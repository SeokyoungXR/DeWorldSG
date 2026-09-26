import os
import argparse
from pathlib import Path
import json
import pickle
import time

import numpy as np
from tqdm import tqdm
import torch
from pycocotools.coco import COCO

from sg_loader import SG_Loader, GT_SG_Loader
from sg_prediction import SG_Predictor
from global_sg import GlobalSG_Gaussian


def main(args):
    # Print configuration
    print(f"Using dataset: {'3RScan' if args.label_categories == 'scannet' else 'ReplicaSSG'}, path: {args.dataset_path}")
    print(f"Using split: {args.split}")
    print("Hyperparameters:")
    print(f"\tObject threshold: {args.obj_thresh}")
    print(f"\tRelation topk: {args.rel_topk}")
    print(f"\tMerging threshold: {args.merging_threshold}")
    print(f"\tClass distribution distance weight: {args.classes_dist_weight}")
    print(f"Using ground truth scene graphs: {args.use_gt_sg}")
    print(f"Using ground truth camera poses: {args.use_gt_pose}")
    print(f"Saving results to: {args.output_path}")

    # Load dataset
    split = args.split
    if args.label_categories == "scannet":
        obj_ann_path = f"2DSG20/{split}.json"
        rel_ann_path = f"2DSG20/rel.json"
        SSG_path = f"3DSSG_subset"
        class_mapping_path = f"{SSG_path}/3dssg_to_scannet.json"
        OBJ_CLASS_NAME = "ScanNet_list"
        REL_CLASS_NAME = "ScanNet_rel"
    elif args.label_categories == "replica":
        assert split == "test" or split == "val", "Replica is only for testing"
        obj_ann_path = f"2DSG/{split}.json"
        rel_ann_path = f"2DSG/rel.json"
        SSG_path = f"ReplicaSSG"
        class_mapping_path = f"{SSG_path}/replica_to_visual_genome.json"
        OBJ_CLASS_NAME = "VisualGenome_list"
        REL_CLASS_NAME = "VisualGenome_rel"

    with open(Path(args.dataset_path) / class_mapping_path) as f:
        class_mapping = json.load(f)
    obj_classes = class_mapping[OBJ_CLASS_NAME]
    rel_classes = class_mapping[REL_CLASS_NAME]

    scan_split = "validation" if split == "val" else split
    with open(Path(args.dataset_path) / SSG_path / f"{scan_split}_scans.txt") as f:
        scan_ids = f.readlines()
    scan_ids = [scan_id.strip() for scan_id in scan_ids]
    scan_ids = scan_ids[:10] if args.debug else scan_ids

    if not args.use_gt_sg:
        sg_predictor = SG_Predictor(args)
    else:
        objCOCO = COCO(Path(args.dataset_path) / obj_ann_path)
        with open(Path(args.dataset_path) / rel_ann_path) as f:
            rel_ann = json.load(f)[split]

    # Inference Start
    predictions = {}
    
    obj_time = 0 # Time taken for object detection
    rel_time = 0 # Time taken for relation extraction
    sam_time = 0 # Time taken for SAM mask prediciton
    post_time = 0 # Time taken for SAM post-processing
    proj_time = 0 # Time taken for projecting SAM masks to 3D
    gaussian_time = 0 # Time taken for computing mean and covariance of projected 3D points
    merge_time = 0 # Time taken for merging local SG into global SG
    frame_time = 0 # Time taken for processing each frame
    frame_cnt = 0 # Total number of frames processed
    keep_temporal_trace = args.visualize_folder is not None or args.save_temporal_trace

    for scan_id in tqdm(scan_ids):
        # Initialize global scene graph
        global_sg = GlobalSG_Gaussian(
            args.merging_threshold,
            args.classes_dist_weight,
            args.classes_dist_method,
            len(obj_classes),
            len(rel_classes),
            keep_temporal_trace,
            obj_class_names=obj_classes,
            rel_class_names=rel_classes,
            refiner_model_path=args.refiner_model_path,
            refiner_probe_path=args.refiner_probe_path,
            refiner_batch_size=args.refiner_batch_size,
        )

        # Initialize scene graph loader
        if args.use_gt_sg:
            sg_loader = GT_SG_Loader(scan_id, objCOCO, rel_ann, args)
        else:
            sg_loader = SG_Loader(scan_id, args)

        camera_intrinsics = sg_loader.color_intrinsic

        if args.visualize_folder:
            cur_obj_2d = []
            cur_rel_2d = []

        # Process each frame
        frame_start_time = time.time()
        for idx, data in enumerate(sg_loader):
            frame_cnt += 1
             
            # Detect objects
            if args.use_gt_sg:
                depth, classes, bboxes, relation_classes, rels, camera_rot, camera_trans = data
            else:
                img, depth, camera_rot, camera_trans = data
                start_time = time.time()
                img_cuda = torch.tensor(img).permute(2, 0, 1).cuda() / 255.0
                obj_det_output, all_scores, classes, class_probs, bboxes = sg_predictor.detect_objects(img_cuda)
                obj_time += time.time() - start_time

            # clip bounding boxes (cx, cy, w, h) to image size
            if isinstance(bboxes, np.ndarray):
                lib = np
            else:
                lib = torch
            xyxy_boxes = lib.concatenate((bboxes[:, :2] - bboxes[:, 2:] / 2, bboxes[:, :2] + bboxes[:, 2:] / 2), axis=1)
            xyxy_boxes[:, 0] = lib.clip(xyxy_boxes[:, 0], 0, img.shape[1] - 1)
            xyxy_boxes[:, 1] = lib.clip(xyxy_boxes[:, 1], 0, img.shape[0] - 1)
            xyxy_boxes[:, 2] = lib.clip(xyxy_boxes[:, 2], 0, img.shape[1] - 1)
            xyxy_boxes[:, 3] = lib.clip(xyxy_boxes[:, 3], 0, img.shape[0] - 1)
            bboxes = lib.stack(((xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2,
                                     (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2,
                                     xyxy_boxes[:, 2] - xyxy_boxes[:, 0],
                                     xyxy_boxes[:, 3] - xyxy_boxes[:, 1]), axis=1)

            # Extract relations
            if not args.use_gt_sg:
                start_time = time.time()
                rels, relation_classes = sg_predictor.extract_relations(obj_det_output, all_scores)
                rel_time += time.time() - start_time

            if args.visualize_folder:
                cur_obj_2d.append({"classes": classes, "bboxes": bboxes, "scores": all_scores})
                cur_rel_2d.append({"rels": rels, "rel_classes": relation_classes})

            if not args.use_gt_sg:
                input_classes = class_probs
            else:
                # Convert GT class indices to one-hot probability distributions
                num_obj_classes = len(obj_classes)
                # one hot ndarry: (N, num_obj_classes) 
                one_hot_classes = np.zeros((classes.shape[0], num_obj_classes), dtype=float)
                one_hot_classes[np.arange(classes.shape[0]), classes] = 1.0
                input_classes = one_hot_classes

            # Lift and merge local scene graph into global scene graph
            frame_sam_time, frame_post_time, frame_proj_time, frame_gaussian_time, frame_merge_time = global_sg.update(input_classes, bboxes, rels, relation_classes, depth, camera_rot, camera_trans, camera_intrinsics, img_cuda)
            sam_time += frame_sam_time
            post_time += frame_post_time
            proj_time += frame_proj_time
            gaussian_time += frame_gaussian_time
            merge_time += frame_merge_time

        frame_time += time.time() - frame_start_time

        print("Step: Running visual refinement (V-JEPA 2)...")
        global_sg.run_visual_refinement()

        # Prepare predictions for the current scene
        prediction = {"pcd": [], "cls": [], "edge_index": [], "edge_cls": []}

        new_points_xyz = np.ndarray((0, 3), dtype=np.float32)
        new_points_rgb = np.ndarray((0, 3), dtype=np.uint8)
        point_clouds = []
        classes = global_sg.global_group.classes
        means = global_sg.global_group.means
        covs = global_sg.global_group.covs
        rels = global_sg.global_group.rels
        pcd = global_sg.global_group.pcd
        for idx in range(classes.shape[0]):
            # Sanity checks
            assert not ((classes[idx] < 0).any() or \
                        np.isnan(means[idx]).any() or \
                        np.isnan(covs[idx]).any() or \
                        (rels[idx] < 0).any() or \
                        pcd[idx] is None), \
                        f"class: {(classes[idx] < 0).any()}, mean: {np.isnan(means[idx]).any()}, cov: {np.isnan(covs[idx]).any()}, rels: {(rels[idx] < 0).any()}, pcd: {pcd[idx] is None}"
            
            pred_points = pcd[idx]
            point_clouds.append(pred_points)
            color = np.random.randint(0, 255, 3)
            new_points_xyz = np.concatenate((new_points_xyz, np.array(pred_points)), axis=0)
            new_points_rgb = np.concatenate((new_points_rgb, np.full((len(pred_points), 3), color)), axis=0)

        prediction["pcd"] = point_clouds
        # change probability distribution to class index
        classes = classes.argmax(axis=1)
        prediction["cls"] = torch.nn.functional.one_hot(torch.tensor(classes), len(obj_classes)).cpu().numpy()
        prediction["mean"] = means
        prediction["cov"] = covs
        s, o = np.nonzero(np.sum(rels, axis=-1))
        prediction["edge_index"] = np.array([s, o])
        prediction["edge_cls"] = np.array(rels[s, o] / np.sum(rels[s, o], axis=-1, keepdims=True))
    
        predictions[scan_id] = prediction

        if args.visualize_folder:
            folder = f"{args.visualize_folder}/{'3RScan' if args.label_categories == 'scannet' else 'ReplicaSSG'}/{scan_id}"
            os.makedirs(folder, exist_ok=True)
            with open(f"{folder}/obj.pkl", "wb") as f:
                pickle.dump(global_sg.cur_obj, f)
            with open(f"{folder}/rel.pkl", "wb") as f:
                pickle.dump(global_sg.cur_rel, f)
            with open(f"{folder}/obj_2d.pkl", "wb") as f:
                pickle.dump(cur_obj_2d, f)
            with open(f"{folder}/rel_2d.pkl", "wb") as f:
                pickle.dump(cur_rel_2d, f)

        if args.save_temporal_trace and hasattr(global_sg, "cur_obj"):
            folder = args.output_path / "temporal_trace" / args.label_categories / scan_id
            os.makedirs(folder, exist_ok=True)
            with open(folder / "obj.pkl", "wb") as f:
                pickle.dump(global_sg.cur_obj, f)
            with open(folder / "rel.pkl", "wb") as f:
                pickle.dump(global_sg.cur_rel, f)

    # Save predictions
    os.makedirs(args.output_path / args.label_categories, exist_ok=True)
    obj_name = f"obj{args.obj_thresh}"
    rel_name = f"rel{args.rel_topk}"
    merging_name = f"merging{args.merging_threshold}"
    class_merge_name = f"classdist_{args.classes_dist_method}{args.classes_dist_weight}"
    mask_name = "_sam3"
    output_filename = f"predictions_gaussian_{obj_name}_{rel_name}_{merging_name}_{class_merge_name}_{args.split}"\
        + f"{'_gt2dsg' if args.use_gt_sg else ''}{'_gtpose' if args.use_gt_pose else ''}"\
        + f"{mask_name}_postprocessing{'_debug' if args.debug else ''}.pkl"
    
    output_path = args.output_path / args.label_categories / output_filename
    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)

    merge_time_per_frame = merge_time / frame_cnt
    obj_time_per_frame = obj_time / frame_cnt
    rel_time_per_frame = rel_time / frame_cnt
    print(f"Number of frames: {frame_cnt}")
    print(f"SAM3 time per frame: {sam_time / frame_cnt}")
    print(f"SAM3 post-processing time per frame: {post_time / frame_cnt}")
    print(f"Projection time per frame: {proj_time / frame_cnt}")
    print(f"Compute mean/cov time per frame: {gaussian_time / frame_cnt}")
    print(f"Merge time per frame: {merge_time_per_frame}")
    print(f"Object time per frame: {obj_time_per_frame}")
    print(f"Relation time per frame: {rel_time_per_frame}")
    print(f"FPS: {frame_cnt / frame_time}")

    os.makedirs(args.output_path / "results" / args.label_categories, exist_ok=True)
    output_path = args.output_path / "results" / args.label_categories / f"{output_filename[:-4]}.txt"
    with open(output_path, "w") as f:
        f.write(f"Number of frames: {frame_cnt}\n")
        f.write(f"Merge time per frame: {merge_time_per_frame}\n")
        f.write(f"Object time per frame: {obj_time_per_frame}\n")
        f.write(f"Relation time per frame: {rel_time_per_frame}\n")
        f.write(f"FPS: {frame_cnt / frame_time}\n")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--artifact_path", type=Path)
    args.add_argument("--output_path", type=Path, default="output/")
    args.add_argument("--label_categories", type=str, choices=["scannet"], default="scannet")
    args.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    args.add_argument("--obj_thresh", type=float, default=0.7)
    args.add_argument("--rel_topk", type=int, default=10)
    args.add_argument("--use_gt_sg", action="store_true", default=False)
    args.add_argument("--not_use_gt_pose", action="store_true", default=False)
    args.add_argument("--merging_threshold", type=float, default=0.7)
    args.add_argument("--classes_dist_method", type=str, default="dot_product", choices=["kl", "l2", "js", "hellinger", "dot_product", "top_class"])
    args.add_argument("--classes_dist_weight", type=float, default=0.3)
    args.add_argument("--refiner_model_path", type=Path, default=None,
                      help="Backbone checkpoint for the selected refiner.")
    args.add_argument("--refiner_probe_path", type=Path, default=None,
                      help="MLP relation probe checkpoint for the selected refiner.")
    args.add_argument("--refiner_batch_size", type=int, default=None,
                      help="Batch size for visual refiner inference.")

    # Debugging arguments
    args.add_argument("--not_preload", action="store_true", default=False, help="Preload all images before each scene. Disable this if you run out of memory. Enable this for runtime evaluation.")
    args.add_argument("--visualize_folder", type=Path, default=None, help="Visualize 2D SG and 3D SSG in each frame and save to the specified folder.")
    args.add_argument("--save_temporal_trace", action="store_true", default=True, help="Save per-frame global graph snapshots for temporal consistency evaluation.")

    args.add_argument("--debug", action="store_true", default=False)


    args = args.parse_args()
    args.use_gt_pose = not args.not_use_gt_pose
    args.preload = not args.not_preload

    assert args.artifact_path is not None or args.use_gt_sg, "Artifact path is required when not using ground truth scene graphs"

    main(args)
