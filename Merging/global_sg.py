from math import floor
from typing import Literal
import time

import torch
import numpy as np

from utils import GaussianSG
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FROSS_ROOT = os.path.dirname(CURRENT_DIR)
SAM3_ROOT = os.path.join(FROSS_ROOT, "sam3")
if SAM3_ROOT not in sys.path:
    sys.path.append(SAM3_ROOT)

import torchvision.transforms as T
import torch.nn.functional as F


SAM3_model = None
SAM3_processor = None


def init_sam3():
    """Initialize SAM3 model and processor with HuggingFace checkpoint download."""
    global SAM3_model, SAM3_processor
    if SAM3_model is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # SAM3 uses HuggingFace for checkpoint download
        SAM3_model = build_sam3_image_model(
            load_from_HF=True,
            device="cuda",
            eval_mode=True,
            enable_inst_interactivity=True
        )
        # Create processor for image embedding
        SAM3_processor = Sam3Processor(SAM3_model, resolution=1008, device="cuda")
        print("Initialized SAM3 model and processor")
    
    return SAM3_model, SAM3_processor


from collections import deque
try:
    from visual_refinement import build_visual_refiner
except ImportError:
    build_visual_refiner = None

# Global 3D scene graph
class GlobalSG_Gaussian:
    def __init__(
        self,
        hellinger_thershold,
        dist_threshold=0.5,
        classes_dist_method='kl',
        num_classes=20,
        num_rel_classes=7,
        visualize=False,
        obj_class_names=None,
        rel_class_names=None,
        refiner_model_path=None,
        refiner_probe_path=None,
        refiner_batch_size=None,
    ):
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.global_group = GaussianSG(num_classes, num_rel_classes, hellinger_thershold, dist_threshold, classes_dist_method)
        
        self.obj_class_names = obj_class_names
        self.rel_class_names = rel_class_names
        
        self.sam3_model, self.sam3_processor = init_sam3()
        
        self.visualize = visualize
        if self.visualize:
            self.cur_obj = []
            self.cur_rel = []
        
        # --- Foundation-model visual refinement ---
        self.frame_buffer = deque(maxlen=64) # Increased for sparse sampling (16 frames * stride 4)
        if build_visual_refiner:
            print("GlobalSG: Initializing visual refiner: V-JEPA 2")
            self.visual_refiner = build_visual_refiner(
                "vjepa",
                model_path=refiner_model_path,
                probe_path=refiner_probe_path,
                rel_classes=rel_class_names,
                batch_size=refiner_batch_size,
            )
        else:
            self.visual_refiner = None
            
        self.visual_evidence_accumulator = {} # Key: (cls_idx_A, cls_idx_B), Value: List of Prob Dicts
        self.evidence_count = 0
        self.vis_refine_counter = 0

    def _predict_masks_with_sam3(self, img_np, xyxy_bboxes):
        from PIL import Image as PILImage

        img_pil = PILImage.fromarray(img_np)
        masks_list = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.sam3_processor.set_image(img_pil)
            for i in range(xyxy_bboxes.shape[0]):
                box = xyxy_bboxes[i].cpu().numpy().astype(np.float32)
                masks_i, _, _ = self.sam3_model.predict_inst(
                    inference_state,
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False
                )
                masks_list.append(torch.from_numpy(masks_i[0]).to("cuda"))
        return torch.stack(masks_list, dim=0)

    def _predict_masks(self, img_np, xyxy_bboxes):
        return self._predict_masks_with_sam3(img_np, xyxy_bboxes).bool()

    def accumulate_visual_evidence(self, img_np, bboxes, classes):
        """
        Runs the selected foundation-model refiner on crops of object pairs.
        """
        if self.visual_refiner is None: return
        
        # Stride Check (Optimize Runtime)
        # ScanNet videos are 30fps. Checking every 5th frame (~6fps) is sufficient.
        self.vis_refine_counter += 1
        if self.vis_refine_counter % 5 != 0:
            return

        # Ensure inputs are numpy/CPU
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()
            
        # Helper imports
        from PIL import Image
        
        # Prepare 16-frame sequence
        # Strategy: Use existing frame_buffer. Stride=1 for now (consecutive).
        # If buffer is short, replicate the first frame (padding).
        current_frames_np = list(self.frame_buffer)
        seq_len = 16
        
        if len(current_frames_np) == 0:
            # Should not happen as we append before calling
            frame_seq_np = [img_np] * seq_len
        elif len(current_frames_np) < seq_len:
            # Pad with the oldest frame (first in list) at the beginning
            # e.g. [f0, f1] -> [f0, ... f0, f0, f1]
            pad_len = seq_len - len(current_frames_np)
            frame_seq_np = [current_frames_np[0]] * pad_len + current_frames_np
        else:
            # Take last 16 frames
            frame_seq_np = current_frames_np[-seq_len:]
            
        # Convert to PIL for cropping
        frame_seq_pil = [Image.fromarray(f) for f in frame_seq_np]
        
        pairs = []
        crop_sequences = [] # List of Lists
        
        num_objs = len(bboxes)
        if num_objs < 2: return
        
        # Limit processing
        for i in range(num_objs):
            for j in range(num_objs):
                if i == j: continue
                
                # Check distance (simple centroid distance)
                # bbox: [cx, cy, w, h]
                xi, yi, wi, hi = bboxes[i]
                xj, yj, wj, hj = bboxes[j]
                
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                # If distance > sum of widths/heights, skip? 
                # Heuristic: dist < max(wi, wj, hi, hj) * 2
                thresh = max(wi, wj, hi, hj) * 2.0
                if dist > thresh: continue 
                
                # Crop Union
                box_i_xyxy = [xi - wi/2, yi - hi/2, xi + wi/2, yi + hi/2]
                box_j_xyxy = [xj - wj/2, yj - hj/2, xj + wj/2, yj + hj/2]
                
                x1 = min(box_i_xyxy[0], box_j_xyxy[0])
                y1 = min(box_i_xyxy[1], box_j_xyxy[1])
                x2 = max(box_i_xyxy[2], box_j_xyxy[2])
                y2 = max(box_i_xyxy[3], box_j_xyxy[3])
                
                # Safety Clip
                W, H = frame_seq_pil[0].size
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                
                # Minimum size check (10px)
                if x2 - x1 < 10 or y2 - y1 < 10: continue
                
                # Crop from ALL frames in the sequence
                seq_crops = []
                for frame in frame_seq_pil:
                     c = frame.crop((x1, y1, x2, y2))
                     seq_crops.append(c)
                
                crop_sequences.append(seq_crops)
                pairs.append((classes[i], classes[j]))
        
        # Run Inference Batch
        if len(crop_sequences) > 0:
            predictions = self.visual_refiner.predict_relations(crop_sequences)
            
            # Store
            for (curr_s_cls, curr_o_cls), probs in zip(pairs, predictions):
                key = (curr_s_cls, curr_o_cls)
                if key not in self.visual_evidence_accumulator:
                    self.visual_evidence_accumulator[key] = []
                # Optional: Only store Top-1 or Full Dist? Store full.
                self.visual_evidence_accumulator[key].append(probs)
                self.evidence_count += 1

    def run_visual_refinement(self):
        """
        Uses accumulated visual evidence to refine relation priors.
        """
        has_visual_refiner = self.visual_refiner is not None

        if has_visual_refiner and len(self.visual_evidence_accumulator) > 0:
            print(f"GlobalSG: Processing accumulated visual evidence from {self.evidence_count} pairs...")

            priors = {}
            for (idx_s, idx_o), prob_list in self.visual_evidence_accumulator.items():
                if not prob_list: continue

                avg_probs = {}
                first_keys = list(prob_list[0].keys())
                for k in first_keys:
                    avg_probs[k] = 0.0

                for p in prob_list:
                    for k in first_keys:
                        avg_probs[k] += p.get(k, 0.0)

                for k in avg_probs:
                    avg_probs[k] /= len(prob_list)

                if idx_s < 0 or idx_o < 0: continue

                if self.obj_class_names:
                    try:
                        s_name = self.obj_class_names[idx_s]
                        o_name = self.obj_class_names[idx_o]
                        priors[(s_name, o_name)] = avg_probs
                    except IndexError:
                        pass

            if self.obj_class_names and self.rel_class_names:
                print(f"GlobalSG: Applying V-JEPA 2 priors to {len(priors)} pairs...")
                self.global_group.apply_bayesian_fusion(
                    priors,
                    self.rel_class_names,
                    self.obj_class_names,
                    alpha=0.6,
                    entropy_thresh=0.6
                )
                print("GlobalSG: Visual Refinement Complete.")
            
        # 5. Structural Pattern Completion (SPC)
        # Visual Autocomplete based on geometric similarity to anchors.
        # Moved from UDBF to maintain geometric refinement capabilities.
        if self.obj_class_names and self.rel_class_names:
            print("GlobalSG: Running Structural Pattern Completion (SPC)...")
            self.global_group.apply_structural_completion(self.obj_class_names, self.rel_class_names)

    def _compute_cluster_inliers(self, depth_values, anchor_depth=None):
        """
        Robust Outlier Rejection: Anchor-based Depth Clustering (DBSCAN-1D equivalent).
        Instead of selecting the largest cluster, select the cluster closest to an anchor depth.
        
        Args:
            depth_values: (N,) or (N,1) torch tensor depth samples for a mask
            anchor_depth: optional scalar anchor depth (torch scalar or float). If None, uses median(depth_values).
        Returns:
            mask: boolean mask (N,) indicating inlier indices belonging to the selected cluster
        """
        depth_vals_flat = depth_values.squeeze()
        device = depth_vals_flat.device

        # Default: keep all
        mask = torch.ones_like(depth_vals_flat, dtype=torch.bool)

        # Too few points: don't cluster
        if depth_vals_flat.numel() <= 15:
            return mask

        # Remove invalid depths (<=0, nan, inf) for stable clustering
        valid_depth = torch.isfinite(depth_vals_flat) & (depth_vals_flat > 0)
        if valid_depth.sum() <= 15:
            return mask  # fallback: keep all

        depth_vals = depth_vals_flat[valid_depth]

        # Anchor: if not given, use robust median depth of valid points
        if anchor_depth is None:
            anchor_depth = torch.median(depth_vals)
        else:
            if not isinstance(anchor_depth, torch.Tensor):
                anchor_depth = torch.tensor(anchor_depth, device=device, dtype=depth_vals.dtype)
            else:
                anchor_depth = anchor_depth.to(device=device, dtype=depth_vals.dtype)

        # Sort depth values
        sorted_d, sort_idx = torch.sort(depth_vals)

        # Compute depth gaps
        diffs = sorted_d[1:] - sorted_d[:-1]

        # Dynamic threshold (depth-aware)
        # - physical floor: 3cm
        # - statistical: 10x median gap
        # - scale-aware: 2% of anchor depth
        med_diff = torch.median(diffs)
        gap_thr = torch.max(
            torch.tensor(0.03, device=device, dtype=sorted_d.dtype),
            torch.max(10.0 * med_diff, 0.02 * anchor_depth)
        )

        # Find split points
        split_indices = torch.nonzero(diffs > gap_thr).reshape(-1) + 1

        # Cluster boundaries
        boundaries = torch.cat([
            torch.tensor([0], device=device),
            split_indices,
            torch.tensor([len(sorted_d)], device=device)
        ])

        # Choose best cluster by anchor proximity (NOT by length)
        best_idx = None
        best_score = None

        for k in range(len(boundaries) - 1):
            s = boundaries[k].item()
            e = boundaries[k + 1].item()
            if e - s <= 2:
                continue

            cluster = sorted_d[s:e]
            # robust cluster representative
            cluster_med = torch.median(cluster)
            # score = |median(cluster) - anchor|
            score = torch.abs(cluster_med - anchor_depth)

            if best_score is None or score < best_score:
                best_score = score
                best_idx = k

        # Fallback: if no cluster found, keep all
        if best_idx is None:
            return mask

        # Map chosen cluster indices back
        start_idx = boundaries[best_idx]
        end_idx = boundaries[best_idx + 1]
        chosen_sorted_idx = sort_idx[start_idx:end_idx]  # indices in "depth_vals"

        # Build final mask in original depth_vals_flat space
        mask[:] = False
        # indices of valid_depth in original tensor
        orig_valid_indices = torch.nonzero(valid_depth, as_tuple=False).squeeze(1)
        chosen_orig_indices = orig_valid_indices[chosen_sorted_idx]
        mask[chosen_orig_indices] = True

        return mask

    def _compute_spatial_consistency(self, depth_map, mask, tolerance=0.1, kernel_size=3):
        """
        Checks local spatial consistency. A pixel is valid if it is close to the median of its neighbors.
        Uses unfolding for efficient local neighborhood processing.
        """
        h, w = depth_map.shape
        # Only process relevant area
        min_y, min_x = torch.where(mask)
        if len(min_y) == 0:
            return torch.zeros_like(mask)
            
        y1, y2 = min_y.min(), min_y.max() + 1
        x1, x2 = min_x.min(), min_x.max() + 1
        
        # Extract patch
        depth_patch = depth_map[y1:y2, x1:x2].float()
        mask_patch = mask[y1:y2, x1:x2]
        
        # Unfold to get neighbors (patches)
        # padding
        padding = kernel_size // 2
        padded_depth = torch.nn.functional.pad(depth_patch.unsqueeze(0).unsqueeze(0), (padding, padding, padding, padding), mode='replicate')
        
        # (1, C, H, W) -> (1, C*kernel*kernel, L)
        unfolded = torch.nn.functional.unfold(padded_depth, kernel_size=kernel_size) 
        # (kernel*kernel, H*W)
        neighbors = unfolded.view(kernel_size*kernel_size, -1)
        
        # Calculate median of neighbors
        local_median = torch.median(neighbors, dim=0)[0].view(depth_patch.shape)
        
        # Check consistency
        consistent_patch = torch.abs(depth_patch - local_median) < (tolerance * local_median + 1e-3)
        
        # Final mask for the patch: must be originally in mask AND consistent
        final_patch_mask = mask_patch & consistent_patch
        
        # Put back into full mask
        full_consistent_mask = torch.zeros_like(mask)
        full_consistent_mask[y1:y2, x1:x2] = final_patch_mask
        
        return full_consistent_mask

    def update(self, classes, bboxes, rels, rel_classes, depth, camera_rot, camera_trans, camera_intrinsic, img=None): # use depth camera intrinsic
        # --- Frame Buffer Update ---
        if img is not None:
             if isinstance(img, torch.Tensor):
                 img_np = img.permute(1, 2, 0).cpu().numpy()
                 if img_np.max() <= 1.0:
                     img_np = (img_np * 255).astype(np.uint8)
                 else:
                     img_np = img_np.astype(np.uint8)
             else:
                 img_np = img # Assume numpy uint8
             self.frame_buffer.append(img_np)
             
             # V-JEPA Accumulation (Pairwise)
             if len(classes) > 0:
                 cls_idx = np.argmax(classes, axis=1) if classes.ndim > 1 else classes
                 self.accumulate_visual_evidence(img_np, bboxes, cls_idx)
        # ---------------------------

        if len(classes) == 0:
            if self.visualize:
                self.cur_obj.append({
                    "ids": self.global_group.valid_indices.copy(),
                    "classes": self.global_group.classes.copy(),
                    "means": self.global_group.means.copy(),
                    "covs": self.global_group.covs.copy(),
                })
                self.cur_rel.append(self.global_group.rels.copy())
            return 0., 0., 0., 0., 0.

        camera_rot = camera_rot[None, ...] # (1, 3, 3)
        camera_trans = camera_trans[None, :, None] # (1, 3, 1)

        # bboxes is [cx, cy, w, h] from main.py
        # Prepare boxes for the selected mask model (xyxy)
        xyxy_bboxes = torch.cat((bboxes[:, :2] - bboxes[:, 2:] / 2, bboxes[:, :2] + bboxes[:, 2:] / 2), dim=1).int()
        bboxes_np = bboxes.cpu().numpy().astype(int)

        start = time.time()
        # Convert image for mask prediction
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        else:
            img_np = img

        masks = self._predict_masks(img_np, xyxy_bboxes)
        
        sam_time = time.time() - start

        start = time.time()
        
        if len(classes) == 0:
             return sam_time, 0., 0., 0., 0.

        # unique class IDs among objects
        max_classes = torch.tensor(classes).argmax(axis=1) if isinstance(classes, np.ndarray) else classes.argmax(dim=1)
        unique_cls = torch.unique(max_classes)
        class_cover = []
        for cls_id in unique_cls:
            cls_masks = masks[max_classes == cls_id] 
            cls_union = torch.any(cls_masks, axis=0) 
            class_cover.append(cls_union)
        if len(class_cover) > 0:
            class_cover = torch.stack(class_cover, axis=0)
            # For each pixel, count how many different classes cover it
            class_cover_count = class_cover.sum(axis=0) 
            ambiguous_pixels = class_cover_count > 1 
            if ambiguous_pixels.any():
                masks[:, ambiguous_pixels] = False

        masks_f = masks.float().unsqueeze(1) 
        kernel_size = 7
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=masks.device)
        required = kernel_size * kernel_size
        conv = torch.nn.functional.conv2d(masks_f, kernel, padding=kernel_size // 2)
        eroded = conv == required
        masks = eroded.squeeze(1)
        post_time = time.time() - start
        
        depth_cuda = torch.tensor(depth, device="cuda")
        camera_rot_cuda = torch.tensor(camera_rot, device="cuda")
        camera_trans_cuda = torch.tensor(camera_trans, device="cuda")

        proj_time, gaussian_time = 0., 0.
        mean_3d, cov_3d, pcds = [], [], []
        valid = np.ones((bboxes_np.shape[0]), dtype=bool)
        
        for mask_idx, mask in enumerate(masks):
            start = time.time()
            mask_y, mask_x = torch.nonzero(mask, as_tuple=True)
            if mask_y.shape[0] < 10:
                valid[mask_idx] = False
                continue
            depth_vals = depth_cuda[mask_y, mask_x].unsqueeze(-1)  # (num_points, 1)
            x = (mask_x - camera_intrinsic.cx) / camera_intrinsic.fx
            y = (mask_y - camera_intrinsic.cy) / camera_intrinsic.fy
            z = torch.ones_like(x)
        
            camera_coords = (torch.stack((x, y, z), dim=-1) * depth_vals).unsqueeze(-1)  # (num_points, 3, 1)
            coords_3d = (camera_rot_cuda @ camera_coords + camera_trans_cuda).squeeze(-1)  # (num_points, 3)
            proj_time += time.time() - start
            
            # === Hybrid Dual-Domain Refinement ===
            
            # 1. Spatial Domain: Geometry Continuity Check
            # Removes "flying pixels" and edge noise using local consistency
            spatial_mask = self._compute_spatial_consistency(depth_cuda, mask, tolerance=0.05)
            
            # 2. Density Domain: 1D Depth Clustering (Gap-based)
            # Removes background/foreground outliers by finding the main object cluster
            # Get depth values that passed the spatial check
            spatial_y, spatial_x = torch.nonzero(spatial_mask, as_tuple=True)
            if spatial_y.shape[0] < 10:
                valid[mask_idx] = False
                continue
                
            depth_vals_spatial = depth_cuda[spatial_y, spatial_x]
            
            # Apply clustering on the spatially filtered points
            # anchor depth: robust median inside spatially consistent region
            anchor_depth = torch.median(depth_vals_spatial)
            cluster_inlier_mask = self._compute_cluster_inliers(depth_vals_spatial, anchor_depth=anchor_depth)

            # Combined Inliers: subset of spatial points that are also in the main cluster
            final_y = spatial_y[cluster_inlier_mask]
            final_x = spatial_x[cluster_inlier_mask]
            
            if final_y.shape[0] < 10:
                valid[mask_idx] = False
                continue

            # Re-project only correct points
            depth_vals_final = depth_cuda[final_y, final_x].unsqueeze(-1)
            x = (final_x - camera_intrinsic.cx) / camera_intrinsic.fx
            y = (final_y - camera_intrinsic.cy) / camera_intrinsic.fy
            z = torch.ones_like(x)
            
            camera_coords = (torch.stack((x, y, z), dim=-1) * depth_vals_final).unsqueeze(-1)
            coords_3d = (camera_rot_cuda @ camera_coords + camera_trans_cuda).squeeze(-1)

            start = time.time()
            mean_3d.append(torch.mean(coords_3d, dim=0).cpu().numpy())
            cov_3d.append((torch.cov(coords_3d.T) + 1e-6 * torch.eye(3, device=coords_3d.device)).cpu().numpy())
            gaussian_time += time.time() - start
            # Extract point clouds for evaluation
            eval_idx = torch.randint(0, coords_3d.shape[0], (max(coords_3d.shape[0] // 2500, 10), ))
            pcds.append(coords_3d[eval_idx].cpu().numpy())

        if len(mean_3d) == 0:
            return sam_time, post_time, proj_time, gaussian_time, 0.

        mean_3d = np.stack(mean_3d, axis=0)
        cov_3d = np.stack(cov_3d, axis=0)

        classes = classes[valid]
        bboxes = bboxes[valid] # keep bboxes consistent if used later (although local var)
        # note: bboxes_np used for logic is already correct, but we might return proper count?
        # actually we don't return bboxes/classes, just updating global SG
        
        if len(rels) > 0:
            new_idx = np.cumsum(valid) - 1
            new_idx[~valid] = -1
            valid_edge_idx = np.logical_and(new_idx[rels[:, 0]] != -1, new_idx[rels[:, 1]] != -1)
            rel_classes = rel_classes[valid_edge_idx]
            rels = new_idx[rels[valid_edge_idx]]

        # Add local 3D SG to global 3D SG
        update_idx = self.global_group.add(classes, mean_3d, cov_3d, rels, rel_classes, pcds)

        # Calculate the Hellinger distance and merge objects
        start_time = time.time() # Renamed 'start' to 'start_time' to avoid conflict with other 'start' variables
        self.global_group.merge(update_idx)
        merge_time = time.time() - start_time


        if self.visualize:
            self.cur_obj.append({
                "ids": self.global_group.valid_indices.copy(),
                "classes": self.global_group.classes.copy(),
                "means": self.global_group.means.copy(),
                "covs": self.global_group.covs.copy(),
            })
            self.cur_rel.append(self.global_group.rels.copy())

        return sam_time, post_time, proj_time, gaussian_time, merge_time
