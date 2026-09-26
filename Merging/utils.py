import numpy as np
import math

def bb_center_xy(bb):
    x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def center_depth_in_bbox(depth_map, bb):
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    
    # Safety Check for Bounds
    cx = max(0, min(W - 1, cx))
    cy = max(0, min(H - 1, cy))
    
    # Depth map usually (H, W), so index [y, x]
    v = float(depth_map[cy, cx])
    return v


def minmax_normalize(vals):
    if not vals: return []
    vmin = float(min(vals))
    vmax = float(max(vals))
    denom = (vmax - vmin) + 1e-8
    return [(float(v) - vmin) / denom for v in vals]


def compute_geo(image_shape, objects_bbs, depth_map,
    tau=0.45,
    lambda1=0.5,
    lambda2=1.0,
    invert_relative=True
):
    # Image shape is (H, W) or (H, W, C). We need (W, H) for diag calculation if consistent with PIL logic, 
    # but here we just need diag.
    H, W = image_shape[:2]
    img_diag = math.hypot(W, H)
    num_objects = len(objects_bbs)

    obj_depth = [center_depth_in_bbox(depth_map, bb) for bb in objects_bbs]
    
    # Invert depth if requested (1/depth for disparity-like behavior, or just relative scaling)
    if invert_relative:
        obj_depth = [1.0 / (v + 1e-6) for v in obj_depth]
    
    obj_depth_norm = minmax_normalize(obj_depth)

    pair_ids = []
    pair_gf = []
    kept_pair_ids = []

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            pair_ids.append([i, j])

            cxi, cyi = bb_center_xy(objects_bbs[i])
            cxj, cyj = bb_center_xy(objects_bbs[j])

            # 2D Euclidean Distance
            x = math.hypot(cxi - cxj, cyi - cyj)
            x_over_y = x / img_diag

            # Normalized Depth Difference
            ddepth = abs(obj_depth_norm[i] - obj_depth_norm[j])
            
            # Geometric Cost
            gf = lambda1 * x_over_y + lambda2 * ddepth

            pair_gf.append(gf)

            # Pruning
            if gf < tau:
                kept_pair_ids.append([i, j])

    return {
        "obj_depth_center": obj_depth,
        "obj_depth_norm": obj_depth_norm,
        "pair_ids": pair_ids,
        "pair_gf": pair_gf,
        "kept_pair_ids": kept_pair_ids,
    }

# 3D scene graph with Gaussian representation
class GaussianSG:
    def __init__(self, num_obj_class, num_rel_class, merge_threshold, class_dist_weight, classes_dist_method, semantic_threshold=0.8):
        initial_size = 1000
        self._growth_factor = 2
        self._max_size = initial_size
        self._valid_mask = np.zeros((initial_size), dtype=bool)
        # self._classes = np.ndarray((initial_size), dtype=int)
        self._means = np.ndarray((initial_size, 3), dtype=float)
        self._covs = np.ndarray((initial_size, 3, 3), dtype=float)
        self._rels = np.zeros((initial_size, initial_size, num_rel_class), dtype=int)
        self._pcd = [None] * initial_size
        self.num_rel_class = num_rel_class
        self.merge_threshold = merge_threshold
        self.classes_dist_method = classes_dist_method

        # introduce class prob distribution distance in merging criteria
        self.num_obj_class = num_obj_class
        self._classes = np.ndarray((initial_size, num_obj_class), dtype=float)
        self.class_dist_weight = class_dist_weight
        self.semantic_threshold = semantic_threshold
        
    @property
    def valid_size(self):
        return np.count_nonzero(self._valid_mask)
    @property
    def valid_indices(self):
        return np.nonzero(self._valid_mask)[0]
    @property
    def classes(self):
        return self._classes[self._valid_mask]
    @property
    def means(self):
        return self._means[self._valid_mask]
    @property
    def covs(self):
        return self._covs[self._valid_mask]
    @property
    def rels(self):
        return self._rels[self._valid_mask][:, self._valid_mask]
    @property
    def pcd(self):
        return [self._pcd[i] for i in range(len(self._pcd)) if self._valid_mask[i]]
    
    def _expand_if_needed(self, add_size):
        if self.valid_size + add_size <= self._max_size:
            return
        new_size = self._max_size * self._growth_factor + add_size
        add_size = new_size - self._max_size
        # self._classes = np.concatenate((self._classes, np.ndarray((add_size), dtype=int)))
        self._classes = np.concatenate((self._classes, np.ndarray((add_size, self.num_obj_class), dtype=float)))
        self._means = np.concatenate((self._means, np.ndarray((add_size, 3), dtype=float)))
        self._covs = np.concatenate((self._covs, np.ndarray((add_size, 3, 3), dtype=float)))
        self._rels = np.concatenate((self._rels, np.zeros((add_size, self._max_size, self.num_rel_class), dtype=int)), axis=0)
        self._rels = np.concatenate((self._rels, np.zeros((new_size, add_size, self.num_rel_class), dtype=int)), axis=1)
        self._pcd.extend([None] * add_size)
        self._valid_mask = np.concatenate((self._valid_mask, np.zeros((add_size), dtype=bool)))
        self._max_size = new_size

    def add(self, new_classes, new_means, new_covs, new_rels, new_rel_classes, new_pcds):
        add_size = len(new_classes)
        self._expand_if_needed(add_size)
        avail_indices = np.nonzero(~self._valid_mask)[0][:add_size]
        # print(f"classes to add: {new_classes.shape}, \
        #     self._classes shape: {self._classes.shape}, \
        #     means to add: {new_means.shape}, \
        #     covs to add: {new_covs.shape}, \
        #     rels to add: {new_rels.shape}, \
        #     rel_classes to add: {new_rel_classes.shape}"
        # )
        self._classes[avail_indices] = new_classes
        self._means[avail_indices] = new_means
        self._covs[avail_indices] = new_covs
        for idx, avail_idx in enumerate(avail_indices):
            self._pcd[avail_idx] = new_pcds[idx]
        if len(new_rels) > 0:
            new_rels = avail_indices[new_rels]
            self._rels[new_rels[:, 0], new_rels[:, 1], new_rel_classes] += 1
        
        self._valid_mask[avail_indices] = True

        return avail_indices # idx to update
    
    # update for FROSS v2: merge with different criteria
    def merge(self, update_idx):
        update_idx = update_idx.tolist()
        while update_idx:
            idx = update_idx.pop()
            # same_class = self._classes == self._classes[idx]
            # same_class[idx] = False # exclude the current Gaussian itself
            # same_class = same_class & self._valid_mask
            # same_class = np.nonzero(same_class)[0]
            # if np.count_nonzero(same_class) == 0:
            #     continue
            if not self._valid_mask[idx]:
                continue
            all_valid_indices = np.nonzero(self._valid_mask)[0]
            other_indices = all_valid_indices[all_valid_indices != idx]
            if len(other_indices) == 0:
                continue
            # check class distribution distance
            class_dists = self._batched_distribution_distance(self._classes[idx], self._classes[other_indices], metric=self.classes_dist_method)
            # same_class = class_dists < self.classes_dist_threshold
            # same_class_indices = other_indices[same_class]
            # if len(same_class_indices) == 0:
            #     continue
            
            mean1 = self._means[idx]
            cov1 = self._covs[idx]

            gaussian_dist = self._batched_hellinger_distance(mean1, cov1, self._means[other_indices], self._covs[other_indices])

            total_dist = self.class_dist_weight * class_dists + (1 - self.class_dist_weight) * gaussian_dist
            min_idx = np.argmin(total_dist)
            
            # Novelty: Geometric Conflict Resolution (GCR)
            # If objects are spatially identical (Hellinger < 0.2), they must be the same object.
            # This resolves "Ghost Objects" where the class flips between frames (e.g. Chair vs Stool).
            # We force merge regardless of semantic disagreement to maintain physical consistency.
            # if gaussian_dist[min_idx] < 0.2:
            #      merge_idx = other_indices[min_idx]
            #      if merge_idx not in update_idx: update_idx.append(merge_idx)
            #      self._merge_gaussians(idx, merge_idx)
            #      continue

            # Standard Merging with Semantic Gating
            if total_dist[min_idx] < self.merge_threshold and class_dists[min_idx] < self.semantic_threshold:
                merge_idx = other_indices[min_idx]
                if merge_idx not in update_idx: update_idx.append(merge_idx)
                assert self._valid_mask[merge_idx] and self._valid_mask[idx]
                self._merge_gaussians(idx, merge_idx)



    def _batched_hellinger_distance(self, mean1, cov1, mean2, cov2):
        """
        Calculate the Hellinger distance between one and many Gaussian distributions
        """
        assert len(mean1.shape) == 1 and len(mean2.shape) == 2
        assert len(cov1.shape) == 2 and len(cov2.shape) == 3
        assert mean1.shape[0] == mean2.shape[1] == 3
        assert cov1.shape[0] == cov1.shape[1] == cov2.shape[1] == cov2.shape[2] == 3
        
        # --- DepSG: Depth-Adaptive Uncertainty Scaling (Strategy B) ---
        # Scale covariance based on Depth (Z) to account for increasing sensor noise at range.
        # RGB-D Error ~ Z^2. We inflate Covariance to be more lenient for distant objects.
        # Formula: cov' = cov * (1 + alpha * z^2)
        alpha = 0.1 # Scaling factor
        
        z1 = mean1[2]
        z2 = mean2[:, 2] # (N,)
        
        # Inflate cov1
        scale1 = 1.0 + alpha * (z1 ** 2)
        cov1 = cov1 * scale1
        
        # Inflate cov2
        # cov2 is (N, 3, 3), scale2 is (N, 1, 1) for broadcasting
        scale2 = 1.0 + alpha * (z2 ** 2)
        cov2 = cov2 * scale2[:, None, None]
        
        # Standard Hellinger continues...
        mean1 = mean1[None, :, None]
        mean2 = mean2[..., None]
        cov1 = cov1[None, ...]
        mean_diff = mean1 - mean2
        cov_mean = (cov1 + cov2) / 2
        cov_mean_inv = np.linalg.inv(cov_mean)
        det_cov_mean = np.linalg.det(cov_mean)
        B_D = (0.125 * mean_diff.transpose(0, 2, 1) @ cov_mean_inv @ mean_diff).flatten() \
            + 0.5 * np.log(det_cov_mean / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
        return np.sqrt(1 - np.exp(-B_D))

    def _merge_gaussians(self, idx1, idx2): # merge idx1 to idx2
        mean1, cov1 = self._means[idx1], self._covs[idx1]
        mean2, cov2 = self._means[idx2], self._covs[idx2]
        # distribution distance check
        dist1 = self._classes[idx1]
        dist2 = self._classes[idx2]
        assert len(mean1.shape) == len(mean2.shape) == 1
        assert len(cov1.shape) == len(cov2.shape) == 2
        assert mean1.shape[0] == mean2.shape[0] == 3
        assert cov1.shape[1] == cov1.shape[0] == cov2.shape[1] == cov2.shape[0] == 3
        num_pnts1, num_pnts2 = len(self._pcd[idx1]), len(self._pcd[idx2])
        if num_pnts1 == 0 and num_pnts2 == 0: # too small so that they don't have pcd
            num_pnts1 = num_pnts2 = 1
        total_pnts = num_pnts1 + num_pnts2
        mean_diff = mean1 - mean2
        # confidence-weighted update (using max probability as confidence proxy)
        conf1 = np.max(dist1) if not np.isnan(dist1).any() else 0.5
        conf2 = np.max(dist2) if not np.isnan(dist2).any() else 0.5
        
        # Avoid zero confidence
        conf1 = max(conf1, 0.1)
        conf2 = max(conf2, 0.1)

        w1 = num_pnts1 * conf1
        w2 = num_pnts2 * conf2
        total_w = w1 + w2

        # Weighted average for mean and cov
        mean_diff = mean1 - mean2
        
        # Standard weighted update for Gaussian parameters
        # Note: Ideally we use w1, w2 for mixing, but for covariance update term (spread), 
        # using counts (num_pnts) is geometrically more robust for "extent". 
        # However, for the *center* (mean), confidence should matter.
        
        self._means[idx2] = (w1 * mean1 + w2 * mean2) / total_w
        
        # For covariance, we stick to count-based or use weighted count
        # Combining two Gaussians: new_cov = (w1*C1 + w2*C2)/W + (w1*w2/W^2)*(m1-m2)(m1-m2)^T
        self._covs[idx2] = (w1 * cov1 + w2 * cov2) / total_w + (w1 * w2 / total_w**2) * np.outer(mean_diff, mean_diff)
        self._rels[idx2, self._valid_mask] += self._rels[idx1, self._valid_mask]
        self._rels[self._valid_mask, idx2] += self._rels[self._valid_mask, idx1]
        self._pcd[idx2] = np.concatenate((self._pcd[idx1], self._pcd[idx2]))
        
        # Entropy-Based Label Gating
        # Weight class updates by Inverse Entropy.
        # High Entropy (Uncertain) -> Low Weight.
        # W = N_points * exp(-Entropy)
        
        def compute_entropy(probs):
             # 1. Normalize ensure sum=1
             p_sum = np.sum(probs)
             if p_sum < 1e-9: return 10.0 # High entropy for zeros
             p = probs / p_sum
             # 2. Compute Entropy
             return -np.sum(p * np.log(p + 1e-9))

        h1 = compute_entropy(dist1)
        h2 = compute_entropy(dist2)
        
        # Beta factor controls sensitivity (e.g. 1.0)
        ent_w1 = num_pnts1 * np.exp(-h1)
        ent_w2 = num_pnts2 * np.exp(-h2)
        
        total_ent_w = ent_w1 + ent_w2
        if total_ent_w < 1e-9: total_ent_w = 1.0 # Safety
        
        # update class distribution using ENTROPY WEIGHTS
        # But we must be careful: if we merge, we want to keep the history of counts?
        # Standard logic was: (N1*d1 + N2*d2) / (N1+N2).
        # We replace N with Entropy-Waited Count.
        
        self._classes[idx2] = (ent_w1 * dist1 + ent_w2 * dist2) / total_ent_w
        self._classes[idx1] = np.nan
        #
        self._means[idx1] = np.nan
        self._covs[idx1] = np.nan
        self._rels[idx1, self._valid_mask] = 0
        self._rels[self._valid_mask, idx1] = 0
        self._pcd[idx1] = None
        self._valid_mask[idx1] = False

    def apply_bayesian_fusion(self, priors, rel_class_names, obj_class_names, alpha=0.5, entropy_thresh=1.0):
        """
        UDBF: Fuses Visual Predictions with LLM Semantic Priors based on Uncertainty (Entropy).
        priors: Dict {(subj_cls, obj_cls): {pred_name: score}}
        """
        if not priors: return
        print(f"UDBF: Applying Bayesian Fusion with alpha={alpha}, thresh={entropy_thresh}...")
        
        name_to_idx = {name: i for i, name in enumerate(rel_class_names)}
        valid_indices = np.nonzero(self._valid_mask)[0]
        n_fused = 0
        
        positions = self._means[valid_indices]
        classes = [obj_class_names[np.argmax(self._classes[i])] for i in valid_indices]
        
        # 1. Softmax on existing relations to get P(Vision)
        # self._rels shape: (N, N, K) - sparse but stored in huge matrix?
        # Actually _rels is full matrix.
        
        for i in range(len(valid_indices)):
            idx1 = valid_indices[i]
            cls1 = classes[i].lower()
            p1 = positions[i]
            
            for j in range(len(valid_indices)):
                if i == j: continue
                idx2 = valid_indices[j]
                cls2 = classes[j].lower()
                p2 = positions[j]
                
                # Check distance gate first (don't fuse far objects)
                dist = np.linalg.norm(p1 - p2)
                if dist > 1.5: continue
                
                # 2. Get Vision Logits & Probabilities
                logits = self._rels[idx1, idx2] # Shape (K,)
                # Ensure existing rels are non-negative for probability calculation? 
                # Actually logits can be negative if they were log-probs, but here they are 'scores' (0 or 10).
                # If existing is < 0, it corrupts data. 
                # But we assume they are >= 0 based on init.
                
                # Check for stability
                max_logit = np.max(logits)
                if np.isnan(max_logit): max_logit = 0
                
                exp_logits = np.exp(logits - max_logit)
                sum_exp = np.sum(exp_logits)
                if sum_exp < 1e-9:
                    p_vis = np.ones_like(logits, dtype=float) / len(logits) # Fallback uniform
                else:
                    p_vis = exp_logits / sum_exp
                
                # 3. Compute Entropy
                entropy = -np.sum(p_vis * np.log(p_vis + 1e-9))
                
                # 4. Check Prior Existence
                prior_key = (cls1, cls2)
                if prior_key not in priors:
                    # Still need to ensure no NaNs in current rels if we touched them?
                    continue
                    
                prior_map = priors[prior_key]
                
                # 5. Fusion Gate: Uncertainty OR Strong Prior?
                # Max Entropy = log(K). 
                k_classes = len(logits)
                max_entropy = np.log(k_classes) if k_classes > 1 else 1.0
                
                vis_pred_idx = np.argmax(logits)
                vis_pred_cls = rel_class_names[vis_pred_idx]
                is_none_pred = (vis_pred_idx == 0) # Assuming 0 is 'None' or 'background'

                # --- ADDITIVE BOOSTING FUSION (World Model Context) ---
                # Reverting Absolute Guard & Interpolation.
                # Strategy: Geometry is the BASE. V-JEPA provides sparse BOOSTS.
                # Formula: P_final = P_geometry + alpha * P_vjepa
                # If P_vjepa is 0 (Silence/Noise Filtered), P_final = P_geometry. (No Dilution)
                
                # 6. Construct Prior Vector (SPARSE)
                # Initialize with ZEROS. Unmentioned classes get 0 boost.
                p_boost = np.zeros_like(p_vis) 
                
                dz = p1[2] - p2[2] 
                
                for pred, score in prior_map.items():
                    if pred not in name_to_idx: continue
                    
                    # === GEOMETRIC PRIOR: P(Geometry | Class) ===
                    # Instead of hard-filtering, we apply a soft probability mask.
                    # This acts as a "Physical Reality Check" for the World Model.
                    
                    def compute_geometric_prob(pred_cls, dz, dist):
                        # Sigmoid-like soft gating parameters
                        # K=50.0 makes the transition very sharp (mimicking hard heuristic)
                        k = 50.0 
                        
                        if pred_cls in ['supported by', 'standing on', 'build in']:
                            # Should be above (dz > -0.1). Prob drops if too far below.
                            # Center slightly adjusted to -0.12 to match heuristic behavior
                            return 1.0 / (1.0 + np.exp(-k * (dz - (-0.12))))
                            
                        elif pred_cls == 'hanging on':
                            # Should be below (dz < 0.2). Prob drops if too far above.
                            # Center adjusted to 0.22
                            return 1.0 / (1.0 + np.exp(k * (dz - 0.22)))
                            
                        elif pred_cls in ['attached to', 'connected to', 'part of']:
                            # Proximity check using Gaussian decay on distance
                            # Stricter proximity (Sigma=0.4m)
                            return np.exp(-0.5 * (dist**2) / (0.4**2))
                        
                        # Non-geometric classes (visual/semantic) have uniform prior P=1.0
                        return 1.0

                    # Calculate geometry score (0.0 ~ 1.0)
                    dist_val = np.linalg.norm(p1 - p2)
                    geo_prob = compute_geometric_prob(pred, dz, dist_val)
                    
                    # Refine Score: P(Final) ~ P(Vision) * P(Geometry)
                    # We treat the input 'score' as the prior confidence from World Model
                    
                    pidx = name_to_idx[pred]
                    val = max(0.0, float(score))
                    
                    # Apply Soft Gating
                    val = val * geo_prob
                    
                    # Filter out physically impossible relations (Prob < 0.1)
                    # Increased cutoff to aggressively remove low-probability relations
                    if val < 0.1: continue

                    p_boost[pidx] = val
                        
                # 7. Apply Additive Update
                # Alpha controls the strength of the boost
                boost_strength = 0.5 # Conservative boost
                
                # Add boost to visual probabilities
                p_fused = p_vis + (boost_strength * p_boost)
                
                # Renormalize? 
                p_fused = p_fused / np.sum(p_fused)
                
                # 8. Update _rels
                # Assign back to Int array. 0 to 10 scale.
                p_fused_norm = p_fused / np.max(p_fused)
                new_scores = p_fused_norm * 10.0
                new_scores = np.nan_to_num(new_scores, 0) # Safety
                new_scores = np.clip(new_scores, 0, 100) # Ensure no negatives
                
                self._rels[idx1, idx2] = new_scores.astype(int)
                n_fused += 1
                
        print(f"Fused {n_fused} relation pairs.") 

    def _batched_distribution_distance(self, p, q_batch, metric='kl'):
        """
        Compute distance between one probability distribution p and a batch of distributions q_batch.
        p is a 1D array. q_batch is a 2D array (N, K).
        Returns an array of distances (N,).
        """
        # Ensure p and q_batch are valid probability distributions
        p = np.clip(p, 1e-12, 1.0)
        q_batch = np.clip(q_batch, 1e-12, 1.0)
        
        if metric == 'hellinger':
            return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q_batch)) ** 2, axis=1))
        elif metric == 'kl':
            return np.sum(p * np.log(p / q_batch), axis=1)
        elif metric == 'js':
            m = 0.5 * (p + q_batch) # m shape is (N, K)
            kl_p_m = np.sum(p * np.log(p / m), axis=1) # shape (N,)
            kl_q_m = np.sum(q_batch * np.log(q_batch / m), axis=1) # shape (N,)
            return 0.5 * kl_p_m + 0.5 * kl_q_m
        elif metric == 'l2':
            return np.sqrt(np.sum((p - q_batch) ** 2, axis=1))
        elif metric == 'dot_product':
            return 1.0 - np.sum(p * q_batch, axis=1)
        elif metric == "top_class":
            p_top = np.argmax(p)
            q_top = np.argmax(q_batch, axis=1)
            return (p_top != q_top).astype(float)
        else:
            raise ValueError(f"Unknown metric: {metric}")


    def apply_structural_completion(self, obj_class_names, rel_class_names, dist_tol=0.2, angle_tol=30.0):
        """
        Structural Pattern Completion (Visual Autocomplete).
        Finds "Anchor" relations (high confidence) and propagates them to 
        geometrically similar pairs that are missing relations.
        """
        print("Running Structural Pattern Completion...")
        valid_indices = np.nonzero(self._valid_mask)[0]
        positions = self._means[valid_indices]
        classes = [obj_class_names[np.argmax(self._classes[i])].lower() for i in valid_indices]
        
        # 1. Mine Anchor Motifs
        # Format: (SubjCls, ObjCls) -> List of {pred_idx, rel_vec, score}
        anchors = {}
        n_anchors = 0
        
        for i in range(len(valid_indices)):
            for j in range(len(valid_indices)):
                if i == j: continue
                idx1, idx2 = valid_indices[i], valid_indices[j]
                
                logits = self._rels[idx1, idx2]
                pred_idx = np.argmax(logits)
                score = logits[pred_idx]
                
                # Check if this is a strong anchor
                # 0 is usually 'None' or 'No Relation'
                if pred_idx == 0: continue 
                if score < 8.0: continue # Must be very confident
                
                cls1, cls2 = classes[i], classes[j]
                key = (cls1, cls2)
                
                p1, p2 = positions[i], positions[j]
                rel_vec = p2 - p1 # Vector from Subj to Obj
                
                if key not in anchors: anchors[key] = []
                anchors[key].append({
                    'pred_idx': pred_idx,
                    'vec': rel_vec,
                    'score': score
                })
                n_anchors += 1
                
        print(f"Mined {n_anchors} geometric anchors.")
        
        # 2. Geometric Analogy Completion
        n_completed = 0
        
        for i in range(len(valid_indices)):
            for j in range(len(valid_indices)):
                if i == j: continue
                idx1, idx2 = valid_indices[i], valid_indices[j]
                
                # Skip if already confident
                current_logits = self._rels[idx1, idx2]
                current_idx = np.argmax(current_logits)
                current_score = current_logits[current_idx]
                
                # We want to fix "None" or "Low Confidence"
                if current_idx != 0 and current_score > 5.0: continue
                
                cls1, cls2 = classes[i], classes[j]
                key = (cls1, cls2)
                
                if key not in anchors: continue
                
                p1, p2 = positions[i], positions[j]
                target_vec = p2 - p1
                target_dist = np.linalg.norm(target_vec)
                
                # Check against all anchors for this class pair
                best_match = None
                min_vec_diff = 1e9
                
                for anchor in anchors[key]:
                    anchor_vec = anchor['vec']
                    anchor_dist = np.linalg.norm(anchor_vec)
                    
                    # Distance check
                    if abs(target_dist - anchor_dist) > dist_tol: continue
                    
                    # Angle check (Cosine Sim)
                    # dot = |a||b|cos(theta)
                    if target_dist < 1e-3 or anchor_dist < 1e-3: continue
                    
                    cos_sim = np.dot(target_vec, anchor_vec) / (target_dist * anchor_dist)
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_sim))
                    
                    if angle > angle_tol: continue
                    
                    # Found a match! Use the one with best vector alignment
                    vec_diff = np.linalg.norm(target_vec - anchor_vec)
                    if vec_diff < min_vec_diff:
                        min_vec_diff = vec_diff
                        best_match = anchor
                
                # Apply Completion
                if best_match is not None:
                    # Inject the relationship
                    # print(f"SPC: Autocompleting {cls1}-{cls2} with {best_match['pred_idx']} (Match Dist {min_vec_diff:.2f})")
                    
                    # Reset logits to low
                    self._rels[idx1, idx2] = np.zeros_like(current_logits)
                    
                    # Set target class to High Score
                    # We copy the anchor's score slightly discounted
                    new_score = best_match['score'] * 0.95 
                    self._rels[idx1, idx2, best_match['pred_idx']] = int(new_score)
                    
                    n_completed += 1
                    
        print(f"Structure-Completed {n_completed} relations.")



    def get_sparse_graph_text(self, obj_class_names, rel_class_names):
        """
        HGR: Serializes the current graph into a text format for LLM inference.
        Returns a tuple: (text_description, id_map)
        """
        valid_indices = np.nonzero(self._valid_mask)[0]
        positions = self._means[valid_indices]
        classes = [obj_class_names[np.argmax(self._classes[i])].lower() for i in valid_indices]
        
        id_map = {i: original_idx for i, original_idx in enumerate(valid_indices)}
        
