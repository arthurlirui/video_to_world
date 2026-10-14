# losses/ — Loss Functions

All loss functions used across the pipeline stages.

---

## `__init__.py`

Re-exports the loss API:
- `PixelReconstructionLossWeights`, `init_lpips`, `pixel_reconstruction_loss` (from `rendering`)
- `tv_loss` (from `tv`)
- `compute_correspondence_loss_with_model_segments` (from `correspondence`)
- `normal_consistency_loss`, `distortion_loss` (from `gaussian`)

---

## `rendering.py`

**Purpose:** Shared image reconstruction losses for GS training.

### `PixelReconstructionLossWeights`
Dataclass with `l1: float` and `lpips: float` weights.

### `init_lpips(device)`
Creates a frozen, eval-mode LPIPS network for perceptual loss. Returns the network or None if LPIPS is not available.

### `pixel_reconstruction_loss(rendered, gt, weights, clamp)`
Computes a combined reconstruction loss:
- MSE (always included)
- L1 (weighted by `weights.l1`)
- LPIPS (weighted by `weights.lpips`, expects inputs in [-1, 1] range)

Returns `(total_loss, component_losses_dict)`.

**Dependencies:** `torch`, `torch.nn.functional`, `lpips` (lazy import).

---

## `tv.py`

**Purpose:** Total-Variation regularization for SE(3) deformation fields. Encourages spatial smoothness of the predicted per-point twists.

### `build_voxel_grid(bbox_min, bbox_max, voxel_size)`
Constructs a grid of voxel-center sample points within the bounding box.

### `voxel_neighbors_6(grid_shape)`
Builds 6-neighborhood index pairs (±x, ±y, ±z) for a 3D voxel grid.

### `tv_loss(deform_net, bbox_min, bbox_max, voxel_size, tv_every_k, sample_ratio, points, ...)`
Computes TV regularization by evaluating the deformation network at sample points and their 6-voxel-offset neighbors, then computing the mean squared twist difference across edges.

Two modes:
- **Point-based** (if `points` provided): Samples at input points plus optional jittered points, finds voxel-offset neighbors.
- **Voxel-grid fallback**: Subsamples voxel-grid centers.

The `sample_ratio` parameter controls what fraction of edges are evaluated per call (stochastic subsampling for efficiency).

**Dependencies:** `torch` only (takes the deformation network as a callable argument).

---

## `correspondence.py`

**Purpose:** Geometry and color correspondence loss for RoMa-matched keypoint pairs, agnostic to how correspondences were generated.

### `compute_correspondence_loss_with_model_segments(src_points, ref_points, roma_matches_data, model_frame_segments, ref_frame_indexes, src_valid_pixel_indices, model_valid_pixel_indices_list, image_height, image_width, max_corr_dist, ...)`
1. Maps RoMa keypoint pixel coordinates to local point indices via `get_local_indices_for_pixels_batch`.
2. Gathers transformed source and reference (model-segment) 3D points.
3. Optionally filters by maximum correspondence distance.
4. Computes a certainty-weighted mean squared 3D distance, gated by a global max-certainty factor.
5. Optionally adds a weighted color-difference term.

Returns `(geometry_loss, color_loss, num_matches)`.

**Dependencies:** `models.roma_matcher.get_local_indices_for_pixels_batch`.

---

## `gaussian.py`

**Purpose:** 2DGS-specific loss functions for Gaussian Splatting training.

### `normal_consistency_loss(normals, surf_normals, alphas)`
Penalizes disagreement between rendered Gaussian normals and depth-derived surface normals. Masked by alpha to only penalize visible surface regions.

### `distortion_loss(distort)`
Mean of the gsplat 2DGS distortion map. Encourages focused, non-spread-out Gaussian splats.

### `opacity_regularization_loss(opacities)`
Binary-cross-entropy-with-self to push opacities toward 0 or 1 (binary occupancy).

### `scale_regularization_loss(log_scales, threshold)`
Penalizes log-scales above a threshold, preventing Gaussians from growing too large.

### `depth_loss(rendered_depth, gt_depth, valid_mask, alphas)`
L1 loss between rendered expected depth and ground-truth camera-z depth, masked by validity and alpha.

**Dependencies:** `torch`, `torch.nn.functional`. No project-internal imports.
