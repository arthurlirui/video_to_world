# utils/ — Utilities

Shared utility modules used across the entire pipeline.

---

## `__init__.py`

Exposes a minimal stable API surface via `__all__`:
- **Logging:** `get_logger`, `try_create_tensorboard_writer`, `tb_log_hparams`
- **Image:** `colors_to_intensity`, `build_intrinsic_matrix`
- **Geometry:** Full SE(3)/SO(3)/quaternion API (`hat`, `vee`, `so3_exp`, `so3_log`, `se3_exp`, `se3_log`, `se3_apply`, `se3_inverse`, `compose_rt`, `compose_se3`, `quaternion_multiply`, `quaternion_to_rotation_matrix`, `rotation_matrix_to_quaternion`, `normal_to_quaternion`, ...)

---

## `geometry.py`

**Purpose:** Rigid-motion / Lie-group utilities with small-angle-stable implementations. The mathematical foundation of the entire pipeline.

### Quaternion Operations
- `quaternion_multiply(q1, q2)` — Hamilton product.
- `quaternion_to_rotation_matrix(q)` — Quaternion → 3×3 rotation matrix.
- `rotation_matrix_to_quaternion(R)` — 3×3 rotation matrix → quaternion (Shepperd's method).
- `normal_to_quaternion(normal, reference)` — Constructs a quaternion aligning a reference direction to a target normal.

### SO(3) Operations
- `hat(v)` — Skew-symmetric matrix from 3-vector.
- `vee(S)` — Extract 3-vector from skew-symmetric matrix.
- `so3_exp(xi)` — SO(3) matrix exponential (Rodrigues' formula with small-angle stabilization).
- `so3_log(R)` — SO(3) matrix logarithm.
- `so3_left_jacobian(xi)` / `so3_left_jacobian_inv(xi)` — Left Jacobian and its inverse.

### SE(3) Operations
- `se3_exp(xi)` — SE(3) matrix exponential from 6D twist. Returns `(R, t)`.
- `se3_log(R, t)` — SE(3) matrix logarithm. Returns 6D twist.
- `se3_inverse(R, t)` — Closed-form inverse of a rigid transform.
- `se3_apply(xi, points)` — Apply SE(3) twist to points: `p' = p @ R.T + t`.

### Composition
- `rt_apply(points, R, t)` — Apply rotation + translation to points.
- `compose_rt(R1, t1, R2, t2)` — Compose two rigid transforms.
- `compose_se3(xi1, xi2)` — Compose two SE(3) twists.

**Convention:** Row-vector — `p' = p @ R.T + t`.

**Dependencies:** `torch` only.

---

## `knn.py`

**Purpose:** Nearest-neighbor queries with pluggable backends.

### `query_knn_with_backend(query_points, reference_points, K, backend, cpu_tree, gpu_tree)`
Unified dispatcher selecting the appropriate backend:
- `"cpu_kdtree"` — scipy `cKDTree`.
- `"gpu_kdtree"` — `torch_kdtree` (GPU KD-tree).
Returns `(indices, squared_distances)`.

### Backend Implementations
- `build_kdtree(points)` — Build a scipy `cKDTree`.
- `nearest_neighbors_kdtree(query, tree, K)` — Query a scipy KD-tree.
- `build_torch_kdtree(points)` — Build a `torch_kdtree` GPU KD-tree.
- `nearest_neighbors_torch_kdtree(query, tree, K)` — Query a GPU KD-tree.
- `nearest_neighbors(query, reference, K, chunk_size)` — Brute-force fallback using chunked `torch.cdist`.

**Dependencies:** `scipy.spatial.cKDTree`, `torch`, `numpy`. Lazy import `torch_kdtree`.

---

## `normals.py`

**Purpose:** Per-point normal estimation via PCA/SVD over KNN neighborhoods.

### `estimate_normals(points, k, backend, viewpoint, tree, index_range)`
- Supports `cpu_kdtree` (scipy) and `gpu_kdtree` (torch_kdtree) backends.
- Optional pre-built tree reuse and index-range subsetting.
- Handles degenerate/NaN/inf cases with an Open3D fallback (`_estimate_normals_open3d_fallback`).
- Optional normal orientation toward a viewpoint.
- Batched processing for memory efficiency.
- Detailed timing logs.
- Returns `(normals, tree)`.

**Dependencies:** `utils.logging`, `utils.knn`, `scipy`, `open3d`, `numpy`, `torch`.

---

## `pointcloud.py`

**Purpose:** Point-cloud merging utilities for incremental model construction in Stage 1.

### `merge_new_points_with_model(model_points, model_colors, model_normals, new_points, new_colors, voxel_size, color_thresh, knn_backend, ...)`
Merges new points into an existing model without modifying existing points:
1. Voxel-hash existing and new point coordinates.
2. Drop new points that fall in already-occupied voxels (optionally keep if color difference exceeds `color_thresh`).
3. Downsample kept new points to voxel centroids.
4. Rebuild KD-tree on the merged cloud.
5. Estimate normals only for the new points (reusing existing normals for old points).
6. Returns `(points, colors, normals, keep_mask, kd_tree)`.

### `merge_point_clouds(pcls)`
Simple Open3D concatenation helper for a list of point clouds.

**Dependencies:** `utils.knn`, `utils.logging`, `utils.normals`, `open3d`, `numpy`, `torch`.

---

## `image.py`

**Purpose:** Image and camera utilities.

### `colors_to_intensity(colors)`
Converts RGB (or single-channel) colors to scalar intensity using luma weights (0.2989, 0.5870, 0.1140).

### `build_intrinsic_matrix(log_focal, principal_point_normalized, width, height, focal_clamp)`
Reconstructs a 3×3 intrinsic matrix from learnable parameters:
- Focal length (exp of learnable log-focal, clamped).
- Principal point (normalized → pixel coordinates).

**Dependencies:** `torch` only.

---

## `downsample.py`

**Purpose:** Point-cloud downsampling utilities.

### `voxel_grid_downsample(points, colors, voxel_size)`
Averages points (and colors) within uniform voxel cells using scatter-add. Returns downsampled points and colors.

### `downsample_to_target(points, colors, target_count, tolerance)`
Incrementally grows the voxel size from a tiny start until the point count falls within a tolerance window of the target. Deterministically chooses the closest result without random subsampling. Returns downsampled points and colors.

**Dependencies:** `torch` only.

---

## `logging.py`

**Purpose:** Shared logging helpers.

### `get_logger(name)`
Returns an idempotent module logger with a tqdm-compatible stream handler (`_TqdmCompatibleStreamHandler` that uses `tqdm.write` to avoid corrupting progress bars).

### `try_create_tensorboard_writer(log_dir)`
Best-effort creation of a `SummaryWriter`. Returns the writer or None if TensorBoard is unavailable.

### `tb_log_hparams(writer, hparams, step)`
Logs hyperparameters to TensorBoard as a JSON blob and a Markdown table.

**Dependencies:** Standard library only. Lazy imports `tqdm`, `torch.utils.tensorboard`.

---

## `export_checkpoint_to_ply.py`

**Purpose:** Script that exports a trained 3DGS checkpoint to a standard 3DGS PLY file.

### `main(config: ExportGSCheckpointToPlyConfig)`
1. Loads the checkpoint, reconstructs a `CanonicalGSModel` via `eval_gs._build_model`.
2. Extracts canonical points, SH coefficients (DC + rest), log scales, logit opacities, and quaternions.
3. Optionally subsamples to `max_points`.
4. Writes a binary PLY with standard 3DGS vertex properties using `plyfile`.

**Dependencies:** `configs.utils`, `data.data_loading`, `eval_gs`, `plyfile`, `tyro`.

---

## `view_checkpoint.py`

**Purpose:** Interactive viewer script for inspecting a trained GS checkpoint in a browser.

### `main(config: ViewGSCheckpointConfig)`
1. Loads the checkpoint, builds a `CanonicalGSModel` via `eval_gs._build_model`.
2. Starts a Viser + nerfview `Viewer` with custom GUI controls:
   - Near/far planes, background color.
   - Render mode: RGB, depth, normal, alpha.
   - 3DGS rasterize mode, max SH degree.
3. `render_fn` performs live 2DGS or 3DGS rasterization via gsplat with closed-form w2c.
4. Includes depth colormap visualization.
5. Optionally overlays training camera frustums.

**Dependencies:** `configs.utils`, `data.data_loading`, `eval_gs`, `viser`, `nerfview`, `gsplat`, `tyro`.
