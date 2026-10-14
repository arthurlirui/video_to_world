# data/ — Data Loading & Checkpoint I/O

Modules for loading DA3 preprocessed data and serializing/deserializing pipeline checkpoints.

---

## `__init__.py`

Empty package init.

---

## `data_loading.py`

**Purpose:** Core data-loading module. Reads DA3 NPZ predictions, back-projects depth maps to point clouds, applies confidence filtering, and provides camera/image loading utilities.

### Key Functions

#### `load_data(root_path, num_frames, stride, device, conf_thresh_percentile, conf_mode, ...)`
The primary data loading entry point. Loads DA3 predictions from `exports/npz/results.npz`:
1. Reads depth maps, confidence maps, extrinsics (w2c), intrinsics, and images.
2. Back-projects depth maps to world-space colored point clouds via `depths_to_world_points_with_colors`.
3. Applies confidence filtering across multiple modes:
   - `"global"` — single percentile threshold across all frames.
   - `"per_frame"` — per-frame percentile.
   - `"per_frame_guided"` — per-frame with global guidance.
   - `"voxel"` — voxel-level filtering via `_voxelized_conf_filter_da3`.
   - `"voxel_guided"` — voxel with global guidance.
4. Caches filtered per-frame PLYs and valid pixel indices to disk.
5. Returns: point clouds, extrinsics, intrinsics, images, valid pixel indices, depth/conf maps, and optionally original-resolution images/intrinsics.

#### `load_da3_camera_images(root_path, num_frames, stride, device, ...)`
Lightweight camera/image loading without point cloud construction. Returns images, c2w poses, and intrinsics.

#### `load_depth_maps_da3(root_path, num_frames, stride, device)`
Loads only depth maps and confidence maps from the NPZ.

#### `load_da3_original_images_from_folder(root_path, images_dir, num_frames, stride, device)`
Loads original-resolution images from a folder (typically `frames_subsampled/`). Scales intrinsics to match the original image resolution.

#### `load_nerf_transforms_json(path, device, override_width, override_height, blender_opengl_to_opencv)`
Parses a NeRF-style `transforms.json` camera trajectory file. Returns c2w poses, intrinsics, width, height. Optionally converts Blender/OpenGL convention to OpenCV.

#### `load_worldexplorer_camera_path_json(path, device)`
Parses a WorldExplorer camera path JSON file.

### Helper Functions
- `depths_to_world_points_with_colors(depth, conf, extrinsic, intrinsic, image)` — Back-projects a depth map to a colored world-space point cloud.
- `torch_to_o3d_pcd(points, colors)` — Converts torch tensors to an Open3D PointCloud.
- `_voxelized_conf_filter_da3(...)` — Voxel-level confidence filtering.

**Dependencies:** `utils.logging`, `open3d`, `numpy`, `torch`, `PIL`, `torchvision`, `tqdm`.

---

## `checkpoint_loading.py`

**Purpose:** Checkpoint serialization/deserialization for the multi-stage pipeline. Handles loading deformation checkpoints, aligned point clouds, RoMa match data, and alignment parameters.

### Key Functions

#### `load_deformation_checkpoints(checkpoint_dir, device, first_local, allow_rigid_fallback)`
Loads per-frame global SE(3) twists and local `DeformationGrid` state dicts from a checkpoint directory.
- `first_local`: Controls how Frame 0's local deformation is handled (`"dummy"`, `"none"`).
- `allow_rigid_fallback`: If True, creates identity deformations when local deformation files are missing.
- Returns: `(per_frame_global_deform, per_frame_local_deform, bbox_min, bbox_max)`.

#### `load_aligned_point_cloud(checkpoint_dir, device)`
Loads the aligned canonical point cloud (points + colors) from `aligned_points.ply`.

#### `load_roma_model_index_data(checkpoint_dir, device)`
Loads RoMa/ICP-filtered model index data: `model_frame_segments` (per-frame point index ranges in the canonical model) and `model_valid_pixel_indices_list`. Returns a `RomaModelIndexData` dataclass or None.

#### `load_alignment_data_params(root_path, run)`
Recovers Stage-1 alignment parameters from the saved `config.json`. Returns an `AlignmentDataParams` containing `num_frames`, `stride`, `offset`, and all confidence-filtering settings. Used by downstream stages to ensure data loading consistency.

#### `load_inverse_local_from_checkpoint(inverse_deform_dir, device)`
Reconstructs a `ViewConditionedInverseDeformation` network from `config.pt` (architecture) and `inverse_local.pt` (weights).

#### `load_roma_match_history(checkpoint_dir, device)`
Loads saved RoMa match history (frame pairs, keypoints, certainty).

#### Other Functions
- `write_point_cloud_ply(path, points, colors)` — Writes a PLY point cloud.
- `load_json_config(path)` / `load_stage_config(root_path, run, subdir)` — JSON config loaders.

### Dataclasses
- `RomaModelIndexData` — Container for `model_frame_segments` and `model_valid_pixel_indices_list`.
- `AlignmentDataParams` — Frozen dataclass holding Stage-1 alignment parameters for downstream reuse.

**Dependencies:** `configs.common.AlignmentDataConfig`, `models.deformation`, `models.roma_matcher`, `open3d`, `numpy`, `torch`.
