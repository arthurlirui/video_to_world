# models/ — Neural Models

Neural network models for Gaussian Splatting, deformation fields, and feature matching.

---

## `__init__.py`

Empty package init.

---

## `canonical_gs_model.py`

**Purpose:** Defines the canonical Gaussian Splatting model that holds learnable Gaussian parameters and renders images via gsplat.

### `CanonicalGSModel(nn.Module)`

The central model for Stage 3.2. Holds:
- **Canonical point positions** — Learnable 3D coordinates (the canonical Gaussian means).
- **SH color coefficients** — DC (`sh_dc`) and higher-order (`sh_rest`) spherical harmonics for view-dependent color.
- **Logit opacities** — Per-point opacity (before sigmoid).
- **Log scales** — Per-point scale (before exp).
- **Quaternions** — Per-point rotation.
- **Per-frame c2w SE(3) twists** — Learnable per-frame camera poses (Frame 0 frozen).
- **Frozen local `DeformationGrid`s** — Per-frame non-rigid deformation fields.
- **Frozen `ViewConditionedInverseDeformation`** — Inverse deformation network for warping canonical Gaussians into per-view camera space.

#### Key Methods

**Camera helpers:**
- `get_c2w(frame_idx)` — Returns the c2w matrix for a frame (via `se3_exp`).
- `get_viewmat(frame_idx)` — Returns the w2c view matrix (inverse of c2w).

**Inverse deformation:**
- `inverse_deform_points(points, frame_idx)` — Warps canonical points to camera space using the inverse deformation network.
- `inverse_deform_gaussians(frame_idx)` — Warps all Gaussian attributes (means, quats, scales) into a frame's camera space. Handles rotation composition for quaternions and scale transformation.

**Rendering:**
- `render_2dgs(points, viewmat, intrinsics)` — Renders via gsplat 2DGS rasterizer. Returns dict with `rgb`, `alpha`, `normals`, `surf_normals`, `distort`, `depth`.
- `render_3dgs(points, viewmat, intrinsics)` — Renders via gsplat 3DGS rasterizer. Returns dict with `rgb`, `alpha`, `depth`.
- `render_frame(frame_idx, intrinsics, use_inverse_deform)` — High-level rendering entry point. Optionally applies inverse deformation before rendering.
- `_rasterize_2dgs_from_gaussians(...)` / `_rasterize_3dgs_from_gaussians(...)` — Low-level rasterization from explicit Gaussian attributes.

#### Helper Functions
- `rgb_to_sh0(rgb)` — Converts RGB to SH DC coefficient.
- `sh0_to_rgb(sh0)` — Converts SH DC coefficient to RGB.

**Dependencies:** `models.deformation`, `utils.geometry` (quaternion, SE3 ops), `gsplat` (lazy import).

---

## `deformation.py`

**Purpose:** Deformation models for non-rigid alignment and inverse deformation.

### `DeformationGrid(nn.Module)`

Hash-grid MLP (via `tinycudann`) that predicts per-point forward SE(3) twists for non-rigid deformation.

- Input: 3D point coordinates (normalized to unit bounding box).
- Output: 6D SE(3) twist (rotation + translation).
- Translation is scaled from normalized space back to world space via the bounding box.
- Architecture: `tinycudann.Encoding` (hash grid) → MLP with configurable layers/neurons.

### `ViewConditionedInverseDeformation(nn.Module)`

Shared-weight network for inverse deformation (canonical → camera space).

- Input: Canonical-space point + view index (embedded as learnable vector).
- Encoding: Spatial hash-grid (`tinycudann`) concatenated with view embedding.
- Output: 6D SE(3) twist for the inverse deformation.
- `inverse_warp(points, view_idx)` — Applies the inverse deformation to points.

### `FullInverseDeformationModel(nn.Module)`

Wrapper combining frozen forward deformations (c2w + local grids) with a learned inverse network.

- `forward_deform(cam_pts, view_idx)` — Forward path: camera → local deform → c2w → canonical.
- `inverse_deform(canonical_pts, view_idx)` — Inverse path (canonical → canonical without local deform). Detaches the c2w⁻¹ step so the inverse-local sees fixed camera-space input.
- `inverse_deform_to_camera(canonical_pts, view_idx)` — Full inverse: canonical → inverse local → c2w → camera.
- `get_inverse_twist(canonical_pts, view_idx)` — Returns the raw SE(3) twist from the inverse network.

### Helper Functions
- `_compute_growth_factor(min_res, max_res, num_levels)` — Computes per-level growth factor for the hash grid.
- `_require_tcnn()` — Validates tinycudann availability.
- `_normalize_points_to_unit_bbox(points, bbox_min, bbox_max)` — Normalizes points to [-1, 1] for hash-grid input.

**Dependencies:** `utils.geometry` (`se3_apply`, `se3_inverse`), `tinycudann` (lazy import), `numpy`, `torch`.

---

## `roma_matcher.py`

**Purpose:** RoMa/RoMaV2 dense feature matching wrapper and utilities for computing inter-frame correspondences.

### `RoMaMatchData`
Container for per-pair match results:
- `src_frame_idx`, `ref_frame_idx` — Frame indices.
- `kpts_src`, `kpts_ref` — Matched keypoint pixel coordinates (Nx2 tensors).
- `certainty` — Per-match certainty scores.

### `RoMaMatcherWrapper`
Loads RoMa v1 or v2 models and provides:
- `match_images(src_img, ref_img)` — Runs dense matching, returns keypoints and certainty.

### `get_local_indices_for_pixels_batch(pixel_coords, valid_pixel_indices, image_height, image_width)`
Maps pixel coordinates to local indices within the filtered valid-pixel set. Uses sorted search for efficient batched lookup. Critical for connecting 2D RoMa matches to 3D point indices.

### `select_reference_frames(current_idx, num_frames, max_references, mode)`
Temporal coverage sampling for reference frame selection:
- `"recent_and_strided"` — Mix of recent frames and uniformly strided frames.
- `"strided"` — Uniformly strided frames.

### Match Caching
- `_get_cache_key(...)` — Deterministic cache key from RoMa config and frame sampling.
- `_get_cache_path(root_path, cache_key)` — Cache file path.
- `load_cached_matches(path, device)` — Loads cached matches from disk.
- `save_matches_to_cache(path, new_matches, existing_cache)` — Saves newly computed matches, merging with existing cache.

### `compute_roma_matches_for_frame(roma_matcher, images, current_frame_idx, ...)`
Computes or loads cached RoMa matches for a frame against selected reference frames. Returns a list of `RoMaMatchData`.

### `MatchHistory`
Accumulates all matches across frames with query helpers:
- `add_matches(matches)` — Add new match results.
- `__len__()` — Total number of match sets.
- `all_matches` — List of all `RoMaMatchData`.

**Dependencies:** `torch`, `romav2`/`romatch` (lazy imports), `PIL`, `numpy`.
