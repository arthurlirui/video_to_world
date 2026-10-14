# algos/ — Core Algorithms

Core algorithmic implementations for point cloud alignment and optimization.

---

## `__init__.py`

Re-exports a curated API surface: `DeformationGrid`, `non_rigid_icp`, `tv_loss`, `build_kdtree`, `nearest_neighbors_kdtree`, `estimate_normals`.

---

## `icp.py`

**Purpose:** Rigid colored point-cloud registration using Adam optimization, inspired by Park et al.'s Colored-ICP (ICCV 2017).

**Key function:**
- `colored_icp_adam(src, src_colors, ref, ref_colors, n_iter, lr, knn_backend, max_corr_dist, lambda_geometric, color_k, progress_callback)` — Optimizes a single global SE(3) transform minimizing a weighted combination of:
  - **Point-to-plane geometric residual** — distance from source points to the reference surface along normals.
  - **Photometric residual** — color difference on virtual tangent planes using precomputed per-point color gradients (least-squares fit).
  
  Returns `(aligned_src, R, t)` where R is the rotation and t is the translation.

**Algorithm details:**
1. Estimate normals on the reference point cloud (if not provided).
2. Precompute color gradients on the reference via least-squares fitting over KNN neighborhoods.
3. Build a KD-tree for nearest-neighbor queries.
4. Iteratively: find correspondences → compute combined Jacobian → Adam step on SE(3) twist → apply transform.

**Dependencies:** `utils.image.colors_to_intensity`, `utils.knn`, `utils.normals.estimate_normals`, `utils.geometry.se3_exp`.

---

## `non_rigid_icp.py`

**Purpose:** Core non-rigid ICP alignment for Stage 1. Jointly optimizes a global SE(3) twist and a hash-grid local deformation field.

**Key function:**
- `non_rigid_icp(src, ref, ref_normals, roma_matches_data, ..., n_iter, lr, method, ...)` — Optimizes:
  - `xi_global` — per-frame SE(3) twist (the c2w rigid transform).
  - `DeformationGrid` — hash-grid MLP predicting per-point local SE(3) twists.
  
  Forward chain: `camera_pts → local_deform(·) → se3_apply(c2w, ·) → canonical`

**Loss terms:**
- **Data term** — point-to-point or point-to-plane distance to the reference model.
- **RoMa correspondence loss** — 3D distance between matched keypoints (via `losses.correspondence`).
- **TV regularization** — smoothness of the deformation field (via `losses.tv`).
- **Color ICP** — optional photometric term on matched correspondences.

**Features:**
- Early stopping with configurable patience, min iterations, and min delta.
- Per-point loss diagnostics for downstream filtering.
- Detailed progress callbacks and TensorBoard logging.
- Configurable KNN backend (CPU/GPU KD-tree).

**Returns:** `(aligned_src, global_deform, local_deform)` — the transformed source points, the SE(3) twist, and the deformation module.

**Dependencies:** `models.deformation.DeformationGrid`, `utils.geometry`, `losses.tv`, `losses.correspondence`, `utils.knn`, `utils.normals`, `utils.image`.

---

## `global_optimization.py`

**Purpose:** Stage 2 global optimization via leave-one-out (LOO) consensus. Jointly refines all per-frame deformations.

**Key function:**
- `global_opt(per_frame_world_points, per_frame_world_colors, per_frame_global_rigid, per_frame_local_deform, ...)` — Optimizes all frames simultaneously.

**Loss terms:**
- **LOO point-to-plane** — Frame i's deformed points compared to the model built from all *other* frames. Both sides of correspondences are differentiable.
- **Anchoring** — Penalizes drift from initial deformations (Stage 1 outputs).
- **TV regularization** — Deformation field smoothness.
- **Thin-shell surface sharpening** — Encourages surface-like structure.
- **LOO Colored-ICP** — Optional photometric term.

**Optimization features:**
- Stochastic LOO pair subsampling (configurable max pairs per iteration, pairs per source).
- Chunked deformation application for memory efficiency.
- Periodic KD-tree / normals rebuild.
- Intermediate checkpoint saving.
- Frame 0 frozen as gauge fix.

**Returns:** Dict with `canonical_points`, `canonical_colors`, `per_frame_global_rigid`, `per_frame_local_deform`, `model_frame_segments`.

**Dependencies:** `losses.tv`, `data.data_loading`, `utils.geometry`, `utils.knn`, `utils.normals`, `utils.image`.
