# Module Reference

Detailed documentation for every module and source file in the project. Each entry describes the file's purpose, key classes/functions, and dependencies.

## Table of Contents

- [Top-Level Pipeline Scripts](#top-level-pipeline-scripts)
- [algos/ — Core Algorithms](algos.md)
- [configs/ — Configuration Dataclasses](configs.md)
- [data/ — Data Loading & Checkpoint I/O](data.md)
- [losses/ — Loss Functions](losses.md)
- [models/ — Neural Models](models.md)
- [utils/ — Utilities](utils.md)
- [patches/ — Third-Party Patches](patches.md)

---

## Top-Level Pipeline Scripts

These are the entry points for running the pipeline stages. Each is a standalone script with a tyro CLI.

### `run_reconstruction.py`

**Purpose:** End-to-end pipeline orchestrator. Chains Stage 0 through Stage 3.2 for a single scene.

**Key class:** `PipelineConfig` — dataclass with all pipeline-level settings (input video/frames, mode preset, renderer selection, stage skip flags, and nested stage configs).

**Key functions:**
- `main(config)` — Orchestrates all stages, managing scene root resolution, mode presets, and stage-to-stage config injection.
- `_iter_config_overrides(prefix, cfg, defaults)` — Converts dataclass differences into tyro-style CLI arguments for subprocess invocation.
- `_find_subdir(parent, prefix)` — Finds the most recent output subdirectory matching a prefix.
- `_run(cmd, dry_run)` — Executes a subprocess command with formatted logging.

**Mode presets:** `fast` (default) skips Stage 2, trains fewer epochs, 3DGS only. `extensive` runs all stages with both 2DGS + 3DGS.

**Dependencies:** `configs.stage1_align`, `configs.stage2_global_optimization`, `configs.stage3_gs`, `configs.stage3_inverse_deformation`, `tyro`.

---

### `preprocess_video.py`

**Purpose:** Stage 0 — DA3 preprocessing. Converts a video or frame folder into per-frame depth maps, confidence maps, camera poses, and point clouds via Depth Anything 3.

**Key functions:**
- `main()` — Entry point (argparse CLI). Resolves scene root, extracts/subsamples frames, runs DA3 inference, saves results.
- `extract_frames(input_video, frames_dir, image_ext)` — Uses ffmpeg to extract frames from a video.
- `subsample_frames(images, max_frames, max_stride)` — Selects a subset of frames respecting max count and stride constraints.

**Output:** `<scene_root>/exports/npz/results.npz`, `<scene_root>/frames_subsampled/`, `<scene_root>/preprocess_frames.json`.

**Dependencies:** `torch`, `depth_anything_3` (external), `ffmpeg`.

---

### `frame_to_model_icp.py`

**Purpose:** Stage 1 — Iterative non-rigid frame-to-model ICP alignment. Incrementally registers each frame's point cloud into a shared canonical coordinate system.

**Key class:** `ICPLossTracker` — Accumulates per-frame ICP loss statistics for adaptive outlier point filtering. Computes robust thresholds via median absolute deviation (MAD).

**Key function:**
- `main(config: FrameToModelICPConfig)` — Loads data, converts to camera space, initializes the canonical model from Frame 0, then iterates over frames: rigid ICP init → RoMa matching → non-rigid ICP → loss-based filtering → model merge. Saves aligned point cloud, per-frame deformations, and convention metadata.

**Dependencies:** `algos.icp`, `algos.non_rigid_icp`, `models.roma_matcher`, `data.data_loading`, `utils.*`, `configs.stage1_align`.

---

### `global_optimization.py`

**Purpose:** Stage 2 — Global optimization via leave-one-out consensus. Jointly refines all per-frame deformations to sharpen the canonical point cloud.

**Key function:**
- `main(config: GlobalOptimizationConfig)` — Loads Stage 1 checkpoints, reconstructs filtered per-frame point sets via saved frame segments and pixel indices, runs `global_opt()`, saves refined deformations and convention metadata.

**Dependencies:** `algos.global_optimization`, `data.checkpoint_loading`, `data.data_loading`, `configs.stage2_global_optimization`.

---

### `train_inverse_deformation.py`

**Purpose:** Stage 3.1 — Trains a view-conditioned inverse deformation network mapping canonical points back to per-view camera space.

**Key functions:**
- `main(config: TrainInverseDeformationConfig)` — Loads checkpoints, builds `FullInverseDeformationModel`, generates training data (forward pairs + interpolated samples), trains, validates, and saves.
- `generate_forward_pairs(model, per_view_cam_points, num_samples_per_view)` — Samples camera-space points and warps them to canonical space.
- `generate_interpolated_samples(model, per_view_cam_points, num_samples_per_view, k_neighbors)` — Generates interpolated interior samples via KNN neighborhoods.
- `compute_inverse_warp_loss(...)` — L2 loss between predicted and target camera points.
- `compute_cycle_consistency_loss(...)` — Forward ∘ inverse ≈ identity.
- `compute_twist_magnitude_loss(...)` — Regularization on SE(3) twist magnitude.
- `compute_spatial_smoothness_loss(...)` — Nearby points should have similar twists.
- `train_inverse_deformation(...)` — Training loop with periodic data regeneration.
- `validate_roundtrip_per_view(...)` — Per-view round-trip RMSE validation with PLY output.

**Dependencies:** `models.deformation`, `data.checkpoint_loading`, `data.data_loading`, `configs.stage3_inverse_deformation`.

---

### `train_gs.py`

**Purpose:** Stage 3.2 — Trains a 2DGS or 3DGS model on the aligned canonical point cloud.

**Key functions:**
- `main(config: GSConfig)` — Loads aligned point cloud + deformations + inverse deformation network, builds `CanonicalGSModel`, runs training loop with L1+LPIPS losses and regularization, saves checkpoints, optionally auto-exports PLY and runs evaluation.
- `_evaluate(model, gt_images, intrinsics_list, num_frames, config, global_step, writer, num_frames_to_save)` — Renders all frames, computes PSNR, saves comparison images.

**Dependencies:** `models.canonical_gs_model`, `data.checkpoint_loading`, `data.data_loading`, `losses.*`, `configs.stage3_gs`, `gsplat`.

---

### `eval_gs.py`

**Purpose:** Evaluates a trained GS checkpoint by rendering novel views from various camera trajectories.

**Key functions:**
- `main(config: EvalGSConfig)` — Loads checkpoint, builds model, renders from gs_video transforms, input DA3 poses, and optimized poses; saves images and MP4 videos.
- `_find_checkpoint_path(checkpoint_dir)` — Locates the final or latest checkpoint file.
- `_build_model(config, device, height, width)` — Reconstructs `CanonicalGSModel` from training config.
- `_w2c_from_c2w(c2w)` — Closed-form inverse of a rigid c2w transform.
- `_load_training_config(checkpoint_dir)` — Reads `config.json` from the checkpoint directory.

**Dependencies:** `models.canonical_gs_model`, `data.checkpoint_loading`, `data.data_loading`, `configs.eval_gs`.
