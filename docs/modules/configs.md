# configs/ — Configuration Dataclasses

All hyperparameters are defined as frozen dataclasses, parsed by [tyro](https://github.com/brentyi/tyro) into CLI arguments. This provides type safety, auto-generated help, and nested configuration without YAML files.

---

## `__init__.py`

Docstring-only init describing the package as the centralized location for tyro dataclass configs.

---

## `common.py`

Shared config mixins used across multiple stages.

### `TensorboardConfig`
- `tensorboard: bool = True` — Enable TensorBoard logging.
- `tensorboard_log_dir: Optional[str] = None` — Override log directory.

### `KnnBackendConfig`
- `knn_backend: str = "cpu_kdtree"` — KNN backend: `"cpu_kdtree"` (scipy) or `"gpu_kdtree"` (torch_kdtree).

### `AlignmentDataConfig` (frozen)
Stage-1 data-loading and confidence-filtering parameters, reused by all downstream stages to ensure consistency:
- `num_frames: int = 50` — Number of frames to align.
- `stride: int = 2` — Take every `stride`-th frame.
- `offset: int = 0` — Starting frame index.
- `conf_thresh_percentile`, `conf_mode`, `conf_local_percentile`, `conf_global_percentile`, `conf_voxel_size`, `conf_voxel_min_count_percentile` — Confidence filtering parameters.

---

## `roma.py`

### `RomaConfig` (frozen)
RoMa dense feature matcher settings:
- `use_roma_matching: bool = True` — Enable RoMa correspondence matching.
- `roma_version: str = "v2"` — RoMa version (`"v1"` or `"v2"`).
- `roma_model: str = "indoor"` — Model type.
- `roma_num_samples`, `roma_certainty_threshold`, `roma_max_references` — Matching parameters.
- `roma_reference_sampling: str = "recent_and_strided"` — Reference frame selection strategy.
- `roma_loss_weight: float` — Weight in the ICP loss.
- `roma_max_corr_dist: float` — Maximum correspondence distance.

---

## `stage1_align.py`

### `FrameToModelICPConfig`
Stage 1 configuration. Mixes in `TensorboardConfig`, `KnnBackendConfig`, and embeds `AlignmentDataConfig` and `RomaConfig`.

Key parameters:
- ICP: `icp_n_iter`, `icp_lr`, `icp_method`, `icp_early_stopping_*`, `icp_color_icp_*`
- Deformation grid: `deform_log2_hashmap_size`, `deform_num_levels`, `deform_n_neurons`, `deform_n_hidden_layers`, `deform_min_res`, `deform_max_res`
- Regularization: `icp_local_twist_reg`, `icp_tv_reg`, `icp_tv_voxel_size`, `icp_tv_every_k`, `icp_tv_sample_ratio`
- Point filtering: `filter_points`, `filter_geom_sigma`, `filter_color_sigma`, `filter_worst_pct`, `filter_min_frames`, `filter_base_percentile`
- Output: `out_path`, `out_suffix`, `save_intermediate_every`

---

## `stage2_global_optimization.py`

### `GlobalOptimizationConfig`
Stage 2 configuration. Mixes in `TensorboardConfig`, `KnnBackendConfig` (default `gpu_kdtree`).

Key parameters:
- Paths: `root_path`, `run`, `checkpoint_subdir`, `out_subdir`
- LOO consensus: `loo_loss_weight`, `loo_k_neighbors`, `loo_max_corr_dist`, `loo_normal_k`, `loo_kdtree_rebuild_every`
- LOO pair subsampling: `loo_max_pairs_per_iter`, `loo_pairs_per_src`
- Anchoring: `anchor_loss_weight`, `anchor_n_samples`
- Regularization: `tv_reg`, `tv_voxel_size`, `tv_every_k`, `tv_sample_ratio`
- Color ICP: `loo_color_icp_weight`, `loo_color_icp_k`, `loo_color_icp_max_color_dist`
- Thin-shell: `thin_shell_weight`
- Optimization: `lr`, `n_iters`, `deform_chunk_size`
- Saving: `save_intermediate_every_n`

---

## `stage3_inverse_deformation.py`

### `TrainInverseDeformationConfig`
Stage 3.1 configuration. Mixes in `TensorboardConfig`, `KnnBackendConfig`.

Key parameters:
- Paths: `root_path`, `run`, `checkpoint_subdir`, `out_path`
- Training: `n_epochs` (sentinel `None` → pipeline decides), `batch_size`, `lr`
- Loss weights: `cycle_weight`, `magnitude_weight`, `smoothness_weight`
- Data generation: `num_forward_samples`, `num_interp_samples`, `regenerate_every`
- Network architecture: `view_embed_dim`, `min_res`, `max_res`, `num_levels`, `log2_hashmap_size`, `n_neurons`, `n_hidden_layers`
- Output: `save_validation_plys`

---

## `stage3_gs.py`

### `GSConfig`
Stage 3.2 Gaussian Splatting training configuration. Mixes in `TensorboardConfig`.

Key parameters:
- Paths: `root_path`, `run`, `global_opt_subdir`, `inverse_deform_dir`, `original_images_dir`, `out_dir`
- Renderer: `renderer: Literal["2dgs", "3dgs"]`
- SH: `sh_degree`, `sh_full_from_iter`, `sh_increase_every`, `sh_freeze_means_when_full_sh`
- Downsampling: `target_num_points`
- Optimization toggles: `optimize_cams`, `optimize_positions`, `deform_inverse_rotations`
- Learning rates: `lr_sh0`, `lr_shN`, `lr_opacities`, `lr_scales`, `lr_quats`, `lr_positions`, `lr_cams`, `lr_colors`, `lr_decay`
- Gaussian init: `initial_opacity`, `initial_scale`, `initial_flat_ratio`, `scale_init`, `knn_neighbors`, `normal_k`
- Loss weights: `l1_weight`, `lpips_weight`
- Regularization: `scale_reg_weight`, `opacity_reg_weight`, `normal_consistency_weight`, `distortion_weight`, `alpha_reg_weight`, `sh_reg_weight`
- Training schedule: `num_iters`, `frames_per_iter`, `log_every`, `save_every`, `eval_every`
- Auto-eval: `auto_eval`

---

## `eval_gs.py`

### `EvalGSConfig`
Evaluation / novel-view rendering configuration.

Key parameters:
- Paths: `root_path`, `run`, `checkpoint_dir`, `global_opt_subdir`, `transforms_path`, `original_images_dir`, `out_dir`
- Rendering options: `render_gs_video_path`, `render_input_poses`, `render_optimised_poses`
- Output: `save_images`, `save_video`, `fps`, `max_frames`, `white_background`
- Resolution: `override_width`, `override_height`
- Point cloud: `target_num_points`

---

## `utils.py`

### `ExportGSCheckpointToPlyConfig`
- Paths: `root_path`, `run`, `checkpoint_dir`, `global_opt_subdir`
- Output: `out_ply`, `max_points` (optional subsampling limit)

### `ViewGSCheckpointConfig`
- Paths: `root_path`, `run`, `checkpoint_dir`
- Viewer: `port`, `show_cameras`, `near_plane`, `far_plane`
