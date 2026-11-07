# Changelog

All notable changes to the **Video-to-World** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Changes that have been merged but not yet released. (Move entries to a version section when releasing.)

### Added
- **Paint2World v2.0 — object-level scene construction (Stage -0.5)**:
  - `paint2world/object_segmentation.py`: depth-aware object instance
    segmentation (LAB + depth clustering, depth-gradient barriers,
    objectness ranking; SAM2 / Mask2Former hooks).
  - `paint2world/triposplat.py`: TripoSplat-style per-object 3D generation —
    triplane encode (colour/density/depth planes), Gaussian decode, pseudo
    multi-view synthesis + ICP cross-view alignment, painting-view crop
    refinement.
  - `paint2world/scene_canvas.py`: 3D spatial-dependency graph
    (rests-on/supports, in-front-of/behind, adjacent, rests-on-ground),
    ground-plane fit, physical placement of floating objects, canvas
    Gaussians + hole-free splat z-buffer depth, and canvas-anchored
    (object-rigid + relation-preservation) GS training.
  - Pipeline integration: `--no_stage_object`, `--no_canvas_depth`,
    `--no_object_rigid` CLI toggles; Stage 0 consumes the canvas-composited
    depth control; Stage 3.2 inits from the canvas.
  - Evaluation: new metrics (`num_objects`, `num_relations`,
    `placement_adjusted`, `canvas_painting_l1`, `object_depth_consistency`,
    `relation_gap_drift`) and three new ablation variants
    (`no_object_scene`, `no_canvas_depth`, `no_object_rigid`).
  - Tests: `tests/test_object_branch.py` (16 tests) and
    `tests/test_pipeline_integration.py` (4 tests) — full suite 20 passed.
  - Docs: `docs/modules/paint2world.md`; README "Paint2World" section.
  - Paper: Stage -0.5 method section, canvas figure, object-branch ablation
    table, updated abstract/intro/related work/conclusion.

### Changed
- `paint2world` version bumped to 2.0.0 (object-level scene branch).

### Deprecated
- _(none yet)_

### Removed
- _(none yet)_

### Fixed
- _(none yet)_

### Security
- _(none yet)_

---

## [1.0.0] — 2026-10-13

Initial documented release. This version establishes the project's documentation system, versioning policy, and contribution workflow alongside the existing codebase.

### Added
- **Documentation system** (`docs/` directory):
  - `INDEX.md` — Documentation hub with navigation and project structure overview.
  - `architecture.md` — System architecture, pipeline stages, data flow, module dependency graph, and key algorithmic concepts.
  - `modules/` — Per-module reference documentation covering every source file:
    - `README.md` — Top-level pipeline scripts.
    - `algos.md` — Core algorithms (ICP, non-rigid ICP, global optimization).
    - `configs.md` — Configuration dataclasses for all stages.
    - `data.md` — Data loading and checkpoint I/O.
    - `losses.md` — Loss functions (rendering, TV, correspondence, Gaussian).
    - `models.md` — Neural models (CanonicalGSModel, deformation, RoMa matcher).
    - `utils.md` — Utilities (geometry, KNN, normals, pointcloud, image, logging, etc.).
    - `patches.md` — Third-party patches and maintenance notes.
  - `CHANGELOG.md` — This file.
  - `VERSIONING.md` — Semantic versioning policy and maintenance workflow.
  - `CONTRIBUTING.md` — Contribution guide and update workflow.
- **Version tracking**:
  - `VERSION` file at project root containing the current version string (`1.0.0`).

### Documentation Coverage (v1.0.0 baseline)

The following modules and files are documented:

| Module | Files | Documented |
|--------|-------|------------|
| Top-level scripts | `run_reconstruction.py`, `preprocess_video.py`, `frame_to_model_icp.py`, `global_optimization.py`, `train_inverse_deformation.py`, `train_gs.py`, `eval_gs.py` | ✅ |
| `algos/` | `__init__.py`, `icp.py`, `non_rigid_icp.py`, `global_optimization.py` | ✅ |
| `configs/` | `__init__.py`, `common.py`, `utils.py`, `roma.py`, `stage1_align.py`, `stage2_global_optimization.py`, `stage3_inverse_deformation.py`, `stage3_gs.py`, `eval_gs.py` | ✅ |
| `data/` | `__init__.py`, `data_loading.py`, `checkpoint_loading.py` | ✅ |
| `losses/` | `__init__.py`, `rendering.py`, `tv.py`, `correspondence.py`, `gaussian.py` | ✅ |
| `models/` | `__init__.py`, `canonical_gs_model.py`, `deformation.py`, `roma_matcher.py` | ✅ |
| `utils/` | `__init__.py`, `geometry.py`, `knn.py`, `normals.py`, `pointcloud.py`, `image.py`, `downsample.py`, `logging.py`, `export_checkpoint_to_ply.py`, `view_checkpoint.py` | ✅ |
| `patches/` | `da3-export-trajectory.patch`, `romav2-dataclasses.patch` | ✅ |

---

## Version History Summary

| Version | Date | Summary |
|---------|------|---------|
| 1.0.0 | 2026-10-13 | Initial documented release with full documentation system and versioning |

---

## Changelog Entry Types

Per [Keep a Changelog](https://keepachangelog.com/), entries are categorized as:

- **Added** — New features, modules, or capabilities.
- **Changed** — Changes to existing functionality.
- **Deprecated** — Features that will be removed in a future version.
- **Removed** — Features removed in this version.
- **Fixed** — Bug fixes.
- **Security** — Security-related changes.

See [VERSIONING.md](VERSIONING.md) for how version numbers are bumped based on these categories.

---

## How to Add a Changelog Entry

1. Add your entry under the `[Unreleased]` section.
2. Use the appropriate category (`Added`, `Changed`, `Fixed`, etc.).
3. Write a concise but descriptive entry. Reference issue/PR numbers if applicable.
4. When releasing, move entries from `[Unreleased]` to a new version section with a date.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full release workflow.

---

## Links

[Unreleased]: https://github.com/lukasHoel/video_to_world/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lukasHoel/video_to_world/releases/tag/v1.0.0
