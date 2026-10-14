# Video-to-World — Documentation

**Current version:** [1.0.0](./VERSION) · **Status:** Initial release · **Last updated:** 2026-10-13

---

## About This Project

**World Reconstruction From Inconsistent Views** reconstructs 3D worlds from video diffusion model outputs. It uses non-rigid alignment to resolve inherent 3D inconsistencies in generated video sequences, producing a canonical 3D Gaussian Splatting representation.

- [arXiv Paper](https://arxiv.org/abs/2603.16736) · [Project Page](https://lukashoel.github.io/video_to_world/) · [Video](https://www.youtube.com/watch?v=qXnUwhVmBzA)

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture.md) | System design, pipeline stages, data flow, and module relationships |
| [Module Reference](modules/README.md) | Per-module and per-file documentation with classes, functions, and dependencies |
| [Changelog](CHANGELOG.md) | Versioned history of changes (follows [Keep a Changelog](https://keepachangelog.com/) format) |
| [Versioning Policy](VERSIONING.md) | Semantic versioning rules and maintenance workflow |
| [Contributing Guide](CONTRIBUTING.md) | How to contribute, update docs, and release new versions |

---

## Quick Links

- **New here?** Start with the [Architecture Overview](architecture.md) for the big picture.
- **Looking for a specific file?** Browse the [Module Reference](modules/README.md).
- **What changed?** Read the [Changelog](CHANGELOG.md).
- **Making changes?** Follow the [Contributing Guide](CONTRIBUTING.md).

---

## Project Structure at a Glance

```
video_to_world/
├── run_reconstruction.py        # End-to-end pipeline orchestrator
├── preprocess_video.py          # Stage 0: DA3 preprocessing
├── frame_to_model_icp.py        # Stage 1: Non-rigid ICP alignment
├── global_optimization.py       # Stage 2: Global optimization
├── train_inverse_deformation.py # Stage 3.1: Inverse deformation training
├── train_gs.py                  # Stage 3.2: Gaussian Splatting training
├── eval_gs.py                   # Evaluation / novel-view rendering
├── algos/                       # Core algorithms (ICP, non-rigid ICP, global opt)
├── configs/                     # Tyro dataclass configurations for all stages
├── data/                        # Data loading and checkpoint serialization
├── losses/                      # Loss functions (rendering, TV, correspondence, Gaussian)
├── models/                      # Neural models (GS, deformation, RoMa matcher)
├── utils/                       # Utilities (geometry, KNN, normals, logging, etc.)
├── patches/                     # Patches for third-party dependencies
├── docs/                        # This documentation
├── VERSION                      # Current version string
├── pyproject.toml               # Project metadata (Ruff config)
└── README.md                    # Quickstart and installation guide
```

---

## Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** — Incompatible API or pipeline changes
- **MINOR** — New features, backward-compatible
- **PATCH** — Bug fixes, backward-compatible

See [VERSIONING.md](VERSIONING.md) for the full policy.
