"""Paint2World: 3D-consistent video and scene generation from human oil paintings.

Paint2World extends the video_to_world reconstruction pipeline with a painting
front-end and three painting-anchored 3D-stability mechanisms:

1. Painting-conditioned video generation with depth-warp + stroke-consistency
   controls (Stage 0).
2. Painting-anchored non-rigid alignment: the canonical frame is initialised
   from the painting's estimated geometry and a per-point consistency score
   downweights generated points that drift too far from the painting-implied
   structure (Stage 1).
3. Stroke-coherent 3D Gaussian Splatting: a stroke-segmentation-aware loss
   encourages 3D brush-stroke arrangement to remain coherent across views
   (Stage 3.2).

The heavy CUDA-only dependencies of the underlying ``video_to_world`` pipeline
(tinycudann, gsplat, DepthAnything-3, RoMa) are imported lazily so that the
Paint2World-specific modules (painting analysis, video-generation controls,
stroke-coherence losses, demo UI) remain importable and testable in a plain
CPU / PyTorch environment. The full reconstruction stages delegate to the
existing project entry points when their backends are available.
"""

from __future__ import annotations

__version__ = "1.0.0"

# Public sub-modules. Imported lazily by callers; do not import heavy deps here.
__all__ = [
    "configs",
    "painting_analysis",
    "video_generation",
    "painting_anchored_alignment",
    "stroke_coherent_gs",
    "losses",
    "models",
    "pipeline",
    "demo",
    "data_utils",
]
