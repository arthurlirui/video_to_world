"""Paint2World: 3D-consistent video and scene generation from human oil paintings.

Paint2World extends the video_to_world reconstruction pipeline with a painting
front-end and four painting-anchored mechanisms:

1. Object-level scene construction (Stage -0.5): the painting is decomposed
   into objects by depth-aware image segmentation, each object is generated
   in 3D with a TripoSplat-style triplane->Gaussian decoder, the objects are
   wired into a 3D spatial-dependency graph (supports / occludes / adjacent)
   and composed onto a 3D scene canvas with a fitted ground plane.
2. Painting-conditioned video generation with depth-warp + stroke-consistency
   controls (Stage 0), conditioned on the scene canvas's composited depth.
3. Painting-anchored non-rigid alignment: the canonical frame is initialised
   from the painting's estimated geometry and a per-point consistency score
   downweights generated points that drift too far from the painting-implied
   structure (Stage 1).
4. Stroke-coherent, object-rigid 3D Gaussian Splatting: a stroke-segmentation-
   aware loss keeps brush strokes coherent across views, while per-object
   SE(3) anchors and a relation-preservation loss keep the canvas's spatial
   dependencies intact during optimisation (Stage 3.2).

The heavy CUDA-only dependencies of the underlying ``video_to_world`` pipeline
(tinycudann, gsplat, DepthAnything-3, RoMa) are imported lazily so that the
Paint2World-specific modules (painting analysis, object segmentation,
triplane object generation, scene canvas, video-generation controls,
stroke-coherence losses, demo UI) remain importable and testable in a plain
CPU / PyTorch environment. The full reconstruction stages delegate to the
existing project entry points when their backends are available.
"""

from __future__ import annotations

__version__ = "2.0.0"

# Public sub-modules. Imported lazily by callers; do not import heavy deps here.
__all__ = [
    "configs",
    "painting_analysis",
    "object_segmentation",
    "triposplat",
    "scene_canvas",
    "video_generation",
    "painting_anchored_alignment",
    "stroke_coherent_gs",
    "losses",
    "models",
    "pipeline",
    "demo",
    "data_utils",
]
