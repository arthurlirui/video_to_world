"""Paint2World configuration dataclasses.

Mirrors the dataclass-driven, tyro-friendly style of ``configs/`` in the
underlying video_to_world pipeline. All hyperparameters for the painting
front-end and the painting-anchored reconstruction extensions live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PaintingAnalysisConfig:
    """Stage -1: oil-painting analysis (depth / normals / strokes)."""

    # ---- Input ----
    painting_path: str = ""
    """Path to the input oil-painting image (PNG/JPG)."""

    max_side: int = 768
    """Longest side after resizing; preserves aspect ratio."""

    # ---- Monocular geometry estimation ----
    depth_backend: Literal["depth_anything_v3", "marigold", "zoedepth", "synthetic"] = "synthetic"
    """Depth backend. ``synthetic`` produces a perceptually plausible depth map
    from colour/gradient/saturation cues and requires no network weights, so the
    demo runs out of the box. Swap to ``depth_anything_v3`` etc. when the
    corresponding checkpoints are available."""

    depth_model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"
    """HuggingFace model id for the real depth backend."""

    # ---- Stroke segmentation ----
    stroke_backend: Literal["sam2", "fastscnn", "graphcut", "synthetic"] = "synthetic"
    """Brush-stroke segmentation backend. ``synthetic`` uses an oriented
    gradient / superpixel grouping that requires no network weights."""

    stroke_min_area: int = 32
    """Minimum stroke segment area in pixels."""

    stroke_merge_threshold: float = 0.10
    """Colour-distance threshold below which adjacent segments are merged."""

    # ---- Normal estimation ----
    normal_method: Literal["depth_gradient", "photometric"] = "depth_gradient"
    """How surface normals are derived. ``depth_gradient`` differentiates the
    (smoothed) depth map; ``photometric`` adds a Lambertian shading prior."""

    # ---- Output ----
    out_dir: str = "data/sample_runs/painting_analysis"
    """Where to write depth.png / normals.png / strokes.png / painting.npz."""


@dataclass
class VideoGenerationConfig:
    """Stage 0: painting-conditioned video generation with 3D-stability controls."""

    # ---- Trajectory ----
    num_frames: int = 50
    """Number of frames to generate along the camera trajectory."""

    trajectory_type: Literal["orbit", "dolly", "pan", "spiral"] = "orbit"
    """Camera trajectory family. ``orbit`` rotates around the painting's
    principal depth; ``dolly`` moves forward; ``pan`` translates laterally;
    ``spiral`` combines orbit + dolly."""

    trajectory_angle_deg: float = 25.0
    """For orbit/spiral: total azimuth sweep in degrees."""

    trajectory_baseline: float = 0.18
    """Stereo baseline fraction (relative to scene scale) for pan/dolly."""

    # ---- Conditioning ----
    video_backend: Literal[
        "wan_i2v", "viewcrafter", "csvd", "stable_video_diffusion", "mock"
    ] = "mock"
    """Video diffusion backend. ``mock`` synthesises a 3D-stable flythrough
    from the painting + depth (no network weights needed) so the demo runs
    end-to-end. The interface is identical to a real backend, so swapping in
    Wan-I2V / ViewCrafter / SVD / CSVD is a one-line change."""

    video_model_name: str = "Wan-AI/Wan2.2-I2V-14B-480P"
    """HuggingFace model id for the real video backend."""

    seed: int = 0
    """Random seed for the video backend."""

    # ---- 3D-stability controls (the novel contribution of Stage 0) ----
    depth_warp_control_strength: float = 0.85
    """Strength of the depth-warp ControlNet signal: at each generated frame
    we warp the painting's depth along the camera trajectory and use it as a
    geometry control. 0 disables, 1 fully constrains geometry to the painting."""

    stroke_consistency_attention: bool = True
    """If True, the painting is locked as a reference frame in the diffusion
    temporal attention and a stroke-mask attention bias prevents brush strokes
    from drifting across generated frames."""

    reference_frame_lock: bool = True
    """Lock the painting as the temporal-attention reference frame."""

    temporal_consistency_loss_weight: float = 0.5
    """Weight of the implicit temporal-consistency guidance during sampling."""

    # ---- Output ----
    out_dir: str = "data/sample_runs/video"
    """Where to write the generated video.mp4 and frames/."""

    fps: int = 8
    """Output video fps."""


@dataclass
class ObjectSegmentationConfig:
    """Stage -0.5a: object-level segmentation of the painting.

    Complements the brush-stroke segmentation (which models *marks*) with an
    object-level decomposition (which models *things*: trees, vessels, faces,
    fruit, buildings). Object masks are the unit of TripoSplat-based 3D object
    generation and of the spatial-dependency graph.
    """

    backend: Literal["sam2", "mask2former", "synthetic"] = "synthetic"
    """Object segmentation backend. ``synthetic`` clusters jointly in LAB colour
    + estimated-depth space with spatial connectivity and needs no network
    weights. Swap to SAM2 / Mask2Former when checkpoints are available; the
    interface (an instance-label map) is identical."""

    min_object_area: int = 400
    """Minimum object mask area in pixels (rejects spurious fragments)."""

    max_objects: int = 12
    """Keep at most this many highest-objectness regions as objects."""

    depth_coherence_weight: float = 0.5
    """Weight of depth coherence in the objectness score. A region whose depth
    varies little is more likely a compact object than a ground/sky sweep."""

    colour_distinctness_weight: float = 0.5
    """Weight of colour distinctness (LAB distance to the scene mean) in the
        objectness score."""

    out_dir: str = "data/sample_runs/objects"
    """Where to write objects.png / objects.npz."""


@dataclass
class TripoSplatConfig:
    """Stage -0.5b: TripoSplat-style per-object 3D generation.

    Follows the TripoSplat architecture at demo scale: each object crop is
    lifted to a *triplane* feature representation (three orthogonal planes,
    ``XY`` / ``YZ`` / ``ZX``) from a painting view plus a depth-warped side
    view, cross-view aligned; 3D Gaussians are then decoded from triplane
    samples and refined against the object's painting-view crop. The real
    TripoSplat checkpoints can replace the synthetic encoder/decoder behind
    the same interface.
    """

    triplane_resolution: int = 32
    """Spatial resolution of each triplane plane (R x R)."""

    triplane_channels: int = 8
    """Feature channels per triplane plane (density + RGB + density variance)."""

    query_resolution: int = 16
    """Query-grid resolution per axis inside the object's unit cube; each
    occupied query cell decodes into one 3D Gaussian."""

    num_side_views: int = 1
    """Number of depth-warped side views fused into the triplane in addition to
    the painting view (the TripoSplat cross-view cue)."""

    side_view_angle_deg: float = 20.0
    """Orbit angle of the first side view around the object's vertical axis."""

    refinement_iters: int = 40
    """Gradient steps refining the decoded Gaussians' scales/opacities/colours
    against the object's painting-view crop (per object)."""

    min_gaussians_per_object: int = 32
    """Objects decoding to fewer Gaussians are padded with jittered samples so
    every object remains renderable."""


@dataclass
class SceneCanvasConfig:
    """Stage -0.5c: spatial-dependency graph + 3D scene canvas.

    The canvas is the explicit 3D scaffold of the scene: a ground plane, the
    placed object set (each with a rigid transform + oriented bounding box),
    the pairwise spatial relations between objects, and the background strokes
    that are not part of any object. The canvas conditions Stage 0 (composited
    depth control) and Stage 3.2 (object-rigid anchors + relation
    preservation).
    """

    ground_plane_weight: float = 0.6
    """Fraction of the scene's lowest points used to fit the ground plane."""

    support_contact_tol: float = 0.12
    """Vertical gap (fraction of scene height) below which 'rests on / supports'
    edges are declared between two objects or an object and the ground."""

    adjacency_tol: float = 0.25
    """Horizontal gap (fraction of scene width) below which two objects are
    declared adjacent."""

    occlusion_min_overlap: float = 0.05
    """Minimum 2D mask overlap (fraction of the smaller mask) plus depth
    ordering needed to declare an in-front-of / behind edge."""

    enable_canvas_depth_control: bool = True
    """If True, Stage 0's per-frame depth-warp control is composited from the
    canvas (objects occlude correctly) instead of warping the raw painting
    depth."""

    enable_object_rigid_gs: bool = True
    """If True, Stage 3.2 keeps each object's Gaussians rigidly coupled by a
    learnable per-object SE(3) delta, so optimisation moves objects as wholes
    and preserves the dependency graph."""

    relation_preservation_weight: float = 0.10
    """Weight of the relation-preservation loss (supports / rests-on contacts
    must stay in contact) during canvas-anchored GS training."""

    out_dir: str = "data/sample_runs/canvas"
    """Where to write canvas.npz / layout.png / relations.json."""


@dataclass
class PaintingAnchoredAlignmentConfig:
    """Stage 1 extension: painting-anchored non-rigid ICP."""

    enable_painting_anchor: bool = True
    """Initialise the canonical frame from the painting's estimated geometry
    instead of treating video frame 0 as the gauge fix."""

    painting_anchor_weight: float = 0.25
    """Weight of the painting-anchor canonical loss in the ICP energy."""

    depth_warp_consistency_weight: float = 0.40
    """Weight of the depth-warp per-point consistency score. Points whose
    depth deviates more than ``depth_warp_tolerance`` from the painting-implied
    depth are downweighted in the data term."""

    depth_warp_tolerance: float = 0.08
    """Relative depth-deviation tolerance (fraction of scene scale) beyond
    which a generated point is considered to drift from the painting prior."""

    stroke_coherence_init: bool = True
    """If True, points belonging to the same painting stroke segment are
    initialised with the same local-deformation seed, encouraging strokes to
    deform as coherent 3D patches."""

    # ---- Delegate to existing pipeline ----
    base_num_frames: int = 50
    base_stride: int = 2
    base_offset: int = 0


@dataclass
class StrokeCoherentGSConfig:
    """Stage 3.2 extension: stroke-coherent 3D Gaussian Splatting."""

    enable_stroke_coherence: bool = True
    """Add the stroke-coherence loss during GS training."""

    stroke_coherence_weight: float = 0.10
    """Weight of the stroke-coherence loss relative to the rendering loss."""

    stroke_intra_smoothness: float = 0.05
    """SH-colour smoothness weight *within* a stroke segment."""

    stroke_inter_sharpness: float = 0.02
    """SH-colour sharpness bonus *across* stroke boundaries (encourages
    strokes to remain visually distinct in 3D)."""

    painting_view_loss_weight: float = 0.30
    """Weight of the painting-view reconstruction loss: the trained scene,
    rendered from the painting's camera, should reproduce the input painting."""

    # ---- Delegate to existing pipeline ----
    base_renderer: Literal["2dgs", "3dgs"] = "3dgs"
    base_iterations: int = 10_000


@dataclass
class Paint2WorldConfig:
    """End-to-end Paint2World pipeline configuration."""

    name: str = "paint2world"
    out_root: str = "data/sample_runs/paint2world"

    painting: PaintingAnalysisConfig = field(default_factory=PaintingAnalysisConfig)
    objects: ObjectSegmentationConfig = field(default_factory=ObjectSegmentationConfig)
    triposplat: TripoSplatConfig = field(default_factory=TripoSplatConfig)
    canvas: SceneCanvasConfig = field(default_factory=SceneCanvasConfig)
    video: VideoGenerationConfig = field(default_factory=VideoGenerationConfig)
    alignment: PaintingAnchoredAlignmentConfig = field(
        default_factory=PaintingAnchoredAlignmentConfig
    )
    gs: StrokeCoherentGSConfig = field(default_factory=StrokeCoherentGSConfig)

    # ---- Pipeline control ----
    run_stage_minus_1: bool = True
    run_stage_object: bool = True
    run_stage_0: bool = True
    run_stage_1: bool = True
    run_stage_2: bool = False
    run_stage_3: bool = True
    """Whether to run each stage. Stage 2 (global optimisation) is skipped by
    default, mirroring the ``fast`` mode of the underlying pipeline."""

    open_demo_ui: bool = False
    """If True, launch a Gradio demo UI after the pipeline finishes."""
