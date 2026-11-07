# paint2world — Oil-Painting-Driven Video & 3D Scene Generation

`paint2world/` is a self-contained extension of the video_to_world pipeline.
It converts a single human oil painting into (i) a 3D-consistent flythrough
video and (ii) a 3D Gaussian Splatting scene, anchored by an explicit
object-level 3D scene scaffold (the *scene canvas*).

Everything runs on CPU with synthetic backends; real backends (DepthAnything-3,
SAM2, pretrained TripoSplat, Wan-2.2 I2V, gsplat) plug in behind identical
interfaces.

## Module map

| File | Stage | Role |
|------|-------|------|
| `configs.py` | all | Dataclass configuration (tyro-friendly), one dataclass per stage. |
| `painting_analysis.py` | -1 | Perceptual depth, normals, brush-stroke segmentation, confidence. |
| `object_segmentation.py` | -0.5a | Depth-aware object instance segmentation + objectness ranking. |
| `triposplat.py` | -0.5b | TripoSplat-style triplane → Gaussian 3D object generation, with pseudo multi-view + ICP cross-view alignment + painting-view crop refinement. |
| `scene_canvas.py` | -0.5c | Spatial-dependency graph, ground-plane fit, physical placement, canvas Gaussians/depth; canvas-anchored (object-rigid) GS training. |
| `video_generation.py` | 0 | Camera trajectories, depth-warp control (canvas-composited), stroke-consistency attention, mock video backend. |
| `painting_anchored_alignment.py` | 1 | Painting-initialised canonical frame + depth-warp consistency weights; non-rigid ICP fallback. |
| `stroke_coherent_gs.py` | 3.2 | Stroke-coherence loss, painting-view loss, differentiable rasteriser fallback, GS training driver. |
| `losses.py` | 1/3.2 | Shared loss implementations (anchor, consistency, stroke coherence, painting view). |
| `pipeline.py` | all | End-to-end orchestrator + CLI. |
| `evaluation.py` | — | Metric harness (align loss, painting L1/LPIPS proxy, stroke coherence, canvas/object/relation metrics). |
| `run_experiments.py` | — | Batch runner producing `results*/{main,ablation,trajectory,scalability}.json`. |
| `make_tables.py` | — | JSON → LaTeX table rows for the paper. |
| `demo.py` | — | Gradio demo UI. |
| `data_utils.py` | — | Synthetic oil-painting generator + sample data. |

## Stage -0.5 in detail

1. **Object segmentation** — pixels are clustered jointly in LAB colour +
   estimated-depth space with spatial connectivity; merging is blocked across
   strong depth gradients; regions are scored by *objectness*
   (area × depth coherence × colour distinctness) and the top-K become
   objects. Interface: an instance-label map + per-object metadata, so SAM2 /
   Mask2Former drop in directly.
2. **TripoSplat-style generation** — each object's masked pixels are
   unprojected to a local point cloud; side views are synthesised by rotating
   the cloud about its vertical axis and ICP-aligned back onto the painting
   view (cross-view alignment); fused points are splatted onto three
   orthogonal feature planes (colour / density / depth per cell); a query
   grid decodes occupied cells into 3D Gaussians; scales/opacities/colours
   are refined against the object's painting-view crop.
3. **Scene canvas** — objects get centroids + bounding boxes; a ground plane
   is fitted to the lowest scene points; a directed relation graph is built
   (`rests_on` / `supports`, `in_front_of` / `behind`, `adjacent_to`,
   `rests_on_ground`); floating grounded objects are snapped onto the ground;
   the canvas exposes composed Gaussians (objects + background), a
   hole-free splat z-buffer depth map, and per-object relation gaps.

## Consumables downstream

- **Stage 0** uses the canvas depth in the depth-warp control
  (`enable_canvas_depth_control`), so the geometry control respects object
  occlusion silhouettes.
- **Stage 3.2** initialises from the canvas and trains with per-object SE(3)
  rigid anchors + a relation-preservation loss (`enable_object_rigid_gs`,
  `relation_preservation_weight`).

## Entry points

```bash
python -m paint2world.pipeline --painting <img> --out_root <dir>   # full pipeline
python -m paint2world.demo                                         # Gradio UI
python -m paint2world.run_experiments --out results_v2             # paper numbers
python -m paint2world.make_tables results_v2                       # LaTeX rows
python -m pytest tests/ -q                                         # 20 tests
```

## Tests

- `tests/test_object_branch.py` — segmentation, triplane encode/decode, ICP,
  refinement, relations, canvas depth, world transform, rigid training,
  persistence (16 tests).
- `tests/test_pipeline_integration.py` — end-to-end orchestrator runs and
  component-toggle routing (4 tests).
