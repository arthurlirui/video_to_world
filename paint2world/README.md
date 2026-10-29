# Paint2World — 3D-Consistent Video and Scene Generation from Human Oil Paintings

Paint2World is a research demo that turns a **human oil painting** into a
**3D-consistent flythrough video** and a **3D Gaussian Splatting scene**. It
extends the *World Reconstruction From Inconsistent Views* pipeline
(`video_to_world`, Höllein & Nießner, 2026) with a painting front-end and
three painting-anchored 3D-stability mechanisms.

```
┌────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐
│  Stage -1  │ → │   Stage 0   │ → │   Stage 1    │ → │      Stage 3.2      │
│  Painting  │   │ Painting-   │   │  Painting-   │   │  Stroke-Coherent    │
│  Analysis  │   │ Conditioned │   │  Anchored    │   │  Gaussian Splatting │
│            │   │ Video Gen.  │   │  Alignment   │   │                     │
│ oil-painting│  │ + 3D-stab.  │   │  (non-rigid  │   │  (painting-view +   │
│ → depth,   │   │   controls  │   │   ICP)       │   │   stroke-coherence) │
│  normals,  │   │             │   │              │   │                     │
│  strokes   │   │ → video.mp4 │   │ → canonical  │   │ → 3DGS scene +      │
│            │   │             │   │   point cloud│   │   novel-view render │
└────────────┘   └─────────────┘   └──────────────┘   └─────────────────────┘
```

## What's novel

1. **Painting-conditioned video generation with 3D-stability controls**
   (`paint2world/video_generation.py`). The painting's depth is warped along a
   camera trajectory and used as a per-frame geometry ControlNet signal
   (`depth_warp_control_strength`), the painting is locked as a reference frame
   in the temporal attention with a stroke-mask attention bias
   (`stroke_consistency_attention`), and a temporal-consistency guidance term
   is applied during sampling. Together these drastically reduce the
   "generative drift" that video diffusion models otherwise produce.

2. **Painting-anchored non-rigid alignment** (`paint2world/painting_anchored_alignment.py`).
   The canonical frame is initialised from the painting's estimated geometry
   (instead of treating video frame 0 as the gauge fix), and each generated
   frame's points are weighted by a **depth-warp consistency score** that
   downweights points whose depth drifts beyond a tolerance from the
   painting-implied structure. This lets the scene plausibly extend beyond the
   painting's frustum while suppressing drift within it.

3. **Stroke-coherent 3D Gaussian Splatting** (`paint2world/stroke_coherent_gs.py`).
   The painting's brush-stroke segmentation is transferred onto the 3D
   Gaussians and a **stroke-coherence loss** encourages each stroke to remain a
   coherent 3D surface patch (intra-stroke SH-colour smoothness) while keeping
   strokes visually distinct across boundaries (inter-stroke sharpness). A
   **painting-view reconstruction loss** ties the trained scene back to the
   input artwork.

## Quickstart (no GPU, no network weights)

The demo ships with **synthetic / mock backends** for the depth estimator, the
video diffusion model, and the Gaussian rasteriser, so it runs end-to-end in a
plain CPU / PyTorch environment.

```bash
# 1. Install the lightweight demo dependencies (PyTorch, NumPy, PIL, Matplotlib).
pip install torch numpy pillow matplotlib imageio

# 2. (Optional) Generate the sample oil paintings.
python -m paint2world.data_utils
#   → data/sample_paintings/sample_00_landscape.png
#   → data/sample_paintings/sample_01_still_life.png
#   → data/sample_paintings/sample_02_portrait.png

# 3. Run the full pipeline on a sample painting (≈40 s on CPU).
python -m paint2world.pipeline \
    --painting data/sample_paintings/sample_00_landscape.png \
    --out_root data/sample_runs/paint2world \
    --num_frames 8 --trajectory orbit --angle_deg 20 --baseline 0.15 \
    --depth_strength 0.85
#   → data/sample_runs/paint2world/stage_minus_1/{depth,normals,strokes,confidence}.png
#   → data/sample_runs/paint2world/stage_0/video.mp4
#   → data/sample_runs/paint2world/stage_1/canonical_preview.png
#   → data/sample_runs/paint2world/stage_3/render_painting_view.png + gaussians.npz

# 4. (Optional) Launch the interactive Gradio demo.
pip install gradio
python -m paint2world.demo
#   → http://localhost:7860
```

## Swapping in real backends

All three painting-anchored mechanisms are backend-agnostic. To use the
production-grade backends of the underlying `video_to_world` pipeline, set the
corresponding configuration fields:

| Stage | Config field | Real backend | Install |
|-------|--------------|--------------|---------|
| -1 | `painting.depth_backend = "depth_anything_v3"` | DepthAnything-3 | see `README.md` (root) |
| -1 | `painting.stroke_backend = "sam2"` | SAM2 | `pip install segment-anything-2` |
| 0 | `video.video_backend = "wan_i2v"` | Wan-2.2 I2V + depth ControlNet | `pip install diffusers` + Wan weights |
| 1 | `alignment.enable_painting_anchor = True` (full backend) | `algos.non_rigid_icp` (tinycudann + RoMa) | see root `README.md` |
| 3.2 | `gs.base_renderer = "3dgs"` (full backend) | `train_gs` (gsplat + tinycudann) | see root `README.md` |

The painting-anchor, depth-warp-consistency, stroke-coherence, and
painting-view losses are injected into the full backends at the same points as
the synthetic / mock fallbacks, so the demo and the production deployment
share identical 3D-stability logic.

## Package layout

```
paint2world/
├── __init__.py
├── configs.py                         # All Paint2World hyperparameters (tyro dataclasses)
├── painting_analysis.py               # Stage -1: depth / normals / strokes / confidence
├── video_generation.py                # Stage 0: painting-conditioned video + 3D-stability controls
├── painting_anchored_alignment.py     # Stage 1: painting-anchored non-rigid ICP
├── stroke_coherent_gs.py              # Stage 3.2: stroke-coherent 3DGS
├── losses.py                          # painting_anchor / depth_warp_consistency / stroke_coherence / painting_view losses
├── pipeline.py                        # End-to-end orchestrator + CLI
├── data_utils.py                      # Synthetic oil-painting generator + sample data
└── demo.py                            # Gradio demo UI
```

## Sample outputs

A reference run on the synthetic landscape painting is checked in at
`data/sample_runs/demo/`:

- `stage_minus_1/` — painting, depth, normals, brush-stroke segmentation, confidence
- `stage_0/video.mp4` — the 3D-stable flythrough generated by Stage 0
- `stage_1/canonical_preview.png` — the painting-anchored canonical point cloud
- `stage_3/render_painting_view.png` — the trained 3DGS scene rendered from the painting's viewpoint

## Citation

If you use Paint2World, please cite both this demo and the underlying
*World Reconstruction From Inconsistent Views* paper (see the root `README.md`
for the BibTeX entry).
