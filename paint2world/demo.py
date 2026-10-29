"""Gradio demo UI for Paint2World.

Provides an interactive interface that:
  * lets the user upload an oil painting (or pick a generated sample),
  * runs Stage -1 (painting analysis) and visualises depth / normals / strokes,
  * runs Stage 0 (painting-conditioned video generation) and plays the video,
  * runs Stage 1 + Stage 3.2 (painting-anchored alignment + stroke-coherent GS)
    and shows the canonical-cloud preview and the painting-view render.

The demo uses the synthetic / mock backends so it runs end-to-end in a CPU
environment. Install ``gradio`` to use it.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import torch

from .configs import Paint2WorldConfig
from . import data_utils, painting_analysis, video_generation, painting_anchored_alignment, stroke_coherent_gs


def _build_config(painting_path: str, out_root: str, num_frames: int, trajectory: str,
                  angle_deg: float, baseline: float, depth_strength: float) -> Paint2WorldConfig:
    cfg = Paint2WorldConfig(out_root=out_root)
    cfg.painting.painting_path = painting_path
    cfg.video.num_frames = num_frames
    cfg.video.trajectory_type = trajectory
    cfg.video.trajectory_angle_deg = angle_deg
    cfg.video.trajectory_baseline = baseline
    cfg.video.depth_warp_control_strength = depth_strength
    return cfg


def run_stage_minus_1(painting_path: str):
    cfg = Paint2WorldConfig(out_root=tempfile.mkdtemp(prefix="p2w_s-1_"))
    cfg.painting.painting_path = painting_path
    cfg.painting.out_dir = os.path.join(cfg.out_root, "stage_minus_1")
    pa = painting_analysis.analyze_painting(cfg.painting)
    return (
        pa["image"],
        _png_path(os.path.join(pa["out_dir"], "depth.png")),
        _png_path(os.path.join(pa["out_dir"], "normals.png")),
        _png_path(os.path.join(pa["out_dir"], "strokes.png")),
        _png_path(os.path.join(pa["out_dir"], "confidence.png")),
        f"Detected {pa['num_strokes']} brush-stroke segments.",
    )


def run_stage_0(painting_path: str, num_frames: int, trajectory: str, angle_deg: float,
                baseline: float, depth_strength: float):
    from .pipeline import _to_torch_img

    # Run Stage -1 first to get painting depth.
    cfg = _build_config(painting_path, tempfile.mkdtemp(prefix="p2w_s0_"), num_frames, trajectory, angle_deg, baseline, depth_strength)
    cfg.painting.out_dir = os.path.join(cfg.out_root, "stage_minus_1")
    pa = painting_analysis.analyze_painting(cfg.painting)
    painting_rgb_t = _to_torch_img(pa["image"], torch.device("cpu"))
    painting_depth_t = torch.from_numpy(pa["depth"]).float()
    painting_strokes_t = torch.from_numpy(pa["strokes"]).long()

    cfg.video.out_dir = os.path.join(cfg.out_root, "stage_0")
    vg = video_generation.generate_video(cfg.video, painting_rgb_t, painting_depth_t, painting_strokes_t)
    return vg["mp4_path"], _png_path(os.path.join(vg["out_dir"], "frame_grid.png"))


def run_full(painting_path: str, num_frames: int, trajectory: str, angle_deg: float,
             baseline: float, depth_strength: float):
    from .pipeline import run_pipeline

    out_root = tempfile.mkdtemp(prefix="p2w_full_")
    cfg = _build_config(painting_path, out_root, num_frames, trajectory, angle_deg, baseline, depth_strength)
    run_pipeline(cfg)
    return (
        _png_path(os.path.join(out_root, "stage_minus_1", "depth.png")),
        _png_path(os.path.join(out_root, "stage_minus_1", "strokes.png")),
        os.path.join(out_root, "stage_0", "video.mp4"),
        _png_path(os.path.join(out_root, "stage_0", "frame_grid.png")),
        _png_path(os.path.join(out_root, "stage_1", "canonical_preview.png")),
        _png_path(os.path.join(out_root, "stage_3", "render_painting_view.png")),
        f"Pipeline complete. Outputs in {out_root}",
    )


def _png_path(p: str) -> str:
    return p if os.path.exists(p) else None


def build_demo():
    """Build and return the Gradio Blocks demo."""
    import gradio as gr

    sample_paths = data_utils.make_sample_paintings()

    with gr.Blocks(title="Paint2World — Oil Painting to 3D World") as demo:
        gr.Markdown(
            "# Paint2World\n"
            "3D-consistent video and scene generation from human oil paintings. "
            "Upload an oil painting (or pick a sample), analyse it, generate a "
            "3D-stable video flythrough, and reconstruct a 3D Gaussian Splatting scene."
        )
        with gr.Row():
            painting_in = gr.Image(label="Input oil painting", type="filepath", value=sample_paths[0])
            sample_picker = gr.Dropdown(
                label="Or pick a sample", choices=sample_paths, value=sample_paths[0]
            )
        with gr.Row():
            num_frames = gr.Slider(8, 80, value=32, step=4, label="Number of generated frames")
            trajectory = gr.Dropdown(
                ["orbit", "dolly", "pan", "spiral"], value="orbit", label="Camera trajectory"
            )
            angle_deg = gr.Slider(5.0, 60.0, value=25.0, step=1.0, label="Orbit/spiral angle (deg)")
            baseline = gr.Slider(0.05, 0.4, value=0.18, step=0.01, label="Trajectory baseline")
            depth_strength = gr.Slider(0.0, 1.0, value=0.85, step=0.05,
                                        label="Depth-warp control strength (3D stability)")
        with gr.Tab("Stage -1: Painting Analysis"):
            btn1 = gr.Button("Run painting analysis")
            with gr.Row():
                out_img = gr.Image(label="Painting")
                out_depth = gr.Image(label="Depth")
                out_normals = gr.Image(label="Normals")
            with gr.Row():
                out_strokes = gr.Image(label="Brush strokes")
                out_conf = gr.Image(label="Confidence")
            out_status_1 = gr.Textbox(label="Status")
            btn1.click(
                run_stage_minus_1, inputs=[painting_in],
                outputs=[out_img, out_depth, out_normals, out_strokes, out_conf, out_status_1],
            )
        with gr.Tab("Stage 0: Video Generation"):
            btn2 = gr.Button("Generate 3D-stable video")
            out_video = gr.Video(label="Generated flythrough")
            out_grid = gr.Image(label="Frame contact sheet")
            btn2.click(
                run_stage_0,
                inputs=[painting_in, num_frames, trajectory, angle_deg, baseline, depth_strength],
                outputs=[out_video, out_grid],
            )
        with gr.Tab("Full Pipeline"):
            btn3 = gr.Button("Run full pipeline (Stage -1 → Stage 3.2)")
            with gr.Row():
                full_depth = gr.Image(label="Painting depth")
                full_strokes = gr.Image(label="Brush strokes")
            with gr.Row():
                full_video = gr.Video(label="Generated video")
                full_grid = gr.Image(label="Frame grid")
            with gr.Row():
                full_canonical = gr.Image(label="Aligned canonical cloud")
                full_render = gr.Image(label="Painting-view render")
            full_status = gr.Textbox(label="Status")
            btn3.click(
                run_full,
                inputs=[painting_in, num_frames, trajectory, angle_deg, baseline, depth_strength],
                outputs=[full_depth, full_strokes, full_video, full_grid, full_canonical, full_render, full_status],
            )
        gr.Markdown(
            "*Demo runs with synthetic / mock backends out of the box. "
            "See the README for swapping in DepthAnything-3, Wan-I2V, and gsplat.*"
        )
    return demo


def main():
    demo = build_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
