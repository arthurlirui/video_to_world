"""End-to-end integration tests for the Paint2World pipeline.

Runs the full orchestrator (all stages, synthetic backends, tiny settings)
on CPU and verifies every stage produces its artifacts, plus the component
toggles (no_object_scene / no_canvas_depth / no_object_rigid) route through
the right code paths.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paint2world.configs import Paint2WorldConfig
from paint2world.pipeline import run_pipeline
from paint2world.data_utils import generate_synthetic_oil_painting


def _tiny_painting(tmp_path: str) -> str:
    """Generate a small sample painting with the bundled generator."""
    os.makedirs(tmp_path, exist_ok=True)
    path = os.path.join(tmp_path, "painting.png")
    img = generate_synthetic_oil_painting(width=192, height=144, seed=3)
    from PIL import Image

    Image.fromarray(img).save(path)
    return path


def _tiny_cfg(painting: str, out_root: str) -> Paint2WorldConfig:
    cfg = Paint2WorldConfig(out_root=out_root)
    cfg.painting.painting_path = painting
    cfg.painting.max_side = 192
    cfg.video.num_frames = 4
    cfg.triposplat.triplane_resolution = 16
    cfg.triposplat.query_resolution = 8
    cfg.triposplat.refinement_iters = 3
    cfg.objects.min_object_area = 200
    cfg.objects.max_objects = 6
    return cfg


def test_full_pipeline_all_stages():
    out = "results/_test_it/pipeline_full"
    painting = _tiny_painting("results/_test_it")
    cfg = _tiny_cfg(painting, out)
    res = run_pipeline(cfg)

    stages = res["stages"]
    assert "painting_analysis" in stages
    assert "object_scene" in stages
    assert "video_generation" in stages
    assert "alignment" in stages
    assert "gs" in stages

    # Stage artifacts on disk.
    assert os.path.exists(os.path.join(out, "stage_minus_1", "painting.npz"))
    assert os.path.exists(os.path.join(out, "stage_object", "objects", "objects.npz"))
    assert os.path.exists(os.path.join(out, "stage_object", "triposplat", "triposplat.npz"))
    assert os.path.exists(os.path.join(out, "stage_object", "canvas", "canvas.npz"))
    assert os.path.exists(os.path.join(out, "stage_object", "canvas", "relations.json"))
    assert os.path.exists(os.path.join(out, "stage_0", "video.npz"))
    assert os.path.exists(os.path.join(out, "stage_0", "video.mp4"))
    assert os.path.exists(os.path.join(out, "stage_1", "alignment.npz"))
    assert os.path.exists(os.path.join(out, "stage_3", "gaussians.npz"))

    # Object-stage summary recorded.
    obj = stages["object_scene"]
    assert obj["num_objects"] >= 1
    assert obj["num_relations"] >= 1


def test_pipeline_no_object_scene_falls_back():
    out = "results/_test_it/pipeline_no_object"
    painting = _tiny_painting("results/_test_it")
    cfg = _tiny_cfg(painting, out)
    cfg.run_stage_object = False
    res = run_pipeline(cfg)
    assert "object_scene" not in res["stages"]
    # GS still produced via the alignment-based path.
    assert os.path.exists(os.path.join(out, "stage_3", "gaussians.npz"))
    assert not os.path.exists(os.path.join(out, "stage_object"))


def test_pipeline_no_object_rigid_uses_free_positions():
    out = "results/_test_it/pipeline_no_rigid"
    painting = _tiny_painting("results/_test_it")
    cfg = _tiny_cfg(painting, out)
    cfg.canvas.enable_object_rigid_gs = False
    res = run_pipeline(cfg)
    assert "object_scene" in res["stages"]
    g = np.load(os.path.join(out, "stage_3", "gaussians.npz"))
    assert "object_label" in g  # canvas init retained
    assert "relation_gap_drift" not in g  # rigid diagnostics only in rigid mode


def test_pipeline_no_canvas_depth_still_runs():
    out = "results/_test_it/pipeline_no_canvas_depth"
    painting = _tiny_painting("results/_test_it")
    cfg = _tiny_cfg(painting, out)
    cfg.canvas.enable_canvas_depth_control = False
    res = run_pipeline(cfg)
    assert "object_scene" in res["stages"]
    assert os.path.exists(os.path.join(out, "stage_0", "video.npz"))
