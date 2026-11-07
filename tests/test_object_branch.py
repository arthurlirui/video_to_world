"""Unit tests for the Paint2World object-level branch.

Covers:
  * Stage -0.5a: object segmentation (synthetic backend),
  * Stage -0.5b: TripoSplat-style triplane encoding / decoding / refinement,
  * Stage -0.5c: spatial-dependency graph, ground plane, scene canvas,
    canvas depth, and canvas-anchored GS training.

All tests run on CPU with small synthetic inputs (no network weights).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paint2world.configs import (
    ObjectSegmentationConfig,
    TripoSplatConfig,
    SceneCanvasConfig,
    StrokeCoherentGSConfig,
)
from paint2world import object_segmentation, triposplat, scene_canvas


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_painting(h: int = 96, w: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """A tiny 'painting': far sky band, a near red box on a mid-distance
    ground band, gently smoothed (paintings are not piecewise constant)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    # Middle band background: distant greenery.
    img[:] = (60, 110, 60)
    depth[:] = 0.75
    # Sky (top quarter): farthest.
    img[: h // 4] = (180, 200, 230)
    depth[: h // 4] = 0.95
    # Ground (bottom quarter): mid-far.
    img[3 * h // 4 :] = (120, 100, 80)
    depth[3 * h // 4 :] = 0.6
    # Object: a red box in the middle band, near.
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 3, 2 * w // 3
    img[y0:y1, x0:x1] = (190, 40, 40)
    depth[y0:y1, x0:x1] = 0.25
    # Gentle 2D box blur so the depth is a plausible smooth surface.
    t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
    k = torch.ones(1, 1, 5, 5) / 25.0
    depth = torch.nn.functional.conv2d(t, k, padding=2).squeeze().numpy()
    return img, np.clip(depth, 0, 1).astype(np.float32)


def _run_object_branch(h=96, w=128, refine_iters=3, with_strokes=True):
    img, depth = _make_synthetic_painting(h, w)
    strokes = None
    if with_strokes:
        from paint2world.painting_analysis import synthetic_stroke_segmentation

        rgb_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        strokes = synthetic_stroke_segmentation(rgb_t, min_area=32).numpy()
    seg = object_segmentation.segment_objects(
        ObjectSegmentationConfig(out_dir="results/_test_ut/objects", min_object_area=200),
        img, depth,
    )
    ts = triposplat.generate_objects(
        TripoSplatConfig(triplane_resolution=16, query_resolution=8, refinement_iters=refine_iters),
        img, depth, seg,
        out_dir="results/_test_ut/triposplat",
    )
    canvas_res = scene_canvas.build_scene_canvas(
        SceneCanvasConfig(), img, depth, seg, ts,
        painting_c2w=torch.eye(4), out_dir="results/_test_ut/canvas",
        strokes=strokes,
    )
    return img, depth, seg, ts, canvas_res


# ---------------------------------------------------------------------------
# Stage -0.5a: object segmentation
# ---------------------------------------------------------------------------

class TestObjectSegmentation:
    def test_finds_objects_on_synthetic_painting(self):
        img, depth = _make_synthetic_painting()
        seg = object_segmentation.segment_objects(
            ObjectSegmentationConfig(out_dir="results/_test_ut/objects"), img, depth
        )
        assert seg["num_objects"] >= 1
        labels = seg["labels"]
        assert labels.shape == (img.shape[0], img.shape[1])
        assert labels.max() == seg["num_objects"]
        # Object areas sum to <= total pixels; masks are disjoint.
        total_mask = sum(int((labels == i + 1).sum()) for i in range(seg["num_objects"]))
        assert 0 < total_mask <= img.shape[0] * img.shape[1]

    def test_object_metadata_fields(self):
        img, depth = _make_synthetic_painting()
        seg = object_segmentation.segment_objects(
            ObjectSegmentationConfig(out_dir="results/_test_ut/objects"), img, depth
        )
        for o in seg["objects"]:
            assert set(["id", "area", "objectness", "bbox", "mean_depth", "mask"]) <= set(o)
            assert o["area"] > 0
            assert 0.0 <= o["mean_depth"] <= 1.0

    def test_objectness_sorted_descending(self):
        img, depth = _make_synthetic_painting()
        seg = object_segmentation.segment_objects(
            ObjectSegmentationConfig(out_dir="results/_test_ut/objects"), img, depth
        )
        scores = [o["objectness"] for o in seg["objects"]]
        assert scores == sorted(scores, reverse=True)

    def test_persistence(self):
        img, depth = _make_synthetic_painting()
        out = "results/_test_ut/objects_persist"
        object_segmentation.segment_objects(
            ObjectSegmentationConfig(out_dir=out), img, depth
        )
        assert os.path.exists(os.path.join(out, "objects.npz"))
        assert os.path.exists(os.path.join(out, "objects.png"))
        data = np.load(os.path.join(out, "objects.npz"))
        assert "labels" in data and "num_objects" in data


# ---------------------------------------------------------------------------
# Stage -0.5b: TripoSplat
# ---------------------------------------------------------------------------

class TestTripoSplat:
    def test_triplane_splat_and_features(self):
        tri = triposplat.Triplane(8)
        pts = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        cols = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        zs = torch.tensor([1.0, 2.0])
        tri.splat(pts, cols, zs)
        feats = tri.features()
        for ax in triposplat.Triplane.AXES:
            assert ax in feats
            assert feats[ax].shape[0] == triposplat.TRIPLANE_FEATURE_DIM
            assert feats[ax].shape[-1] == 8
            assert float(feats[ax][3].sum()) > 0  # density present

    def test_icp_recovers_rotation(self):
        torch.manual_seed(0)
        src = torch.rand(200, 3) * 2 - 1
        # Rotate around y by a known angle.
        ang = 0.35
        c, s = np.cos(ang), np.sin(ang)
        R = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
        moved = src @ R.T
        aligned, T = triposplat.icp_align(moved, src, num_iters=25)
        err = (aligned - src).norm(dim=1).mean()
        assert float(err) < 0.05, f"ICP failed to align, err={err}"

    def test_decode_emits_gaussians(self):
        tri = triposplat.Triplane(16)
        pts = torch.rand(500, 3) * 1.2 - 0.6
        cols = torch.rand(500, 3)
        zs = torch.rand(500) * 3 + 1
        tri.splat(pts, cols, zs)
        g = triposplat.decode_gaussians_from_triplane(tri, query_resolution=12, min_gaussians=16)
        assert g["position"].shape[0] >= 16
        assert g["position"].shape[1] == 3
        assert g["sh_dc"].shape == g["position"].shape
        assert g["opacity"].shape[0] == g["position"].shape[0]
        assert float(g["sh_dc"].min()) >= 0.0 and float(g["sh_dc"].max()) <= 1.0

    def test_generate_object_refinement_reduces_loss(self):
        img, depth = _make_synthetic_painting()
        rgb = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        dep = torch.from_numpy(depth).float().unsqueeze(0)
        mask = torch.zeros(img.shape[:2], dtype=torch.bool)
        mask[32:64, 42:86] = True
        meta = {"id": 1, "mask": mask.numpy(), "mean_depth": 0.3, "bbox": [42, 32, 86, 64], "area": int(mask.sum())}
        cfg = TripoSplatConfig(triplane_resolution=16, query_resolution=10, refinement_iters=8)
        entry = triposplat.generate_object(cfg, meta, rgb, dep)
        assert entry["num_gaussians"] > 0
        assert entry["refinement_loss_initial"] is not None
        assert entry["refinement_loss_final"] <= entry["refinement_loss_initial"] + 1e-3

    def test_generate_objects_combined_bundle(self):
        img, depth = _make_synthetic_painting()
        seg = object_segmentation.segment_objects(
            ObjectSegmentationConfig(out_dir="results/_test_ut/objects", min_object_area=200), img, depth
        )
        res = triposplat.generate_objects(
            TripoSplatConfig(triplane_resolution=12, query_resolution=8, refinement_iters=2),
            img, depth, seg,
        )
        assert res["num_objects"] == seg["num_objects"]
        n_lab = res["object_label"].shape[0]
        assert n_lab == res["combined"]["position"].shape[0]
        assert set(res["object_label"].tolist()) <= set(range(0, seg["num_objects"] + 1))


# ---------------------------------------------------------------------------
# Stage -0.5c: scene canvas
# ---------------------------------------------------------------------------

class TestSceneCanvas:
    def test_ground_plane_fit(self):
        pts = torch.rand(1000, 3) * 4 - 2
        pts[:, 1] = 1.0 + 0.01 * torch.rand(1000) + 2 * pts[:, 0] * 0.0  # flat ground at y=1
        plane = scene_canvas.fit_ground_plane(pts, bottom_fraction=0.3)
        y = scene_canvas.ground_y_at(plane, 0.0, 0.0)
        assert abs(y - 1.0) < 0.05

    def test_canvas_build_and_relations(self):
        _, _, seg, ts, canvas_res = _run_object_branch()
        canvas = canvas_res["canvas"]
        assert len(canvas.objects) == seg["num_objects"]
        assert canvas_res["num_relations"] == len(canvas.relations)
        for r in canvas.relations:
            assert r["type"] in ("rests_on", "rests_on_ground", "in_front_of", "adjacent_to")
            assert "a" in r and "b" in r
        # Relation endpoints reference existing objects.
        ids = set(o["id"] for o in canvas.objects)
        for r in canvas.relations:
            assert r["a"] in ids
            assert r["b"] in ids or r["b"] == "ground"

    def test_canvas_gaussians_and_labels(self):
        _, _, seg, ts, canvas_res = _run_object_branch()
        g = canvas_res["canvas"].canvas_gaussians_camera()
        n = g["position"].shape[0]
        assert n > 0
        assert g["object_label"].shape[0] == n
        # Background label 0 present when some pixels are unassigned.
        assert 0 in set(g["object_label"].tolist())

    def test_canvas_depth_no_holes_and_finite(self):
        img, depth, seg, ts, canvas_res = _run_object_branch()
        H, W = img.shape[:2]
        cd = canvas_res["canvas"].canvas_depth(H, W)
        assert cd.shape == (H, W)
        assert bool((cd >= 0).all()) and bool((cd <= 1).all())
        assert float(cd.min()) > 0.0  # hole-free

    def test_world_transform_consistency(self):
        _, _, seg, ts, canvas_res = _run_object_branch()
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor([0.5, -0.3, 0.0])
        canvas = canvas_res["canvas"]
        canvas.painting_c2w = c2w
        g_cam = canvas.canvas_gaussians_camera()["position"]
        g_world = canvas.world_gaussians()["position"]
        # world = R @ cam + t with R = I.
        assert torch.allclose(g_world, g_cam + c2w[:3, 3].unsqueeze(0), atol=1e-5)

    def test_canvas_anchored_training_preserves_relations(self):
        img, depth, seg, ts, canvas_res = _run_object_branch(refine_iters=2)
        canvas = canvas_res["canvas"]
        rgb = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        frames = rgb.unsqueeze(0).expand(2, -1, -1, -1).contiguous()
        out = scene_canvas.train_canvas_anchored_gs(
            canvas,
            SceneCanvasConfig(relation_preservation_weight=0.2),
            StrokeCoherentGSConfig(),
            frames, [torch.eye(4), torch.eye(4)],
            rgb, torch.eye(4),
            out_dir="results/_test_ut/canvas_gs",
            num_iters=10,
        )
        assert out["position"].shape[0] > 0
        # Relation gaps stay close to their initial values.
        assert out["relation_gap_drift"] < 0.1
        # Every object has a (small) learned delta.
        assert len(out["object_deltas"]) >= 1
        for d in out["object_deltas"].values():
            assert "translation" in d and "yaw_deg" in d

    def test_canvas_owns_stroke_labels(self):
        """The canvas must carry brush-stroke labels so the stroke-coherence
        loss is active in canvas-anchored training (regression test)."""
        _, _, seg, ts, canvas_res = _run_object_branch(with_strokes=True)
        canvas = canvas_res["canvas"]
        assert canvas.stroke_labels is not None
        g = canvas.canvas_gaussians_camera()
        assert "stroke_label" in g
        assert g["stroke_label"].shape[0] == g["position"].shape[0]
        assert int((g["stroke_label"] > 0).sum()) > 0, "no stroke labels survived projection"
        # Persisted copy also carries them.
        data = np.load(os.path.join(canvas_res["out_dir"], "canvas.npz"))
        assert "stroke_label" in data
        assert int((data["stroke_label"] > 0).sum()) > 0

    def test_canvas_persistence(self):
        img, depth, seg, ts, canvas_res = _run_object_branch(refine_iters=1)
        out = canvas_res["out_dir"]
        assert os.path.exists(os.path.join(out, "canvas.npz"))
        assert os.path.exists(os.path.join(out, "relations.json"))
        assert os.path.exists(os.path.join(out, "layout.png"))
        data = np.load(os.path.join(out, "canvas.npz"))
        assert "object_label" in data and "painting_c2w" in data
        import json

        with open(os.path.join(out, "relations.json")) as f:
            rel = json.load(f)
        assert "relations" in rel and "ground" in rel
