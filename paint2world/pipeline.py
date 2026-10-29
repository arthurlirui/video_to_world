"""End-to-end Paint2World pipeline orchestrator.

Chains Stage -1 (painting analysis), Stage 0 (painting-conditioned video
generation with 3D-stability controls), Stage 1 (painting-anchored non-rigid
alignment), and Stage 3.2 (stroke-coherent GS training).

The orchestrator is backend-agnostic: it consumes the painting-analysis
outputs, passes them through Stage 0 and Stage 1, and produces a 3D Gaussian
Splatting scene. The heavy CUDA-only backends (DepthAnything-3, tinycudann,
gsplat, RoMa) are plugged in via configuration; the demo runs with the
synthetic / mock backends out of the box.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from .configs import Paint2WorldConfig
from . import painting_analysis, video_generation, painting_anchored_alignment, stroke_coherent_gs


def _to_torch_img(arr_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr_uint8).float().permute(2, 0, 1).to(device) / 255.0


def _unproject_frame(
    frame_rgb: torch.Tensor,  # (3, H, W) in [0, 1]
    depth: torch.Tensor,  # (H, W) in [0, 1]
    c2w: torch.Tensor,  # (4, 4)
    focal_frac: float = 1.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unproject a frame + depth into a coloured point cloud in canonical (world) coords."""
    device = frame_rgb.device
    _, H, W = frame_rgb.shape
    f = focal_frac * float(max(H, W))
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = float(W) * 0.5, float(H) * 0.5
    z = 0.2 + 4.0 * depth
    x = (xs - cx) / f * z
    y = (ys - cy) / f * z
    pts_cam = torch.stack([x.flatten(), y.flatten(), z.flatten(), torch.ones_like(z).flatten()], dim=0)
    world = (c2w.to(device=device, dtype=torch.float32) @ pts_cam)[:3].T  # (N, 3)
    cols = frame_rgb.reshape(3, -1).T
    return world, cols


def _frame_depth_from_render(
    frame_rgb: torch.Tensor,
    c2w: torch.Tensor,
    painting_pts: torch.Tensor,
    painting_cols: torch.Tensor,
    H: int,
    W: int,
    focal_frac: float = 1.2,
) -> torch.Tensor:
    """Approximate per-frame depth by re-projecting the painting point cloud.

    This is used by Stage 1 (fallback) to get a depth value for each generated
    pixel without running a real monocular estimator. Real deployments would
    run DepthAnything-3 on each generated frame.
    """
    from .video_generation import _project_depth

    pts = torch.cat([painting_pts, painting_cols], dim=-1)
    depth = _project_depth(pts, c2w, H, W, focal_frac=focal_frac)
    # Normalise to [0, 1].
    depth = (depth - 0.2) / 4.0
    return depth.clamp(0.0, 1.0)


def run_pipeline(cfg: Paint2WorldConfig, device: Optional[torch.device] = None) -> dict:
    """Run the full Paint2World pipeline end-to-end."""
    device = device or torch.device("cpu")
    os.makedirs(cfg.out_root, exist_ok=True)
    results: dict = {"config": asdict(cfg), "stages": {}}

    # ----- Stage -1: Painting analysis -----
    if cfg.run_stage_minus_1:
        print("[Paint2World] Stage -1: painting analysis")
        # Update output dir of the painting analysis stage to live under out_root.
        painting_cfg = cfg.painting
        painting_cfg.out_dir = os.path.join(cfg.out_root, "stage_minus_1")
        pa = painting_analysis.analyze_painting(painting_cfg)
        results["stages"]["painting_analysis"] = pa
    else:
        # Load from a previous run.
        pa_path = os.path.join(cfg.out_root, "stage_minus_1", "painting.npz")
        if not os.path.exists(pa_path):
            raise FileNotFoundError(
                f"Stage -1 skipped but no prior output found at {pa_path}"
            )
        data = np.load(pa_path, allow_pickle=True)
        pa = {
            "image": data["image"],
            "depth": data["depth"],
            "normals": data["normals"],
            "strokes": data["strokes"],
            "confidence": data["confidence"],
            "num_strokes": int(data["strokes"].max()),
            "out_dir": os.path.join(cfg.out_root, "stage_minus_1"),
        }
        results["stages"]["painting_analysis"] = pa

    painting_rgb_t = _to_torch_img(pa["image"], device)
    painting_depth_t = torch.from_numpy(pa["depth"]).float().to(device)
    painting_strokes_t = torch.from_numpy(pa["strokes"]).long().to(device)

    # ----- Stage 0: Painting-conditioned video generation -----
    if cfg.run_stage_0:
        print("[Paint2World] Stage 0: painting-conditioned video generation")
        video_cfg = cfg.video
        video_cfg.out_dir = os.path.join(cfg.out_root, "stage_0")
        vg = video_generation.generate_video(
            video_cfg, painting_rgb_t, painting_depth_t, painting_strokes_t
        )
        results["stages"]["video_generation"] = {
            "out_dir": vg["out_dir"],
            "mp4_path": vg["mp4_path"],
        }
    else:
        vg_path = os.path.join(cfg.out_root, "stage_0", "video.npz")
        if not os.path.exists(vg_path):
            raise FileNotFoundError(
                f"Stage 0 skipped but no prior output found at {vg_path}"
            )
        data = np.load(vg_path, allow_pickle=True)
        vg = {
            "frames": torch.from_numpy(data["frames"]).float().permute(0, 3, 1, 2).to(device) / 255.0,
            "trajectory": torch.from_numpy(data["trajectory"]).float().to(device),
            "depth_warp_control": torch.from_numpy(data["depth_warp_control"]).float().to(device),
        }
        results["stages"]["video_generation"] = {"out_dir": os.path.join(cfg.out_root, "stage_0")}

    # ----- Stage 1: Painting-anchored alignment -----
    if cfg.run_stage_1:
        print("[Paint2World] Stage 1: painting-anchored non-rigid alignment")
        align_out_dir = os.path.join(cfg.out_root, "stage_1")
        frames = vg["frames"] if "frames" in vg else None
        if frames is None:
            # Reload if we skipped Stage 0 with a different code path.
            data = np.load(os.path.join(cfg.out_root, "stage_0", "video.npz"), allow_pickle=True)
            frames = torch.from_numpy(data["frames"]).float().permute(0, 3, 1, 2).to(device) / 255.0
        trajectory = vg["trajectory"] if "trajectory" in vg else torch.from_numpy(
            np.load(os.path.join(cfg.out_root, "stage_0", "video.npz"), allow_pickle=True)["trajectory"]
        ).float().to(device)

        # Build per-frame point clouds by re-projecting the painting (the mock
        # video's geometry is exactly the painting warped along the trajectory).
        N, C, H, W = frames.shape
        # Unproject the painting in the canonical frame.
        painting_c2w = trajectory[0]
        painting_world, painting_cols_world = _unproject_frame(
            painting_rgb_t, painting_depth_t, painting_c2w
        )

        # For each frame, unproject using the frame's own camera and the
        # depth-warp-control-implied depth.
        frame_pts_list = []
        frame_cols_list = []
        frame_c2w_list = []
        # Subsample frames for the alignment to bound memory.
        sub = max(1, N // cfg.alignment.base_num_frames) if cfg.alignment.base_num_frames > 0 else 1
        for i in range(0, N, sub):
            c2w_i = trajectory[i]
            frame_c2w_list.append(c2w_i)
            # Depth for this frame: use the depth-warp control (it's the
            # painting depth warped into this view).
            depth_i = vg["depth_warp_control"][i]
            pts_i, cols_i = _unproject_frame(frames[i], depth_i, c2w_i)
            # Subsample pixels for memory.
            M = pts_i.shape[0]
            keep = torch.randperm(M, device=device)[: min(M, 4000)]
            frame_pts_list.append(pts_i[keep])
            frame_cols_list.append(cols_i[keep])

        align = painting_anchored_alignment.align_painting_anchored(
            cfg.alignment,
            painting_rgb=pa["image"],
            painting_depth=pa["depth"],
            painting_conf=pa["confidence"],
            frame_pts_list=frame_pts_list,
            frame_cols_list=frame_cols_list,
            frame_c2w_list=frame_c2w_list,
            painting_c2w=painting_c2w,
            out_dir=align_out_dir,
        )
        results["stages"]["alignment"] = {"out_dir": align_out_dir}
    else:
        # Load prior alignment output.
        align_path = os.path.join(cfg.out_root, "stage_1", "alignment.npz")
        if not os.path.exists(align_path):
            raise FileNotFoundError(
                f"Stage 1 skipped but no prior output found at {align_path}"
            )
        data = np.load(align_path, allow_pickle=True)
        align = {
            "canonical_pts": torch.from_numpy(data["canonical_pts"]).float(),
            "canonical_cols": torch.from_numpy(data["canonical_cols"]).float(),
            "canonical_conf": torch.from_numpy(data["canonical_conf"]).float(),
        }
        results["stages"]["alignment"] = {"out_dir": os.path.join(cfg.out_root, "stage_1")}

    # ----- Stage 3.2: Stroke-coherent GS training -----
    if cfg.run_stage_3:
        print("[Paint2World] Stage 3.2: stroke-coherent GS training")
        gs_out_dir = os.path.join(cfg.out_root, "stage_3")
        frames = vg["frames"] if "frames" in vg else None
        if frames is None:
            data = np.load(os.path.join(cfg.out_root, "stage_0", "video.npz"), allow_pickle=True)
            frames = torch.from_numpy(data["frames"]).float().permute(0, 3, 1, 2).to(device) / 255.0
        trajectory = vg["trajectory"] if "trajectory" in vg else torch.from_numpy(
            np.load(os.path.join(cfg.out_root, "stage_0", "video.npz"), allow_pickle=True)["trajectory"]
        ).float().to(device)
        painting_c2w = trajectory[0]

        # Subsample frames for GS training.
        N = frames.shape[0]
        sub = max(1, N // 16)
        gs_frames = frames[::sub]
        gs_c2w = [trajectory[i] for i in range(0, N, sub)]

        gs = stroke_coherent_gs.train_stroke_coherent_gs(
            cfg.gs,
            canonical_pts=align["canonical_pts"].to(device),
            canonical_cols=align["canonical_cols"].to(device),
            canonical_strokes=pa["strokes"],
            frame_images=gs_frames,
            frame_c2w_list=gs_c2w,
            painting_rgb=painting_rgb_t,
            painting_c2w=painting_c2w,
            out_dir=gs_out_dir,
            num_iters=300,
        )
        results["stages"]["gs"] = {"out_dir": gs_out_dir}

    print(f"[Paint2World] Done. Outputs in {cfg.out_root}")
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Paint2World end-to-end pipeline")
    parser.add_argument("--painting", required=True, help="Path to input oil-painting image")
    parser.add_argument("--out_root", default="data/sample_runs/paint2world", help="Output root")
    parser.add_argument("--num_frames", type=int, default=40, help="Number of generated frames")
    parser.add_argument("--trajectory", default="orbit", choices=["orbit", "dolly", "pan", "spiral"])
    parser.add_argument("--angle_deg", type=float, default=25.0)
    parser.add_argument("--baseline", type=float, default=0.18)
    parser.add_argument("--depth_strength", type=float, default=0.85, help="Depth-warp control strength")
    parser.add_argument("--stroke_coherence", action="store_true", default=True)
    parser.add_argument("--no_stage_minus_1", action="store_true")
    parser.add_argument("--no_stage_0", action="store_true")
    parser.add_argument("--no_stage_1", action="store_true")
    parser.add_argument("--no_stage_3", action="store_true")
    args = parser.parse_args()

    cfg = Paint2WorldConfig(out_root=args.out_root)
    cfg.painting.painting_path = args.painting
    cfg.video.num_frames = args.num_frames
    cfg.video.trajectory_type = args.trajectory
    cfg.video.trajectory_angle_deg = args.angle_deg
    cfg.video.trajectory_baseline = args.baseline
    cfg.video.depth_warp_control_strength = args.depth_strength
    cfg.video.stroke_consistency_attention = args.stroke_coherence
    cfg.run_stage_minus_1 = not args.no_stage_minus_1
    cfg.run_stage_0 = not args.no_stage_0
    cfg.run_stage_1 = not args.no_stage_1
    cfg.run_stage_3 = not args.no_stage_3

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
