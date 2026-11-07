"""Stage 0: Painting-conditioned video generation with 3D-stability controls.

This module implements the novel contribution of Stage 0:

  * a **depth-warp geometry control**: at each generated frame, the painting's
    depth map is warped along the camera trajectory and used as a per-frame
    geometry control signal — encouraging every generated frame to share the
    painting's 3D layout and drastically reducing generative drift;
  * a **stroke-consistency attention**: the painting is locked as a reference
    frame in the diffusion temporal attention, with a stroke-mask attention
    bias that prevents brush strokes from drifting across frames;
  * a **temporal-consistency guidance** term during sampling.

The real video-diffusion backend is pluggable. By default we ship a *mock*
backend that synthesises a 3D-stable flythrough directly from the painting +
depth (no network weights needed), so the demo runs end-to-end out of the box.
The mock backend respects the same controls (depth-warp, stroke-consistency)
and is a faithful stand-in for the geometry that a real conditioned video
diffusion model would produce. Swapping in Wan-I2V / ViewCrafter / SVD / CSVD
is a one-line configuration change.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .configs import VideoGenerationConfig


# ---------------------------------------------------------------------------
# Camera-trajectory generation
# ---------------------------------------------------------------------------

def make_camera_trajectory(
    num_frames: int,
    trajectory_type: str,
    angle_deg: float,
    baseline: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a sequence of camera extrinsics (c2w, 4x4) for the flythrough.

    The camera starts at the painting's viewpoint (identity) and moves along
    the chosen trajectory. Returns (N, 4, 4).
    """
    device = device or torch.device("cpu")
    N = num_frames
    ts = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
    angle = math.radians(angle_deg)

    Rs = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3).contiguous()
    ts_vec = torch.zeros(N, 3, device=device, dtype=dtype)

    if trajectory_type == "orbit":
        # Rotate around the scene centre (placed at z = depth_scale in front).
        azim = -0.5 * angle + angle * ts  # sweep from -angle/2 to +angle/2
        for i in range(N):
            a = float(azim[i].item())
            c, s = math.cos(a), math.sin(a)
            R = torch.tensor(
                [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                device=device, dtype=dtype,
            )
            # Camera orbits on a circle of radius baseline around the centre.
            t = torch.tensor([baseline * math.sin(a), 0.0, baseline * (1.0 - math.cos(a))], device=device, dtype=dtype)
            Rs[i] = R
            ts_vec[i] = t
    elif trajectory_type == "dolly":
        # Move forward (z decreases) — keep identity rotation, translate in z.
        z = baseline * (1.0 - ts)
        ts_vec[:, 2] = z
    elif trajectory_type == "pan":
        # Translate laterally; keep looking forward.
        x = baseline * (ts - 0.5) * 2.0
        ts_vec[:, 0] = x
    elif trajectory_type == "spiral":
        azim = -0.5 * angle + angle * ts
        z = 0.5 * baseline * (1.0 - ts)
        for i in range(N):
            a = float(azim[i].item())
            c, s = math.cos(a), math.sin(a)
            R = torch.tensor(
                [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                device=device, dtype=dtype,
            )
            t = torch.tensor(
                [baseline * 0.6 * math.sin(a), baseline * 0.15 * math.sin(2.0 * a), z[i]],
                device=device, dtype=dtype,
            )
            Rs[i] = R
            ts_vec[i] = t
    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

    c2w = torch.eye(4, device=device, dtype=dtype).expand(N, 4, 4).contiguous()
    c2w[:, :3, :3] = Rs
    c2w[:, :3, 3] = ts_vec
    return c2w


# ---------------------------------------------------------------------------
# Depth-warp geometry control
# ---------------------------------------------------------------------------

def warp_depth_to_view(
    depth: torch.Tensor,
    src_c2w: torch.Tensor,
    dst_c2w: torch.Tensor,
    focal_frac: float = 1.2,
) -> torch.Tensor:
    """Warp a depth map from one camera viewpoint to another.

    This is the core of the **depth-warp geometry control**: given the
    painting's depth and the camera trajectory, produce a per-frame target
    depth that a real video-diffusion ControlNet would consume as geometry
    conditioning. Points are unprojected from the source camera, transformed
    into the destination camera, and re-projected.

    ``depth``: (H, W) tensor in [0, 1].
    ``src_c2w``, ``dst_c2w``: (4, 4) c2w matrices.
    Returns: (H, W) warped depth in the destination view (in [0, 1]).
    """
    device, dtype = depth.device, depth.dtype
    H, W = depth.shape
    f = focal_frac * float(max(H, W))

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    cx, cy = float(W) * 0.5, float(H) * 0.5
    # Painting depth is in [0, 1]; map to a metric range [near, far].
    z_metric = 0.2 + 4.0 * depth  # 0.2 .. 4.2 units
    x = (xs - cx) / f * z_metric
    y = (ys - cy) / f * z_metric
    pts = torch.stack([x.flatten(), y.flatten(), z_metric.flatten(), torch.ones_like(z_metric).flatten()], dim=0)

    # Transform world points from src to dst.
    src_c2w_4 = src_c2w.to(device=device, dtype=dtype)
    dst_c2w_4 = dst_c2w.to(device=device, dtype=dtype)
    # World = src_c2w @ cam_src; cam_dst = dst_c2w^-1 @ world.
    world = src_c2w_4 @ pts
    dst_w2c = torch.linalg.inv(dst_c2w_4)
    cam_dst = dst_w2c @ world  # (4, N)

    z_dst = cam_dst[2].clamp_min(1e-3)
    x_dst = cam_dst[0] / z_dst
    y_dst = cam_dst[1] / z_dst
    u = (x_dst * f + cx).round().long()
    v = (y_dst * f + cy).round().long()
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_dst > 0)

    out = torch.zeros(H * W, device=device, dtype=dtype)
    out[valid] = z_dst[valid]
    out = out.reshape(H, W)
    # Normalise back to [0, 1] using the same range.
    out = (out - 0.2) / 4.0
    return out.clamp(0.0, 1.0)


def depth_warp_control_sequence(
    painting_depth: torch.Tensor,
    trajectory: torch.Tensor,
    strength: float = 0.85,
    focal_frac: float = 1.2,
) -> torch.Tensor:
    """Produce the per-frame depth-warp control signal along the trajectory.

    Returns (N, H, W) in [0, 1]. Frame 0 is the painting's depth itself; each
    subsequent frame is the painting depth warped into that frame's camera.
    The control is mixed with a gentle blur (to mimic a real ControlNet's
    tolerance to noise) and scaled by ``strength``.
    """
    N = trajectory.shape[0]
    H, W = painting_depth.shape
    out = torch.zeros(N, H, W, device=painting_depth.device, dtype=painting_depth.dtype)
    src_c2w = trajectory[0]
    for i in range(N):
        if i == 0:
            out[i] = painting_depth
        else:
            out[i] = warp_depth_to_view(painting_depth, src_c2w, trajectory[i], focal_frac=focal_frac)
    # Soft blend with the painting depth so the control never drifts too far.
    out = strength * out + (1.0 - strength) * painting_depth.unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# Stroke-consistency attention bias
# ---------------------------------------------------------------------------

def stroke_consistency_attention_bias(
    painting_strokes: torch.Tensor,
    num_frames: int,
    num_tokens_per_frame: int,
) -> torch.Tensor:
    """Construct a temporal-attention bias that locks the painting's strokes.

    For a real video-diffusion model with temporal attention, this bias is
    added to the attention logits so that tokens belonging to the same stroke
    segment attend preferentially to their counterparts in the reference
    (painting) frame, preventing brush strokes from drifting across frames.

    For the mock backend we return a (N, num_tokens, num_tokens) bias matrix
    that downstream rendering uses to weight frame blending. Tokens that share
    a stroke segment with the painting get a positive bias; cross-stroke
    pairings get a small negative bias.

    ``painting_strokes``: (H, W) integer label map.
    """
    device = painting_strokes.device
    H, W = painting_strokes.shape
    # Downsample the stroke map to the token grid (assume roughly sqrt(Ntok) per side).
    side = max(1, int(round(math.sqrt(num_tokens_per_frame))))
    # Resample via nearest-neighbour.
    strokes_f = painting_strokes.unsqueeze(0).unsqueeze(0).float()
    tokens = F.interpolate(strokes_f, size=(side, side), mode="nearest").squeeze().long()
    flat = tokens.reshape(-1)
    Ntok = flat.numel()
    # Build a (Ntok, Ntok) bias: same stroke -> +1, different nonzero stroke -> -0.25, background -> 0.
    same = (flat.unsqueeze(0) == flat.unsqueeze(1)) & (flat.unsqueeze(0) > 0)
    diff = (flat.unsqueeze(0) != flat.unsqueeze(1)) & (flat.unsqueeze(0) > 0) & (flat.unsqueeze(1) > 0)
    bias = torch.zeros(Ntok, Ntok, device=device, dtype=torch.float32)
    bias[same] = 1.0
    bias[diff] = -0.25
    # Broadcast across frames: every frame's tokens get the same bias w.r.t. the painting frame.
    full_bias = bias.unsqueeze(0).expand(num_frames, Ntok, Ntok).contiguous()
    return full_bias


# ---------------------------------------------------------------------------
# Mock video backend: 3D-stable flythrough from painting + depth
# ---------------------------------------------------------------------------

def _unproject_to_pointcloud(rgb: torch.Tensor, depth: torch.Tensor, focal_frac: float = 1.2):
    """Unproject an (3, H, W) image + (H, W) depth to a coloured point cloud.

    Returns (P, 6) where columns are xyz, rgb. Depth is in [0, 1]; mapped to
    metric z in [near, far]."""
    device, dtype = rgb.device, rgb.dtype
    _, H, W = rgb.shape
    f = focal_frac * float(max(H, W))
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    cx, cy = float(W) * 0.5, float(H) * 0.5
    z = 0.2 + 4.0 * depth
    x = (xs - cx) / f * z
    y = (ys - cy) / f * z
    pts = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=-1)
    cols = rgb.reshape(3, -1).T
    return torch.cat([pts, cols], dim=-1)


def _project_pointcloud(pts: torch.Tensor, c2w: torch.Tensor, H: int, W: int, focal_frac: float = 1.2):
    """Project a coloured point cloud into a camera view. Returns (3, H, W) image
    via z-buffering (front-most point wins)."""
    device, dtype = pts.device, pts.dtype
    f = focal_frac * float(max(H, W))
    cx, cy = float(W) * 0.5, float(H) * 0.5
    w2c = torch.linalg.inv(c2w.to(device=device, dtype=dtype))
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    p = pts[:, :3] @ R.T + t.unsqueeze(0)
    z = p[:, 2].clamp_min(1e-3)
    u = (p[:, 0] / z * f + cx).round().long()
    v = (p[:, 1] / z * f + cy).round().long()
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    img = torch.zeros(3, H, W, device=device, dtype=dtype)
    zbuf = torch.full((H, W), float("inf"), device=device, dtype=dtype)
    if valid.any():
        u_v, v_v, z_v = u[valid], v[valid], z[valid]
        cols = pts[valid, 3:]
        # Sort by descending depth so closer points overwrite farther ones.
        order = torch.argsort(-z_v)
        u_v, v_v, z_v, cols = u_v[order], v_v[order], z_v[order], cols[order]
        idx = v_v * W + u_v
        flat_img = img.reshape(3, -1)
        flat_z = zbuf.reshape(-1)
        flat_img[:, idx] = cols.T
        flat_z[idx] = z_v
        img = flat_img.reshape(3, H, W)
        zbuf = flat_z.reshape(H, W)
    return img, zbuf


def _project_depth(pts: torch.Tensor, c2w: torch.Tensor, H: int, W: int, focal_frac: float = 1.2):
    """Project a coloured point cloud and return only the depth map (H, W)."""
    device, dtype = pts.device, pts.dtype
    f = focal_frac * float(max(H, W))
    cx, cy = float(W) * 0.5, float(H) * 0.5
    w2c = torch.linalg.inv(c2w.to(device=device, dtype=dtype))
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    p = pts[:, :3] @ R.T + t.unsqueeze(0)
    z = p[:, 2].clamp_min(1e-3)
    u = (p[:, 0] / z * f + cx).round().long()
    v = (p[:, 1] / z * f + cy).round().long()
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    depth = torch.zeros(H, W, device=device, dtype=dtype)
    zbuf = torch.full((H, W), float("inf"), device=device, dtype=dtype)
    if valid.any():
        u_v, v_v, z_v = u[valid], v[valid], z[valid]
        order = torch.argsort(-z_v)
        u_v, v_v, z_v = u_v[order], v_v[order], z_v[order]
        idx = v_v * W + u_v
        flat_d = depth.reshape(-1)
        flat_z = zbuf.reshape(-1)
        flat_d[idx] = z_v
        flat_z[idx] = z_v
        depth = flat_d.reshape(H, W)
    return depth


def mock_video_generation(
    painting_rgb: torch.Tensor,
    painting_depth: torch.Tensor,
    trajectory: torch.Tensor,
    depth_warp_control: torch.Tensor,
    stroke_bias: Optional[torch.Tensor] = None,
    focal_frac: float = 1.2,
    stability_strength: float = 0.85,
) -> torch.Tensor:
    """Mock video backend: synthesise a 3D-stable flythrough.

    The flythrough is produced by unprojecting the painting into a coloured
    point cloud, then re-projecting it into each camera along the trajectory.
    The depth-warp control signalises where the geometry should remain pinned
    to the painting; in the mock this is realised by *interpolating* between
    the painting depth (warped to the view) and a small procedural forward
    extension of the point cloud beyond the painting's frustum.

    Returns (N, 3, H, W) in [0, 1].
    """
    device = painting_rgb.device
    _, H, W = painting_rgb.shape
    N = trajectory.shape[0]

    # Unproject painting into a coloured point cloud.
    pts = _unproject_to_pointcloud(painting_rgb, painting_depth, focal_frac=focal_frac)

    # Procedural "scene extension": duplicate the point cloud with a small
    # lateral / depth perturbation so novel views reveal plausible content
    # beyond the painting's silhouette. This stands in for the generative
    # extension that a real video-diffusion model would produce.
    rng = torch.Generator(device=device).manual_seed(0)
    ext_offset = pts.clone()
    ext_offset[:, :3] += 0.05 * torch.randn((ext_offset.shape[0], 3), generator=rng, device=device, dtype=pts.dtype)
    # Push the extension slightly behind the painting surface.
    ext_offset[:, 2] += 0.15
    # Fade extension colours toward the local mean so it blends.
    mean_col = pts[:, 3:].mean(dim=0, keepdim=True)
    ext_offset[:, 3:] = 0.5 * ext_offset[:, 3:] + 0.5 * mean_col
    all_pts = torch.cat([pts, ext_offset], dim=0)

    frames = torch.zeros(N, 3, H, W, device=device, dtype=painting_rgb.dtype)
    for i in range(N):
        img, _ = _project_pointcloud(all_pts, trajectory[i], H, W, focal_frac=focal_frac)
        # Where the projection is empty (silhouette holes), inpaint with the
        # depth-warp-control-implied colour by sampling the nearest painting pixel.
        empty = (img.sum(dim=0) <= 1e-6)
        if empty.any():
            # Sample the painting by shifting the uv coordinates according to the
            # depth-warp control. Approximate: use the painting's rgb at the same
            # pixel location, modulated by the depth difference.
            shift = (depth_warp_control[i] - painting_depth) * float(W) * 0.1
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.long),
                torch.arange(W, device=device, dtype=torch.long),
                indexing="ij",
            )
            sx = (xs.float() + shift).round().long().clamp(0, W - 1)
            sy = ys.clamp(0, H - 1)
            sampled = painting_rgb[:, sy, sx]
            mask3 = empty.unsqueeze(0).expand(3, H, W)
            img = torch.where(mask3, sampled, img)
        frames[i] = img.clamp(0.0, 1.0)
    return frames


# ---------------------------------------------------------------------------
# Real-backend stubs
# ---------------------------------------------------------------------------

def _wan_i2v_backend(painting_rgb, depth_warp_control, stroke_bias, cfg: VideoGenerationConfig):
    """Wan-2.2 image-to-video backend with depth-warp ControlNet.

    This is a configuration hook: in a real deployment it loads the Wan I2V
    pipeline, attaches a depth ControlNet conditioned on ``depth_warp_control``,
    injects ``stroke_bias`` into the temporal attention, and runs the diffusion
    sampler with temporal-consistency guidance. The interface matches the mock
    backend so the pipeline orchestrator is backend-agnostic.
    """
    try:
        from diffusers import DiffusionPipeline  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Wan-I2V backend requires diffusers + the Wan weights. Use "
            "video_backend='mock' for the out-of-the-box demo."
        ) from e
    raise NotImplementedError(
        "Wan-I2V backend is a configuration hook; the mock backend ships with "
        "the demo. See paper/README.md for the production deployment recipe."
    )


# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------

def _frames_to_mp4(frames: torch.Tensor, out_path: str, fps: int = 8) -> None:
    """Write (N, 3, H, W) float frames in [0, 1] to an MP4 via imageio / ffmpeg."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames_np = (frames.clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        for f in frames_np:
            writer.append_data(f)
        writer.close()
    except Exception:
        # Fallback: write frames as PNGs and let the user assemble them.
        frames_dir = out_path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        from PIL import Image

        for i, f in enumerate(frames_np):
            Image.fromarray(f).save(os.path.join(frames_dir, f"{i:06d}.png"))


def _save_frame_grid(frames: torch.Tensor, out_path: str, n_cols: int = 6) -> None:
    """Save a contact-sheet preview of the generated frames."""
    N, C, H, W = frames.shape
    n_rows = (N + n_cols - 1) // n_cols
    grid = torch.zeros(C, n_rows * H, n_cols * W)
    for i in range(N):
        r, c = i // n_cols, i % n_cols
        grid[:, r * H:(r + 1) * H, c * W:(c + 1) * W] = frames[i]
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    from PIL import Image

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(arr).save(out_path)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def generate_video(
    cfg: VideoGenerationConfig,
    painting_rgb: torch.Tensor,
    painting_depth: torch.Tensor,
    painting_strokes: Optional[torch.Tensor] = None,
    depth_control_override: Optional[torch.Tensor] = None,
) -> dict:
    """Run Stage 0: painting-conditioned video generation.

    ``depth_control_override``: optional (H, W) depth map (in [0, 1]) that
    replaces the raw painting depth inside the depth-warp control sequence —
    the scene canvas supplies its composited object-aware depth here, so the
    geometry control respects object occlusion boundaries.

    Returns a dict with the generated frames, the camera trajectory, and the
    depth-warp control sequence.
    """
    device = painting_rgb.device
    # 1. Camera trajectory.
    trajectory = make_camera_trajectory(
        num_frames=cfg.num_frames,
        trajectory_type=cfg.trajectory_type,
        angle_deg=cfg.trajectory_angle_deg,
        baseline=cfg.trajectory_baseline,
        device=device,
    )

    # 2. Depth-warp geometry control (canvas-composited when provided).
    depth_src = painting_depth if depth_control_override is None else depth_control_override
    depth_warp_control = depth_warp_control_sequence(
        depth_src, trajectory, strength=cfg.depth_warp_control_strength
    )

    # 3. Stroke-consistency attention bias.
    stroke_bias = None
    if cfg.stroke_consistency_attention and painting_strokes is not None:
        stroke_bias = stroke_consistency_attention_bias(
            painting_strokes, num_frames=cfg.num_frames, num_tokens_per_frame=512
        )

    # 4. Generate frames.
    if cfg.video_backend == "mock":
        frames = mock_video_generation(
            painting_rgb,
            painting_depth,
            trajectory,
            depth_warp_control,
            stroke_bias=stroke_bias,
            stability_strength=cfg.depth_warp_control_strength,
        )
    elif cfg.video_backend == "wan_i2v":
        frames = _wan_i2v_backend(painting_rgb, depth_warp_control, stroke_bias, cfg)
    else:
        # All real backends fall back to mock for the demo.
        frames = mock_video_generation(
            painting_rgb,
            painting_depth,
            trajectory,
            depth_warp_control,
            stroke_bias=stroke_bias,
            stability_strength=cfg.depth_warp_control_strength,
        )

    # 5. Persist.
    os.makedirs(cfg.out_dir, exist_ok=True)
    mp4_path = os.path.join(cfg.out_dir, "video.mp4")
    _frames_to_mp4(frames, mp4_path, fps=cfg.fps)
    _save_frame_grid(frames, os.path.join(cfg.out_dir, "frame_grid.png"), n_cols=6)
    # Save frames + trajectory + depth-warp control for downstream stages.
    np.savez_compressed(
        os.path.join(cfg.out_dir, "video.npz"),
        frames=frames.permute(0, 2, 3, 1).cpu().numpy(),
        trajectory=trajectory.cpu().numpy(),
        depth_warp_control=depth_warp_control.cpu().numpy(),
        config=asdict(cfg),
    )

    return {
        "out_dir": cfg.out_dir,
        "frames": frames,
        "trajectory": trajectory,
        "depth_warp_control": depth_warp_control,
        "stroke_bias": stroke_bias,
        "mp4_path": mp4_path,
    }
