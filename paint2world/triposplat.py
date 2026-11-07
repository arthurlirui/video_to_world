"""Stage -0.5b: TripoSplat-style per-object 3D generation from the painting.

This module follows the TripoSplat architecture at demo scale. TripoSplat
generates 3D objects as Gaussians decoded from a **triplane** feature
representation, built from multi-view images and *cross-view aligned* before
decoding. Paint2World instantiates the same recipe per segmented object:

  1. **Lift** — the object's painting-view pixels are unprojected into a local
     point cloud (camera space, centred and scaled into a unit cube).
  2. **Pseudo multi-view** — because a painting is a single view, additional
     side views are synthesised by rotating the local cloud around the
     object's vertical axis, mimicking the multi-view image inputs of
     TripoSplat.
  3. **Cross-view alignment** — each side view is registered back onto the
     painting view with a small point-to-point ICP (the analogue of
     TripoSplat's cross-view Gaussian/TSDF alignment stage), and the corrected
     points are fused.
  4. **Triplane encoding** — the fused points are splatted onto three
     orthogonal feature planes (``XY`` / ``YZ`` / ``ZX``), each storing
     colour, density, and camera-space depth per cell.
  5. **Triplane decoding** — a query grid inside the unit cube samples the
     triplane bilinearly; occupied cells decode into 3D Gaussians
     (position / scale / opacity / SH colour).
  6. **Refinement** — the decoded Gaussians' scales, opacities and colours are
     optimised by rendering the object's painting-view crop with the
     differentiable rasteriser and minimising L1 to the painting.

The real TripoSplat encoder/decoder (a trained feed-forward network) can be
plugged in behind ``generate_objects`` without changing any downstream
consumer: the output is a per-object set of world-frame Gaussians plus a
rigid placement transform.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from .configs import TripoSplatConfig


# ---------------------------------------------------------------------------
# 1. Lift: object pixels -> local point cloud
# ---------------------------------------------------------------------------

def lift_object_to_camera_space(
    rgb: torch.Tensor,  # (3, H, W) in [0, 1]
    depth: torch.Tensor,  # (1, H, W) in [0, 1]
    mask: torch.Tensor,  # (H, W) bool
    focal_frac: float = 1.2,
    max_points: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unproject the masked object pixels into painting-camera space.

    Camera-space convention matches the pipeline's unprojection: x right,
    y down, z forward, with z = z_min + (z_max - z_min) * depth. Returns
    (points (N, 3), colours (N, 3)), subsampled to ``max_points``.
    """
    device = rgb.device
    _, H, W = rgb.shape
    f = focal_frac * float(max(H, W))
    cx, cy = float(W) * 0.5, float(H) * 0.5

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    z = 0.2 + 4.0 * depth[0]
    x = (xs - cx) / f * z
    y = (ys - cy) / f * z
    pts = torch.stack([x[mask], y[mask], z[mask]], dim=1)
    cols = rgb[:, mask].T
    if pts.shape[0] > max_points:
        keep = torch.randperm(pts.shape[0], device=device)[:max_points]
        pts, cols = pts[keep], cols[keep]
    return pts, cols


def normalize_object(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Centre and scale a point cloud into the unit cube.

    Returns (local_points in [-1, 1]^3, centroid (3,), half_extent scalar).
    """
    centroid = points.mean(dim=0)
    local = points - centroid
    half = float(local.abs().max().item())
    half = max(half, 1e-6)
    local = local / half
    return local, centroid, torch.tensor(half, device=points.device)


# ---------------------------------------------------------------------------
# 2. Pseudo multi-view: rotate around the object's vertical axis
# ---------------------------------------------------------------------------

def _rot_y(points: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rotate points around the y axis (camera 'down' axis; the object's
    vertical axis in the local frame)."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        device=points.device, dtype=points.dtype,
    )
    return points @ R.T


# ---------------------------------------------------------------------------
# 3. Cross-view alignment: point-to-point ICP (TripoSplat's alignment stage)
# ---------------------------------------------------------------------------

def icp_align(
    src: torch.Tensor,  # (M, 3) points to move
    dst: torch.Tensor,  # (N, 3) reference points
    num_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigid ICP (rotation + translation, no scale) aligning src onto dst.

    Returns (aligned_src, transform (4, 4)). Point-to-point correspondences
    via nearest neighbours; a small closed-form Procrustes update per
    iteration. This is the analogue of TripoSplat's cross-view Gaussian
    alignment, simplified to a point setting.
    """
    device = src.device
    T = torch.eye(4, device=device, dtype=src.dtype)
    cur = src.clone()
    for _ in range(num_iters):
        d = torch.cdist(cur, dst)
        nn = d.argmin(dim=1)
        p = cur
        q = dst[nn]
        mu_p = p.mean(dim=0)
        mu_q = q.mean(dim=0)
        H = (p - mu_p).T @ (q - mu_q)
        U, _, Vt = torch.linalg.svd(H)
        S = torch.eye(3, device=device, dtype=src.dtype)
        S[2, 2] = float(torch.det(Vt.T @ U.T).clamp(-1.0, 1.0).item())
        R = Vt.T @ S @ U.T
        t = mu_q - R @ mu_p
        cur = p @ R.T + t
        step = torch.eye(4, device=device, dtype=src.dtype)
        step[:3, :3] = R
        step[:3, 3] = t
        T = step @ T
    return cur, T


# ---------------------------------------------------------------------------
# 4. Triplane encoding
# ---------------------------------------------------------------------------

TRIPLANE_FEATURE_DIM = 8  # r, g, b, density, depth, r_var, g_var, b_var


class Triplane:
    """Three orthogonal feature planes (XY, YZ, ZX) in the unit cube.

    Each plane stores ``TRIPLANE_FEATURE_DIM`` channels per cell:
    (r, g, b, density, mean camera-depth, and per-channel colour variance
    proxies). Built by splatting a fused multi-view point cloud.
    """

    AXES = ("xy", "yz", "zx")

    def __init__(self, resolution: int, device=None, dtype=torch.float32):
        self.resolution = resolution
        device = device or torch.device("cpu")
        # features: accumulated (sum_r, sum_g, sum_b, count, sum_z, sum_r2, sum_g2, sum_b2)
        self._acc = {
            ax: torch.zeros(8, resolution, resolution, device=device, dtype=dtype)
            for ax in self.AXES
        }

    def _plane_coords(self, axis: str, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map (N, 3) unit-cube points to a plane's (row, col) float coords.

        Row/col conventions keep a right-handed layout: rows index the second
        axis of the pair, columns the first, both in [-1, 1] mapped to
        [0, R-1].
        """
        if axis == "xy":
            a, b = pts[:, 0], pts[:, 1]
        elif axis == "yz":
            a, b = pts[:, 1], pts[:, 2]
        else:  # zx
            a, b = pts[:, 2], pts[:, 0]
        r = self.resolution
        col = (a + 1.0) * 0.5 * (r - 1)
        row = (b + 1.0) * 0.5 * (r - 1)
        return row, col

    def splat(self, points: torch.Tensor, colours: torch.Tensor, depths: torch.Tensor) -> None:
        """Accumulate points (N, 3) with colours (N, 3) and camera depths (N,)
        into all three planes (cell-average splatting)."""
        feats = torch.cat(
            [
                colours,
                torch.ones_like(colours[:, :1]),  # count
                depths.unsqueeze(1),
                colours ** 2,
            ],
            dim=1,
        )  # (N, 8): r, g, b, 1, z, r^2, g^2, b^2
        for ax in self.AXES:
            row, col = self._plane_coords(ax, points)
            ri = row.round().long().clamp(0, self.resolution - 1)
            ci = col.round().long().clamp(0, self.resolution - 1)
            flat = (ri * self.resolution + ci)
            self._acc[ax].reshape(8, -1).index_add_(
                1, flat, feats.T.to(self._acc[ax].dtype)
            )

    def features(self) -> dict[str, torch.Tensor]:
        """Return per-axis (C=8, R, R) feature planes with unit semantics."""
        out = {}
        for ax in self.AXES:
            acc = self._acc[ax]
            cnt = acc[3:4].clamp_min(1.0)
            rgb = acc[0:3] / cnt
            density = acc[3:4]
            z = acc[4:5] / cnt
            var = (acc[5:8] / cnt - rgb**2).clamp_min(0.0)
            out[ax] = torch.cat([rgb, density / density.max().clamp_min(1e-8), z, var], dim=0)
        return out


def encode_triplane(
    local_pts: torch.Tensor,
    local_cols: torch.Tensor,
    cam_depths: torch.Tensor,
    resolution: int,
) -> Triplane:
    tri = Triplane(resolution, device=local_pts.device, dtype=local_pts.dtype)
    tri.splat(local_pts, local_cols, cam_depths)
    return tri


# ---------------------------------------------------------------------------
# 5. Triplane decoding -> 3D Gaussians
# ---------------------------------------------------------------------------

def _bilinear_sample(plane: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    """Bilinear sample a (C, R, R) plane at float (row, col) coordinates.

    Returns (N, C). Coordinates are clamped inside the plane.
    """
    R = plane.shape[-1]
    r = row.clamp(0.0, R - 1.001)
    c = col.clamp(0.0, R - 1.001)
    r0 = r.floor().long()
    c0 = c.floor().long()
    r1 = (r0 + 1).clamp(max=R - 1)
    c1 = (c0 + 1).clamp(max=R - 1)
    fr = (r - r0.float()).view(-1, 1)
    fc = (c - c0.float()).view(-1, 1)
    p00 = plane[:, r0, c0].T
    p01 = plane[:, r0, c1].T
    p10 = plane[:, r1, c0].T
    p11 = plane[:, r1, c1].T
    return (p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc
            + p10 * fr * (1 - fc) + p11 * fr * fc)


def decode_gaussians_from_triplane(
    tri: Triplane,
    query_resolution: int,
    occupancy_threshold: float = 0.02,
    min_gaussians: int = 32,
    jitter: float = 0.02,
    seed: int = 0,
) -> dict:
    """Decode triplane features into a set of object-local Gaussians.

    A query grid of ``query_resolution^3`` cells covers the unit cube; each
    cell fuses bilinear features from the three planes (density-weighted) and
    emits one Gaussian when its fused occupancy exceeds the threshold. The
    emitted scales derive from the local point density (via nearest observed
    cell distance), colours from the fused RGB.
    """
    device = next(iter(tri._acc.values())).device
    dtype = next(iter(tri._acc.values())).dtype
    feats = tri.features()
    R = tri.resolution
    Q = query_resolution

    qs = torch.linspace(-1.0, 1.0, Q, device=device, dtype=dtype)
    gz, gy, gx = torch.meshgrid(qs, qs, qs, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)  # (Q^3, 3)

    # Fused per-query features: density-weighted mean over the three planes.
    per_plane = []
    for ax in Triplane.AXES:
        row, col = tri._plane_coords(ax, grid)
        per_plane.append(_bilinear_sample(feats[ax], row, col))
    stacked = torch.stack(per_plane, dim=0)  # (3, Q^3, 8)
    dens = stacked[:, :, 3:4].clamp_min(0.0)  # (3, Q^3, 1)
    w = dens + 1e-6
    fused = (stacked * w).sum(dim=0) / w.sum(dim=0)  # (Q^3, 8)

    occ = fused[:, 3]
    keep = occ > occupancy_threshold
    if int(keep.sum().item()) < min_gaussians:
        # Keep the top-occupancy cells so every object remains renderable.
        k = min(min_gaussians, occ.numel())
        keep = torch.zeros_like(occ, dtype=torch.bool)
        keep[occ.topk(k).indices] = True

    pos = grid[keep]
    feat = fused[keep]
    n = pos.shape[0]
    if n == 0:
        empty = torch.zeros(0, 3, device=device, dtype=dtype)
        return {
            "position": empty, "sh_dc": empty.clone(),
            "opacity": torch.zeros(0, device=device, dtype=dtype),
            "scale": empty.clone(), "rotation": torch.zeros(0, 4, device=device, dtype=dtype),
        }

    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = pos + (torch.rand(pos.shape, generator=g).to(device) - 0.5) * jitter

    # Scale from cell size, modulated by occupancy (denser -> tighter).
    cell = 2.0 / Q
    scale = cell * (0.9 - 0.4 * (feat[:, 3] / feat[:, 3].max().clamp_min(1e-8)))
    scale = scale.clamp_min(cell * 0.3).unsqueeze(1).expand(-1, 3)

    # Opacity: logit of (a soft-squashed occupancy), biased to visible.
    op_raw = (feat[:, 3] / feat[:, 3].max().clamp_min(1e-8)).clamp(0.05, 1.0)
    opacity = torch.logit(op_raw * 0.85 + 0.05)

    return {
        "position": pos,
        "sh_dc": feat[:, 0:3].clamp(0.0, 1.0),
        "opacity": opacity,
        "scale": scale,
        "rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        .unsqueeze(0).expand(n, -1).clone(),
    }


# ---------------------------------------------------------------------------
# 6. Refinement: render the object's painting-view crop and optimise
# ---------------------------------------------------------------------------

def _object_crop_camera(
    world_points: torch.Tensor,
    focal_frac: float,
    base_img_size: int,
):
    """Build a tight virtual camera for rendering one object.

    The virtual camera keeps the painting's viewing direction (identity
    rotation in camera space) but translates so the object's centre lands on
    the optical axis, with an image size covering the object's screen extent.
    Returns (c2w (4, 4), H, W, focal).
    """
    f = focal_frac * float(base_img_size)
    center = world_points.mean(dim=0)
    z = world_points[:, 2].clamp_min(0.05)  # true camera depth (z axis untouched)
    hx = float(((world_points[:, 0] - center[0]).abs() / z * f).max().item())
    hy = float(((world_points[:, 1] - center[1]).abs() / z * f).max().item())
    max_half = 1.5 * base_img_size
    Wc = int(max(24, min(math.ceil(2.4 * hx), int(2 * max_half))))
    Hc = int(max(24, min(math.ceil(2.4 * hy), int(2 * max_half))))
    c2w = torch.eye(4, dtype=world_points.dtype)
    c2w[:3, 3] = torch.tensor([center[0].item(), center[1].item(), 0.0], dtype=world_points.dtype)
    return c2w, Hc, Wc, f


def refine_object(
    gaussians: dict,
    target_crop: torch.Tensor,  # (3, Hc, Wc) in [0, 1]
    c2w: torch.Tensor,
    f: float,
    num_iters: int,
    lr: float = 2e-2,
) -> list[float]:
    """Optimise scale / opacity / SH colour against the object's crop view.

    Positions stay fixed (they encode the triplane-decoded structure); the
    photometric fit adjusts how the structure renders, which is where the
    triplane decode is most under-constrained. Returns the loss history.
    """
    Hc, Wc = target_crop.shape[-2:]
    scale = gaussians["scale"].clone().requires_grad_(True)
    opacity = gaussians["opacity"].clone().requires_grad_(True)
    sh = gaussians["sh_dc"].clone().requires_grad_(True)
    opt = torch.optim.Adam([scale, opacity, sh], lr=lr)
    history = []
    for _ in range(num_iters):
        opt.zero_grad()
        g = dict(gaussians)
        g["scale"], g["opacity"], g["sh_dc"] = scale, opacity, sh
        from .stroke_coherent_gs import rasterise_gaussians

        rendered = rasterise_gaussians(g, c2w, Hc, Wc, focal_frac=f / max(Hc, Wc))
        # Background of the crop: mean colour of the target (flat prior).
        bg = target_crop.mean(dim=(1, 2)).view(3, 1, 1).expand(3, Hc, Wc)
        alpha_target = (target_crop - bg).abs().mean(dim=0, keepdim=True)
        # Weight the L1 more where the object actually deviates from its mean
        # background (soft mask from colour contrast).
        w = (alpha_target / alpha_target.max().clamp_min(1e-6)).clamp(0.1, 1.0)
        loss = ((rendered - target_crop).abs() * w).mean()
        loss = loss + 1e-3 * torch.sigmoid(opacity).mean()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    gaussians["scale"] = scale.detach()
    gaussians["opacity"] = opacity.detach()
    gaussians["sh_dc"] = sh.detach().clamp(0.0, 1.0)
    return history


# ---------------------------------------------------------------------------
# Top-level: per-object generation and batch driver
# ---------------------------------------------------------------------------

def generate_object(
    cfg: TripoSplatConfig,
    obj_meta: dict,
    rgb: torch.Tensor,  # (3, H, W)
    depth: torch.Tensor,  # (1, H, W)
    seed: int = 0,
) -> dict:
    """Generate one object's Gaussians in *camera space* (painting view).

    Returns a dict with the object id, world(painting-camera)-frame Gaussians,
    the local placement transform (centroid, half-extent), the triplane
    feature summary, and refinement diagnostics.
    """
    mask = torch.from_numpy(obj_meta["mask"]).to(rgb.device)
    pts, cols = lift_object_to_camera_space(rgb, depth, mask, max_points=4096)
    if pts.shape[0] < 8:
        # Degenerate object: emit a tiny flat patch so downstream stages stay
        # well-defined.
        pts = torch.zeros(8, 3, device=rgb.device)
        cols = torch.full((8, 3), 0.5, device=rgb.device)
    local, centroid, half = normalize_object(pts)
    cam_depths = pts[:, 2]

    # --- Pseudo multi-view + cross-view alignment ---
    fused_pts = [local]
    fused_cols = [cols]
    fused_depths = [cam_depths]
    for v in range(cfg.num_side_views):
        ang = math.radians(cfg.side_view_angle_deg) * (v + 1) * (-1.0 if v % 2 else 1.0)
        side_pts = _rot_y(local, ang)
        aligned, _T = icp_align(side_pts, local, num_iters=15)
        fused_pts.append(aligned)
        fused_cols.append(cols)
        fused_depths.append(cam_depths)
    all_pts = torch.cat(fused_pts, dim=0)
    all_cols = torch.cat(fused_cols, dim=0)
    all_depths = torch.cat(fused_depths, dim=0)

    # --- Triplane encode + decode ---
    tri = encode_triplane(all_pts, all_cols, all_depths, cfg.triplane_resolution)
    g_local = decode_gaussians_from_triplane(
        tri,
        query_resolution=cfg.query_resolution,
        min_gaussians=cfg.min_gaussians_per_object,
        seed=seed,
    )

    # Local -> camera-space transform.
    half_f = float(half.item())
    g_world = {
        "position": g_local["position"] * half_f + centroid.unsqueeze(0),
        "sh_dc": g_local["sh_dc"],
        "opacity": g_local["opacity"],
        "scale": g_local["scale"] * half_f,
        "rotation": g_local["rotation"],
    }

    # --- Painting-view crop refinement ---
    history = []
    if cfg.refinement_iters > 0 and g_world["position"].shape[0] > 0:
        c2w, Hc, Wc, f = _object_crop_camera(
            g_world["position"], focal_frac=1.2, base_img_size=max(rgb.shape[-2:])
        )
        # Target crop: render the object's actual pixels through the same
        # virtual camera. Build it by projecting the lifted points.
        target = _render_point_crop(pts, cols, c2w, Hc, Wc, f)
        history = refine_object(g_world, target, c2w, f, num_iters=cfg.refinement_iters)

    plane_stats = {
        ax: {
            "density_max": float(tri.features()[ax][3].max().item()),
            "occupied_cells": int((tri.features()[ax][3] > 0.01).sum().item()),
        }
        for ax in Triplane.AXES
    }
    return {
        "id": obj_meta["id"],
        "gaussians": g_world,
        "centroid": centroid.detach(),
        "half_extent": half_f,
        "num_gaussians": int(g_world["position"].shape[0]),
        "triplane_stats": plane_stats,
        "refinement_loss_initial": history[0] if history else None,
        "refinement_loss_final": history[-1] if history else None,
        "mean_depth": obj_meta["mean_depth"],
        "bbox": obj_meta["bbox"],
        "area": obj_meta["area"],
    }


def _render_point_crop(
    pts: torch.Tensor,
    cols: torch.Tensor,
    c2w: torch.Tensor,
    Hc: int,
    Wc: int,
    f: float,
) -> torch.Tensor:
    """Render a point cloud into its crop view (z-buffered nearest-point
    splat) — the refinement target for one object."""
    device = pts.device
    w2c = torch.linalg.inv(c2w.to(device))
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    p = pts @ R.T + t.unsqueeze(0)
    z = p[:, 2].clamp_min(1e-3)
    u = p[:, 0] / z * f + Wc * 0.5
    v = p[:, 1] / z * f + Hc * 0.5
    img = torch.zeros(3, Hc, Wc, device=device, dtype=pts.dtype)
    zbuf = torch.full((Hc, Wc), float("inf"), device=device, dtype=pts.dtype)
    order = torch.argsort(-z)  # far to near so near overwrites
    for i in order.tolist():
        ui, vi = int(round(u[i].item())), int(round(v[i].item()))
        if 0 <= ui < Wc and 0 <= vi < Hc and z[i].item() < zbuf[vi, ui].item():
            img[:, vi, ui] = cols[i]
            zbuf[vi, ui] = z[i]
    # Dilate 1px so isolated points form visible blobs.
    img = torch.nn.functional.max_pool2d(
        (img > 0).float() * img.unsqueeze(0), 3, stride=1, padding=1
    ).squeeze(0) + torch.nn.functional.avg_pool2d(
        img.unsqueeze(0), 3, stride=1, padding=1
    ).squeeze(0) * 0.5
    return (img / img.max().clamp_min(1e-6) * 0.5 + img * 0.5).clamp(0.0, 1.0)


def generate_objects(
    cfg: TripoSplatConfig,
    image: np.ndarray,  # (H, W, 3) uint8
    depth: np.ndarray,  # (H, W) in [0, 1]
    seg: dict,  # output of object_segmentation.segment_objects
    out_dir: Optional[str] = None,
) -> dict:
    """Stage -0.5b driver: generate every segmented object's 3D Gaussians.

    Returns a dict with the per-object entries and a combined all-object
    Gaussian bundle. Persists ``triposplat.npz`` when ``out_dir`` is given.
    """
    rgb = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
    dep = torch.from_numpy(depth.copy()).float().unsqueeze(0)

    entries = []
    for k, obj_meta in enumerate(seg["objects"]):
        entry = generate_object(cfg, obj_meta, rgb, dep, seed=1000 + k)
        entries.append(entry)

    # Combined bundle (all objects, camera space).
    def _cat(key):
        return torch.cat([e["gaussians"][key] for e in entries if e["num_gaussians"] > 0], dim=0)

    combined = {
        "position": _cat("position"),
        "sh_dc": _cat("sh_dc"),
        "opacity": _cat("opacity"),
        "scale": _cat("scale"),
        "rotation": _cat("rotation"),
    }
    obj_ids = torch.cat(
        [torch.full((e["num_gaussians"],), e["id"], dtype=torch.long) for e in entries]
    )

    result = {
        "objects": entries,
        "combined": combined,
        "object_label": obj_ids,
        "num_objects": len(entries),
        "config": asdict(cfg),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "triposplat.npz"),
            position=combined["position"].cpu().numpy(),
            sh_dc=combined["sh_dc"].cpu().numpy(),
            opacity=combined["opacity"].cpu().numpy(),
            scale=combined["scale"].cpu().numpy(),
            rotation=combined["rotation"].cpu().numpy(),
            object_label=obj_ids.cpu().numpy(),
            config=asdict(cfg),
        )
    return result
