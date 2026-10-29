"""Stage -1: Oil-painting analysis.

Given an input oil painting, estimate:
  * a monocular depth map (synthetic perceptual backend by default),
  * a surface-normal map (from the depth gradient),
  * a brush-stroke segmentation map (oriented-gradient / superpixel grouping),
  * a confidence map (high in textured stroke regions, low in flat sky / ground).

The synthetic backends are designed to be *perceptually plausible* for oil
paintings and require no network weights, so the demo runs end-to-end in a
plain CPU / PyTorch environment. They expose the same interface as the real
backends (DepthAnything-3, SAM2, etc.), so swapping in a real model is a
one-line configuration change.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .configs import PaintingAnalysisConfig


# ---------------------------------------------------------------------------
# Image I/O helpers (PIL-based, no OpenCV hard dependency)
# ---------------------------------------------------------------------------

def _load_image(path: str, max_side: int = 768) -> np.ndarray:
    """Load an image as an (H, W, 3) uint8 RGB array, resized so the longest
    side is ``max_side`` (preserving aspect ratio)."""
    from PIL import Image  # lazy import

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = float(max_side) / float(max(w, h))
    if scale < 1.0:
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _to_tensor(img_uint8: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
    # Copy to avoid PyTorch's non-writable-array warning when the source is
    # a read-only PIL buffer.
    t = torch.from_numpy(img_uint8.copy()).float().permute(2, 0, 1) / 255.0
    if device is not None:
        t = t.to(device)
    return t


def _save_rgb(arr_uint8: np.ndarray, path: str) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_uint8).save(path)


def _save_gray(arr_float: np.ndarray, path: str, cmap: str = "inferno") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm

    a = np.clip(arr_float, 0.0, 1.0)
    rgba = cm.get_cmap(cmap)(a)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    _save_rgb(rgb, path)


# ---------------------------------------------------------------------------
# Synthetic depth estimation for oil paintings
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(radius: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _smooth(img: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Separable Gaussian smoothing on (C, H, W) tensor."""
    radius = max(1, int(round(3.0 * sigma)))
    k = _gaussian_kernel_1d(radius, sigma, img.device, img.dtype)
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    c = img.shape[0]
    out = F.conv2d(img.unsqueeze(0), kx.expand(c, 1, 1, -1), padding=(0, radius), groups=c)
    out = F.conv2d(out, ky.expand(c, 1, -1, 1), padding=(radius, 0), groups=c)
    return out.squeeze(0)


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.601 luminance on (C=3, H, W) -> (1, H, W)."""
    r, g, b = rgb[0:1], rgb[1:2], rgb[2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _relative_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Perceptual relative luminance (sRGB -> linear -> luma)."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[0:1] + 0.7152 * lin[1:2] + 0.0722 * lin[2:3]


def _saturation(rgb: torch.Tensor) -> torch.Tensor:
    mx = rgb.max(dim=0, keepdim=True).values
    mn = rgb.min(dim=0, keepdim=True).values
    denom = mx + 1e-6
    return 1.0 - mn / denom


def _gradient_magnitude(img: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude on a single-channel (1, H, W) tensor."""
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=img.device, dtype=img.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(img.unsqueeze(0), sobel_x, padding=1)
    gy = F.conv2d(img.unsqueeze(0), sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze(0)


def _vertical_position_prior(h: int, w: int, device, dtype) -> torch.Tensor:
    """Atmospheric-perspective / horizon prior: upper region tends to be
    farther (sky / distance), lower region nearer (ground). The horizon is
    placed at ~0.62 of the image height by default, with a soft ramp."""
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    horizon = 0.62
    # Above the horizon: depth grows toward the top (sky). Below: depth grows
    # toward the horizon (ground recedes). Both encoded as "near = 0, far = 1".
    above = torch.clamp((horizon - ys) / horizon, min=0.0)
    below = torch.clamp((ys - horizon) / max(1.0 - horizon, 1e-6), min=0.0)
    # Combined: distance from horizon, normalised.
    dist_from_horizon = torch.where(ys <= horizon, above, below)
    # Near the horizon is "far"; near top/bottom edges can be either. We treat
    # the horizon band as the far field, and add a gentle falloff toward the
    # bottom (foreground ground plane).
    depth_prior = 1.0 - torch.exp(-3.5 * dist_from_horizon)
    # Bottom-of-frame foreground bias: pull the bottom 15% nearer.
    bottom_mask = torch.clamp((ys - 0.85) / 0.15, min=0.0, max=1.0)
    depth_prior = depth_prior * (1.0 - 0.6 * bottom_mask)
    return depth_prior.unsqueeze(0)


def synthetic_painting_depth(rgb: torch.Tensor) -> torch.Tensor:
    """Perceptually plausible monocular depth for oil paintings.

    Combines four cues:
      1. **Atmospheric perspective**: distant regions tend to be desaturated
         and lower-contrast (lighter, hazier).
      2. **Vertical position**: a soft horizon prior.
      3. **Local contrast / texture**: high-frequency brush texture is more
         common in the foreground; smooth areas recede.
      4. **Relative luminance**: a gentle prior that darker regions are nearer
         (chiaroscuro convention) — used only as a weak tie-breaker.

    The result is a depth map in [0, 1] where 0 = nearest, 1 = farthest. This
    is *not* a metric depth; it is a layout prior that the downstream
    painting-anchored alignment consumes as a canonical-frame geometry anchor.
    """
    device, dtype = rgb.device, rgb.dtype
    c, h, w = rgb.shape
    assert c == 3

    smooth_rgb = _smooth(rgb, sigma=1.5)
    lum = _relative_luminance(smooth_rgb)
    sat = _saturation(smooth_rgb)
    grad = _gradient_magnitude(_smooth(lum, sigma=0.8))
    vert = _vertical_position_prior(h, w, device, dtype)

    # Cue 1: atmospheric perspective — high saturation + high contrast = near.
    # Normalise each cue to [0, 1] (near = 0, far = 1) then combine.
    def to_unit(t: torch.Tensor) -> torch.Tensor:
        lo = float(t.min().item())
        hi = float(t.max().item())
        if hi - lo < 1e-6:
            return torch.full_like(t, 0.5)
        return (t - lo) / (hi - lo)

    # Saturation: low sat -> far (haze), high sat -> near.
    sat_far = 1.0 - to_unit(sat)
    # Contrast / texture: low texture -> far (smooth distance), high -> near.
    grad_far = 1.0 - to_unit(grad)
    # Luminance: brighter -> farther (haze), darker -> nearer (chiaroscuro).
    lum_far = to_unit(lum)
    # Vertical: the horizon prior (already in [0,1], 1=far).
    vert_far = vert

    # Weighted combination. The vertical prior dominates the global layout;
    # the local cues refine within that layout.
    depth = (
        0.45 * vert_far
        + 0.22 * sat_far
        + 0.23 * grad_far
        + 0.10 * lum_far
    )
    # Light smoothing so the depth map is a plausible surface, not per-pixel noise.
    depth = _smooth(depth, sigma=2.0)
    depth = to_unit(depth)
    return depth  # (1, H, W) in [0, 1]


# ---------------------------------------------------------------------------
# Surface normals from depth
# ---------------------------------------------------------------------------

def depth_to_normals(depth: torch.Tensor, focal_frac: float = 1.2) -> torch.Tensor:
    """Compute per-pixel surface normals from a depth map.

    ``depth`` is (1, H, W) in [0, 1]. We treat the depth as a z-buffer with a
    synthetic focal length ``focal_frac * max(H, W)`` and unproject neighbours
    to 3D to estimate normals via cross products. Returns (3, H, W) normals
    in [-1, 1] (world space, +z = toward camera).
    """
    assert depth.dim() == 3 and depth.shape[0] == 1
    _, h, w = depth.shape
    device, dtype = depth.device, depth.dtype
    f = focal_frac * float(max(h, w))

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    cx, cy = float(w) * 0.5, float(h) * 0.5
    x = (xs - cx) / f
    y = (ys - cy) / f
    z = depth.squeeze(0)

    # Backward differences.
    dx = torch.zeros_like(z)
    dy = torch.zeros_like(z)
    dx[:, 1:] = z[:, 1:] - z[:, :-1]
    dy[1:, :] = z[1:, :] - z[:-1, :]

    # 3D points: P = (x*z, y*z, z). Tangent vectors along x and y.
    # dP/dx = (x*dx + z, 0, dx); dP/dy = (0, y*dy + z, dy).
    px_x = x * dx + z
    px_z = dx
    py_y = y * dy + z
    py_z = dy

    # Normal = cross(dP/dy, dP/dx) so that it points toward the camera (+z).
    nx = py_y * px_z - py_z * 0.0  # cross((0, py_y, py_z), (px_x, 0, px_z))
    ny = py_z * px_x - 0.0 * px_z
    nz = 0.0 * 0.0 - py_y * px_x
    normals = torch.stack([nx, ny, nz], dim=0)
    norms = torch.linalg.norm(normals, dim=0, keepdim=True).clamp_min(1e-8)
    normals = normals / norms
    # Ensure +z (toward camera).
    flip = (normals[2:3] < 0).float() * 2.0 - 1.0
    normals = normals * flip
    return normals  # (3, H, W)


# ---------------------------------------------------------------------------
# Brush-stroke segmentation (synthetic: oriented-gradient superpixel grouping)
# ---------------------------------------------------------------------------

def _lab_from_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Approximate LAB colour space from sRGB (C, H, W). Only the perceptual
    brightness and colour-opponent channels are needed for segmentation."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[0:1], lin[1:2], lin[2:3]
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    xn, yn, zn = 0.95047, 1.0, 1.08883
    def f(t):
        return torch.where(t > 0.008856, t.clamp_min(1e-9) ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)
    fx = f(x / xn)
    fy = f(y / yn)
    fz = f(z / zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.cat([L, a, bb], dim=0)


def synthetic_stroke_segmentation(
    rgb: torch.Tensor,
    min_area: int = 64,
    merge_threshold: float = 0.15,
) -> torch.Tensor:
    """Brush-stroke segmentation via oriented-gradient superpixel grouping.

    Returns an integer label map (H, W) of dtype long, where each label
    identifies one stroke segment. Background / untextured regions get label 0.
    """
    device = rgb.device
    c, h, w = rgb.shape
    lab = _lab_from_rgb(rgb)
    lab = _smooth(lab, sigma=1.0)

    # Texture energy: gradient magnitude on the L channel.
    grad_l = _gradient_magnitude(lab[0:1])
    grad_a = _gradient_magnitude(lab[1:2])
    grad_b = _gradient_magnitude(lab[2:3])
    grad = (grad_l + grad_a + grad_b) * 0.5

    # Quantise into a grid of seeds, then grow seeds outward while colour
    # distance is below threshold — a simple SLIC-like superpixel.
    step = max(8, int(round(math.sqrt(max(h * w, 1) / max(400, (h * w) // 4000)))))
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    # Seed grid.
    sy = torch.arange(step // 2, h, step, device=device)
    sx = torch.arange(step // 2, w, step, device=device)
    seeds_y, seeds_x = torch.meshgrid(sy, sx, indexing="ij")
    seeds_y = seeds_y.flatten()
    seeds_x = seeds_x.flatten()
    num_seeds = seeds_y.numel()
    if num_seeds == 0:
        return torch.zeros((h, w), dtype=torch.long, device=device)

    seed_lab = lab[:, seeds_y, seeds_x].T  # (num_seeds, 3)
    pix_lab = lab.permute(1, 2, 0).reshape(h * w, 3)
    pix_xy = torch.stack([ys.flatten(), xs.flatten()], dim=1).float()
    step_f = float(step)

    # SLIC-style assignment: each pixel -> nearest seed in combined
    # colour+spatial distance. Process seeds in chunks to bound memory.
    best_dist = torch.full((h * w,), float("inf"), device=device)
    best_seed = torch.zeros((h * w,), dtype=torch.long, device=device)
    chunk_seeds = max(1, 4096 // max(1, h * w // 4096))
    for s in range(0, num_seeds, chunk_seeds):
        e = min(num_seeds, s + chunk_seeds)
        sl = seed_lab[s:e]  # (m, 3)
        sxy = torch.stack([seeds_y[s:e], seeds_x[s:e]], dim=1).float()  # (m, 2)
        d_col = torch.cdist(pix_lab, sl)  # (HW, m)
        d_spa = torch.cdist(pix_xy, sxy) / step_f  # (HW, m)
        d = d_col + 2.0 * d_spa
        sub_best, sub_idx = d.min(dim=1)  # (HW,)
        improved = sub_best < best_dist
        best_dist = torch.where(improved, sub_best, best_dist)
        best_seed = torch.where(improved, sub_idx + s, best_seed)
    label_map = (best_seed + 1).reshape(h, w)

    # Mask out untextured regions (low gradient) -> label 0. Use a low
    # threshold so most of the painting is segmented into strokes; only
    # truly flat regions (sky, smooth ground) are dropped.
    grad_mean = float(grad.mean().item())
    grad_max = float(grad.max().item())
    tex_threshold = max(1e-3, 0.15 * grad_mean + 0.02 * grad_max)
    tex_mask = grad.squeeze(0) > tex_threshold
    label_map = torch.where(tex_mask, label_map, torch.zeros_like(label_map))

    # Remove tiny segments (assign to 0).
    label_map = _drop_small_segments(label_map, min_area=min_area)

    # Merge adjacent segments with similar mean colour.
    label_map = _merge_similar_segments(label_map, lab, merge_threshold=merge_threshold)

    return label_map.long()


def _drop_small_segments(label_map: torch.Tensor, min_area: int) -> torch.Tensor:
    labels = label_map.unique()
    out = torch.zeros_like(label_map)
    for lab in labels.tolist():
        if lab == 0:
            continue
        m = label_map == lab
        if int(m.sum().item()) >= min_area:
            out[m] = lab
    return out


def _merge_similar_segments(label_map: torch.Tensor, lab: torch.Tensor, merge_threshold: float) -> torch.Tensor:
    """Iteratively merge segments whose mean LAB colour distance is below threshold."""
    labels = [int(l) for l in label_map.unique().tolist() if l != 0]
    if len(labels) < 2:
        return label_map
    means = {}
    for l in labels:
        m = label_map == l
        if m.any():
            means[l] = lab[:, m].mean(dim=1)
    # Union-find style merge.
    parent = {l: l for l in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            d = (means[li] - means[lj]).norm().item()
            if d < merge_threshold * 100.0:  # LAB distances are ~0-100
                ri, rj = find(li), find(lj)
                if ri != rj:
                    parent[ri] = rj
    remap = {l: find(l) for l in labels}
    # Compact labels.
    roots = sorted(set(remap.values()))
    root_to_new = {r: i + 1 for i, r in enumerate(roots)}
    out = torch.zeros_like(label_map)
    for l in labels:
        out[label_map == l] = root_to_new[remap[l]]
    return out


# ---------------------------------------------------------------------------
# Confidence map
# ---------------------------------------------------------------------------

def painting_confidence(rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Per-pixel confidence in the (synthetic) depth estimate.

    High in textured stroke regions, low in flat sky / ground. Used as a
    soft weight by downstream painting-anchored alignment.
    """
    smooth = _smooth(rgb, sigma=1.0)
    grad = _gradient_magnitude(_relative_luminance(smooth))
    sat = _saturation(smooth)
    conf = (to_unit_local(grad) + to_unit_local(sat)) * 0.5
    conf = _smooth(conf, sigma=1.5)
    return to_unit_local(conf)


def to_unit_local(t: torch.Tensor) -> torch.Tensor:
    lo = float(t.min().item())
    hi = float(t.max().item())
    if hi - lo < 1e-6:
        return torch.full_like(t, 0.5)
    return (t - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Real-backend stubs (delegate to external models when available)
# ---------------------------------------------------------------------------

def _depth_anything_v3(rgb: torch.Tensor, model_name: str) -> torch.Tensor:
    """DepthAnything-3 backend. Imported lazily; requires the DA3 package."""
    try:
        from depth_anything_v3 import DepthAnythingV3  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "DepthAnything-3 is not installed. Use depth_backend='synthetic' for "
            "the out-of-the-box demo, or follow the video_to_world README to "
            "install DA3."
        ) from e
    model = DepthAnythingV3.from_pretrained(model_name)
    # DA3 returns metric depth; we normalise to [0, 1] (far = 1).
    depth = model.infer_image(rgb.permute(1, 2, 0).cpu().numpy())  # placeholder API
    depth = torch.from_numpy(depth).float().to(rgb.device).unsqueeze(0)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth


def _sam2_strokes(rgb: torch.Tensor) -> torch.Tensor:
    """SAM2-based stroke segmentation. Imported lazily."""
    try:
        from segment_anything_2 import sam2  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "SAM2 is not installed. Use stroke_backend='synthetic' for the "
            "out-of-the-box demo."
        ) from e
    raise NotImplementedError("SAM2 backend is a configuration hook; not bundled.")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyze_painting(cfg: PaintingAnalysisConfig) -> dict:
    """Run Stage -1: painting -> depth, normals, strokes, confidence.

    Returns a dict with paths and tensors. Tensors are kept on CPU as numpy
    arrays for downstream stages.
    """
    if not cfg.painting_path:
        raise ValueError("PaintingAnalysisConfig.painting_path must be set")

    img_uint8 = _load_image(cfg.painting_path, max_side=cfg.max_side)
    rgb = _to_tensor(img_uint8)

    # --- Depth ---
    if cfg.depth_backend == "synthetic":
        depth = synthetic_painting_depth(rgb)
    elif cfg.depth_backend == "depth_anything_v3":
        depth = _depth_anything_v3(rgb, cfg.depth_model_name)
    else:
        # Other backends share the synthetic fallback interface.
        depth = synthetic_painting_depth(rgb)

    # --- Normals ---
    normals = depth_to_normals(depth)

    # --- Strokes ---
    if cfg.stroke_backend == "synthetic":
        strokes = synthetic_stroke_segmentation(
            rgb,
            min_area=cfg.stroke_min_area,
            merge_threshold=cfg.stroke_merge_threshold,
        )
    else:
        strokes = synthetic_stroke_segmentation(
            rgb,
            min_area=cfg.stroke_min_area,
            merge_threshold=cfg.stroke_merge_threshold,
        )

    # --- Confidence ---
    conf = painting_confidence(rgb, depth)

    # --- Persist ---
    os.makedirs(cfg.out_dir, exist_ok=True)
    _save_rgb(img_uint8, os.path.join(cfg.out_dir, "painting.png"))
    _save_gray(depth.squeeze(0).cpu().numpy(), os.path.join(cfg.out_dir, "depth.png"), cmap="inferno")
    # Normal map as a coloured image (xyz -> rgb).
    n_vis = ((normals.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8)
    _save_rgb(n_vis, os.path.join(cfg.out_dir, "normals.png"))
    # Strokes as a coloured label image.
    strokes_np = strokes.cpu().numpy()
    num_strokes = int(strokes_np.max())
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(max(num_strokes + 1, 1), 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    strokes_rgb = palette[strokes_np]
    _save_rgb(strokes_rgb, os.path.join(cfg.out_dir, "strokes.png"))
    _save_gray(conf.squeeze(0).cpu().numpy(), os.path.join(cfg.out_dir, "confidence.png"), cmap="viridis")

    # NPZ bundle for downstream stages.
    np.savez_compressed(
        os.path.join(cfg.out_dir, "painting.npz"),
        image=img_uint8,
        depth=depth.squeeze(0).cpu().numpy(),
        normals=normals.permute(1, 2, 0).cpu().numpy(),
        strokes=strokes_np,
        confidence=conf.squeeze(0).cpu().numpy(),
        config=asdict(cfg),
    )

    return {
        "out_dir": cfg.out_dir,
        "image": img_uint8,
        "depth": depth.squeeze(0).cpu().numpy(),
        "normals": normals.permute(1, 2, 0).cpu().numpy(),
        "strokes": strokes_np,
        "confidence": conf.squeeze(0).cpu().numpy(),
        "num_strokes": num_strokes,
    }
