"""Stage -0.5a: Object-level segmentation of an oil painting.

While the brush-stroke segmentation of Stage -1 models *marks* (individual
painterly gestures), this module decomposes the painting into *things* —
the objects a human painter composes into a scene: trees, hills, vessels,
fruit, faces, buildings. Object masks are the unit of

  * TripoSplat-based per-object 3D generation (``triposplat.py``), and
  * the 3D spatial-dependency graph and scene canvas (``scene_canvas.py``).

The default ``synthetic`` backend clusters pixels jointly in LAB colour and
estimated-depth space with spatial connectivity (a depth-aware superpixel
merge), scores each region by *objectness* (area x depth coherence x colour
distinctness) and keeps the top-k regions as objects. It requires no network
weights so the demo runs end-to-end on CPU; SAM2 / Mask2Former checkpoints
plug in behind the same interface (an instance-label map + per-object
metadata).
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from .configs import ObjectSegmentationConfig
from .painting_analysis import _lab_from_rgb, _smooth, _gradient_magnitude, _relative_luminance


# ---------------------------------------------------------------------------
# Synthetic object segmentation
# ---------------------------------------------------------------------------

def _connected_component_labels(
    assign: torch.Tensor,  # (H, W) long, seed-cluster ids per pixel
    edge_mask: torch.Tensor,  # (H, W) bool, True where a depth boundary blocks merging
    min_area: int,
) -> torch.Tensor:
    """Split the seed-cluster map into spatially connected components.

    Components are grown over 4-neighbours; pixels across an ``edge_mask``
    boundary are never merged, so depth discontinuities (the strongest object
    cue) survive colour merging. Uses ``scipy.ndimage.label`` per seed cluster
    when available (fast C implementation) and falls back to an iterative
    min-label propagation in pure torch otherwise.
    """
    h, w = assign.shape
    labels = torch.zeros((h, w), dtype=torch.long, device=assign.device)

    try:
        from scipy import ndimage  # lazy; optional speed-up

        cur = 1
        assign_np = assign.cpu().numpy()
        edge_np = edge_mask.cpu().numpy()
        for seed in np.unique(assign_np):
            comp_mask, n = ndimage.label(assign_np == seed)
            for i in range(1, n + 1):
                m = comp_mask == i
                if m.sum() < min_area:
                    continue
                # Split the component at depth-boundary edges.
                sub, n2 = ndimage.label(m & ~edge_np if edge_np.any() else m)
                for j in range(1, n2 + 1):
                    m2 = sub == j
                    if m2.sum() >= min_area:
                        labels[torch.from_numpy(m2).to(assign.device)] = cur
                        cur += 1
        return labels
    except Exception:
        pass

    # ---- Pure-torch fallback: iterative min-label propagation ----
    # Pixels start with unique ids; we repeatedly take the running min over the
    # 4-neighbourhood restricted to (same seed AND same edge side), which
    # converges to connected-component ids.
    idx = torch.arange(h * w, device=assign.device, dtype=torch.long).reshape(h, w)
    cur = idx.clone()
    same_seed = torch.ones((h, w), dtype=torch.bool, device=assign.device)
    for _ in range(max(h, w)):
        changed = False
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ys_a = slice(max(0, -dy), h - max(0, dy))
            xs_a = slice(max(0, -dx), w - max(0, dx))
            ys_b = slice(max(0, dy), h + min(0, dy))
            xs_b = slice(max(0, dx), w + min(0, dx))
            ok = (
                (assign[ys_a, xs_a] == assign[ys_b, xs_b])
                & (edge_mask[ys_a, xs_a] == edge_mask[ys_b, xs_b])
            )
            cand = cur[ys_b, xs_b].where(ok, cur[ys_a, xs_a])
            upd = cand < cur[ys_a, xs_a]
            if upd.any():
                changed = True
                cur[ys_a, xs_a] = cand
        if not changed:
            break

    # Compact ids and drop tiny fragments.
    uniq = [int(v) for v in cur.unique().tolist()]
    nxt = 1
    for u in uniq:
        m = cur == u
        if int(m.sum().item()) >= min_area:
            labels[m] = nxt
            nxt += 1
    return labels


def synthetic_object_segmentation(
    rgb: torch.Tensor,  # (3, H, W) in [0, 1]
    depth: torch.Tensor,  # (1, H, W) in [0, 1], 0 = near, 1 = far
    min_area: int = 400,
    max_objects: int = 12,
    depth_coherence_weight: float = 0.5,
    colour_distinctness_weight: float = 0.5,
) -> tuple[torch.Tensor, list[dict]]:
    """Depth-aware colour clustering into object instances.

    Returns ``(labels, objects)`` where ``labels`` is an (H, W) long map
    (0 = background, i > 0 = i-th object) and ``objects`` is a list of
    per-object metadata dicts (mask, bbox, area, mean colour/depth, depth
    coherence, objectness).
    """
    device = rgb.device
    c, h, w = rgb.shape
    assert depth.shape == (1, h, w)

    # 1. Coarse seed grid (much coarser than stroke superpixels: objects are
    #    large and contiguous).
    step = max(16, int(round(math.sqrt(h * w / 64.0))))
    lab = _smooth(_lab_from_rgb(rgb), sigma=2.0)
    dep = _smooth(depth, sigma=2.0)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    sy = torch.arange(step // 2, h, step, device=device)
    sx = torch.arange(step // 2, w, step, device=device)
    gy, gx = torch.meshgrid(sy, sx, indexing="ij")
    seeds = torch.stack([gy.flatten(), gx.flatten()], dim=1)
    n_seeds = seeds.shape[0]
    if n_seeds == 0:
        return torch.zeros((h, w), dtype=torch.long, device=device), []

    # 2. Assign each pixel to the nearest seed in (L, a, b, depth) space with
    #    a spatial regulariser — joint colour+layout clustering.
    seed_feat = torch.cat(
        [lab[:, seeds[:, 0], seeds[:, 1]].T, dep[0, seeds[:, 0], seeds[:, 1]].unsqueeze(1) * 60.0],
        dim=1,
    )  # (S, 4): depth scaled to be comparable with LAB distances
    pix_feat = torch.cat(
        [lab.permute(1, 2, 0).reshape(-1, 3), dep.reshape(-1, 1) * 60.0], dim=1
    )  # (HW, 4)
    pix_xy = torch.stack([ys.flatten().float(), xs.flatten().float()], dim=1)

    best_d = torch.full((h * w,), float("inf"), device=device)
    best_s = torch.zeros((h * w,), dtype=torch.long, device=device)
    chunk = max(1, 4096 * 16 // max(1, h * w // 256))
    for s in range(0, n_seeds, chunk):
        e = min(n_seeds, s + chunk)
        d_col = torch.cdist(pix_feat, seed_feat[s:e])
        d_spa = torch.cdist(pix_xy, seeds[s:e].float()) / (2.5 * step)
        d = d_col + d_spa
        sub_best, sub_idx = d.min(dim=1)
        imp = sub_best < best_d
        best_d = torch.where(imp, sub_best, best_d)
        best_s = torch.where(imp, sub_idx + s, best_s)
    assign = best_s.reshape(h, w)

    # 3. Depth-boundary edges: strong depth gradients block component merging
    #    so that objects sitting at different depths stay separate.
    dep_grad = _gradient_magnitude(dep)
    thr = float(dep_grad.mean().item()) + 1.5 * float(dep_grad.std().item())
    edge_mask = dep_grad.squeeze(0) > max(thr, 1e-3)

    # 4. Connected components with the depth-boundary barrier.
    labels = _connected_component_labels(assign, edge_mask, min_area=min_area)

    # 5. Merge adjacent components with very similar colour AND depth
    #    (over-segmentation repair).
    labels = _merge_adjacent_similar(labels, lab, dep, colour_tol=14.0, depth_tol=0.06)

    # 6. Objectness scoring and top-k selection.
    labels, objects = _select_objects(
        labels, lab, dep, rgb,
        max_objects=max_objects,
        depth_coherence_weight=depth_coherence_weight,
        colour_distinctness_weight=colour_distinctness_weight,
    )
    return labels, objects


def _merge_adjacent_similar(
    labels: torch.Tensor,
    lab: torch.Tensor,
    dep: torch.Tensor,
    colour_tol: float,
    depth_tol: float,
) -> torch.Tensor:
    """One pass merging 4-adjacent label pairs that are similar in colour and depth."""
    h, w = labels.shape
    uniq = [int(l) for l in labels.unique().tolist() if l != 0]
    if len(uniq) < 2:
        return labels
    means = {}
    for l in uniq:
        m = labels == l
        means[l] = (
            lab[:, m].mean(dim=1),
            float(dep[0, m].mean().item()),
            int(m.sum().item()),
        )
    # Adjacency via shifted comparisons.
    parent = {l: l for l in uniq}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def try_union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        ca, da, _ = means[ra]
        cb, db, _ = means[rb]
        if (ca - cb).norm().item() < colour_tol and abs(da - db) < depth_tol:
            # Union by larger area for stability.
            if means[ra][2] >= means[rb][2]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for dy, dx in ((0, 1), (1, 0)):
        a = labels[: h - dy, : w - dx] if dy else labels[:h, : w - 1]
        b = labels[dy:, dx:] if dy else labels[:, 1:]
        both = (a > 0) & (b > 0) & (a != b)
        if both.any():
            pairs = torch.stack([a[both], b[both]], dim=1).unique(dim=0)
            for pa, pb in pairs.tolist():
                try_union(int(pa), int(pb))

    # Compact and re-filter tiny fragments.
    remap = {}
    out = torch.zeros_like(labels)
    nxt = 1
    for l in uniq:
        r = find(l)
        if r not in remap:
            remap[r] = nxt
            nxt += 1
        out[labels == l] = remap[r]
    return out


def _select_objects(
    labels: torch.Tensor,
    lab: torch.Tensor,
    dep: torch.Tensor,
    rgb: torch.Tensor,
    max_objects: int,
    depth_coherence_weight: float,
    colour_distinctness_weight: float,
) -> tuple[torch.Tensor, list[dict]]:
    """Score each candidate region by objectness; keep the top-k as objects.

    Objectness = area_fraction x (w_d * depth_coherence + w_c * colour_distinctness):
      * depth coherence — low depth spread suggests a compact solid at one
        distance rather than a ground/sky sweep;
      * colour distinctness — LAB distance from the global scene mean marks
        regions the painter deliberately differentiated.
    """
    device = labels.device
    h, w = labels.shape
    total = float(h * w)
    scene_lab_mean = lab.reshape(3, -1).mean(dim=1)

    candidates = []
    for l in [int(v) for v in labels.unique().tolist() if v != 0]:
        m = labels == l
        area = int(m.sum().item())
        if area == 0:
            continue
        depth_vals = dep[0, m]
        d_std = float(depth_vals.std().item())
        depth_coh = float(torch.exp(-torch.tensor(d_std) / 0.12).item())  # ~1 for flat-in-depth
        col_dist = float((lab[:, m].mean(dim=1) - scene_lab_mean).norm().item()) / 100.0
        col_dist = min(1.0, col_dist)
        area_frac = area / total
        objectness = area_frac * (
            depth_coherence_weight * depth_coh + colour_distinctness_weight * col_dist
        )
        ys_idx, xs_idx = torch.nonzero(m, as_tuple=True)
        candidates.append(
            {
                "label": l,
                "area": area,
                "objectness": objectness,
                "depth_coherence": depth_coh,
                "colour_distinctness": col_dist,
                "bbox": [
                    int(xs_idx.min().item()), int(ys_idx.min().item()),
                    int(xs_idx.max().item()), int(ys_idx.max().item()),
                ],
                "mean_depth": float(depth_vals.mean().item()),
                "mean_colour_rgb": rgb[:, m].mean(dim=1).cpu().numpy(),
            }
        )

    candidates.sort(key=lambda d: d["objectness"], reverse=True)
    kept = candidates[: max_objects]

    # Relabel to consecutive ids; everything else becomes background (0).
    out = torch.zeros_like(labels)
    objects = []
    for new_id, cand in enumerate(kept, start=1):
        m = labels == cand["label"]
        out[m] = new_id
        meta = dict(cand)
        meta.pop("label")
        meta["id"] = new_id
        meta["mask"] = m.cpu().numpy()
        objects.append(meta)
    return out, objects


# ---------------------------------------------------------------------------
# Real-backend hook
# ---------------------------------------------------------------------------

def _sam2_objects(rgb: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
    """SAM2-based object segmentation. Imported lazily; requires sam2."""
    try:
        from segment_anything_2 import sam2  # type: ignore  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "SAM2 is not installed. Use backend='synthetic' for the "
            "out-of-the-box demo."
        ) from e
    raise NotImplementedError("SAM2 backend is a configuration hook; not bundled.")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def segment_objects(
    cfg: ObjectSegmentationConfig,
    image: np.ndarray,  # (H, W, 3) uint8
    depth: np.ndarray,  # (H, W) float in [0, 1]
    strokes: Optional[np.ndarray] = None,  # (H, W) int, used for visualisation only
) -> dict:
    """Run Stage -0.5a: painting -> object instance masks + metadata.

    Returns a dict with the label map, the per-object metadata list, and
    visualisation / persistence paths. Persists ``objects.npz`` and a
    coloured ``objects.png`` overlay in ``cfg.out_dir``.
    """
    rgb = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
    dep = torch.from_numpy(depth.copy()).float().unsqueeze(0)

    if cfg.backend in ("sam2", "mask2former"):
        labels, objects = _sam2_objects(rgb)
    else:
        labels, objects = synthetic_object_segmentation(
            rgb, dep,
            min_area=cfg.min_object_area,
            max_objects=cfg.max_objects,
            depth_coherence_weight=cfg.depth_coherence_weight,
            colour_distinctness_weight=cfg.colour_distinctness_weight,
        )

    labels_np = labels.cpu().numpy().astype(np.int32)
    num_objects = len(objects)

    # ---- Persist ----
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    palette = rng.integers(60, 255, size=(num_objects + 1, 3), dtype=np.uint8)
    palette[0] = (24, 24, 24)
    overlay = (rgb.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
    vis = overlay * 255.0 * 0.6 + palette[labels_np] * 0.4
    from PIL import Image

    Image.fromarray(vis.clip(0, 255).astype(np.uint8)).save(
        os.path.join(cfg.out_dir, "objects.png")
    )

    np.savez_compressed(
        os.path.join(cfg.out_dir, "objects.npz"),
        labels=labels_np,
        num_objects=num_objects,
        objectness=np.array([o["objectness"] for o in objects], dtype=np.float64),
        bbox=np.array([o["bbox"] for o in objects], dtype=np.int64).reshape(-1, 4),
        mean_depth=np.array([o["mean_depth"] for o in objects], dtype=np.float64),
        config=asdict(cfg),
    )

    return {
        "out_dir": cfg.out_dir,
        "labels": labels_np,
        "objects": objects,
        "num_objects": num_objects,
    }
