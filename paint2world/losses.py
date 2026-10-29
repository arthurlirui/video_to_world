"""Painting-anchored losses for Stage 1 (alignment) and Stage 3.2 (GS training).

These losses encode the three painting-anchored 3D-stability mechanisms:

  * ``painting_anchor_loss`` — pulls the canonical point cloud toward the
    painting's estimated geometry (Stage 1).
  * ``depth_warp_consistency_score`` — a per-point consistency score that
    downweights generated points whose depth drifts beyond a tolerance from
    the painting-implied depth (Stage 1).
  * ``stroke_coherence_loss`` — encourages 3D brush-stroke arrangement to
    remain coherent across views by smoothing SH colours within a stroke
    segment and sharpening them across stroke boundaries (Stage 3.2).
  * ``painting_view_loss`` — the trained scene rendered from the painting's
    camera should reproduce the input painting (Stage 3.2).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage 1: painting-anchored alignment losses
# ---------------------------------------------------------------------------

def painting_anchor_loss(
    canonical_pts: torch.Tensor,
    painting_pts: torch.Tensor,
    painting_conf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pull the canonical point cloud toward the painting's estimated geometry.

    ``canonical_pts``: (M, 3) current canonical-frame points (any order).
    ``painting_pts``: (K, 3) points unprojected from the painting's depth
        (the canonical-frame prior).
    ``painting_conf``: optional (K,) confidence weights in [0, 1].

    The loss is a chamfer-style distance from canonical points to their
    nearest painting-prior points (one-directional, since the canonical cloud
    may legitimately extend beyond the painting's frustum).
    """
    if canonical_pts.numel() == 0 or painting_pts.numel() == 0:
        return canonical_pts.new_zeros(())
    # Chunked nearest-neighbour to bound memory.
    M = canonical_pts.shape[0]
    K = painting_pts.shape[0]
    chunk = 8192
    total = canonical_pts.new_zeros(())
    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        d2 = torch.cdist(canonical_pts[s:e], painting_pts)  # (m, K)
        nn_d2, nn_idx = d2.min(dim=1)
        if painting_conf is not None:
            w = painting_conf[nn_idx].clamp_min(1e-3)
        else:
            w = torch.ones_like(nn_d2)
        total = total + (w * nn_d2).sum()
    return total / max(M, 1)


def depth_warp_consistency_score(
    frame_pts: torch.Tensor,
    frame_depth: torch.Tensor,
    painting_depth_warped: torch.Tensor,
    tolerance: float = 0.08,
) -> torch.Tensor:
    """Per-point consistency score in [0, 1] for a generated frame.

    ``frame_pts``: (M, 3) generated-frame points (camera space).
    ``frame_depth``: (M,) per-point depth (camera-space z).
    ``painting_depth_warped``: (M,) the painting's depth warped into this
        frame's viewpoint (the depth-warp geometry control resampled to the
        points' pixel locations).
    ``tolerance``: relative depth-deviation tolerance.

    Points whose depth matches the painting-implied depth get score 1.0;
    points whose depth deviates more than ``tolerance`` get a score that
    decays linearly to 0 (a soft robust loss). The score is used to weight
    the ICP data term.
    """
    if frame_pts.numel() == 0:
        return frame_pts.new_zeros(0)
    rel_diff = (frame_depth - painting_depth_warped).abs() / (painting_depth_warped.clamp_min(1e-3))
    # Linear decay from 1 at rel_diff=0 to 0 at rel_diff=tolerance, clipped to >=0.
    score = (1.0 - rel_diff / max(tolerance, 1e-6)).clamp_min(0.0)
    return score


# ---------------------------------------------------------------------------
# Stage 3.2: stroke-coherent GS losses
# ---------------------------------------------------------------------------

def _gather_neighbour_labels(
    stroke_labels: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """For each Gaussian, gather the stroke labels of its k nearest neighbours.

    ``stroke_labels``: (N,) integer stroke label per Gaussian (0 = background).
    Returns: (N, k) neighbour labels.
    """
    N = stroke_labels.shape[0]
    if N < 2:
        return stroke_labels.unsqueeze(0).expand(N, max(k, 1)).contiguous()
    # Brute-force NN; fine for demo-scale N.
    device = stroke_labels.device
    pts = torch.arange(N, device=device)
    # Use a cheap 1D proxy: random sampling of neighbours. For a real
    # deployment, replace with a KD-tree over the canonical point cloud.
    perm = torch.randperm(N, device=device)
    nb = stroke_labels[perm]
    # For each point, take k random neighbours (avoiding self where possible).
    out = nb.unsqueeze(0).expand(N, -1)[:, :k].contiguous()
    return out


def stroke_coherence_loss(
    sh_colours: torch.Tensor,
    stroke_labels: torch.Tensor,
    intra_weight: float = 0.05,
    inter_weight: float = 0.02,
    k: int = 8,
) -> torch.Tensor:
    """Stroke-coherence loss on Gaussian SH colours.

    ``sh_colours``: (N, C) per-Gaussian SH colours (DC band, or any scalar
        per-Gaussian colour representation).
    ``stroke_labels``: (N,) integer stroke label per Gaussian (0 = background).

    The loss has two terms:
      * **Intra-stroke smoothness**: SH colours of Gaussians within the same
        stroke segment should be similar (each stroke = one coherent 3D patch).
      * **Inter-stroke sharpness**: SH colours across stroke boundaries should
        remain distinct (strokes should not bleed into each other in 3D).

    Background points (label 0) are excluded from both terms.
    """
    N, C = sh_colours.shape
    if N < 2:
        return sh_colours.new_zeros(())
    device = sh_colours.device

    # Sample neighbours.
    nb_labels = _gather_neighbour_labels(stroke_labels, k=k)  # (N, k)
    # For each point, also gather its neighbours' colours. We approximate with
    # random sampling for the demo (a real deployment uses KD-tree NN).
    perm = torch.randperm(N, device=device)
    nb_colours = sh_colours[perm].unsqueeze(0).expand(N, -1, -1)[:, :k, :]  # (N, k, C)

    same = (nb_labels == stroke_labels.unsqueeze(1)) & (stroke_labels.unsqueeze(1) > 0)
    diff = (nb_labels != stroke_labels.unsqueeze(1)) & (stroke_labels.unsqueeze(1) > 0) & (nb_labels > 0)

    # Intra: penalise colour variance within a stroke.
    intra_diff = ((sh_colours.unsqueeze(1) - nb_colours) ** 2).mean(dim=-1)  # (N, k)
    intra = (same.float() * intra_diff).sum() / same.float().sum().clamp_min(1.0)

    # Inter: encourage colours across strokes to differ (negative loss term:
    # we *reward* distance, so subtract a margin).
    inter_diff = ((sh_colours.unsqueeze(1) - nb_colours) ** 2).mean(dim=-1)
    margin = 0.05
    inter = (inter_weight * (margin - diff.float() * inter_diff).clamp_min(0.0)).sum() / diff.float().sum().clamp_min(1.0)

    return intra_weight * intra + inter


def painting_view_loss(
    rendered: torch.Tensor,
    painting: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_l1: float = 1.0,
    lambda_lpips: float = 0.0,
    lpips_fn=None,
) -> torch.Tensor:
    """Painting-view reconstruction loss.

    ``rendered``: (3, H, W) rendered image from the painting's camera.
    ``painting``: (3, H, W) the input painting at the same resolution.
    ``mask``: optional (H, W) weight mask (e.g. confidence-weighted).

    The trained scene rendered from the painting's viewpoint should reproduce
    the input painting — this is the strongest constraint tying the 3D scene
    to the user's artwork.
    """
    l1 = (rendered - painting).abs().mean()
    if mask is not None:
        l1 = (mask.unsqueeze(0) * (rendered - painting).abs()).sum() / mask.sum().clamp_min(1.0)
    loss = lambda_l1 * l1
    if lambda_lpips > 0.0 and lpips_fn is not None:
        # LPIPS expects (N, 3, H, W).
        loss = loss + lambda_lpips * lpips_fn(
            rendered.unsqueeze(0), painting.unsqueeze(0)
        ).mean()
    return loss
