"""Synthetic oil-painting generator and data utilities for the Paint2World demo.

The synthetic painting generator produces a perceptually plausible oil-style
image with layered brush strokes, impasto texture, and a clear depth layout
(foreground / midground / background). It is used to ship sample data with
the demo so users can try Paint2World end-to-end without bringing their own
artwork.
"""

from __future__ import annotations

import math
import os
import random
from typing import Optional

import numpy as np


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """h in [0, 1), s, v in [0, 1]."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def _draw_brush_stroke(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    angle: float,
    length: float,
    width: float,
    colour: tuple[float, float, float],
    opacity: float = 0.85,
    rng: Optional[random.Random] = None,
) -> None:
    """Draw a single oriented brush stroke (an ellipse with impasto noise)."""
    rng = rng or random
    H, W, _ = canvas.shape
    # Bounding box of the rotated ellipse.
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    radius = int(max(length, width)) + 2
    y0, y1 = max(0, int(cy - radius)), min(H, int(cy + radius))
    x0, x1 = max(0, int(cx - radius)), min(W, int(cx + radius))
    if y0 >= y1 or x0 >= x1:
        return
    ys, xs = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )
    dx = xs - cx
    dy = ys - cy
    # Rotate into stroke frame.
    u = cos_a * dx + sin_a * dy
    v = -sin_a * dx + cos_a * dy
    # Ellipse mask.
    ellipse = (u / max(length, 1e-3)) ** 2 + (v / max(width, 1e-3)) ** 2
    mask = ellipse <= 1.0
    # Soft edge.
    edge = np.clip((1.05 - ellipse) / 0.05, 0.0, 1.0)
    alpha = (mask.astype(np.float32) * edge * opacity)
    # Impasto noise: per-pixel brightness jitter.
    noise = (rng.random() - 0.5) * 0.15
    col = np.array(colour, dtype=np.float32) + noise
    col = np.clip(col, 0.0, 1.0)
    for c in range(3):
        canvas[y0:y1, x0:x1, c] = (
            alpha * col[c] + (1.0 - alpha) * canvas[y0:y1, x0:x1, c]
        )


def generate_synthetic_oil_painting(
    width: int = 512,
    height: int = 384,
    seed: int = 0,
    palette: Optional[list[tuple[float, float, float]]] = None,
    style: str = "landscape",
) -> np.ndarray:
    """Generate a synthetic oil-style painting as a (H, W, 3) uint8 RGB array.

    ``style`` controls the scene layout: ``landscape`` (sky / hills / ground),
    ``still_life`` (table + objects), or ``portrait`` (figure silhouette).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), dtype=np.float32)

    if palette is None:
        # A warm landscape palette.
        palette = [
            _hsv_to_rgb(0.55 + 0.05 * rng.random(), 0.4, 0.85),  # sky
            _hsv_to_rgb(0.55 + 0.05 * rng.random(), 0.5, 0.7),   # distant hills
            _hsv_to_rgb(0.30 + 0.05 * rng.random(), 0.6, 0.6),   # midground
            _hsv_to_rgb(0.10 + 0.05 * rng.random(), 0.7, 0.5),   # foreground
            _hsv_to_rgb(0.05 + 0.05 * rng.random(), 0.8, 0.4),   # warm accent
        ]

    if style == "landscape":
        # Sky gradient (top 60%).
        horizon = int(height * 0.6)
        for y in range(horizon):
            t = y / max(horizon, 1)
            col = (
                palette[0][0] * (1.0 - t) + palette[0][0] * t,
                palette[0][1] * (1.0 - t) + 0.85 * t,
                palette[0][2] * (1.0 - t) + 0.95 * t,
            )
            canvas[y, :, :] = col
        # Distant hills.
        for y in range(horizon - 10, horizon + 20):
            t = (y - (horizon - 10)) / 30.0
            col = (
                palette[1][0] * (1.0 - t) + palette[2][0] * t,
                palette[1][1] * (1.0 - t) + palette[2][1] * t,
                palette[1][2] * (1.0 - t) + palette[2][2] * t,
            )
            canvas[y, :, :] = col
        # Foreground ground.
        for y in range(horizon + 20, height):
            t = (y - (horizon + 20)) / max(height - horizon - 20, 1)
            col = (
                palette[3][0] * (1.0 - t) + palette[4][0] * t,
                palette[3][1] * (1.0 - t) + palette[4][1] * t,
                palette[3][2] * (1.0 - t) + palette[4][2] * t,
            )
            canvas[y, :, :] = col

        # Add brush strokes everywhere, oriented along local gradients.
        num_strokes = 600
        for _ in range(num_strokes):
            cx = rng.uniform(0, width)
            cy = rng.uniform(0, height)
            # Orient: horizontal in sky/foreground, diagonal on hills.
            if cy < horizon * 0.8:
                angle = rng.uniform(-0.2, 0.2)
            elif cy < horizon + 25:
                angle = rng.uniform(-0.8, 0.8)
            else:
                angle = rng.uniform(-0.3, 0.3)
            length = rng.uniform(8, 22)
            width_s = rng.uniform(2, 5)
            # Sample colour from local region.
            ix = int(min(width - 1, max(0, cx)))
            iy = int(min(height - 1, max(0, cy)))
            base = canvas[iy, ix].tolist()
            # Perturb colour slightly.
            col = tuple(min(1.0, max(0.0, c + rng.uniform(-0.1, 0.1))) for c in base)
            _draw_brush_stroke(canvas, cx, cy, angle, length, width_s, col, opacity=rng.uniform(0.5, 0.9), rng=rng)

        # A warm focal accent (e.g., a sun / tree).
        sx = width * rng.uniform(0.3, 0.7)
        sy = horizon * rng.uniform(0.5, 0.9)
        for _ in range(80):
            cx = sx + rng.uniform(-15, 15)
            cy = sy + rng.uniform(-15, 15)
            _draw_brush_stroke(canvas, cx, cy, rng.uniform(0, math.pi), 10, 4, palette[4], opacity=0.6, rng=rng)

    elif style == "still_life":
        # Table.
        canvas[:, :, :] = _hsv_to_rgb(0.08, 0.3, 0.5)
        # Background wall.
        canvas[: height // 2, :, :] = _hsv_to_rgb(0.1, 0.2, 0.7)
        # Objects: a few coloured ellipses (fruit / vase).
        for _ in range(5):
            cx = rng.uniform(width * 0.2, width * 0.8)
            cy = rng.uniform(height * 0.4, height * 0.8)
            col = palette[rng.randint(0, len(palette) - 1)]
            for _ in range(60):
                _draw_brush_stroke(canvas, cx + rng.uniform(-20, 20), cy + rng.uniform(-20, 20),
                                   rng.uniform(0, math.pi), 12, 4, col, opacity=0.7, rng=rng)
        # Brush texture across the whole canvas.
        for _ in range(400):
            cx = rng.uniform(0, width)
            cy = rng.uniform(0, height)
            ix = int(min(width - 1, max(0, cx)))
            iy = int(min(height - 1, max(0, cy)))
            base = canvas[iy, ix].tolist()
            col = tuple(min(1.0, max(0.0, c + rng.uniform(-0.08, 0.08))) for c in base)
            _draw_brush_stroke(canvas, cx, cy, rng.uniform(-0.4, 0.4), 10, 3, col, opacity=0.5, rng=rng)

    else:  # portrait
        canvas[:, :, :] = _hsv_to_rgb(0.05, 0.4, 0.3)
        # Figure silhouette.
        cx_fig = width // 2
        cy_fig = int(height * 0.55)
        for y in range(height):
            for x in range(width):
                dx = (x - cx_fig) / (width * 0.18)
                dy = (y - cy_fig) / (height * 0.35)
                if dx * dx + dy * dy < 1.0:
                    canvas[y, x, :] = _hsv_to_rgb(0.07, 0.5, 0.6)
        # Brush texture.
        for _ in range(500):
            cx = rng.uniform(0, width)
            cy = rng.uniform(0, height)
            ix = int(min(width - 1, max(0, cx)))
            iy = int(min(height - 1, max(0, cy)))
            base = canvas[iy, ix].tolist()
            col = tuple(min(1.0, max(0.0, c + rng.uniform(-0.1, 0.1))) for c in base)
            _draw_brush_stroke(canvas, cx, cy, rng.uniform(-0.5, 0.5), 12, 3, col, opacity=0.6, rng=rng)

    # Add subtle canvas-weave texture.
    weave = (np_rng.random((height, width, 1)) - 0.5) * 0.05
    canvas = np.clip(canvas + weave, 0.0, 1.0)
    return (canvas * 255.0).astype(np.uint8)


def make_sample_paintings(out_dir: str = "data/sample_paintings", num: int = 3) -> list[str]:
    """Generate a small set of sample synthetic oil paintings for the demo."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    styles = ["landscape", "still_life", "portrait"]
    paths = []
    for i in range(min(num, len(styles))):
        img = generate_synthetic_oil_painting(width=512, height=384, seed=i + 1, style=styles[i])
        path = os.path.join(out_dir, f"sample_{i:02d}_{styles[i]}.png")
        Image.fromarray(img).save(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    paths = make_sample_paintings()
    print("Wrote sample paintings:")
    for p in paths:
        print(" ", p)
