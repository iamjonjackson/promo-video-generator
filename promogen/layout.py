"""Geometry: cover-crop math, rectangle collision, deterministic logo placement."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Rect = Tuple[int, int, int, int]


def cover_size(sw: int, sh: int, tw: int, th: int) -> Tuple[int, int]:
    """Scale factors for cover-crop: image scaled so it fills (tw, th) entirely."""
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        raise ValueError(f"invalid dimensions for cover crop: {sw}x{sh} -> {tw}x{th}")
    scale = max(tw / sw, th / sh)
    return int(round(sw * scale)), int(round(sh * scale))


def cover_crop_rect(sw: int, sh: int, tw: int, th: int) -> Rect:
    """Centered crop box (left, top, right, bottom) on a cover-scaled image."""
    cw, ch = cover_size(sw, sh, tw, th)
    left = (cw - tw) // 2
    top = (ch - th) // 2
    return (left, top, left + tw, top + th)


def rects_overlap(a: Rect, b: Rect, pad: int = 0) -> bool:
    """True if axis-aligned rectangles a and b overlap when padded by `pad`."""
    ax0, ay0, ax1, ay1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    bx0, by0, bx1, by1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


@dataclass
class LogoPlacement:
    image: object          # PIL RGBA image, already scaled and rotated
    x: int
    y: int
    opacity: float         # 1.0 normally; 0.6 when placement gives up
    attempts: int


def _rotated_bbox(img) -> Rect:
    a = img.getbbox() or (0, 0, img.width, img.height)
    return (a[0], a[1], a[2], a[3])


def place_logos(
    logos: Sequence,
    frame_w: int,
    frame_h: int,
    avoid: Sequence[Rect],
    rng: random.Random,
    max_fig_attempts: int = 50,
    rounds: int = 3,
) -> List[LogoPlacement]:
    """Place logos randomly without overlapping each other or any `avoid` rect.

    Each logo width is 6-14% of frame width, rotation in [-8, +8] degrees,
    position within an ~8% safe margin. On failure the logo is shrunk 20% and
    retried; after `rounds` rounds it is placed anywhere with opacity ~60%.
    """
    from PIL import Image

    placements: List[LogoPlacement] = []
    occupied: List[Rect] = []
    margin_x = int(frame_w * 0.08)
    margin_y = int(frame_h * 0.08)
    pad = int(frame_w * 0.01)

    for logo in logos:
        placed = None
        attempts = 0
        scale_w = rng.uniform(0.06, 0.14)
        for _ in range(rounds):
            w = max(16, int(frame_w * scale_w))
            base = logo.copy()
            base.thumbnail((w, int(w * logo.height / max(logo.width, 1))), Image.LANCZOS)
            angle = rng.uniform(-8.0, 8.0)
            rot = base.rotate(angle, expand=True, resample=Image.BICUBIC)
            for _ in range(max_fig_attempts):
                attempts += 1
                x = rng.randint(margin_x, max(margin_x, frame_w - margin_x - rot.width))
                y = rng.randint(margin_y, max(margin_y, frame_h - margin_y - rot.height))
                box = (x, y, x + rot.width, y + rot.height)
                if not any(rects_overlap(box, o, pad) for o in occupied + list(avoid)):
                    placed = LogoPlacement(rot, x, y, 1.0, attempts)
                    break
            if placed is not None:
                break
            scale_w *= 0.8  # shrink 20% and retry the next round
        if placed is None:
            x = rng.randint(margin_x, max(margin_x, frame_w - margin_x - rot.width))
            y = rng.randint(margin_y, max(margin_y, frame_h - margin_y - rot.height))
            placed = LogoPlacement(rot, x, y, 0.6, attempts)
        if placed.opacity >= 1.0:
            occupied.append((placed.x, placed.y, placed.x + placed.image.width,
                             placed.y + placed.image.height))
        placements.append(placed)
    return placements
