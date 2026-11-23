import random

import pytest
from PIL import Image

from promogen.layout import (cover_crop_rect, cover_size, place_logos,
                              rects_overlap)


def test_cover_crop_fills_target():
    # 4:3 source into a 9:16 frame: must cover, never letterbox
    cw, ch = cover_size(4000, 3000, 1080, 1920)
    assert cw >= 1080 and ch >= 1920
    assert cw / ch == pytest.approx(4 / 3)
    l, t, r, b = cover_crop_rect(4000, 3000, 1080, 1920)
    assert (r - l, b - t) == (1080, 1920)
    # portrait source into 9:16: full width used
    cw2, ch2 = cover_size(1000, 3000, 1080, 1920)
    assert cw2 == 1000 * (3000 * 1080 // 1000 // 1920 + 1) or cw2 >= 1080


def test_cover_crop_math_exact():
    # exact-aspect source: crop equals the target exactly
    assert cover_size(1080, 1920, 1080, 1920) == (1080, 1920)


def test_rects_overlap():
    a = (0, 0, 100, 100)
    assert rects_overlap(a, (50, 50, 150, 150))
    assert not rects_overlap(a, (101, 0, 200, 100))
    assert rects_overlap(a, (99, 0, 200, 100), pad=5)
    assert not rects_overlap(a, (120, 0, 200, 100), pad=5)


def _logo(w=200, h=80):
    return Image.new("RGBA", (w, h), (255, 0, 0, 255))


def test_logo_placement_no_overlap_intro():
    rng = random.Random(7)
    avoid = [(300, 500, 780, 620), (0, 60, 1080, 130)]
    placements = place_logos([_logo() for _ in range(3)], 1080, 1920,
                             avoid, rng)
    boxes = [(p.x, p.y, p.x + p.image.width, p.y + p.image.height)
             for p in placements]
    for i, b1 in enumerate(boxes):
        for j, b2 in enumerate(boxes):
            if i < j and placements[i].opacity == placements[j].opacity == 1.0:
                assert not rects_overlap(b1, b2, pad=10), "logos overlap"
        for a in avoid:
            if placements[i].opacity == 1.0:
                assert not rects_overlap(b1, a), "logo hits text block"


def test_logo_placement_deterministic():
    def run(seed):
        rng = random.Random(seed)
        return [(p.x, p.y, p.opacity) for p in
                place_logos([_logo()], 1080, 1920, [(300, 500, 780, 620)], rng)]
    assert run(9) == run(9)
    # no infinite loops on a hopeless case: tiny frame, huge avoid zone
    rng = random.Random(1)
    place_logos([_logo(50, 50)], 120, 120, [(0, 0, 120, 120)], rng,
                max_fig_attempts=5, rounds=2)
