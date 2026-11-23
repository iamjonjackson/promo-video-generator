"""Shared palette and easing helpers."""
from __future__ import annotations

import math

CYAN = (41, 232, 255)
MAGENTA = (255, 61, 176)
YELLOW = (255, 214, 41)
TEXT = (235, 245, 255)
BG_TOP = (10, 12, 32)
BG_BOTTOM = (36, 16, 74)
CHIP_FILL = (10, 12, 32)

ACCENTS = (CYAN, MAGENTA, YELLOW)

VIGNETTE_CORNER = 0.70


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def out_back(t: float, overshoot: float = 1.70158) -> float:
    """Ease-out-back: overshoots above 1 then settles at 1."""
    t = clamp01(t)
    c3 = overshoot + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + overshoot * (t - 1.0) ** 2


def beat_pulse(t: float, beat_sec: float, decay: float = 6.0) -> float:
    """Exponentially decaying pulse within each beat; 1 at each beat boundary."""
    if beat_sec <= 0:
        return 0.0
    phase = (t % beat_sec) / beat_sec
    return math.exp(-decay * phase)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
