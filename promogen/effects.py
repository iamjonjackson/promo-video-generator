"""Transition effect registry.

Every effect is a small function with the common signature::

    fx(out_getter, in_getter, progress, frame_index, ctx) -> PIL.Image (RGB)

- ``out_getter`` / ``in_getter``: zero-argument callables returning the
  outgoing / incoming full-frame PIL image (RGB, ctx.size).
- ``progress``: 0..1 through the ~0.4s transition window.
- ``frame_index``: global output frame index (used for per-frame jitter).
- ``ctx``: :class:`EffectsContext` carrying frame size, a per-frame rng
  (seeded from the run seed) and the current transition index.

Register a new effect with the :func:`effect` decorator; it is automatically
eligible for random selection (and unit-tested by
``tests/test_effects.py::test_registry``).
"""
from __future__ import annotations

import math
import random
from collections import OrderedDict
from typing import Callable

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .style import ACCENTS, CYAN

WHITE = (255, 255, 255)

EFFECTS: OrderedDict[str, Callable] = OrderedDict()


def effect(fn):
    """Register a transition effect function under its function name."""
    EFFECTS[fn.__name__] = fn
    return fn


class EffectsContext:
    """Per-transition context handed to every effect."""

    def __init__(self, size, rng: random.Random, seed: int, index: int,
                 fps: int = 30, transition_sec: float = 0.4):
        self.size = (int(size[0]), int(size[1]))
        self.rng = rng                  # per-frame rng, seeded from the run seed
        self.seed = seed
        self.index = index               # transition index in the timeline
        self.fps = fps
        self.transition_sec = transition_sec

    def tr_rng(self) -> random.Random:
        """Deterministic per-transition rng: stable structure across frames."""
        return random.Random(f"{self.seed}:{self.index}")


def _np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _img(arr: np.ndarray, size) -> Image.Image:
    arr = np.ascontiguousarray(np.clip(arr, 0, 255), dtype=np.uint8)
    return Image.fromarray(arr)


def _ss(p: float) -> float:
    """0..1 -> 0..1 with eased ends (smoothstep)."""
    q = min(1.0, max(0.0, p))
    return q * q * (3 - 2 * q)


# ---------------------------------------------------------------- flash

@effect
def flash(out_getter, in_getter, progress, frame_index, ctx):
    """White crossfade with a bright flash overshoot at the midpoint."""
    t = _ss(progress)
    a = _np(out_getter()).astype(np.float32)
    b = _np(in_getter()).astype(np.float32)
    mixed = a * (1 - t) + b * t
    spike = math.exp(-((progress - 0.5) / 0.12) ** 2)
    mixed += np.array([255.0, 250.0, 235.0]) * spike
    return _img(mixed, ctx.size)


# -------------------------------------------------------- diagonal wipe

@effect
def diagonal_wipe(out_getter, in_getter, progress, frame_index, ctx):
    """45-degree expanding triangle mask plus a striped accent band."""
    w, h = ctx.size
    a = _np(out_getter())
    b = _np(in_getter())
    yy, xx = np.mgrid[0:h, 0:w]
    d = (xx + yy).astype(np.int32)
    edge = int(progress * (w + h))
    band = max(24, w // 24)
    res = np.where((d < edge)[:, :, None], b, a)
    in_band = np.abs(d - edge) < band
    stripe = in_band & ((d // 28) % 2 == 0)
    res[stripe] = CYAN
    res[in_band & ~stripe] = 0
    return _img(res, ctx.size)


# ---------------------------------------------------------------- glitch

@effect
def glitch(out_getter, in_getter, progress, frame_index, ctx):
    """12 horizontal rows flipping between slides with x-jitter and scanlines."""
    w, h = ctx.size
    rows = 12
    rh = max(1, h // rows)
    a = _np(out_getter())
    b = _np(in_getter())
    res = np.empty_like(a)
    flip_seed = ctx.tr_rng()
    thresholds = [flip_seed.random() * 0.9 + 0.05 for _ in range(rows)]
    rng = ctx.rng
    for i in range(rows):
        y0 = i * rh
        y1 = min(h, (i + 1) * rh) if i < rows - 1 else h
        src = b if progress > thresholds[i] else a
        row = src[y0:y1].copy()
        if rng.random() < 0.65:
            jx = int(rng.gauss(0, 0.05 * w * (1.0 - progress)))
            row = np.roll(row, jx, axis=1)
        if i % 2 == 0:                       # scanline darkening
            row = (row.astype(np.float32) * 0.84).astype(np.uint8)
        res[y0:y1] = row
    if rng.random() < 0.3:                    # occasional color overlay band
        i = rng.randrange(rows)
        y0 = i * rh
        y1 = min(h, (i + 1) * rh)
        tint = np.array(ACCENTS[rng.randrange(3)], dtype=np.float32)
        res[y0:y1] = (res[y0:y1].astype(np.float32)
                      + tint * 0.5).clip(0, 255).astype(np.uint8)
    return _img(res, ctx.size)


# --------------------------------------------------------------- circles

@effect
def circles(out_getter, in_getter, progress, frame_index, ctx):
    """Growing circle reveal, dashed accent ring, confetti dots on the edge."""
    w, h = ctx.size
    canvas = out_getter().convert("RGB").copy()
    max_r = math.hypot(w, h) / 2 * 1.05
    r = _ss(progress) * max_r
    if r > 1:
        inc = in_getter().convert("RGB")
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).ellipse(
            [w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r], fill=255)
        canvas.paste(inc, (0, 0), mask)
    draw = ImageDraw.Draw(canvas)
    tr = ctx.tr_rng()
    segs = 36
    width = max(2, w // 360)
    for k in range(0, segs, 2):
        a0 = k * 360.0 / segs
        a1 = (k + 1) * 360.0 / segs
        draw.arc([w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r], a0, a1,
                 fill=ACCENTS[k % 3], width=width)
    s = max(3, w // 220)
    for _ in range(12):
        ang = tr.uniform(0, 360)
        x = w / 2 + math.cos(math.radians(ang)) * (r + w * 0.015)
        y = h / 2 + math.sin(math.radians(ang)) * (r + w * 0.015)
        draw.rectangle([x - s, y - s, x + s, y + s],
                       fill=ACCENTS[tr.randrange(3)])
    return canvas


# ------------------------------------------------------------------ rows

@effect
def rows(out_getter, in_getter, progress, frame_index, ctx):
    """12 horizontal bars sliding in alternately from left/right."""
    w, h = ctx.size
    bars = 12
    bh = max(1, h // bars)
    a = _np(out_getter())
    b = _np(in_getter())
    tr = ctx.tr_rng()
    bias = [tr.random() * 0.35 for _ in range(bars)]
    res = a.copy()
    bar_w = max(3, w // 180)
    for i in range(bars):
        g = progress * 1.35 - bias[i]
        if g <= 0:
            continue
        g = min(1.0, g)
        y0 = i * bh
        y1 = min(h, (i + 1) * bh) if i < bars - 1 else h
        src = b[y0:y1]
        cut = max(1, int(g * w))
        if i % 2 == 0:                        # from the left
            part = src[:, :cut] if g < 1 else src
            res[y0:y1, :part.shape[1]] = part
            if g < 1:
                ex = max(1, part.shape[1])
                res[y0:y1, max(0, ex - bar_w):ex] = ACCENTS[i % 3]
        else:                                 # from the right
            part = src[:, w - cut:] if g < 1 else src
            res[y0:y1, w - part.shape[1]:] = part
            if g < 1:
                ex = w - part.shape[1]
                res[y0:y1, ex:min(w, ex + bar_w)] = ACCENTS[i % 3]
    return _img(res, ctx.size)


# ---------------------------------------------------------------- mosaic

@effect
def mosaic(out_getter, in_getter, progress, frame_index, ctx):
    """Pixelate-out then pixelate-in with a dark wash."""
    w, h = ctx.size
    q = 1 + math.sin(progress * math.pi) * 11
    src = in_getter if progress > 0.5 else out_getter
    img = src().convert("RGB")
    sw = max(2, int(w / q))
    sh = max(2, int(h / q))
    up = img.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.NEAREST)
    wash = 0.45 * math.sin(progress * math.pi)
    return _img(_np(up).astype(np.float32) * (1 - wash), ctx.size)


# ----------------------------------------------------------------- swipe

@effect
def swipe(out_getter, in_getter, progress, frame_index, ctx):
    """Incoming slide sweeps left-to-right with ghosts, cyan edge and chevrons."""
    w, h = ctx.size
    t = _ss(progress)
    canvas = out_getter().convert("RGB").copy()
    inc = in_getter().convert("RGB")
    off = int((1 - t) * w)
    for k, alpha in ((2, 60), (1, 110), (0, 255)):
        x = off + k * max(8, w // 12)          # ghosts trail the leading edge
        if x < w:
            strip = inc.crop((0, 0, min(inc.width, w - x), h))
            if strip.width > 0:
                if alpha < 255:
                    ghost = strip.convert("RGBA")
                    ghost.putalpha(alpha)
                    canvas.paste(ghost, (x, 0), ghost)
                else:
                    canvas.paste(strip, (x, 0))
    draw = ImageDraw.Draw(canvas)
    ex0, ex1 = off, min(w - 1, off + max(4, w // 250))
    if ex0 < w and ex1 > ex0:
        draw.rectangle([ex0, 0, ex1, h], fill=CYAN)
    arm = max(14, w // 40)                     # 3 chevrons ahead of the edge
    for k in range(3):
        cx = off - (k + 1) * arm * 2 - 20
        if cx > 8:
            draw.polygon([(cx, h // 2 - arm), (cx + arm, h // 2),
                          (cx, h // 2 + arm)],
                         outline=ACCENTS[k % 3], width=max(4, w // 200))
    return canvas


# ------------------------------------------------------------ zoom punch

@effect
def zoom_punch(out_getter, in_getter, progress, frame_index, ctx):
    """Outgoing shrinks + rotates away with rays; incoming punches in with a flash tail."""
    w, h = ctx.size
    p = progress
    if p < 0.5:
        q = p / 0.5
        base = out_getter().convert("RGB").rotate(
            14 * q * q, resample=Image.BILINEAR)
        scale = 1.0 - 0.55 * q
        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        small = base.resize((sw, sh), Image.BILINEAR)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(small, ((w - sw) // 2, (h - sh) // 2))
        draw = ImageDraw.Draw(canvas, "RGBA")
        for k in range(10):                    # radial rays
            ang = math.radians(k * 36 + q * 60)
            r0, r1 = 0.30 * w, 0.72 * w
            col = ACCENTS[k % 3] + (120,)
            draw.line([(w / 2 + math.cos(ang) * r0, h / 2 + math.sin(ang) * r0),
                       (w / 2 + math.cos(ang) * r1, h / 2 + math.sin(ang) * r1)],
                      fill=col, width=max(2, w // 300))
        return canvas
    q = (p - 0.5) / 0.5
    scale = 1.5 - 0.5 * _ss(q)
    sw = int(w * scale)
    sh = int(h * scale)
    inc = in_getter().convert("RGB")
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(inc.resize((sw, sh), Image.BILINEAR),
                ((w - sw) // 2, (h - sh) // 2))
    spike = math.exp(-((q - 0.1) / 0.25) ** 2)  # white flash tail
    if spike > 0.02:
        level = int(240 * spike)
        canvas = ImageChops.add(canvas, Image.new("RGB", (w, h), (level,) * 3))
    return canvas


# ------------------------------------------------------------------ dots

@effect
def dots(out_getter, in_getter, progress, frame_index, ctx):
    """Grid of square cells filling randomly; falling colored specks."""
    w, h = ctx.size
    cell = max(2, w // 24)
    cols = w // cell
    rows_n = h // cell
    res = _np(out_getter()).copy()
    b = _np(in_getter())
    tr = ctx.tr_rng()
    cells = [(cx, cy) for cy in range(rows_n) for cx in range(cols)]
    tr.shuffle(cells)
    k = int(progress * len(cells))
    ch = np.zeros((rows_n, cols), dtype=bool)
    for cx, cy in cells[:k]:
        ch[cy, cx] = True
    k = np.kron(ch, np.ones((cell, cell), dtype=bool))
    pick = np.zeros((h, w), dtype=bool)
    pick[:k.shape[0], :k.shape[1]] = k
    res[pick] = b[pick]
    draw = ImageDraw.Draw(Image.fromarray(res))
    s = max(2, cell // 3)
    for i in range(30):                       # falling colored specks
        base = tr.random()
        sx = int(tr.random() * max(1, w - s))
        sy = int((base * h + frame_index * (2 + int(base * 6)) * 3) % h)
        draw.rectangle([sx, sy, sx + s, sy + s], fill=ACCENTS[i % 3])
    return Image.fromarray(res)


EFFECT_NAMES = list(EFFECTS.keys())
