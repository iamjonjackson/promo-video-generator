"""Background, intro slide and outro slide rendering.

All planners are deterministic given (Config, seed-derived rng): rendering a
frame is a pure function of (planner, frame_index).
"""
from __future__ import annotations

import math
import random
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .config import Config, FPS, FONT_PATH
from .layout import LogoPlacement, Rect, place_logos
from .style import (ACCENTS, BG_BOTTOM, BG_TOP, CHIP_FILL, CYAN, MAGENTA,
                   TEXT, clamp01, out_back)


# ------------------------------------------------------------------ fonts

_FONT_CACHE: dict = {}


def get_font(size: int, path: str = FONT_PATH) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]


def fit_font(text: str, path: str, max_w: int, max_h: int,
             start: int = 400) -> ImageFont.FreeTypeFont:
    """Largest font size whose rendered `text` fits in (max_w, max_h)."""
    lo, hi = 8, max(9, start)
    best = ImageFont.truetype(path, 8)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(path, mid)
        bbox = f.getbbox(text)
        if bbox[2] - bbox[0] <= max_w and bbox[3] - bbox[1] <= max_h:
            best = f
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ------------------------------------------------------------- background

class Background:
    """Vertical gradient + radial glows + drifting confetti + intro stripes."""

    def __init__(self, w: int, h: int, rng: random.Random):
        self.w, self.h = w, h
        yy = np.linspace(0.0, 1.0, h)[:, None]
        top = np.array(BG_TOP, dtype=np.float32)
        bottom = np.array(BG_BOTTOM, dtype=np.float32)
        col = top[None, :] * (1 - yy) + bottom[None, :] * yy       # (h, 3)
        self.base = np.repeat(col[:, None, :], w, axis=1)          # (h, w, 3)
        yy2, xx2 = np.mgrid[0:h, 0:w]
        for (cx, cy, radius, color, alpha) in (
            (0.22 * w, 0.28 * h, 0.65 * w, CYAN, 0.16),
            (0.80 * w, 0.72 * h, 0.70 * w, MAGENTA, 0.13),
        ):
            r2 = (xx2 - cx) ** 2 + (yy2 - cy) ** 2
            falloff = np.exp(-r2 / (radius * radius)).astype(np.float32)
            self.base += np.array(color, dtype=np.float32)[None, None, :] * \
                (falloff * alpha)[:, :, None]
        self.base = np.clip(self.base, 0, 255).astype(np.uint8)

        self.confetti = [
            {
                "x": rng.uniform(0, 1),
                "y": rng.uniform(0, 1),
                "speed": rng.uniform(0.008, 0.03),
                "size": rng.uniform(0.002, 0.006) * w,
                "color": ACCENTS[rng.randrange(3)],
                "alpha": rng.uniform(35, 110),
            }
            for _ in range(26)
        ]
        self.stripe_tile = self._make_stripes()
        self.cache: dict = {}

    def _make_stripes(self) -> Image.Image:
        """Tileable 45-degree diagonal stripe mask (constant along x+y)."""
        period = max(48, self.w // 9)
        on = period // 3
        yy, xx = np.mgrid[0:self.h, 0:period]
        arr = (((xx + yy) % period) < on).astype(np.uint8) * 70
        return Image.fromarray(arr, "L")

    def frame(self, t: float, stripes: bool = False) -> Image.Image:
        """Background frame at time `t` seconds; confetti drifts upward."""
        img = Image.fromarray(self.base.copy(), "RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        for pt in self.confetti:
            y01 = (pt["y"] - t * pt["speed"]) % 1.0
            x = int(pt["x"] * self.w)
            y = int(y01 * self.h)
            s = int(pt["size"])
            draw.rectangle([x, y, x + s, y + s],
                           fill=pt["color"] + (int(pt["alpha"]),))
        if stripes:
            period = self.stripe_tile.width
            drift = int(t * 60) % period
            overlay = Image.new("L", (self.w, self.h), 0)
            for x0 in range(-1, self.w // period + 2):
                overlay.paste(self.stripe_tile, (x0 * period + drift, 0))
            white = Image.new("RGBA", (self.w, self.h), (255, 255, 255, 0))
            white.putalpha(overlay)
            img = Image.alpha_composite(img.convert("RGBA"), white).convert("RGB")
        return img


# ------------------------------------------------------- text rendering

def text_bbox(font: ImageFont.FreeTypeFont, text: str) -> Tuple[int, int, int, int]:
    return font.getbbox(text)


def draw_chroma_text(canvas: Image.Image, center_xy, text: str,
                     font: ImageFont.FreeTypeFont, entrance_t: float,
                     base_scale: float, rng: random.Random) -> Rect:
    """Draw centered text with an out-back entrance and chromatic-ghost RGB.

    Ghosting is strong during the first ~0.55 of the entrance. Returns the
    bounding rect of the settled text (used for logo collision).
    """
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = max(40, int(tw * 0.25))
    tile_w, tile_h = tw + 2 * pad, th + 2 * pad
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    td.text((pad - bbox[0], pad - bbox[1]), text, font=font,
            fill=TEXT + (255,))
    cx = center_xy[0] - tile_w // 2
    cy = center_xy[1] - tile_h // 2
    scale = base_scale * max(0.02, out_back(entrance_t))
    if scale != 1.0:
        nw, nh = max(1, int(tile_w * scale)), max(1, int(tile_h * scale))
        tile = tile.resize((nw, nh), Image.BICUBIC)
        cx += (tile_w - nw) // 2
        cy += (tile_h - nh) // 2
    ghost = 1.0 - clamp01(entrance_t / 0.55)
    if ghost > 0.01:
        alpha = tile.getchannel("A")
        d = max(3, int(tile_h * 0.05 * ghost) + 2)
        for off, col in ((-d, (255, 40, 90)), (d, CYAN)):
            solid = Image.new("RGBA", tile.size, col + (255,))
            solid.putalpha(alpha.point(lambda a, g=ghost: int(a * g * 0.85)))
            canvas.alpha_composite(solid, (cx + off, cy))
    canvas.alpha_composite(tile, (cx, cy))
    return (cx + pad, cy + pad, cx + pad + tw, cy + pad + th)


def draw_chip(canvas: Image.Image, center_xy, text: str,
              font: ImageFont.FreeTypeFont, entrance_t: float,
              accent=CYAN) -> Rect:
    """Rounded pill chip: dark fill, 4px accent stroke, white text."""
    scale = max(0.02, out_back(entrance_t))
    pad_x = int(font.size * 1.1)
    pad_y = int(font.size * 0.55)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w = int((tw + 2 * pad_x) * scale)
    h = int((th + 2 * pad_y) * scale)
    if w < 4 or h < 4:
        return (0, 0, 0, 0)
    chip = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cd = ImageDraw.Draw(chip)
    radius = h // 2
    cd.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                         fill=CHIP_FILL + (235,), outline=accent + (255,),
                         width=max(2, w // 260))
    f = font if scale == 1.0 else ImageFont.truetype(
        font.path, max(8, int(font.size * scale)))
    cb = f.getbbox(text)
    cd.text(((w - (cb[2] - cb[0])) // 2 - cb[0],
             (h - (cb[3] - cb[1])) // 2 - cb[1]),
            text, font=f, fill=(255, 255, 255, 255))
    x = int(center_xy[0] - w / 2)
    y = int(center_xy[1] - h / 2)
    canvas.alpha_composite(chip, (x, y))
    return (x, y, x + w, y + h)


def draw_logos(canvas: Image.Image, placements: Sequence[LogoPlacement]):
    for p in placements:
        img = p.image
        if p.opacity < 1.0:
            img = img.copy()
            a = img.getchannel("A").point(lambda v: int(v * p.opacity))
            img.putalpha(a)
        canvas.paste(img, (p.x, p.y), img)


# ------------------------------------------------------------ intro slide

class IntroPlanner:
    """Deterministic intro slide plan: timings, logo layout, fonts."""

    def __init__(self, cfg: Config, w: int, h: int, intro_frames: int,
                 rng: random.Random, logos: Sequence):
        self.cfg = cfg
        self.w, self.h = w, h
        self.intro_frames = intro_frames
        self.title = cfg.title.upper() or " "
        self.title2 = (cfg.title2 or "").upper() or " "

        t1_max_w = int(w * 0.86)
        self.font1 = fit_font(self.title, FONT_PATH, t1_max_w, int(h * 0.16), start=h // 4)
        self.font2 = fit_font(self.title2, FONT_PATH, int(w * 0.86), int(h * 0.16),
                              start=int(self.font1.size * 1.35))
        self.chip_font = get_font(max(18, w // 22))
        self.footer_font = get_font(max(14, w // 40))

        # per-letter layout: x offsets from getlength(), total centered
        self.letter_metrics = []
        cursor = 0.0
        for ch in self.title:
            wch = self.font1.getlength(ch)
            self.letter_metrics.append((cursor, wch))
            cursor += wch
        self.title_total_w = cursor
        b1 = self.font1.getbbox(self.title)
        th1 = b1[3] - b1[1]
        x0 = w / 2 - self.title_total_w / 2
        self.title_y_center = int(h * 0.36)
        self.title1_rect = (int(x0), self.title_y_center - th1 // 2,
                            int(x0 + self.title_total_w),
                            self.title_y_center + th1 // 2)
        b2 = self.font2.getbbox(self.title2)
        n = len(self.title)
        stagger = rng.choice([3, 4])
        entrance = 14
        tail = 26                              # title2 punch + chip + settle
        budget = max(4, intro_frames - tail - entrance - 2)
        while n > 1 and (n - 1) * stagger > budget:
            stagger -= 1
            if stagger <= 1:
                stagger = 1
                break
        self.stagger = stagger
        self.letter_start = 2
        self.entrance = entrance
        nat_end = self.letter_start + (n - 1) * stagger + entrance
        self.t2_start = max(nat_end - 2, intro_frames - tail + 2)
        self.t2_start = min(self.t2_start, max(0, intro_frames - 14))
        self.t2_entrance = 10
        self.chip_start = self.t2_start + 6
        self.chip_entrance = 10

        # title text blocks (settled) for logo collision avoidance
        cb = self.chip_font.getbbox(cfg.date_line.upper())
        self.title2_rect = self._rect_around(b2, (w // 2, int(h * 0.50))) if self.title2 else None
        chip_w = (cb[2] - cb[0]) + 2 * int(self.chip_font.size * 1.1)
        chip_h = (cb[3] - cb[1]) + 2 * int(self.chip_font.size * 0.55)
        self.chip_rect = (w // 2 - chip_w // 2, int(h * 0.62) - chip_h // 2,
                          w // 2 + chip_w // 2, int(h * 0.62) + chip_h // 2)
        self.footer_rect = (0, int(h * 0.02), w, int(h * 0.02) + self.footer_font.size * 3)
        avoid = [self.title1_rect, self.footer_rect]
        if self.title2_rect:
            avoid.append(self.title2_rect)
        avoid.append(self.chip_rect)
        self.logo_placements = (place_logos(logos, w, h, avoid, rng)
                                if logos else [])

    def _rect_around(self, bbox, center) -> Rect:
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return (center[0] - tw // 2, center[1] - th // 2,
                center[0] + tw // 2, center[1] + th // 2)

    def et(self, f: int, start: int) -> float:
        return clamp01((f - start) / self.entrance)

    def frame(self, bg: Background, f: int) -> Image.Image:
        cfg = self.cfg
        t = f / FPS
        img = bg.frame(t, stripes=True).convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")

        # footer watermark, from frame 0
        ft = cfg.footer_text()
        fb = self.footer_font.getbbox(ft)
        draw.text((self.w / 2 - (fb[2] - fb[0]) / 2 - fb[0],
                   self.h * 0.03), ft, font=self.footer_font,
                  fill=TEXT + (150,))

        draw_logos(img, self.logo_placements)

        # title letters, staggered, out-back + chroma ghosts
        for i, ch in enumerate(self.title):
            if ch == " ":
                continue
            start = self.letter_start + i * self.stagger
            if f < start:
                break
            et = clamp01((f - start) / self.entrance)
            off, wch = self.letter_metrics[i]
            x = int(self.title1_rect[0] + off + wch / 2)
            draw_chroma_text(img, (x, self.title_y_center), ch,
                             self.font1, et, 1.0, None)
        if self.title2 and f >= self.t2_start:
            et = clamp01((f - self.t2_start) / self.t2_entrance)
            draw_chroma_text(img, (self.w // 2, (self.title2_rect[1] +
                                                 self.title2_rect[3]) // 2),
                             self.title2, self.font2, et, 1.0, None)
        if f >= self.chip_start:
            et = clamp01((f - self.chip_start) / self.chip_entrance)
            draw_chip(img, ((self.chip_rect[0] + self.chip_rect[2]) // 2,
                            (self.chip_rect[1] + self.chip_rect[3]) // 2),
                      cfg.date_line.upper(), self.chip_font, et)
        return img.convert("RGB")


# ------------------------------------------------------------- outro slide

class OutroPlanner:
    """Deterministic outro slide plan: text with glow, chip, equalizer, logos."""

    def __init__(self, cfg: Config, w: int, h: int, rng: random.Random,
                 beat_sec: float, logos: Sequence):
        self.cfg = cfg
        self.w, self.h = w, h
        self.beat_sec = beat_sec
        self.title = cfg.title.upper() or " "
        self.title2 = (cfg.title2 or "").upper() or " "
        self.font1 = fit_font(self.title, FONT_PATH, int(w * 0.8), int(h * 0.12),
                              start=h // 5)
        self.font2 = fit_font(self.title2, FONT_PATH, int(w * 0.8), int(h * 0.12),
                              start=int(self.font1.size * 1.2))
        self.chip_font = get_font(max(18, w // 22))

        b1 = self.font1.getbbox(self.title)
        b2 = self.font2.getbbox(self.title2)
        self.title1_rect = self._rect_around(b1, (w // 2, int(h * 0.40)), 1.25)
        self.title2_rect = self._rect_around(b2, (w // 2, int(h * 0.52)), 1.25) if self.title2 else None
        cb = self.chip_font.getbbox(cfg.date_line.upper())
        chip_w = (cb[2] - cb[0]) + 2 * int(self.chip_font.size * 1.1)
        chip_h = (cb[3] - cb[1]) + 2 * int(self.chip_font.size * 0.55)
        self.chip_rect = (w // 2 - chip_w // 2, int(h * 0.63) - chip_h // 2,
                          w // 2 + chip_w // 2, int(h * 0.63) + chip_h // 2)
        self.eq_top = int(h * 0.76)
        self.eq_bottom = int(h * 0.90)
        self.eq_w = max(6, w // 40)
        self.eq_gap = self.eq_w // 2
        # logos first, avoiding text/chip/equalizer zones
        avoid = [self.title1_rect, self.chip_rect,
                 (0, self.eq_top - 40, w, h)]
        if self.title2_rect:
            avoid.append(self.title2_rect)
        self.logo_placements = (place_logos(logos, w, h, avoid, rng)
                                if logos else [])
        self.bar_phase = [rng.uniform(0, 1) for _ in range(12)]

    def _rect_around(self, bbox, center, grow: float = 1.0) -> Rect:
        tw = int((bbox[2] - bbox[0]) * grow)
        th = int((bbox[3] - bbox[1]) * grow)
        return (center[0] - tw // 2, center[1] - th // 2,
                center[0] + tw // 2, center[1] + th // 2)

    def _glow_text(self, img: Image.Image, text: str, font, center) -> None:
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = max(30, tw // 6)
        tile = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((pad - bbox[0], pad - bbox[1]), text,
                                  font=font, fill=TEXT + (255,))
        glow = tile.copy().filter(ImageFilter.GaussianBlur(max(6, tw // 40)))
        gc = Image.new("RGBA", glow.size, CYAN + (110,))
        gc.putalpha(glow.getchannel("A").point(lambda a: a // 2))
        gc2 = Image.new("RGBA", glow.size, MAGENTA + (90,))
        gc2.putalpha(glow.getchannel("A").point(lambda a: a // 3))
        x, y = center[0] - tile.width // 2, center[1] - tile.height // 2
        img.alpha_composite(gc2, (x + 6, y + 4))
        img.alpha_composite(gc, (x - 6, y - 4))
        img.alpha_composite(tile, (x, y))

    def frame(self, bg: Background, f: int, beat_sec: float) -> Image.Image:
        t = f / FPS
        img = bg.frame(t).convert("RGBA")
        draw_logos(img, self.logo_placements)
        self._glow_text(img, self.title, self.font1,
                        ((self.title1_rect[0] + self.title1_rect[2]) // 2,
                         (self.title1_rect[1] + self.title1_rect[3]) // 2))
        if self.title2:
            self._glow_text(img, self.title2, self.font2,
                            ((self.title2_rect[0] + self.title2_rect[2]) // 2,
                             (self.title2_rect[1] + self.title2_rect[3]) // 2))
        draw_chip(img, ((self.chip_rect[0] + self.chip_rect[2]) // 2,
                        (self.chip_rect[1] + self.chip_rect[3]) // 2),
                  self.cfg.date_line.upper(), self.chip_font, 1.0)
        draw = ImageDraw.Draw(img, "RGBA")
        # 12 equalizer bars bouncing with the beat pulse
        span = self.eq_bottom - self.eq_top
        n = 12
        total_w = n * self.eq_w + (n - 1) * self.eq_gap
        x0 = (self.w - total_w) // 2
        for j in range(n):
            phase = (t / beat_sec + self.bar_phase[j]) % 1.0
            pulse = math.exp(-6.0 * phase)
            bar_h = int(span * (0.22 + 0.70 * pulse))
            x = x0 + j * (self.eq_w + self.eq_gap)
            col = ACCENTS[j % 3]
            draw.rounded_rectangle([x, self.eq_bottom - bar_h,
                                    x + self.eq_w, self.eq_bottom],
                                   radius=self.eq_w // 2, fill=col + (235,))
        return img.convert("RGB")
