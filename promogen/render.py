"""Rendering pipeline: frames from the Plan, piped raw into ffmpeg."""
from __future__ import annotations

import random
import subprocess
import time
from typing import List, Optional

import numpy as np
from PIL import Image

from .config import FPS
from .effects import EFFECTS, EffectsContext
from .plan import Plan, PhotoSpec, photo_boundaries
from .slides import Background, IntroPlanner, OutroPlanner
from .style import VIGNETTE_CORNER, beat_pulse, clamp01, lerp, smoothstep

CACHE_ZOOM = 1.35       # headroom above max Ken Burns scale (~1.22 * 1.016)
TRANSITION_HALF = 0.2


def vignette_mask(w: int, h: int) -> np.ndarray:
    """Multiplicative mask falling to ~70% gray at the corners."""
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    k = 1.0 - VIGNETTE_CORNER
    mask = 1.0 - k * np.clip(r, 0, 1) ** 2
    return mask.astype(np.float32)


class PhotoSlide:
    """One photo rendered with Ken Burns + beat bump + vignette."""

    def __init__(self, spec: PhotoSpec, w: int, h: int):
        self.spec = spec
        self.w, self.h = w, h
        self.cache: Optional[Image.Image] = None
        self.vmask = vignette_mask(w, h)

    def _load(self):
        if self.cache is None:
            img = Image.open(self.spec.path)
            try:
                img.load()
            except OSError:
                raise
            scale = max(self.w * CACHE_ZOOM / img.width,
                        self.h * CACHE_ZOOM / img.height)
            cw, ch = int(img.width * scale), int(img.height * scale)
            self.cache = img.convert("RGB").resize((cw, ch), Image.BILINEAR)
        return self.cache

    def frame(self, local_t: float, dur: float, pulse: float) -> Image.Image:
        cache = self._load()
        m = self.spec.motion
        q = smoothstep(local_t / dur if dur > 0 else 1.0)
        if m.kind == "pan":
            s = lerp(m.scale0, m.scale1, q)
        else:
            s = lerp(m.scale0, m.scale1, q)
        s *= (1.0 + 0.016 * pulse)
        s = min(s, CACHE_ZOOM * 0.999)
        cw, ch = cache.width, cache.height
        crop_w = min(cw, max(1, int(cw / s)))
        crop_h = min(ch, max(1, int(ch / s)))
        x0 = int((clamp01(lerp(m.ax0, m.ax1, q))) * (cw - crop_w))
        y0 = int((clamp01(lerp(m.ay0, m.ay1, q))) * (ch - crop_h))
        crop = cache.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        out = crop.resize((self.w, self.h), Image.BILINEAR)
        arr = np.asarray(out, dtype=np.float32) * self.vmask[:, :, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class Renderer:
    """Renders frame N of the promo video from a Plan."""

    def __init__(self, plan: Plan, logos: List[Image.Image]):
        self.plan = plan
        cfg = plan.cfg
        self.w, self.h = cfg.width, cfg.height
        self.rng = random.Random(plan.seed ^ 0x5EED)   # per-frame jitter rng
        self.bg = Background(self.w, self.h, random.Random(plan.seed ^ 0xC0F))
        self.logos = logos
        intro_frames = int(round(plan.intro_sec * FPS))
        self.intro = IntroPlanner(cfg, self.w, self.h, intro_frames,
                                  random.Random(plan.seed ^ 0x1A), logos)
        self.outro = OutroPlanner(cfg, self.w, self.h,
                                  random.Random(plan.seed ^ 0x0E),
                                  plan.beat_sec, logos)
        self.slides = [PhotoSlide(spec, self.w, self.h)
                       for spec in plan.photos]
        self.intro_frames = intro_frames
        self.outro_start = plan.intro_sec + len(plan.photos) * plan.photo_sec
        self.fade_frames = int(round(plan.fade_sec * FPS))
        self.total_frames = plan.total_frames
        self._boundaries = photo_boundaries(plan)
        # effects list: entry i is the transition INTO photo i; last is outro
        self._transitions = [spec.effect for spec in plan.photos]
        if plan.photos:
            self._transitions.append(plan.outro_effect)

    # ------------------------------------------------------------ frames

    def _photo_frame(self, i: int, f: int) -> Image.Image:
        plan = self.plan
        start = plan.intro_sec + i * plan.photo_sec
        local_t = max(0.0, f / FPS - start)
        pulse = beat_pulse(f / FPS, plan.beat_sec)
        img = self.slides[i].frame(local_t, plan.photo_sec, pulse)
        return img

    def _outro_frame(self, f: int) -> Image.Image:
        return self.outro.frame(self.bg, f, self.plan.beat_sec)

    def _intro_frame(self, f: int) -> Image.Image:
        return self.intro.frame(self.bg, f)

    def frame(self, f: int) -> Image.Image:
        plan = self.plan
        t = f / FPS
        effects = self._transitions
        n = len(self.slides)
        half = TRANSITION_HALF
        for bi, boundary in enumerate(self._boundaries):
            if boundary - half <= t < boundary + half:
                idx_out = bi - 1                     # -1 = intro, n = outro
                idx_in = bi
                if idx_out < 0:
                    out_getter = lambda: self._intro_frame(f)
                elif idx_out <= n - 1:
                    out_getter = lambda io=idx_out: self._photo_frame(io, f)
                else:
                    out_getter = lambda: self._outro_frame(f)
                if idx_in <= n - 1:
                    in_getter = lambda ii=idx_in: self._photo_frame(ii, f)
                else:
                    in_getter = lambda: self._outro_frame(f)
                p = clamp01((t - (boundary - half)) / (2 * half))
                ctx = EffectsContext((self.w, self.h), self.rng,
                                      plan.seed, bi, FPS,
                                      plan.transition_sec)
                return EFFECTS[effects[bi]](out_getter, in_getter, p, f, ctx)
        if t < plan.intro_sec:
            return self._intro_frame(f)
        if t < self.outro_start:
            idx = min(n - 1, int((t - plan.intro_sec) // plan.photo_sec))
            return self._photo_frame(idx, f)
        img = self._outro_frame(f)
        # fade to black over the tail of the outro
        fade_start_t = self.outro_start + plan.outro_sec
        if t > fade_start_t:
            p = clamp01((t - fade_start_t) / plan.fade_sec)
            arr = np.asarray(img, dtype=np.uint8).astype(np.float32) * (1.0 - p)
            img = Image.fromarray(arr.astype(np.uint8))
        return img


def ffmpeg_encode(plan: Plan, audio_wav: str, out_path: str, ffmpeg: str,
                 frames_iter, on_progress=None) -> None:
    """Encode: raw RGB frames -> H.264 high@4.1 + AAC 192k mp4 (+faststart)."""
    w, h = plan.cfg.width, plan.cfg.height
    vf = (f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,"
          f"atrim=0:{plan.total_sec:.6f}")
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-i", audio_wav,
        "-filter_complex", f"[1:a]{vf}[aud]",
        "-map", "0:v", "-map", "[aud]",
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.1",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", f"{plan.total_sec:.6f}",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    t0 = time.time()
    rendered = 0
    try:
        for img in frames_iter:
            data = img.convert("RGB").tobytes()
            proc.stdin.write(data)
            rendered += 1
            if on_progress and rendered % 30 == 0:
                on_progress(rendered, time.time() - t0)
        proc.stdin.close()
    except BrokenPipeError:
        stderr = proc.stderr.read().decode("utf-8", "replace")
        raise RuntimeError(f"ffmpeg failed early:\n{stderr[-2000:]}")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    stderr = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}:\n{stderr[-2000:]}")
    if on_progress:
        on_progress(rendered, time.time() - t0)
