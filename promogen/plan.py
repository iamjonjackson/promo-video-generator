"""Deterministic plan: beat-snapping math, timeline, transition/KenBurns choices.

The plan is built once (all rng consumed here, from one random.Random(seed)),
then rendering is a pure function of (plan, frame_index).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config, FPS
from .effects import EFFECT_NAMES
from .errors import PromogenError


def beat_sec_for(bpm: float) -> float:
    return 60.0 / bpm


def snap_to_beat(seconds: float, beat: float) -> float:
    """Snap a duration to the nearest nonzero multiple of beat_sec."""
    if beat <= 0:
        return seconds
    return max(beat, round(seconds / beat) * beat)


@dataclass
class KenBurns:
    kind: str                  # "zoom_in" | "zoom_out" | "pan"
    scale0: float
    scale1: float
    ax0: float
    ay0: float
    ax1: float
    ay1: float


@dataclass
class PhotoSpec:
    path: str
    effect: str                # transition INTO this photo
    motion: KenBurns


@dataclass
class Plan:
    cfg: Config
    seed: int
    bpm: float
    beat_sec: float
    intro_sec: float
    photo_sec: float
    outro_sec: float
    fade_sec: float
    transition_sec: float
    photos: List[PhotoSpec] = field(default_factory=list)
    outro_effect: str = ""
    total_sec: float = 0.0
    total_frames: int = 0
    photo_paths: List[str] = field(default_factory=list)
    logo_placements: dict = field(default_factory=dict)
    audio_source: Optional[str] = None


def choose_motions(rng: random.Random) -> KenBurns:
    kind = rng.choice(("zoom_in", "zoom_out", "pan"))
    if kind == "zoom_in":
        s0 = 1.02
        s1 = rng.uniform(1.14, 1.18)
    elif kind == "zoom_out":
        s0 = rng.uniform(1.18, 1.22)
        s1 = 1.02
    else:
        s0 = 1.10
        s1 = 1.12

    def anchor(_rng):
        slack_lo = (s0 - 1.0) / 2.0 * 0.8
        return _rng.uniform(max(0.0, 0.5 - slack_lo), min(1.0, 0.5 + slack_lo))

    return KenBurns(kind, s0, s1, anchor(rng), anchor(rng), anchor(rng), anchor(rng))


def build_plan(cfg: Config, seed: int, photo_paths: List[str],
               bpm: float, audio_source: Optional[str]) -> Plan:
    rng = random.Random(seed)
    beat = beat_sec_for(bpm)

    intro_sec = snap_to_beat(2.0, beat)
    photo_sec = snap_to_beat(cfg.sec_per_frame, beat)
    if abs(photo_sec - cfg.sec_per_frame) > 1e-9:
        print(f"[beat] snapped --sec-per-frame {cfg.sec_per_frame:.3f}s "
              f"-> {photo_sec:.3f}s (nearest beat multiple at {bpm:g} BPM)")
    outro_sec = snap_to_beat(cfg.outro_sec, beat)
    if photo_sec < 0.45:
        raise PromogenError(
            f"snapped --sec-per-frame is {photo_sec:.2f}s, too short for the "
            "0.4s transition overlap; use a larger --sec-per-frame or a lower --bpm")
    fade_sec = 0.6
    transition_sec = 0.4

    photos: List[PhotoSpec] = []
    prev_effect = None
    for p in photo_paths:
        names = [n for n in EFFECT_NAMES if n != prev_effect]
        eff = rng.choice(names)
        prev_effect = eff
        photos.append(PhotoSpec(p, eff, choose_motions(rng)))
    outro_eff = ""
    if photos:
        names = [n for n in EFFECT_NAMES if n != prev_effect]
        outro_eff = rng.choice(names)

    n = len(photos)
    total_sec = intro_sec + n * photo_sec + outro_sec + fade_sec
    total_frames = int(round(total_sec * FPS))

    return Plan(
        cfg=cfg, seed=seed, bpm=bpm, beat_sec=beat,
        intro_sec=intro_sec, photo_sec=photo_sec, outro_sec=outro_sec,
        fade_sec=fade_sec, transition_sec=transition_sec,
        photos=photos, outro_effect=outro_eff, total_sec=total_sec,
        total_frames=total_frames, photo_paths=photo_paths,
        audio_source=audio_source,
    )


def photo_boundaries(plan: Plan) -> List[float]:
    """Cut boundaries (seconds): intro->p1, p1->p2, ..., pN->outro."""
    b = [plan.intro_sec + i * plan.photo_sec for i in range(len(plan.photos) + 1)]
    return b


