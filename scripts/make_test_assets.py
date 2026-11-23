#!/usr/bin/env python3
"""Generate gradient placeholder photos + a 30s 120 BPM click-ish synth wav.

Usage:
    python scripts/make_test_assets.py OUT_DIR [N_PHOTOS]

Creates OUT_DIR/photo-01.png ... and OUT_DIR/track.wav (30s, 44.1kHz, 120 BPM).
Drop that wav (renamed as default-audio.mp3 is not needed: wav is accepted via
--audio) into assets/ or pass it explicitly. No third-party deps beyond
pillow + numpy.
"""
from __future__ import annotations

import math
import os
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw

SR = 44100
BPM = 120.0
DURATION = 30.0


def make_photo(path: str, idx: int, w: int = 1080, h: int = 1920) -> None:
    t = idx / 7.0
    top = (int(128 + 120 * math.sin(t)), int(90 + 120 * math.cos(t * 1.3)), 200)
    bottom = (30, int(60 + 150 * math.sin(t * 0.7)), int(120 + 100 * math.cos(t)))
    yy = np.linspace(0, 1, h)[:, None]
    col = (np.array(top)[None, :] * (1 - yy) + np.array(bottom)[None, :] * yy)
    arr = np.repeat(col[:, None, :], w, axis=1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    d.text((80, 120 + 40 * idx), f"PLACEHOLDER {idx:02d}", fill=(255, 255, 255))
    img.save(path)


def make_wav(path: str) -> None:
    n = int(SR * DURATION)
    t = np.arange(n) / SR
    beat = 60.0 / BPM
    sig = np.zeros(n, dtype=np.float32)
    # soft pad chord
    for f in (110.0, 164.81, 220.0):
        sig += 0.12 * np.sin(2 * np.pi * f * t)
    # kick-ish click on each beat
    k = np.arange(int(0.12 * SR)) / SR
    click = np.sin(2 * np.pi * 60 * k) * np.exp(-k * 22)
    for b in range(int(DURATION / beat)):
        i = int(b * beat * SR)
        sig[i:i + len(click)] += click * 0.8
    sig *= 0.7
    pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "assets"
    n_photos = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, n_photos + 1):
        make_photo(os.path.join(out_dir, f"photo-{i:02d}.png"), i)
    make_wav(os.path.join(out_dir, "track.wav"))
    print(f"wrote {n_photos} photos + track.wav to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
