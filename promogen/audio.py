"""Audio preparation: decode, loop with crossfade, fade out, BPM estimate.

Loudness normalization to ~-14 LUFS happens in the final ffmpeg call via the
``loudnorm`` filter (see render.py); everything else is numpy so run output is
deterministic.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import wave
from typing import Optional, Tuple

import numpy as np

from .errors import FfmpegMissing

SAMPLE_RATE = 44100
CROSSFADE_SEC = 0.2
FADE_OUT_SEC = 2.5


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FfmpegMissing(
            "ffmpeg was not found on PATH.\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   apt-get install ffmpeg   (or your distro's package manager)\n"
            "  Windows: winget install Gyan.FFmpeg   (or download from ffmpeg.org)"
        )
    return path


def decode(path: str, ffmpeg: str) -> Tuple[np.ndarray, int]:
    """Decode an audio file to float32 stereo [-1, 1] via ffmpeg raw output."""
    cmd = [
        ffmpeg, "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2",
        "-ar", str(SAMPLE_RATE), "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if arr.size % 2:
        arr = arr[:-1]
    return arr.reshape(-1, 2), SAMPLE_RATE


def loop_to(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Loop signal to target_len samples with ~200ms crossfaded loop joints."""
    n = sig.shape[0]
    if n == 0:
        return np.zeros((target_len, sig.shape[1]), dtype=np.float32)
    if n >= target_len:
        return sig[:target_len].copy()
    reps = math.ceil(target_len / n)
    out = np.concatenate([sig] * reps, axis=0)[:target_len].copy()
    xf = min(int(CROSSFADE_SEC * SAMPLE_RATE), n // 4)
    if xf > 0:
        fade_out = np.cos(np.linspace(0, math.pi / 2, xf)) ** 2
        fade_in = 1.0 - fade_out
        for joint in range(n, target_len, n):
            end = min(joint + xf, target_len)
            L = end - joint
            if L <= 0:
                break
            out[joint:end] = (
                sig[n - L:n] * fade_out[-L:][:, None]
                + sig[:L] * fade_in[-L:][:, None]).astype(np.float32)
    return out


def fade_out_tail(sig: np.ndarray, seconds: float) -> np.ndarray:
    """Apply a linear fade over the last `seconds` of the signal."""
    if seconds <= 0:
        return sig
    n = sig.shape[0]
    xf = min(int(seconds * SAMPLE_RATE), n)
    if xf <= 0:
        return sig
    out = sig.copy()
    ramp = np.linspace(1.0, 0.0, xf, dtype=np.float32)[:, None]
    out[n - xf:] *= ramp
    return out


def write_wav(path: str, sig: np.ndarray) -> None:
    """Write float stereo [-1, 1] as a 16-bit PCM stereo 44.1k wav."""
    clipped = np.clip(sig, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def prepare_track(source: Optional[str], duration_sec: float,
                   ffmpeg: str) -> str:
    """Return a temp wav path: looped/trimmed to duration, faded out, silent
    if source is None (--no-audio)."""
    import tempfile

    if source is None:
        sig = np.zeros((int(duration_sec * SAMPLE_RATE), 2), dtype=np.float32)
    else:
        decoded, _ = decode(source, ffmpeg)
        target = int(duration_sec * SAMPLE_RATE)
        sig = fade_out_tail(loop_to(decoded, target), FADE_OUT_SEC)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="promogen_audio_")
    import os
    os.close(fd)
    write_wav(path, sig)
    return path


def estimate_bpm(path: str, ffmpeg: str,
                  lo: float = 90.0, hi: float = 180.0) -> float:
    """Estimate BPM from onset-energy autocorrelation (numpy only)."""
    sig, sr = decode(path, ffmpeg)
    mono = sig.mean(axis=1)
    hop = int(sr / 200.0)                     # 200 fps onset envelope
    win = 512
    n = mono.shape[0] // hop
    if n < 400:
        raise ValueError("audio too short for BPM estimation")
    frames = np.lib.stride_tricks.sliding_window_view(mono, win)[::hop][:n]
    energy = (frames ** 2).mean(axis=1)
    onset = np.diff(energy, prepend=energy[0])
    onset = np.clip(onset, 0, None)
    onset -= onset.mean()
    onset = np.maximum(onset, 0)
    fps = sr / hop
    lag_lo = max(2, int(60.0 / hi * fps))
    lag_hi = min(n - 1, int(60.0 / lo * fps))
    if lag_hi <= lag_lo:
        return 120.0
    ac = np.correlate(onset, onset, mode="full")[n - 1:]
    ac = ac[: lag_hi + 1]
    if ac[0] != 0:
        ac = ac / ac[0]
    best_lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi + 1]))
    bpm = 60.0 * fps / best_lag
    # fold into [lo, hi]
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return round(float(bpm), 1)
