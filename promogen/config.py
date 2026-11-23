"""Configuration dataclass and minimal config.yaml loading."""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, fields
from typing import Optional

FONT_NAME = "DejaVuSans-Bold.ttf"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
FONT_PATH = os.path.join(ASSETS_DIR, "fonts", FONT_NAME)
DEFAULT_AUDIO = os.path.join(ASSETS_DIR, "default-audio.mp3")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
LOGO_EXTS = {".png"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}

FPS = 30


@dataclass
class Config:
    images_dir: str = ""
    audio_path: Optional[str] = None
    no_audio: bool = False
    logos_dir: Optional[str] = None
    sec_per_frame: float = 2.0
    outro_sec: float = 3.0
    title: str = "DIGITECH FEST"
    title2: str = "FEST"
    date_line: str = "NOVEMBER 2026"
    footer: Optional[str] = None
    seed: Optional[int] = None
    out: Optional[str] = None
    shuffle_photos: bool = False
    bpm: float = 120.0
    bpm_auto: bool = False
    dry_run: bool = False
    width: int = 1080
    height: int = 1920
    config_file: Optional[str] = None

    def resolve_seed(self) -> int:
        if self.seed is None:
            self.seed = random.SystemRandom().randrange(0, 2**31 - 1)
        return self.seed

    def footer_text(self) -> str:
        return self.footer if self.footer is not None else self.title.upper()

    def out_path(self, seed: int) -> str:
        if self.out:
            return self.out
        return os.path.join("output", f"promo-{seed}.mp4")

    def manifest_path(self, out_path: str) -> str:
        root, _ = os.path.splitext(out_path)
        return root + ".manifest.json"


def _coerce(value: str, current):
    """Coerce a YAML-ish string to the type of the current default."""
    if isinstance(current, bool):
        v = value.strip().lower()
        return v in ("1", "true", "yes", "on")
    if isinstance(current, float):
        return float(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if current is None:
        return value
    return value


def load_config_file(path: str, cfg: Config) -> None:
    """Load a flat `key: value` config file into an existing Config.

    Minimal on purpose (no PyYAML dependency): one `key: value` pair per line,
    `#` comments allowed, string values may be quoted.
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            value = value.strip().strip("'\"")
            mapping[key.strip()] = value
    valid = {f.name: f.type for f in fields(Config)}
    for key, value in mapping.items():
        if key not in valid:
            raise ValueError(f"unknown config key in {path}: {key!r}")
        setattr(cfg, key, _coerce(value, getattr(cfg, key)))
    if isinstance(cfg.bpm, str):
        if cfg.bpm.strip().lower() == "auto":
            cfg.bpm_auto = True
            cfg.bpm = 120.0
        else:
            cfg.bpm = float(cfg.bpm)
    cfg.config_file = path


def apply_args(cfg: Config, args: argparse.Namespace) -> None:
    """Apply explicit (non-None) CLI arguments on top of a config-file-loaded Config.

    Boolean flags are argparse store_true booleans, so only True matters.
    """
    flag_to_field = {
        "images": "images_dir",
        "audio": "audio_path",
        "date_line": "date_line",
        "outro_sec": "outro_sec",
        "logos": "logos_dir",
        "config": "config_file",
    }
    for f in fields(Config):
        flag = f.name
        for attr, fld in flag_to_field.items():
            if fld == f.name:
                flag = attr
        if not hasattr(args, flag):
            continue
        value = getattr(args, flag)
        if isinstance(getattr(cfg, f.name), bool):
            if value:
                setattr(cfg, f.name, True)
        elif value is not None:
            setattr(cfg, f.name, value)
    if isinstance(cfg.bpm, str):
        s = cfg.bpm.strip().lower()
        if s == "auto":
            cfg.bpm_auto = True
            cfg.bpm = 120.0
        else:
            cfg.bpm = float(cfg.bpm)
