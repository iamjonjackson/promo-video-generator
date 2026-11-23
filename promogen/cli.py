"""Command-line interface for promogen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import List, Optional

from PIL import Image, UnidentifiedImageError

from . import __version__
from .audio import estimate_bpm, ffmpeg_path, prepare_track
from .config import (DEFAULT_AUDIO, FPS, FONT_PATH, IMAGE_EXTS, LOGO_EXTS,
                     Config, apply_args, load_config_file)
from .errors import AudioMissing, BadImages, PromogenError
from .plan import Plan, build_plan
from .render import Renderer, ffmpeg_encode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m promogen",
        description="Generate a CapCut-style vertical (9:16) promo video "
                    "from a folder of photos.")
    p.add_argument("--images", required=True, metavar="DIR",
                   help="folder of photos (JPG/PNG/WebP/HEIC), folder order")
    p.add_argument("--sec-per-frame", type=float, metavar="FLOAT",
                   help="seconds per photo, excluding transitions (default 2.0)")
    p.add_argument("--audio", metavar="PATH",
                   help="soundtrack (mp3/wav/m4a); default assets/default-audio.mp3")
    p.add_argument("--no-audio", action="store_true",
                   help="render with a silent audio track instead of music")
    p.add_argument("--logos", metavar="DIR",
                   help="folder of transparent PNG logos (intro + outro only)")
    p.add_argument("--title", metavar="STR",
                   help='main title line (default "DIGITECH FEST")')
    p.add_argument("--title2", metavar="STR",
                   help='punch-in second line (default "FEST")')
    p.add_argument("--date-line", dest="date_line", metavar="STR",
                   help='date/chip line (default "NOVEMBER 2026")')
    p.add_argument("--footer", metavar="STR",
                   help="small top-center watermark (default: title uppercased)")
    p.add_argument("--outro-sec", type=float, metavar="FLOAT",
                   help="outro slide hold time before the fade (default 3.0)")
    p.add_argument("--seed", type=int, metavar="INT",
                   help="random seed; default: random per run")
    p.add_argument("--shuffle-photos", action="store_true",
                   help="randomize photo order (seeded)")
    p.add_argument("--bpm", metavar="FLOAT|auto",
                   help="visual beat grid BPM (default 120; 'auto' estimates "
                        "BPM from the audio)")
    p.add_argument("--out", metavar="PATH",
                   help="output .mp4 path (default output/promo-<seed>.mp4)")
    p.add_argument("--config", metavar="PATH",
                   help="flat key: value config file (applied before flags)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the manifest plan and exit without rendering")
    p.add_argument("--width", type=int, help="output width (default 1080)")
    p.add_argument("--height", type=int, help="output height (default 1920)")
    p.add_argument("--version", action="version",
                   version=f"promogen {__version__}")
    return p


# ----------------------------------------------------------------- inputs

def find_photos(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        raise BadImages(f"--images is not a directory: {images_dir!r}")
    names = sorted(os.listdir(images_dir),
                   key=lambda n: (n.lower(), n))
    paths = [os.path.join(images_dir, n) for n in names
             if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
    if not paths:
        raise BadImages(
            f"no JPG/PNG/WebP/HEIC images found in {images_dir!r}")
    return paths


def find_logos(logos_dir: str) -> List[Image.Image]:
    names = sorted(os.listdir(logos_dir),
                   key=lambda n: (n.lower(), n))
    paths = [os.path.join(logos_dir, n) for n in names
             if os.path.splitext(n)[1].lower() in LOGO_EXTS]
    if not paths:
        return []
    if len(paths) > 4:
        print(f"[logos] warning: {len(paths)} logos found; 4 are supported "
              "gracefully, continuing with the first 8")
    logos: List[Image.Image] = []
    for p in paths[:8]:
        try:
            img = Image.open(p)
            img.load()
        except (UnidentifiedImageError, OSError) as e:
            raise BadImages(f"cannot open logo {p!r}: {e}")
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        logos.append(img)
    return logos


def load_photos(paths: List[str]) -> List[str]:
    """Verify each photo is readable; the loader handles JPG/PNG/WebP natively."""
    for p in paths:
        try:
            img = Image.open(p)
            img.verify()
        except (UnidentifiedImageError, OSError) as e:
            base = os.path.basename(p).lower()
            if base.endswith((".heic", ".heif")):
                raise BadImages(
                    f"cannot read {p!r}: HEIC support requires the optional "
                    "extra `pip install pillow-heif`")
            raise BadImages(f"cannot open image {p!r}: {e}")
    return paths


def resolve_audio(cfg: Config, ffmpeg: str) -> Optional[str]:
    """Return audio source path or None for silence; raise if nothing usable."""
    if cfg.no_audio:
        return None
    if cfg.audio_path:
        if not os.path.isfile(cfg.audio_path):
            raise AudioMissing(f"--audio file not found: {cfg.audio_path!r}")
        return cfg.audio_path
    if os.path.isfile(DEFAULT_AUDIO):
        return DEFAULT_AUDIO
    raise AudioMissing(
        f"no soundtrack: pass --audio PATH, use --no-audio, or drop a file "
        f"at {DEFAULT_AUDIO!r} (see assets/README.md)")


def font_hash() -> str:
    h = hashlib.sha256()
    with open(FONT_PATH, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def probe_bpm(cfg: Config, audio: Optional[str], ffmpeg: str) -> float:
    if not cfg.bpm_auto:
        return float(cfg.bpm)
    if audio is None:
        print("[bpm] --bpm auto requested with --no-audio; falling back to 120")
        return 120.0
    bpm = estimate_bpm(audio, ffmpeg)
    print(f"[bpm] estimated {bpm:g} BPM from {os.path.basename(audio)!r}")
    return float(bpm)


# ------------------------------------------------------------- main flow

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config()
    if args.config:
        if not os.path.isfile(args.config):
            print(f"error: config file not found: {args.config!r}",
                  file=sys.stderr)
            return 2
        load_config_file(args.config, cfg)
    apply_args(cfg, args)

    try:
        return _run(cfg)
    except (PromogenError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[abort] interrupted; partial output cleaned up",
              file=sys.stderr)
        return 130


def _run(cfg: Config) -> int:
    ffmpeg = ffmpeg_path()      # prints install instructions when missing

    photo_paths = load_photos(find_photos(cfg.images_dir))
    logos = find_logos(cfg.logos_dir) if cfg.logos_dir else []

    seed = cfg.resolve_seed()
    rng = random.Random(seed)
    if cfg.shuffle_photos:
        rng.shuffle(photo_paths)

    audio_source = resolve_audio(cfg, ffmpeg)
    bpm = probe_bpm(cfg, audio_source, ffmpeg)

    plan = build_plan(cfg, seed, photo_paths, bpm, audio_source)
    out_path = cfg.out_path(seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    renderer = None
    if not cfg.dry_run:
        renderer = Renderer(plan, logos)
    manifest = manifest_for(plan, renderer, out_path)
    manifest["ffmpeg_version"] = _ffmpeg_version(ffmpeg)
    manifest["font_sha256"] = font_hash()
    print(f"seed: {seed}")

    if cfg.dry_run:
        print(json.dumps(manifest, indent=2, default=str))
        return 0

    audio_wav = prepare_track(audio_source, plan.total_sec, ffmpeg)
    sys.stdout.write(f"[plan] {len(plan.photos)} photos, "
                     f"{plan.total_frames} frames, "
                     f"{plan.total_sec:.2f}s @ {FPS} fps, bpm {plan.bpm:g}\n")

    removal = None
    try:
        def progress(rendered: int, elapsed: float):
            fps = rendered / elapsed if elapsed > 0 else 0.0
            eta = (plan.total_frames - rendered) / fps if fps > 0 else 0.0
            sys.stdout.write(
                f"\r[render] frame {rendered}/{plan.total_frames} "
                f"({fps:.1f} fps, eta {eta:.0f}s)   ")
            sys.stdout.flush()

        def frames_iterator():
            for f in range(plan.total_frames):
                yield renderer.frame(f)
                # keep only the active photo + neighbors decoded
                _evict_photos(renderer, f)

        removal = out_path
        ffmpeg_encode(plan, audio_wav, out_path, ffmpeg,
                      frames_iterator(), on_progress=progress)
    except BaseException:
        removal = removal or out_path
        try:
            if os.path.exists(removal):
                os.remove(removal)
        finally:
            if os.path.exists(audio_wav):
                os.unlink(audio_wav)
        raise
    if os.path.exists(audio_wav):
        os.unlink(audio_wav)

    with open(cfg.manifest_path(out_path), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print()
    print(f"[done] rendered {plan.total_frames} frames, "
          f"{plan.total_sec:.2f}s @ {FPS} fps -> {out_path}")
    print(f"[done] manifest: {cfg.manifest_path(out_path)}")
    return 0


def _evict_photos(renderer: Renderer, frame_idx: int) -> None:
    """Keep decoded photo caches bounded (active photo + neighbors)."""
    plan = renderer.plan
    idx = int((frame_idx / FPS - plan.intro_sec - 0.3) // plan.photo_sec)
    for i, slide in enumerate(renderer.slides):
        if abs(i - idx) > 1 and slide.cache is not None:
            slide.cache = None


def _ffmpeg_version(ffmpeg: str) -> str:
    import subprocess
    out = subprocess.run([ffmpeg, "-version"], capture_output=True,
                         text=True).stdout
    return out.splitlines()[0] if out else "unknown"


def manifest_for(plan: Plan, renderer: Optional[Renderer], out_path: str) -> dict:
    cfg = plan.cfg
    m: dict = {
        "seed": plan.seed,
        "config": {
            "images_dir": cfg.images_dir,
            "logos_dir": cfg.logos_dir,
            "sec_per_frame": cfg.sec_per_frame,
            "outro_sec": cfg.outro_sec,
            "title": cfg.title,
            "title2": cfg.title2,
            "date_line": cfg.date_line,
            "footer": cfg.footer,
            "shuffle_photos": cfg.shuffle_photos,
            "width": cfg.width,
            "height": cfg.height,
            "bpm": plan.bpm,
        },
        "timeline": {
            "intro_sec": plan.intro_sec,
            "photo_sec": plan.photo_sec,
            "outro_hold_sec": plan.outro_sec,
            "fade_sec": plan.fade_sec,
            "transition_sec": plan.transition_sec,
            "total_sec": round(plan.total_sec, 4),
            "total_frames": plan.total_frames,
            "fps": FPS,
            "bpm": plan.bpm,
            "beat_sec": round(plan.beat_sec, 6),
        },
        "out_path": out_path,
        "audio": plan.audio_source or "silence",
        "photos": [
            {
                "path": spec.path,
                "effect": spec.effect,
                "motion": {
                    "kind": spec.motion.kind,
                    "scale0": round(spec.motion.scale0, 4),
                    "scale1": round(spec.motion.scale1, 4),
                    "anchor0": [round(spec.motion.ax0, 4), round(spec.motion.ay0, 4)],
                    "anchor1": [round(spec.motion.ax1, 4), round(spec.motion.ay1, 4)],
                },
            }
            for spec in plan.photos
        ],
        "outro_effect": plan.outro_effect,
    }
    if renderer is not None:
        m["logo_placements"] = {
            "intro": [
                {"x": p.x, "y": p.y, "opacity": p.opacity,
                 "size": [p.image.width, p.image.height]}
                for p in renderer.intro.logo_placements
            ],
            "outro": [
                {"x": p.x, "y": p.y, "opacity": p.opacity,
                 "size": [p.image.width, p.image.height]}
                for p in renderer.outro.logo_placements
            ],
        }
    return m


if __name__ == "__main__":
    raise SystemExit(main())
