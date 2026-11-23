# promogen

A local CLI tool that turns a folder of photos into a CapCut-style vertical
(9:16) promo video — dark neon-gradient background, big chromatic-aberration
title cards, beat-synced cuts, punchy random transitions, and a branded, timed
outro. Fully parameterized, so it can brand any event, not just a festival.

Pure Python (Pillow + numpy) + ffmpeg. No browser, no headless Chrome, no
playwright.

```
python -m promogen --images myphotos/ [options]
```

## Quickstart

```bash
python3 -m pip install -r requirements.txt      # pillow + numpy (ffmpeg must be on PATH)
python scripts/make_test_assets.py myphotos 5    # optional: gradient placeholders + 120 BPM wav
cp myphotos/track.wav assets/default-audio.mp3   # any mp3/wav/m4a soundtrack
python3 -m promogen --images myphotos/ --title "MY EVENT" --date-line "JUNE 2027"
```

Output lands at `output/promo-<seed>.mp4` with a `promo-<seed>.manifest.json`
next to it.

## Requirements

- Python 3.11+
- `pillow`, `numpy` (see `requirements.txt`)
- **ffmpeg on PATH** — checked at startup with install instructions:
  - macOS: `brew install ffmpeg`
  - Linux: `apt-get install ffmpeg` (or your distro's package manager)
  - Windows: `winget install Gyan.FFmpeg` (or download a build from ffmpeg.org)

HEIC photos need the optional extra `pip install pillow-heif`.

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--images DIR` | required | Folder of photos (JPG/PNG/WebP/HEIC). Order = folder order (case-insensitive by name); `--shuffle-photos` randomizes. |
| `--sec-per-frame FLOAT` | `2.0` | Seconds each photo stays on screen. Excludes transition overlap (transitions overlap neighbors, they don't add time). Snapped to the nearest beat multiple; the adjusted value is logged. |
| `--audio PATH` | `assets/default-audio.mp3` | Custom soundtrack (mp3/wav/m4a). If the default file is missing and no `--audio` is passed, promogen exits with a clear error unless `--no-audio` is given. |
| `--no-audio` | off | Render with a silent track instead of music. |
| `--logos DIR` | none | One or more transparent PNG logos. Placed randomly (seeded) on the intro and outro slides only — never over photos. 4 are supported gracefully; more triggers a warning (up to 8 used). |
| `--title STR` | `DIGITECH FEST` | Main title line. |
| `--title2 STR` | `FEST` | Punch-in second line. |
| `--date-line STR` | `NOVEMBER 2026` | Text in the pill chip. |
| `--footer STR` | title uppercased | Small watermark, top-center. |
| `--outro-sec FLOAT` | `3.0` | Outro slide hold time before a ~0.6s fade to black. A real, timed slide — the video never ends on the last photo. |
| `--seed INT` | random per run | All randomness derives from this seed. The seed used is printed and recorded in the manifest; re-running with `--seed X` reproduces a byte-identical video. |
| `--shuffle-photos` | off | Randomize photo order (seeded). |
| `--bpm FLOAT\|auto` | `120` | Visual beat grid. `auto` estimates BPM from the audio (onset-energy autocorrelation, numpy only) and prints it. |
| `--out PATH` | `output/promo-<seed>.mp4` | Output video path. Manifest is written next to it as `<out>.manifest.json`. |
| `--config PATH` | none | Flat `key: value` config file (same names as flags without `--`), applied before any CLI flags. |
| `--dry-run` | off | Print the manifest plan (timeline, per-photo effects/motions, logo layout) without rendering. |
| `--width/--height` | `1080/1920` | Output resolution (advanced; used by the test suite). |

The manifest records the seed, config, per-photo effect + Ken Burns anchors,
logo placements, computed duration/BPM, font SHA-256 and ffmpeg version.

## Video structure

- **Intro (~2s):** letter-by-letter title reveal with out-back overshoot and
  RGB chromatic ghosts, second title line punch-in, date pill chip, top-center
  footer from frame 0, drifting 45° stripes, plus randomized logos.
- **Photos:** each cover-cropped to 9:16 (never letterbox, never distort) with
  one random Ken Burns move (zoom-in 1.02→~1.18, zoom-out ~1.22→1.02, or pan
  between two anchors, smoothstep-eased), a beat-synced scale bump
  (`1 + 0.016 × pulse`), and a mild vignette. A ~0.4s transition centered on
  every boundary: flash, diagonal wipe, glitch, circles, rows, mosaic, swipe,
  zoom punch, dots — chosen uniformly with no immediate repeats.
- **Outro:** brand text with soft glow, the date chip, 12 equalizer bars
  bouncing with the beat (cyan/magenta/yellow), logos. Holds `--outro-sec`,
  then fades to black over ~0.6s as part of the rendered video.

Total duration = intro + N × photo_sec + outro + fade. Video is 30 fps
H.264 (High profile, level 4.1, yuv420p, `+faststart`) + AAC 192k in .mp4.

## Audio handling

The track is decoded in-process (ffmpeg raw PCM), looped with a ~200ms equal-power
crossfade at the loop point if shorter than the video, faded out over the last
~2.5s, then normalized to ~-14 LUFS by ffmpeg's `loudnorm` filter during the
final mux (AAC 192k).

## Reproducibility

One `random.Random(seed)` instance is threaded through every choice (transitions,
Ken Burns, logo layout, particles, jitter) — no bare `random.*` calls.
Determinism notes:

- The bundled font `assets/fonts/DejaVuSans-Bold.ttf` is used everywhere (font
  substitution is the usual reproducibility killer); its SHA-256 is recorded in
  the manifest.
- Same machine + same ffmpeg build → same-seed reruns are byte-identical
  (verified by the integration test). ffmpeg version is recorded in the
  manifest; a different ffmpeg build may encode slightly different bytes with
  identical visual output.

## Adding a custom transition

`promogen/effects.py` holds a registry of small, individually testable
functions with a common signature. To add a new effect:

1. Open `promogen/effects.py` and add a function, decorated with `@effect`:

   ```python
   @effect
   def my_wipe(out_getter, in_getter, progress, frame_index, ctx):
       """One-sentence description of the effect."""
       a = np.asarray(out_getter().convert("RGB"))   # outgoing frame
       b = np.asarray(in_getter().convert("RGB"))    # incoming frame
       t = progress                                    # 0..1 across ~0.4s
       mixed = (a * (1 - t) + b * t).clip(0, 255).astype(np.uint8)
       return Image.fromarray(mixed)
   ```

   - `out_getter()` / `in_getter()` return full-frame PIL images (outgoing and
     incoming slides, each still animating — call them once per frame).
   - `progress` is 0..1 across the transition window; `frame_index` is the
     global frame number (use for per-frame jitter).
   - `ctx.size` is `(width, height)`; `ctx.rng` is the seeded per-frame rng for
     jitter; `ctx.tr_rng()` is a deterministic per-transition rng for stable
     structures (row orders, cell shuffles).

2. That's it — the `@effect` decorator makes it eligible for random selection
   immediately, and `tests/test_effects.py::test_registry` automatically runs
   your effect over dummy images and asserts a correct-size RGB result.

3. Optionally re-run the tests: `python -m pytest tests/test_effects.py`.

## Tests

```bash
python -m pytest tests/ -q
```

Fast unit tests cover beat-snapping math, the duration formula, cover-crop
math, logo collision/overlap rejection, and every registered transition over
dummy images. One integration smoke test renders a tiny 270x480, 3-photo video
and asserts a valid mp4 with video + audio streams and byte-identical output on
a same-seed rerun (skipped automatically when ffmpeg is missing).

## Offline assets

`scripts/make_test_assets.py OUT_DIR [N]` generates gradient placeholder
photos and a 30s 120 BPM click-ish synth wav (numpy), so tests and first runs
work offline. The repo ships no sample images and no audio.

## Font

The repo bundles DejaVu Sans Bold (`assets/fonts/DejaVuSans-Bold.ttf`, a free
font under its permissive license) for stable, reproducible title rendering.

## Repo layout

```
promogen/          cli.py, plan.py, render.py, effects.py, slides.py,
                  audio.py, layout.py, config.py, style.py, errors.py
assets/           fonts/ (bundled) — drop default-audio.mp3 here (see assets/README.md)
scripts/          make_test_assets.py
tests/            unit + integration tests
```
