# assets/

Place your default soundtrack here as `default-audio.mp3`.

- Formats: mp3, wav or m4a. If `assets/default-audio.mp3` is missing you can
  still pass `--audio PATH` or `--no-audio` (silent track).
- Nothing in this directory is committed except the bundled font in
  `assets/fonts/DejaVuSans-Bold.ttf`.
- For offline test runs, `python scripts/make_test_assets.py assets` generates
  placeholder gradient photos and a 30s 120 BPM synth wav.
