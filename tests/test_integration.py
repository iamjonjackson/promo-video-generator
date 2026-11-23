"""Integration smoke test: render a tiny 3-photo video end to end.

Skipped when ffmpeg is missing. Uses 270x480 @ 2s beatable settings so it runs
in a couple of seconds.
"""
import os
import shutil
import subprocess
import sys

import pytest

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(ffmpeg is None, reason="ffmpeg not on PATH")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_assets(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_test_assets as mta
    assets = tmp_path / "assets"
    assets.mkdir()
    for i in range(1, 4):
        mta.make_photo(str(assets / f"photo-{i:02d}.png"), i, w=270, h=480)
    wav = assets / "track.wav"
    mta.make_wav(str(wav))
    return assets


def test_tiny_render(tmp_path):
    from promogen.cli import main
    from promogen.config import FONT_PATH

    assets = _make_assets(tmp_path)
    assert os.path.isfile(FONT_PATH)   # bundled font must ship with the repo

    out = tmp_path / "out.mp4"
    rc = main([
        "--images", str(assets),
        "--audio", str(assets / "track.wav"),
        "--sec-per-frame", "0.5",
        "--outro-sec", "0.5",
        "--seed", "123",
        "--width", "270",
        "--height", "480",
        "--out", str(out),
    ])
    assert rc == 0, "cli main returned nonzero"
    assert out.exists() and out.stat().st_size > 10_000

    # manifest written next to the output
    manifest = str(out).rsplit(".", 1)[0] + ".manifest.json"
    assert os.path.exists(manifest)

    # probe: valid mp4 with video + audio streams
    probe = subprocess.run([ffmpeg, "-i", str(out)], capture_output=True,
                           text=True).stderr
    assert "Duration: 00:00:04" in probe
    assert "Video: h264" in probe
    assert "Audio: aac" in probe

    # reproducibility: same seed -> byte-identical output
    out2 = tmp_path / "out2.mp4"
    main([
        "--images", str(assets),
        "--audio", str(assets / "track.wav"),
        "--sec-per-frame", "0.5",
        "--outro-sec", "0.5",
        "--seed", "123",
        "--width", "270",
        "--height", "480",
        "--out", str(out2),
    ])
    h1 = _sha(out)
    h2 = _sha(out2)
    assert h1 == h2, f"same seed produced different bytes: {h1} != {h2}"


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()
