import pytest

from promogen.config import FPS
from promogen.effects import EFFECT_NAMES
from promogen.plan import build_plan, snap_to_beat, beat_sec_for, photo_boundaries
from promogen.errors import PromogenError


def test_snap_to_beat_multiples():
    beat = 0.5
    assert snap_to_beat(2.0, beat) == 2.0
    assert snap_to_beat(1.9, beat) == 2.0
    assert snap_to_beat(1.24, beat) == 1.0
    assert snap_to_beat(0.2, beat) == 0.5      # never below one beat


def test_beat_sec():
    assert beat_sec_for(120) == pytest.approx(0.5)
    assert beat_sec_for(60) == pytest.approx(1.0)


def _plan(sec_per_frame=2.0, outro=3.0, bpm=120.0, n=3):
    from promogen.config import Config
    cfg = Config(images_dir="x", sec_per_frame=sec_per_frame,
                outro_sec=outro, bpm=bpm)
    return build_plan(cfg, 42, [f"p{i}.png" for i in range(n)], bpm, None)


def test_duration_formula():
    p = _plan(n=4, sec_per_frame=2.0, outro=3.0)
    # intro ~2 + N*sec + outro + fade(0.6)
    assert p.intro_sec == pytest.approx(2.0)
    assert p.photo_sec == pytest.approx(2.0)
    assert p.outro_sec == pytest.approx(3.0)
    total = p.total_sec
    assert total == pytest.approx(2.0 + 4 * 2.0 + 3.0 + 0.6)
    assert p.total_frames == int(round(total * FPS))
    # every boundary lands on a beat multiple
    for b in photo_boundaries(p):
        k = round(b / p.beat_sec)
        assert abs(b - k * p.beat_sec) < 1e-9


def test_snap_logged_value():
    p = _plan(sec_per_frame=1.7, bpm=120.0)   # 1.7 -> 1.5 (3 beats)
    assert p.photo_sec == pytest.approx(1.5)


def test_effect_choices_no_immediate_repeat():
    p = _plan(n=10)
    effects = [s.effect for s in p.photos] + [p.outro_effect]
    assert all(e in EFFECT_NAMES for e in effects)
    for a, b in zip(effects, effects[1:]):
        assert a != b, "same effect twice in a row"


def test_short_photos_rejected():
    with pytest.raises(PromogenError):
        _plan(sec_per_frame=0.2, bpm=200.0)    # beat 0.3s -> below 0.45s guard
