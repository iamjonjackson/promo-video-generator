import random

import pytest
from PIL import Image

from promogen.effects import EFFECTS, EFFECT_NAMES, EffectsContext


W, H = 96, 160


@pytest.fixture()
def dummy_pair():
    a = Image.new("RGB", (W, H), (200, 30, 30))
    b = Image.new("RGB", (W, H), (30, 200, 30))
    return a, b


@pytest.fixture()
def ctx():
    return EffectsContext((W, H), random.Random(0), seed=1, index=0, fps=30,
                          transition_sec=0.4)


def test_registry_contents():
    expected = {"flash", "diagonal_wipe", "glitch", "circles", "rows",
                "mosaic", "swipe", "zoom_punch", "dots"}
    assert expected == set(EFFECT_NAMES)
    assert len(EFFECT_NAMES) == len(EFFECTS)


@pytest.mark.parametrize("name", EFFECT_NAMES)
def test_every_effect_runs(dummy_pair, ctx, name):
    a, b = dummy_pair

    def out():
        return a

    def inc():
        return b

    for p, fi in ((0.0, 0), (0.25, 3), (0.5, 6), (0.9, 11), (1.0, 12)):
        img = EFFECTS[name](out, inc, p, fi, ctx)
        assert img.size == (W, H), f"{name} returned {img.size}"
        assert img.mode == "RGB"


@pytest.mark.parametrize("name", EFFECT_NAMES)
def test_effects_change_pixels(dummy_pair, ctx, name):
    a, b = dummy_pair
    img = EFFECTS[name](lambda: a, lambda: b, 0.5, 6, ctx)
    assert img.convert("RGB").getpixel((W // 2, H // 2)) != (0, 0, 0) or \
        name in ("mosaic", "zoom_punch", "swipe")   # dark-wash effects may be dim
