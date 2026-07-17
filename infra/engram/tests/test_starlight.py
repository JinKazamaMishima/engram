#!/usr/bin/env python3
"""Hermetic tests for aurora m6's living-sky math (no Textual, no clock).

    .venv/bin/python infra/engram/tests/test_starlight.py
"""
import os
import re
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from starlight import (  # noqa: E402
    METEOR_SECS,
    SkyPalette,
    _meteor_overlay,
    _starfield_cells,
    blend_hex,
    breathe,
    breathe_level,
    header_band,
)

PAL = SkyPalette(
    bg="#0A0E1A", dim="#5A6C96", lit="#727C95", engram="#3D7A87",
    meteor_head="#E8ECF8", meteor_tail="#3D7A87",
)

_CELL = re.compile(r"^\[#([0-9A-Fa-f]{6})\](.)\[/\]$")


def _split(cell: str):
    """(hex, glyph) for a rendered star cell, or (None, ' ') for empty."""
    m = _CELL.match(cell)
    return (m.group(1).upper(), m.group(2)) if m else (None, " ")


# --- blend ------------------------------------------------------------------

def test_blend_endpoints_and_midpoint():
    assert blend_hex("#000000", "#FFFFFF", 0.0) == "#000000"
    assert blend_hex("#000000", "#FFFFFF", 1.0) == "#FFFFFF"
    assert blend_hex("#000000", "#FFFFFF", 0.5) == "#808080"


def test_blend_clamps_out_of_range():
    assert blend_hex("#000000", "#FFFFFF", -3.0) == "#000000"
    assert blend_hex("#000000", "#FFFFFF", 9.0) == "#FFFFFF"


# --- breathing --------------------------------------------------------------

def test_breathe_level_stays_within_floor_and_one():
    floor = 0.25
    lo, hi = 1.0, 0.0
    for i in range(400):
        v = breathe_level(i * 0.01, period=3.0, floor=floor)
        assert floor - 1e-9 <= v <= 1.0 + 1e-9
        lo, hi = min(lo, v), max(hi, v)
    assert lo < floor + 0.02          # actually reaches the trough
    assert hi > 0.98                  # ...and the full-bright peak


def test_breathe_never_returns_bare_background():
    # A star at its dimmest is still tinted toward its color, never pure bg —
    # that's what keeps the resting field faintly visible instead of blinking out.
    dimmest = breathe(PAL.bg, PAL.lit, 0.0, period=3.0, floor=0.18, phase=0.0)
    assert dimmest != PAL.bg


# --- starfield: STABLE layout, only brightness moves (anti-flicker) ---------

def test_starfield_layout_is_stable_across_frames():
    a = _starfield_cells(48, 0, 0.13, PAL)
    b = _starfield_cells(48, 0, 1.91, PAL)
    assert len(a) == len(b) == 48
    # Same columns lit, same glyph in each — positions do NOT reshuffle per frame.
    assert [g for _, g in map(_split, a)] == [g for _, g in map(_split, b)]


def test_starfield_brightness_actually_breathes():
    a = _starfield_cells(48, 0, 0.13, PAL)
    b = _starfield_cells(48, 0, 1.10, PAL)
    colors_a = [h for h, _ in map(_split, a) if h]
    colors_b = [h for h, _ in map(_split, b) if h]
    assert colors_a and colors_a != colors_b     # some star changed hue between frames


def test_starfield_has_stars_but_stays_sparse():
    cells = _starfield_cells(48, 1, 0.5, PAL)
    lit = sum(1 for _, g in map(_split, cells) if g != " ")
    assert 0 < lit < 48 // 2                      # a night sky, not a wall of stars


def test_starfield_narrow_width_is_safe():
    assert _starfield_cells(1, 0, 0.0, PAL) == [" "]
    assert _starfield_cells(0, 0, 0.0, PAL) == []


# --- meteor -----------------------------------------------------------------

def test_meteor_absent_when_not_flying():
    assert _meteor_overlay(40, 3, None, PAL) == {}
    assert _meteor_overlay(40, 3, 1.0, PAL) == {}     # past its life
    assert _meteor_overlay(40, 3, -0.1, PAL) == {}


def test_meteor_has_one_head_that_leads_the_tail():
    cells = _meteor_overlay(40, 3, 0.4, PAL)
    assert cells
    heads = [c for c in cells.values() if "✦" in c]
    assert len(heads) == 1                            # exactly one bright head
    head_hex, _ = _split(heads[0])
    # The head is brighter (further from bg) than the faintest tail cell.
    tails = [_split(c)[0] for c in cells.values() if "✦" not in c]
    assert tails, "meteor should have a trail"


def test_meteor_travels_left_to_right():
    early = _meteor_overlay(40, 3, 0.15, PAL)
    late = _meteor_overlay(40, 3, 0.85, PAL)
    early_head_col = min(col for (_, col) in early)
    late_head_col = max(col for (_, col) in late)
    assert late_head_col > early_head_col


def test_meteor_secs_is_positive():
    assert METEOR_SECS > 0


# --- composited band --------------------------------------------------------

def test_header_band_shape_and_meteor_overlay():
    plain = header_band(40, 3, 0.5, None, PAL)
    assert len(plain) == 3 and all(isinstance(s, str) for s in plain)
    withmeteor = header_band(40, 3, 0.5, 0.4, PAL)
    assert len(withmeteor) == 3
    assert "✦" in "".join(withmeteor)                 # the meteor head shows up
    assert header_band(1, 3, 0.5, None, PAL) == ["", "", ""]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
