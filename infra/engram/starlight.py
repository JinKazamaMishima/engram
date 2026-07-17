"""aurora m6 — the living sky.

Pure, framework-free math for the Engram header's starfield. Two effects, one
discipline borrowed from the grok-build TUI ([[engram-tui-density-follows-function]]):
modulate the *brightness* of a held glyph with ``sin²`` blended toward the cell
background, so chrome *breathes* instead of strobing.

- ``breathe`` / ``header_band`` — every star holds its position and glyph across
  frames; only its brightness oscillates (per-star phase + period, so the field
  scintillates out of sync like a real sky, not a slot machine). Engram's own ``✦``
  rides a slower, higher-floor pulse — she shines while the field twinkles.
- ``_meteor_overlay`` — a one-shot shooting star that streaks the band diagonally
  when a turn completes, then vanishes. Driven by a wall-clock ``progress∈[0,1)``
  the caller owns, so there is no timer to leak.

Everything returns Rich-markup strings (``[#RRGGBB]g[/]``), so it is unit-testable
headless — no Textual, no Rich objects, no clock of its own (the caller passes the
elapsed seconds). ``app.py``'s header uses this exact code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Star glyphs by apparent magnitude — weighted toward the faint dot so the field
# reads as mostly dim sky with the occasional brighter point. Engram is always ``✦``.
_FIELD_GLYPHS = ("·", "·", "·", "˖", "⋆", "✧")
_ENGRAM_GLYPH = "✦"

# One star in ~this-many is Engram herself (steady, bright, slow).
_ENGRAM_EVERY = 8
# Roughly one star per this-many columns — a sparse, elegant night sky.
_DENSITY = 7

# One-shot meteor lifetime, seconds. Short and quick, like a real streak.
METEOR_SECS = 0.75
_TAIL_LEN = 5


@dataclass(frozen=True)
class SkyPalette:
    """Hex colors the sky blends between (owned by the app's theme, passed in)."""

    bg: str          # the header cell background — every star dims *toward* this
    dim: str         # faint stars
    lit: str         # brighter field stars
    engram: str        # Engram's own star (and the meteor's kin)
    meteor_head: str  # the shooting star's bright head
    meteor_tail: str  # its fading trail


# --- color ------------------------------------------------------------------

def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def blend_hex(bg: str, fg: str, t: float) -> str:
    """Linear sRGB blend ``bg → fg`` by ``t∈[0,1]``. ``t=0`` is ``bg``, ``t=1`` ``fg``."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    (r1, g1, b1), (r2, g2, b2) = _rgb(bg), _rgb(fg)
    return "#{:02X}{:02X}{:02X}".format(
        round(r1 + (r2 - r1) * t),
        round(g1 + (g2 - g1) * t),
        round(b1 + (b2 - b1) * t),
    )


# --- breathing --------------------------------------------------------------

def breathe_level(t: float, *, period: float, floor: float = 0.30, phase: float = 0.0) -> float:
    """``sin²`` brightness in ``[floor, 1]``.

    One full dim→bright→dim cycle per ``period`` seconds; ``phase`` (in cycles)
    desyncs elements. ``sin²`` never goes negative and is smooth at the trough —
    the calm, non-strobing breath.
    """
    s = math.sin(math.pi * (t / max(period, 1e-6) + phase))
    return floor + (1.0 - floor) * (s * s)


def breathe(bg: str, color: str, t: float, *, period: float,
            floor: float = 0.30, phase: float = 0.0) -> str:
    """``color`` blended toward ``bg`` by a breathing ``sin²`` level → the live hex."""
    return blend_hex(bg, color, breathe_level(t, period=period, floor=floor, phase=phase))


# --- starfield --------------------------------------------------------------

def _layout_seed(width: int, row: int) -> int:
    """Stable per-(width,row) seed: the layout must NOT change frame-to-frame — only
    brightness breathes with ``t``. Reshuffling per tick is the flicker we're killing."""
    return (width * 2654435761) ^ (row * 40503) ^ 0x9E3779B9


def _starfield_cells(width: int, row: int, t: float, palette: SkyPalette) -> list[str]:
    """``width`` markup cells for one row. Positions, glyphs, and per-star phase/period
    are seeded by ``(width, row)`` — deterministic across frames — so only the colors
    move. A ``deterministic`` local RNG keeps it pure (no global ``random`` state)."""
    cells = [" "] * max(0, width)
    if width <= 1:
        return cells
    import random  # local: keep module import-light and the global RNG untouched

    rng = random.Random(_layout_seed(width, row))
    count = min(width, max(2, width // _DENSITY))
    for i, col in enumerate(sorted(rng.sample(range(width), count))):
        phase = rng.random()
        is_engram = (rng.randrange(_ENGRAM_EVERY) == 0)
        if is_engram:
            glyph = _ENGRAM_GLYPH
            color = breathe(palette.bg, palette.engram, t, period=5.5, floor=0.55, phase=phase)
        else:
            glyph = _FIELD_GLYPHS[rng.randrange(len(_FIELD_GLYPHS))]
            base = palette.lit if rng.random() < 0.35 else palette.dim
            color = breathe(palette.bg, base, t,
                            period=rng.uniform(2.3, 4.6), floor=0.18, phase=phase)
        cells[col] = f"[{color}]{glyph}[/]"
    return cells


# --- meteor (one-shot) ------------------------------------------------------

def _meteor_overlay(width: int, rows: int, progress: float | None,
                    palette: SkyPalette) -> dict[tuple[int, int], str]:
    """Cells for a diagonal shooting star at ``progress∈[0,1)`` → ``{(row, col): markup}``.

    The head runs left→right and drifts down across ``rows``; a short tail fades to
    ``bg`` behind it and trails up-left. Past the right edge only the tail lingers,
    then it's gone. Empty dict when no meteor is flying."""
    if progress is None or not (0.0 <= progress < 1.0) or width <= 1 or rows < 1:
        return {}
    span = width + _TAIL_LEN
    head_x = progress * span
    head_row = progress * (rows - 1)
    out: dict[tuple[int, int], str] = {}
    for k in range(_TAIL_LEN + 1):
        col = round(head_x - k)
        if not (0 <= col < width):
            continue
        r = round(head_row - k * 0.5)
        if not (0 <= r < rows):
            continue
        frac = 1.0 - k / (_TAIL_LEN + 1)          # 1 at the head → faint at the tail
        color = blend_hex(palette.bg, palette.meteor_head if k == 0 else palette.meteor_tail, frac)
        glyph = "✦" if k == 0 else ("✧" if k <= 2 else "·")
        out[(r, col)] = f"[{color}]{glyph}[/]"
    return out


# --- the composited band ----------------------------------------------------

def header_band(width: int, rows: int, t: float, meteor_progress: float | None,
                palette: SkyPalette) -> list[str]:
    """The ``rows`` markup strings for the header's right field: the breathing
    starfield with the one-shot meteor (if any) overlaid on top."""
    if width <= 1:
        return [""] * rows
    band = [_starfield_cells(width, r, t, palette) for r in range(rows)]
    for (r, col), cell in _meteor_overlay(width, rows, meteor_progress, palette).items():
        band[r][col] = cell
    return ["".join(row) for row in band]
