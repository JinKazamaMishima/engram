#!/usr/bin/env python3
"""Hermetic tests for aurora m4 — the context/provenance meter + compaction telemetry.

Pure-render tests (no Textual run) for ``render_context_meter`` / ``_meter_bar``, plus
the provider-agnostic driver seam: the base ``ModelDriver`` default, ``AgentSDKDriver``'s
PreCompact counter bump (which must never veto compaction), and ``GrokDriver`` reporting
xAI usage into the same shape the gauge reads. No network, no GPU, no creds.

    .venv/bin/python infra/engram/tests/test_context_meter.py
"""
import os
import re
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from app import _MET_FLOOR, _MET_WARN, _meter_bar, render_context_meter  # noqa: E402
from core import AgentSDKDriver, ModelDriver  # noqa: E402
from grok_driver import CONTEXT_WINDOW, GrokDriver  # noqa: E402


# --- pure render: fail-open + shape -----------------------------------------
def test_meter_blank_without_usage():
    """No usable usage → '' so the cell stays blank and the chrome quiets."""
    assert render_context_meter({}, 0, 0) == ""
    assert render_context_meter(None, 500, 1) == ""
    assert render_context_meter({"totalTokens": 0, "rawMaxTokens": 0}, 0, 0) == ""
    assert render_context_meter({"totalTokens": 100, "rawMaxTokens": 0}, 0, 0) == ""


def test_meter_renders_percent_and_bar():
    out = render_context_meter(
        {"totalTokens": 100_000, "rawMaxTokens": 1_000_000,
         "isAutoCompactEnabled": True}, 4_000, 0)
    assert "🧠" in out and "10%" in out          # brain glyph, not a "ctx" label
    assert "⣿" in out and "⣀" in out            # braille used/free bar was drawn


def test_meter_floor_clamped_over_total():
    """A floor larger than the whole window must clamp, not overflow or crash."""
    out = render_context_meter(
        {"totalTokens": 5_000, "rawMaxTokens": 1_000_000,
         "isAutoCompactEnabled": True}, 999_999, 0)
    assert out and "🧠" in out


def test_meter_compaction_badge():
    base = {"totalTokens": 100_000, "rawMaxTokens": 1_000_000, "isAutoCompactEnabled": True}
    assert "⎇" not in render_context_meter(base, 0, 0)     # no badge at zero
    assert "⎇3" in render_context_meter(base, 0, 3)        # badge shows the count


def test_meter_no_autocompact_warning_when_full():
    """Grok-shaped: auto-compact OFF + high fill → the rose no-net warning fires;
    a LOW fill on the same netless backend must stay calm (no false alarm)."""
    full = {"totalTokens": 220_000, "rawMaxTokens": 256_000, "isAutoCompactEnabled": False}
    out = render_context_meter(full, 3_000, 0)
    assert "no auto-compact" in out and _MET_WARN in out
    low = {"totalTokens": 5_000, "rawMaxTokens": 256_000, "isAutoCompactEnabled": False}
    assert "no auto-compact" not in render_context_meter(low, 3_000, 0)


def test_meter_claude_full_stays_calm():
    """Auto-compact ON (Claude) → even near-full shows no no-net warning: the net saves it."""
    out = render_context_meter(
        {"totalTokens": 950_000, "rawMaxTokens": 1_000_000, "isAutoCompactEnabled": True},
        4_000, 0)
    assert "no auto-compact" not in out


def test_meter_bar_is_pure_and_bounded():
    assert _meter_bar(0, 0, 0) == ""                       # no window → blank
    bar = _meter_bar(2, 5, 10, width=10)
    cells = re.sub(r"\[/?[^\]]*\]", "", bar)               # strip Rich markup
    assert len(cells) == 10 and set(cells) <= {"⣿", "⣀"}   # braille full/low-rail cells
    assert _MET_FLOOR in bar                                # green re-derived head present


# --- provider-agnostic seam --------------------------------------------------
def test_base_driver_has_zero_count():
    """The TUI reads compaction_count/last_compaction_ts off ANY driver, no hasattr guard."""
    assert ModelDriver().compaction_count == 0
    assert ModelDriver().last_compaction_ts is None


async def test_precompact_bumps_count_and_never_vetoes():
    """Every compaction is counted (before the fragile curation path) and the hook
    still returns {} — it can never block or steer compaction."""
    d = AgentSDKDriver(store=None)               # store-less: no curation, still counts
    assert d.compaction_count == 0
    assert await d._on_precompact({}, None, {}) == {}
    assert await d._on_precompact({"session_id": "s"}, None, {}) == {}
    assert d.compaction_count == 2
    assert d.last_compaction_ts is not None


async def test_grok_reports_usage_shape():
    """GrokDriver normalizes xAI usage into the gauge's shape — blank until the first
    response, then prompt_tokens vs the window, honestly netless."""
    d = GrokDriver(cwd=".")
    assert await d.get_context_usage() == {}               # nothing sent yet → blank
    d._last_usage = {"prompt_tokens": 64_000, "completion_tokens": 500}
    u = await d.get_context_usage()
    assert u["totalTokens"] == 64_000
    assert u["rawMaxTokens"] == CONTEXT_WINDOW
    assert u["isAutoCompactEnabled"] is False              # honest: Grok has no net
    assert abs(u["percentage"] - 100.0 * 64_000 / CONTEXT_WINDOW) < 1e-6
    assert "🧠" in render_context_meter(u, 3_000, 0)       # feeds the meter end-to-end


if __name__ == "__main__":
    import asyncio
    import inspect
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                asyncio.run(fn()) if inspect.iscoroutinefunction(fn) else fn()
                print(f"ok   {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("---", "all green" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
