#!/usr/bin/env python3
"""Hermetic tests for the aurora activity indicator's pure render (no Textual run).

    .venv/bin/python infra/engram/tests/test_activity.py
"""
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from app import (  # noqa: E402
    _ACT_DELEG,
    _ACT_READ,
    _ACT_SHELL,
    _ACT_TALK,
    _ACT_WRITE,
    SPINNER,
    _activity_color,
    render_activity,
)


def test_idle_is_blank():
    assert render_activity("", 0) == ""
    assert render_activity("   ", 7) == ""


def test_color_by_tool_category():
    # Assert the tool→color MAPPING via the constants, not literal hexes, so palette
    # tweaks (e.g. the night-dim pass) don't break the test.
    assert _activity_color("Bash") == _ACT_SHELL                      # shell = amber
    assert _activity_color("Read") == _ACT_READ                      # read = cyan
    assert _activity_color("Grep") == _ACT_READ
    assert _activity_color("Edit") == _ACT_WRITE                     # write = green
    assert _activity_color("Agent→Explore: find callers") == _ACT_DELEG  # delegate = violet
    assert _activity_color("Workflow→deep-research") == _ACT_DELEG
    assert _activity_color("MysteryTool") == _ACT_TALK              # default = blue


def test_thinking_and_responding_colors():
    assert _ACT_DELEG in render_activity("thinking", 0)              # violet
    assert _ACT_TALK in render_activity("responding", 0)            # blue


def test_spinner_advances_with_frame():
    a = render_activity("Bash", 0)
    b = render_activity("Bash", 1)
    assert a != b                                                    # glyph changes
    assert SPINNER[0] in a and SPINNER[1] in b


def test_label_shown_and_truncated():
    assert "Read" in render_activity("Read", 0)
    long = "Agent→Explore: " + "x" * 80
    assert "…" in render_activity(long, 0)                           # truncated


def test_glint_is_occasional():
    # the cyan ✦ glint fires on some frames, not all — proves it's a periodic accent
    glints = [("✦" in render_activity("Bash", f)) for f in range(34)]
    assert any(glints) and not all(glints)


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("---", "all green" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
