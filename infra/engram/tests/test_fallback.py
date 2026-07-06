#!/usr/bin/env python3
"""Headless tests for fallback-model SURFACING: the family classifier and the
active-fallback detector (actual-vs-primary across the alias/id boundary). The
_options() wiring (default Opus 4.8, same-family skip, env override) lives in
test_options.py.

    .venv/bin/python infra/engram/tests/test_fallback.py
"""
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(ENGRAM, "..", "..")), "src"))

from core import AgentSDKDriver, _model_family  # noqa: E402


def _driver(model="fable"):
    return AgentSDKDriver(model=model, store=None, buffer_dir=False)


def test_model_family():
    assert _model_family("fable") == "fable"
    assert _model_family("claude-fable-5") == "fable"
    assert _model_family("opus[1m]") == "opus"
    assert _model_family("claude-opus-4-8") == "opus"
    assert _model_family("claude-sonnet-4-6") == "sonnet"
    assert _model_family("haiku") == "haiku"
    assert _model_family(None) is None
    assert _model_family("some-unknown-model") is None
    print("✓ _model_family maps aliases AND resolved ids to a family, None otherwise")


def test_active_fallback_detects_cross_family_only():
    d = _driver(model="fable")
    assert d.active_fallback is None                 # no turn yet → no actual model
    d.actual_model = "claude-fable-5"                # SDK reports the primary
    assert d.active_fallback is None                 # same family → not a fallback
    d.actual_model = "claude-opus-4-8"               # rotated to Opus
    assert d.active_fallback == "claude-opus-4-8"    # cross-family → surfaced verbatim
    d.actual_model = "something-weird"               # unclassifiable → don't cry wolf
    assert d.active_fallback is None
    print("✓ active_fallback fires only on a real cross-family rotation")


def main() -> int:
    test_model_family()
    test_active_fallback_detects_cross_family_only()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
