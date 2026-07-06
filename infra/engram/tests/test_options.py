#!/usr/bin/env python3
"""Unit tests for driver option plumbing: the fallback_model rotation and the
Event.data payload channel (both backward-compatible additions).

    .venv/bin/python infra/engram/tests/test_options.py
"""
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from core import AgentSDKDriver, Event  # noqa: E402


def test_fallback_model_default_is_opus():
    # Cross-family primary (fable) → the default Opus 4.8 fallback applies.
    d = AgentSDKDriver(store=None, model="fable")
    assert d._options().fallback_model == "claude-opus-4-8", d._options().fallback_model
    print("✓ fallback_model defaults to Opus 4.8 (fable primary)")


def test_fallback_skipped_when_same_family():
    # An opus-primary session must NOT 'fall back' to Opus — same family is
    # pointless and the CLI rejects a fallback equal to the primary. This covers
    # the DEFAULT model (opus[1m]) too.
    for m in ("opus[1m]", "claude-opus-4-8"):
        d = AgentSDKDriver(store=None, model=m)
        assert d._options().fallback_model is None, f"{m}: same-family must be skipped"
    print("✓ same-family fallback skipped (opus primary → no opus fallback)")


def test_fallback_env_override_and_disable():
    os.environ["ENGRAM_FALLBACK_MODEL"] = "haiku"
    try:
        d = AgentSDKDriver(store=None, model="fable")
        assert d._options().fallback_model == "haiku"
        os.environ["ENGRAM_FALLBACK_MODEL"] = ""          # explicit disable
        assert d._options().fallback_model is None
    finally:
        del os.environ["ENGRAM_FALLBACK_MODEL"]
    print("✓ ENGRAM_FALLBACK_MODEL overrides; empty disables")


def test_event_data_channel_backward_compatible():
    plain = Event("text", "hello")                      # every existing call site
    assert plain.data is None and plain.kind == "text" and plain.text == "hello"
    rich = Event("todos", "", data={"todos": [{"content": "x", "status": "pending"}]})
    assert rich.data["todos"][0]["content"] == "x"
    print("✓ Event grows a data payload; existing two-arg constructors untouched")


def main() -> int:
    test_fallback_model_default_is_opus()
    test_fallback_skipped_when_same_family()
    test_fallback_env_override_and_disable()
    test_event_data_channel_backward_compatible()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
