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


def test_fallback_model_default():
    d = AgentSDKDriver(store=None)                      # model defaults to opus[1m]
    assert d._options().fallback_model == "sonnet", d._options().fallback_model
    print("✓ fallback_model defaults to sonnet")


def test_fallback_skipped_when_equal_to_primary():
    d = AgentSDKDriver(store=None, model="sonnet")
    assert d._options().fallback_model is None, "fallback == primary must be skipped"
    print("✓ fallback skipped when it equals the primary model")


def test_fallback_env_override_and_disable():
    os.environ["ENGRAM_FALLBACK_MODEL"] = "haiku"
    try:
        d = AgentSDKDriver(store=None)
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
    test_fallback_model_default()
    test_fallback_skipped_when_equal_to_primary()
    test_fallback_env_override_and_disable()
    test_event_data_channel_backward_compatible()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
