#!/usr/bin/env python3
"""Hermetic unit tests for the GrokDriver (broker m4 — Engram running ON Grok 4.5).

No network, no key, no spend — ``xai_common`` and the recall preamble are stubbed.
Covers persistent multi-turn history, the tool-call loop with the operator-supervised
write tools (write_file / edit_file / bash), Event kinds, effort/reset, and fail-open
paths (no key, transport error → error Event with history left consistent).

    .venv/bin/python infra/engram/tests/test_grok_driver.py
"""
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import envoy  # noqa: E402  (after sys.path shim, like the sibling tests)
import grok_driver  # noqa: E402
import xai_common  # noqa: E402
from grok_driver import GrokDriver  # noqa: E402


def _tc(name, args, ticks=1_000_000):
    return {"model": "grok-4.5", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"cost_in_usd_ticks": ticks}}


def _final(text, ticks=1_000_000):
    return {"model": "grok-4.5",
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"cost_in_usd_ticks": ticks}}


@contextlib.contextmanager
def _patched(responses, *, key="k", inject=""):
    o_post, o_key = xai_common.post_json, xai_common.load_key
    o_inj, o_inp = envoy._recall_inject, envoy._inprocess_tools
    it = iter(responses)

    def fake_post(path, payload, key_, timeout):
        return next(it)

    xai_common.post_json = fake_post
    xai_common.load_key = lambda: key
    envoy._recall_inject = lambda *a, **k: inject
    envoy._inprocess_tools = lambda cwd: []
    try:
        yield
    finally:
        xai_common.post_json, xai_common.load_key = o_post, o_key
        envoy._recall_inject, envoy._inprocess_tools = o_inj, o_inp


async def _drain(agen):
    return [ev async for ev in agen]


# --- persistent conversation -------------------------------------------------
async def test_persistent_history_across_turns():
    with _patched([(_final("Hi the operator."), None), (_final("Still here."), None)]):
        d = GrokDriver(cwd=".")
        evs1 = await _drain(d.query("hello"))
        await _drain(d.query("again?"))
    assert any(e.kind == "text" and "Hi the operator." in e.text for e in evs1)
    roles = [m["role"] for m in d._messages]
    assert roles[0] == "system"
    assert roles.count("user") == 2          # both turns landed in ONE history
    assert roles.count("assistant") == 2


async def test_reset_clears_history():
    d = GrokDriver(cwd=".")
    d._messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    d.reset()
    assert d._messages == []


async def test_set_effort_validates():
    d = GrokDriver(cwd=".")
    await d.set_effort("high")
    assert d.effort == "high"
    await d.set_effort("bogus")              # invalid ignored, not stored
    assert d.effort == "high"


# --- driver selection (broker m5: which backend does /model target?) ---------
def test_is_grok_detection():
    """The one predicate both surfaces read to swap AgentSDKDriver ↔ GrokDriver."""
    from core import _is_grok, _model_family
    assert _is_grok("grok-4.5") and _is_grok("grok-4.5-fast") and _is_grok("GROK")
    assert _model_family("grok-4.5") == "grok"                 # header/family display too
    for claude in ("opus[1m]", "opus", "sonnet", "fable", "haiku", "claude-opus-4-8"):
        assert not _is_grok(claude), claude
    assert not _is_grok(None) and not _is_grok("")


# --- the supervised write tools ---------------------------------------------
async def test_write_file_executor():
    with tempfile.TemporaryDirectory() as d:
        out = await grok_driver._write_file({"path": "a/b.txt", "content": "x"}, Path(d))
        assert out.startswith("[wrote]") and Path(d, "a/b.txt").read_text() == "x"
        denied = await grok_driver._write_file({"path": "../esc", "content": "x"}, Path(d))
        assert denied.startswith("[denied]")


async def test_edit_file_executor_uniqueness_guard():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, "f.txt")
        p.write_text("one two two")
        assert (await grok_driver._edit_file(
            {"path": "f.txt", "old": "one", "new": "1"}, Path(d))).startswith("[edited]")
        assert p.read_text() == "1 two two"
        nonuniq = await grok_driver._edit_file({"path": "f.txt", "old": "two", "new": "2"}, Path(d))
        assert "not unique" in nonuniq
        assert (await grok_driver._edit_file(
            {"path": "f.txt", "old": "two", "new": "2", "replace_all": True},
            Path(d))).startswith("[edited]")
        assert p.read_text() == "1 2 2"
        assert "not found" in await grok_driver._edit_file(
            {"path": "f.txt", "old": "zzz", "new": "q"}, Path(d))


async def test_bash_executor():
    with tempfile.TemporaryDirectory() as d:
        assert "hi" in await grok_driver._bash({"command": "echo hi"}, Path(d))
        assert (await grok_driver._bash({"command": ""}, Path(d))).startswith("[error]")


async def test_write_tool_loop_end_to_end():
    """Grok requests write_file, the driver executes it for real and feeds the result
    back, then Grok concludes. Verifies dispatch + tool Event + real side effect."""
    with tempfile.TemporaryDirectory() as d:
        with _patched([(_tc("write_file", {"path": "out.txt", "content": "hello"}), None),
                       (_final("Done — wrote out.txt."), None)]):
            drv = GrokDriver(cwd=d)
            evs = await _drain(drv.query("write hello to out.txt"))
        assert Path(d, "out.txt").read_text() == "hello"
        assert any(e.kind == "tool" and e.text == "write_file" for e in evs)
        assert any(e.kind == "text" and "Done" in e.text for e in evs)
        # the tool result was fed back before the final answer
        assert any(m.get("role") == "tool" and "[wrote]" in (m.get("content") or "")
                   for m in drv._messages)


async def test_full_toolset_has_read_and_write():
    with tempfile.TemporaryDirectory() as d:
        with _patched([]):
            specs, dispatch = grok_driver._full_toolset(Path(d))
        names = {s["function"]["name"] for s in specs}
        assert {"read_file", "grep", "glob"} <= names          # envoy read layer reused
        assert {"write_file", "edit_file", "bash"} <= names     # supervised writes added


# --- memory + fail-open ------------------------------------------------------
async def test_recall_injected_per_turn():
    with _patched([(_final("ok"), None)], inject="MEMORY BLOCK"):
        d = GrokDriver(cwd=".")
        evs = await _drain(d.query("hi"))
    assert any(e.kind == "recall" for e in evs)
    assert any(m["role"] == "system" and "MEMORY BLOCK" in m["content"] for m in d._messages)


async def test_no_key_yields_error_event():
    with _patched([], key=None):
        d = GrokDriver(cwd=".")
        evs = await _drain(d.query("hi"))
    assert any(e.kind == "text" and "no xAI key" in e.text for e in evs)


async def test_transport_error_fails_open_history_consistent():
    with _patched([(None, "boom")]):
        d = GrokDriver(cwd=".")
        evs = await _drain(d.query("hi"))
    assert any(e.kind == "text" and "boom" in e.text for e in evs)
    assert d._messages[-1]["role"] == "user"      # turn recorded, conversation intact


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
