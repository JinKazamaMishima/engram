#!/usr/bin/env python3
"""Hermetic unit tests for the envoy (broker m3 — Engram's Grok 4.5 research subagent).

No network, no key, no spend — ``xai_common.post_json`` / ``load_key`` and the recall
preamble are stubbed. Covers the tool-call loop (dispatch → feed result back → answer),
the read-only native executors + workspace path safety, JSON-Schema conversion, the
model/steps/cost footer, and every fail-open path (no key, empty task, transport error,
step-limit truncation).

    .venv/bin/python infra/engram/tests/test_envoy.py
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
import xai_common  # noqa: E402


def _tc_resp(name, args, ticks=5_000_000):
    """A chat/completions response that requests one tool call."""
    return {"model": "grok-4.5", "choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}}],
        "usage": {"cost_in_usd_ticks": ticks}}


def _final_resp(text, ticks=3_000_000):
    """A chat/completions response that ends the loop with a text answer."""
    return {"model": "grok-4.5",
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"cost_in_usd_ticks": ticks}}


@contextlib.contextmanager
def _patched(responses, captured, *, key="k"):
    """Stub the xAI transport with a scripted response queue, and neutralise the
    recall preamble + in-process toolset so the loop is tested in isolation."""
    o_post, o_key = xai_common.post_json, xai_common.load_key
    o_inj, o_inp = envoy._recall_inject, envoy._inprocess_tools
    it = iter(responses)

    def fake_post(path, payload, key_, timeout):
        captured.append(payload)
        return next(it)

    xai_common.post_json = fake_post
    xai_common.load_key = lambda: key
    envoy._recall_inject = lambda *a, **k: ""
    envoy._inprocess_tools = lambda cwd: []
    try:
        yield
    finally:
        xai_common.post_json, xai_common.load_key = o_post, o_key
        envoy._recall_inject, envoy._inprocess_tools = o_inj, o_inp


# --- pure helpers ------------------------------------------------------------
def test_json_schema():
    s = envoy._json_schema({"query": str, "k": int})
    assert s["type"] == "object"
    assert s["properties"]["query"] == {"type": "string"}
    assert s["properties"]["k"] == {"type": "integer"}
    assert envoy._json_schema({})["properties"] == {}


def test_mcp_text():
    assert envoy._mcp_text({"content": [{"type": "text", "text": "hi"}]}) == "hi"
    assert envoy._mcp_text({"content": [], "is_error": True}).startswith("[error]")
    assert envoy._mcp_text({"content": []}) == "(no output)"


def test_format_footer():
    f = envoy._format(envoy.EnvoyResult(text="answer", cost_usd=0.012,
                                        model="grok-4.5", steps=3))
    assert "answer" in f and "[envoy ·" in f
    assert "model grok-4.5" in f and "3 steps" in f and "$0.012" in f
    assert "1 step" in envoy._format(envoy.EnvoyResult(text="x", steps=1))   # singular
    assert "truncated" in envoy._format(
        envoy.EnvoyResult(text="x", truncated=True, steps=2))
    assert "(envoy returned no text)" in envoy._format(envoy.EnvoyResult())


# --- native read-only executors + path safety --------------------------------
async def test_read_file_reads():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "a.txt").write_text("content here")
        assert await envoy._read_file({"path": "a.txt"}, Path(d)) == "content here"
        assert (await envoy._read_file({"path": "missing"}, Path(d))).startswith("[not found]")


async def test_read_file_denied_outside_workspace():
    with tempfile.TemporaryDirectory() as d:
        out = await envoy._read_file({"path": "../../../etc/passwd"}, Path(d))
        assert out.startswith("[denied]")


async def test_glob_lists_workspace_matches():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "x.py").write_text("x")
        Path(d, "y.txt").write_text("y")
        out = await envoy._glob({"pattern": "*.py"}, Path(d))
        assert "x.py" in out and "y.txt" not in out
        assert await envoy._glob({"pattern": ""}, Path(d)) == "[error] glob: empty pattern"


async def test_grep_guards():
    with tempfile.TemporaryDirectory() as d:
        assert (await envoy._grep({"pattern": ""}, Path(d))).startswith("[error]")
        denied = await envoy._grep({"pattern": "x", "path": "../../.."}, Path(d))
        assert denied.startswith("[denied]")


# --- the agentic loop --------------------------------------------------------
async def test_full_tool_loop():
    """The core mechanic: Grok requests read_file, we execute it, feed the result
    back, and Grok answers off that result. Verifies dispatch + result feed-back +
    cost accumulation + advertised toolset."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "note.txt").write_text("hello from the file")
        cap: list = []
        with _patched([(_tc_resp("read_file", {"path": "note.txt"}), None),
                       (_final_resp("The note says hello."), None)], cap):
            r = await envoy.run_envoy("what's in note.txt?", cwd=d)
        assert r.ok and r.text == "The note says hello."
        assert r.steps == 2 and not r.truncated
        assert round(r.cost_usd, 6) == 0.008          # 5e6 + 3e6 nano-USD ticks
        # the file contents were fed back as a tool message on the 2nd call
        second = cap[1]["messages"]
        assert any(m.get("role") == "tool" and "hello from the file" in (m.get("content") or "")
                   for m in second)
        # the first call advertised the native read-only toolset + auto tool choice
        assert cap[0]["tool_choice"] == "auto"
        names = {t["function"]["name"] for t in cap[0]["tools"]}
        assert {"read_file", "grep", "glob"} <= names


async def test_unknown_tool_is_a_result_not_a_crash():
    with tempfile.TemporaryDirectory() as d:
        cap: list = []
        with _patched([(_tc_resp("nope", {}), None),
                       (_final_resp("recovered"), None)], cap):
            r = await envoy.run_envoy("go", cwd=d)
        assert r.ok and r.text == "recovered"
        assert any("unknown tool" in (m.get("content") or "") for m in cap[1]["messages"])


async def test_no_key():
    with _patched([], [], key=None):
        r = await envoy.run_envoy("task", cwd=".")
    assert not r.ok and "no xAI key" in r.error


async def test_empty_task():
    with _patched([], []):
        r = await envoy.run_envoy("   ", cwd=".")
    assert not r.ok and "empty task" in r.error


async def test_transport_error_fails_open():
    with _patched([(None, "boom")], []):
        r = await envoy.run_envoy("task", cwd=".")
    assert not r.ok and r.error == "boom"


async def test_step_limit_truncates():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "note.txt").write_text("x")
        with _patched([(_tc_resp("read_file", {"path": "note.txt"}), None),
                       (_tc_resp("read_file", {"path": "note.txt"}), None)], []):
            r = await envoy.run_envoy("loop forever", cwd=d, max_steps=2)
        assert r.truncated and r.steps == 2 and "step limit" in r.text


# --- server build gating -----------------------------------------------------
def test_build_server_key_gated():
    o_key = xai_common.load_key
    try:
        xai_common.load_key = lambda: None
        assert envoy.build_envoy_server(".") is None            # no key → nothing to expose
        assert envoy.build_envoy_server(".", require_key=False) is not None
    finally:
        xai_common.load_key = o_key


def test_registry_shares_recall_handlers():
    """The no-drift claim: the native loop pulls the SAME recall handlers the SDK
    server exposes, via build_recall_tools."""
    import memory_tools
    tools = memory_tools.build_recall_tools(".")
    assert {t.name for t in tools} == {"recall_search", "recall_read_note", "code_search"}
    assert all(callable(t.handler) for t in tools)


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
