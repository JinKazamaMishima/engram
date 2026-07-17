#!/usr/bin/env python3
"""Hermetic unit tests for grok_agent (Engram's native Grok worker).

No network, no key, no spend — ``xai_common.post_json`` and ``load_key`` are
stubbed. Covers the request shape (model, effort mapping, banned reasoning
params, structured-output plumbing), cost decode, and fail-open paths.

    .venv/bin/python infra/engram/tests/test_grok_agent.py
"""
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import grok_agent  # noqa: E402  (module handle for server-build patching)
import pytest  # noqa: E402
import xai_common  # noqa: E402  (after sys.path shim, like the sibling tests)
from grok_agent import GrokResult, grok_task, map_effort  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_xai_globals():
    """Restore the xai_common seams every test here monkeypatches. ``_stub`` and
    ``test_no_key`` set ``load_key`` / ``post_json`` without restoring — harmless per
    file, but under ONE pytest collection of the whole engram suite it leaks a stubbed
    ``xai_common.load_key`` into ``test_x_search::test_load_key`` (green apart, red
    together). Autouse save/restore closes the leak; a no-op under the ``__main__``
    script runner (single file → nothing to leak into)."""
    saved = (xai_common.post_json, xai_common.load_key)
    try:
        yield
    finally:
        xai_common.post_json, xai_common.load_key = saved


def _stub(resp=None, err=None, key="k", capture=None):
    def fake_post(path, payload, key_, timeout):
        if capture is not None:
            capture.update(path=path, payload=payload, key=key_, timeout=timeout)
        return resp, err
    xai_common.post_json = fake_post
    xai_common.load_key = lambda: key


def _resp(content, ticks=1_000_000, model="grok-4.5"):
    return {"model": model,
            "choices": [{"message": {"content": content}}],
            "usage": {"cost_in_usd_ticks": ticks}}


def test_effort_map():
    assert map_effort("low") == "low"
    assert map_effort("medium") == "medium"
    assert map_effort("high") == "high"
    assert map_effort("xhigh") == "high"     # hotter Engram rungs collapse
    assert map_effort("max") == "high"
    assert map_effort(None) == "low"
    assert map_effort("bogus") == "low"


def test_request_shape():
    cap = {}
    _stub(resp=_resp("hi"), capture=cap)
    r = grok_task("hello", effort="max", system="be terse")
    p = cap["payload"]
    assert p["model"] == "grok-4.5"
    assert p["reasoning_effort"] == "high"                 # max -> high
    assert p["messages"][0] == {"role": "system", "content": "be terse"}
    assert p["messages"][-1] == {"role": "user", "content": "hello"}
    for banned in ("presence_penalty", "frequency_penalty", "stop"):
        assert banned not in p                             # reasoning rejects these
    assert "response_format" not in p                      # no schema -> none
    assert "max_tokens" not in p                           # omitted unless asked
    assert r.ok and r.text == "hi" and r.cost_usd == 0.001


def test_structured_output():
    cap = {}
    _stub(resp=_resp('{"sentiment": "positive"}'), capture=cap)
    schema = {"type": "object", "properties": {"sentiment": {"type": "string"}}}
    r = grok_task("classify", schema=schema)
    assert cap["payload"]["response_format"]["type"] == "json_schema"
    assert r.ok and r.data == {"sentiment": "positive"}


def test_no_key():
    xai_common.load_key = lambda: None
    r = grok_task("x")
    assert not r.ok and "key" in r.error.lower()


def test_api_error_passthrough():
    _stub(resp=None, err="xAI API error: boom")
    r = grok_task("x")
    assert not r.ok and "boom" in r.error


def test_malformed_response_fails_open():
    _stub(resp={"unexpected": True})
    r = grok_task("x")
    assert not r.ok and r.error.startswith("parse:")


# --- the in-process MCP tool (build_grok_server) ----------------------------

def test_format():
    """Trimmed answer + a model/cost footer; honest fallbacks when text/cost absent."""
    out = grok_agent._format(GrokResult(text="  answer  ", cost_usd=0.001, model="grok-4.5"))
    assert out.startswith("answer") and "[grok · model grok-4.5 · ~$0.001]" in out
    bare = grok_agent._format(GrokResult())               # no text, no cost, no model
    assert "(Grok returned no text)" in bare
    assert "model grok-4.5" in bare                       # falls back to MODEL
    assert "$" not in bare                                # no cost bit when cost is None


async def test_handler():
    """The exposed ``grok`` tool: forwards prompt+effort, formats the answer, and
    fails open on empty prompt / worker error — worker MOCKED, zero spend."""
    orig_load, orig_task, orig_create = (
        xai_common.load_key, grok_agent.grok_task, grok_agent.create_sdk_mcp_server)
    try:
        xai_common.load_key = lambda: "test-key"          # build passes the key gate
        captured = {}
        grok_agent.create_sdk_mcp_server = (
            lambda name, version="1.0.0", tools=None:
            captured.update({t.name: t for t in tools})
            or orig_create(name=name, version=version, tools=tools))
        grok_agent.build_grok_server()
        tool = captured["grok"]

        seen = {}
        def fake_ok(prompt, *, effort="low"):
            seen.update(prompt=prompt, effort=effort)
            return GrokResult(text="hi from grok", cost_usd=0.002, model="grok-4.5")
        grok_agent.grok_task = fake_ok

        out = await tool.handler({"prompt": "hello", "effort": "high"})
        assert not out.get("is_error"), out
        body = out["content"][0]["text"]
        assert "hi from grok" in body and "grok-4.5" in body and "~$0.002" in body
        assert seen == {"prompt": "hello", "effort": "high"}   # prompt+effort forwarded

        await tool.handler({"prompt": "x"})                    # effort omitted → default low
        assert seen["effort"] == "low"

        out = await tool.handler({"prompt": "   "})            # empty prompt, never calls worker
        assert out.get("is_error") and "empty prompt" in out["content"][0]["text"]

        grok_agent.grok_task = lambda prompt, *, effort="low": GrokResult(error="boom")
        out = await tool.handler({"prompt": "x"})              # worker error → is_error, no crash
        assert out.get("is_error") and "boom" in out["content"][0]["text"]
    finally:
        xai_common.load_key, grok_agent.grok_task, grok_agent.create_sdk_mcp_server = (
            orig_load, orig_task, orig_create)


def test_build_requires_key():
    """build_grok_server returns None with no key (nothing to expose), builds with one."""
    orig = xai_common.load_key
    try:
        xai_common.load_key = lambda: None
        assert grok_agent.build_grok_server() is None, "no key → no server"
        assert grok_agent.build_grok_server(require_key=False) is not None
    finally:
        xai_common.load_key = orig


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
