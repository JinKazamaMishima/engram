"""Tests for the recall Telegram bridge's pure helpers: the markdown->Telegram-HTML
renderer (recall.notify, always tested) and the bridge's text-splitting / command
helpers (infra/telegram/agent_bridge.py, skipped if the optional telegram extra
isn't installed). No network and no SDK connection — importing the bridge only
reads env + defines functions; it connects to Telegram/Claude solely in main()."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from recall import notify

try:
    import claude_agent_sdk  # noqa: F401
    _spec = importlib.util.spec_from_file_location(
        "recall_agent_bridge",
        Path(__file__).resolve().parent.parent / "infra" / "telegram" / "agent_bridge.py")
    bridge = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(bridge)
    _HAVE_SDK = True
except Exception:  # noqa: BLE001 — telegram extra not installed; skip those tests
    bridge = None
    _HAVE_SDK = False

requires_sdk = pytest.mark.skipif(
    not _HAVE_SDK, reason="telegram extra (claude-agent-sdk) not installed")


# ---- markdown -> Telegram HTML (recall.notify) ---------------------------

def test_md_bold_code_header():
    out = notify._md_to_telegram_html("## Title\n**bold** and `recall_inject.py`")
    assert "<b>Title</b>" in out and "<b>bold</b>" in out
    assert "<code>recall_inject.py</code>" in out   # underscores survive escaping


def test_md_h1_is_underlined():
    assert "<b><u>Top</u></b>" in notify._md_to_telegram_html("# Top")


def test_md_escapes_and_spoiler():
    out = notify._md_to_telegram_html("a < b & c ||secret||")
    assert "&lt;" in out and "&amp;" in out
    assert "<tg-spoiler>secret</tg-spoiler>" in out


def test_md_fenced_block_escaped_in_pre():
    out = notify._md_to_telegram_html("```\nx = a<b\n```")
    assert "<pre>" in out and "</pre>" in out and "x = a&lt;b" in out  # escaped inside pre


def test_md_link():
    assert '<a href="https://x.io">t</a>' in notify._md_to_telegram_html("[t](https://x.io)")


def test_md_never_raises_on_odd_input():
    for s in ["", "**", "`unclosed", "||", ">quote", "[t](http://x)", "###"]:
        notify._md_to_telegram_html(s)   # must not raise


# ---- bridge text + command helpers ---------------------------------------

@requires_sdk
def test_u16_len_counts_utf16_code_units():
    assert bridge._u16_len("abc") == 3
    assert bridge._u16_len("a😀b") == 4   # emoji = a surrogate pair (2 units)


@requires_sdk
def test_split_respects_budget_and_keeps_content():
    text = "\n\n".join(f"para {i} " + "x" * 100 for i in range(50))
    chunks = bridge._split_for_telegram(text, 200)
    assert chunks and all(bridge._u16_len(c) <= 200 for c in chunks)
    joined = "".join(chunks)
    assert "para 0 " in joined and "para 49 " in joined   # nothing dropped


@requires_sdk
def test_split_hard_wraps_one_overlong_line():
    chunks = bridge._split_for_telegram("y" * 500, 100)
    assert len(chunks) >= 5 and all(bridge._u16_len(c) <= 100 for c in chunks)


@requires_sdk
def test_matches_cmd_handles_at_suffix():
    bridge.BOT_USERNAME = "EngramRecallBot"
    assert bridge._matches_cmd("/ping", "ping")
    assert bridge._matches_cmd("/ping@EngramRecallBot", "ping")
    assert not bridge._matches_cmd("/pingpong", "ping")
    assert not bridge._matches_cmd("ping", "ping")


@requires_sdk
def test_build_options_pins_effort_and_model():
    # setting_sources=["project"] excludes the user ~/.claude/settings.json, so the
    # bridge must pin these itself or fall back to CLI defaults. Default = the
    # operator's box: max effort + Opus 1M.
    o = bridge._build_options(None)
    assert o.effort == "max"
    assert o.model == "opus[1m]"


@requires_sdk
def test_cmd_arg_parses_model_switch():
    bridge.BOT_USERNAME = ""
    assert bridge._cmd_arg("/model", "model") == ""            # bare -> show/list
    assert bridge._cmd_arg("/model fable", "model") == "fable"  # arg -> switch target
    assert bridge._cmd_arg("/model  opus[1m] ", "model") == "opus[1m]"  # trimmed
    assert bridge._cmd_arg("/status", "model") is None         # different command
    assert bridge._cmd_arg("/modelfoo", "model") is None       # not a prefix match


@requires_sdk
def test_build_options_reflects_switched_model():
    # /model recycles the client onto a new model; _build_options must read the live
    # value, not the frozen startup default. Guard the module global for order-independence.
    saved = bridge._current_model
    try:
        bridge._current_model = "fable"
        assert bridge._build_options(None).model == "fable"
    finally:
        bridge._current_model = saved


@requires_sdk
def test_build_options_wires_recall_inject_hook():
    # setting_sources=["project"] excludes the user settings where a terminal
    # install's UserPromptSubmit hook lives — the bridge must wire injection itself
    # or phone sessions silently get no corpus recall.
    o = bridge._build_options(None)
    assert o.hooks and "UserPromptSubmit" in o.hooks
    assert bridge._recall_inject_hook in o.hooks["UserPromptSubmit"][0].hooks


@requires_sdk
def test_recall_inject_hook_passes_script_json_through(tmp_path, monkeypatch):
    import asyncio
    import json as _json
    stub = tmp_path / "stub_inject.py"
    payload = {"suppressOutput": True, "systemMessage": "🧠 recalled: recall:x",
               "hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                      "additionalContext": "## Recalled"}}
    stub.write_text("import sys, json\n"
                    "hook = json.load(sys.stdin)\n"           # must be valid hook JSON
                    "assert 'prompt' in hook\n"
                    f"print(json.dumps({payload!r}))\n")
    monkeypatch.setattr(bridge, "RECALL_INJECT", str(stub))
    out = asyncio.run(bridge._recall_inject_hook({"prompt": "where is x?"}, None, {}))
    assert out == payload
    assert _json.dumps(out)  # round-trips


@requires_sdk
def test_recall_inject_hook_fails_open(tmp_path, monkeypatch):
    import asyncio
    # Script explodes -> {} (inject nothing), never an exception up to the SDK.
    boom = tmp_path / "boom.py"
    boom.write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(bridge, "RECALL_INJECT", str(boom))
    assert asyncio.run(bridge._recall_inject_hook({"prompt": "hi there"}, None, {})) == {}
    # Disabled (empty path) -> {} without spawning anything.
    monkeypatch.setattr(bridge, "RECALL_INJECT", "")
    assert asyncio.run(bridge._recall_inject_hook({"prompt": "hi there"}, None, {})) == {}


@requires_sdk
def test_session_curation_cmd_targets_the_ended_session():
    # brick 2: /new and /end fire this argv to curate the ended session in the
    # background — the recall CLI, --session <id>, scoped to AGENT_CWD, committing.
    cmd = bridge._session_curation_cmd("sess-123")
    assert cmd[0] == bridge.RECALL_BIN and cmd[1] == "curate"
    assert "--session" in cmd and "sess-123" in cmd
    assert "--project-dir" in cmd and str(bridge.AGENT_CWD) in cmd
    assert "--commit" in cmd


# ---- LiveBuffer bridge parity (tier 1) -------------------------------------

def _tmp_buffer(bridge_mod, tmp_path, monkeypatch):
    """Point the bridge's LiveBuffer at a tmp dir + a fresh launch id, and its
    SESSION_FILE at tmp (so _save_session's persistence stays sandboxed)."""
    from buffer import LiveBuffer  # importable: the bridge put infra/engram on sys.path
    monkeypatch.setattr(bridge_mod, "_buf_launch_id", "launch-testaaaa")
    monkeypatch.setattr(bridge_mod, "_buf_convo_id", "launch-testaaaa")
    monkeypatch.setattr(bridge_mod, "_buffer",
                        LiveBuffer(tmp_path, lambda: bridge_mod._buf_convo_id))
    monkeypatch.setattr(bridge_mod, "SESSION_FILE", tmp_path / "session_id")
    return tmp_path


@requires_sdk
def test_bridge_buffers_rows_and_migrates_on_sid_mint(tmp_path, monkeypatch):
    """Rows land under the provisional launch id; the sid mint (via
    _save_session) MIGRATES the file and seq continues — the driver's exact
    launch->sid semantics, now on the bridge path."""
    import json as _json
    _tmp_buffer(bridge, tmp_path, monkeypatch)
    bridge._buffer.append("user", "hello engram")
    bridge._buffer.append("assistant", "hello operator")
    assert (tmp_path / "launch-testaaaa.jsonl").exists()
    bridge._save_session("sid-12345")                      # mint
    assert not (tmp_path / "launch-testaaaa.jsonl").exists()
    rows = [_json.loads(x) for x in
            (tmp_path / "sid-12345.jsonl").read_text().splitlines()]
    assert [(r["role"], r["seq"]) for r in rows] == [("user", 1), ("assistant", 2)]
    bridge._buffer.append("user", "second turn")
    rows = (tmp_path / "sid-12345.jsonl").read_text().splitlines()
    assert _json.loads(rows[-1])["seq"] == 3               # seq continued after rekey


@requires_sdk
def test_bridge_new_conversation_leaves_old_file_and_mints_fresh_launch(tmp_path, monkeypatch):
    """/new (sid->None) must NOT drag the finished conversation's rows into the
    next one: fresh launch id, old file untouched."""
    _tmp_buffer(bridge, tmp_path, monkeypatch)
    bridge._save_session("sid-old")
    bridge._buffer.append("user", "old convo row")
    bridge._save_session(None)                             # /new
    assert (tmp_path / "sid-old.jsonl").exists()           # finished file stays
    assert bridge._buf_convo_id.startswith("launch-")
    assert bridge._buf_convo_id != "launch-testaaaa"       # fresh id, not the boot one
    bridge._buffer.append("user", "new convo row")
    new_file = tmp_path / f"{bridge._buf_convo_id}.jsonl"
    assert new_file.exists()
    assert "old convo row" not in new_file.read_text()


@requires_sdk
def test_bridge_buffer_disabled_is_total_noop(tmp_path, monkeypatch):
    """Gate off (dir=None): appends and rekeys must be silent no-ops."""
    from buffer import LiveBuffer
    monkeypatch.setattr(bridge, "_buffer", LiveBuffer(None, lambda: "x"))
    monkeypatch.setattr(bridge, "SESSION_FILE", tmp_path / "session_id")
    bridge._buffer.append("user", "never written")
    bridge._save_session("sid-x")                          # rekey path must not raise
    assert list(tmp_path.glob("*.jsonl")) == []
