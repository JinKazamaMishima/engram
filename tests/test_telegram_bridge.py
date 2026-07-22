"""Tests for the recall Telegram bridge's pure helpers: the markdown->Telegram-HTML
renderer (recall.notify, always tested) and the bridge's text-splitting / command
helpers (infra/telegram/agent_bridge.py, skipped if the optional telegram extra
isn't installed). No network and no SDK connection — importing the bridge only
reads env + defines functions; it connects to Telegram/Claude solely in main()."""
from __future__ import annotations

import asyncio
import contextlib
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


# ---- /btw — steer the in-flight turn --------------------------------------
# handle_message-level: a /btw while a turn is busy must write the bare note
# straight onto the live client's stdin (steering the running reply) and must
# NOT touch the _pending queue; idle (or on a failed write) it degrades to the
# normal coalesce path so the note is never lost.

class _BtwClient:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def query(self, prompt, session_id="default"):
        self.sent.append(prompt)


class _DeadClient:
    async def query(self, prompt, session_id="default"):
        raise RuntimeError("transport down")


class _BufStub:
    def append(self, *a, **k): ...


def _btw_env(monkeypatch, tmp_path, sent):
    async def fake_send(text, chat_id=None):
        sent.append(text)

    async def fake_typing():
        pass

    monkeypatch.setattr(bridge, "CHAT_ID_RAW", "42")
    monkeypatch.setattr(bridge, "LOCK_FILE", tmp_path / "bridge.lock")
    monkeypatch.setattr(bridge, "send", fake_send)
    monkeypatch.setattr(bridge, "send_typing", fake_typing)
    monkeypatch.setattr(bridge, "audit", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "_buffer", _BufStub())
    monkeypatch.setattr(bridge, "_pending", [])
    monkeypatch.setattr(bridge, "_queued_ack_sent", False)


def _msg(text):
    return {"message": {"chat": {"id": 42}, "text": text}}


@requires_sdk
def test_btw_mid_turn_writes_to_live_client_not_queue(monkeypatch, tmp_path):
    sent: list[str] = []
    _btw_env(monkeypatch, tmp_path, sent)
    client = _BtwClient()
    monkeypatch.setattr(bridge, "_client", client)

    async def scenario():
        turn = asyncio.create_task(asyncio.sleep(30))      # a busy in-flight turn
        monkeypatch.setattr(bridge, "_turn_task", turn)
        try:
            await bridge.handle_message(_msg("/btw check the dates"))
        finally:
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn

    asyncio.run(scenario())
    assert client.sent == ["check the dates"]              # bare note, straight to stdin
    assert bridge._pending == []                           # steered, NOT queued
    assert any("folded" in s for s in sent)                # the ack Alex sees


@requires_sdk
def test_btw_idle_falls_back_to_normal_message(monkeypatch, tmp_path):
    sent: list[str] = []
    _btw_env(monkeypatch, tmp_path, sent)
    monkeypatch.setattr(bridge, "_client", None)
    monkeypatch.setattr(bridge, "_turn_task", None)

    async def scenario():
        try:
            await bridge.handle_message(_msg("/btw check the dates"))
            assert bridge._pending == [{"text": "check the dates", "paths": []}]
        finally:
            bridge._clear_pending()                        # cancel the armed drain task

    asyncio.run(scenario())


@requires_sdk
def test_btw_failed_write_falls_back_to_queue(monkeypatch, tmp_path):
    sent: list[str] = []
    _btw_env(monkeypatch, tmp_path, sent)
    monkeypatch.setattr(bridge, "_client", _DeadClient())

    async def scenario():
        turn = asyncio.create_task(asyncio.sleep(30))
        monkeypatch.setattr(bridge, "_turn_task", turn)
        try:
            await bridge.handle_message(_msg("/btw check the dates"))
            assert bridge._pending == [{"text": "check the dates", "paths": []}]
        finally:
            bridge._clear_pending()
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn

    asyncio.run(scenario())
    assert any("queued" in s for s in sent)                # the one busy-period ack


@requires_sdk
def test_btw_bare_shows_usage(monkeypatch, tmp_path):
    sent: list[str] = []
    _btw_env(monkeypatch, tmp_path, sent)
    monkeypatch.setattr(bridge, "_client", None)
    monkeypatch.setattr(bridge, "_turn_task", None)
    asyncio.run(bridge.handle_message(_msg("/btw")))
    assert bridge._pending == []
    assert any("usage: /btw" in s for s in sent)


# ---- ripple m1 — live draft streaming --------------------------------------
# The streamer is cosmetic BY CONTRACT: drafts may fail (old server, flood, bad
# client) but finalized text must always be cut from the streamed buffer and
# prefix-checked against the authoritative reply — degrade, never corrupt.

def _http_429(retry_after):
    import io
    import urllib.error
    body = (b"{}" if retry_after is None else
            f'{{"ok":false,"parameters":{{"retry_after":{retry_after}}}}}'.encode())
    return urllib.error.HTTPError("u", 429, "Too Many Requests", None, io.BytesIO(body))


@requires_sdk
def test_retry_after_parsed_floored_and_capped():
    assert bridge._retry_after_secs(_http_429(2.5)) == 2.5
    assert bridge._retry_after_secs(_http_429(None)) == 1.0   # missing -> never 0
    assert bridge._retry_after_secs(_http_429(0)) == 1.0      # zero would amplify
    assert bridge._retry_after_secs(_http_429(300)) == 30.0   # bounded


@requires_sdk
def test_telegram_post_honors_429_then_succeeds(monkeypatch):
    calls, sleeps = [], []

    class _Ok:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise _http_429(3)
        return _Ok()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", sleeps.append)
    out = bridge._telegram_post("sendMessage", {"chat_id": 1, "text": "x"})
    assert out == {"ok": True} and len(calls) == 2
    assert sleeps == [3.0]                                    # server's window, honored


@requires_sdk
def test_telegram_post_retries_zero_fails_fast(monkeypatch):
    import urllib.error
    sleeps = []
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(_http_429(5)))
    monkeypatch.setattr("time.sleep", sleeps.append)
    with pytest.raises(urllib.error.HTTPError):
        bridge._telegram_post("sendMessageDraft", {"chat_id": 1}, 10, 0)
    assert sleeps == []                    # a cosmetic draft never waits out a flood


@requires_sdk
def test_stream_cut_paragraph_seams_never_inside_fences():
    p1 = "intro paragraph"
    fenced = "```\ncode top\n\ncode bottom\n```"
    text = f"{p1}\n\n{fenced}\n\ntail still growing"
    cut = bridge._stream_cut(text)
    # Largest safe seam = after the fence CLOSES; never at the blank line inside
    # the fence, never at the end (the tail paragraph is still being generated).
    assert cut == len(p1) + 2 + len(fenced)
    assert bridge._stream_cut(f"{p1}\n\n```\nopen fence\n\nstill code") == len(p1)
    assert bridge._stream_cut("one growing paragraph, no seam") == 0
    assert bridge._stream_cut("a\n\nb") == 1


@requires_sdk
def test_draft_tail_passthrough_and_line_trim():
    assert bridge._draft_tail("short text") == "short text"
    long = "\n".join(f"line {i}" for i in range(1000))
    tail = bridge._draft_tail(long, budget=100)
    assert tail.startswith("… ") and bridge._u16_len(tail) <= 102
    assert tail.endswith("line 999")       # newest lines win
    monster = "y" * 500
    t = bridge._draft_tail(monster, budget=100)
    assert t.startswith("… ") and bridge._u16_len(t) <= 100 and t.endswith("y")


def _streamer_env(monkeypatch, sends, drafts, typing=None):
    """Patch the streamer's world: recording send(), a _telegram_post that records
    sendMessageDraft calls (raising if drafts is None), recording send_typing."""
    async def fake_send(text, chat_id=None):
        sends.append(text)

    async def fake_typing():
        (typing if typing is not None else []).append(1)

    def fake_post(method, params, timeout=15.0, retries=2):
        if drafts is None:
            raise RuntimeError("draft transport down")
        assert method == "sendMessageDraft" and retries == 0
        drafts.append(dict(params))
        return {"ok": True}

    monkeypatch.setattr(bridge, "send", fake_send)
    monkeypatch.setattr(bridge, "send_typing", fake_typing)
    monkeypatch.setattr(bridge, "_telegram_post", fake_post)


@requires_sdk
def test_streamer_ticks_post_throttled_drafts_with_cursor(monkeypatch):
    sends, drafts = [], []
    _streamer_env(monkeypatch, sends, drafts)

    async def scenario():
        s = bridge._TurnStreamer(42)
        await s._post_draft("")                     # start(): Thinking… placeholder
        s.feed("hello ")
        await s._tick()
        s.feed("world")
        await s._tick()
        await s._tick()                             # unchanged text -> no repost
        s._last_post = -999                         # keepalive window elapsed
        await s._tick()                             # -> repost same text (30s expiry)
        return s

    s = asyncio.run(scenario())
    assert [d["text"] for d in drafts] == ["", "hello  ▌", "hello world ▌", "hello world ▌"]
    assert all(d["draft_id"] == s.draft_id for d in drafts)   # one animated bubble
    assert sends == [] and s.draft_ok


@requires_sdk
def test_streamer_finalizes_chunk_then_streams_remainder(monkeypatch):
    sends, drafts = [], []
    _streamer_env(monkeypatch, sends, drafts)
    monkeypatch.setattr(bridge, "STREAM_FINALIZE_U16", 40)
    para1 = "first paragraph " + "x" * 40
    para2 = "second paragraph, still short"

    async def scenario():
        s = bridge._TurnStreamer(42)
        first_id = s.draft_id
        s.feed(para1 + "\n\n" + para2)
        await s._tick()                             # over threshold -> finalize para1
        assert sends == [para1] and s.chunks_sent == 1
        assert s.draft_id != first_id               # continuation = fresh bubble
        await s._tick()                             # remainder rides the new draft
        assert drafts[-1]["text"] == para2 + " ▌"
        assert drafts[-1]["draft_id"] == s.draft_id
        remainder = await s.finish(para1 + "\n\n" + para2)
        return remainder

    remainder = asyncio.run(scenario())
    assert remainder == para2                       # exactly the un-finalized tail
    assert sends == [para1]                         # nothing duplicated, nothing lost


@requires_sdk
def test_streamer_draft_failure_degrades_to_typing_keepalive(monkeypatch):
    sends, typing = [], []
    _streamer_env(monkeypatch, sends, drafts=None, typing=typing)
    monkeypatch.setattr(bridge, "STREAM_FINALIZE_U16", 40)

    async def scenario():
        s = bridge._TurnStreamer(42)
        await s._post_draft("")                     # transport down -> degrade
        assert not s.draft_ok
        s.feed("small update")
        s._last_post = -999
        await s._tick()                             # degraded tick -> typing ping
        assert typing == [1]
        s.feed(" and now enough text to cross the finalize threshold\n\nnext para")
        await s._tick()                             # finalize still works draft-less
        return s

    s = asyncio.run(scenario())
    assert s.chunks_sent == 1 and len(sends) == 1   # delivery survives dead drafts


@requires_sdk
def test_streamer_finish_divergence_sends_full_reply(monkeypatch):
    sends, drafts = [], []
    _streamer_env(monkeypatch, sends, drafts)

    async def scenario():
        s = bridge._TurnStreamer(42)
        s.buf = "streamed prefix\n\nmore text"
        s.finalized = len("streamed prefix") + 2    # as if a chunk was finalized
        return await s.finish("a completely different final reply")

    assert asyncio.run(scenario()) == "a completely different final reply"


@requires_sdk
def test_streamer_finish_without_finalize_returns_reply_verbatim():
    async def scenario():
        s = bridge._TurnStreamer(42)
        s.feed("draft-only text, never finalized")
        return await s.finish("the authoritative reply")

    assert asyncio.run(scenario()) == "the authoritative reply"


@requires_sdk
def test_stream_event_text_extracts_only_top_level_text_deltas():
    from types import SimpleNamespace

    def ev(dtype=None, parent=None, text="hi", etype="content_block_delta"):
        e = {"type": etype}
        if dtype:
            e["delta"] = {"type": dtype, "text" if dtype == "text_delta" else "thinking": text}
        return SimpleNamespace(parent_tool_use_id=parent, event=e)

    assert bridge._stream_event_text(ev("text_delta")) == "hi"
    assert bridge._stream_event_text(ev("thinking_delta")) == ""   # never hits the chat
    assert bridge._stream_event_text(ev("text_delta", parent="tool-1")) == ""  # subagent
    assert bridge._stream_event_text(ev(etype="message_start")) == ""


@requires_sdk
def test_build_options_toggles_partial_messages(monkeypatch):
    monkeypatch.setattr(bridge, "STREAM_ENABLED", True)
    assert bridge._build_options(None).include_partial_messages is True
    monkeypatch.setattr(bridge, "STREAM_ENABLED", False)
    assert bridge._build_options(None).include_partial_messages is False


@requires_sdk
def test_process_turn_delivers_chunks_live_then_remainder(monkeypatch):
    """End-to-end wiring: run_turn feeds the streamer mid-flight, a chunk lands
    BEFORE the turn returns, the remainder lands after, nothing duplicates."""
    sends, drafts = [], []
    _streamer_env(monkeypatch, sends, drafts)
    monkeypatch.setattr(bridge, "STREAM_ENABLED", True)
    monkeypatch.setattr(bridge, "STREAM_FINALIZE_U16", 40)
    monkeypatch.setattr(bridge, "DRAFT_TICK_SECS", 0.01)
    monkeypatch.setattr(bridge, "CHAT_ID_RAW", "42")
    monkeypatch.setattr(bridge, "audit", lambda *a: None)
    monkeypatch.setattr(bridge, "_buffer", _BufStub())
    para1 = "streamed early " + "z" * 40
    para2 = "and the closing thought"
    mid_flight = []

    async def fake_run_turn(text, streamer=None):
        streamer.feed(para1 + "\n\n")
        await asyncio.sleep(0.2)                    # ~20 ticks: plenty to finalize
        mid_flight.append(list(sends))              # what the user saw mid-turn
        streamer.feed(para2)
        return (para1 + "\n\n" + para2).strip()

    monkeypatch.setattr(bridge, "run_turn", fake_run_turn)
    asyncio.run(bridge._process_turn("hola"))
    assert mid_flight == [[para1]]                  # chunk 1 arrived DURING the turn
    assert sends == [para1, para2]                  # remainder after, no dupes
    assert drafts and drafts[0]["text"] == ""       # the Thinking… placeholder led
