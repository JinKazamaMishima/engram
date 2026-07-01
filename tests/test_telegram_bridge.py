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
