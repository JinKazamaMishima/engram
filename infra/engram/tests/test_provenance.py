#!/usr/bin/env python3
"""Headless tests for the 2026-07-07 TUI trio: the per-turn recall provenance line
(memory made visible — including the zero-hit and hook-silent miss-detector states),
serialized interaction cards (parallel AskUserQuestion calls must ask one → answer →
next, not stack and clobber), and the clipboard fix (every copy path goes through the
overridden copy_to_clipboard: OSC52 + a real clipboard tool + status feedback).

    .venv/bin/python infra/engram/tests/test_provenance.py
"""
import asyncio
import json
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import app as app_mod  # noqa: E402 — to patch its subprocess seam
from app import EngramApp, PromptArea, render_recall_line  # noqa: E402
from claude_agent_sdk import HookEventMessage  # noqa: E402
from core import Event, ModelDriver, _recall_line  # noqa: E402
from textual.widgets import OptionList  # noqa: E402


class FakeDriver(ModelDriver):
    """Yields a scripted list of Events; on_interaction optionally fired first
    (concurrently, mirroring the SDK's parallel permission-callback tasks)."""
    def __init__(self, events, interactions=None):
        self.model = "opus[1m]"
        self.effort = "max"
        self.session_id = None
        self.actual_model = None
        self.resumed = False
        self.stderr_tail = ""
        self.on_interaction = None
        self.permission_mode = "bypassPermissions"
        self._events = events
        self._interactions = interactions or []
        self.verdicts = []

    async def query(self, text, *, prepend=""):
        if self._interactions:
            # The SDK spawns one task per permission callback — fire them all at
            # once so the app's serialization (not our call order) is what's tested.
            self.verdicts = await asyncio.gather(
                *(self.on_interaction(req) for req in self._interactions))
        for ev in self._events:
            yield ev

    async def disconnect(self): ...
    def reset(self): self.session_id = None
    async def set_effort(self, level): self.effort = level
    async def set_model(self, name): self.model = name
    async def set_permission_mode(self, mode): self.permission_mode = mode


async def wait_until(cond, pilot, what, limit=200):
    for _ in range(limit):
        if cond():
            return
        await pilot.pause()
    raise AssertionError(f"timeout waiting for: {what}")


# ---- pure helpers: core._recall_line (wire → text) + app.render_recall_line ----

def _hook_msg(**kw):
    d = dict(subtype="hook_response", hook_event_name="UserPromptSubmit",
             data={}, session_id=None, uuid=None)
    d.update(kw)
    return HookEventMessage(**d)


def test_recall_line_extraction() -> None:
    out = json.dumps({"suppressOutput": True,
                      "systemMessage": "🧠 recalled: recall:note-a, global:note-b"})
    assert _recall_line(_hook_msg(data={"output": out})) == "recall:note-a, global:note-b"
    # Hook ran, surfaced nothing (no output at all) → '' — an honest zero, not None.
    assert _recall_line(_hook_msg(data={})) == ""
    # Someone else's hook printing non-JSON must read as '' (ran), never crash.
    assert _recall_line(_hook_msg(data={"output": "not json"})) == ""
    # Not our event → None (hook_started, or another hook's response).
    assert _recall_line(_hook_msg(subtype="hook_started")) is None
    assert _recall_line(_hook_msg(hook_event_name="PreCompact")) is None
    print("✓ core._recall_line: slugs out of hook_response; ''=zero-hit; None=not ours")


def test_render_recall_line() -> None:
    line = render_recall_line("recall:a, global:b, recall:c, recall:d, global:e")
    assert "5 notes" in line and "soul:b" in line and "+2" in line, line
    assert "recall:a" not in line and " a" in line, line   # project prefix dropped
    assert "1 note ·" in render_recall_line("recall:only-one")
    assert "no notes" in render_recall_line("")
    assert "silent" in render_recall_line(None)
    print("✓ render_recall_line: caps at 3 + count · soul: kept · zero + silent states")


# ---- the provenance line in the live UI ----

async def scenario_recall_line_renders() -> None:
    driver = FakeDriver([Event("recall", "recall:note-a, global:note-b"),
                         Event("text", "grounded reply")])
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("what do you remember about note a?"))
        await wait_until(lambda: not app._busy and len(app.query(".recall-line")) > 0,
                         pilot, "recall line mounted")
        assert len(app.query(".recall-line")) == 1
        for _ in range(8):                       # let stream mounts settle pre-teardown
            await pilot.pause()
    print("✓ UI: Event('recall') → one provenance line above the reply")


async def scenario_recall_silent_detector() -> None:
    # No recall event at all (hook not firing) → the line renders the outage tell.
    driver = FakeDriver([Event("text", "reply with no injection")])
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("a prompt long enough to recall"))
        await wait_until(lambda: not app._busy and len(app.query(".recall-line")) == 1,
                         pilot, "silent-detector line mounted exactly once")
        for _ in range(8):
            await pilot.pause()
        assert len(app.query(".recall-line")) == 1, "must not mount twice"
    print("✓ UI: no hook event → 'silent' miss-detector line (exactly one)")


# ---- serialized interaction cards ----

async def scenario_parallel_questions_serialize() -> None:
    q = lambda i: {"kind": "question", "questions": [{  # noqa: E731
        "header": f"Q{i}", "question": f"question {i}?",
        "options": [{"label": f"answer {i}"}]}]}
    driver = FakeDriver([Event("text", "done")], interactions=[q(1), q(2)])
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("ask me two things"))
        await wait_until(lambda: app._interact_open, pilot, "first card open")
        # SERIALIZED: exactly one live (enabled) chooser, even though both
        # callbacks fired concurrently. The second must not have mounted yet.
        cards = list(app.query(OptionList).filter(".interact"))
        assert len(cards) == 1, f"expected 1 card while Q1 is open, got {len(cards)}"
        assert not cards[0].disabled, "the open card must be answerable (not greyed)"
        await pilot.press("enter")                       # pick "answer 1"
        await wait_until(lambda: app._interact_open and len(
            app.query(OptionList).filter(".interact")) == 2, pilot, "second card open")
        cards = list(app.query(OptionList).filter(".interact"))
        assert cards[0].disabled and not cards[1].disabled, \
            "Q1 frozen as a record; Q2 live — never greyed before its answer"
        await pilot.press("enter")                       # pick "answer 2"
        await wait_until(lambda: len(driver.verdicts) == 2, pilot, "both verdicts")
        a1, a2 = (v["message"] for v in driver.verdicts)
        assert "answer 1" in a1 and "answer 2" in a2, driver.verdicts
        await wait_until(lambda: not app._busy, pilot, "turn finishes")
    print("✓ parallel AskUserQuestion calls serialize: ask → answer → next, both delivered")


# ---- clipboard: one robust path for every copy ----

async def scenario_copy_paths() -> None:
    calls = []
    real_run = app_mod.subprocess.run

    class _Done:
        returncode = 0

    def fake_run(cmd, **kw):
        calls.append((list(cmd), kw.get("input", b"")))
        return _Done()

    app_mod.subprocess.run = fake_run
    try:
        driver = FakeDriver([Event("text", "the reply to copy")])
        app = EngramApp(driver=driver)
        async with app.run_test() as pilot:
            await pilot.pause()
            # The override chases OSC52 with a real clipboard tool + status feedback
            # — this is the path Textual's own Screen ctrl+c (drag-selection) calls.
            app.copy_to_clipboard("hello from engram")
            assert calls and calls[0][1] == b"hello from engram", calls
            assert "copied" in str(app.query_one("#status", app_mod.Static).content)
            # /copy = the discoverable last-reply path (same as ctrl+y).
            app.post_message(PromptArea.Submitted("say something"))
            await wait_until(lambda: app._last_reply == "the reply to copy",
                             pilot, "turn done (reply recorded)")
            calls.clear()
            app.post_message(PromptArea.Submitted("/copy"))
            await wait_until(lambda: bool(calls), pilot, "/copy ran")
            assert calls[0][1] == b"the reply to copy", calls
    finally:
        app_mod.subprocess.run = real_run
    print("✓ copy: copy_to_clipboard chases OSC52 with a clipboard tool · /copy copies the reply")


async def main() -> int:
    test_recall_line_extraction()
    test_render_recall_line()
    await scenario_recall_line_renders()
    await scenario_recall_silent_detector()
    await scenario_parallel_questions_serialize()
    await scenario_copy_paths()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
