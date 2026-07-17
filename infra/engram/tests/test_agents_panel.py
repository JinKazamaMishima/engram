#!/usr/bin/env python3
"""Headless pilot tests for the aurora m2 agents panel: toggle (ctrl+t and
/agents), navigation, detail open with a real tmp jsonl fixture tailed via a
DIRECT _tick_tail() call (no wall-clock waits), enter-with-text passthrough,
esc precedence, and fail-open on unresolvable rows. Same FakeDriver + pilot
style as test_commands.py.

    .venv/bin/python infra/engram/tests/test_agents_panel.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import agent_tail  # noqa: E402
from app import EngramApp, PromptArea  # noqa: E402
from core import Event, ModelDriver  # noqa: E402
from textual.widgets import OptionList, RichLog  # noqa: E402

TASKS = [
    {"task_id": "t1", "tool_use_id": "tu1", "name": "Explore", "desc": "map code",
     "status": "running", "tokens": 9000, "last_tool": "Grep"},
    {"task_id": "t2", "name": "⚙ research", "status": "running", "workflow": True,
     "wf": {"phases": [{"title": "Scan", "agents": [
         {"label": "scan:a", "state": "progress", "model": "sonnet"}]}],
         "done": 0, "total": 1, "phase": "Scan"}},
]


class FakeDriver(ModelDriver):
    """query() emits a task snapshot then blocks on a gate (agents stay 'live')."""
    def __init__(self, tasks=None):
        self.model = "opus[1m]"
        self.effort = "max"
        self.session_id = "sess-1"
        self.cwd = Path("/home/user/repos/engram")
        self.resumed = False
        self.stderr_tail = ""
        self.calls: list[str] = []
        self.gates: list[asyncio.Event] = []
        self._tasks = tasks if tasks is not None else TASKS

    async def query(self, text, *, prepend=""):
        self.calls.append(text)
        yield Event("task", "", data={"tasks": [dict(t) for t in self._tasks]})
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        yield Event("text", "ok")

    async def disconnect(self): ...
    def reset(self): self.session_id = None
    async def set_effort(self, level): self.effort = level
    async def set_model(self, name): self.model = name


async def wait_until(cond, pilot, what, limit=200):
    for _ in range(limit):
        if cond():
            return
        await pilot.pause()
    raise AssertionError(f"timeout waiting for: {what}")


def _panel(app) -> OptionList:
    return app.query_one("#agents", OptionList)


async def scenario_toggle_and_navigate() -> None:
    app = EngramApp(driver=FakeDriver())
    async with app.run_test() as pilot:
        # closed by default; ctrl+t opens with the placeholder (no tasks yet)
        assert not app._agents_open
        await pilot.press("ctrl+t")
        await wait_until(lambda: app._agents_open, pilot, "panel open")
        assert _panel(app).display
        # a turn brings the snapshot in → rows render, highlight lands on row 0
        app.post_message(PromptArea.Submitted("go"))
        await wait_until(lambda: len(app._agent_rows) == 3, pilot, "3 rows")
        assert [r["id"] for r in app._agent_rows] == ["t1", "t2", "t2/scan:a"]
        assert app._agents_sel == "t1"
        # ↓ moves and wraps; highlight id survives a re-render
        await pilot.press("down")
        assert app._agents_sel == "t2"
        app._render_agents()
        assert app._agents_sel == "t2"
        await pilot.press("down", "down")
        assert app._agents_sel == "t1", "wraps past the end"
        # esc closes the panel (and only the panel); prompt keeps focus throughout
        assert app.focused.id == "prompt"
        await pilot.press("escape")
        assert not app._agents_open and not _panel(app).display
        print("✓ ctrl+t toggles; rows render from task events; ↑/↓ wrap; esc closes; "
              "prompt keeps focus")


async def scenario_slash_agents_and_placeholder() -> None:
    app = EngramApp(driver=FakeDriver(tasks=[]))
    async with app.run_test() as pilot:
        app.post_message(PromptArea.Submitted("/agents"))
        await wait_until(lambda: app._agents_open, pilot, "/agents opens the panel")
        assert _panel(app).option_count == 1, "placeholder row when nothing is live"
        app.post_message(PromptArea.Submitted("/agents"))
        await wait_until(lambda: not app._agents_open, pilot, "/agents toggles closed")
        print("✓ /agents toggles the panel; empty snapshot → one placeholder row")


async def scenario_detail_tail() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "agent-x.jsonl"
        jsonl.write_text(json.dumps(
            {"type": "assistant",
             "message": {"content": [{"type": "text", "text": "hello from agent"}]}}
        ) + "\n")
        orig = agent_tail.resolve_task_file

        def fake_resolve(row, cwd, sid, **kw):
            return (jsonl, "direct") if row.get("task_id") == "t1" else (None, "none")

        # app.py imported the symbol directly — patch it there.
        import app as app_mod
        app_mod.resolve_task_file = fake_resolve
        try:
            app = EngramApp(driver=FakeDriver())
            async with app.run_test() as pilot:
                await pilot.press("ctrl+t")
                app.post_message(PromptArea.Submitted("go"))
                await wait_until(lambda: len(app._agent_rows) == 3, pilot, "rows")
                # enter on the empty prompt opens the detail for t1
                await pilot.press("enter")
                await wait_until(lambda: app._detail_id == "t1", pilot, "detail open")
                view = app.query_one("#agentview", RichLog)
                assert view.display
                # first tick seeds + renders the existing line
                app._tick_tail()
                await pilot.pause()
                lines = "\n".join(str(s) for s in view.lines)
                assert "hello from agent" in lines
                # append → next tick picks it up (no wall-clock wait)
                with open(jsonl, "a") as f:
                    f.write(json.dumps(
                        {"type": "assistant",
                         "message": {"content": [{"type": "text",
                                                  "text": "second line"}]}}) + "\n")
                app._tick_tail()
                await pilot.pause()
                lines = "\n".join(str(s) for s in view.lines)
                assert "second line" in lines
                # enter again closes the detail; tail stops
                await pilot.press("enter")
                assert app._detail_id is None and app._tail is None
                assert not view.display
                # unresolvable row (t2 parent) → state-only card, no crash
                await pilot.press("down")
                await pilot.press("enter")
                await wait_until(lambda: app._detail_id == "t2", pilot, "t2 detail")
                assert app._tail is None
                lines = "\n".join(str(s) for s in
                                  app.query_one("#agentview", RichLog).lines)
                assert "state only" in lines
                print("✓ detail pane: resolve→card→tail (seed + incremental via direct "
                      "_tick_tail); toggle closes; unresolvable row → state-only card")
        finally:
            app_mod.resolve_task_file = orig


async def scenario_enter_with_text_submits() -> None:
    d = FakeDriver()
    app = EngramApp(driver=d)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+t")
        app.post_message(PromptArea.Submitted("go"))
        await wait_until(lambda: len(d.calls) == 1, pilot, "first turn")
        prompt = app.query_one("#prompt", PromptArea)
        prompt.load_text("second message")
        await pilot.press("enter")
        await wait_until(lambda: len(d.calls) == 2 or len(app._queue) == 1, pilot,
                         "enter with text still submits (or queues while busy)")
        assert app._agents_open, "panel stays open while chatting"
        print("✓ enter with text submits a normal message; panel stays open")


async def main() -> None:
    await scenario_toggle_and_navigate()
    await scenario_slash_agents_and_placeholder()
    await scenario_detail_tail()
    await scenario_enter_with_text_submits()
    print("\nALL PASS")


if __name__ == "__main__":
    asyncio.run(main())
