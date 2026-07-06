#!/usr/bin/env python3
"""Unit tests for task/todo visibility: TodoWrite → 'todos' Events, the session
task registry → 'task' Events, and the pure panel renderer.

    .venv/bin/python infra/engram/tests/test_tasks.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from core import AgentSDKDriver  # noqa: E402
from app import render_tasks_line  # noqa: E402
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage, ResultMessage, TaskNotificationMessage, TaskProgressMessage,
    TaskStartedMessage, TextBlock, ToolUseBlock,
)


def AM(*blocks):
    return AssistantMessage(content=list(blocks), model="m")


def RESULT():
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                         is_error=False, num_turns=1, session_id="s")


class FakeClient:
    def __init__(self, messages):
        self._messages = messages

    async def query(self, text, *, prepend=""):
        pass

    async def receive_messages(self):
        for m in self._messages:
            yield m


TODOS = [{"content": "map code", "status": "completed", "activeForm": "mapping code"},
         {"content": "write fix", "status": "in_progress", "activeForm": "writing the fix"},
         {"content": "run tests", "status": "pending", "activeForm": "running tests"}]


async def test_todowrite_yields_todos_event():
    d = AgentSDKDriver(store=None)
    d._client = FakeClient([
        AM(ToolUseBlock(id="t1", name="TodoWrite", input={"todos": TODOS})),
        AM(TextBlock(text="on it")),
        RESULT(),
    ])
    evs = [ev async for ev in d._stream("hi")]
    todos = [e for e in evs if e.kind == "todos"]
    assert len(todos) == 1 and todos[0].data["todos"] == TODOS
    assert not any(e.kind == "tool" and "TodoWrite" in e.text for e in evs), \
        "the panel is the render — no status blip for TodoWrite"
    print("✓ TodoWrite → one 'todos' Event with the list; no tool blip")


async def test_task_registry_and_events():
    d = AgentSDKDriver(store=None)
    d._client = FakeClient([
        TaskStartedMessage(subtype="task_started", data={"subagent_type": "Explore"},
                           task_id="tA", description="find callers", uuid="u",
                           session_id="s"),
        TaskProgressMessage(subtype="task_progress", data={}, task_id="tA",
                            description="running",
                            usage={"total_tokens": 12400, "tool_uses": 2,
                                   "duration_ms": 900},
                            uuid="u", session_id="s", last_tool_name="Grep"),
        TaskNotificationMessage(subtype="task_notification", data={}, task_id="tA",
                                status="completed", output_file="/x",
                                summary="done", uuid="u", session_id="s"),
        AM(TextBlock(text="synthesis")),
        RESULT(),
    ])
    evs = [ev async for ev in d._stream("hi")]
    task_evs = [e for e in evs if e.kind == "task"]
    assert len(task_evs) == 3, [e.kind for e in evs]
    assert task_evs[0].data["tasks"][0]["status"] == "running"
    assert task_evs[1].data["tasks"][0]["tokens"] == 12400
    assert task_evs[1].data["tasks"][0]["last_tool"] == "Grep"
    assert task_evs[2].data["tasks"][0]["status"] == "completed"
    assert d.tasks["tA"]["status"] == "completed", "registry persists after the turn"
    d.reset()
    assert d.tasks == {}, "reset clears the registry with the thread"
    print("✓ task registry: start → progress → terminal snapshots; persists; reset clears")


def test_render_tasks_line():
    line = render_tasks_line(TODOS, [
        {"name": "Explore", "status": "running", "tokens": 12400},
        {"name": "Plan", "status": "completed"},
        {"name": "general-purpose", "status": "failed"},
    ])
    assert "☑ 1/3" in line and "▶ writing the fix" in line, line
    assert "🛰 Explore ⏳ 12k" in line, line
    # Finished agents never stack up by name — they collapse to counters
    # (the transcript's inline ✓/✗ lines are the record).
    assert "Plan" not in line and "general-purpose" not in line, line
    assert "✓ 1 done" in line and "✗ 1 failed" in line, line
    assert render_tasks_line([], []) == ""
    print("✓ panel renderer: live agents listed, finished ones collapse to counters")


def test_render_tasks_line_nine_done_collapse():
    # The reported bug: 9 completed sub-agents stacked above the vision card.
    nine = [{"name": f"agent-{i}", "status": "completed"} for i in range(9)]
    line = render_tasks_line([], nine)
    assert line == "🛰 ✓ 9 done", line
    print("✓ nine finished agents render as one counter cell, not nine rows")


async def test_query_prunes_terminal_tasks():
    d = AgentSDKDriver(store=None)
    d.tasks = {"old-done": {"name": "Explore", "status": "completed"},
               "old-dead": {"name": "Plan", "status": "failed"},
               "still-going": {"name": "bg", "status": "running"}}

    async def _noop():
        return None
    d.connect = _noop

    async def _one_text(text):
        from core import Event
        yield Event("text", "hi")
    d._stream = _one_text

    async for _ev in d.query("next turn"):
        pass
    assert set(d.tasks) == {"still-going"}, d.tasks
    print("✓ query() prunes finished sub-agents at the turn boundary; live ones stay")


async def main() -> int:
    await test_todowrite_yields_todos_event()
    await test_task_registry_and_events()
    test_render_tasks_line()
    test_render_tasks_line_nine_done_collapse()
    await test_query_prunes_terminal_tasks()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
