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

from app import render_tasks_line  # noqa: E402
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ToolUseBlock,
)
from core import AgentSDKDriver  # noqa: E402


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
                           session_id="s", tool_use_id="tuA"),
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
    # aurora m2: identity rides every snapshot so the panel can resolve the
    # on-disk transcript (task_id from the registry key, tool_use_id from start)
    assert task_evs[0].data["tasks"][0]["task_id"] == "tA"
    assert task_evs[0].data["tasks"][0]["tool_use_id"] == "tuA"
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


WF_PROGRESS = [
    {"type": "workflow_phase", "index": 1, "title": "Scan"},
    {"type": "workflow_agent", "index": 1, "label": "scan a", "phaseIndex": 1,
     "phaseTitle": "Scan", "model": "claude-sonnet-4-6", "state": "done"},
    {"type": "workflow_agent", "index": 2, "label": "scan b", "phaseIndex": 1,
     "phaseTitle": "Scan", "model": "claude-opus-4-8", "state": "progress"},
]


async def test_workflow_task_labeled_backgrounded_and_snapshotted():
    d = AgentSDKDriver(store=None)
    d._client = FakeClient([
        TaskStartedMessage(subtype="task_started",
                           data={"task_type": "local_workflow",
                                 "workflow_name": "audit-routes"},
                           task_id="wf1", description="audit the routes",
                           uuid="u", session_id="s", tool_use_id="tu9"),
        TaskProgressMessage(subtype="task_progress",
                            data={"workflow_progress": WF_PROGRESS},
                            task_id="wf1", description="Scan: scan b",
                            usage={"total_tokens": 40100, "tool_uses": 3,
                                   "duration_ms": 900},
                            uuid="u", session_id="s", last_tool_name="scan b"),
        AM(TextBlock(text="running — I'll report back")),
        RESULT(),
    ])
    evs = [ev async for ev in d._stream("hi")]
    # Named + flagged as a workflow and tracked as BACKGROUND — the run outlives
    # the turn by design, so it must never hold the prompt open via `pending`.
    assert d.tasks["wf1"]["name"] == "⚙ audit-routes", d.tasks
    assert d.tasks["wf1"]["workflow"] is True
    assert "wf1" in d._bg_tasks and d.has_background_tasks
    launch = [e for e in evs if e.kind == "text" and "workflow" in e.text]
    assert launch and "audit-routes" in launch[0].text
    # The progress heartbeat snapshots the phase/agent tree for the panel.
    snap = d.tasks["wf1"]["wf"]
    assert snap["done"] == 1 and snap["total"] == 2 and snap["phase"] == "Scan"
    assert snap["phases"][0]["agents"][1]["state"] == "progress"
    assert any("1/2 agents" in e.text for e in evs if e.kind == "status")
    print("✓ workflow task: ⚙-named, background-tracked, phase/agent tree snapshotted")


def test_workflow_snapshot_shapes():
    from core import workflow_snapshot
    assert workflow_snapshot([]) == {"phases": [], "done": 0, "total": 0,
                                     "phase": ""}
    snap = workflow_snapshot([{"type": "workflow_agent", "label": "solo",
                               "phaseTitle": "Fix", "state": "done"}])
    assert snap["total"] == snap["done"] == 1 and snap["phase"] == "Fix"
    print("✓ workflow_snapshot: empty + phase-less agent lists both hold shape")


def test_workflow_tool_label():
    from core import _tool_label
    blk = ToolUseBlock(id="t", name="Workflow", input={
        "script": "export const meta = {\n  name: 'find-bugs',\n"})
    assert _tool_label(blk) == "Workflow→find-bugs"
    assert _tool_label(ToolUseBlock(id="t2", name="Workflow", input={})) == "Workflow"
    print("✓ Workflow tool-use labels with the script's meta name")


def test_render_tasks_line_workflow():
    line = render_tasks_line([], [
        {"name": "⚙ audit-routes", "status": "running", "workflow": True,
         "tokens": 40100, "wf": {"phase": "Scan", "done": 1, "total": 2}}])
    assert "⚙ audit-routes ⏳ Scan 1/2 40k" in line, line
    print("✓ panel renderer: workflow row shows phase + agents done/total")


async def main() -> int:
    await test_todowrite_yields_todos_event()
    await test_task_registry_and_events()
    test_render_tasks_line()
    test_render_tasks_line_nine_done_collapse()
    await test_query_prunes_terminal_tasks()
    await test_workflow_task_labeled_backgrounded_and_snapshotted()
    test_workflow_snapshot_shapes()
    test_workflow_tool_label()
    test_render_tasks_line_workflow()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
