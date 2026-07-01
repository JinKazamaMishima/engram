#!/usr/bin/env python3
"""Unit test for `_stream`'s async sub-agent capture (the auto-delegation path).

The CLI's Agent tool runs sub-agents asynchronously: their progress + completion
arrive as Task* messages AFTER the parent turn's ResultMessage. `_stream` must keep
reading until they finish, stream only TOP-LEVEL text (incl. the main agent's final
synthesis) while skipping the sub-agent's own monologue, surface lifecycle markers +
progress, and still stop at the first ResultMessage on a normal (no sub-agent) turn.

We drive `_stream` directly with a fake client that yields scripted real SDK messages.

    .venv/bin/python infra/engram/tests/test_async_subagent.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import core  # noqa: E402  (for monkeypatching the idle timeout)
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TextBlock,
    ToolUseBlock,
)
from core import AgentSDKDriver  # noqa: E402


def AM(text=None, *, tool=None, parent=None):
    blocks = []
    if text is not None:
        blocks.append(TextBlock(text=text))
    if tool is not None:
        blocks.append(ToolUseBlock(id=tool[0], name=tool[1], input=tool[2]))
    return AssistantMessage(content=blocks, model="m", parent_tool_use_id=parent)


def RESULT(sid="s"):
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                         is_error=False, num_turns=1, session_id=sid)


class FakeClient:
    """Yields a scripted message list; optionally hangs at the end (a silent,
    still-running sub-agent) to exercise the idle-timeout detach path."""
    def __init__(self, messages, hang=False):
        self._messages = messages
        self._hang = hang

    async def query(self, text):
        pass

    async def receive_messages(self):
        for m in self._messages:
            yield m
        if self._hang:
            await asyncio.Event().wait()


async def collect(messages, hang=False):
    d = AgentSDKDriver(store=None)
    d._client = FakeClient(messages, hang=hang)
    return [ev async for ev in d._stream("hi")]


def texts(evs):
    return "".join(e.text for e in evs if e.kind == "text")


async def test_subagent_capture():
    AGENT_TU = "toolu_xyz"
    msgs = [
        SystemMessage(subtype="init", data={"session_id": "s", "model": "opus[1m]"}),
        AM("I'll delegate this."),                                  # top-level narration
        AM(tool=(AGENT_TU, "Agent", {"subagent_type": "Explore",
                                     "description": "Count files"})),  # top-level tool-use
        TaskStartedMessage(subtype="task_started", data={"subagent_type": "Explore"},
                           task_id="t1", description="Count files", uuid="u1", session_id="s"),
        RESULT(),                                                   # parent result — keep going
        AM("SECRET sub-agent monologue", parent=AGENT_TU),          # sub-internal — must skip
        TaskProgressMessage(subtype="task_progress", data={"subagent_type": "Explore"},
                            task_id="t1", description="running",
                            usage={"total_tokens": 8454, "tool_uses": 1, "duration_ms": 2578},
                            uuid="u2", session_id="s", last_tool_name="Bash"),
        TaskNotificationMessage(subtype="task_notification", data={}, task_id="t1",
                                status="completed", output_file="/x",
                                summary='Agent "Count files" finished', uuid="u3", session_id="s"),
        AM("The sub-agent reports 4 files."),                       # top-level synthesis
        RESULT(),                                                   # no pending — STOP here
        AM("LEAKED — must not be read"),                            # past the stop point
    ]
    evs = await collect(msgs)
    body = texts(evs)
    assert "I'll delegate this." in body, body
    assert "delegated to **Explore**" in body and "Count files" in body, body
    assert 'Agent "Count files" finished' in body and "✓" in body, body
    assert "The sub-agent reports 4 files." in body, "must read PAST the parent ResultMessage"
    assert "SECRET sub-agent monologue" not in body, "sub-agent-internal text must be skipped"
    assert "LEAKED" not in body, "must stop at the ResultMessage once no task is pending"
    assert any(e.kind == "tool" and e.text == "Agent→Explore: Count files" for e in evs), evs
    prog = [e.text for e in evs if e.kind == "status"]
    assert any("Explore" in p and "Bash" in p and "8,454 tok" in p for p in prog), prog
    print("✓ auto-delegation: reads past parent result, streams synthesis, skips monologue, marks lifecycle")


async def test_normal_turn_stops_at_first_result():
    msgs = [
        SystemMessage(subtype="init", data={"session_id": "s"}),
        AM("hello"),
        RESULT(),                          # no sub-agent → stop immediately
        AM("LEAKED — must not be read"),
    ]
    body = texts(await collect(msgs))
    assert "hello" in body and "LEAKED" not in body, body
    print("✓ normal turn still stops at the first ResultMessage (no regression)")


async def test_idle_detach():
    core.SUBAGENT_IDLE_TIMEOUT = 0.2       # don't hang the test
    msgs = [
        AM("working on it"),
        TaskStartedMessage(subtype="task_started", data={"subagent_type": "Explore"},
                           task_id="t1", description="long job", uuid="u1", session_id="s"),
        RESULT(),                          # pending non-empty → keep waiting … then silence
    ]
    body = texts(await collect(msgs, hang=True))
    assert "detached" in body and "Explore" in body, body
    print("✓ a silent, still-running sub-agent is detached (turn doesn't hang)")


async def main() -> int:
    await test_subagent_capture()
    await test_normal_turn_stops_at_first_result()
    await test_idle_detach()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
