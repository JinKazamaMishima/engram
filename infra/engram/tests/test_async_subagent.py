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
    TaskUpdatedMessage,
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


def TSTART(tid, desc="job", tool_use_id=None):
    """tool_use_id links the task to the Agent tool call that launched it — the
    driver uses it to recognize run_in_background launches."""
    return TaskStartedMessage(subtype="task_started", data={"subagent_type": "Explore"},
                              task_id=tid, description=desc, uuid="u", session_id="s",
                              tool_use_id=tool_use_id)


def TUPD(tid, status):
    """Inline lifecycle patch — the CLI emits status=completed here BEFORE the parent
    ResultMessage (this is the ordering that used to stop _stream too early)."""
    return TaskUpdatedMessage(subtype="task_updated", data={}, task_id=tid,
                              patch={"status": status}, status=status,
                              session_id="s", uuid="u")


def TNOTIF(tid, status, summary):
    """The async completion ping that re-invokes the model — arrives AFTER the parent result."""
    return TaskNotificationMessage(subtype="task_notification", data={}, task_id=tid,
                                   status=status, output_file="/x", summary=summary,
                                   uuid="u", session_id="s")


class FakeClient:
    """Yields a scripted message list; optionally hangs at the end (a silent,
    still-running sub-agent) to exercise the idle-timeout detach path."""
    def __init__(self, messages, hang=False):
        self._messages = messages
        self._hang = hang

    async def query(self, text, *, prepend=""):
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


async def test_timeout_converts_stragglers_to_background():
    """A sync delegation that goes silent past the idle timeout must release the
    turn with HONEST copy (no 'detached; result wasn't captured' — the result IS
    captured later) and keep tracking the straggler as a background task so the
    idle drain / next stream paints its result when it lands."""
    core.SUBAGENT_IDLE_TIMEOUT = 0.2       # don't hang the test
    msgs = [
        AM("working on it"),
        TSTART("t1", "long job"),          # sync (no bg link) → pending
        RESULT(),                          # pending non-empty → keep waiting … then silence
    ]
    d = AgentSDKDriver(store=None)
    d._client = FakeClient(msgs, hang=True)
    body = texts([ev async for ev in d._stream("hi")])
    assert "taking a while" in body and "Explore" in body, body
    assert "detached" not in body and "wasn't captured" not in body, body
    assert d._bg_tasks == {"t1"} and d.has_background_tasks, (d._bg_tasks, d._bg_owed)
    print("✓ a silent sync delegation times out into a BACKGROUND task (honest copy, still tracked)")


async def test_background_launch_releases_turn_with_marker():
    """An Agent call with run_in_background must NOT hold the turn open via
    `pending` — that hold + its idle timeout was the '⏳ detached; result wasn't
    captured this turn' noise. The turn ends at its own Result with a calm marker,
    and the task is tracked on the DRIVER for the idle drain."""
    core.SUBAGENT_IDLE_TIMEOUT = 0.5       # tripwire: correct code never waits at all
    TU = "toolu_bg"
    msgs = [
        SystemMessage(subtype="init", data={"session_id": "s"}),
        AM("Kicking off a background architect."),
        AM(tool=(TU, "Agent", {"subagent_type": "Explore", "description": "long research",
                               "run_in_background": True})),
        TSTART("tBG", "long research", tool_use_id=TU),   # background → NOT pending
        RESULT(),                                          # → STOP HERE, cleanly
    ]
    d = AgentSDKDriver(store=None)
    d._client = FakeClient(msgs, hang=True)               # silence after — must not matter
    body = texts([ev async for ev in d._stream("hi")])
    assert "delegated to **Explore** (background)" in body, body
    assert "1 background agent still working" in body, body
    assert "taking a while" not in body and "detached" not in body, body
    assert d._bg_tasks == {"tBG"} and d.has_background_tasks, (d._bg_tasks, d._bg_owed)
    print("✓ background launch: turn ends at its own result — calm marker, no detach noise")


async def test_background_result_paints_via_drain():
    """After the turn released, drain_background keeps reading: the ✓ ping, the
    model's follow-up turn, then returns once nothing more is owed."""
    d = AgentSDKDriver(store=None)
    d._task_names["tBG"] = "Explore"
    d._bg_tasks.add("tBG")
    d._client = FakeClient([
        TNOTIF("tBG", "completed", "research finished"),
        AM("Here is what the architect found."),          # notification-driven turn
        RESULT(),                                          # debt paid → drain returns
        AM("LEAKED — after the drain should have stopped"),
    ])
    body = texts([ev async for ev in d.drain_background()])
    assert "research finished" in body and "✓" in body, body
    assert "Here is what the architect found." in body, body
    assert "LEAKED" not in body, body
    assert not d.has_background_tasks, (d._bg_tasks, d._bg_owed)
    print("✓ drain paints the ✓ + the follow-up turn, then stops (nothing owed)")


async def test_background_completion_inside_a_typed_turn():
    """A background task finishing WHILE a typed turn streams: its ✓ renders
    inline, and the follow-up turn is still owed afterwards — has_background_tasks
    stays True so the idle drain runs next and reads it."""
    d = AgentSDKDriver(store=None)
    d._task_names["tBG"] = "Explore"
    d._bg_tasks.add("tBG")
    d._client = FakeClient([
        AM("answering your question"),
        TNOTIF("tBG", "completed", "bg agent finished"),   # lands mid-typed-turn
        RESULT(),                                          # typed turn ends (pending empty)
    ])
    body = texts([ev async for ev in d._stream("hi")])
    assert "bg agent finished" in body and "✓" in body, body
    assert d._bg_tasks == set() and d._bg_owed == 1 and d.has_background_tasks, \
        (d._bg_tasks, d._bg_owed)
    print("✓ bg completion mid-typed-turn: ✓ inline, follow-up turn still owed to the drain")


async def test_killed_background_task_clears_via_task_updated():
    """A killed background task may report ONLY via TaskUpdated (the CLI sometimes
    suppresses the notification) — it must clear from tracking (no eternal
    has_background_tasks) and paint ✗, with no follow-up turn owed."""
    d = AgentSDKDriver(store=None)
    d._task_names["tBG"] = "Explore"
    d._bg_tasks.add("tBG")
    d._client = FakeClient([
        AM("hello"),
        TUPD("tBG", "killed"),
        RESULT(),
    ])
    body = texts([ev async for ev in d._stream("hi")])
    assert "✗" in body and "killed" in body, body
    assert not d.has_background_tasks, (d._bg_tasks, d._bg_owed)
    print("✓ killed background task (TaskUpdated only) clears tracking, paints ✗")


async def test_real_ordering_inline_completion_before_parent_result():
    """Regression for the 'agents finished but no reply' bug. The LIVE CLI ordering
    (captured via probe): sub-agents run inline and emit a terminal TaskUpdatedMessage
    BEFORE the parent ResultMessage; the async TaskNotificationMessage arrives AFTER it and
    re-invokes the model to synthesise. _stream must NOT stop at the parent result — the
    real answer lands in the notification-driven turn after it."""
    A, B = "toolu_A", "toolu_B"
    msgs = [
        SystemMessage(subtype="init", data={"session_id": "s", "model": "opus[1m]"}),
        AM("Launching two agents."),                             # top-level narration
        AM(tool=(A, "Agent", {"subagent_type": "Explore", "description": "count py"})),
        TSTART("tA", "count py"),                                # pending={tA}
        AM(tool=(B, "Agent", {"subagent_type": "Explore", "description": "count md"})),
        TSTART("tB", "count md"),                                # pending={tA,tB}
        AM("AGENT_A_MONOLOGUE", parent=A),                       # sub-internal → skip
        AM("AGENT_B_MONOLOGUE", parent=B),                       # sub-internal → skip
        TUPD("tA", "completed"),                                 # inline — must NOT drain
        TUPD("tB", "completed"),                                 # inline — must NOT drain
        AM("Both launched; waiting on results."),               # parent narration
        RESULT(),                       # parent result — pending STILL {tA,tB} → MUST CONTINUE
        TNOTIF("tA", "completed", "Agent A finished"),          # async ping → drain tA, ✓
        AM("Agent A is done."),                                 # re-invocation synthesis
        RESULT(),                                               # pending {tB} → continue
        TNOTIF("tB", "completed", "Agent B finished"),          # async ping → drain tB, ✓
        AM("FINAL py=25 md=12"),                                # the real answer
        RESULT(),                                               # pending {} → STOP here
        AM("LEAKED past the true end"),
    ]
    body = texts(await collect(msgs))
    assert "Launching two agents." in body, body
    assert "Both launched; waiting on results." in body, body
    assert "FINAL py=25 md=12" in body, "must render the answer that lands AFTER the parent result"
    assert "Agent A is done." in body, body
    assert body.count("✓") == 2, f"one completion mark per task; got {body.count('✓')}"
    assert "AGENT_A_MONOLOGUE" not in body and "AGENT_B_MONOLOGUE" not in body, "skip monologue"
    assert "LEAKED" not in body, "must stop at the final ResultMessage (pending empty)"
    print("✓ real ordering: inline TaskUpdated doesn't stop the stream; post-result answer renders")


async def test_failed_subagent_via_task_updated_ends_promptly():
    """A sub-agent that fails/among terminal-but-not-completed via TaskUpdatedMessage (no
    notification) must clear pending so the turn ends at the next result instead of idling
    to the timeout."""
    msgs = [
        AM("Delegating."),
        TSTART("tX", "risky job"),                              # pending={tX}
        TUPD("tX", "failed"),                                   # terminal, non-completed → drain
        AM("It failed; moving on."),
        RESULT(),                                               # pending {} → stop
        AM("LEAKED"),
    ]
    body = texts(await collect(msgs))
    assert "It failed; moving on." in body and "✗" in body, body
    assert "LEAKED" not in body, body
    print("✓ failed delegation (TaskUpdated only) drains pending and ends at the next result")


async def main() -> int:
    await test_subagent_capture()
    await test_normal_turn_stops_at_first_result()
    await test_timeout_converts_stragglers_to_background()
    await test_background_launch_releases_turn_with_marker()
    await test_background_result_paints_via_drain()
    await test_background_completion_inside_a_typed_turn()
    await test_killed_background_task_clears_via_task_updated()
    await test_real_ordering_inline_completion_before_parent_result()
    await test_failed_subagent_via_task_updated_ends_promptly()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
