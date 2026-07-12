"""Headless pilot test for the Engram Textual TUI — proves the UI plumbing
(input → background turn → streamed render) with a FAKE driver, so no model call
is made. The real AgentSDKDriver is smoke-tested separately on the subscription."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

pytest.importorskip("textual")          # optional engram-harness deps; skip if absent
pytest.importorskip("claude_agent_sdk")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "infra", "engram"))

import app as engram_app  # noqa: E402
from attach import parse_dropped_paths  # noqa: E402
from core import AgentSDKDriver, Event, ModelDriver  # noqa: E402
from textual.widgets import Markdown, Static  # noqa: E402


class FakeDriver(ModelDriver):
    """Canned events; records what it was asked. No network, no model."""
    model = "fake"
    effort = "low"
    cwd = "/x"
    stderr_tail = ""

    def __init__(self) -> None:
        self.session_id = None
        self.queried: list[str] = []
        self.reset_calls = 0

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    def reset(self) -> None:
        self.reset_calls += 1
        self.session_id = None

    async def query(self, text: str, *, prepend: str = ""):
        self.queried.append(text)
        yield Event("tool", "Read")
        yield Event("text", "hello from ")
        yield Event("text", "engram")


def test_engram_tui_runs_a_turn_headless():
    async def scenario():
        drv = FakeDriver()
        app = engram_app.EngramApp(driver=drv)
        async with app.run_test() as pilot:
            p = app.query_one("#prompt", engram_app.PromptArea)
            p.focus()
            p.load_text("hi engram")
            await pilot.press("enter")
            for _ in range(100):                 # let the worker finish
                if drv.queried and not app._busy:
                    break
                await pilot.pause(0.02)
            assert drv.queried == ["hi engram"]    # the turn reached the driver
            assert app._busy is False            # and completed cleanly
            assert list(app.query(Markdown))     # an assistant reply was mounted
    asyncio.run(scenario())


def test_engram_tui_new_thread_resets_driver():
    async def scenario():
        drv = FakeDriver()
        app = engram_app.EngramApp(driver=drv)
        async with app.run_test() as pilot:
            p = app.query_one("#prompt", engram_app.PromptArea)
            p.focus()
            p.load_text("/new")
            await pilot.press("enter")
            await pilot.pause()
            assert drv.reset_calls == 1          # /new reset the conversation
            assert drv.queried == []             # and did NOT call the model
    asyncio.run(scenario())


def test_parse_dropped_paths(tmp_path):
    f = tmp_path / "shot.png"
    f.write_text("x")
    g = tmp_path / "doc.txt"
    g.write_text("y")
    sp = tmp_path / "a b.txt"          # path with a space
    sp.write_text("z")
    assert parse_dropped_paths(f"'{f}'") == [f]                 # single-quoted
    assert parse_dropped_paths(f"file://{g}") == [g]            # file:// URI
    assert parse_dropped_paths(f"'{sp}'") == [sp]              # quoted, has a space
    assert set(parse_dropped_paths(f"'{f}' '{g}'")) == {f, g}  # multi-file drop
    assert parse_dropped_paths("/nope/missing.xyz") == []       # nonexistent → dropped
    assert parse_dropped_paths("just some words") == []         # plain text → none


def test_engram_tui_attach_then_send_injects_path(tmp_path):
    img = tmp_path / "pic.png"
    img.write_text("fake")

    async def scenario():
        drv = FakeDriver()
        app = engram_app.EngramApp(driver=drv)
        async with app.run_test() as pilot:
            app.attach_files([img])                # drop / paste a file path
            await pilot.pause()
            assert img in app._attachments
            app.query_one("#chips", Static)        # chip area mounted; _render_chips ran
            p = app.query_one("#prompt", engram_app.PromptArea)
            p.focus()
            p.load_text("what is this?")
            await pilot.press("enter")
            for _ in range(100):
                if drv.queried and not app._busy:
                    break
                await pilot.pause(0.02)
            assert drv.queried, "no turn was sent"
            assert str(img) in drv.queried[0]      # path folded in for the Read tool
            assert app._attachments == []          # cleared after sending
    asyncio.run(scenario())


# ---- /btw — steer the in-flight turn ---------------------------------------

class GatedDriver(ModelDriver):
    """Turns block on per-call gates (so tests control busy); records inject()
    asides. For the /btw mid-turn steering tests."""
    model = "fake"
    effort = "low"
    cwd = "/x"
    stderr_tail = ""

    def __init__(self) -> None:
        self.session_id = None
        self.calls: list[str] = []
        self.gates: list[asyncio.Event] = []
        self.injects: list[str] = []
        self.inject_ok = True

    async def disconnect(self) -> None: ...
    def reset(self) -> None: ...

    async def query(self, text: str, *, prepend: str = ""):
        self.calls.append(text)
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        yield Event("text", "ok")

    async def inject(self, text: str) -> bool:
        self.injects.append(text)
        return self.inject_ok


async def _submit(app, pilot, text: str) -> None:
    p = app.query_one("#prompt", engram_app.PromptArea)
    p.focus()
    p.load_text(text)
    await pilot.press("enter")


async def _until(pilot, cond, what: str) -> None:
    for _ in range(150):
        if cond():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timeout waiting for: {what}")


def test_btw_mid_turn_steers_without_queue_or_interrupt():
    async def scenario():
        drv = GatedDriver()
        app = engram_app.EngramApp(driver=drv)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "long job")
            await _until(pilot, lambda: drv.calls, "turn to start")
            assert app._busy is True
            await _submit(app, pilot, "/btw check the dates")
            await _until(pilot, lambda: drv.injects, "aside to reach the driver")
            assert drv.injects == ["check the dates"]   # bare note, prefix stripped
            assert app._queue == []                     # steered, NOT queued
            assert drv.calls == ["long job"]            # no second turn started
            assert app._busy is True                    # and the turn wasn't interrupted
            drv.gates[0].set()
            await _until(pilot, lambda: not app._busy, "turn to finish")
    asyncio.run(scenario())


def test_btw_falls_back_to_queue_when_inject_refused():
    async def scenario():
        drv = GatedDriver()
        drv.inject_ok = False                           # turn ends in the same instant
        app = engram_app.EngramApp(driver=drv)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "long job")
            await _until(pilot, lambda: drv.calls, "turn to start")
            await _submit(app, pilot, "/btw check the dates")
            await _until(pilot, lambda: app._queue, "aside to queue")
            assert app._queue == [("check the dates", [])]   # never lost
            drv.gates[0].set()
            await _until(pilot, lambda: len(drv.calls) == 2, "queued aside to dispatch")
            assert drv.calls[1] == "check the dates"
            drv.gates[1].set()
            await _until(pilot, lambda: not app._busy, "to go idle")
    asyncio.run(scenario())


def test_btw_idle_is_just_a_message():
    async def scenario():
        drv = GatedDriver()
        app = engram_app.EngramApp(driver=drv)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "/btw remember the tests")
            await _until(pilot, lambda: drv.calls, "message to dispatch")
            assert drv.calls == ["remember the tests"]   # normal turn, bare note
            assert drv.injects == []                     # no mid-turn write
            drv.gates[0].set()
            await _until(pilot, lambda: not app._busy, "turn to finish")
    asyncio.run(scenario())


def test_btw_bare_shows_usage_only():
    async def scenario():
        drv = GatedDriver()
        app = engram_app.EngramApp(driver=drv)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "/btw")
            await pilot.pause()
            assert drv.calls == [] and drv.injects == [] and app._queue == []
    asyncio.run(scenario())


# ---- AgentSDKDriver.inject — the mid-turn stdin write itself --------------

class _StdinClient:
    """Records what inject() writes onto the live turn's stdin."""
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def query(self, prompt, session_id="default"):
        self.sent.append(prompt)


class _BufRecorder:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []

    def append(self, role: str, text: str) -> None:
        self.rows.append((role, text))


def test_driver_inject_gates_on_live_turn():
    async def scenario():
        d = AgentSDKDriver(store=None)
        d._client = _StdinClient()
        d._buffer = _BufRecorder()
        assert await d.inject("too early") is False        # idle → refuse (phantom turn)
        d._turn_live = True
        assert await d.inject("mid-turn note") is True
        assert d._client.sent == ["mid-turn note"]         # bare note on the live stdin
        assert d._buffer.rows == [("user", "mid-turn note")]   # log-raw invariant
        d._client = None
        assert await d.inject("during recycle") is False   # client mid-recycle → refuse
    asyncio.run(scenario())


def test_driver_turn_live_follows_query_lifecycle():
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    class _OneTurnClient:
        async def query(self, prompt, session_id="default"): ...

        async def receive_messages(self):
            yield AssistantMessage(content=[TextBlock(text="hi")], model="m")
            yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                                is_error=False, num_turns=1, session_id="s")

    async def scenario():
        d = AgentSDKDriver(store=None)
        d._client = _OneTurnClient()
        d._maybe_evict = lambda: None
        assert d._turn_live is False
        gen = d.query("hello")
        await gen.__anext__()
        assert d._turn_live is True         # live while the stream is open
        async for _ in gen:
            pass
        assert d._turn_live is False        # cleared in the finally, even on early exit
    asyncio.run(scenario())
