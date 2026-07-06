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
from core import Event, ModelDriver  # noqa: E402
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
