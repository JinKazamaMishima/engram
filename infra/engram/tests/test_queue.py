#!/usr/bin/env python3
"""Headless test: the type-ahead message queue in the Engram TUI.

Drives the real EngramApp with a fake driver whose turns are held open by per-call
gates, so we can: start turn A, submit B and C while busy (they must QUEUE, not
run), then release each turn and assert the queue drains in order A→B→C — plus
that a state-changing command typed mid-turn is blocked rather than queued.

    .venv/bin/python infra/engram/tests/test_queue.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from core import Event, ModelDriver  # noqa: E402
from app import PromptArea, EngramApp  # noqa: E402
from textual.widgets import Static  # noqa: E402


class FakeDriver(ModelDriver):
    """Each query() blocks on its own gate until the test releases it."""
    def __init__(self):
        self.model = "fake-model"
        self.effort = "max"
        self.session_id = None
        self.actual_model = None
        self.resumed = False
        self.stderr_tail = ""
        self.calls: list[str] = []
        self.gates: list[asyncio.Event] = []

    async def query(self, text, *, prepend=""):
        self.calls.append(text)
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()           # held open until the test releases this turn
        yield Event("text", f"reply to: {text}")

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


def queued_texts(app):
    return [t for t, _ in app._queue]


def strip_text(app):
    w = app.query_one("#queued", Static)
    for attr in ("_content", "renderable", "_renderable"):
        v = getattr(w, attr, None)
        if v is not None:
            return str(v)
    return str(w.render())


async def main() -> int:
    driver = FakeDriver()
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()

        # 1) Start turn A → it should begin running and mark busy.
        app.post_message(PromptArea.Submitted("message A"))
        await wait_until(lambda: len(driver.calls) == 1, pilot, "turn A to start")
        assert app._busy is True, "should be busy while A streams"
        assert driver.calls == ["message A"]

        # 2) Type B and C WHILE BUSY → must queue, not run, not be dropped.
        app.post_message(PromptArea.Submitted("message B"))
        app.post_message(PromptArea.Submitted("message C"))
        await pilot.pause()
        assert queued_texts(app) == ["message B", "message C"], \
            f"both should queue, got {queued_texts(app)}"
        assert len(driver.calls) == 1, "no new turn should start while busy"
        strip = strip_text(app)
        assert "⏳" in strip and "message B" in strip, f"queue strip wrong: {strip!r}"
        print("✓ messages typed mid-turn are QUEUED (not dropped), shown in the strip")

        # 3) Release A → B should auto-send; queue now [C].
        driver.gates[0].set()
        await wait_until(lambda: len(driver.calls) == 2, pilot, "B to auto-send")
        assert driver.calls[1] == "message B"
        assert queued_texts(app) == ["message C"], "C should remain queued"
        assert app._busy is True
        print("✓ on turn end, the next queued message auto-sends (A→B)")

        # 4) Release B → C should auto-send; queue empty.
        driver.gates[1].set()
        await wait_until(lambda: len(driver.calls) == 3, pilot, "C to auto-send")
        assert driver.calls[2] == "message C"
        assert queued_texts(app) == [], "queue should be empty"
        assert strip_text(app) == "", "strip should clear"
        print("✓ queue drains in order, strip clears when empty")

        # 5) Release C → idle.
        driver.gates[2].set()
        await wait_until(lambda: not app._busy, pilot, "to go idle")
        assert driver.calls == ["message A", "message B", "message C"], driver.calls

        # 6) A command typed while busy must be BLOCKED, not queued.
        gate_idx = len(driver.gates)
        app.post_message(PromptArea.Submitted("hold me open"))
        await wait_until(lambda: len(driver.calls) == 4, pilot, "a turn to hold")
        app.post_message(PromptArea.Submitted("/new"))
        await pilot.pause()
        assert queued_texts(app) == [], "/new must NOT be queued"
        assert len(driver.calls) == 4, "/new must not start a turn"
        driver.gates[gate_idx].set()
        await wait_until(lambda: not app._busy, pilot, "final idle")
        print("✓ state-changing command (/new) typed mid-turn is blocked, not queued")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
