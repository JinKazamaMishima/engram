#!/usr/bin/env python3
"""Headless tests for ESC-interrupt + block-seam reply formatting.

Covers (1) ESC while a reply streams stops the turn — the driver's interrupt fires,
the held turn unwinds, and _busy clears; (2) ESC is a no-op when idle; and (3)
consecutive streamed text blocks render as separate paragraphs — a block ending in
':' no longer glues onto the next on the same line. Same FakeDriver + pilot style as
test_mode.py / test_commands.py.

    .venv/bin/python infra/engram/tests/test_interrupt.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from core import Event, ModelDriver, REGULAR_MODE  # noqa: E402
from app import PromptArea, EngramApp  # noqa: E402


class FakeDriver(ModelDriver):
    """query() either blocks on a gate (to hold a turn open for the interrupt test)
    or replays a fixed list of text blocks (for the seam test); interrupt() records
    the call and releases the gate — exactly the seam app.py's ESC path drives."""
    def __init__(self, script=None, block=False):
        self.model = "opus[1m]"
        self.effort = "max"
        self.session_id = None
        self.actual_model = None
        self.resumed = False
        self.stderr_tail = ""
        self.permission_mode = REGULAR_MODE
        self.script = script or []
        self.block = block
        self.interrupted = False
        self.gate = asyncio.Event()

    async def query(self, text, *, prepend=""):
        if self.block:
            await self.gate.wait()           # hold the turn open until interrupt()
            return                           # released → stream ends (no more text)
        for chunk in self.script:
            yield Event("text", chunk)

    async def interrupt(self):
        self.interrupted = True
        self.gate.set()                      # release a held turn so it can unwind

    async def set_permission_mode(self, mode): self.permission_mode = mode
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


async def scenario_esc_interrupts() -> None:
    driver = FakeDriver(block=True)
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("hello"))     # start a turn, held open
        await wait_until(lambda: app._busy, pilot, "turn to start")
        await pilot.press("escape")                          # ESC mid-reply → stop now
        await wait_until(lambda: driver.interrupted, pilot, "interrupt sent")
        await wait_until(lambda: not app._busy, pilot, "turn to unwind")
    print("✓ ESC mid-reply interrupts the turn (driver.interrupt fires, _busy clears)")


async def scenario_esc_idle_noop() -> None:
    driver = FakeDriver(block=True)
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")                          # nothing running
        await pilot.pause()
        assert not driver.interrupted, "ESC must be a no-op when idle"
        assert not app._busy
    print("✓ ESC is a no-op when idle (no interrupt, nothing breaks)")


async def scenario_block_seam() -> None:
    driver = FakeDriver(script=["Here's the result:", "It works."])
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("go"))
        await wait_until(lambda: app._last_reply, pilot, "reply to land")
        reply = app._last_reply
        assert "result:\n\nIt works." in reply, repr(reply)  # paragraph break inserted
        assert "result:It works" not in reply, "blocks must not glue on one line"
    print("✓ consecutive text blocks get a paragraph break (no ':'-glued same line)")


async def main() -> int:
    await scenario_esc_interrupts()
    await scenario_esc_idle_noop()
    await scenario_block_seam()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
