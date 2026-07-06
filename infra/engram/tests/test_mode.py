#!/usr/bin/env python3
"""Headless tests for plan ↔ regular mode cycling (shift+tab · /mode).

Covers (1) the shift+tab KEYPRESS through a real Textual pilot flips the driver's
permission mode and the header subtitle, (2) /mode toggles and sets explicitly, and
(3) shift+tab mid-reply ARMS the mode (driver untouched until the turn ends, then
applied — governing the next turn). Same FakeDriver + pilot style as test_commands.py.

    .venv/bin/python infra/engram/tests/test_mode.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from core import (  # noqa: E402
    AgentSDKDriver, Event, ModelDriver, PLAN_MODE, REGULAR_MODE,
)
from app import PromptArea, EngramApp  # noqa: E402


class FakeDriver(ModelDriver):
    """query() blocks on a per-call gate (to hold a turn open); set_permission_mode
    records the live switch — exactly the seam app.py drives."""
    def __init__(self):
        self.model = "opus[1m]"
        self.effort = "max"
        self.session_id = None
        self.actual_model = None
        self.resumed = False
        self.stderr_tail = ""
        self.permission_mode = REGULAR_MODE
        self.mode_calls: list[str] = []
        self.gates: list[asyncio.Event] = []

    async def query(self, text, *, prepend=""):
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        yield Event("text", "ok")

    async def set_permission_mode(self, mode):
        self.permission_mode = mode
        self.mode_calls.append(mode)

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


async def scenario_shift_tab() -> None:
    driver = FakeDriver()
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Starts in regular; the subtitle says nothing about plan.
        assert driver.permission_mode == REGULAR_MODE
        assert "plan" not in app._subtitle(), app._subtitle()
        # The real keypress (prompt is focused on mount) → plan (no ask stop anymore).
        await pilot.press("shift+tab")
        await wait_until(lambda: driver.permission_mode == PLAN_MODE, pilot, "shift+tab → plan")
        assert "plan" in app._subtitle(), app._subtitle()
        assert "plan mode" in widget_status(app).lower(), widget_status(app)
        # Again → back to regular (full cycle).
        await pilot.press("shift+tab")
        await wait_until(lambda: driver.permission_mode == REGULAR_MODE, pilot, "shift+tab → regular")
        assert "plan" not in app._subtitle(), app._subtitle()
    print("✓ shift+tab cycles regular → plan → regular (driver mode + subtitle follow)")


async def scenario_mode_command() -> None:
    driver = FakeDriver()
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("/mode plan"))
        await wait_until(lambda: driver.permission_mode == PLAN_MODE, pilot, "/mode plan")
        app.post_message(PromptArea.Submitted("/mode regular"))
        await wait_until(lambda: driver.permission_mode == REGULAR_MODE, pilot, "/mode regular")
        app.post_message(PromptArea.Submitted("/mode"))            # bare → cycle: regular → plan
        await wait_until(lambda: driver.permission_mode == PLAN_MODE, pilot, "/mode cycle")
        # Garbage arg → usage hint, mode unchanged.
        app.post_message(PromptArea.Submitted("/mode sideways"))
        await pilot.pause()
        assert driver.permission_mode == PLAN_MODE, "bad /mode arg must not change the mode"
        assert "usage" in widget_status(app).lower(), widget_status(app)
    print("✓ /mode sets plan|regular explicitly, bare /mode cycles, bad arg → usage")


async def scenario_arm_mid_reply() -> None:
    driver = FakeDriver()
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("hello"))            # start a turn, hold it open
        await wait_until(lambda: app._busy, pilot, "turn to start")
        await pilot.press("shift+tab")                            # arm plan mid-reply
        await pilot.pause()
        # Driver NOT switched yet (a live control request would race the stream)…
        assert driver.mode_calls == [], "mode must not switch live mid-reply"
        assert app._pending_mode == PLAN_MODE, app._pending_mode
        assert "plan" in app._subtitle(), "subtitle should preview the armed mode"
        # …it applies the instant the turn ends.
        driver.gates[0].set()
        await wait_until(lambda: driver.permission_mode == PLAN_MODE, pilot, "armed mode applied")
        assert app._pending_mode is None
    print("✓ shift+tab mid-reply arms the mode; it applies when the turn ends (next turn)")


async def scenario_leaving_plan_recycles_client() -> None:
    """The wedge fix (driver level): AgentSDKDriver.set_permission_mode must RECYCLE the
    warm client — drop it so the next turn reconnects in the new mode — not fire a live
    control request. The live switch can't release plan mode and would leave you stuck in
    plan with no way out from the UI. Mirrors set_effort / set_model."""
    class FakeClient:
        def __init__(self): self.disconnected = False
        async def disconnect(self): self.disconnected = True

    driver = AgentSDKDriver(store=None)
    fc = FakeClient()
    driver._client = fc
    driver.permission_mode = PLAN_MODE
    await driver.set_permission_mode(REGULAR_MODE)        # leave plan
    assert fc.disconnected, "leaving plan must drop the warm client"
    assert driver._client is None, "client must be recycled (so the next turn reconnects)"
    assert driver.permission_mode == REGULAR_MODE
    print("✓ set_permission_mode recycles the client — no live-switch plan-mode wedge")


async def scenario_plan_approval_restores_pre_plan_mode() -> None:
    """The prompt-hell fix (2026-07-03): approving a plan releases plan mode at the CLI
    level — but into the CLI's own "default" mode, NOT the bypass mode you act in. The
    driver must remember the pre-plan mode and restore it LIVE on approval so the
    implementation turn runs free."""
    class FakeClient:
        def __init__(self): self.modes: list[str] = []
        async def disconnect(self): ...
        async def set_permission_mode(self, mode): self.modes.append(mode)

    async def approve(req): return {"approved": True}

    # Planned from regular: approval restores regular, via a live control request.
    driver = AgentSDKDriver(store=None)
    driver.permission_mode = REGULAR_MODE
    await driver.set_permission_mode(PLAN_MODE)            # enter plan (from regular)
    assert driver.plan_restore_target == REGULAR_MODE
    fc = FakeClient()
    driver._client = fc                                    # the client reconnected in plan
    driver.on_interaction = approve
    res = await driver._can_use_tool("ExitPlanMode", {"plan": "do it"}, None)
    assert res is not None
    assert driver.permission_mode == REGULAR_MODE, "approval must restore the PRE-plan mode"
    assert fc.modes == [REGULAR_MODE], f"live restore expected, got {fc.modes}"

    # Manually cycling OUT of plan consumes the memory (next approval can't restore stale).
    driver3 = AgentSDKDriver(store=None)
    driver3.permission_mode = REGULAR_MODE
    await driver3.set_permission_mode(PLAN_MODE)
    await driver3.set_permission_mode(REGULAR_MODE)        # manual exit, no approval
    assert driver3._pre_plan_mode is None, "manual exit must clear the pre-plan memory"
    print("✓ plan approval restores the pre-plan mode (live restore to regular)")


def widget_status(app) -> str:
    from textual.widgets import Static
    w = app.query_one("#status", Static)
    for attr in ("_content", "renderable", "_renderable"):
        v = getattr(w, attr, None)
        if v is not None:
            return str(v)
    return str(w.render())


async def main() -> int:
    await scenario_shift_tab()
    await scenario_mode_command()
    await scenario_arm_mid_reply()
    await scenario_leaving_plan_recycles_client()
    await scenario_plan_approval_restores_pre_plan_mode()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
