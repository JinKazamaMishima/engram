#!/usr/bin/env python3
"""Headless tests for the interactive tools — plan approval (ExitPlanMode) and option
questions (AskUserQuestion) — driven through the real Textual UI with a fake driver that
invokes the app's ``on_interaction`` seam exactly the way core.AgentSDKDriver._can_use_tool
does. Covers the five behaviours: render the plan, approve → leave plan mode, keep-planning
with feedback, pick an option, and type-your-own / chat. Same FakeDriver + pilot style as
test_mode.py.

    .venv/bin/python infra/engram/tests/test_interact.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from app import EngramApp, PromptArea  # noqa: E402
from core import PLAN_MODE, REGULAR_MODE, Event, ModelDriver  # noqa: E402
from textual.widgets import OptionList  # noqa: E402


class InteractDriver(ModelDriver):
    """query() calls the app's on_interaction handler (set by EngramApp) with a scripted
    request mid-turn — mirroring how the SDK's can_use_tool hands ExitPlanMode /
    AskUserQuestion to the front-end — records the verdict, then yields one text block."""
    def __init__(self, request):
        self.model = "opus[1m]"
        self.effort = "max"
        self.session_id = None
        self.actual_model = None
        self.resumed = False
        self.stderr_tail = ""
        self.on_interaction = None
        self.permission_mode = PLAN_MODE if request.get("kind") == "plan" else REGULAR_MODE
        self._request = request
        self.result = None            # what on_interaction returned (the verdict)
        self.mode_calls: list[str] = []

    async def query(self, text):
        self.result = await self.on_interaction(self._request)
        yield Event("text", "done")

    async def disconnect(self): ...
    def reset(self): self.session_id = None
    async def set_effort(self, level): self.effort = level
    async def set_model(self, name): self.model = name
    async def set_permission_mode(self, mode):
        self.permission_mode = mode
        self.mode_calls.append(mode)


async def wait_until(cond, pilot, what, limit=200):
    for _ in range(limit):
        if cond():
            return
        await pilot.pause()
    raise AssertionError(f"timeout waiting for: {what}")


def _card_present(app) -> bool:
    return len(app.query(".plancard")) >= 1


def _type(app, text: str) -> None:
    """Load free text into the (focused) prompt so an Enter routes it as the answer."""
    app.query_one("#prompt", PromptArea).load_text(text)


async def scenario_plan_approve() -> None:
    driver = InteractDriver({"kind": "plan", "plan": "# Plan\n\n- step one\n- step two\n"})
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert driver.permission_mode == PLAN_MODE
        app.post_message(PromptArea.Submitted("build the thing"))
        await wait_until(lambda: app._interact_open, pilot, "plan card open")
        assert app._busy, "the turn parks (busy) while the card is up"
        assert _card_present(app), "the plan is rendered as a markdown card"
        assert len(app.query(OptionList)) >= 1
        # Enter with an empty prompt → accept the highlighted option (Approve is first).
        await pilot.press("enter")
        await wait_until(lambda: driver.result is not None, pilot, "verdict returned")
        assert driver.result == {"approved": True}, driver.result
        assert driver.permission_mode == REGULAR_MODE, "approving must leave plan mode"
        assert "plan" not in app._subtitle(), app._subtitle()
        assert not app._interact_open
        await wait_until(lambda: not app._busy, pilot, "turn finishes after approve")
    print("✓ plan rendered as a card; Enter approves → {approved:True} + leaves plan mode")


async def scenario_plan_keep_with_feedback() -> None:
    driver = InteractDriver({"kind": "plan", "plan": "# Plan\n- do X\n"})
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("plan it"))
        await wait_until(lambda: app._interact_open, pilot, "plan card open")
        # Type feedback + Enter → keep planning, feedback delivered, still in plan mode.
        _type(app, "also add tests first")
        await pilot.press("enter")
        await wait_until(lambda: driver.result is not None, pilot, "verdict returned")
        assert driver.result["approved"] is False, driver.result
        assert "also add tests first" in driver.result["message"], driver.result
        assert driver.permission_mode == PLAN_MODE, "keep-planning stays in plan mode"
        await wait_until(lambda: not app._busy, pilot, "turn finishes")
    print("✓ plan: type feedback + Enter → {approved:False, message:<feedback>}, stays in plan")


async def scenario_question_pick_option() -> None:
    driver = InteractDriver({"kind": "question", "questions": [{
        "header": "Indentation", "question": "tabs or spaces?",
        "options": [{"label": "Tabs", "description": "tab chars"},
                    {"label": "Spaces", "description": "space chars"}],
    }]})
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("how should I indent?"))
        await wait_until(lambda: app._interact_open, pilot, "question card open")
        assert _card_present(app), "the question is rendered as a card"
        # ↓ to the 2nd option (Spaces), Enter (empty prompt) → pick it.
        await pilot.press("down")
        await pilot.press("enter")
        await wait_until(lambda: driver.result is not None, pilot, "answer returned")
        assert "Spaces" in driver.result["message"], driver.result
        assert "Tabs" not in driver.result["message"], driver.result
        await wait_until(lambda: not app._busy, pilot, "turn finishes")
    print("✓ question: ↓ + Enter selects the option → answer delivered as the tool result")


async def scenario_question_type_your_own() -> None:
    driver = InteractDriver({"kind": "question", "questions": [{
        "header": "Indentation", "question": "tabs or spaces?",
        "options": [{"label": "Tabs"}, {"label": "Spaces"}],
    }]})
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.post_message(PromptArea.Submitted("how should I indent?"))
        await wait_until(lambda: app._interact_open, pilot, "question card open")
        # Ignore the options — type a custom answer (the "add my own / chat" path).
        _type(app, "4-wide tabs, actually")
        await pilot.press("enter")
        await wait_until(lambda: driver.result is not None, pilot, "answer returned")
        assert "4-wide tabs, actually" in driver.result["message"], driver.result
        await wait_until(lambda: not app._busy, pilot, "turn finishes")
    print("✓ question: type your own answer + Enter → delivered verbatim (add-my-own / chat)")


async def main() -> int:
    await scenario_plan_approve()
    await scenario_plan_keep_with_feedback()
    await scenario_question_pick_option()
    await scenario_question_type_your_own()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
