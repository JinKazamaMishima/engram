#!/usr/bin/env python3
"""Unit tests for _can_use_tool after ASK MODE WAS REMOVED (a deliberate design call): an ordinary
tool call reaching the permission callback AUTO-ALLOWS — no card, no prompt; Engram acts
and the persona is the guardrail — while the two interactive CLI tools (ExitPlanMode /
AskUserQuestion) still route to their cards. Plus a guard that no "ask" permission mode
survives anywhere (cycle / aliases / constant / ENGRAM_DEFAULT_MODE).

    .venv/bin/python infra/engram/tests/test_permissions.py
"""
import asyncio
import importlib
import os
import sys
from dataclasses import dataclass, field

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import core  # noqa: E402
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny  # noqa: E402
from core import (  # noqa: E402
    _MODE_ALIASES,
    MODE_CYCLE,
    PLAN_MODE,
    REGULAR_MODE,
    AgentSDKDriver,  # noqa: E402
)


@dataclass
class FakeCtx:
    """Shape-compatible stand-in for the SDK's ToolPermissionContext."""
    title: str | None = None
    display_name: str | None = None
    decision_reason: str | None = None
    suggestions: list = field(default_factory=list)


def scripted(driver, reply):
    """Wire an on_interaction handler that records the request and returns `reply`."""
    seen = []
    async def handler(req):
        seen.append(req)
        return reply
    driver.on_interaction = handler
    return seen


async def test_ordinary_tool_auto_allows_with_handler():
    """An ordinary tool reaching can_use_tool auto-allows — no card, and the interaction
    handler is never consulted (there is no permission prompt anymore)."""
    d = AgentSDKDriver(store=None)
    seen = scripted(d, {"allow": False, "message": "should never be asked"})
    res = await d._can_use_tool("Bash", {"command": "rm -rf /tmp/x"}, FakeCtx())
    assert isinstance(res, PermissionResultAllow) and res.updated_permissions is None
    assert seen == [], "ordinary tools must NOT be routed to an interaction handler"
    print("✓ ordinary tool → auto-allow, no card, handler untouched")


async def test_ordinary_tool_auto_allows_headless():
    """No UI wired (perceiving loop / --once / tests): still auto-allow — never deny, never hang."""
    d = AgentSDKDriver(store=None)          # on_interaction never set
    res = await d._can_use_tool("WebSearch", {"query": "x"}, FakeCtx())
    assert isinstance(res, PermissionResultAllow), res
    print("✓ headless (no UI) auto-allows — never denies, never hangs")


async def test_plan_still_routes():
    d = AgentSDKDriver(store=None)
    seen = scripted(d, {"approved": False, "message": "keep planning"})
    res = await d._can_use_tool("ExitPlanMode", {"plan": "do X"}, FakeCtx())
    assert isinstance(res, PermissionResultDeny) and res.message == "keep planning"
    assert seen[0]["kind"] == "plan"
    print("✓ ExitPlanMode still routes to the plan card")


async def test_question_still_routes():
    d = AgentSDKDriver(store=None)
    seen = scripted(d, {"message": "chose option A"})
    res = await d._can_use_tool(
        "AskUserQuestion", {"questions": [{"question": "which?"}]}, FakeCtx())
    assert isinstance(res, PermissionResultDeny) and res.message == "chose option A"
    assert seen[0]["kind"] == "question"
    print("✓ AskUserQuestion still routes to the option card")


def test_ask_mode_is_gone():
    """No 'ask' permission mode anywhere: two-stop cycle, no alias, no constant, and
    ENGRAM_DEFAULT_MODE=ask falls back to regular (bypass)."""
    assert MODE_CYCLE == (REGULAR_MODE, PLAN_MODE), MODE_CYCLE
    assert "ask" not in _MODE_ALIASES
    assert not hasattr(core, "ASK_MODE"), "the ASK_MODE constant must be gone"
    assert core.DEFAULT_MODE == REGULAR_MODE, "deliberate: bypass stays the daily default"
    os.environ["ENGRAM_DEFAULT_MODE"] = "ask"
    try:
        importlib.reload(core)
        assert core.DEFAULT_MODE == core.REGULAR_MODE, "removed 'ask' must fall back to regular"
        assert core.MODE_CYCLE == (core.REGULAR_MODE, core.PLAN_MODE)
    finally:
        del os.environ["ENGRAM_DEFAULT_MODE"]
        importlib.reload(core)
    print("✓ ask mode removed: 2-stop cycle, no alias/constant, ask→regular fallback")


async def main() -> int:
    await test_ordinary_tool_auto_allows_with_handler()
    await test_ordinary_tool_auto_allows_headless()
    await test_plan_still_routes()
    await test_question_still_routes()
    test_ask_mode_is_gone()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
