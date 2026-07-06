#!/usr/bin/env python3
"""Headless tests for harness slash-commands: /context, /agent, /model, /ultracode.

Covers (1) the pure markdown renderer for the SDK's context-usage payload,
(2) /context renders when idle but is BLOCKED (driver never polled) mid-turn,
and (3) /agent <name> <task> rewrites to a Task-tool delegation prompt while a
task-less /agent is rejected. Same FakeDriver + Textual-pilot style as
test_queue.py.

    .venv/bin/python infra/engram/tests/test_commands.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from core import Event, ModelDriver, render_context_md  # noqa: E402
from app import MODELS, PromptArea, EngramApp  # noqa: E402
from textual.widgets import Static  # noqa: E402


USAGE = {
    "model": "opus[1m]",
    "totalTokens": 131072,
    "rawMaxTokens": 1000000,
    "maxTokens": 920000,
    "percentage": 13.1072,
    "isAutoCompactEnabled": True,
    "categories": [
        {"name": "System prompt", "tokens": 3200, "color": "#fff"},
        {"name": "Messages", "tokens": 110000, "color": "#abc"},
        {"name": "Empty", "tokens": 0, "color": "#000"},          # zero → filtered out
        {"name": "Free space", "tokens": 886800, "color": "#000"},  # remainder → filtered out
    ],
    "memoryFiles": [{"path": "CLAUDE.md"}],
    "mcpTools": [{"name": "x"}],
    "agents": [{"name": "Explore"}, {"name": "Plan"}],
}


class FakeDriver(ModelDriver):
    """query() blocks on a per-call gate; get_context_usage() returns a canned dict
    and counts calls so we can assert it's NOT polled mid-turn."""
    def __init__(self, usage=None):
        self.model = "opus[1m]"
        self.effort = "max"
        self.session_id = None
        self.actual_model = None
        self.resumed = False
        self.stderr_tail = ""
        self.calls: list[str] = []          # prepend + text (what the SDK sees)
        self.raw_calls: list[str] = []       # raw text only (what the buffer logs)
        self.subagent_calls: list[tuple[str, str]] = []
        self.gates: list[asyncio.Event] = []
        self.context_calls = 0
        self._usage = usage or {}

    async def query(self, text, *, prepend=""):
        # `text` is the operator's raw message; `prepend` is the model-only
        # block (working memory + markers). Record the full thing the SDK sees
        # so marker assertions still hold, and the raw text separately.
        self.calls.append(prepend + text)
        self.raw_calls.append(text)
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        yield Event("text", "ok")

    async def run_subagent(self, name, task):
        self.subagent_calls.append((name, task))
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        yield Event("text", f"{name} reports: ok")

    async def get_context_usage(self):
        self.context_calls += 1
        return dict(self._usage)

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


def widget_text(app, sel) -> str:
    w = app.query_one(sel, Static)
    for attr in ("_content", "renderable", "_renderable"):
        v = getattr(w, attr, None)
        if v is not None:
            return str(v)
    return str(w.render())


def test_render_context() -> None:
    out = render_context_md(USAGE)
    assert "opus[1m]" in out, out
    assert "131,072" in out and "1,000,000" in out, out
    assert "13%" in out, out
    assert "█" in out and "░" in out, out                 # the usage bar
    assert "System prompt" in out and "Messages" in out, out
    assert "Empty" not in out, "zero-token categories should be filtered"
    assert "Free space" not in out, "the free-space remainder should be filtered"
    assert "sub-agents (2): Explore, Plan" in out, out
    assert "auto-compact on" in out, out
    assert "usable window 920,000 of 1,000,000" in out, out
    # Real payloads return agent entries without a usable name → fall back to a count.
    nameless = render_context_md({**USAGE, "agents": [{}, {}, {}]})
    assert "sub-agents: 3" in nameless, nameless
    assert render_context_md({}) == "**Context** — no usage data available."
    print("✓ render_context_md formats the usage payload (bar, table, rollups, footer)")


def test_model_menu() -> None:
    """The /model dropdown offers the model roster (incl. Fable 5); free-form still works."""
    app = EngramApp(driver=FakeDriver())

    def vals(text):
        return [v for v, _ in app._menu_items(text)]

    assert vals("/model ") == [f"/model {n}" for n, _ in MODELS], vals("/model ")
    assert "/model fable" in vals("/model "), "Fable 5 must be offered"
    assert vals("/model f") == ["/model fable"], vals("/model f")
    assert vals("/model op") == ["/model opus[1m]", "/model opus"], vals("/model op")
    labels = {v: lbl for v, lbl in app._menu_items("/model f")}
    assert "Fable 5" in labels["/model fable"], labels
    assert "/model" in vals("/model"), vals("/model")           # still in the top-level menu
    print("✓ /model dropdown offers the roster incl. Fable 5 (free-form entry still works)")


async def scenario_context() -> None:
    driver = FakeDriver(usage=USAGE)
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Idle → /context polls the driver once and renders.
        app.post_message(PromptArea.Submitted("/context"))
        await wait_until(lambda: driver.context_calls == 1, pilot, "/context to render")
        # Start a turn and hold it open.
        app.post_message(PromptArea.Submitted("hello"))
        await wait_until(lambda: len(driver.calls) == 1, pilot, "turn to start")
        assert app._busy
        # /context mid-turn → BLOCKED: driver is not polled, status says busy.
        app.post_message(PromptArea.Submitted("/context"))
        await pilot.pause()
        assert driver.context_calls == 1, "must not poll the warm client mid-turn"
        assert "busy" in widget_text(app, "#status").lower(), widget_text(app, "#status")
        driver.gates[0].set()
        await wait_until(lambda: not app._busy, pilot, "idle")
    print("✓ /context renders when idle, is blocked (not polled) mid-turn")


async def scenario_agent() -> None:
    driver = FakeDriver()
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Task-less /agent → rejected, no sub-agent runs.
        app.post_message(PromptArea.Submitted("/agent"))
        await wait_until(lambda: "usage" in widget_text(app, "#status").lower(),
                         pilot, "task-less /agent → usage hint")   # status can lag a pause
        assert driver.subagent_calls == [], "task-less /agent must not start a sub-agent"
        app.post_message(PromptArea.Submitted("/agent Explore"))   # name but no task
        await pilot.pause()
        assert driver.subagent_calls == [], "name-only /agent must not start a sub-agent"
        # /agent <name> <task> → the named sub-agent runs as an isolated sub-query
        # (run_subagent), NOT the main query path.
        app.post_message(PromptArea.Submitted("/agent Explore find every caller"))
        await wait_until(lambda: len(driver.subagent_calls) == 1, pilot, "sub-agent to start")
        assert driver.subagent_calls[0] == ("Explore", "find every caller"), driver.subagent_calls
        assert driver.calls == [], "explicit /agent must not hit the main query path"
        driver.gates[0].set()
        await wait_until(lambda: not app._busy, pilot, "idle")
    print("✓ /agent runs the named sub-agent as a sub-query; task-less/name-only is rejected")


async def scenario_ultracode() -> None:
    driver = FakeDriver()
    app = EngramApp(driver=driver)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._ultracode is False and app._ultracode_marker() == ""
        # Toggle on → subtitle badge + a non-empty standing marker.
        app.post_message(PromptArea.Submitted("/ultracode"))
        await wait_until(lambda: app._ultracode is True, pilot, "ultracode on")
        assert "ultracode" in app._subtitle(), app._subtitle()
        assert app._ultracode_marker().startswith("<system-reminder>"), "marker present when on"
        # A typed turn now carries the standing opt-in to the model (prepended, model-only).
        app.post_message(PromptArea.Submitted("hello"))
        await wait_until(lambda: len(driver.calls) == 1, pilot, "turn to start")
        sent = driver.calls[0]
        assert sent.startswith("<system-reminder>") and "Ultracode is on" in sent, sent[:80]
        assert sent.rstrip().endswith("hello"), sent[-40:]
        driver.gates[0].set()
        await wait_until(lambda: not app._busy, pilot, "idle")
        # Explicit off → marker empty, badge gone, next turn is clean.
        app.post_message(PromptArea.Submitted("/ultracode off"))
        await wait_until(lambda: app._ultracode is False, pilot, "ultracode off")
        assert app._ultracode_marker() == "" and "ultracode" not in app._subtitle()
        app.post_message(PromptArea.Submitted("hi again"))
        await wait_until(lambda: len(driver.calls) == 2, pilot, "second turn")
        assert driver.calls[1] == "hi again", driver.calls[1]
    print("✓ /ultracode toggles the standing opt-in: subtitle badge + per-turn system-reminder")


async def main() -> int:
    test_render_context()
    test_model_menu()
    await scenario_context()
    await scenario_agent()
    await scenario_ultracode()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
