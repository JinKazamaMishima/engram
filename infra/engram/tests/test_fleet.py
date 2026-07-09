#!/usr/bin/env python3
"""Unit tests for the Engram fleet — parallel sibling sessions steered from one
TUI. Fake driver, tmp lock roots: no SDK, no subprocess, no real locks.

    .venv/bin/python infra/engram/tests/test_fleet.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import core  # noqa: E402
from core import Event  # noqa: E402
from fleet import Fleet, render_fleet_line, render_member_event  # noqa: E402


class FakeFleetDriver:
    """Minimal ModelDriver: records queries, streams a scripted turn."""

    def __init__(self, *, events=None, delay=0.0):
        self.connected = False
        self.queries: list[str] = []
        self.resumed = False
        self.has_background_tasks = False
        self._events = events or [Event("text", "did the thing. "),
                                  Event("tool", "Bash"),
                                  Event("text", "done — result: 42")]
        self._delay = delay

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def query(self, text, *, prepend=""):
        self.queries.append(text)
        for e in self._events:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield e

    async def drain_background(self):
        return
        yield  # pragma: no cover — marks async generator


async def _until(cond, *, tries=200):
    for _ in range(tries):
        if cond():
            return True
        await asyncio.sleep(0.01)
    return False


def _fleet(tmp, **kw):
    drivers: list[FakeFleetDriver] = []

    def factory(*, cwd, name):
        d = FakeFleetDriver(**kw)
        drivers.append(d)
        return d

    return Fleet(driver_factory=factory, lock_root=Path(tmp) / "locks"), drivers


async def test_spawn_runs_initial_task_to_idle():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "dealerflow"
        repo.mkdir()
        fleet, drivers = _fleet(tmp)
        m, note = fleet.spawn(repo, task="audit the auth module")
        assert m is not None and "spawned" in note
        assert await _until(lambda: m.status == "idle" and m.turns == 1)
        assert drivers[0].queries == ["audit the auth module"]
        tail = m.tail()
        assert "did the thing" in tail and "· Bash" in tail and "❯ audit" in tail
        rows = fleet.rows()
        assert rows[0]["name"] == "dealerflow" and rows[0]["status"] == "idle"
        await fleet.shutdown()
    print("✓ spawn → initial task runs → transcript + idle")


async def test_spawn_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        fleet, _ = _fleet(tmp)
        m, note = fleet.spawn(Path(tmp) / "nope")
        assert m is None and "no such folder" in note
        repo = Path(tmp) / "r"
        repo.mkdir()
        m1, _ = fleet.spawn(repo)
        m2, note2 = fleet.spawn(repo)
        assert m1 is not None and m2 is None and "already works in" in note2
        await fleet.shutdown()
    print("✓ spawn refuses a missing folder and a duplicate cwd")


async def test_spawn_refuses_live_engram_lock():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "r"
        repo.mkdir()
        fleet, _ = _fleet(tmp)
        # Another LIVE engram already owns the folder: pre-write its lock and make
        # the liveness probe say yes (restore after — module-level patch).
        lock_dir = Path(tmp) / "locks"
        lock_dir.mkdir(parents=True)
        import re as _re
        safe = _re.sub(r"[^A-Za-z0-9._-]", "-", str(repo.resolve()))
        (lock_dir / safe).write_text("99999")
        orig = core._pid_is_live_engram
        core._pid_is_live_engram = lambda pid: True
        try:
            m, note = fleet.spawn(repo)
        finally:
            core._pid_is_live_engram = orig
        assert m is None and "already driven" in note, note
    print("✓ spawn refuses a folder a live engram already drives")


async def test_send_queues_while_working_then_drains():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "r"
        repo.mkdir()
        fleet, drivers = _fleet(tmp, delay=0.03)
        m, _ = fleet.spawn(repo, task="first job")
        assert await _until(lambda: m.status == "working")
        note = fleet.send("r", "second job")
        assert "queued" in note, note
        assert await _until(lambda: m.turns == 2 and m.status == "idle")
        assert drivers[0].queries == ["first job", "second job"]
        assert fleet.send("ghost", "x").startswith("no fleet member")
        await fleet.shutdown()
    print("✓ steering queues behind a working turn and drains in order")


async def test_kill_releases_lock_and_name():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "r"
        repo.mkdir()
        fleet, drivers = _fleet(tmp)
        m, _ = fleet.spawn(repo)
        assert await _until(lambda: m.status == "idle")
        lock_path = m.lock.path
        assert lock_path.exists()
        note = await fleet.kill("r")
        assert "stopped" in note
        assert not lock_path.exists(), "kill must release the folder lock"
        assert "r" not in fleet.members
        assert not drivers[0].connected, "kill disconnects the driver"
        # The folder is spawnable again immediately.
        m2, note2 = fleet.spawn(repo)
        assert m2 is not None, note2
        await fleet.shutdown()
    print("✓ kill cancels, disconnects, releases the lock, frees the name")


async def test_dead_member_reports_error():
    class BoomDriver(FakeFleetDriver):
        async def query(self, text, *, prepend=""):
            raise RuntimeError("subprocess died")
            yield  # pragma: no cover

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "r"
        repo.mkdir()
        fleet = Fleet(driver_factory=lambda *, cwd, name: BoomDriver(),
                      lock_root=Path(tmp) / "locks")
        m, _ = fleet.spawn(repo, task="explode")
        assert await _until(lambda: m.status == "dead")
        assert "RuntimeError" in m.error
        assert fleet.send("r", "more").startswith("r is dead")
        await fleet.shutdown()
    print("✓ a crashed member reads dead with its error, and refuses new work")


async def test_name_dedupe_two_repos_same_basename():
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "one" / "app"
        b = Path(tmp) / "two" / "app"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        fleet, _ = _fleet(tmp)
        m1, _ = fleet.spawn(a)
        m2, _ = fleet.spawn(b)
        assert m1.name == "app" and m2.name == "app-2"
        await fleet.shutdown()
    print("✓ same-basename repos get deduped member names")


def test_render_fleet_line():
    assert render_fleet_line([]) == ""
    line = render_fleet_line([
        {"name": "dealerflow", "dir": "dealerflow", "status": "working",
         "pending": 1, "turns": 2, "last": "npm test", "error": ""},
        {"name": "polytrade", "dir": "polytrade", "status": "idle",
         "pending": 0, "turns": 5, "last": "", "error": ""},
        {"name": "x", "dir": "x", "status": "dead", "pending": 0, "turns": 0,
         "last": "", "error": "RuntimeError: boom"},
    ])
    assert "⚑ dealerflow ⏳ npm test (+1 queued)" in line, line
    assert "⚑ polytrade ✦" in line
    assert "⚑ x ✗ RuntimeError: boom" in line
    print("✓ fleet strip: working members show last action + queue depth")


def test_render_member_event_shapes():
    assert render_member_event(Event("text", "chunk")) == "chunk"
    assert render_member_event(Event("tool", "Bash")) == "\n· Bash\n"
    assert "◆ recall" in render_member_event(Event("recall", "soul:rule-x"))
    assert render_member_event(Event("status", "thinking")) == ""
    assert render_member_event(Event("task", "", data={})) == ""
    print("✓ member-event renderer: text raw, tools as lines, panel feeds skip")


async def main() -> int:
    await test_spawn_runs_initial_task_to_idle()
    await test_spawn_refusals()
    await test_spawn_refuses_live_engram_lock()
    await test_send_queues_while_working_then_drains()
    await test_kill_releases_lock_and_name()
    await test_dead_member_reports_error()
    await test_name_dedupe_two_repos_same_basename()
    test_render_fleet_line()
    test_render_member_event_shapes()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
