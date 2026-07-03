#!/usr/bin/env python3
"""Tests for crash-resilient session resume + the per-folder launch lock — the
hardening for remote (VPN) launches where the process can die mid-turn.

Covers:
  * a session id is flushed the moment the session EXISTS (on the init message),
    not only at end-of-turn — so a turn interrupted before its ResultMessage still
    leaves a resumable pointer (the first-turn-in-a-fresh-folder data-loss hole);
  * a stale resume is SURFACED to the front-end (and the dead pointer cleared),
    not silently swapped for a fresh thread;
  * LaunchLock refuses a second LIVE engram in one folder but reclaims a stale lock
    left behind by a crash.

    .venv/bin/python infra/engram/tests/test_resume.py
"""
import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import core  # noqa: E402  (for monkeypatching _pid_is_live_engram)
from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
)
from core import AgentSDKDriver, LaunchLock  # noqa: E402


def AM(text):
    return AssistantMessage(content=[TextBlock(text=text)], model="m",
                            parent_tool_use_id=None)


def RESULT(sid="s"):
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                         is_error=False, num_turns=1, session_id=sid)


class RecordingStore:
    """A SessionStore stand-in that records every save in order."""
    def __init__(self, initial=None):
        self.saved = []
        self._initial = initial

    def load(self, cwd):
        return self._initial

    def save(self, cwd, sid):
        self.saved.append(sid)


class FakeClient:
    def __init__(self, messages, raise_on_recv=False):
        self._messages = messages
        self._raise = raise_on_recv

    async def query(self, text):
        pass

    async def receive_messages(self):
        if self._raise:
            raise RuntimeError("stale resume")
        for m in self._messages:
            yield m


async def test_session_flushed_on_init_before_result():
    """The id must reach the store as soon as the init message arrives — even if the
    turn is cut off before its ResultMessage (the mid-turn-crash case)."""
    store = RecordingStore()
    d = AgentSDKDriver(store=store)
    # init only, then the stream ends abruptly — NO ResultMessage (crash mid-turn).
    d._client = FakeClient([SystemMessage(subtype="init",
                                          data={"session_id": "sess-1"})])
    _ = [ev async for ev in d._stream("hi")]
    assert store.saved and store.saved[0] == "sess-1", \
        f"session id must be flushed at init, got {store.saved}"
    print("✓ session id is flushed on init, before end-of-turn (crash-safe)")


async def test_stale_resume_is_surfaced_and_cleared():
    """A resume that blows up mid-turn: drop the dead pointer, tell the user, retry fresh."""
    store = RecordingStore(initial="old-dead")
    d = AgentSDKDriver(store=store)
    assert d.resumed is True and d.session_id == "old-dead"

    good = FakeClient([SystemMessage(subtype="init", data={"session_id": "new"}),
                       AM("fresh answer"), RESULT("new")])
    step = {"n": 0}

    async def fake_connect():
        # first connect → a client that raises on receive; second → the good one.
        d._client = FakeClient([], raise_on_recv=True) if step["n"] == 0 else good
        step["n"] += 1

    async def fake_disconnect():
        d._client = None

    d.connect = fake_connect
    d.disconnect = fake_disconnect

    evs = [ev async for ev in d.query("hi")]
    body = "".join(e.text for e in evs if e.kind == "text")
    assert "couldn't resume" in body, body
    assert "fresh answer" in body, body
    assert d.resumed is False
    assert None in store.saved, f"dead pointer must be cleared, got {store.saved}"
    assert store.saved[-1] == "new", f"new session must be persisted, got {store.saved}"
    print("✓ stale resume: dead pointer cleared, user told, retried on a fresh thread")


def test_launch_lock_refuses_double_but_reclaims_stale():
    root = Path(tempfile.mkdtemp())
    cwd = Path(tempfile.mkdtemp())          # a neutral project dir for the test
    lk = LaunchLock(cwd, root=root)

    # free folder → acquired; re-acquire by the SAME process is fine (we own it).
    assert lk.acquire() is None
    assert lk.acquire() is None

    orig = core._pid_is_live_engram
    core._pid_is_live_engram = lambda pid: True   # force "alive" — the test proc isn't engram
    try:
        lk.path.write_text("4242")                # pretend pid 4242 holds it, and is alive
        lk2 = LaunchLock(cwd, root=root)
        assert lk2.acquire() == 4242, "must refuse when a live engram holds the folder"

        core._pid_is_live_engram = lambda pid: False   # a crash left a DEAD-pid lock
        lk3 = LaunchLock(cwd, root=root)
        assert lk3.acquire() is None, "a stale (dead-pid) lock must be reclaimed"
        assert lk3.path.read_text().strip() == str(os.getpid())
    finally:
        core._pid_is_live_engram = orig

    # a genuinely dead pid is detected as not-live (spawn → reap → its pid is gone).
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert core._pid_is_live_engram(p.pid) is False
    print("✓ launch lock: refuses a live double-launch, reclaims a crashed lock's stale file")


async def main() -> int:
    await test_session_flushed_on_init_before_result()
    await test_stale_resume_is_surfaced_and_cleared()
    test_launch_lock_refuses_double_but_reclaims_stale()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
