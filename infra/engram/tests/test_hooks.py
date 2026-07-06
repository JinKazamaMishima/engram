#!/usr/bin/env python3
"""Unit tests for the memory-lifecycle seams: the PreCompact provisional-curation
hook and the detached shutdown-curation spawner (Brick 3's harness side).

    .venv/bin/python infra/engram/tests/test_hooks.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import core  # noqa: E402
from core import AgentSDKDriver, session_curation_cmd, spawn_session_curate  # noqa: E402


class FakeStore:
    """A non-default store → the driver is a real thread-of-record (curates on
    PreCompact/shutdown) but does NOT auto-open a buffer (that needs the default
    store), so it exercises the transcript-fallback path."""
    def load(self, cwd):
        return None
    def save(self, cwd, sid):
        pass


def test_curation_cmd_shape():
    cmd = session_curation_cmd("sid-1", Path("/repo"))
    assert cmd[1:] == ["curate", "--session", "sid-1", "--project-dir", "/repo",
                       "--commit"], cmd
    prov = session_curation_cmd("sid-1", Path("/repo"), provisional=True)
    assert "--provisional" in prov and prov[-1] == "--commit", prov
    os.environ["RECALL_BIN"] = "/opt/recall"
    try:
        assert session_curation_cmd("s", Path("/r"))[0] == "/opt/recall"
    finally:
        del os.environ["RECALL_BIN"]
    print("✓ curation argv matches the bridge's proven shape (+ --provisional variant)")


def test_hooks_registration():
    d = AgentSDKDriver(store=None)
    hooks = d._hooks()
    assert hooks and "PreCompact" in hooks and len(hooks["PreCompact"]) == 1
    matcher = hooks["PreCompact"][0]
    assert matcher.hooks == [d._on_precompact] and matcher.timeout == 30
    assert d._options().hooks is not None, "hooks must be wired into the options"
    os.environ["ENGRAM_CURATE_ON_COMPACT"] = "0"
    try:
        assert d._hooks() is None, "kill switch must unregister the hook"
    finally:
        del os.environ["ENGRAM_CURATE_ON_COMPACT"]
    print("✓ PreCompact matcher registered into options; ENGRAM_CURATE_ON_COMPACT=0 disables")


async def test_precompact_session_fallback():
    """No buffer (custom store) → PreCompact fires the transcript --session pass
    for the right session, provisionally, and never vetoes."""
    d = AgentSDKDriver(store=FakeStore())
    calls = []
    orig = core.spawn_session_curate
    core.spawn_session_curate = lambda sid, cwd, provisional=False: calls.append(
        (sid, str(cwd), provisional))
    try:
        out = await d._on_precompact({"session_id": "s-compact", "trigger": "auto"},
                                     None, {})
        assert out == {}, "must never veto/steer compaction"
        d.session_id = "s-fallback"
        await d._on_precompact({}, None, {})       # no sid in payload → driver's
    finally:
        core.spawn_session_curate = orig
    assert calls == [("s-compact", str(d.cwd), True),
                     ("s-fallback", str(d.cwd), True)], calls
    print("✓ PreCompact (no buffer) → provisional --session pass; returns {}")


async def test_precompact_buffer_path_ignores_size_gate():
    """With a LiveBuffer, PreCompact curates the cooled edge REGARDLESS of the
    size gate (last chance before the summary) — but still excludes the hot
    window (it survives via working-memory re-injection)."""
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        os.environ["RECALL_DATA_ROOT"] = str(dr)
        d = AgentSDKDriver(store=FakeStore(), buffer_dir=Path(tmp))
        d.cwd = Path(tmp)
        spawned = []
        orig = core.spawn_buffer_curate
        core.spawn_buffer_curate = (
            lambda path, cwd, *, until=None, provisional=True:
            spawned.append(until) or FakeProc())
        hot, core.EVICT_HOT_TURNS = core.EVICT_HOT_TURNS, 2
        try:
            for i in range(5):                     # tiny cooled tail (below any gate)
                d._buffer.append("user", f"m{i}")
            out = await d._on_precompact({}, None, {})
            assert out == {}
            assert len(spawned) == 1 and spawned[0] is not None  # cooled edge, not full
        finally:
            core.spawn_buffer_curate = orig
            core.EVICT_HOT_TURNS = hot
    print("✓ PreCompact (buffer) curates the cooled edge, size gate ignored, hot kept")


async def test_precompact_store_none_is_isolated():
    """A store-less driver (the perceiving mind) never curates — transient
    perception turns are not thread-of-record."""
    d = AgentSDKDriver(store=None)
    calls = []
    o1, o2 = core.spawn_session_curate, core.spawn_buffer_curate
    core.spawn_session_curate = lambda *a, **k: calls.append("session")
    core.spawn_buffer_curate = lambda *a, **k: calls.append("buffer")
    try:
        d.session_id = "s-perceive"
        out = await d._on_precompact({"session_id": "s-perceive"}, None, {})
        assert out == {} and calls == [], calls
    finally:
        core.spawn_session_curate, core.spawn_buffer_curate = o1, o2
    print("✓ PreCompact on a store-less (perceiving) driver curates nothing")


def test_spawn_detached_and_guarded():
    import subprocess as sp
    calls = []
    orig = sp.Popen
    def fake_popen(argv, **kw):
        calls.append((argv, kw))
        class P:  # noqa: N801
            pid = 1
        return P()
    sp.Popen = fake_popen
    try:
        spawn_session_curate(None, Path("/r"))          # no sid → no spawn
        spawn_session_curate("sid-2", Path("/r"), provisional=True)
    finally:
        sp.Popen = orig
    assert len(calls) == 1, calls
    argv, kw = calls[0]
    assert "--provisional" in argv and kw.get("start_new_session") is True
    print("✓ shutdown spawner: detached (start_new_session), skips empty sid")


class FakeProc:
    pid = 99
    def wait(self):
        return 0


async def main() -> int:
    test_curation_cmd_shape()
    test_hooks_registration()
    await test_precompact_session_fallback()
    await test_precompact_buffer_path_ignores_size_gate()
    await test_precompact_store_none_is_isolated()
    test_spawn_detached_and_guarded()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
