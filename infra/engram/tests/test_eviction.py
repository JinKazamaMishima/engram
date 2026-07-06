#!/usr/bin/env python3
"""Headless tests for eviction-is-curation (Brick 3 A6): the cooled-tail size
gate, the detached `curate --buffer` spawn + PID reaper, the _evicting guard,
the core-owned watermark read-back, and the shutdown full-flush.

    .venv/bin/python infra/engram/tests/test_eviction.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)
REPO = os.path.abspath(os.path.join(ENGRAM, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import core  # noqa: E402
from core import AgentSDKDriver  # noqa: E402


class FakeStore:
    def __init__(self):
        self._sid = None
    def load(self, cwd):
        return self._sid
    def save(self, cwd, sid):
        self._sid = sid


class FakeProc:
    def __init__(self, rc=0):
        self.pid = 4321
        self._rc = rc
    def wait(self):
        return self._rc


def _driver(tmp, data_root):
    """A buffered driver whose watermark reads land in an isolated data root."""
    os.environ["RECALL_DATA_ROOT"] = str(data_root)
    d = AgentSDKDriver(store=FakeStore(), buffer_dir=Path(tmp))
    d.cwd = Path(tmp)                       # project slug = this tmp folder
    return d


def _fill(d, n, text="padding text for a turn"):
    for i in range(n):
        d._buffer.append("user" if i % 2 == 0 else "assistant", f"{text} {i}")


def _patch_spawn():
    """Capture spawn_buffer_curate calls; return (calls, restore)."""
    calls = []
    orig = core.spawn_buffer_curate
    def fake(path, cwd, *, until=None, provisional=True):
        calls.append({"path": str(path), "until": until, "provisional": provisional})
        return FakeProc()
    core.spawn_buffer_curate = fake
    return calls, (lambda: setattr(core, "spawn_buffer_curate", orig))


def test_cooled_edge_excludes_hot_window():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        hot, core.EVICT_HOT_TURNS = core.EVICT_HOT_TURNS, 3
        try:
            _fill(d, 5)                     # 5 rows, hot window = last 3
            edge = d._cooled_edge()
            assert edge is not None
            rows = d._buffer.tail(10)
            assert edge[0] == rows[1]["ts"]  # cooled = rows[:-3] → last cooled is row index 1
            # not enough rows to cool → None
            d2 = _driver(tmp + "/x" if False else tmp, dr)
        finally:
            core.EVICT_HOT_TURNS = hot
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        hot, core.EVICT_HOT_TURNS = core.EVICT_HOT_TURNS, 12
        try:
            _fill(d, 5)                     # fewer than the hot window → nothing cooled
            assert d._cooled_edge() is None
        finally:
            core.EVICT_HOT_TURNS = hot
    print("✓ cooled edge = tail after watermark minus the hot window; None below it")


def test_size_gate_fires_only_when_cooled_crosses_threshold():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        calls, restore = _patch_spawn()
        hot, core.EVICT_HOT_TURNS = core.EVICT_HOT_TURNS, 2
        chars, core.EVICT_CHARS = core.EVICT_CHARS, 200
        try:
            _fill(d, 5, text="short")       # cooled chars well under 200
            d._maybe_evict()
            assert calls == [], "must not fire below the char gate"
            _fill(d, 6, text="x" * 80)      # now the cooled tail is large
            d._maybe_evict()
            assert len(calls) == 1, calls
            assert calls[0]["until"] is not None and calls[0]["provisional"] is True
        finally:
            core.EVICT_HOT_TURNS, core.EVICT_CHARS = hot, chars
            restore()
    print("✓ size gate: no spawn below ENGRAM_EVICT_CHARS, one detached spawn above")


def test_evicting_guard_serializes():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        calls, restore = _patch_spawn()
        hot, core.EVICT_HOT_TURNS = core.EVICT_HOT_TURNS, 1
        chars, core.EVICT_CHARS = core.EVICT_CHARS, 10
        try:
            _fill(d, 6, text="y" * 40)
            d._evicting = True              # a curate is already in flight
            d._maybe_evict()
            assert calls == [], "guard must block a second concurrent spawn"
            d._evicting = False
            d._maybe_evict()
            assert len(calls) == 1
        finally:
            core.EVICT_HOT_TURNS, core.EVICT_CHARS = hot, chars
            restore()
    print("✓ _evicting guard serializes eviction: one detached curate at a time")


def test_disabled_switches():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        calls, restore = _patch_spawn()
        hot, core.EVICT_HOT_TURNS = core.EVICT_HOT_TURNS, 1
        chars, core.EVICT_CHARS = core.EVICT_CHARS, 10
        on, core.EVICT_ON = core.EVICT_ON, False
        try:
            _fill(d, 6, text="z" * 40)
            d._maybe_evict()
            assert calls == [], "ENGRAM_EVICT=0 must suppress size-gate eviction"
        finally:
            core.EVICT_HOT_TURNS, core.EVICT_CHARS, core.EVICT_ON = hot, chars, on
            restore()
    print("✓ ENGRAM_EVICT off suppresses size-gate eviction (capture/inject untouched)")


def test_watermark_read_back_from_curated_json():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        assert d._evict_watermark() == ""            # nothing curated yet
        # core writes the watermark keyed on the convo id (= buffer stem)
        from recall import config
        sf = config.curation_dir() / config.project_slug(d.cwd) / "curated.json"
        sf.parent.mkdir(parents=True, exist_ok=True)
        sf.write_text(json.dumps({"watermarks": {d._buf_convo_id: "2026-07-03T20:00:00+00:00"}}))
        assert d._evict_watermark() == "2026-07-03T20:00:00+00:00"
        # rows before the watermark are already curated → excluded from the tail
        _fill(d, 3)
        after = d._buffer.tail_after(d._evict_watermark())
        assert len(after) == 3                        # all newer than the mark
    print("✓ watermark read-back from core's curated.json scopes the un-evicted tail")


async def test_reaper_clears_guard_on_exit():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        calls, restore = _patch_spawn()
        hot, core.EVICT_HOT_TURNS = core.EVICT_HOT_TURNS, 1
        chars, core.EVICT_CHARS = core.EVICT_CHARS, 10
        try:
            _fill(d, 6, text="w" * 40)
            d._maybe_evict()
            assert d._evicting is True and len(calls) == 1
            # let the reaper task run: it waits on FakeProc.wait() → clears guard
            for _ in range(50):
                await asyncio.sleep(0)
                if not d._evicting:
                    break
            assert d._evicting is False, "reaper must release the guard on PID exit"
        finally:
            core.EVICT_HOT_TURNS, core.EVICT_CHARS = hot, chars
            restore()
    print("✓ reaper waits on the PID and releases the guard on exit (success or crash)")


def test_shutdown_full_flush_and_fallback():
    # buffer present → full flush (until=None)
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as dr:
        d = _driver(tmp, dr)
        calls, restore = _patch_spawn()
        try:
            _fill(d, 3)
            d.evict_on_shutdown()
            assert len(calls) == 1 and calls[0]["until"] is None   # whole tail
        finally:
            restore()
    # no buffer but a store → legacy session --session pass
    sess = []
    orig = core.spawn_session_curate
    core.spawn_session_curate = lambda sid, cwd, provisional=False: sess.append((sid, provisional))
    try:
        d2 = AgentSDKDriver(store=FakeStore())     # buffer disabled (custom store)
        d2.session_id = "s-live"
        d2.evict_on_shutdown()
        assert sess == [("s-live", True)]
        # store=None (perceiving mind) → nothing curated
        sess.clear()
        d3 = AgentSDKDriver(store=None)
        d3.session_id = "s-perceive"
        d3.evict_on_shutdown()
        assert sess == []
    finally:
        core.spawn_session_curate = orig
    print("✓ shutdown: buffer full-flush / session fallback / perceive isolated")


async def main() -> int:
    test_cooled_edge_excludes_hot_window()
    test_size_gate_fires_only_when_cooled_crosses_threshold()
    test_evicting_guard_serializes()
    test_disabled_switches()
    test_watermark_read_back_from_curated_json()
    await test_reaper_clears_guard_on_exit()
    test_shutdown_full_flush_and_fallback()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
