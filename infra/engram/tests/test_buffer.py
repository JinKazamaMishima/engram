#!/usr/bin/env python3
"""Headless tests for the LiveBuffer (Brick 3 tier 1) and its driver wiring:
append/tail/seq, rekey (rename + merge + copy-on-fork), fail-open, the
log-raw/inject-derived invariant at query(), stale-resume revert, interrupt
partial capture — and the cross-repo contract: rows written by the harness
LiveBuffer must round-trip through recall.transcripts.iter_buffer_exchanges.

    .venv/bin/python infra/engram/tests/test_buffer.py
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

from buffer import LiveBuffer  # noqa: E402
from core import AgentSDKDriver, Event  # noqa: E402

# The Brick-3 cross-repo reader (iter_buffer_exchanges) lives in the recall ENGINE,
# which mirrors on a separate track — skip that one round-trip check when this repo's
# engine predates Brick-3 (the rest of the LiveBuffer suite is engine-independent).
from recall import transcripts as _T  # noqa: E402
_ENGINE_HAS_BRICK3 = hasattr(_T, "iter_buffer_exchanges")


def _mk(dir_, cid="conv-1"):
    holder = {"id": cid}
    return LiveBuffer(Path(dir_), lambda: holder["id"]), holder


def test_append_tail_seq() -> None:
    with tempfile.TemporaryDirectory() as d:
        buf, _ = _mk(d)
        buf.append("user", "first")
        buf.append("assistant", "reply one")
        buf.append("user", "second")
        rows = buf.tail(2)
        assert [r["text"] for r in rows] == ["reply one", "second"]
        assert [r["seq"] for r in buf.tail(10)] == [1, 2, 3]
        assert buf.last_seq() == 3
        # reseed continues, never restarts
        buf2, _ = _mk(d)
        buf2.reseed()
        buf2.append("user", "fourth")
        assert [r["seq"] for r in buf2.tail(10)] == [1, 2, 3, 4]
        # empty text never writes a row
        buf.append("assistant", "")
        assert buf.last_seq() == 4
    print("✓ append/tail/seq monotonic; reseed continues; empty text skipped")


def test_disabled_and_fail_open() -> None:
    off = LiveBuffer(None, lambda: "x")
    off.append("user", "never lands")
    assert off.tail(5) == [] and off.path() is None and not off.enabled
    # unwritable dir: every call swallows, nothing raises
    ro = Path("/proc/definitely-not-writable")
    buf = LiveBuffer(ro, lambda: "x")
    buf.append("user", "hi")
    assert buf.tail(3) == []
    buf.migrate("x", "y")
    print("✓ dir=None disables cleanly; unwritable dir is fail-open everywhere")


def test_tolerant_reader_and_tail_after() -> None:
    with tempfile.TemporaryDirectory() as d:
        buf, _ = _mk(d)
        buf.append("user", "early")
        p = buf.path()
        with p.open("a") as f:
            f.write('{"torn": ')                     # crash mid-append
            f.write("\nnot json\n")
        buf.append("assistant", "late")
        rows = buf.tail(10)
        assert [r["text"] for r in rows] == ["early", "late"]
        early_ts = rows[0]["ts"]
        after = buf.tail_after(early_ts)
        assert [r["text"] for r in after] == ["late"]     # strictly after
        assert len(buf.tail_after("")) == 2               # no mark -> everything
        assert len(buf.tail_after("garbage-ts")) == 2     # garbled -> everything
    print("✓ torn/garbage lines skipped; tail_after strict, fail-open on bad marks")


def test_migrate_rename_merge_copy() -> None:
    with tempfile.TemporaryDirectory() as d:
        buf, holder = _mk(d, "launch-abc")
        buf.append("user", "hello")
        # rename: launch -> sid
        buf.migrate("launch-abc", "sid-1")
        holder["id"] = "sid-1"
        assert not (Path(d) / "launch-abc.jsonl").exists()
        assert [r["text"] for r in buf.tail(5)] == ["hello"]
        # merge: rows land after the existing file's rows
        other, _ = _mk(d, "launch-zzz")
        other.append("user", "from the retry")
        other.migrate("launch-zzz", "sid-1")
        assert [r["text"] for r in buf.tail(5)] == ["hello", "from the retry"]
        assert not (Path(d) / "launch-zzz.jsonl").exists()
        # copy (fork): parent intact, child seeded
        buf.migrate("sid-1", "sid-2", copy=True)
        assert (Path(d) / "sid-1.jsonl").exists()
        holder["id"] = "sid-2"
        assert [r["text"] for r in buf.tail(5)] == ["hello", "from the retry"]
    print("✓ migrate: rename, merge-then-unlink, copy-on-fork (parent intact)")


def test_roundtrip_through_recall_reader() -> None:
    """THE cross-repo contract: harness rows must parse as recall Exchanges and
    render through the canonical bundle path."""
    if not _ENGINE_HAS_BRICK3:
        print("↷ SKIP round-trip — recall engine lacks Brick-3 (iter_buffer_exchanges)")
        return
    from recall import transcripts as T
    with tempfile.TemporaryDirectory() as d:
        buf, _ = _mk(d, "sid-rt")
        buf.append("user", "why does X happen?")
        buf.append("assistant", "Because of Y.")
        exs = list(T.iter_buffer_exchanges(buf.path()))
        assert [(e.role, e.text) for e in exs] == [
            ("user", "why does X happen?"), ("assistant", "Because of Y.")]
        assert all(e.session_id == "sid-rt" and e.ts.tzinfo for e in exs)
        text, stats = T.build_buffer_bundle(buf.path())
        assert stats.exchanges == 2 and "### USER" in text
    print("✓ LiveBuffer rows round-trip through recall.transcripts (contract holds)")


# ---- driver wiring ---------------------------------------------------------

def _driver(tmp) -> AgentSDKDriver:
    d = AgentSDKDriver(store=None, buffer_dir=Path(tmp))

    async def _noop_connect():
        return None
    d.connect = _noop_connect
    return d


async def scenario_capture_and_invariant() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = _driver(tmp)
        seen_by_sdk = {}

        async def fake_stream(text):
            seen_by_sdk["text"] = text
            yield Event("text", "the ")
            yield Event("tool", "Bash: ls")
            yield Event("text", "answer")
        d._stream = fake_stream

        out = [ev async for ev in d.query("raw question",
                                          prepend="<working-memory>wm</working-memory>\n")]
        assert any(ev.kind == "tool" for ev in out)
        assert seen_by_sdk["text"].startswith("<working-memory>")   # SDK got the block …
        rows = d._buffer.tail(10)
        assert [(r["role"], r["text"]) for r in rows] == [
            ("user", "raw question"),                               # … the buffer did NOT
            ("assistant", "the answer")]
        assert rows[0]["convo_id"].startswith("launch-")
    print("✓ query(): raw text buffered, prepend reaches only the SDK; one assistant row")


async def scenario_interrupt_partial() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = _driver(tmp)

        async def endless(text):
            yield Event("text", "partial ")
            yield Event("text", "reply")
            await asyncio.sleep(999)
            yield Event("text", "never")
        d._stream = endless

        gen = d.query("hi")
        assert (await gen.__anext__()).text == "partial "
        assert (await gen.__anext__()).text == "reply"
        await gen.aclose()                       # ESC — front-end abandons the turn
        rows = d._buffer.tail(5)
        assert [(r["role"], r["text"]) for r in rows] == [
            ("user", "hi"), ("assistant", "partial reply")]
    print("✓ interrupt (GeneratorExit): the partial reply the operator saw is captured")


async def scenario_stale_resume_single_user_row() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = _driver(tmp)
        d.session_id = "sid-dead"
        d._buf_convo_id = "sid-dead"
        calls = {"n": 0}

        async def flaky(text):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("stale resume")
                yield  # pragma: no cover — make it a generator
            yield Event("text", "recovered")
        d._stream = flaky

        async def _noop_disconnect():
            return None
        d.disconnect = _noop_disconnect

        out = [ev async for ev in d.query("are you there?")]
        assert any("couldn't resume" in ev.text for ev in out if ev.kind == "text")
        # reverted to the launch file: ONE user row + the recovery reply live on
        assert d._buf_convo_id == d._launch_id
        rows = d._buffer.tail(10)
        assert [(r["role"], r["text"]) for r in rows] == [
            ("user", "are you there?"), ("assistant", "recovered")]
    print("✓ stale-resume: one user row, buffer reverts to launch id, reply captured")


async def scenario_rekey_matrix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = _driver(tmp)
        d._buffer.append("user", "before the sid exists")
        launch = d._buf_convo_id
        # turn-1: sid learned -> rename
        d.session_id = "sid-real"
        d._sync_buf_convo()
        assert d._buf_convo_id == "sid-real"
        assert not (Path(tmp) / f"{launch}.jsonl").exists()
        assert (Path(tmp) / "sid-real.jsonl").exists()
        # fork: copy — parent intact, child continues
        d._fork_buf_copy = True
        d.session_id = "sid-branch"
        d._sync_buf_convo()
        assert (Path(tmp) / "sid-real.jsonl").exists()      # parent untouched
        assert d._buf_convo_id == "sid-branch"
        d._buffer.append("user", "branch-only turn")
        assert len(d._buffer.tail(10)) == 2
        # reset (/new): fresh launch id, old files stay
        d.reset()
        assert d._buf_convo_id.startswith("launch-")
        assert d._buffer.tail(10) == []
        # picker resume: switch to an existing conversation's buffer
        await_resume = d.resume_session("sid-real")
        await await_resume
        assert d._buf_convo_id == "sid-real"
        assert [r["text"] for r in d._buffer.tail(10)] == ["before the sid exists"]
        d._buffer.append("user", "continued after pick")
        assert [r["seq"] for r in d._buffer.tail(10)] == [1, 2]   # seq continued
    print("✓ rekey matrix: rename / copy-on-fork / reset-fresh / picker-continue")


def test_store_none_disables_buffer() -> None:
    d = AgentSDKDriver(store=None)          # no explicit buffer_dir
    assert not d._buffer.enabled            # perceive-mind isolation (A7)
    print("✓ store=None (perceiving mind / transient) never buffers by default")


async def scenario_working_memory_prepend_not_buffered() -> None:
    """A4+A5 end to end: the working-memory block built from PRIOR turns rides
    the prepend to the SDK, but only the raw turn text lands in the buffer — so
    the derived block can never feed back into its own source."""
    import working_set as ws
    with tempfile.TemporaryDirectory() as tmp:
        d = _driver(tmp)
        seen = {}

        async def echo(text):
            seen["sdk"] = text
            yield Event("text", "ok")
        d._stream = echo

        # turn 1 — nothing prior, so no working memory yet
        wm1 = ws.build_working_memory(d._buffer, d.cwd, notes=0)
        assert wm1 == ""
        async for _ in d.query("first question", prepend=wm1):
            pass
        # turn 2 — the builder now sees turn 1 in the buffer
        wm2 = ws.build_working_memory(d._buffer, d.cwd, notes=0)
        assert "first question" in wm2 and "<working-memory>" in wm2
        async for _ in d.query("second question", prepend=wm2 + "\n\n"):
            pass
        # the SDK saw the block; the buffer holds ONLY raw turns
        assert "<working-memory>" in seen["sdk"]
        assert seen["sdk"].endswith("second question")
        texts = [(r["role"], r["text"]) for r in d._buffer.tail(10)]
        assert texts == [("user", "first question"), ("assistant", "ok"),
                         ("user", "second question"), ("assistant", "ok")]
        assert not any("<working-memory>" in t for _, t in texts)
    print("✓ working-memory prepend reaches the SDK; buffer logs raw turns only")


async def main() -> int:
    test_append_tail_seq()
    test_disabled_and_fail_open()
    test_tolerant_reader_and_tail_after()
    test_migrate_rename_merge_copy()
    test_roundtrip_through_recall_reader()
    await scenario_capture_and_invariant()
    await scenario_interrupt_partial()
    await scenario_stale_resume_single_user_row()
    await scenario_rekey_matrix()
    test_store_none_disables_buffer()
    await scenario_working_memory_prepend_not_buffered()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
