#!/usr/bin/env python3
"""Unit tests for the session picker / fork / switch plumbing.

    .venv/bin/python infra/engram/tests/test_sessions.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(ENGRAM, "..", ".."))
sys.path.insert(0, ENGRAM)
sys.path.insert(0, os.path.join(REPO, "src"))

from claude_agent_sdk import ResultMessage, SystemMessage  # noqa: E402
from core import AgentSDKDriver, SessionStore  # noqa: E402


class FakeClient:
    def __init__(self, messages):
        self._messages = messages
        self.disconnected = False

    async def query(self, text, *, prepend=""):
        pass

    async def receive_messages(self):
        for m in self._messages:
            yield m

    async def disconnect(self):
        self.disconnected = True


def test_list_sessions_ordering_and_fallback():
    import recall.transcripts as rt
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        old = tdir / "sess-old.jsonl"
        new = tdir / "sess-new.jsonl"
        old.write_text("not json at all\n")
        new.write_text("also junk\n")
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        orig = rt.project_transcript_dir
        rt.project_transcript_dir = lambda cwd, base=None: tdir
        try:
            d = AgentSDKDriver(store=None)
            d.session_id = "sess-new"
            out = d.list_sessions()
        finally:
            rt.project_transcript_dir = orig
    assert [s["sid"] for s in out] == ["sess-new", "sess-old"], out
    assert out[0]["current"] is True and out[1]["current"] is False
    assert out[0]["preview"] == "sess-new"[:8], "junk transcript → id fallback"
    print("✓ list_sessions: newest first, current flagged, junk transcripts degrade to id")


async def test_fork_takes_new_session_id():
    with tempfile.TemporaryDirectory() as td:
        store = SessionStore(root=Path(td))
        d = AgentSDKDriver(store=store)
        d.session_id = "old-session"
        d._client = FakeClient([])
        await d.fork()
        assert d._fork_next and d._client is None, "fork must recycle the client"
        assert d._options().fork_session is True, "next connect must fork"
        d._client = FakeClient([
            SystemMessage(subtype="init",
                          data={"session_id": "forked-123", "model": "m"}),
            ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                          is_error=False, num_turns=1, session_id="forked-123"),
        ])
        async for _ in d._stream("hi"):
            pass
        assert d.session_id == "forked-123", \
            "the init's NEW id must be taken unconditionally when forking"
        assert d._fork_next is False
        assert store.load(d.cwd) == "forked-123", "the branch becomes the resumable session"
        assert d._options().fork_session is None or not d._options().fork_session, \
            "fork applies to ONE connect only"
    print("✓ /fork: one-shot fork_session; the branched id replaces the old one")


async def test_resume_session_switches_and_persists():
    with tempfile.TemporaryDirectory() as td:
        store = SessionStore(root=Path(td))
        d = AgentSDKDriver(store=store)
        fc = FakeClient([])
        d._client = fc
        await d.resume_session("other-42")
        assert fc.disconnected and d._client is None
        assert d.session_id == "other-42" and d.resumed
        assert store.load(d.cwd) == "other-42"
    print("✓ resume_session: recycles the client, points resume=, persists per-cwd")


async def main() -> int:
    test_list_sessions_ordering_and_fallback()
    await test_fork_takes_new_session_id()
    await test_resume_session_switches_and_persists()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
