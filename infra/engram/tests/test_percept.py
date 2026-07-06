#!/usr/bin/env python3
"""Headless tests for PerceptMemory (perceiving-loop step 5): the clean-
perception gate (stable-only eye reads, scene-change dedup, fail-closed on
unknown kinds), provenance-carrying rows the engine can read back, the
size-gated detached curate spawn + watermark respect, day-file rollover
with a full flush of the closing day, and the cutouts.

    .venv/bin/python infra/engram/tests/test_percept.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)
sys.path.insert(0, os.path.join(ENGRAM, "perceive"))
REPO = os.path.abspath(os.path.join(ENGRAM, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from percept import PerceptMemory  # noqa: E402

T0 = 1751500000.0                  # a fixed instant; day math offsets from here


class Ev:
    """Duck-typed stand-in for loop.Event — percept reads t/kind/detail/data."""
    def __init__(self, kind, detail="detail", data=None, t=T0):
        self.t, self.kind, self.detail, self.data = t, kind, detail, data or {}


class FakeProc:
    def __init__(self, rc=None):
        self._rc = rc              # None = still running
    def poll(self):
        return self._rc


def _mem(tmp, **kw):
    calls = []
    def spawn(path, cwd, *, until=None, provisional=True):
        calls.append({"path": str(path), "until": until,
                      "provisional": provisional})
        return FakeProc()          # rc=None: "still running" until a test says otherwise
    kw.setdefault("enabled", True)
    kw.setdefault("evict_on", True)
    m = PerceptMemory(Path(tmp) / "percept", cwd=Path(tmp), spawn=spawn, **kw)
    return m, calls


def _rows(m):
    p = m._buffer.path()
    if p is None or not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_gate_transitions_persist_unknown_dropped():
    with tempfile.TemporaryDirectory() as tmp:
        m, _ = _mem(tmp)
        m.on_event(Ev("engage", "Ada is here — engaging", {"person": "Ada"}))
        m.on_event(Ev("ambient", "background hum"))       # never memory
        m.on_event(Ev("overheard", "someone else talking"))
        m.on_event(Ev("wtf", "novel kind"))               # fail-closed
        m.on_event(Ev("heard", "Engram, hello", {"text": "hello"}))
        m.on_event(Ev("idle", "frame is clear — resting"))
        rows = _rows(m)
        assert [r["kind"] for r in rows] == ["engage", "heard", "idle"]
        assert all(r["role"] == "perception" for r in rows)
        assert rows[0]["text"].startswith("[engage] Ada is here")
        assert rows[0]["data"] == {"person": "Ada"}      # provenance rides along
    print("✓ gate: transitions + heard persist; ambient/overheard/unknown never do")


def test_eye_requires_stable_and_scene_change():
    with tempfile.TemporaryDirectory() as tmp:
        m, _ = _mem(tmp)
        m.on_event(Ev("eye", "a cup?", {"stable": False, "corroborated": []}))
        assert _rows(m) == []                              # unconfirmed → dropped
        m.on_event(Ev("eye", "desk + laptop [✓]",
                      {"stable": True, "corroborated": ["desk", "laptop"]}))
        m.on_event(Ev("eye", "desk + laptop again [✓]",
                      {"stable": True, "corroborated": ["laptop", "desk"]}))
        assert len(_rows(m)) == 1                          # same scene → one row
        m.on_event(Ev("eye", "now a guitar [✓]",
                      {"stable": True, "corroborated": ["guitar"]}))
        rows = _rows(m)
        assert len(rows) == 2 and "guitar" in rows[1]["text"]
    print("✓ eye: only STABLE reads persist, and only when the scene CHANGES")


def test_rows_readable_by_engine():
    with tempfile.TemporaryDirectory() as tmp:
        m, _ = _mem(tmp)
        m.on_event(Ev("engage", "Ada is here — engaging"))
        m.on_event(Ev("eye", "(1.2s) a desk  [✓ desk]",
                      {"stable": True, "corroborated": ["desk"]}))
        from recall import transcripts
        exs = list(transcripts.iter_buffer_exchanges(m._buffer.path()))
        assert [e.role for e in exs] == ["perception", "perception"]
        text, stats = transcripts.build_buffer_bundle(m._buffer.path())
        assert stats.exchanges == 2 and "### PERCEPTION" in text
    print("✓ engine round-trip: percept rows read back + render as ### PERCEPTION")


def test_size_gate_watermark_and_guard():
    with tempfile.TemporaryDirectory() as tmp, \
         tempfile.TemporaryDirectory() as dr:
        os.environ["RECALL_DATA_ROOT"] = str(dr)
        m, calls = _mem(tmp, evict_chars=200, hot_rows=2)
        for i in range(4):
            m.on_event(Ev("engage", f"short {i}", t=T0 + i))
        assert calls == [], "below the char gate → no spawn"
        for i in range(6):
            m.on_event(Ev("engage", "x" * 80, t=T0 + 10 + i))
        assert len(calls) == 1 and calls[0]["until"] is not None
        assert calls[0]["provisional"] is True
        # in-flight guard: FakeProc.poll() is None → no second spawn
        for i in range(6):
            m.on_event(Ev("engage", "y" * 80, t=T0 + 30 + i))
        assert len(calls) == 1, "one detached curate at a time"
        # watermark respect: mark everything curated → cooled tail is empty
        from recall import config
        sf = (config.curation_dir() / config.project_slug(Path(tmp))
              / "curated.json")
        sf.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        mark = datetime.now(timezone.utc).isoformat()
        sf.write_text(json.dumps(
            {"watermarks": {m._buffer.path().stem: mark}}))
        m._proc = FakeProc(rc=0)                # previous curate finished
        m.on_event(Ev("engage", "z" * 80, t=T0 + 60))
        assert len(calls) == 1, "everything before the watermark is curated"
    print("✓ size gate + one-at-a-time guard + watermark scope the spawn")


def test_day_rollover_flushes_closing_day():
    with tempfile.TemporaryDirectory() as tmp:
        m, calls = _mem(tmp)
        m.on_event(Ev("engage", "day one", t=T0))
        day1 = m._buffer.path()
        m.on_event(Ev("engage", "day two", t=T0 + 3 * 86400))
        assert len(calls) == 1 and calls[0]["until"] is None   # full flush
        assert calls[0]["path"] == str(day1)
        assert m._buffer.path() != day1                        # new day file
        rows = _rows(m)
        assert [r["text"][-7:] for r in rows] == ["day two"]
        assert rows[0]["seq"] == 1                             # seq reseeded
    print("✓ day rollover: closing day fully flushed, fresh file + seq for the new day")


def test_flush_and_cutouts():
    # flush() folds the whole tail
    with tempfile.TemporaryDirectory() as tmp:
        m, calls = _mem(tmp)
        m.on_event(Ev("engage", "something"))
        m.flush()
        assert len(calls) == 1 and calls[0]["until"] is None
    # enabled=False → no file ever
    with tempfile.TemporaryDirectory() as tmp:
        m, calls = _mem(tmp, enabled=False)
        m.on_event(Ev("engage", "invisible"))
        m.flush()
        assert _rows(m) == [] and calls == []
        assert m._buffer.path() is None
    # evict_on=False → rows persist, nothing spawns (manual sweep later)
    with tempfile.TemporaryDirectory() as tmp:
        m, calls = _mem(tmp, evict_on=False, evict_chars=1, hot_rows=0)
        for i in range(5):
            m.on_event(Ev("engage", "kept" * 20, t=T0 + i))
        m.flush()
        assert len(_rows(m)) == 5 and calls == []
    print("✓ flush folds the tail; ENGRAM_PERCEPT=0 / ENGRAM_EVICT=0 cutouts hold")


def test_wrap_forwards_even_if_memory_breaks():
    with tempfile.TemporaryDirectory() as tmp:
        m, _ = _mem(tmp)
        seen = []
        cb = m.wrap(lambda ev: seen.append(ev.kind))
        cb(Ev("engage", "fine"))
        m._gate = None                      # sabotage: on_event now raises
        cb(Ev("idle", "still forwarded"))
        assert seen == ["engage", "idle"], "the mind must always hear the event"
    print("✓ wrap: memory failure never eats the event for the mind")


def main() -> int:
    test_gate_transitions_persist_unknown_dropped()
    test_eye_requires_stable_and_scene_change()
    test_rows_readable_by_engine()
    test_size_gate_watermark_and_guard()
    test_day_rollover_flushes_closing_day()
    test_flush_and_cutouts()
    test_wrap_forwards_even_if_memory_breaks()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
