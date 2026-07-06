"""curate-sessions-all — the nightly safety-net sweep (brick 2). It discovers each
project's day sessions and forwards one ``curate --session <id>`` per file; curate.run,
list_projects and transcript discovery are patched so no transcripts, model or git are
touched."""
from __future__ import annotations

from pathlib import Path

from recall import registry


class _Out:
    def __init__(self, code=0):
        self.exit_code = code


def test_sweep_forwards_one_call_per_session(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("recall.curate.run",
                        lambda argv: (calls.append(list(argv)) or _Out()))
    monkeypatch.setattr(registry, "list_projects", lambda: [Path("/x/proj-a")])
    monkeypatch.setattr("recall.transcripts.project_transcript_dir",
                        lambda d: Path("/t/proj-a"))
    monkeypatch.setattr("recall.transcripts.discover_transcripts",
                        lambda tdir, target: [Path("/t/proj-a/aaa.jsonl"),
                                              Path("/t/proj-a/bbb.jsonl")])
    assert registry.curate_sessions_all(["--date", "2026-06-01", "--commit"]) == 0
    sids = [c[c.index("--session") + 1] for c in calls]
    assert sids == ["aaa", "bbb"]                       # one call per session, by id
    for c in calls:
        assert "--project-dir" in c and "/x/proj-a" in c
        assert "--commit" in c                          # flag forwards …
        assert "--date" not in c                        # … but the sweep is per-session


def test_sweep_aggregates_failure(monkeypatch):
    def run(argv):
        return _Out(1 if "bad" in argv else 0)          # one session fails
    monkeypatch.setattr("recall.curate.run", run)
    monkeypatch.setattr(registry, "list_projects", lambda: [Path("/x/p")])
    monkeypatch.setattr("recall.transcripts.project_transcript_dir", lambda d: Path("/t"))
    monkeypatch.setattr("recall.transcripts.discover_transcripts",
                        lambda tdir, target: [Path("/t/ok.jsonl"), Path("/t/bad.jsonl")])
    assert registry.curate_sessions_all(["--date", "2026-06-01"]) == 1   # non-zero if any fail


def test_sweep_no_projects_is_clean(monkeypatch):
    monkeypatch.setattr(registry, "list_projects", lambda: [])
    assert registry.curate_sessions_all([]) == 0


def test_sweep_bad_date_returns_1(monkeypatch):
    monkeypatch.setattr(registry, "list_projects", lambda: [Path("/x/p")])
    assert registry.curate_sessions_all(["--date", "nope"]) == 1


def test_sweep_forwards_incremental(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr("recall.curate.run",
                        lambda argv: (calls.append(list(argv)) or _Out()))
    monkeypatch.setattr(registry, "list_projects", lambda: [Path("/x/p")])
    monkeypatch.setattr("recall.transcripts.project_transcript_dir", lambda d: Path("/t"))
    monkeypatch.setattr("recall.transcripts.discover_transcripts",
                        lambda tdir, target: [Path("/t/s.jsonl")])
    assert registry.curate_sessions_all(
        ["--date", "2026-06-01", "--incremental"]) == 0
    assert calls and "--incremental" in calls[0]
