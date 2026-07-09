"""recall doctor — the deterministic doctrine drift checks, on tmp dirs and
throwaway git repos (no model, no daemon, no Telegram)."""
from __future__ import annotations

import os
import subprocess
import time

from recall import doctor
from recall.doctor import STUB_MEMORY_MD


def _write_stub(mem_dir):
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text(STUB_MEMORY_MD)


# ---- memory stub -----------------------------------------------------------

def test_memory_stub_clean(tmp_path):
    mem = tmp_path / "memory"
    _write_stub(mem)
    (mem / "ARCHIVE.md").write_text("- old index lines live here\n")
    (mem / "archive").mkdir()
    (mem / "archive" / "old-fact.md").write_text("archived, sanctioned")
    assert doctor.check_memory_stub(mem, "proj") == []


def test_memory_stub_whitespace_is_not_drift(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(STUB_MEMORY_MD + "\n\n   \n")
    assert doctor.check_memory_stub(mem, "proj") == []


def test_memory_stub_drift_flagged(tmp_path):
    mem = tmp_path / "memory"
    _write_stub(mem)
    with (mem / "MEMORY.md").open("a") as f:
        f.write("- [A new fact](new-fact.md) — someone bypassed the corpus\n")
    findings = doctor.check_memory_stub(mem, "proj")
    assert len(findings) == 1 and findings[0].level == "warn"
    assert "drifted" in findings[0].message


def test_memory_stray_note_flagged(tmp_path):
    mem = tmp_path / "memory"
    _write_stub(mem)
    (mem / "sneaky-fact.md").write_text("---\nname: x\n---\nfact outside recall")
    findings = doctor.check_memory_stub(mem, "proj")
    assert len(findings) == 1 and "sneaky-fact.md" in findings[0].message


def test_memory_absent_is_fine(tmp_path):
    assert doctor.check_memory_stub(tmp_path / "nope", "proj") == []


# ---- machine commit lane ---------------------------------------------------

_GIT_ENV = {**os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.t"}


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, env=_GIT_ENV)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs" / "knowledge").mkdir(parents=True)
    _git(repo.parent, "init", "-q", str(repo))
    return repo


def test_machine_lane_clean(tmp_path):
    repo = _repo(tmp_path)
    (repo / "docs" / "knowledge" / "a-note.md").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "[curator] 2026-07-09 (proj): a note")
    assert doctor.check_machine_lane(repo, "proj", days=8) == []


def test_machine_lane_escape_flagged(tmp_path):
    repo = _repo(tmp_path)
    (repo / "CLAUDE.md").write_text("instructions")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "[curator] 2026-07-09 (proj): oops")
    findings = doctor.check_machine_lane(repo, "proj", days=8)
    assert len(findings) == 1 and findings[0].level == "error"
    assert "CLAUDE.md" in findings[0].message


def test_machine_lane_ignores_human_commits(tmp_path):
    repo = _repo(tmp_path)
    (repo / "CLAUDE.md").write_text("instructions")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "ada: hand-written CLAUDE.md")
    assert doctor.check_machine_lane(repo, "proj", days=8) == []


def test_machine_lane_no_git_is_silent(tmp_path):
    assert doctor.check_machine_lane(tmp_path, "proj", days=8) == []


# ---- index freshness -------------------------------------------------------

def test_index_missing_is_error(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("---\nname: a\ndescription: d\n---\nb\n")
    findings = doctor.check_index(corpus, tmp_path / "idx.sqlite", "proj")
    assert len(findings) == 1 and findings[0].level == "error"
    assert "NO index" in findings[0].message


def test_index_fresh_is_clean(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("x")
    idx = tmp_path / "idx.sqlite"
    idx.write_text("db")
    assert doctor.check_index(corpus, idx, "proj") == []


def test_index_stale_is_warned(tmp_path):
    # Stale = the index predates the newest NOTE (content-based), not merely
    # old on the wall clock.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("x")
    idx = tmp_path / "idx.sqlite"
    idx.write_text("db")
    old = time.time() - (doctor.INDEX_LAG_HOURS + 5) * 3600
    os.utime(idx, (old, old))
    findings = doctor.check_index(corpus, idx, "proj")
    assert len(findings) == 1 and findings[0].level == "warn"


def test_index_dormant_project_is_not_stale(tmp_path):
    # Old notes + an equally old index = a dormant project, not an outage.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    note = corpus / "a.md"
    note.write_text("x")
    idx = tmp_path / "idx.sqlite"
    idx.write_text("db")
    month_ago = time.time() - 30 * 24 * 3600
    os.utime(note, (month_ago, month_ago))
    os.utime(idx, (month_ago + 60, month_ago + 60))  # rebuilt just after
    assert doctor.check_index(corpus, idx, "proj") == []


def test_index_empty_corpus_is_silent(tmp_path):
    assert doctor.check_index(tmp_path / "nope", tmp_path / "idx", "proj") == []


# ---- rules tier ------------------------------------------------------------

def test_rules_broken_note_is_error(tmp_path):
    (tmp_path / "rule-bad.md").write_text(
        "---\nname: rule-bad\nkind: rule\n---\nno description\n")
    findings = doctor.check_rules(tmp_path, "global")
    assert len(findings) == 1 and findings[0].level == "error"
    assert "silent rule outage" in findings[0].message


def test_rules_over_budget_is_warned(tmp_path, monkeypatch):
    (tmp_path / "rule-long.md").write_text(
        f'---\nname: rule-long\ndescription: "{"x" * 120}"\nkind: rule\n---\nb\n')
    monkeypatch.setenv("RECALL_RULES_BUDGET_CHARS", "50")
    findings = doctor.check_rules(tmp_path, "global")
    assert len(findings) == 1 and findings[0].level == "warn"
    assert "omitted" in findings[0].message


def test_rules_clean(tmp_path):
    (tmp_path / "rule-ok.md").write_text(
        '---\nname: rule-ok\ndescription: "do the thing"\nkind: rule\n---\nb\n')
    assert doctor.check_rules(tmp_path, "global") == []


# ---- run_checks composition ------------------------------------------------

def test_run_checks_clean_on_empty_world(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    assert doctor.run_checks() == 0
    assert "OK" in capsys.readouterr().out


def test_run_checks_flags_gone_project(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "projects.txt").write_text(str(tmp_path / "vanished") + "\n")
    monkeypatch.setenv("RECALL_DATA_ROOT", str(data))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    rc = doctor.run_checks()
    assert rc == 2
    assert "registered path is gone" in capsys.readouterr().out
