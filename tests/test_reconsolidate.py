"""Tests for recall.reconsolidate — the weekly maintenance wrapper, with
claude/model/git injected as fakes (no subprocess, no model, no git). Covers the
merge-by-supersede path, per-week idempotency, the empty-worklist skip, the reused
manifest validator, and the hermetic trivial-corpus precompute."""
from __future__ import annotations

import json
import subprocess
from datetime import date

from recall import reconsolidate

TARGET = date(2026, 6, 8)


def _seed_global(tmp_path):
    corpus = tmp_path / "data" / "global"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "survivor.md").write_text(
        "---\nname: survivor\ndescription: the kept note\ntags: [t]\n"
        "last_updated: 2026-06-01\nsources: [2026-06-01]\n---\nThe insight.\n")
    (corpus / "dup.md").write_text(
        "---\nname: dup\ndescription: a near twin\ntags: [t]\n"
        "last_updated: 2026-05-01\nsources: [2026-05-01]\n---\nSame, dimmer.\n")
    return corpus


def _cands(scope="global"):
    return {"scope": scope,
            "duplicate_pairs": [{"a": "dup", "b": "survivor", "score": 0.9}],
            "link_candidates": [], "stale": []}


def _merge_claude(ctx, env, timeout):
    # supersede 'dup' in place + write a valid manifest listing it as updated
    (ctx.corpus_dir / "dup.md").write_text(
        "---\nname: dup\ndescription: a near twin\ntags: [t]\n"
        "superseded: true\nsuperseded_by: survivor\n"
        "last_updated: 2026-06-08\nsources: [2026-05-01, 2026-06-08]\n---\n"
        "Superseded by [[survivor]] (2026-06-08).\n")
    ctx.manifest_path.write_text(json.dumps({
        "schema_version": 1, "date": ctx.target.isoformat(),
        "summary": "merged dup into survivor",
        "notes": [{"slug": "dup", "action": "updated",
                   "title": "superseded by survivor", "scope": ctx.scope}]}))
    return subprocess.CompletedProcess(args=[], returncode=0)


def _run(tmp_path, monkeypatch, *, invoke_claude=None, compute_candidates=None,
         force=False):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    _seed_global(tmp_path)
    argv = ["--scope", "global"] + (["--force"] if force else [])
    return reconsolidate.run(
        argv, invoke_claude=invoke_claude or _merge_claude,
        compute_candidates=compute_candidates or (lambda ctx: _cands()),
        rebuild_index=lambda ctx: 2, autocommit=lambda ctx, m: None,
        today_et=TARGET)


def test_reconsolidate_merges_by_supersede(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch)
    assert out.kind == "curated", out
    dup = tmp_path / "data" / "global" / "dup.md"
    assert dup.exists()                       # never deleted
    assert "superseded: true" in dup.read_text()


def test_reconsolidate_idempotent_per_week(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    out = _run(tmp_path, monkeypatch)
    assert out.kind == "skipped" and out.reason == "already_reconsolidated"
    # --force overrides
    forced = _run(tmp_path, monkeypatch, force=True)
    assert forced.kind == "curated"


def test_reconsolidate_nothing_to_do(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, compute_candidates=lambda ctx: {
        "scope": "global", "duplicate_pairs": [], "link_candidates": [],
        "stale": []})
    assert out.kind == "skipped" and out.reason == "nothing_to_consolidate"


def test_reconsolidate_rejects_missing_note(tmp_path, monkeypatch):
    def bad(ctx, env, timeout):
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "claims a ghost note",
            "notes": [{"slug": "ghost", "action": "updated", "title": "x",
                       "scope": ctx.scope}]}))
        return subprocess.CompletedProcess(args=[], returncode=0)
    out = _run(tmp_path, monkeypatch, invoke_claude=bad)
    assert out.kind == "failed" and out.reason == "note_missing"


def test_reconsolidate_dry_run_no_claude(tmp_path, monkeypatch):
    def boom(ctx, env, timeout):
        raise AssertionError("claude must not be invoked on --dry-run")
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    _seed_global(tmp_path)
    out = reconsolidate.run(
        ["--scope", "global", "--dry-run"], invoke_claude=boom,
        compute_candidates=lambda ctx: _cands(), rebuild_index=lambda ctx: 2,
        autocommit=lambda ctx, m: None, today_et=TARGET)
    assert out.kind == "skipped" and out.reason == "dry_run"


def test_compute_candidates_trivial_corpus(tmp_path, monkeypatch):
    # <2 notes -> empty worklist, returned BEFORE any numpy/torch import.
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    corpus = tmp_path / "data" / "global"
    corpus.mkdir(parents=True)
    (corpus / "only.md").write_text(
        "---\nname: only\ndescription: d\ntags: [t]\n---\nbody\n")
    ctx = reconsolidate._resolve_context(
        reconsolidate._parse_args(["--scope", "global"]), TARGET, TARGET)
    c = reconsolidate._compute_candidates(ctx)
    assert c["duplicate_pairs"] == [] and c["link_candidates"] == []
