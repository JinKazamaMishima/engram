"""Tests for recall.curate — the wrapper orchestration, with claude/index/git
injected as fakes (no subprocess, no model, no git). Covers scope routing
(project vs global), idempotency, the no-conversations skip, and rejection of a
manifest that references a note the skill never wrote."""
from __future__ import annotations

import json
import subprocess
from datetime import date

from recall import curate, dynamics
from recall.schema import KnowledgeNote

TARGET = date(2026, 6, 1)


def _transcript(dir_, name="s.jsonl"):
    dir_.mkdir(parents=True, exist_ok=True)
    ev = [
        {"type": "user",
         "message": {"role": "user", "content": "why does X happen?"},
         "timestamp": "2026-06-01T16:00:00Z"},
        {"type": "assistant",
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": "Because of Y."}]},
         "timestamp": "2026-06-01T16:01:00Z"},
    ]
    (dir_ / name).write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    return dir_


def _fake_claude(project_note=True, global_note=True):
    def _inner(ctx, env, timeout):
        notes = []
        if project_note:
            (ctx.project_knowledge_dir / "proj-insight.md").write_text(
                "---\nname: proj-insight\ndescription: a durable project insight\n"
                "tags: [t]\n---\nThe mechanic, with the why.\n")
            notes.append({"slug": "proj-insight", "action": "created",
                          "title": "a project insight", "scope": "project"})
        if global_note:
            (ctx.global_dir / "owner-prefers-hard-route.md").write_text(
                "---\nname: owner-prefers-hard-route\ndescription: owner values the "
                "complicated route\nkind: identity\n---\nLearns by doing the hard thing.\n")
            notes.append({"slug": "owner-prefers-hard-route", "action": "created",
                          "title": "values the hard route", "scope": "global"})
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "captured one project + one global note", "notes": notes}))
        return subprocess.CompletedProcess(args=[], returncode=0)
    return _inner


def _run(tmp_path, monkeypatch, invoke_claude=None, compute_neighbors=None,
         compute_surprise=None):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True, exist_ok=True)
    tdir = _transcript(tmp_path / "transcripts")
    argv = ["--project-dir", str(proj), "--date", TARGET.isoformat(),
            "--transcript-dir", str(tdir)]
    out = curate.run(argv, invoke_claude=invoke_claude or _fake_claude(),
                     rebuild_indices=lambda ctx: {ctx.slug: 1, "global": 1},
                     autocommit=lambda ctx, m: [],
                     compute_neighbors=compute_neighbors or (lambda ctx: []),
                     compute_surprise=compute_surprise or (lambda ctx, created: {}),
                     today_et=TARGET)
    return proj, out


def test_curate_writes_both_scopes(tmp_path, monkeypatch):
    proj, out = _run(tmp_path, monkeypatch)
    assert out.kind == "curated", out
    assert (proj / "docs" / "knowledge" / "proj-insight.md").exists()
    soul = tmp_path / "data" / "global" / "owner-prefers-hard-route.md"
    assert soul.exists()  # global-scope note routed to the soul corpus
    state = json.loads(
        (tmp_path / "data" / "curation" / "proj" / "curated.json").read_text())
    assert TARGET.isoformat() in state["dates"]
    log = tmp_path / "data" / "curation" / "proj" / "sessions" / "2026-06-01.md"
    assert log.exists() and "global" in log.read_text()


def test_curate_idempotent(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    _proj, out = _run(tmp_path, monkeypatch)
    assert out.kind == "skipped" and out.reason == "already_curated"


def test_curate_no_conversations_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    out = curate.run(["--project-dir", str(proj), "--date", TARGET.isoformat(),
                      "--transcript-dir", str(empty)],
                     invoke_claude=_fake_claude(), rebuild_indices=lambda c: {},
                     autocommit=lambda c, m: [], today_et=TARGET)
    assert out.kind == "skipped" and out.reason == "no_conversations"


def test_curate_injects_neighbors_sidecar(tmp_path, monkeypatch):
    from pathlib import Path
    seen = {}

    def claude(ctx, env, timeout):
        p = env.get("RECALL_CURATE_NEIGHBORS")
        seen["path"] = p
        seen["data"] = json.loads(Path(p).read_text())
        return _fake_claude()(ctx, env, timeout)

    nbrs = [{"slug": "existing-note", "scope": "proj",
             "description": "d", "score": 0.5}]
    _proj, out = _run(tmp_path, monkeypatch, invoke_claude=claude,
                      compute_neighbors=lambda ctx: nbrs)
    assert out.kind == "curated", out
    assert seen["path"] and seen["data"]["neighbors"][0]["slug"] == "existing-note"


def test_curate_real_neighbors_empty_without_index(tmp_path, monkeypatch):
    # The real _compute_neighbors runs (not injected). No index -> [] BEFORE any
    # torch import, and the sidecar is written empty. Hermetic.
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True)
    tdir = _transcript(tmp_path / "transcripts")
    out = curate.run(["--project-dir", str(proj), "--date", TARGET.isoformat(),
                      "--transcript-dir", str(tdir)],
                     invoke_claude=_fake_claude(),
                     rebuild_indices=lambda ctx: {ctx.slug: 1, "global": 1},
                     autocommit=lambda ctx, m: [], today_et=TARGET)
    nb = tmp_path / "data" / "curation" / "proj" / "neighbors" / "2026-06-01.json"
    assert out.kind == "curated"
    assert nb.exists() and json.loads(nb.read_text())["neighbors"] == []


def test_curate_rejects_manifest_referencing_missing_note(tmp_path, monkeypatch):
    def bad_claude(ctx, env, timeout):
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "claims a note it never wrote",
            "notes": [{"slug": "ghost", "action": "created", "title": "x",
                       "scope": "project"}]}))
        return subprocess.CompletedProcess(args=[], returncode=0)
    _proj, out = _run(tmp_path, monkeypatch, invoke_claude=bad_claude)
    assert out.kind == "failed" and out.reason == "note_missing"


# ---- Phase III: birth stability (surprise at encoding + permanence) -------

def test_curate_birth_stability_identity_permanent_project_surprise(tmp_path, monkeypatch):
    # project note born with surprise-scaled stability; identity note born permanent
    proj, out = _run(tmp_path, monkeypatch,
                     compute_surprise=lambda ctx, created: {("project", "proj-insight"): 0.4})
    assert out.kind == "curated", out

    pn = KnowledgeNote.parse((proj / "docs" / "knowledge" / "proj-insight.md").read_text())
    assert pn.surprise == 0.6                       # σ = 1 − max_sim(0.4)
    assert pn.stability == round(dynamics.initial_stability(0.6), 3)
    assert pn.last_used == TARGET.isoformat() and pn.uses == 0

    sn = KnowledgeNote.parse(
        (tmp_path / "data" / "global" / "owner-prefers-hard-route.md").read_text())
    assert sn.stability == round(dynamics.S_PERM, 1)   # identity → born permanent
    assert sn.importance == 1.0 and sn.surprise == 0.9 and dynamics.is_permanent(sn.stability)


def test_curate_birth_stability_does_not_clobber_seed(tmp_path, monkeypatch):
    # a note that already carries a stability (a manual seed like Engram) is left alone
    def claude(ctx, env, timeout):
        (ctx.global_dir / "engram.md").write_text(
            "---\nname: engram\ndescription: my name is engram\nkind: identity\n"
            "stability: 400.0\nimportance: 1.0\n---\nThe name Ada gave me.\n")
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "seeded engram",
            "notes": [{"slug": "engram", "action": "created", "title": "engram",
                       "scope": "global"}]}))
        return subprocess.CompletedProcess(args=[], returncode=0)
    _proj, out = _run(tmp_path, monkeypatch, invoke_claude=claude)
    assert out.kind == "curated"
    note = KnowledgeNote.parse((tmp_path / "data" / "global" / "engram.md").read_text())
    assert note.stability == 400.0     # preserved; NOT reset to S_PERM by birth-stability


# ---- brick 1: session-scoped curation ------------------------------------

def _run_session(tmp_path, monkeypatch, session_id="s", invoke_claude=None):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True, exist_ok=True)
    tdir = _transcript(tmp_path / "transcripts", name=f"{session_id}.jsonl")
    argv = ["--project-dir", str(proj), "--session", session_id,
            "--transcript-dir", str(tdir)]
    out = curate.run(argv, invoke_claude=invoke_claude or _fake_claude(),
                     rebuild_indices=lambda ctx: {ctx.slug: 1, "global": 1},
                     autocommit=lambda ctx, m: [],
                     compute_neighbors=lambda ctx: [],
                     compute_surprise=lambda ctx, created: {},
                     today_et=TARGET)
    return proj, out


def test_curate_session_scoped(tmp_path, monkeypatch):
    proj, out = _run_session(tmp_path, monkeypatch)
    assert out.kind == "curated", out
    assert (proj / "docs" / "knowledge" / "proj-insight.md").exists()
    state = json.loads(
        (tmp_path / "data" / "curation" / "proj" / "curated.json").read_text())
    assert "s" in state.get("sessions", [])     # tracked in the sessions bucket …
    assert state.get("dates", []) == []         # … NOT the dates bucket
    assert (tmp_path / "data" / "curation" / "proj" / "bundles"
            / "session-s.md").exists()          # artifacts under a session stem


def test_curate_session_idempotent(tmp_path, monkeypatch):
    _run_session(tmp_path, monkeypatch)
    _proj, out = _run_session(tmp_path, monkeypatch)
    assert out.kind == "skipped" and out.reason == "already_curated"


def test_curate_session_missing_is_clean_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True)
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    out = curate.run(["--project-dir", str(proj), "--session", "ghost",
                      "--transcript-dir", str(tdir)],
                     invoke_claude=_fake_claude(), rebuild_indices=lambda c: {},
                     autocommit=lambda c, m: [], today_et=TARGET)
    assert out.kind == "skipped" and out.reason == "session_missing"


def test_curate_date_and_session_buckets_independent(tmp_path, monkeypatch):
    # A day-sweep and a session curation on the same project keep separate state,
    # so neither clobbers the other's idempotency (the nightly<->live guarantee).
    _run(tmp_path, monkeypatch)                        # date-scoped
    _proj, out = _run_session(tmp_path, monkeypatch)   # session-scoped, same proj
    assert out.kind == "curated", out
    state = json.loads(
        (tmp_path / "data" / "curation" / "proj" / "curated.json").read_text())
    assert TARGET.isoformat() in state["dates"] and "s" in state["sessions"]
