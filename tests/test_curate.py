"""Tests for recall.curate — the wrapper orchestration, with claude/index/git
injected as fakes (no subprocess, no model, no git). Covers scope routing
(project vs global), idempotency, the no-conversations skip, and rejection of a
manifest that references a note the skill never wrote."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

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

def _run_session(tmp_path, monkeypatch, session_id="s", invoke_claude=None,
                 extra=()):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True, exist_ok=True)
    tdir = _transcript(tmp_path / "transcripts", name=f"{session_id}.jsonl")
    argv = ["--project-dir", str(proj), "--session", session_id,
            "--transcript-dir", str(tdir), *extra]
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


# ---- --provisional (live pass over a possibly-still-open conversation) ----

def test_curate_provisional_hints_bundle_and_skips_mark(tmp_path, monkeypatch):
    seen = {}

    def capture(ctx, env, timeout):
        seen["bundle"] = ctx.bundle_text
        return _fake_claude()(ctx, env, timeout)

    _proj, out = _run_session(tmp_path, monkeypatch, invoke_claude=capture,
                              extra=("--provisional",))
    assert out.kind == "curated", out
    # The curator was told this pass is provisional …
    assert seen["bundle"].startswith("> PROVISIONAL PASS")
    # … and the session was NOT claimed: the canonical pass can rerun it.
    state_file = tmp_path / "data" / "curation" / "proj" / "curated.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    assert "s" not in state.get("sessions", [])


def test_curate_provisional_then_canonical_reruns(tmp_path, monkeypatch):
    _run_session(tmp_path, monkeypatch, extra=("--provisional",))
    _proj, out = _run_session(tmp_path, monkeypatch)   # canonical, same session
    assert out.kind == "curated", "provisional must not block the canonical pass"
    state = json.loads(
        (tmp_path / "data" / "curation" / "proj" / "curated.json").read_text())
    assert "s" in state.get("sessions", [])


def test_session_curation_cmd_argv_parses():
    """Cross-boundary contract: the harness spawns `recall curate <argv>`
    DETACHED with stderr to DEVNULL — an unrecognized flag dies silently (the
    61fef58 inert-seam bug, where --provisional didn't exist). Whatever
    session_curation_cmd() builds must parse in curate._parse_args, forever."""
    pytest.importorskip("claude_agent_sdk")   # the harness core imports it at module top
    harness = os.path.join(os.path.dirname(__file__), "..", "infra", "engram")
    sys.path.insert(0, harness)
    try:
        from core import session_curation_cmd
    finally:
        sys.path.remove(harness)
    cmd = session_curation_cmd("abc-123", Path("/tmp/p"), provisional=True)
    assert cmd[1] == "curate"
    ns = curate._parse_args(cmd[2:])          # drop binary + subcommand
    assert ns.session == "abc-123"
    assert ns.provisional is True and ns.commit is True
    # The non-provisional variant must parse too.
    ns2 = curate._parse_args(session_curation_cmd("abc", Path("/tmp/p"))[2:])
    assert ns2.provisional is False


# ---- --buffer (Engram LiveBuffer as the curation source) ---------------------

def _buffer(tmp_path, name="sid-42.jsonl"):
    buf = tmp_path / "buffers" / name
    buf.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"convo_id": buf.stem, "seq": 1, "ts": "2026-06-01T16:00:00+00:00",
         "role": "user", "text": "why does X happen?"},
        {"convo_id": buf.stem, "seq": 2, "ts": "2026-06-01T16:01:00+00:00",
         "role": "assistant", "text": "Because of Y."},
    ]
    buf.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return buf


def _run_buffer(tmp_path, monkeypatch, buf, extra=(), invoke_claude=None):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True, exist_ok=True)
    out = curate.run(["--project-dir", str(proj), "--buffer", str(buf), *extra],
                     invoke_claude=invoke_claude or _fake_claude(),
                     rebuild_indices=lambda ctx: {ctx.slug: 1, "global": 1},
                     autocommit=lambda ctx, m: [],
                     compute_neighbors=lambda ctx: [],
                     compute_surprise=lambda ctx, created: {},
                     today_et=TARGET)
    return proj, out


def test_curate_buffer_mode_shares_sessions_bucket(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)
    _proj, out = _run_buffer(tmp_path, monkeypatch, buf)
    assert out.kind == "curated", out
    state = json.loads(
        (tmp_path / "data" / "curation" / "proj" / "curated.json").read_text())
    # The buffer's convo id IS the SDK session id: one shared state key, so the
    # nightly --session sweep and live eviction see each other's work.
    assert "sid-42" in state.get("sessions", [])
    assert (tmp_path / "data" / "curation" / "proj" / "bundles"
            / "session-sid-42.md").exists()


def test_curate_buffer_missing_is_clean_skip(tmp_path, monkeypatch):
    _proj, out = _run_buffer(tmp_path, monkeypatch,
                             tmp_path / "buffers" / "ghost.jsonl")
    assert out.kind == "skipped" and out.reason == "buffer_missing"
    assert out.exit_code == 0


def test_curate_buffer_provisional_composes(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)
    _proj, out = _run_buffer(tmp_path, monkeypatch, buf, extra=("--provisional",))
    assert out.kind == "curated", out
    state_file = tmp_path / "data" / "curation" / "proj" / "curated.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    assert "sid-42" not in state.get("sessions", [])   # provisional never claims


# ---- --incremental (per-conversation watermark) -----------------------------

def _grow_buffer(buf, seq, ts, role="user", text="later question"):
    row = {"convo_id": buf.stem, "seq": seq, "ts": ts, "role": role, "text": text}
    with buf.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _watermarks(tmp_path):
    state_file = tmp_path / "data" / "curation" / "proj" / "curated.json"
    if not state_file.exists():
        return {}
    return json.loads(state_file.read_text()).get("watermarks", {})


def test_incremental_first_pass_sets_watermark_then_skips(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)
    _proj, out = _run_buffer(tmp_path, monkeypatch, buf, extra=("--incremental",))
    assert out.kind == "curated", out
    assert _watermarks(tmp_path)["sid-42"] == "2026-06-01T16:01:00+00:00"
    # Buckets untouched: the watermark IS the incremental state.
    state = json.loads(
        (tmp_path / "data" / "curation" / "proj" / "curated.json").read_text())
    assert "sid-42" not in state.get("sessions", [])
    # Second pass, nothing new → clean skip, exit 0, watermark unchanged.
    _proj, out2 = _run_buffer(tmp_path, monkeypatch, buf, extra=("--incremental",))
    assert out2.kind == "skipped" and out2.reason == "no_new_exchanges"
    assert out2.exit_code == 0
    assert _watermarks(tmp_path)["sid-42"] == "2026-06-01T16:01:00+00:00"


def test_incremental_curates_only_new_tail(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)
    _run_buffer(tmp_path, monkeypatch, buf, extra=("--incremental",))
    _grow_buffer(buf, 3, "2026-06-01T17:00:00+00:00", text="a NEW question")
    seen = {}

    def capture(ctx, env, timeout):
        seen["bundle"] = ctx.bundle_text
        return _fake_claude()(ctx, env, timeout)

    _proj, out = _run_buffer(tmp_path, monkeypatch, buf,
                             extra=("--incremental",), invoke_claude=capture)
    assert out.kind == "curated", out
    assert "a NEW question" in seen["bundle"]
    assert "why does X happen?" not in seen["bundle"]   # already-curated head sliced off
    assert _watermarks(tmp_path)["sid-42"] == "2026-06-01T17:00:00+00:00"


def test_incremental_watermark_not_advanced_on_failure(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)

    def broken_claude(ctx, env, timeout):
        return subprocess.CompletedProcess(args=[], returncode=3)

    _proj, out = _run_buffer(tmp_path, monkeypatch, buf,
                             extra=("--incremental",), invoke_claude=broken_claude)
    assert out.kind == "failed"
    assert _watermarks(tmp_path) == {}      # failed pass never advances → no tail lost


def test_incremental_until_caps_slice_and_watermark(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)
    _grow_buffer(buf, 3, "2026-06-01T17:00:00+00:00", text="still hot")
    seen = {}

    def capture(ctx, env, timeout):
        seen["bundle"] = ctx.bundle_text
        return _fake_claude()(ctx, env, timeout)

    _proj, out = _run_buffer(
        tmp_path, monkeypatch, buf,
        extra=("--incremental", "--until", "2026-06-01T16:30:00+00:00"),
        invoke_claude=capture)
    assert out.kind == "curated", out
    assert "still hot" not in seen["bundle"]            # hot row stays uncurated
    assert _watermarks(tmp_path)["sid-42"] == "2026-06-01T16:01:00+00:00"


def test_incremental_force_recurates_whole_and_rewrites(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)
    _run_buffer(tmp_path, monkeypatch, buf, extra=("--incremental",))
    seen = {}

    def capture(ctx, env, timeout):
        seen["bundle"] = ctx.bundle_text
        return _fake_claude()(ctx, env, timeout)

    _proj, out = _run_buffer(tmp_path, monkeypatch, buf,
                             extra=("--incremental", "--force"),
                             invoke_claude=capture)
    assert out.kind == "curated", out
    assert "why does X happen?" in seen["bundle"]       # whole convo re-read
    assert _watermarks(tmp_path)["sid-42"] == "2026-06-01T16:01:00+00:00"


def test_incremental_ignores_sessions_bucket(tmp_path, monkeypatch):
    # A session canonically curated (bucket-marked) that then GROWS after a
    # resume: incremental must still curate it — the watermark alone governs.
    buf = _buffer(tmp_path)
    _run_buffer(tmp_path, monkeypatch, buf)             # canonical: marks bucket
    _grow_buffer(buf, 3, "2026-06-01T18:00:00+00:00", text="post-resume growth")
    _proj, out = _run_buffer(tmp_path, monkeypatch, buf, extra=("--incremental",))
    assert out.kind == "curated", "bucket mark must not orphan post-resume growth"
    assert _watermarks(tmp_path)["sid-42"] == "2026-06-01T18:00:00+00:00"


def test_incremental_provisional_advances_watermark_only(tmp_path, monkeypatch):
    buf = _buffer(tmp_path)
    _proj, out = _run_buffer(tmp_path, monkeypatch, buf,
                             extra=("--incremental", "--provisional"))
    assert out.kind == "curated", out
    assert _watermarks(tmp_path)["sid-42"]              # position tracked …
    state = json.loads(
        (tmp_path / "data" / "curation" / "proj" / "curated.json").read_text())
    assert "sid-42" not in state.get("sessions", [])    # … unit never claimed


def test_incremental_bad_flag_combos_fail_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True)
    tdir = _transcript(tmp_path / "transcripts")
    common = dict(invoke_claude=_fake_claude(), rebuild_indices=lambda c: {},
                  autocommit=lambda c, m: [], today_et=TARGET)
    # --incremental needs a per-conversation unit …
    out = curate.run(["--project-dir", str(proj), "--date", TARGET.isoformat(),
                      "--transcript-dir", str(tdir), "--incremental"], **common)
    assert out.kind == "failed" and out.reason == "bad_flags"
    # … and --until means nothing outside --incremental.
    out = curate.run(["--project-dir", str(proj), "--session", "s",
                      "--transcript-dir", str(tdir),
                      "--until", "2026-06-01T16:00:00+00:00"], **common)
    assert out.kind == "failed" and out.reason == "bad_flags"
    # Garbled --until fails loud, not silent.
    out = curate.run(["--project-dir", str(proj), "--session", "s",
                      "--transcript-dir", str(tdir), "--incremental",
                      "--until", "not-a-ts"], **common)
    assert out.kind == "failed" and out.reason == "bad_until"


def test_incremental_session_mode_watermarks_transcript(tmp_path, monkeypatch):
    # The nightly sweep's form: --session <id> --incremental over a CC transcript.
    _proj, out = _run_session(tmp_path, monkeypatch, extra=("--incremental",))
    assert out.kind == "curated", out
    assert _watermarks(tmp_path)["s"] == "2026-06-01T16:01:00+00:00"
    _proj, out2 = _run_session(tmp_path, monkeypatch, extra=("--incremental",))
    assert out2.kind == "skipped" and out2.reason == "no_new_exchanges"


# ---- supersession → valid_to backstop (Brick 3, B5) -------------------------

def test_supersession_stamps_valid_to(tmp_path, monkeypatch):
    def superseding_claude(ctx, env, timeout):
        # New note replaces an old one; curator sets superseded_by but FORGETS
        # valid_to — the backstop must stamp it with the filing date.
        (ctx.project_knowledge_dir / "new-way.md").write_text(
            "---\nname: new-way\ndescription: the new approach\n---\nUse B.\n")
        (ctx.project_knowledge_dir / "old-way.md").write_text(
            "---\nname: old-way\ndescription: the old approach\n"
            "superseded: true\nsuperseded_by: new-way\n---\nUse A.\n")
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "superseded old-way with new-way",
            "notes": [
                {"slug": "new-way", "action": "created", "title": "new",
                 "scope": "project"},
                {"slug": "old-way", "action": "updated", "title": "old",
                 "scope": "project"}]}))
        return subprocess.CompletedProcess(args=[], returncode=0)

    proj, out = _run(tmp_path, monkeypatch, invoke_claude=superseding_claude)
    assert out.kind == "curated", out
    old = KnowledgeNote.parse((proj / "docs" / "knowledge" / "old-way.md").read_text())
    assert old.valid_to == TARGET.isoformat()      # backstop stamped it
    new = KnowledgeNote.parse((proj / "docs" / "knowledge" / "new-way.md").read_text())
    assert new.valid_to == ""                      # the replacement stays open


def test_supersession_never_overwrites_existing_valid_to(tmp_path, monkeypatch):
    def claude(ctx, env, timeout):
        (ctx.project_knowledge_dir / "new-way.md").write_text(
            "---\nname: new-way\ndescription: the new approach\n---\nUse B.\n")
        (ctx.project_knowledge_dir / "old-way.md").write_text(
            "---\nname: old-way\ndescription: the old approach\n"
            "superseded: true\nsuperseded_by: new-way\n"
            "valid_to: 2026-05-15\n---\nUse A.\n")   # curator knew the real date
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "superseded with explicit reversal date",
            "notes": [
                {"slug": "new-way", "action": "created", "title": "n",
                 "scope": "project"},
                {"slug": "old-way", "action": "updated", "title": "o",
                 "scope": "project"}]}))
        return subprocess.CompletedProcess(args=[], returncode=0)

    proj, out = _run(tmp_path, monkeypatch, invoke_claude=claude)
    assert out.kind == "curated", out
    old = KnowledgeNote.parse((proj / "docs" / "knowledge" / "old-way.md").read_text())
    assert old.valid_to == "2026-05-15"            # curator's date wins


# ---- index rebuild: daemon-first embedding + scream-on-failure ------------

def _fake_fire_ctx(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(slug="proj",
                           project_knowledge_dir=tmp_path / "proj" / "docs" / "knowledge",
                           project_index_path=tmp_path / "idx" / "proj.sqlite",
                           global_dir=tmp_path / "global",
                           global_index_path=tmp_path / "idx" / "global.sqlite")


def test_rebuild_indices_prefers_the_warm_daemon(tmp_path, monkeypatch):
    # Loading a second in-process model while the daemon holds the GPU is the
    # CUDA OOM that silently froze the index 2026-07-04..06 — daemon first.
    import recall.index as rindex
    daemon = object()
    monkeypatch.setattr(rindex, "DaemonEmbedder", lambda: daemon)
    monkeypatch.setattr(rindex, "SentenceTransformerEmbedder",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("must not load an in-process model")))
    used = []
    monkeypatch.setattr(rindex, "build_index",
                        lambda d, p, e: used.append(e) or 1)
    out = curate._rebuild_indices(_fake_fire_ctx(tmp_path))
    assert out == {"proj": 1, "global": 1}
    assert used == [daemon, daemon]


def test_rebuild_indices_falls_back_when_daemon_down(tmp_path, monkeypatch):
    import recall.index as rindex
    inproc = object()
    monkeypatch.setattr(rindex, "DaemonEmbedder",
                        lambda: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setattr(rindex, "SentenceTransformerEmbedder", lambda: inproc)
    used = []
    monkeypatch.setattr(rindex, "build_index",
                        lambda d, p, e: used.append(e) or 2)
    out = curate._rebuild_indices(_fake_fire_ctx(tmp_path))
    assert out == {"proj": 2, "global": 2}
    assert used == [inproc, inproc]


def test_rebuild_failure_screams_but_curation_stands(tmp_path, monkeypatch):
    # A failed rebuild must NOT fail the curation (corpus + manifest landed; the
    # index is derived) — but it must SCREAM: the silent WARN here is what let
    # production recall serve a stale index for two invisible days.
    alerts = []
    monkeypatch.setattr(
        curate, "notify_alert",
        lambda *, title, body, priority: alerts.append((title, body, priority)) or True)
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    proj = tmp_path / "proj"
    (proj / "docs" / "knowledge").mkdir(parents=True, exist_ok=True)
    tdir = _transcript(tmp_path / "transcripts")

    def boom(ctx):
        raise RuntimeError("CUDA out of memory")

    out = curate.run(["--project-dir", str(proj), "--date", TARGET.isoformat(),
                      "--transcript-dir", str(tdir)],
                     invoke_claude=_fake_claude(), rebuild_indices=boom,
                     autocommit=lambda ctx, m: [],
                     compute_neighbors=lambda ctx: [],
                     compute_surprise=lambda ctx, created: {},
                     today_et=TARGET)
    assert out.kind == "curated" and out.exit_code == 0     # curation stands
    assert len(alerts) == 1                                  # exactly one scream
    title, body, priority = alerts[0]
    assert priority == "urgent"
    assert "index_rebuild_failed" in title
    assert "STALE" in body and "CUDA out of memory" in body
