"""Tests for recall.dream — the dream pass + bleed membrane. The claude
subprocess, the embedder pairing, soul promotion and git are injected, so these
run with no model and no repo. Covers: hypothesis generation lands QUARANTINED in
the subconscious (never the corpus), the bleed valve (corroboration- and
blessing-gated, rate-limited promotion), decay of stale hypotheses, the digest,
and idempotency."""
from __future__ import annotations

import json
import subprocess
from datetime import date

from recall import config, dream
from recall.schema import KnowledgeNote

TARGET = date(2026, 6, 24)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    config.global_corpus_dir().mkdir(parents=True, exist_ok=True)  # the soul always exists


def _note(corpus_dir, slug, *, last_used="", first_seen="2026-01-01",
          body="A durable note body, with the why."):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    fm = [f"name: {slug}", f'description: "{slug}: a thing"', f"first_seen: {first_seen}"]
    if last_used:
        fm.append(f"last_used: {last_used}")
    (corpus_dir / f"{slug}.md").write_text(
        "---\n" + "\n".join(fm) + "\n---\n" + body + "\n")


def _hyp(sub_dir, slug, *, parents=("p-a", "p-b"), first_seen="2026-06-01",
         corroborations=0, blessed="false", status="unverified"):
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: \"a conjecture: x relates to y\"\n"
        f"kind: hypothesis\nparents: [{parents[0]}, {parents[1]}]\n"
        f"status: {status}\nfirst_seen: {first_seen}\n"
        f"corroborations: {corroborations}\nblessed: {blessed}\n"
        f"stability: 1.0\n---\nThe conjecture body.\n")


def _fake_dream_claude(hyp_slug="latent-link", parents=("today-note", "old-note")):
    def _inner(ctx, env, timeout):
        ctx.subconscious_dir.mkdir(parents=True, exist_ok=True)
        (ctx.subconscious_dir / f"{hyp_slug}.md").write_text(
            f"---\nname: {hyp_slug}\ndescription: \"a latent link: X relates to Y\"\n"
            f"kind: hypothesis\nparents: [{parents[0]}, {parents[1]}]\n"
            f"confidence: 0.4\n---\nThe conjecture. [[{parents[0]}]] [[{parents[1]}]].\n")
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "one hypothesis tonight",
            "notes": [{"slug": hyp_slug, "action": "created",
                       "title": "latent link", "scope": "global"}]}))
        return subprocess.CompletedProcess(args=[], returncode=0)
    return _inner


def _run_global(monkeypatch, *, compute_pairs=None, invoke_claude=None,
                promote=None, **kw):
    return dream.run(
        ["--scope", "global"],
        compute_pairs=compute_pairs or (lambda ctx, corpus: []),
        invoke_claude=invoke_claude or _fake_dream_claude(),
        promote=promote or (lambda ctx, h, raw: None),
        rebuild_index=lambda ctx: 0,
        autocommit=lambda ctx, promoted: None,
        today_et=TARGET, **kw)


# ---- generation + quarantine ---------------------------------------------

def test_dream_generates_and_quarantines_hypothesis(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "today-note", last_used=TARGET.isoformat())   # today's experience
    _note(g, "old-note")                                   # an older memory
    out = _run_global(
        monkeypatch,
        compute_pairs=lambda ctx, corpus: [
            {"seed": "today-note", "older": "old-note", "cos": 0.45}])
    assert out.kind == "dreamed", out

    sub = config.subconscious_dir("global")
    hyp = sub / "latent-link.md"
    assert hyp.exists()                              # quarantined in subconscious…
    assert not (g / "latent-link.md").exists()       # …NOT in the soul corpus
    fm, _body = dream._split_fm(hyp.read_text())
    assert fm["kind"] == "hypothesis" and fm["status"] == "unverified"
    assert float(fm["stability"]) == dream.DREAM_S0 and int(fm["corroborations"]) == 0
    assert (sub / "digest" / f"{TARGET.isoformat()}.md").exists()   # morning digest


def test_dream_quiet_night_skips_skill(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "today-note", last_used=TARGET.isoformat())
    _note(g, "old-note")
    calls = {"n": 0}

    def claude(ctx, env, t):
        calls["n"] += 1
        return subprocess.CompletedProcess(args=[], returncode=0)

    out = _run_global(monkeypatch, compute_pairs=lambda c, corp: [],
                      invoke_claude=claude)
    assert out.kind == "dreamed" and calls["n"] == 0   # no pairs -> no skill call
    digest = (config.subconscious_dir("global") / "digest"
              / f"{TARGET.isoformat()}.md").read_text()
    assert "Quiet night" in digest


def test_partition_keeps_old_reactivated_notes_in_the_background_pool(tmp_path, monkeypatch):
    """Older-pool fix: the background is every note BORN before today, so a note that
    resurfaced today (last_used=today) but was created long ago stays an eligible
    recombination partner. Defining 'older' by birth date — not 'untouched today' —
    stops the pool collapsing to whatever happened not to surface on a busy day, when
    consolidation has stamped last_used=today across most of the corpus."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "fresh-today", first_seen=TARGET.isoformat())            # born today -> seed only
    _note(g, "old-reactivated", first_seen="2026-01-01",
          last_used=TARGET.isoformat())                              # old AND active today
    _note(g, "old-dormant", first_seen="2026-02-01")                 # old, untouched today
    seeds, older = dream._partition(dream._load_corpus(g), TARGET)
    assert set(seeds) == {"fresh-today", "old-reactivated"}          # today's experience
    assert set(older) == {"old-reactivated", "old-dormant"}          # born before today
    assert "old-reactivated" in seeds and "old-reactivated" in older  # overlap is intended
    assert "fresh-today" not in older                                 # born-today isn't background


# ---- the bleed membrane ---------------------------------------------------

def test_dream_promotes_when_corroborated(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "p-a", last_used=TARGET.isoformat())   # both parents re-activated…
    _note(g, "p-b", last_used=TARGET.isoformat())   # …together today
    sub = config.subconscious_dir("global")
    _hyp(sub, "hyp", parents=("p-a", "p-b"),
         corroborations=dream.DREAM_PROMOTE_N - 1)   # one short of promotion

    promoted = []
    out = _run_global(monkeypatch,
                      promote=lambda ctx, h, raw: (promoted.append(h["slug"]) or "insight-hyp"))
    assert out.kind == "dreamed" and promoted == ["hyp"]
    fm, _ = dream._split_fm((sub / "hyp.md").read_text())
    assert fm["status"] == "promoted"
    assert int(fm["corroborations"]) == dream.DREAM_PROMOTE_N


def test_dream_blessed_promotes_without_corroboration(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    _hyp(sub, "hyp", corroborations=0, blessed="true")   # operator endorsed it
    promoted = []
    out = _run_global(monkeypatch,
                      promote=lambda ctx, h, raw: (promoted.append(h["slug"]) or "insight-hyp"))
    assert out.kind == "dreamed" and promoted == ["hyp"]


def test_dream_bleed_is_rate_limited(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    for i in range(3):
        _hyp(sub, f"hyp{i}", blessed="true")   # three eligible, but cap is DREAM_BLEED_MAX
    promoted = []
    _run_global(monkeypatch,
                promote=lambda ctx, h, raw: (promoted.append(h["slug"]) or f"insight-{h['slug']}"))
    assert len(promoted) == dream.DREAM_BLEED_MAX   # never bleed more than the cap/night


def test_dream_decays_stale_uncorroborated(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    _hyp(sub, "old-hyp", first_seen="2026-01-01", corroborations=0)   # ~175d old
    out = _run_global(monkeypatch)
    assert out.kind == "dreamed"
    fm, _ = dream._split_fm((sub / "old-hyp.md").read_text())
    assert fm["status"] == "discarded"             # faded, unblessed, uncorroborated


def test_promote_to_soul_writes_valid_reversible_lesson(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    hyp = {"slug": "hyp", "fm": {"description": "a durable conjecture worth keeping"},
           "body": "The lesson, stated plainly.", "parents": ["p-a", "p-b"]}
    slug = dream._promote_to_soul(ctx, hyp, "")
    assert slug == "insight-hyp"
    text = (g / "insight-hyp.md").read_text()
    note = KnowledgeNote.parse(text, expect_slug="insight-hyp")
    assert note.kind == "lesson" and note.stability > 0    # earned, not permanent
    assert not dream.dynamics.is_permanent(note.stability)
    assert "[[p-a]]" in text and "[[p-b]]" in text         # linked + reversible


# ---- lifecycle ------------------------------------------------------------

def test_dream_idempotent(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "a", last_used=TARGET.isoformat())   # today's experience -> authoritative run
    _run_global(monkeypatch)
    out = _run_global(monkeypatch)
    assert out.kind == "skipped" and out.reason == "already_dreamed"


def test_premature_run_leaves_date_open_for_authoritative_nightly(tmp_path, monkeypatch):
    """The burned-night guard. A dream fired before the day's curate+consolidate have
    stamped today onto any note (no seeds, nothing for bleed to mature) must NOT consume
    the date — otherwise the authoritative post-consolidation nightly skips as
    already_dreamed and the soul never dreams (the 2026-06-25 regression)."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "old-note")                      # exists, but is not *today's* experience
    state = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET).state_file
    calls = {"n": 0}

    def claude(ctx, env, t):
        calls["n"] += 1
        return _fake_dream_claude(parents=("today-note", "old-note"))(ctx, env, t)

    # Premature run (no seeds): leaves the date open, never invokes the skill.
    out1 = _run_global(monkeypatch, compute_pairs=lambda c, corp: [], invoke_claude=claude)
    assert out1.kind == "skipped" and out1.reason == "no_experience_yet"
    assert calls["n"] == 0
    assert TARGET.isoformat() not in dream._done_dates(state)   # date NOT consumed

    # The day is then lived; the authoritative run recombines and consumes the date.
    _note(g, "today-note", last_used=TARGET.isoformat())
    out2 = _run_global(
        monkeypatch, invoke_claude=claude,
        compute_pairs=lambda c, corp: [
            {"seed": "today-note", "older": "old-note", "cos": 0.45}])
    assert out2.kind == "dreamed" and calls["n"] == 1          # skill finally invoked
    assert TARGET.isoformat() in dream._done_dates(state)      # now consumed


def test_dream_rejects_manifest_referencing_missing_hypothesis(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "today-note", last_used=TARGET.isoformat())
    _note(g, "old-note")

    def bad_claude(ctx, env, t):
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "claims a hypothesis it never wrote",
            "notes": [{"slug": "ghost", "action": "created", "title": "x",
                       "scope": "global"}]}))
        return subprocess.CompletedProcess(args=[], returncode=0)

    out = _run_global(
        monkeypatch, invoke_claude=bad_claude,
        compute_pairs=lambda ctx, corpus: [
            {"seed": "today-note", "older": "old-note", "cos": 0.45}])
    assert out.kind == "failed" and out.reason == "note_missing"
