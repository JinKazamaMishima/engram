"""Tests for recall.dream — the dream pass + bleed membrane. The claude
subprocess, the embedder pairing, soul promotion and git are injected, so these
run with no model and no repo. Covers: hypothesis generation lands QUARANTINED in
the subconscious (never the corpus), the bleed valve (corroboration- and
blessing-gated, rate-limited promotion), decay of stale hypotheses, the digest,
and idempotency."""
from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

from recall import config, dream
from recall.schema import CurationManifest, KnowledgeNote

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


def _note_cf(corpus_dir, slug, *, surprise, body, first_seen="2026-01-01"):
    """A note stamped as today's experience, with an optional measured surprise —
    for exercising L1 counterfactual seed selection. ``surprise=None`` leaves it unset."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    fm = [f"name: {slug}", f'description: "{slug}: a thing"',
          f"first_seen: {first_seen}", f"last_used: {TARGET.isoformat()}"]
    if surprise is not None:
        fm.append(f"surprise: {surprise}")
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


def _cf_hyp(sub_dir, slug, *, parents=("some-episode",), status="unverified",
            blessed="false", first_seen="2026-06-10", body=None):
    """A quarantined kind:counterfactual note (single parent) for corroboration tests."""
    sub_dir.mkdir(parents=True, exist_ok=True)
    b = body or ("Real: we shipped X untested.\nCounterfactual: had we tested first, it "
                 "survives.\nLesson: measure before you enable.\nPredicts: the next "
                 "rollout ships measured-first.\n")
    plist = ", ".join(parents)
    (sub_dir / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: \"what-if: a causal guess\"\n"
        f"kind: counterfactual\nparents: [{plist}]\npivot: \"shipped X untested\"\n"
        f"status: {status}\nblessed: {blessed}\ncorroborations: 0\n"
        f"first_seen: {first_seen}\nstability: 1.0\n---\n{b}")


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


def _fake_cf_claude(slug="cf-shipped-untested", seed="shipped-untested"):
    """Stand-in /dream servicing a COUNTERFACTUAL worklist entry: asserts the wrapper
    handed it CF seeds, then writes one quarantined kind:counterfactual note (single
    parent) + a manifest — the §B contract, with no model."""
    def _inner(ctx, env, timeout):
        wl = json.loads(Path(env["RECALL_DREAM_WORKLIST"]).read_text())
        assert wl.get("counterfactuals"), "wrapper must pass CF seeds to the skill"
        ctx.subconscious_dir.mkdir(parents=True, exist_ok=True)
        (ctx.subconscious_dir / f"{slug}.md").write_text(
            f"---\nname: {slug}\ndescription: \"what-if: measure before you enable\"\n"
            f"kind: counterfactual\nparents: [{seed}]\npivot: \"shipped untested\"\n"
            f"confidence: 0.5\n---\nReal: shipped untested, it thrashed.\n"
            f"Counterfactual: had we tested first, prod survives.\n"
            f"Lesson: measure before you enable.\nPredicts: the next rollout ships "
            f"measured-first. [[{seed}]].\n")
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "one counterfactual tonight",
            "notes": [{"slug": slug, "action": "created", "title": "cf what-if",
                       "scope": "global"}]}))
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


# ---- L1: counterfactual seed selection ------------------------------------

def test_counterfactual_seeds_gate_charge_and_forkability(tmp_path, monkeypatch):
    """From today's experience, keep episodes that (a) name a decision/outcome
    (forkable) and (b) aren't a measured-dull memory; rank by charge. Unmeasured
    surprise (-1) falls through the gate; a measured low surprise is filtered; a
    static fact with no decision is filtered even at high surprise."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note_cf(g, "shipped-untested", surprise=0.8,
             body="We decided to ship the cache to prod before load-testing it; it turned out to thrash.")
    _note_cf(g, "chose-postgres", surprise=None,   # unmeasured -> falls through, ranks below measured
             body="We chose Postgres instead of the old store because it fails closed.")
    _note_cf(g, "routine-tweak", surprise=0.05,   # forkable but measured-dull -> filtered
             body="Decided to bump a timeout; nothing notable happened.")
    _note_cf(g, "static-fact", surprise=0.9,       # high charge but no decision/outcome -> filtered
             body="Recall uses Qwen3 embeddings with 1024 dimensions.")
    corpus = dream._load_corpus(g)
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    slugs = [s["seed"] for s in dream._counterfactual_seeds(ctx, corpus)]
    assert slugs == ["shipped-untested", "chose-postgres"]   # forkable+charged, ranked by charge
    assert "routine-tweak" not in slugs                       # measured-dull filtered
    assert "static-fact" not in slugs                         # unforkable filtered


def test_counterfactual_seeds_capped(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(dream, "CF_MAX_SEEDS", 2)
    g = config.global_corpus_dir()
    for i in range(4):
        _note_cf(g, f"decision-{i}", surprise=0.5 + i / 100,
                 body=f"We decided thing {i}; it turned out to matter.")
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    seeds = dream._counterfactual_seeds(ctx, dream._load_corpus(g))
    assert len(seeds) == 2                                     # never exceed the cap
    assert [s["seed"] for s in seeds] == ["decision-3", "decision-2"]   # top charge first


def test_counterfactual_enabled_writes_quarantined_note(tmp_path, monkeypatch):
    """With --counterfactual, the wrapper computes CF seeds, puts them in the worklist,
    invokes the skill, and the kind:counterfactual note lands in the subconscious (single
    parent) — never the corpus — with lifecycle fields stamped."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note_cf(g, "shipped-untested", surprise=0.8,
             body="We decided to ship the cache to prod before load-testing it; it turned out to thrash.")
    out = dream.run(
        ["--scope", "global", "--counterfactual"],
        compute_pairs=lambda ctx, corpus: [],                    # blend quiet tonight
        compute_cf_seeds=lambda ctx, corpus: [{"seed": "shipped-untested", "charge": 0.8}],
        invoke_claude=_fake_cf_claude(),
        promote=lambda ctx, h, raw: None,
        rebuild_index=lambda ctx: 0, autocommit=lambda ctx, p: None,
        today_et=TARGET)
    assert out.kind == "dreamed", out
    sub = config.subconscious_dir("global")
    note = sub / "cf-shipped-untested.md"
    assert note.exists()                                          # quarantined…
    assert not (g / "cf-shipped-untested.md").exists()           # …never the soul
    fm, _ = dream._split_fm(note.read_text())
    assert fm["kind"] == "counterfactual"
    assert fm["parents"] == ["shipped-untested"]                 # single parent
    assert fm["status"] == "unverified"                           # wrapper stamped lifecycle


def test_counterfactual_off_by_default(tmp_path, monkeypatch):
    """Without --counterfactual the operator is inert: CF seeds are never computed and,
    with no blend pairs, the skill is never invoked — the live nightly path is unchanged."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note_cf(g, "shipped-untested", surprise=0.8,
             body="We decided to ship it; it turned out to matter.")
    calls = {"cf": 0}

    def cf(ctx, corpus):
        calls["cf"] += 1
        return [{"seed": "shipped-untested", "charge": 0.8}]

    out = _run_global(monkeypatch, compute_pairs=lambda c, corp: [], compute_cf_seeds=cf)
    assert out.kind == "dreamed" and calls["cf"] == 0             # CF not computed without the flag


# ---- L1: counterfactual corroboration (prospective match) -----------------

def _cf_ctx():
    return dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)


def test_open_counterfactuals_filters(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    _cf_hyp(sub, "cf-open")                              # open -> kept
    _cf_hyp(sub, "cf-promoted", status="promoted")      # terminal -> skip
    _cf_hyp(sub, "cf-blessed", blessed="true")          # manual fast-path -> skip
    _hyp(sub, "plain-hyp")                              # kind:hypothesis, not a what-if -> skip
    slugs = sorted(str(fm.get("name")) for fm, _b, _p in dream._open_counterfactuals(sub))
    assert slugs == ["cf-open"]


def test_apply_cf_verdicts_confirm_promotes(tmp_path, monkeypatch):
    """One clean confirm graduates the what-if — single parent, stamped corroborated_by."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "new-episode", last_used=TARGET.isoformat())   # today's arrival = the evidence
    sub = config.subconscious_dir("global")
    _cf_hyp(sub, "cf-x", parents=("some-parent",))
    promoted = []
    summary = dream.apply_cf_verdicts(
        _cf_ctx(), dream._load_corpus(g),
        [{"cf": "cf-x", "verdict": "confirm", "evidence": "new-episode", "why": "recurred"}],
        promote=lambda ctx, h, raw: (promoted.append(h["slug"]) or "insight-cf-x"),
        cap_used=0)
    assert summary["promoted"] == ["insight-cf-x"] and promoted == ["cf-x"]
    fm, _ = dream._split_fm((sub / "cf-x.md").read_text())
    assert fm["status"] == "promoted" and fm["corroborated_by"] == "new-episode"


def test_apply_cf_verdicts_refute_retires(tmp_path, monkeypatch):
    """A refuting match kills the what-if now — falsifiable, faster than TTL decay."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "new-episode", last_used=TARGET.isoformat())
    sub = config.subconscious_dir("global")
    _cf_hyp(sub, "cf-x")
    promoted = []
    summary = dream.apply_cf_verdicts(
        _cf_ctx(), dream._load_corpus(g),
        [{"cf": "cf-x", "verdict": "refute", "evidence": "new-episode", "why": "prediction failed"}],
        promote=lambda ctx, h, raw: promoted.append(h["slug"]), cap_used=0)
    assert summary["retired"] == 1 and summary["promoted"] == [] and promoted == []
    fm, _ = dream._split_fm((sub / "cf-x.md").read_text())
    assert fm["status"] == "discarded" and fm["refuted_by"] == "new-episode"


def test_apply_cf_verdicts_rejects_evidence_not_from_today(tmp_path, monkeypatch):
    """We never take the skill's word that reality moved — cited evidence must be one of
    today's episodes, else the ruling is ignored and the what-if stays open."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "seed-today", last_used=TARGET.isoformat())    # a today note, but not the cited one
    sub = config.subconscious_dir("global")
    _cf_hyp(sub, "cf-x")
    promoted = []
    summary = dream.apply_cf_verdicts(
        _cf_ctx(), dream._load_corpus(g),
        [{"cf": "cf-x", "verdict": "confirm", "evidence": "ghost", "why": "..."}],
        promote=lambda ctx, h, raw: (promoted.append(h["slug"]) or "x"), cap_used=0)
    assert summary["promoted"] == [] and promoted == []
    fm, _ = dream._split_fm((sub / "cf-x.md").read_text())
    assert str(fm.get("status")) == "unverified"            # left open


def test_apply_cf_verdicts_respects_bleed_cap(tmp_path, monkeypatch):
    """Confirmations share tonight's DREAM_BLEED_MAX — a cap already spent by bleed() blocks
    the CF promotion, and it stays open for tomorrow rather than being lost."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "new-episode", last_used=TARGET.isoformat())
    sub = config.subconscious_dir("global")
    _cf_hyp(sub, "cf-x")
    promoted = []
    summary = dream.apply_cf_verdicts(
        _cf_ctx(), dream._load_corpus(g),
        [{"cf": "cf-x", "verdict": "confirm", "evidence": "new-episode"}],
        promote=lambda ctx, h, raw: (promoted.append(h["slug"]) or "x"),
        cap_used=dream.DREAM_BLEED_MAX)
    assert summary["promoted"] == [] and promoted == []
    fm, _ = dream._split_fm((sub / "cf-x.md").read_text())
    assert str(fm.get("status")) == "unverified"


def test_run_counterfactual_corroboration_confirms(tmp_path, monkeypatch):
    """End-to-end with --counterfactual: the harness funnels an open what-if into the worklist,
    the skill writes a confirm verdict, and the harness promotes it on the single clean match."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "new-episode", last_used=TARGET.isoformat())   # today's arrival + seeds_present
    sub = config.subconscious_dir("global")
    _cf_hyp(sub, "cf-x", parents=("some-parent",))

    def cf_corrob(ctx, corpus):
        return [{"cf": {"slug": "cf-x", "description": "what-if", "pivot": "shipped X",
                        "predicts": "the next rollout ships measured-first",
                        "parents": ["some-parent"], "body": "..."},
                 "candidates": [{"slug": "new-episode", "description": "d", "body": "b"}]}]

    def claude(ctx, env, t):
        Path(env["RECALL_DREAM_VERDICTS"]).write_text(json.dumps(
            [{"cf": "cf-x", "verdict": "confirm", "evidence": "new-episode", "why": "recurred"}]))
        ctx.manifest_path.write_text(json.dumps({
            "schema_version": 1, "date": ctx.target.isoformat(),
            "summary": "corroborated one what-if", "notes": []}))
        return subprocess.CompletedProcess(args=[], returncode=0)

    promoted = []
    out = dream.run(
        ["--scope", "global", "--counterfactual"],
        compute_pairs=lambda c, corp: [], compute_cf_seeds=lambda c, corp: [],
        compute_cf_corrob=cf_corrob, invoke_claude=claude,
        promote=lambda ctx, h, raw: (promoted.append(h["slug"]) or "insight-cf-x"),
        rebuild_index=lambda ctx: 0, autocommit=lambda ctx, p: None, today_et=TARGET)
    assert out.kind == "dreamed" and promoted == ["cf-x"]
    fm, _ = dream._split_fm((sub / "cf-x.md").read_text())
    assert fm["status"] == "promoted"


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


# ---- Palate m1: taste scoring + durable trace -----------------------------

def _palate_taste(**over):
    e = {"slug": "latent-link", "taste": 0.72, "pursue": "novelty",
         "axes": [{"lens": "novelty", "weight": 0.9, "score": 0.8},
                  {"lens": "mission_fit", "weight": 0.5, "score": 0.4}],
         "why": "a genuinely non-obvious transfer"}
    e.update(over)
    return e


def _manifest_of(*slugs):
    return CurationManifest.from_dict(
        {"schema_version": 1, "date": TARGET.isoformat(), "summary": "s",
         "notes": [{"slug": s, "action": "created", "title": "t", "scope": "global"}
                   for s in slugs]})


def test_palate_read_taste_tolerant(tmp_path):
    """_read_taste is best-effort: missing/garbled -> [], and non-dict entries are dropped."""
    assert dream._read_taste(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert dream._read_taste(bad) == []
    good = tmp_path / "t.json"
    good.write_text(json.dumps([_palate_taste(), "junk", 3]))
    assert [t["slug"] for t in dream._read_taste(good)] == ["latent-link"]


def test_palate_apply_stamps_frontmatter_and_traces(tmp_path, monkeypatch):
    """apply_taste stamps the two scalars (taste, pursue) into the quarantined note and
    appends the full per-lens record to the durable palate trace under the data root."""
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    _hyp(sub, "latent-link")
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    n = dream.apply_taste(ctx, _manifest_of("latent-link"), [_palate_taste()])
    assert n == 1
    fm, _ = dream._split_fm((sub / "latent-link.md").read_text())
    assert float(fm["taste"]) == 0.72 and fm["pursue"] == "novelty"
    lines = ctx.palate_trace.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["slug"] == "latent-link" and rec["taste"] == 0.72
    assert rec["parents"] == ["p-a", "p-b"] and len(rec["axes"]) == 2


def test_palate_clamps_and_only_scores_notes_written_this_run(tmp_path, monkeypatch):
    """Out-of-range taste is clamped to [0,1]; an entry naming a slug the manifest never
    listed is ignored entirely (no stamp, no trace) — taste scores only what was written."""
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    _hyp(sub, "latent-link")
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    # not in the manifest -> ignored
    assert dream.apply_taste(ctx, _manifest_of("other"), [_palate_taste()]) == 0
    fm, _ = dream._split_fm((sub / "latent-link.md").read_text())
    assert "taste" not in fm and not ctx.palate_trace.exists()
    # in the manifest but taste out of range -> clamped
    assert dream.apply_taste(ctx, _manifest_of("latent-link"),
                             [_palate_taste(taste=9.9)]) == 1
    fm, _ = dream._split_fm((sub / "latent-link.md").read_text())
    assert float(fm["taste"]) == 1.0


def _fake_dream_claude_with_taste(hyp_slug="latent-link", parents=("today-note", "old-note")):
    """Stand-in /dream that writes the §A hypothesis + manifest AND the §E taste file."""
    base = _fake_dream_claude(hyp_slug, parents)

    def _inner(ctx, env, timeout):
        cp = base(ctx, env, timeout)
        Path(env["RECALL_DREAM_TASTE"]).write_text(json.dumps([
            {"slug": hyp_slug, "taste": 0.66, "pursue": "elegance",
             "axes": [{"lens": "elegance", "weight": 0.8, "score": 0.7}],
             "why": "clean shared abstraction"}]))
        return cp
    return _inner


def test_palate_end_to_end_persists_from_skill_output(tmp_path, monkeypatch):
    """Full run: the skill writes RECALL_DREAM_TASTE, the wrapper reads it after stamping
    lifecycle defaults, and both the frontmatter scalars and the trace line land."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _note(g, "today-note", last_used=TARGET.isoformat())
    _note(g, "old-note")
    out = _run_global(
        monkeypatch, invoke_claude=_fake_dream_claude_with_taste(),
        compute_pairs=lambda ctx, corpus: [
            {"seed": "today-note", "older": "old-note", "cos": 0.45}])
    assert out.kind == "dreamed"
    sub = config.subconscious_dir("global")
    fm, _ = dream._split_fm((sub / "latent-link.md").read_text())
    assert float(fm["taste"]) == 0.66 and fm["pursue"] == "elegance"
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    assert json.loads(ctx.palate_trace.read_text().splitlines()[0])["slug"] == "latent-link"


# ---- Palate m2: taste-scaled quarantine lifetime --------------------------

def _hyp_tasted(sub_dir, slug, *, taste, first_seen, corroborations=0, blessed="false"):
    """A quarantined hypothesis carrying a Palate m1 taste score (for m2 decay tests)."""
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: \"a conjecture: x relates to y\"\n"
        f"kind: hypothesis\nparents: [p-a, p-b]\nstatus: unverified\n"
        f"first_seen: {first_seen}\ncorroborations: {corroborations}\nblessed: {blessed}\n"
        f"stability: 1.0\ntaste: {taste}\n---\nThe conjecture body.\n")


def test_ttl_for_taste_scales_symmetrically():
    """Unmeasured taste keeps the base lifetime; a measured taste stretches or shrinks it."""
    base = dream.DREAM_TTL_DAYS
    assert dream._ttl_for_taste(None) == base                      # unmeasured -> unchanged
    assert dream._ttl_for_taste(0.0) == round(base * dream.DREAM_TASTE_TTL_LO)
    assert dream._ttl_for_taste(1.0) == round(base * dream.DREAM_TASTE_TTL_HI)
    assert dream._ttl_for_taste(0.0) < base < dream._ttl_for_taste(1.0)
    assert dream._as_taste("0.7") == 0.7 and dream._as_taste(None) is None   # parse + sentinel


def test_high_taste_survives_past_base_ttl(tmp_path, monkeypatch):
    """A conjecture aged BEYOND the base 30-day TTL is NOT retired when its taste is high —
    the palate bought it a longer quarantine to keep earning corroboration."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    sub = config.subconscious_dir("global")
    old = (TARGET - timedelta(days=40)).isoformat()          # past base 30…
    _hyp_tasted(sub, "loved", taste=0.9, first_seen=old)     # …but taste 0.9 -> ttl ~56d
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    summary = dream.bleed(ctx, dream._load_corpus(g), promote=lambda c, h, r: None)
    assert summary["retired"] == 0
    fm, _ = dream._split_fm((sub / "loved.md").read_text())
    assert str(fm.get("status")) == "unverified"             # survived — not yet at its TTL


def test_low_taste_retires_early_but_unmeasured_keeps_base(tmp_path, monkeypatch):
    """A poorly-rated conjecture clears out before the base TTL; an UNMEASURED one still gets
    the full base lifetime — no regression for pre-palate hypotheses."""
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    sub = config.subconscious_dir("global")
    age20 = (TARGET - timedelta(days=20)).isoformat()
    _hyp_tasted(sub, "meh", taste=0.0, first_seen=age20)     # taste 0 -> ttl 15 -> 20>=15 retire
    _hyp(sub, "unscored", first_seen=age20)                  # no taste -> base 30 -> survives
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    summary = dream.bleed(ctx, dream._load_corpus(g), promote=lambda c, h, r: None)
    assert summary["retired"] == 1
    assert str(dream._split_fm((sub / "meh.md").read_text())[0].get("status")) == "discarded"
    assert str(dream._split_fm((sub / "unscored.md").read_text())[0].get("status")) == "unverified"


# ---- Palate m3: the cross-night chase -------------------------------------

def _pursuit_taste(slug, taste, pursue="novelty"):
    return {"slug": slug, "taste": taste, "pursue": pursue,
            "axes": [{"lens": pursue, "weight": 0.9, "score": taste}], "why": "w"}


def test_record_pursuits_enrolls_only_top_taste(tmp_path, monkeypatch):
    """After taste, only conjectures at/above the chase bar (MIN) enrol as open pursuits."""
    _setup(tmp_path, monkeypatch)
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    entries = [_pursuit_taste("hot", 0.9), _pursuit_taste("warm", 0.7),
               _pursuit_taste("cold", 0.2)]                # cold < MIN -> excluded
    dream._record_pursuits(ctx, entries, _manifest_of("hot", "warm", "cold"))
    got = {p["slug"]: p for p in dream._read_pursuits(ctx.pursuits_path)}
    assert set(got) == {"hot", "warm"}
    assert got["hot"]["chased"] == 0 and got["hot"]["born"] == TARGET.isoformat()


def test_record_pursuits_caps_and_dedups(tmp_path, monkeypatch):
    """Keep the highest-taste up to MAX, and never enrol the same conjecture twice."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(dream, "DREAM_PURSUE_MAX", 2)
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    entries = [_pursuit_taste(f"h{i}", 0.70 + i / 100) for i in range(4)]
    dream._record_pursuits(ctx, entries, _manifest_of("h0", "h1", "h2", "h3"))
    assert [p["slug"] for p in dream._read_pursuits(ctx.pursuits_path)] == ["h3", "h2"]
    dream._record_pursuits(ctx, [_pursuit_taste("h3", 0.99)], _manifest_of("h3"))
    assert [p["slug"] for p in dream._read_pursuits(ctx.pursuits_path)].count("h3") == 1


def test_plan_pursuits_prunes_ages_and_cards(tmp_path, monkeypatch):
    """Before the dream: drop graduated/discarded/cooled/missing pursuits; keep the live ones,
    bump their chase count, and hand the skill a card to develop."""
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    _hyp(sub, "live", status="unverified")
    _hyp(sub, "graduated", status="promoted")
    _hyp(sub, "cooled", status="unverified")
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    dream._write_pursuits(ctx.pursuits_path, [
        {"slug": "live", "pursue": "novelty", "taste": 0.9, "born": "2026-06-01", "chased": 0},
        {"slug": "graduated", "pursue": "x", "taste": 0.8, "born": "2026-06-01", "chased": 0},
        {"slug": "cooled", "pursue": "x", "taste": 0.8, "born": "2026-06-01",
         "chased": dream.DREAM_PURSUE_TTL},
        {"slug": "gone", "pursue": "x", "taste": 0.8, "born": "2026-06-01", "chased": 0}])
    cards, nxt = dream._plan_pursuits(ctx)
    assert [c["slug"] for c in cards] == ["live"]          # only the live thread is chased
    assert cards[0]["pursue"] == "novelty" and cards[0]["parents"] == ["p-a", "p-b"]
    assert [p["slug"] for p in nxt] == ["live"] and nxt[0]["chased"] == 1   # aged one night


def test_pursuit_roundtrip_record_then_plan(tmp_path, monkeypatch):
    """A high-taste conjecture recorded one night surfaces as a chase card the next."""
    _setup(tmp_path, monkeypatch)
    sub = config.subconscious_dir("global")
    _hyp(sub, "spark", status="unverified")
    ctx = dream._resolve(dream._parse_args(["--scope", "global"]), TARGET)
    dream._record_pursuits(ctx, [_pursuit_taste("spark", 0.9, pursue="elegance")],
                           _manifest_of("spark"))
    cards, nxt = dream._plan_pursuits(ctx)
    assert [c["slug"] for c in cards] == ["spark"] and cards[0]["pursue"] == "elegance"
    assert nxt[0]["chased"] == 1
