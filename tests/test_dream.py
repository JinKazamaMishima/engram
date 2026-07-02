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
from pathlib import Path

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
