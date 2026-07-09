#!/usr/bin/env python3
"""The dream pass — nightly offline recombination + the bleed membrane.

Runs after curate + consolidate. Two phases, because sleep does two opposite jobs
(see docs/dynamic-memory.md):

  REM / recombine  — take the day's *experience* (notes activated or created
    today) and pair each with an OLDER note at MEDIUM semantic distance (the
    cosine band where creativity lives — near enough to relate, far enough to be
    non-obvious), then ask the dream skill for a latent hypothesis linking them.
    Output is a typed `kind: hypothesis` note in the QUARANTINED subconscious
    store — never the corpus, never the live index. Most are noise and fade.

  Bleed membrane  — the valve, not a wall. Each night, existing hypotheses are
    checked for corroboration (their two parent memories were independently
    re-activated together) or operator blessing; a hypothesis that earns it is
    PROMOTED into the soul — rate-limited, affect-stripped (the durable lesson,
    not the drama), reversible, and ACCUMULATING alongside real notes (never
    replacing them — the guardrail against self-reinforcing collapse). Stale
    uncorroborated hypotheses decay and are retired. A morning digest surfaces the
    night's recombinations to the operator, whose blessing is itself corroboration.

Deterministic harness (selection, pairing, promotion, digest) + an LLM skill for
the hypothesis text. The subprocess + the embedder + git are injectable so tests
run with no model and no repo. Idempotent per (scope, date).

Usage:
    recall dream --scope global
    recall dream --scope project --project-dir /x/foo --commit
    recall dream --scope global --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from recall import config, dynamics
from recall.curate import (
    CLAUDE_BIN,
    CLAUDE_TIMEOUT_S,
    ET,
    Outcome,
    _et_clock,
    _git_commit_scoped,
    run_claude_with_backoff,
    validate_manifest_against,
)
from recall.notify import notify_alert
from recall.schema import CurationManifest, CurationSchemaError, KnowledgeNote, set_frontmatter_keys

_DREAM_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit"]
# Dreaming is the deep creative pass (counterfactual / blind-variation-selective-
# retention over 0.30-0.60 cosine note pairs) — pin it to Opus 4.8 at the 1M
# window at xhigh effort. (Bump RECALL_DREAM_EFFORT to max if the counterfactuals
# want more depth; xhigh stays clear of the 900s per-call timeout on a heavy
# night.) Model + effort are env-overridable.
DREAM_MODEL = os.environ.get("RECALL_DREAM_MODEL", "claude-opus-4-8[1m]")
DREAM_EFFORT = os.environ.get("RECALL_DREAM_EFFORT", "xhigh")

# Recombination band: pair a fresh memory with an older one whose cosine sits in
# [LO, HI] — wider/further than reconsolidate's 0.60–0.80 link band, because a
# dream should reach for the non-obvious. Bounded count so a night is cheap.
DREAM_LO = float(os.environ.get("RECALL_DREAM_LO", "0.30"))
DREAM_HI = float(os.environ.get("RECALL_DREAM_HI", "0.60"))
DREAM_MAX_PAIRS = int(os.environ.get("RECALL_DREAM_MAX_PAIRS", "6"))
# Bleed membrane.
DREAM_PROMOTE_N = int(os.environ.get("RECALL_DREAM_PROMOTE_N", "2"))   # corroborations to promote
DREAM_BLEED_MAX = int(os.environ.get("RECALL_DREAM_BLEED_MAX", "1"))   # max promotions/night/scope
DREAM_TTL_DAYS = int(os.environ.get("RECALL_DREAM_TTL_DAYS", "30"))    # uncorroborated lifetime
DREAM_S0 = float(os.environ.get("RECALL_DREAM_S0", "1.0"))            # hypotheses are born fragile
# Counterfactual (L1) seeds — a single charged, forkable episode from today's
# experience (not a pair). "Forkable" = the note names a decision / cause / outcome
# to intervene on; "charged" = surprising enough to be worth re-processing. A KNOWN-low
# surprise is a measured-dull memory and is skipped; an UNSET surprise (-1) falls THROUGH
# the gate (unmeasured ≠ boring — see KnowledgeNote.surprise). The pivot itself is chosen
# downstream by the skill; this only decides WHICH episodes deserve a 'what-if'.
CF_CHARGE_MIN = float(os.environ.get("RECALL_DREAM_CF_CHARGE_MIN", "0.30"))
CF_MAX_SEEDS = int(os.environ.get("RECALL_DREAM_CF_MAX_SEEDS", "3"))
# Counterfactual corroboration (the "bleed" for what-ifs). We do NOT scan the corpus: each
# night we match today's episodes against the small set of OPEN counterfactuals (capped per
# night, gone in TTL days), so cost is bounded by open-bets × today — never by corpus size.
# A coarse cosine funnel (CF_CORROB_LO) narrows which of today's episodes are even in a
# what-if's neighbourhood; the skill (stage 2) makes the confirm/refute call. The bar is ONE
# clean match — a causal rule confirmed once outranks an association seen twice — and because
# the note's `Predicts:` line is falsifiable, a refuting match RETIRES it (faster than TTL).
# Confirmations share tonight's DREAM_BLEED_MAX cap.
CF_CORROB_LO = float(os.environ.get("RECALL_DREAM_CF_CORROB_LO", "0.45"))
CF_CORROB_MAX_CAND = int(os.environ.get("RECALL_DREAM_CF_CORROB_MAX_CAND", "3"))


@dataclass(frozen=True)
class DreamContext:
    scope: str
    label: str
    corpus_dir: Path
    index_path: Path
    repo: Path
    target: date
    subconscious_dir: Path
    worklist_path: Path
    manifest_path: Path
    verdicts_path: Path
    digest_path: Path
    session_log_path: Path
    state_file: Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="recall dream",
                                description=__doc__.splitlines()[0])
    p.add_argument("--scope", choices=("project", "global"), default="global")
    p.add_argument("--project-dir", type=Path, default=None)
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--force", action="store_true",
                   help="re-dream even if this date already ran")
    p.add_argument("--commit", action="store_true",
                   help="on success, scoped-commit promoted soul notes ([dream])")
    p.add_argument("--dry-run", action="store_true",
                   help="compute seeds+pairs+promotions; do not invoke the skill")
    p.add_argument("--counterfactual", action="store_true",
                   help="also run the L1 counterfactual operator on today's charged, "
                        "forkable seeds (opt-in; blend recombination runs regardless)")
    return p.parse_args(argv)


def _resolve(args: argparse.Namespace, target: date) -> DreamContext:
    if args.scope == "global":
        label = config.GLOBAL_SCOPE
        corpus_dir = config.global_corpus_dir()
        repo = corpus_dir
    else:
        project_dir = (Path(args.project_dir).resolve() if args.project_dir
                       else Path.cwd())
        label = config.project_slug(project_dir)
        corpus_dir = config.project_corpus_dir(project_dir)
        repo = project_dir
    ddir = config.data_root() / "dream" / label
    return DreamContext(
        scope=args.scope, label=label, corpus_dir=corpus_dir,
        index_path=config.index_path(label), repo=repo, target=target,
        subconscious_dir=config.subconscious_dir(label),
        worklist_path=ddir / "worklists" / f"{target.isoformat()}.json",
        manifest_path=ddir / "manifests" / f"{target.isoformat()}.json",
        verdicts_path=ddir / "verdicts" / f"{target.isoformat()}.json",
        digest_path=config.subconscious_dir(label) / "digest" / f"{target.isoformat()}.md",
        session_log_path=ddir / "sessions" / f"{target.isoformat()}.md",
        state_file=ddir / "done.json")


# ---- idempotency ---------------------------------------------------------

def _done_dates(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        return set(json.loads(state_file.read_text()).get("dates", []))
    except (json.JSONDecodeError, OSError):
        return set()


def _mark_done(state_file: Path, d: date) -> None:
    dates = _done_dates(state_file)
    dates.add(d.isoformat())
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"dates": sorted(dates)}, indent=2))


# ---- corpus + subconscious helpers ---------------------------------------

def _load_corpus(corpus_dir: Path) -> dict[str, tuple[KnowledgeNote, Path]]:
    out: dict[str, tuple[KnowledgeNote, Path]] = {}
    for path in sorted(Path(corpus_dir).glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            note = KnowledgeNote.parse(path.read_text(), expect_slug=path.stem)
        except CurationSchemaError:
            continue
        out[note.slug] = (note, path)
    return out


def _is_today(note: KnowledgeNote, target: date) -> bool:
    """Was this note part of *today's experience* — activated or created today?"""
    iso = target.isoformat()
    return note.last_used[:10] == iso or note.first_seen[:10] == iso


def _load_hypotheses(subconscious_dir: Path) -> list[tuple[dict, str, Path]]:
    """(frontmatter, body, path) for every parseable subconscious hypothesis."""
    out: list[tuple[dict, str, Path]] = []
    if not subconscious_dir.is_dir():
        return out
    for path in sorted(subconscious_dir.glob("*.md")):
        try:
            fm, body = _split_fm(path.read_text())
        except CurationSchemaError:
            continue
        out.append((fm, body, path))
    return out


def _split_fm(text: str) -> tuple[dict, str]:
    from recall.schema import _split_frontmatter
    return _split_frontmatter(text)


# ---- REM: recombination pairs (model) ------------------------------------

def _partition(corpus: dict[str, tuple[KnowledgeNote, Path]],
               target: date) -> tuple[list[str], list[str]]:
    """Split the corpus into the dream's REM seeds and its older background pool.

    seeds — today's *experience*: notes created or re-activated today (``_is_today``).
    older — the established background: every note BORN before today (``first_seen``
        predates the target), whether or not it was *also* reinforced today.

    Defining "older" by birth date — not by "untouched today" — is what keeps the
    background from collapsing. On a busy day consolidation stamps ``last_used = today``
    across most of the corpus, so an "untouched today" pool shrinks to the handful of
    notes that happened not to surface, and every dream reaches into the same tiny,
    arbitrary set. An old memory that resurfaced today is still an old memory: it stays
    eligible as a partner (overlap with seeds is intended — self-pairs are skipped and
    reversed duplicates collapse in the dedup below)."""
    iso = target.isoformat()
    seeds = [s for s, (n, _p) in corpus.items() if _is_today(n, target)]
    older = [s for s, (n, _p) in corpus.items() if n.first_seen[:10] < iso]
    return seeds, older


def _recombination_pairs(ctx: DreamContext,
                         corpus: dict[str, tuple[KnowledgeNote, Path]]) -> list[dict]:
    """Pair each of today's memories with an OLDER note at medium cosine distance
    [LO, HI]. Lazy torch AFTER the trivial guard; best-effort -> [] (no model / a
    tiny corpus just means no dreams tonight). Deterministic given the corpus +
    date (a date-seeded shuffle gives variety without breaking tests)."""
    slugs = list(corpus.keys())
    seeds, older = _partition(corpus, ctx.target)
    if not seeds or not older:
        return []
    try:
        import random

        import numpy as np

        from recall.index import best_embedder
        emb = best_embedder(alert_degraded=True)
        notes = [corpus[s][0] for s in slugs]
        vecs = np.asarray(emb.embed([f"{n.description}\n\n{n.body}"
                                     for n in notes]), dtype="float32")
        pos = {s: i for i, s in enumerate(slugs)}
        cands: list[dict] = []
        for s in seeds:
            i = pos[s]
            for o in older:
                if o == s:
                    continue
                cos = float(vecs[i] @ vecs[pos[o]])
                if DREAM_LO <= cos <= DREAM_HI:
                    cands.append({"seed": s, "older": o, "cos": round(cos, 3)})
    except Exception as e:  # noqa: BLE001 — dreams are optional; never fail the run
        print(f"[dream] WARN pair computation failed: {e}", flush=True)
        return []
    # Mid-band first (most evocative), then a date-seeded shuffle for variety.
    mid = (DREAM_LO + DREAM_HI) / 2
    cands.sort(key=lambda c: abs(c["cos"] - mid))
    random.Random(ctx.target.toordinal()).shuffle(cands)
    seen: set[frozenset] = set()
    picked: list[dict] = []
    for c in cands:
        key = frozenset((c["seed"], c["older"]))
        if key in seen:
            continue
        seen.add(key)
        picked.append(c)
        if len(picked) >= DREAM_MAX_PAIRS:
            break
    return picked


# ---- L1: counterfactual seeds (model-free selection) ---------------------

# A note is "forkable" if its prose names a decision, a cause, or an outcome —
# something a single ``do()`` can intervene on. Lexical prefilter ONLY: the skill
# makes the final call (and yields nothing for a seed that turns out unforkable).
# Over-including here is cheap (the skill drops it); under-including loses a dream.
_FORK_MARKERS = re.compile(
    r"\b(decid\w*|chos\w*|choos\w*|opt(?:ed|ing)?|instead|rather than|because|"
    r"so we|turned out|ended up|should(?:n't| not)? have|could have|would have|"
    r"regret\w*|mistake|failed?|fix(?:ed|es)?|broke|avoid\w*|prevent\w*)\b", re.I)


def _is_forkable(note: KnowledgeNote) -> bool:
    return bool(_FORK_MARKERS.search(f"{note.description}\n{note.body}"))


def _charge(note: KnowledgeNote) -> float | None:
    """The seed's affective charge = its encoding surprise, or None when unmeasured
    (``surprise == -1``). None falls THROUGH the charge gate (unknown ≠ boring); a real
    low σ is a measured-dull note and is filtered out."""
    return note.surprise if note.surprise >= 0.0 else None


def _counterfactual_seeds(ctx: DreamContext,
                          corpus: dict[str, tuple[KnowledgeNote, Path]]) -> list[dict]:
    """Pick today's charged, forkable episodes to run the L1 counterfactual on — the
    single-note analog of ``_recombination_pairs``. Gate: the note is part of today's
    experience (``_partition`` seeds), NAMES a decision/cause/outcome (``_is_forkable``),
    and is not a measured-dull memory (``_charge`` ≥ CF_CHARGE_MIN when measured). Rank by
    charge (most worth re-processing first), deterministic, capped at CF_MAX_SEEDS. No
    model, no torch — pure selection, so it is hermetic and unit-testable."""
    seeds, _older = _partition(corpus, ctx.target)
    scored: list[dict] = []
    for s in seeds:
        note = corpus[s][0]
        if not _is_forkable(note):
            continue
        charge = _charge(note)
        if charge is not None and charge < CF_CHARGE_MIN:
            continue
        scored.append({"seed": s,
                       "charge": round(charge, 3) if charge is not None else None})
    # Most charged first; an unmeasured (None) charge sinks below any measured one but
    # stays eligible. Slug tiebreak keeps it deterministic (tests + idempotent re-runs).
    scored.sort(key=lambda c: (-(c["charge"] if c["charge"] is not None else -1.0),
                               c["seed"]))
    return scored[:CF_MAX_SEEDS]


# ---- L1: counterfactual corroboration — stage 1, the coarse funnel -------

def _open_counterfactuals(subconscious_dir: Path) -> list[tuple[dict, str, Path]]:
    """The open bets — kind:counterfactual notes still unverified (not promoted, not
    discarded) and not blessed. Blessed ones ride the manual fast-path in ``bleed``; these
    await reality's verdict via prospective match."""
    out: list[tuple[dict, str, Path]] = []
    for fm, body, path in _load_hypotheses(subconscious_dir):
        if str(fm.get("kind") or "").strip().lower() != "counterfactual":
            continue
        if str(fm.get("status") or "unverified").strip() in ("promoted", "discarded"):
            continue
        if _as_bool(fm.get("blessed")):
            continue
        out.append((fm, body, path))
    return out


def _cf_parents(fm: dict) -> list[str]:
    parents = fm.get("parents") or []
    if not isinstance(parents, list):
        parents = [parents]
    return [str(p).strip() for p in parents if str(p).strip()]


def _predicts_line(body: str) -> str:
    """The what-if's falsifiable claim — the ``Predicts:`` line the skill staked."""
    for line in body.splitlines():
        if line.strip().lower().startswith("predicts:"):
            return line.strip()[len("predicts:"):].strip()
    return ""


def _cf_corroboration_worklist(ctx: DreamContext,
                               corpus: dict[str, tuple[KnowledgeNote, Path]]) -> list[dict]:
    """Stage 1 — the coarse funnel. Do NOT scan the corpus: for each OPEN counterfactual, keep
    the few of TODAY's episodes (``_partition`` seeds, minus the what-if's own parents) whose
    cosine to it clears CF_CORROB_LO. Cost is bounded by open-bets × today, never by corpus
    size — the whole point of matching arrivals against open bets instead of scanning. The
    skill (stage 2) makes the confirm/refute call on what survives. Best-effort: no open bets /
    no today-episodes / no model -> [] (lazy torch, after the trivial guards)."""
    open_cfs = _open_counterfactuals(ctx.subconscious_dir)
    if not open_cfs:
        return []
    seeds, _older = _partition(corpus, ctx.target)
    if not seeds:
        return []
    try:
        import numpy as np

        from recall.index import best_embedder
        emb = best_embedder(alert_degraded=True)
        cf_vecs = np.asarray(
            emb.embed([f"{fm.get('description', '')}\n{fm.get('pivot', '')}\n{body}"
                       for fm, body, _ in open_cfs]), dtype="float32")
        seed_notes = [corpus[s][0] for s in seeds]
        seed_vecs = np.asarray(
            emb.embed([f"{n.description}\n\n{n.body}" for n in seed_notes]), dtype="float32")
    except Exception as e:  # noqa: BLE001 — corroboration is optional; never fail the run
        print(f"[dream] WARN cf-corroboration funnel failed: {e}", flush=True)
        return []
    work: list[dict] = []
    for i, (fm, body, _path) in enumerate(open_cfs):
        slug = str(fm.get("name") or "")
        parents = set(_cf_parents(fm))
        scored = [(float(cf_vecs[i] @ seed_vecs[j]), s)
                  for j, s in enumerate(seeds) if s != slug and s not in parents]
        cands = [s for cos, s in sorted(scored, key=lambda c: -c[0])
                 if cos >= CF_CORROB_LO][:CF_CORROB_MAX_CAND]
        if not cands:
            continue
        work.append({
            "cf": {"slug": slug, "description": str(fm.get("description") or ""),
                   "pivot": str(fm.get("pivot") or ""), "predicts": _predicts_line(body),
                   "parents": sorted(parents), "body": body},
            "candidates": [{"slug": s, "description": corpus[s][0].description,
                            "body": corpus[s][0].body} for s in cands]})
    return work


def _materialize_worklist(ctx: DreamContext, pairs: list[dict], cf_seeds: list[dict],
                          cf_corrob: list[dict],
                          corpus: dict[str, tuple[KnowledgeNote, Path]]) -> None:
    config.ensure_dirs(ctx.subconscious_dir, ctx.worklist_path.parent,
                       ctx.manifest_path.parent, ctx.verdicts_path.parent)

    def _card(slug: str) -> dict:
        n = corpus[slug][0]
        return {"slug": slug, "description": n.description, "body": n.body,
                "kind": n.kind}

    work = [{"seed": _card(p["seed"]), "older": _card(p["older"]), "cos": p["cos"]}
            for p in pairs]
    cf = [{"seed": _card(c["seed"]), "charge": c.get("charge")} for c in cf_seeds]
    ctx.worklist_path.write_text(json.dumps(
        {"date": ctx.target.isoformat(), "scope": ctx.scope,
         "pairs": work, "counterfactuals": cf, "corroborate": cf_corrob},
        indent=2))


def _build_env(ctx: DreamContext) -> dict[str, str]:
    return {
        **os.environ,
        "RECALL_DREAM_WORKLIST": str(ctx.worklist_path),
        "RECALL_DREAM_SUBCONSCIOUS": str(ctx.subconscious_dir),
        "RECALL_DREAM_MANIFEST": str(ctx.manifest_path),
        "RECALL_DREAM_VERDICTS": str(ctx.verdicts_path),
        "RECALL_DREAM_DATE": ctx.target.isoformat(),
        "RECALL_DREAM_SCOPE": ctx.scope,
    }


def _invoke_claude(ctx: DreamContext, env: dict[str, str],
                   timeout_s: int = CLAUDE_TIMEOUT_S
                   ) -> subprocess.CompletedProcess[bytes]:
    return run_claude_with_backoff(
        [CLAUDE_BIN, "-p", "/dream",
         "--model", DREAM_MODEL, "--effort", DREAM_EFFORT,
         "--add-dir", str(config.data_root()),
         "--allowedTools", *_DREAM_ALLOWED_TOOLS],
        env=env, cwd=str(ctx.repo), timeout=timeout_s)


def _stamp_hypothesis_defaults(ctx: DreamContext, manifest: CurationManifest) -> int:
    """The skill writes the hypothesis text + parents + confidence; the wrapper
    OWNS the lifecycle fields — stamp any the skill left unset so every hypothesis
    is born quarantined, fragile, and unverified."""
    n = 0
    for note in manifest.notes:
        path = ctx.subconscious_dir / f"{note.slug}.md"
        if not path.exists():
            continue
        fm, _body = _split_fm(path.read_text())
        updates: dict[str, object] = {}
        if "status" not in fm:
            updates["status"] = "unverified"
        if "kind" not in fm:
            updates["kind"] = "hypothesis"
        if "first_seen" not in fm:
            updates["first_seen"] = ctx.target.isoformat()
        if "last_updated" not in fm:
            updates["last_updated"] = ctx.target.isoformat()
        if "stability" not in fm:
            updates["stability"] = DREAM_S0
        if "corroborations" not in fm:
            updates["corroborations"] = 0
        if "blessed" not in fm:
            updates["blessed"] = "false"
        if updates:
            try:
                path.write_text(set_frontmatter_keys(path.read_text(), updates))
                n += 1
            except (CurationSchemaError, OSError):
                continue
    return n


# ---- the bleed membrane: corroboration, promotion, decay -----------------

def _today_activated(corpus: dict[str, tuple[KnowledgeNote, Path]],
                     target: date) -> set[str]:
    iso = target.isoformat()
    return {s for s, (n, _) in corpus.items() if n.last_used[:10] == iso}


def bleed(ctx: DreamContext, corpus: dict[str, tuple[KnowledgeNote, Path]],
          *, promote: Callable[[DreamContext, dict, str], str | None]) -> dict:
    """Walk the subconscious and apply the valve. For each unverified hypothesis:
    if BOTH parents were re-activated together today (reality re-deriving the
    connection) bump its corroboration; if it has earned promotion (≥ N
    corroborations, or operator-blessed) and we are under tonight's bleed cap,
    PROMOTE it into the soul; if it is old and uncorroborated, retire it. Returns
    a summary. ``promote`` is injected so tests don't write the corpus."""
    activated = _today_activated(corpus, ctx.target)
    promoted: list[str] = []
    corroborated = retired = 0
    for fm, body, path in _load_hypotheses(ctx.subconscious_dir):
        status = str(fm.get("status") or "unverified").strip()
        if status in ("promoted", "discarded"):
            continue
        slug = str(fm.get("name") or path.stem)
        parents = fm.get("parents") or []
        if not isinstance(parents, list):
            parents = [parents]
        parents = [str(p).strip() for p in parents if str(p).strip()]
        corr = _as_int(fm.get("corroborations"), 0)
        blessed = _as_bool(fm.get("blessed"))

        # corroboration: both parents re-activated together today (and not the
        # night this dream was born)
        born = str(fm.get("first_seen") or "")[:10]
        if (born != ctx.target.isoformat() and len(parents) >= 2
                and all(p in activated for p in parents)):
            corr += 1
            corroborated += 1
            _set_keys(path, {"corroborations": corr,
                             "last_updated": ctx.target.isoformat()})

        if (corr >= DREAM_PROMOTE_N or blessed) and len(promoted) < DREAM_BLEED_MAX:
            c = promote(ctx, {"slug": slug, "fm": fm, "body": body,
                              "parents": parents}, path.read_text())
            if c:
                promoted.append(c)
                _set_keys(path, {"status": "promoted",
                                 "last_updated": ctx.target.isoformat()})
            continue

        # decay: an old, uncorroborated, unblessed hypothesis fades
        age = _age_days(born, ctx.target)
        if age >= DREAM_TTL_DAYS and corr == 0 and not blessed:
            _set_keys(path, {"status": "discarded",
                             "last_updated": ctx.target.isoformat()})
            retired += 1
    return {"promoted": promoted, "corroborated": corroborated, "retired": retired}


def _promote_to_soul(ctx: DreamContext, hyp: dict, _raw: str) -> str | None:
    """Promote a corroborated/blessed hypothesis into the soul as a `kind: lesson`
    note — affect-stripped (we keep the conjecture + its provenance, not drama),
    born at MODEST stability (earned, not flashbulb — it must keep earning its
    place), linked to its parents, ACCUMULATED alongside real notes. Reversible:
    a plain note the weekly reconsolidate can supersede. Returns the new slug."""
    fm, parents = hyp["fm"], hyp["parents"]
    desc = str(fm.get("description") or hyp["slug"]).strip().replace('"', "'")
    slug = f"insight-{hyp['slug']}"[:80]
    links = " ".join(f"[[{p}]]" for p in parents)
    note = (
        f"---\nname: {slug}\n"
        f'description: "{desc}"\n'
        f"kind: lesson\ntags: [dream-derived]\n"
        f"first_seen: {ctx.target.isoformat()}\n"
        f"last_updated: {ctx.target.isoformat()}\n"
        f"sources: [{ctx.target.isoformat()}]\n"
        f"stability: {round(dynamics.initial_stability(0.6), 3)}\n"
        f"last_used: {ctx.target.isoformat()}\nuses: 0\n"
        f"---\n{hyp['body'].strip()}\n\n"
        f"Promoted from a subconscious hypothesis ([[{hyp['slug']}]]) after it was "
        f"independently corroborated. Connects {links}.\n")
    path = ctx.corpus_dir / f"{slug}.md"
    try:
        KnowledgeNote.parse(note, expect_slug=slug)   # validate before it lands
        path.write_text(note)
    except (CurationSchemaError, OSError) as e:
        print(f"[dream] WARN promotion of {hyp['slug']} failed: {e}", flush=True)
        return None
    return slug


# ---- L1: counterfactual corroboration — stage 3, act on the verdict ------

def _read_verdicts(path: Path) -> list[dict]:
    """The skill's confirm/refute rulings on the corroboration worklist. Best-effort: a
    missing or garbled file is a quiet no-op — corroboration never fails the run."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return [v for v in data if isinstance(v, dict)] if isinstance(data, list) else []


def apply_cf_verdicts(ctx: DreamContext,
                      corpus: dict[str, tuple[KnowledgeNote, Path]],
                      verdicts: list[dict], *,
                      promote: Callable[[DreamContext, dict, str], str | None],
                      cap_used: int) -> dict:
    """Stage 3 — act on the ruling. ONE clean ``confirm`` graduates the what-if into the soul
    (reusing ``_promote_to_soul``; single-parent link), sharing tonight's DREAM_BLEED_MAX cap;
    a ``refute`` RETIRES it now (``status: discarded`` + ``refuted_by``), faster than TTL; any
    other verdict leaves it open. We do NOT take the skill's word that reality moved: the cited
    ``evidence`` must be one of TODAY's episodes, else the ruling is ignored. ``promote`` is
    injected so tests never write the corpus."""
    today = set(_partition(corpus, ctx.target)[0])
    promoted: list[str] = []
    retired = 0
    for v in verdicts:
        cf_slug = str(v.get("cf") or "").strip()
        verdict = str(v.get("verdict") or "").strip().lower()
        evidence = str(v.get("evidence") or "").strip()
        if not cf_slug or verdict not in ("confirm", "refute"):
            continue
        path = ctx.subconscious_dir / f"{cf_slug}.md"
        if not path.exists():
            continue
        try:
            fm, body = _split_fm(path.read_text())
        except CurationSchemaError:
            continue
        if str(fm.get("status") or "unverified").strip() in ("promoted", "discarded"):
            continue
        if not evidence or evidence not in today:   # reality must actually have surfaced it
            continue
        if verdict == "confirm" and (cap_used + len(promoted)) < DREAM_BLEED_MAX:
            c = promote(ctx, {"slug": cf_slug, "fm": fm, "body": body,
                              "parents": _cf_parents(fm)}, path.read_text())
            if c:
                promoted.append(c)
                _set_keys(path, {"status": "promoted", "corroborations": 1,
                                 "corroborated_by": evidence,
                                 "last_updated": ctx.target.isoformat()})
        elif verdict == "refute":
            _set_keys(path, {"status": "discarded", "refuted_by": evidence,
                             "last_updated": ctx.target.isoformat()})
            retired += 1
    return {"promoted": promoted, "retired": retired}


# ---- small frontmatter scalar helpers ------------------------------------

def _set_keys(path: Path, updates: dict) -> None:
    try:
        path.write_text(set_frontmatter_keys(path.read_text(), updates))
    except (CurationSchemaError, OSError):
        pass


def _as_int(v, default: int) -> int:
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "yes", "1", "on")


def _age_days(ref_iso: str, target: date) -> int:
    if not ref_iso:
        return 0
    try:
        return max(0, (target - date.fromisoformat(ref_iso[:10])).days)
    except ValueError:
        return 0


# ---- digest + session log ------------------------------------------------

def _write_digest(ctx: DreamContext, new_hyps: list, bleed_summary: dict) -> None:
    ctx.digest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# dream digest ({ctx.label}) — {ctx.target.isoformat()}\n",
             "\n_Overnight recombinations — conjectures, not facts. A recombination "
             "hypothesis bleeds into the soul once reality corroborates it twice (both "
             "parent memories resurface together on two separate days); a counterfactual "
             "what-if graduates on a single prospective match — a later day proves (or "
             "refutes) its prediction. Setting `blessed: true` is an optional manual "
             "fast-path. An uncorroborated conjecture fades after 30 days._\n"]
    if bleed_summary.get("promoted"):
        lines.append("\n## Bled into the soul (corroborated)\n")
        for s in bleed_summary["promoted"]:
            lines.append(f"- `{s}`\n")
    if new_hyps:
        lines.append("\n## New hypotheses tonight\n")
        for h in new_hyps:
            lines.append(f"- `{h.slug}` — {h.title}\n")
    if not new_hyps and not bleed_summary.get("promoted"):
        lines.append("\n_Quiet night — nothing recombined._\n")
    ctx.digest_path.write_text("".join(lines))


def _append_session_block(ctx: DreamContext, outcome: Outcome, n_new: int,
                          bleed_summary: dict) -> None:
    path = ctx.session_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"# dream ({ctx.label}) — {ctx.target.isoformat()}\n")
    when = _et_clock()
    if outcome.kind == "dreamed":
        body = (f"\n## {when} — {n_new} hypothesis(es); "
                f"{len(bleed_summary.get('promoted', []))} promoted, "
                f"{bleed_summary.get('corroborated', 0)} corroborated, "
                f"{bleed_summary.get('retired', 0)} retired\n")
    else:
        body = f"\n## {when} — {outcome.kind} ({outcome.reason}): {outcome.detail}\n"
    path.write_text(path.read_text() + body)


# ---- orchestration -------------------------------------------------------

def _print(o: Outcome) -> None:
    print(f"[dream] {o.kind}: {o.reason} — {o.detail}", flush=True)


def run(argv: list[str] | None = None, *,
        invoke_claude: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
        compute_pairs: Callable[[DreamContext, dict], list[dict]] | None = None,
        compute_cf_seeds: Callable[[DreamContext, dict], list[dict]] | None = None,
        compute_cf_corrob: Callable[[DreamContext, dict], list[dict]] | None = None,
        promote: Callable[[DreamContext, dict, str], str | None] | None = None,
        rebuild_index: Callable[[DreamContext], int] | None = None,
        autocommit: Callable[[DreamContext, list], str | None] | None = None,
        today_et: date | None = None) -> Outcome:
    """Entry point. All side-effecting collaborators (the claude subprocess, the
    embedder pairing, soul promotion, index rebuild, git) are injectable so tests
    run hermetic; ``today_et`` drives the date."""
    invoke_claude = invoke_claude or _invoke_claude
    compute_pairs = compute_pairs or _recombination_pairs
    compute_cf_seeds = compute_cf_seeds or _counterfactual_seeds
    compute_cf_corrob = compute_cf_corrob or _cf_corroboration_worklist
    promote = promote or _promote_to_soul
    rebuild_index = rebuild_index or _rebuild_index
    autocommit = autocommit or _autocommit

    args = _parse_args(argv)
    if today_et is None:
        today_et = datetime.now(timezone.utc).astimezone(ET).date()
    try:
        target = date.fromisoformat(args.date) if args.date else today_et
    except ValueError:
        o = Outcome(kind="failed", reason="bad_date",
                    detail=f"--date {args.date!r} is not an ISO date", exit_code=1)
        _print(o)
        return o
    ctx = _resolve(args, target)

    if target.isoformat() in _done_dates(ctx.state_file) and not args.force:
        o = Outcome(kind="skipped", reason="already_dreamed",
                    detail=f"{target.isoformat()} already dreamed for {ctx.label}",
                    exit_code=0)
        _print(o)
        return o
    if not ctx.corpus_dir.is_dir():
        o = Outcome(kind="skipped", reason="no_corpus",
                    detail=f"corpus dir absent: {ctx.corpus_dir}", exit_code=0)
        _print(o)
        return o

    corpus = _load_corpus(ctx.corpus_dir)
    # "Today's experience" — the notes curate+consolidate stamped with today's date
    # earlier in the nightly chain. These are the REM seeds; with none of them, there
    # is nothing *of today* to recombine (see the burned-night guard below).
    seeds_present = any(_is_today(n, target) for (n, _p) in corpus.values())

    # The bleed membrane runs every invocation, independent of whether new hypotheses
    # are generated — yesterday's dreams keep maturing toward (or away from) the soul.
    # It is idempotent: promoted/discarded are terminal and skipped on any re-run.
    bleed_summary = bleed(ctx, corpus, promote=promote)
    bled = bool(bleed_summary["promoted"] or bleed_summary["corroborated"]
                or bleed_summary["retired"])

    pairs = compute_pairs(ctx, corpus)
    cf_seeds = compute_cf_seeds(ctx, corpus) if args.counterfactual else []
    cf_corrob = compute_cf_corrob(ctx, corpus) if args.counterfactual else []
    if args.dry_run:
        print(f"[dream] DRY-RUN {ctx.label} {target}: {len(pairs)} pair(s), "
              f"{len(cf_seeds)} counterfactual(s), {len(cf_corrob)} to-corroborate, "
              f"{len(bleed_summary['promoted'])} would-promote", flush=True)
        return Outcome(kind="skipped", reason="dry_run",
                       detail="dry-run only; no skill call", exit_code=0)

    # The burned-night guard. Recombination needs today's experience as seeds, which
    # the earlier curate+consolidate stages stamp onto the day's notes. A run with no
    # seeds and nothing for bleed to mature is either premature (fired before
    # consolidation — a daytime or manual invocation) or a genuinely empty day; either
    # way the experiential half hasn't happened. Returning here WITHOUT marking the
    # date done leaves it open for the authoritative post-consolidation nightly run.
    # (Without this, a stray morning run silently burned the night — the 2026-06-25
    # regression, where the real 23:23 nightly skipped as already_dreamed and the soul
    # never actually dreamed.) --force overrides, for deliberate manual re-dreaming.
    if not seeds_present and not bled and not args.force:
        o = Outcome(kind="skipped", reason="no_experience_yet",
                    detail=f"nothing lived on {target.isoformat()} yet for {ctx.label}; "
                           f"date left open for the nightly run", exit_code=0)
        _print(o)
        return o

    manifest = None
    if pairs or cf_seeds or cf_corrob:
        _materialize_worklist(ctx, pairs, cf_seeds, cf_corrob, corpus)
        print(f"[dream] invoking claude for {ctx.label} {target} "
              f"({len(pairs)} recombination pair(s), {len(cf_seeds)} counterfactual(s), "
              f"{len(cf_corrob)} to-corroborate)", flush=True)
        try:
            cp = invoke_claude(ctx, _build_env(ctx), CLAUDE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            o = Outcome(kind="failed", reason="claude_timeout",
                        detail=f"claude did not return within {CLAUDE_TIMEOUT_S}s",
                        exit_code=1, alert_priority="urgent")
            _print(o); _append_session_block(ctx, o, 0, bleed_summary)
            return o
        except FileNotFoundError as e:
            o = Outcome(kind="failed", reason="claude_bin_missing",
                        detail=f"claude binary not found: {e}", exit_code=1)
            _print(o); _append_session_block(ctx, o, 0, bleed_summary)
            return o
        if cp.returncode != 0:
            o = Outcome(kind="failed", reason="claude_nonzero",
                        detail=f"claude exited with code {cp.returncode}",
                        exit_code=1, alert_priority="urgent")
            _print(o); _append_session_block(ctx, o, 0, bleed_summary)
            return o
        manifest, fail = validate_manifest_against(
            ctx.manifest_path, target.isoformat(), lambda _s: ctx.subconscious_dir)
        if fail is not None:
            _print(fail); _append_session_block(ctx, fail, 0, bleed_summary)
            return fail
        _stamp_hypothesis_defaults(ctx, manifest)
        # Stage 3 of corroboration: consume the skill's confirm/refute rulings on open
        # what-ifs. Promotions share tonight's bleed cap (bleed ran first, above), so we
        # fold them into bleed_summary for the digest, index rebuild, notify and commit.
        verdicts = _read_verdicts(ctx.verdicts_path)
        if verdicts:
            cf_apply = apply_cf_verdicts(ctx, corpus, verdicts, promote=promote,
                                         cap_used=len(bleed_summary["promoted"]))
            bleed_summary["promoted"] += cf_apply["promoted"]
            bleed_summary["retired"] += cf_apply["retired"]

    new_notes = manifest.notes if manifest else []
    # Only an *authoritative* run — one that had today's experience to recombine, or an
    # explicit --force — consumes the date. A bleed-only run (matured yesterday's
    # hypotheses on a day not yet lived) leaves the date open; bleed is idempotent so
    # the nightly re-runs it harmlessly and then recombines once the seeds exist.
    if seeds_present or args.force:
        _mark_done(ctx.state_file, target)
    _write_digest(ctx, new_notes, bleed_summary)
    if bleed_summary["promoted"]:
        try:
            rebuild_index(ctx)
        except Exception as e:  # noqa: BLE001 — derived index, never fatal
            print(f"[dream] WARN index rebuild failed (corpus intact): {e}", flush=True)

    detail = (f"{ctx.label}: {len(new_notes)} new hypothesis(es), "
              f"{len(bleed_summary['promoted'])} promoted, "
              f"{bleed_summary['corroborated']} corroborated, "
              f"{bleed_summary['retired']} retired")
    ok = Outcome(kind="dreamed", reason="ok", detail=detail, exit_code=0)
    _print(ok)
    _append_session_block(ctx, ok, len(new_notes), bleed_summary)
    if bleed_summary["promoted"]:
        notify_alert(title=f"[dream] {ctx.label}: {len(bleed_summary['promoted'])} "
                           f"insight(s) bled into the soul",
                     body=", ".join(bleed_summary["promoted"]), priority="low")
    if args.commit and bleed_summary["promoted"]:
        try:
            c = autocommit(ctx, bleed_summary["promoted"])
            if c:
                print(f"[dream] {c}", flush=True)
        except Exception as e:  # noqa: BLE001 — corpus already written, never fatal
            print(f"[dream] WARN auto-commit failed (corpus intact): {e}", flush=True)
    return ok


def _rebuild_index(ctx: DreamContext) -> int:
    from recall.index import best_embedder, build_index
    return build_index(ctx.corpus_dir, ctx.index_path,
                       best_embedder(alert_degraded=True))


def _autocommit(ctx: DreamContext, promoted: list) -> str | None:
    paths = ["."] if ctx.scope == "global" else [str(ctx.corpus_dir)]
    return _git_commit_scoped(ctx.repo, paths, ctx.target,
                              f"bled {len(promoted)} corroborated insight(s) into the soul",
                              f"dream {ctx.label}")


def main(argv: list[str] | None = None) -> int:
    return run(argv).exit_code


if __name__ == "__main__":
    sys.exit(main())
