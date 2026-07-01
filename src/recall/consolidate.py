#!/usr/bin/env python3
"""Activation consolidation — the deterministic nightly fold (no LLM, GPU-free).

Where ``curate`` distills new notes and ``reconsolidate`` re-examines the whole
corpus, ``consolidate`` does the cheap mechanical step in between: read a scope's
activation events (the hippocampal log the recall hook appends to), bump each
touched note's **stability** via the DSR/FSRS law, surgically persist
``stability``/``last_used``/``uses`` into the note's frontmatter, sync those
columns into the derived index in place (no re-embedding), and scoped-commit.

Decay is **lazy** — computed at query time from ``last_used`` + ``stability`` (see
``dynamics.retrievability``) — so this fold only ever PERSISTS reinforcement. It is
a clean no-op on a scope with no activity (notes still decay implicitly, no write
needed). One stability bump per note per run, honoring the spacing effect; a note
the operator actually *cited* that day is reinforced harder than one merely
surfaced.

Idempotent by construction: claiming the log atomically removes the consumed
events, so a second run the same day finds nothing.

Usage:
    recall consolidate --scope global
    recall consolidate --scope project --project-dir /x/foo --commit
    recall consolidate --scope global --dry-run
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from recall import activation, config, dynamics, index
from recall.curate import ET, Outcome, _et_clock, _git_commit_scoped
from recall.schema import CurationSchemaError, KnowledgeNote, set_frontmatter_keys


@dataclass(frozen=True)
class ConsolidateContext:
    scope: str              # "project" | "global"
    label: str              # scope id (project slug, or "global")
    corpus_dir: Path
    index_path: Path
    repo: Path
    target: date
    session_log_path: Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="recall consolidate",
                                description=__doc__.splitlines()[0])
    p.add_argument("--scope", choices=("project", "global"), default="global")
    p.add_argument("--project-dir", type=Path, default=None,
                   help="project to consolidate (--scope project; default cwd)")
    p.add_argument("--date", type=str, default=None,
                   help="ISO run date (default: today ET)")
    p.add_argument("--commit", action="store_true",
                   help="on success, scoped-commit the corpus ([consolidate])")
    p.add_argument("--dry-run", action="store_true",
                   help="report pending activations; persist nothing")
    return p.parse_args(argv)


def _resolve(args: argparse.Namespace, target: date) -> ConsolidateContext:
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
    return ConsolidateContext(
        scope=args.scope, label=label, corpus_dir=corpus_dir,
        index_path=config.index_path(label), repo=repo, target=target,
        session_log_path=(config.data_root() / "consolidation" / label
                          / "sessions" / f"{target.isoformat()}.md"))


# ---- corpus + reinforcement ----------------------------------------------

def _load_corpus(corpus_dir: Path) -> dict[str, tuple[KnowledgeNote, Path]]:
    """slug -> (note, path) for every parseable note (README + malformed skipped)."""
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


def _age_days(ref_iso: str, target: date) -> int:
    if not ref_iso:
        return 0
    try:
        d = date.fromisoformat(ref_iso[:10])
    except ValueError:
        return 0
    return max(0, (target - d).days)


def _reinforced_stability(note: KnowledgeNote, *, cited: bool, target: date) -> float:
    """New stability after this run's activation of ``note``. Bootstraps an unset
    stability from the note's surprise (flashbulb S₀) or the neutral default, ages
    it from the last activation (falling back to the last content edit), and
    applies one reinforcement at the resulting retrievability."""
    if note.stability and note.stability > 0:
        s = note.stability
    elif note.surprise >= 0:
        s = dynamics.initial_stability(note.surprise)
    else:
        s = dynamics.S_DEFAULT
    ref = note.last_used or note.last_updated or note.first_seen
    r = dynamics.retrievability(_age_days(ref, target), s)
    gain = dynamics.CITE_GAIN if cited else 1.0
    return dynamics.reinforce(s, r, gain=gain)


def _detect_cited(ctx: ConsolidateContext, candidates: set[str]) -> set[str]:
    """Best-effort: which surfaced slugs the model actually *used* that day —
    detected by the slug string appearing in the day's denoised transcript
    bundle(s). For a project scope, its own bundle; for global, every project's
    bundle (a soul note can be cited in any project's session). No bundle -> no
    citations (all activations count as mere surfacings). Never raises."""
    if not candidates:
        return set()
    texts: list[str] = []
    cdir = config.curation_dir()
    try:
        if ctx.scope == "global":
            for sub in sorted(p for p in cdir.glob("*") if p.is_dir()):
                b = sub / "bundles" / f"{ctx.target.isoformat()}.md"
                if b.exists():
                    texts.append(b.read_text())
        else:
            b = cdir / ctx.label / "bundles" / f"{ctx.target.isoformat()}.md"
            if b.exists():
                texts.append(b.read_text())
    except OSError:
        return set()
    if not texts:
        return set()
    blob = "\n".join(texts)
    return {s for s in candidates if s in blob}


# ---- session log ---------------------------------------------------------

def _append_session_block(ctx: ConsolidateContext, outcome: Outcome,
                          n_notes: int, n_cited: int) -> None:
    path = ctx.session_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"# consolidate ({ctx.label}) — {ctx.target.isoformat()}\n")
    when = _et_clock()
    if outcome.kind == "consolidated":
        body = (f"\n## {when} — reinforced {n_notes} note(s) ({n_cited} cited)\n"
                f"\n- {outcome.detail}\n")
    else:
        body = f"\n## {when} — {outcome.kind} ({outcome.reason}): {outcome.detail}\n"
    path.write_text(path.read_text() + body)


# ---- orchestration -------------------------------------------------------

def _print(o: Outcome) -> None:
    print(f"[consolidate] {o.kind}: {o.reason} — {o.detail}", flush=True)


def run(argv: list[str] | None = None, *,
        claim_events: Callable[[str], tuple[list[dict], Path | None]] | None = None,
        detect_cited: Callable[[ConsolidateContext, set[str]], set[str]] | None = None,
        sync_index: Callable[[Path, list[tuple]], int] | None = None,
        autocommit: Callable[[ConsolidateContext], str | None] | None = None,
        today_et: date | None = None) -> Outcome:
    """Entry point. ``claim_events`` / ``detect_cited`` / ``sync_index`` /
    ``autocommit`` are injectable so tests run fully hermetic (no model, no git);
    ``today_et`` drives the run date."""
    claim_events = claim_events or activation.claim_events
    detect_cited = detect_cited or _detect_cited
    sync_index = sync_index or index.update_dynamics
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

    events, claimed = claim_events(ctx.label)
    if not events:
        o = Outcome(kind="skipped", reason="no_activations",
                    detail=f"{ctx.label}: no pending activations", exit_code=0)
        _print(o)
        return o

    agg = activation.rollup(events)
    if args.dry_run:
        # Don't persist; but a claimed log must not silently swallow the events —
        # leave the consuming file in place for the real run (we read, didn't discard).
        print(f"[consolidate] DRY-RUN {ctx.label} {ctx.target}: "
              f"{len(agg)} note(s) across {len(events)} event(s)", flush=True)
        return Outcome(kind="skipped", reason="dry_run",
                       detail="dry-run only; nothing persisted", exit_code=0)

    if not ctx.corpus_dir.is_dir():
        # Corpus gone but events pending: drop them (nothing to reinforce) so they
        # don't accumulate forever.
        activation.discard_claimed(ctx.label)
        o = Outcome(kind="skipped", reason="no_corpus",
                    detail=f"corpus dir absent: {ctx.corpus_dir}", exit_code=0)
        _print(o)
        return o

    corpus = _load_corpus(ctx.corpus_dir)
    cited = detect_cited(ctx, set(agg) & set(corpus))

    index_rows: list[tuple[str, float, str, int]] = []
    n_notes = 0
    for slug, stats in agg.items():
        entry = corpus.get(slug)
        if entry is None:
            continue  # surfaced note no longer in the corpus (renamed/pruned)
        note, path = entry
        new_s = _reinforced_stability(note, cited=(slug in cited), target=target)
        new_uses = note.uses + int(stats.get("count", 0))
        updates = {"stability": round(new_s, 3),
                   "last_used": target.isoformat(),
                   "uses": new_uses}
        try:
            path.write_text(set_frontmatter_keys(path.read_text(), updates))
        except (CurationSchemaError, OSError) as e:
            print(f"[consolidate] WARN skip {slug}: {e}", flush=True)
            continue
        index_rows.append((slug, round(new_s, 3), target.isoformat(), new_uses))
        n_notes += 1

    try:
        synced = sync_index(ctx.index_path, index_rows)
        print(f"[consolidate] index synced: {ctx.label} ({synced} rows)", flush=True)
    except Exception as e:  # noqa: BLE001 — derived index; never fatal
        print(f"[consolidate] WARN index sync failed (corpus intact): {e}", flush=True)

    activation.discard_claimed(ctx.label)

    detail = (f"{ctx.label}: reinforced {n_notes} note(s), "
              f"{len(cited)} cited, from {len(events)} activation(s)")
    ok = Outcome(kind="consolidated", reason="ok", detail=detail, exit_code=0)
    _print(ok)
    _append_session_block(ctx, ok, n_notes, len(cited))
    if args.commit:
        try:
            c = autocommit(ctx)
            if c:
                print(f"[consolidate] {c}", flush=True)
        except Exception as e:  # noqa: BLE001 — corpus already written, never fatal
            print(f"[consolidate] WARN auto-commit failed (corpus intact): {e}",
                  flush=True)
    return ok


def _autocommit(ctx: ConsolidateContext) -> str | None:
    paths = ["."] if ctx.scope == "global" else [str(ctx.corpus_dir)]
    return _git_commit_scoped(ctx.repo, paths, ctx.target,
                              f"reinforced activations ({ctx.label})",
                              f"consolidate {ctx.label}")


def main(argv: list[str] | None = None) -> int:
    return run(argv).exit_code


if __name__ == "__main__":
    sys.exit(main())
