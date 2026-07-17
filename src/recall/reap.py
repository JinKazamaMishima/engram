#!/usr/bin/env python3
"""ebb — the reaper: recall's active-forgetting pass (deterministic, GPU-free).

Where ``consolidate`` REINFORCES the notes the day used, ``reap`` does the other
half of memory that recall was missing entirely: it EVICTS the notes the corpus
has stopped using. A note is archived when it is either

  - **superseded** — ``superseded: true`` (its content already lives in the note
    that replaced it, so keeping it in the live set is pure noise), or
  - **cold** — dormant past ``REAP_DORMANT_DAYS`` AND its FSRS
    ``effective_retrievability`` has fallen below ``REAP_R_FLOOR`` AND it has been
    used at most ``REAP_USES_MAX`` times.

"Archive" is a **reversible move**, never a delete: the note file is moved from
``<corpus>/*.md`` into ``<corpus>/archive/`` (frontmatter stamped with why). The
corpus loaders glob non-recursively (``glob("*.md")``), so an archived note drops
out of the rebuilt index automatically while staying in the same git repo — still
on disk, still ``cat``-able (so the miss-log can catch a wrong eviction), and one
``recall reap --restore <slug>`` away from coming back.

Three classes of note are NEVER reaped: operator-owned rules (``kind: rule``),
notes that have graduated to permanence (``is_permanent``), and identity/soul
anchors (``importance >= REAP_IMPORTANCE_FLOOR``). All thresholds are
env-overridable (``RECALL_REAP_*``); the defaults are the "aggressive" preset.

Runs every night (``reap-all`` after ``consolidate-all``); a day with nothing to
evict makes no commit. Not week-gated — forgetting is continuous, not lumpy.

Usage:
    recall reap --scope global
    recall reap --scope project --project-dir /x/foo --commit
    recall reap --scope project --project-dir /x/foo --dry-run
    recall reap --scope global --restore some-note-slug
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from recall import config, dynamics, rules
from recall.curate import ET, Outcome, _git_commit_scoped
from recall.schema import CurationSchemaError, KnowledgeNote, set_frontmatter_keys


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Selection thresholds — the "aggressive" preset (the shipped defaults). All tunable.
REAP_DORMANT_DAYS = _env_int("RECALL_REAP_DORMANT_DAYS", 30)     # min days since last touch
REAP_R_FLOOR = _env_float("RECALL_REAP_R_FLOOR", 0.20)          # evict below this retrievability
REAP_USES_MAX = _env_int("RECALL_REAP_USES_MAX", 2)             # only lightly-used notes
REAP_IMPORTANCE_FLOOR = _env_float("RECALL_REAP_IMPORTANCE_FLOOR", 1.0)  # protect anchors >= this
# Stability to assume for a note the consolidate fold never touched (never
# activated, no surprise recorded): S_MIN decays fast — an unremarkable, unused
# note is exactly what we want to let go. A note born SURPRISING keeps its
# flashbulb S0 (protected); an activated note keeps its real S.
REAP_BOOTSTRAP_S = _env_float("RECALL_REAP_BOOTSTRAP_S", dynamics.S_MIN)
REAP_MAX_PER_RUN = _env_int("RECALL_REAP_MAX_PER_RUN", 0)       # 0 == unbounded (full send)


@dataclass(frozen=True)
class ReapContext:
    scope: str            # "project" | "global"
    label: str            # scope id for paths/commits (project slug, or "global")
    corpus_dir: Path
    archive_dir: Path
    index_path: Path
    repo: Path            # git repo the scoped commit runs in
    target: date


@dataclass(frozen=True)
class Candidate:
    slug: str
    path: Path
    reason: str           # "superseded" | "cold"
    r: float              # effective_retrievability at judging time
    age_days: int


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="recall reap",
                                description=__doc__.splitlines()[0])
    p.add_argument("--scope", choices=("project", "global"), default="global")
    p.add_argument("--project-dir", type=Path, default=None,
                   help="project to reap (--scope project; default cwd)")
    p.add_argument("--date", type=str, default=None,
                   help="ISO run date (default: today ET)")
    p.add_argument("--commit", action="store_true",
                   help="on success, scoped-commit the corpus ([curator] reap)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be archived; move/commit nothing")
    p.add_argument("--max-per-run", type=int, default=None,
                   help="archive at most N coldest notes this run (default: "
                        "RECALL_REAP_MAX_PER_RUN, 0 == unbounded)")
    p.add_argument("--restore", type=str, default=None, metavar="SLUG",
                   help="move a note back OUT of archive/ into the live corpus")
    return p.parse_args(argv)


def _resolve(args: argparse.Namespace, target: date) -> ReapContext:
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
    return ReapContext(
        scope=args.scope, label=label, corpus_dir=corpus_dir,
        archive_dir=config.archive_dir(corpus_dir),
        index_path=config.index_path(label), repo=repo, target=target)


# ---- corpus + selection --------------------------------------------------

def _load_corpus(corpus_dir: Path) -> list[tuple[KnowledgeNote, Path]]:
    """(note, path) for every parseable live note. Non-recursive glob, so the
    archive/ subdir is never re-examined; README + malformed are skipped."""
    out: list[tuple[KnowledgeNote, Path]] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            note = KnowledgeNote.parse(path.read_text(), expect_slug=path.stem)
        except CurationSchemaError:
            continue
        out.append((note, path))
    return out


def _age_days(ref_iso: str, target: date) -> int:
    if not ref_iso:
        return 0
    try:
        d = date.fromisoformat(ref_iso[:10])
    except ValueError:
        return 0
    return max(0, (target - d).days)


def _most_recent_touch(note: KnowledgeNote) -> str:
    """Latest of last_used / last_updated / first_seen (ISO dates compare
    lexicographically) — a note edited or used recently is not dormant."""
    return max((s[:10] for s in (note.last_used, note.last_updated, note.first_seen)
                if s), default="")


def _stability_for(note: KnowledgeNote) -> float:
    """The stability to judge coldness by — mirrors the consolidate bootstrap so
    the reaper and the fold share one notion of a note's strength, except an
    entirely un-signalled note falls to REAP_BOOTSTRAP_S (fast decay)."""
    if note.stability and note.stability > 0:
        return note.stability
    if note.surprise >= 0:
        return dynamics.initial_stability(note.surprise)
    return REAP_BOOTSTRAP_S


def _is_exempt(note: KnowledgeNote) -> bool:
    """Never-reap classes: operator-owned rules, graduated-permanent notes, and
    identity/soul anchors (high importance)."""
    return (note.kind == rules.RULE_KIND
            or dynamics.is_permanent(note.stability)
            or note.importance >= REAP_IMPORTANCE_FLOOR)


def _classify(note: KnowledgeNote, path: Path, target: date) -> Candidate | None:
    """Decide if a note should be evicted, and why. None == keep."""
    if _is_exempt(note):
        return None
    s = _stability_for(note)
    age = _age_days(_most_recent_touch(note), target)
    r = dynamics.effective_retrievability(age, s)
    if note.superseded:
        return Candidate(note.slug, path, "superseded", r, age)
    if age >= REAP_DORMANT_DAYS and r < REAP_R_FLOOR and note.uses <= REAP_USES_MAX:
        return Candidate(note.slug, path, "cold", r, age)
    return None


def _select(corpus: list[tuple[KnowledgeNote, Path]], target: date,
            max_per_run: int) -> list[Candidate]:
    cands = [c for note, path in corpus
             if (c := _classify(note, path, target)) is not None]
    # Coldest first, so a --max-per-run cap keeps the least-retrievable.
    cands.sort(key=lambda c: c.r)
    return cands[:max_per_run] if max_per_run > 0 else cands


# ---- archive move + restore ----------------------------------------------

def _archive(cand: Candidate, ctx: ReapContext) -> None:
    """Move one note into archive/, stamping why. Reversible; content untouched
    apart from the added frontmatter keys."""
    config.ensure_dirs(ctx.archive_dir)
    stamped = set_frontmatter_keys(cand.path.read_text(), {
        "archived_on": ctx.target.isoformat(),
        "archived_reason": cand.reason,
        "archived_r": round(cand.r, 3),
    })
    (ctx.archive_dir / cand.path.name).write_text(stamped)
    cand.path.unlink()


def _restore_one(ctx: ReapContext, slug: str) -> bool:
    """Move a note back from archive/ into the live corpus. False if absent."""
    src = ctx.archive_dir / f"{slug}.md"
    if not src.exists():
        return False
    config.ensure_dirs(ctx.corpus_dir)
    stamped = set_frontmatter_keys(src.read_text(),
                                   {"restored_on": ctx.target.isoformat()})
    (ctx.corpus_dir / f"{slug}.md").write_text(stamped)
    src.unlink()
    return True


# ---- index + commit ------------------------------------------------------

def _rebuild_index(ctx: ReapContext) -> int:
    from recall.index import best_embedder, build_index
    return build_index(ctx.corpus_dir, ctx.index_path,
                       best_embedder(alert_degraded=True))


def _autocommit(ctx: ReapContext, summary: str) -> str | None:
    paths = ["."] if ctx.scope == "global" else [str(ctx.corpus_dir)]
    return _git_commit_scoped(ctx.repo, paths, ctx.target, summary,
                              f"reap {ctx.label}")


# ---- orchestration -------------------------------------------------------

def _print(o: Outcome) -> None:
    print(f"[reap] {o.kind}: {o.reason} — {o.detail}", flush=True)


def run(argv: list[str] | None = None, *,
        load_corpus: Callable[[Path], list[tuple[KnowledgeNote, Path]]] | None = None,
        rebuild_index: Callable[[ReapContext], int] | None = None,
        autocommit: Callable[[ReapContext, str], str | None] | None = None,
        today_et: date | None = None) -> Outcome:
    """Entry point. ``load_corpus`` / ``rebuild_index`` / ``autocommit`` are
    injectable so tests run fully hermetic (no embedder, no git); ``today_et``
    drives the run date."""
    load_corpus = load_corpus or _load_corpus
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

    if not ctx.corpus_dir.is_dir():
        o = Outcome(kind="skipped", reason="no_corpus",
                    detail=f"corpus dir absent: {ctx.corpus_dir}", exit_code=0)
        _print(o)
        return o

    # --restore is a manual reversal: move one note back, rebuild, commit.
    if args.restore:
        if not _restore_one(ctx, args.restore):
            o = Outcome(kind="skipped", reason="not_archived",
                        detail=f"{args.restore!r} not found in {ctx.archive_dir}",
                        exit_code=0)
            _print(o)
            return o
        _safe_rebuild(ctx, rebuild_index)
        o = Outcome(kind="restored", reason="ok",
                    detail=f"{ctx.label}: restored {args.restore} to the corpus",
                    exit_code=0)
        _print(o)
        if args.commit:
            _safe_commit(ctx, autocommit, f"restored {args.restore} from archive")
        return o

    max_per_run = (args.max_per_run if args.max_per_run is not None
                   else REAP_MAX_PER_RUN)
    cands = _select(load_corpus(ctx.corpus_dir), target, max_per_run)

    if not cands:
        o = Outcome(kind="skipped", reason="nothing_cold",
                    detail=f"{ctx.label}: no superseded or cold notes", exit_code=0)
        _print(o)
        return o

    n_sup = sum(1 for c in cands if c.reason == "superseded")
    n_cold = len(cands) - n_sup
    if args.dry_run:
        print(f"[reap] DRY-RUN {ctx.label} {target}: would archive {len(cands)} "
              f"note(s) — {n_sup} superseded, {n_cold} cold:", flush=True)
        for c in cands:
            print(f"[reap]   - {c.slug}  ({c.reason}, R={c.r:.3f}, "
                  f"dormant {c.age_days}d)", flush=True)
        return Outcome(kind="skipped", reason="dry_run",
                       detail=f"{len(cands)} would be archived; nothing moved",
                       exit_code=0)

    for c in cands:
        try:
            _archive(c, ctx)
        except OSError as e:
            print(f"[reap] WARN skip {c.slug}: {e}", flush=True)

    _safe_rebuild(ctx, rebuild_index)
    summary = (f"archived {len(cands)} note(s) — {n_sup} superseded, "
               f"{n_cold} cold ({ctx.label})")
    ok = Outcome(kind="reaped", reason="ok", detail=summary, exit_code=0)
    _print(ok)
    if args.commit:
        _safe_commit(ctx, autocommit, summary)
    return ok


def _safe_rebuild(ctx: ReapContext,
                  rebuild_index: Callable[[ReapContext], int]) -> None:
    try:
        n = rebuild_index(ctx)
        print(f"[reap] index rebuilt: {ctx.label} ({n} live notes)", flush=True)
    except Exception as e:  # noqa: BLE001 — derived index; never fatal
        print(f"[reap] WARN index rebuild failed (corpus intact): {e}", flush=True)


def _safe_commit(ctx: ReapContext,
                 autocommit: Callable[[ReapContext, str], str | None],
                 summary: str) -> None:
    try:
        c = autocommit(ctx, summary)
        if c:
            print(f"[reap] {c}", flush=True)
    except Exception as e:  # noqa: BLE001 — corpus already written, never fatal
        print(f"[reap] WARN auto-commit failed (corpus intact): {e}", flush=True)


def main(argv: list[str] | None = None) -> int:
    return run(argv).exit_code


if __name__ == "__main__":
    sys.exit(main())
