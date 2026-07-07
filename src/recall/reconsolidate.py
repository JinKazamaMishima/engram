#!/usr/bin/env python3
"""Corpus-wide re-consolidation — the weekly "sleep-time" maintenance pass.

Where ``curate`` distills ONE day's bundle, ``reconsolidate`` re-examines the
WHOLE standing corpus for ONE scope. The wrapper precomputes (the skill has no
Bash, so it can't run the embedder itself):
  - near-duplicate note pairs (cosine ≥ DUP_THRESHOLD),
  - missing-[[link]] candidates (similar notes that don't yet cross-link),
  - stale flags (notes untouched for a long time),
hands them to the ``/reconsolidate-memory`` skill, then validates + rebuilds +
scoped-commits exactly like curate. The skill MERGES duplicates by **superseding
in place** (it never deletes — the manifest validator requires every listed note
to exist on disk, and the action vocabulary is only created/updated) and adds the
missing cross-links.

Idempotent per (scope, ISO-week). Reuses curate's manifest validator, git commit,
and schema wholesale.

Exit codes: 0 — consolidated, or cleanly skipped; 1 — unexpected failure.

Usage:
    recall reconsolidate --scope global
    recall reconsolidate --scope project --project-dir /x/foo
    recall reconsolidate --scope global --dry-run | --commit | --force
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from recall import config, index
from recall.curate import (
    CLAUDE_BIN,
    CLAUDE_TIMEOUT_S,
    ET,
    Outcome,
    _et_clock,
    _git_commit_scoped,
    validate_manifest_against,
)
from recall.notify import notify_alert
from recall.schema import CurationManifest

_RECON_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit"]
# Thresholds are calibrated to Qwen3-Embedding's (compressed) cosine range — on a
# real 41-note corpus, max pairwise ≈ 0.79, p95 ≈ 0.61, median ≈ 0.43. So DUP at
# 0.80 only fires on a genuine near-twin (related notes top out ~0.79 and should
# be LINKED, not merged); LINK at 0.60 ≈ the p95 band of "related, maybe link it".
# Env-overridable for tuning as the corpus / model evolve.
DUP_THRESHOLD = float(os.environ.get("RECALL_RECON_DUP", "0.80"))
LINK_THRESHOLD = float(os.environ.get("RECALL_RECON_LINK", "0.60"))
K_NEIGHBORS = 5           # nearest neighbors examined per note
STALE_AGE_DAYS = 120      # last_updated older than this -> stale flag (advisory)


# ---- context -------------------------------------------------------------

@dataclass(frozen=True)
class ReconContext:
    scope: str            # "project" | "global"
    label: str            # scope id for paths/commits (project slug, or "global")
    corpus_dir: Path
    index_path: Path
    repo: Path            # git repo the scoped commit runs in
    target: date          # run date == manifest.date
    candidates_path: Path
    manifest_path: Path
    session_log_path: Path
    state_file: Path
    today_et: date


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="recall reconsolidate",
                                description=__doc__.splitlines()[0])
    p.add_argument("--scope", choices=("project", "global"), default="global")
    p.add_argument("--project-dir", type=Path, default=None,
                   help="project to reconsolidate (--scope project; default cwd)")
    p.add_argument("--date", type=str, default=None,
                   help="ISO run date (default: today ET) — becomes manifest.date")
    p.add_argument("--force", action="store_true",
                   help="re-run even if this ISO-week is already done")
    p.add_argument("--dry-run", action="store_true",
                   help="compute + print candidate counts; do not invoke claude")
    p.add_argument("--commit", action="store_true",
                   help="on success, scoped-commit the corpus ([reconsolidate])")
    return p.parse_args(argv)


def _recon_dir(label: str) -> Path:
    return config.data_root() / "reconsolidation" / label


def _resolve_context(args: argparse.Namespace, target: date,
                     today_et: date) -> ReconContext:
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
    rdir = _recon_dir(label)
    return ReconContext(
        scope=args.scope, label=label, corpus_dir=corpus_dir,
        index_path=config.index_path(label), repo=repo, target=target,
        candidates_path=rdir / "candidates" / f"{target.isoformat()}.json",
        manifest_path=rdir / "manifests" / f"{target.isoformat()}.json",
        session_log_path=rdir / "sessions" / f"{target.isoformat()}.md",
        state_file=rdir / "done.json", today_et=today_et)


# ---- idempotency (per ISO-week) ------------------------------------------

def _week_key(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _done_weeks(state_file: Path) -> set[str]:
    if not state_file.exists():
        return set()
    try:
        return set(json.loads(state_file.read_text()).get("weeks", []))
    except (json.JSONDecodeError, OSError):
        return set()


def _mark_done(state_file: Path, wk: str) -> None:
    weeks = _done_weeks(state_file)
    weeks.add(wk)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"weeks": sorted(weeks)}, indent=2))


# ---- candidate precompute (lazy torch) -----------------------------------

def _compute_candidates(ctx: ReconContext) -> dict:
    """All-pairs cosine over the scope's notes -> duplicate pairs, missing-link
    candidates, and stale flags. Lazy torch import AFTER the trivial-corpus guard;
    best-effort -> empty lists (a worklist hint, never a hard dependency)."""
    notes = [n for n, _ in index._load_notes(ctx.corpus_dir)]
    empty = {"scope": ctx.scope, "duplicate_pairs": [], "link_candidates": [],
             "stale": []}
    if len(notes) < 2:
        return empty
    try:
        import numpy as np

        from recall.index import _extract_links, best_embedder
        emb = best_embedder(alert_degraded=True)
        vecs = np.asarray(emb.embed([f"{n.description}\n\n{n.body}"
                                     for n in notes]), dtype="float32")
        sims = vecs @ vecs.T          # cosine (vectors are L2-normalized)
        known = {n.slug for n in notes}
        out_links = [set(_extract_links(n.body, known, exclude=n.slug))
                     for n in notes]
        dup_pairs: list[dict] = []
        link_cands: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for i, n in enumerate(notes):
            order = list(np.argsort(-sims[i]))
            for j in order[1:K_NEIGHBORS + 1]:
                j = int(j)
                a, b = n.slug, notes[j].slug
                key = tuple(sorted((a, b)))
                if key in seen:
                    continue
                seen.add(key)
                s = float(sims[i][j])
                if s >= DUP_THRESHOLD:
                    dup_pairs.append({"a": a, "b": b, "score": round(s, 3)})
                elif s >= LINK_THRESHOLD and b not in out_links[i] and a not in out_links[j]:
                    link_cands.append({"a": a, "b": b, "score": round(s, 3)})
        stale = _stale(notes, ctx.target)
    except Exception as e:  # noqa: BLE001 — worklist hint; never fail the run
        print(f"[reconsolidate] WARN candidate precompute failed: {e}", flush=True)
        return empty
    return {"scope": ctx.scope, "duplicate_pairs": dup_pairs,
            "link_candidates": link_cands, "stale": stale}


def _stale(notes, target: date) -> list[dict]:
    out: list[dict] = []
    for n in notes:
        lu = (n.last_updated or "").strip()
        if not lu:
            continue
        try:
            d = date.fromisoformat(lu[:10])
        except ValueError:
            continue
        age = (target - d).days
        if age >= STALE_AGE_DAYS:
            out.append({"slug": n.slug, "last_updated": lu, "age_days": age})
    return out


# ---- subprocess + materialization ----------------------------------------

def _build_env(ctx: ReconContext) -> dict[str, str]:
    return {
        **os.environ,
        "RECALL_RECON_CANDIDATES": str(ctx.candidates_path),
        "RECALL_RECON_MANIFEST": str(ctx.manifest_path),
        "RECALL_RECON_SCOPE": ctx.scope,
        "RECALL_RECON_DATE": ctx.target.isoformat(),
        "RECALL_RECON_CORPUS_DIR": str(ctx.corpus_dir),
    }


def _materialize(ctx: ReconContext, candidates: dict) -> None:
    config.ensure_dirs(ctx.corpus_dir, ctx.candidates_path.parent,
                       ctx.manifest_path.parent)
    ctx.candidates_path.write_text(json.dumps(candidates, indent=2))


def _invoke_claude(ctx: ReconContext, env: dict[str, str],
                   timeout_s: int = CLAUDE_TIMEOUT_S
                   ) -> subprocess.CompletedProcess[bytes]:
    """Run the reconsolidation skill scoped to the corpus's repo, granting the
    recall data root (candidates sidecar + manifest live there)."""
    return subprocess.run(
        [CLAUDE_BIN, "-p", "/reconsolidate-memory",
         "--add-dir", str(config.data_root()),
         "--allowedTools", *_RECON_ALLOWED_TOOLS],
        env=env, cwd=str(ctx.repo), timeout=timeout_s, check=False)


def _rebuild_index(ctx: ReconContext) -> int:
    """Rebuild the scope's derived index from its (now-consolidated) corpus."""
    from recall.index import best_embedder, build_index
    return build_index(ctx.corpus_dir, ctx.index_path,
                       best_embedder(alert_degraded=True))


def _autocommit(ctx: ReconContext, manifest: CurationManifest) -> str | None:
    paths = ["."] if ctx.scope == "global" else [str(ctx.corpus_dir)]
    return _git_commit_scoped(ctx.repo, paths, ctx.target, manifest.summary,
                              f"reconsolidate {ctx.label}")


# ---- session log ---------------------------------------------------------

def _append_session_block(ctx: ReconContext, outcome: Outcome,
                          manifest: CurationManifest | None) -> None:
    path = ctx.session_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    when = _et_clock()
    head = (f"# reconsolidate ({ctx.label}) — {ctx.target.isoformat()}\n"
            if not path.exists() else "")
    if outcome.kind == "curated" and manifest is not None:
        n_upd = sum(1 for n in manifest.notes if n.action == "updated")
        n_new = sum(1 for n in manifest.notes if n.action == "created")
        body = (f"\n## {when} — reconsolidated: ~{n_upd} updated / +{n_new} new\n"
                f"\n- {manifest.summary}\n"
                + "".join(f"- `{n.slug}` — {n.title}\n" for n in manifest.notes))
    else:
        body = f"\n## {when} — {outcome.kind} ({outcome.reason}): {outcome.detail}\n"
    path.write_text((path.read_text() if path.exists() else "") + head + body)


# ---- orchestration -------------------------------------------------------

def _print(o: Outcome) -> None:
    print(f"[reconsolidate] {o.kind}: {o.reason} — {o.detail}", flush=True)


def run(argv: list[str] | None = None, *,
        invoke_claude: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
        compute_candidates: Callable[[ReconContext], dict] | None = None,
        rebuild_index: Callable[[ReconContext], int] | None = None,
        autocommit: Callable[[ReconContext, CurationManifest], str | None] | None = None,
        today_et: date | None = None) -> Outcome:
    """Entry point. ``invoke_claude`` / ``compute_candidates`` / ``rebuild_index``
    / ``autocommit`` are injectable so tests swap the subprocess + the model work
    + git; ``today_et`` drives the run date."""
    invoke_claude = invoke_claude or _invoke_claude
    compute_candidates = compute_candidates or _compute_candidates
    rebuild_index = rebuild_index or _rebuild_index
    autocommit = autocommit or _autocommit

    args = _parse_args(argv)
    if today_et is None:
        today_et = datetime.now(timezone.utc).astimezone(ET).date()
    try:
        target = date.fromisoformat(args.date) if args.date else today_et
    except ValueError:
        o = Outcome(kind="failed", reason="bad_date",
                    detail=f"--date {args.date!r} is not an ISO date",
                    exit_code=1, alert_priority="urgent")
        _print(o)
        return o
    ctx = _resolve_context(args, target, today_et)

    wk = _week_key(target)
    if wk in _done_weeks(ctx.state_file) and not args.force:
        o = Outcome(kind="skipped", reason="already_reconsolidated",
                    detail=f"{wk} already reconsolidated for {ctx.label} "
                           f"(use --force)", exit_code=0)
        _print(o)
        return o
    if not ctx.corpus_dir.is_dir():
        o = Outcome(kind="skipped", reason="no_corpus",
                    detail=f"corpus dir absent: {ctx.corpus_dir}", exit_code=0)
        _print(o)
        return o

    cands = compute_candidates(ctx)
    n_dup = len(cands.get("duplicate_pairs", []))
    n_link = len(cands.get("link_candidates", []))
    n_stale = len(cands.get("stale", []))
    if not (n_dup or n_link or n_stale):
        o = Outcome(kind="skipped", reason="nothing_to_consolidate",
                    detail=f"{ctx.label}: no dup/link/stale candidates", exit_code=0)
        _print(o)
        _append_session_block(ctx, o, None)
        return o

    if args.dry_run:
        print(f"[reconsolidate] DRY-RUN {ctx.label} {ctx.target}: "
              f"{n_dup} dup pairs, {n_link} link candidates, {n_stale} stale",
              flush=True)
        return Outcome(kind="skipped", reason="dry_run",
                       detail="dry-run only; no LLM call", exit_code=0)

    _materialize(ctx, cands)
    print(f"[reconsolidate] invoking claude for {ctx.label} {ctx.target} "
          f"({n_dup} dup / {n_link} link / {n_stale} stale)", flush=True)
    try:
        cp = invoke_claude(ctx, _build_env(ctx), CLAUDE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        o = Outcome(kind="failed", reason="claude_timeout",
                    detail=f"claude did not return within {CLAUDE_TIMEOUT_S}s",
                    exit_code=1, alert_priority="urgent")
        _print(o); notify_alert(title=f"[reconsolidate] {o.reason}",
                                body=o.detail, priority="urgent")
        _append_session_block(ctx, o, None)
        return o
    except FileNotFoundError as e:
        o = Outcome(kind="failed", reason="claude_bin_missing",
                    detail=f"claude binary not found: {e}", exit_code=1,
                    alert_priority="urgent")
        _print(o); _append_session_block(ctx, o, None)
        return o

    if cp.returncode != 0:
        o = Outcome(kind="failed", reason="claude_nonzero",
                    detail=f"claude exited with code {cp.returncode}",
                    exit_code=1, alert_priority="urgent")
        _print(o); notify_alert(title=f"[reconsolidate] {o.reason}",
                                body=o.detail, priority="urgent")
        _append_session_block(ctx, o, None)
        return o

    manifest, fail = validate_manifest_against(
        ctx.manifest_path, ctx.target.isoformat(), lambda _s: ctx.corpus_dir)
    if fail is not None:
        _print(fail); notify_alert(title=f"[reconsolidate] {fail.reason}",
                                   body=fail.detail, priority="urgent")
        _append_session_block(ctx, fail, None)
        return fail

    _mark_done(ctx.state_file, wk)
    try:
        n = rebuild_index(ctx)
        print(f"[reconsolidate] index rebuilt: {ctx.label} ({n} notes)", flush=True)
    except Exception as e:  # noqa: BLE001 — derived index, never fatal
        print(f"[reconsolidate] WARN index rebuild failed (corpus intact): {e}",
              flush=True)

    ok = Outcome(kind="curated", reason="ok",
                 detail=manifest.summary if manifest else "", exit_code=0)
    _print(ok)
    _append_session_block(ctx, ok, manifest)
    if args.commit:
        try:
            c = autocommit(ctx, manifest)
            if c:
                print(f"[reconsolidate] {c}", flush=True)
        except Exception as e:  # noqa: BLE001 — corpus already written, never fatal
            print(f"[reconsolidate] WARN auto-commit failed (corpus intact): {e}",
                  flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    return run(argv).exit_code


if __name__ == "__main__":
    sys.exit(main())
