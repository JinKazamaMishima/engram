#!/usr/bin/env python3
"""Conversation-memory curator — nightly fire wrapper (owns all control flow).

For ONE project: discovers the day's Claude Code transcripts, denoises them to
clean human<->assistant prose, refuses if the day was already curated (unless
--force), invokes ``claude -p "/curate-memory"`` with a tight env contract,
validates the manifest + every note the skill wrote, rebuilds the derived
indices, and scoped-commits the corpora. The skill's only job is the
*distillation*: read the bundle, decide what's durable, and write notes to the
project corpus (``scope: project``) or the shared global/"soul" corpus
(``scope: global``) + one manifest.

Generalized from a production predecessor: no project-specific deps, project is
a parameter, and curation now writes to TWO corpora.

Exit codes:
   0 — curated, OR cleanly skipped (no conversations / already curated).
   1 — unexpected failure (claude nonzero/timeout, manifest/note invalid).

Usage:
    recall curate                       # today ET, cwd project
    recall curate --project-dir /x/foo --date 2026-06-01
    recall curate --force | --dry-run | --commit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from recall import config, dynamics
from recall import transcripts as T
from recall.notify import notify_alert
from recall.schema import (
    CurationManifest,
    CurationSchemaError,
    KnowledgeNote,
    set_frontmatter_keys,
)

CLAUDE_BIN = os.environ.get("RECALL_CLAUDE_BIN") or shutil.which("claude") or "claude"
CLAUDE_TIMEOUT_S = 900
# Curation is a bounded extraction/dedup task, not deep reasoning — run it on
# Sonnet 5 (the top Sonnet, 1M-context) at xhigh effort, which has far more
# headroom than Opus during peak hours so it rarely eats the 529 that pages the
# operator. Model + effort are env-overridable for A/B or rollback without a
# redeploy.
CURATE_MODEL = os.environ.get("RECALL_CURATE_MODEL", "claude-sonnet-5")
CURATE_EFFORT = os.environ.get("RECALL_CURATE_EFFORT", "xhigh")
ET = ZoneInfo("America/New_York")
_CURATE_ALLOWED_TOOLS = ["Read", "Glob", "Grep", "Write", "Edit"]
_SIMILAR_QUERY_MAX_CHARS = 8000   # cap the bundle text embedded for dedup lookup

# A `claude` failure whose output matches this is a TRANSIENT upstream capacity
# signal (HTTP 429/529/5xx, "Overloaded", rate-limit) — the kind a 529 literally
# tells you to "try again in a moment" for. run_claude_with_backoff rides these
# out; every other nonzero exit (bad flag, auth, missing bin) is returned/raised
# at once so a real fault still fails fast and still pages.
_TRANSIENT_RE = re.compile(
    r"(?:\b(?:429|529|502|503|504)\b|overload|rate[ _-]?limit"
    r"|service[ _-]?unavailable|temporarily unavailable)", re.IGNORECASE)


def _emit(data: bytes, stream) -> None:
    """Write captured child output to the parent stream so journalctl still shows
    the skill's reasoning even though we capture to inspect it. Robust under
    pytest, which may swap sys.stdout for an object with no .buffer."""
    if not data:
        return
    try:
        stream.buffer.write(data)
        stream.flush()
    except (AttributeError, ValueError):
        stream.write(data.decode("utf-8", "replace"))
        stream.flush()


def run_claude_with_backoff(
    argv: list[str], *, env: dict[str, str], cwd: str, timeout: int,
    attempts: int = 4, base_delay: float = 5.0, max_delay: float = 60.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> subprocess.CompletedProcess[bytes]:
    """Run `claude` and transparently retry ONLY the transient-overload class
    (see _TRANSIENT_RE) with exponential backoff + jitter, so the nightly memory
    cycle rides out an Anthropic capacity blip instead of failing and paging the
    operator. Non-transient nonzero exits return immediately (no wasted retries,
    real faults still fail fast). TimeoutExpired / FileNotFoundError propagate to
    the caller's existing handlers — a timeout or a missing binary is not a thing
    a retry fixes. Output is captured to be inspected, then re-emitted each
    attempt so the journal keeps the skill's stdout/stderr."""
    last: subprocess.CompletedProcess[bytes] | None = None
    for attempt in range(1, attempts + 1):
        cp = runner(argv, env=env, cwd=cwd, timeout=timeout,
                    check=False, capture_output=True)
        _emit(cp.stdout, sys.stdout)
        _emit(cp.stderr, sys.stderr)
        last = cp
        if cp.returncode == 0:
            return cp
        blob = ((cp.stdout or b"") + b"\n" + (cp.stderr or b"")).decode(
            "utf-8", "replace")
        if attempt < attempts and _TRANSIENT_RE.search(blob):
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += jitter() * base_delay   # decorrelate concurrent retries
            print(f"[claude] transient upstream error (attempt {attempt}/{attempts}); "
                  f"retrying in {delay:.1f}s", flush=True)
            sleep(delay)
            continue
        return cp
    return last  # type: ignore[return-value]  # attempts >= 1, so last is set


# ---- result / outcome types ----------------------------------------------

@dataclass(frozen=True)
class FireContext:
    target: date
    project_dir: Path
    slug: str
    transcript_dir: Path
    project_knowledge_dir: Path
    global_dir: Path
    bundle_text: str
    bundle_stats: T.BundleStats
    bundle_path: Path
    manifest_path: Path
    neighbors_path: Path
    project_index_path: Path
    global_index_path: Path
    session_log_path: Path
    state_file: Path
    state_bucket: str          # "dates" (nightly sweep) | "sessions" (--session)
    state_key: str             # idempotency key within that bucket
    today_et: date
    # Incremental mode only: ISO ts of the newest exchange in this pass — the
    # value the watermark advances to at the success point. "" = non-incremental.
    watermark_advance: str = ""


@dataclass(frozen=True)
class Outcome:
    kind: str           # "curated" | "skipped" | "failed"
    reason: str
    detail: str
    exit_code: int
    alert_priority: str | None = None


# ---- argument parsing ----------------------------------------------------

PROVISIONAL_HINT = (
    "> PROVISIONAL PASS — this conversation may still be OPEN (mid-session "
    "eviction or shutdown). Facts below can still reverse: write them with "
    "`confidence: 0.3` unless clearly durable, leave `valid_to` empty, and "
    "prefer raising an existing provisional note's confidence over minting a "
    "duplicate.\n\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="recall curate",
                                description=__doc__.splitlines()[0])
    p.add_argument("--project-dir", type=Path, default=None,
                   help="project to curate (default: cwd)")
    p.add_argument("--date", type=str, default=None,
                   help="ISO date to curate (default: today ET)")
    p.add_argument("--session", type=str, default=None,
                   help="curate ONE session by id (<id>.jsonl) instead of a whole "
                        "day; idempotency tracked per-session (ignores --date)")
    p.add_argument("--buffer", type=Path, default=None,
                   help="curate an Engram LiveBuffer JSONL instead of a Claude Code "
                        "transcript; unit id = filename stem, tracked in the "
                        "sessions bucket (overrides --session/--date)")
    p.add_argument("--force", action="store_true",
                   help="re-curate even if this date is already curated")
    p.add_argument("--dry-run", action="store_true",
                   help="build the bundle + print stats; do not invoke claude")
    p.add_argument("--transcript-dir", type=Path, default=None,
                   help="override the Claude Code transcript dir (tests)")
    p.add_argument("--commit", action="store_true",
                   help="on success, scoped-commit the corpora ([curator], no push)")
    p.add_argument("--provisional", action="store_true",
                   help="live pass over a possibly-still-open conversation: hint "
                        "the curator toward low-confidence facts and do NOT mark "
                        "the unit curated (the canonical pass can rerun it)")
    p.add_argument("--incremental", action="store_true",
                   help="curate only the tail after this conversation's watermark "
                        "(requires --session or --buffer); on success the watermark "
                        "advances — buckets are never written, the watermark IS the "
                        "state. --force ignores + rewrites the watermark")
    p.add_argument("--until", type=str, default=None,
                   help="upper slice bound (ISO ts, inclusive) for --incremental — "
                        "the harness passes the cooled edge so still-hot turns stay "
                        "uncurated")
    return p.parse_args(argv)


# ---- idempotency state (per project) --------------------------------------

def _curated_keys(state_file: Path, bucket: str = "dates") -> set[str]:
    """Keys already curated in ``bucket``: "dates" for the nightly day-sweep,
    "sessions" for per-session curation. The buckets are independent so live and
    nightly curation never clobber each other's idempotency state."""
    if not state_file.exists():
        return set()
    try:
        return set(json.loads(state_file.read_text()).get(bucket, []))
    except (json.JSONDecodeError, OSError):
        return set()


def _mark_curated(state_file: Path, bucket: str, key: str) -> None:
    """Record ``key`` in ``bucket``, preserving every other bucket already on
    disk (so marking a session never drops the recorded dates, and vice versa)."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(state_file.read_text()) if state_file.exists() else {}
        if not isinstance(data, dict):
            data = {}
    except (json.JSONDecodeError, OSError):
        data = {}
    data[bucket] = sorted(set(data.get(bucket, [])) | {key})
    state_file.write_text(json.dumps(data, indent=2))


def _read_watermark(state_file: Path, convo_id: str) -> datetime | None:
    """The per-conversation incremental watermark: everything with
    ``ts <= watermark`` has already been curated. ``None`` = never watermarked
    (a first incremental pass curates the whole conversation — at-least-once;
    the curator's dedup absorbs any overlap with a past canonical pass)."""
    if not state_file.exists():
        return None
    try:
        raw = json.loads(state_file.read_text()).get("watermarks", {})
        raw = raw.get(convo_id) if isinstance(raw, dict) else None
    except (json.JSONDecodeError, OSError):
        return None
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _mark_watermark(state_file: Path, convo_id: str, iso_ts: str) -> None:
    """Advance ``convo_id``'s watermark, preserving every other bucket (same
    read-modify-write discipline as ``_mark_curated``). Called ONLY at the
    post-validation success point — a failed pass never advances, so no tail
    is ever silently lost."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(state_file.read_text()) if state_file.exists() else {}
        if not isinstance(data, dict):
            data = {}
    except (json.JSONDecodeError, OSError):
        data = {}
    marks = data.get("watermarks", {})
    if not isinstance(marks, dict):
        marks = {}
    marks[convo_id] = iso_ts
    data["watermarks"] = marks
    state_file.write_text(json.dumps(data, indent=2))


# ---- context resolution --------------------------------------------------

def _resolve_context(args: argparse.Namespace, today_et: date,
                     target: date) -> FireContext | Outcome:
    project_dir = (Path(args.project_dir).resolve() if args.project_dir
                   else Path.cwd())
    slug = config.project_slug(project_dir)
    transcript_dir = (args.transcript_dir if args.transcript_dir is not None
                      else T.project_transcript_dir(project_dir))
    cdir = config.curation_dir() / slug
    state_file = cdir / "curated.json"

    # Unit of work: a whole day (nightly sweep), one session (live --session), or
    # one Engram LiveBuffer (eviction --buffer; shares the sessions bucket + stem —
    # the buffer's convo id IS the SDK session id, so the nightly sweep and live
    # eviction see one another's state). Buckets stay independent of dates.
    if args.buffer:
        sid = args.buffer.stem
        bucket, state_key, stem = "sessions", sid, f"session-{sid}"
    elif args.session:
        bucket, state_key, stem = "sessions", args.session, f"session-{args.session}"
    else:
        bucket, state_key, stem = "dates", target.isoformat(), target.isoformat()

    # Incremental slicing window: since = this convo's watermark (strictly
    # after — the watermarked exchange was already curated), until = the
    # caller's cooled edge (inclusive).
    if args.incremental and not (args.buffer or args.session):
        return Outcome(kind="failed", reason="bad_flags",
                       detail="--incremental requires --session or --buffer "
                              "(the watermark is per-conversation)",
                       exit_code=1, alert_priority="urgent")
    if args.until and not args.incremental:
        return Outcome(kind="failed", reason="bad_flags",
                       detail="--until only applies with --incremental",
                       exit_code=1, alert_priority="urgent")
    until: datetime | None = None
    if args.until:
        try:
            until = datetime.fromisoformat(args.until.replace("Z", "+00:00"))
        except ValueError:
            return Outcome(kind="failed", reason="bad_until",
                           detail=f"--until {args.until!r} is not an ISO "
                                  f"timestamp", exit_code=1,
                           alert_priority="urgent")
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
    since = (_read_watermark(state_file, state_key)
             if args.incremental and not args.force else None)

    # Whole-unit idempotency (buckets) gates canonical passes only. Incremental
    # passes answer to the watermark alone: a bucket-marked session that GREW
    # after a resume must still get its tail curated — the first incremental
    # pass on a legacy session re-reads it whole (at-least-once; the curator's
    # dedup absorbs the overlap) and the watermark takes over from there.
    if (not args.incremental and not args.force
            and state_key in _curated_keys(state_file, bucket)):
        return Outcome(kind="skipped", reason="already_curated",
                       detail=f"{state_key} already curated for {slug} "
                              f"(use --force to redo)", exit_code=0)

    watermark_advance = ""
    if args.buffer:
        path = args.buffer
        if not path.exists():
            return Outcome(kind="skipped", reason="buffer_missing",
                           detail=f"no buffer {path} for {slug}", exit_code=0)
        last = T.buffer_last_ts(path, until=until)
        target = (last.astimezone(T.ET).date() if last else today_et)
        if args.incremental and last is not None:
            watermark_advance = last.isoformat()
        paths = [path]
        bundle_text, stats = T.build_buffer_bundle(path, since=since,
                                                   until=until)
    elif args.session:
        path = T.session_transcript_path(transcript_dir, args.session)
        if not path.exists():
            return Outcome(kind="skipped", reason="session_missing",
                           detail=f"no transcript {path.name} for {slug} "
                                  f"in {transcript_dir}", exit_code=0)
        target = T.session_date(path) or today_et   # file it under the session's day
        if args.incremental:
            last = T.transcript_last_ts(path, until=until)
            if last is not None:
                watermark_advance = last.isoformat()
        paths = [path]
        bundle_text, stats = T.build_bundle(paths, None, since=since,
                                            until=until)
    else:
        paths = T.discover_transcripts(transcript_dir, target)
        bundle_text, stats = T.build_bundle(paths, target)

    if stats.exchanges == 0:
        if args.incremental and since is not None:
            return Outcome(kind="skipped", reason="no_new_exchanges",
                           detail=f"nothing after watermark "
                                  f"{since.isoformat()} for {slug} "
                                  f"({state_key})", exit_code=0)
        return Outcome(kind="skipped", reason="no_conversations",
                       detail=f"no human conversation for {slug} "
                              f"({state_key}; {len(paths)} transcript(s) scanned)",
                       exit_code=0)

    return FireContext(
        target=target, project_dir=project_dir, slug=slug,
        transcript_dir=transcript_dir,
        project_knowledge_dir=config.project_corpus_dir(project_dir),
        global_dir=config.global_corpus_dir(),
        bundle_text=bundle_text, bundle_stats=stats,
        bundle_path=cdir / "bundles" / f"{stem}.md",
        manifest_path=cdir / "manifests" / f"{stem}.json",
        neighbors_path=cdir / "neighbors" / f"{stem}.json",
        project_index_path=config.index_path(slug),
        global_index_path=config.index_path(config.GLOBAL_SCOPE),
        session_log_path=cdir / "sessions" / f"{target.isoformat()}.md",
        state_file=state_file, state_bucket=bucket, state_key=state_key,
        today_et=today_et, watermark_advance=watermark_advance)


# ---- env contract + input materialization ---------------------------------

def _build_env(ctx: FireContext) -> dict[str, str]:
    return {
        **os.environ,
        "RECALL_CURATE_INPUT": str(ctx.bundle_path),
        "RECALL_CURATE_DATE": ctx.target.isoformat(),
        "RECALL_CURATE_SESSION": ctx.state_key if ctx.state_bucket == "sessions" else "",
        "RECALL_CURATE_MANIFEST": str(ctx.manifest_path),
        "RECALL_CURATE_NEIGHBORS": str(ctx.neighbors_path),
        "RECALL_PROJECT_KNOWLEDGE_DIR": str(ctx.project_knowledge_dir),
        "RECALL_GLOBAL_DIR": str(ctx.global_dir),
        "RECALL_PROJECT_SLUG": ctx.slug,
    }


def _materialize_inputs(ctx: FireContext, neighbors: list[dict]) -> None:
    """Write the bundle the skill reads, the precomputed dedup-neighbors sidecar,
    and a state sidecar; ensure all output dirs exist before claude runs."""
    config.ensure_dirs(ctx.project_knowledge_dir, ctx.global_dir,
                       ctx.bundle_path.parent, ctx.manifest_path.parent,
                       ctx.neighbors_path.parent)
    ctx.bundle_path.write_text(ctx.bundle_text)
    ctx.neighbors_path.write_text(json.dumps({
        "date": ctx.target.isoformat(),
        "query_chars": min(len(ctx.bundle_text), _SIMILAR_QUERY_MAX_CHARS),
        "neighbors": neighbors,
    }, indent=2))
    sidecar = config.curation_dir() / ctx.slug / ".next.json"
    sidecar.write_text(json.dumps({
        "date": ctx.target.isoformat(),
        "project_slug": ctx.slug,
        "bundle_path": str(ctx.bundle_path),
        "project_knowledge_dir": str(ctx.project_knowledge_dir),
        "global_dir": str(ctx.global_dir),
        "manifest_path": str(ctx.manifest_path),
        "fire_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, sort_keys=True))


# ---- subprocess invocation (extracted for tests) -------------------------

def _invoke_claude(ctx: FireContext, env: dict[str, str],
                   timeout_s: int = CLAUDE_TIMEOUT_S
                   ) -> subprocess.CompletedProcess[bytes]:
    """Run the curator skill as a scoped, non-interactive agent from the
    project dir (so the project corpus is in cwd). The recall data root is
    granted via --add-dir so the skill can read the bundle and write global
    notes + the manifest, which live outside cwd."""
    return run_claude_with_backoff(
        [CLAUDE_BIN, "-p", "/curate-memory",
         "--model", CURATE_MODEL, "--effort", CURATE_EFFORT,
         "--add-dir", str(config.data_root()),
         "--allowedTools", *_CURATE_ALLOWED_TOOLS],
        env=env, cwd=str(ctx.project_dir), timeout=timeout_s)


def _note_dir(ctx: FireContext, scope: str) -> Path:
    return ctx.global_dir if scope == "global" else ctx.project_knowledge_dir


def _compute_neighbors(ctx: FireContext) -> list[dict]:
    """Nearest existing notes to today's bundle (passage-side), fused over the
    project + global indices, so the curator can UPDATE a near-match instead of
    writing a twin (Mem0's "retrieve similar before deciding ADD/UPDATE"). Lazy
    torch import AFTER the no-index guard; best-effort -> [] (a dedup hint, never
    a hard dependency — a cold model or missing index must not fail curation)."""
    scopes = [(ctx.slug, ctx.project_index_path),
              (config.GLOBAL_SCOPE, ctx.global_index_path)]
    if not any(Path(p).exists() for _, p in scopes):
        return []
    try:
        from recall.index import best_embedder, search_corpora
        query = ctx.bundle_text[:_SIMILAR_QUERY_MAX_CHARS]
        vec = best_embedder(alert_degraded=True).embed([query], is_query=False)[0]
        hits = search_corpora(scopes, query, query_vector=vec, k=12)
    except Exception as e:  # noqa: BLE001 — dedup hint; never fail the run
        print(f"[curate] WARN neighbor precompute failed: {e}", flush=True)
        return []
    return [{"slug": h.slug, "scope": h.corpus, "description": h.description,
             "score": round(h.score, 4)} for h in hits]


def _compute_surprise(ctx: FireContext,
                      created: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Max cosine similarity of each newly-created note to the PRIOR corpus (the
    notes NOT created today), per scope — the novelty signal σ = 1 − max_sim that
    sets a note's *birth* stability (the flashbulb route to durability). Mirrors
    reconsolidate's all-pairs cosine. Lazy torch AFTER the trivial guard;
    best-effort -> {} (a cold model or tiny corpus just means new notes fall back
    to default birth stability — surprise is a nice-to-have, never a hard dep)."""
    by_scope: dict[str, set[str]] = {}
    for slug, scope in created:
        by_scope.setdefault(scope, set()).add(slug)
    out: dict[tuple[str, str], float] = {}

    # Torch-free first pass: load+parse each scope's notes (no model) and decide
    # which scopes actually have a PRIOR corpus to compare against. A brand-new
    # corpus (the new notes are all there is) is maximally surprised by definition
    # and needs no embedding — so the model is loaded ONLY when there's real work.
    from recall.index import _load_notes
    todo: dict[str, list] = {}
    for scope, new_slugs in by_scope.items():
        notes = [n for n, _ in _load_notes(_note_dir(ctx, scope))]
        if any(n.slug not in new_slugs for n in notes):
            todo[scope] = notes
        else:
            for s in new_slugs:        # nothing prior to be unlike
                out[(scope, s)] = 0.0
    if not todo:
        return out

    try:
        import numpy as np

        from recall.index import best_embedder
        emb = best_embedder(alert_degraded=True)
        for scope, notes in todo.items():
            new_slugs = by_scope[scope]
            vecs = np.asarray(emb.embed([f"{n.description}\n\n{n.body}"
                                         for n in notes]), dtype="float32")
            idx = {n.slug: i for i, n in enumerate(notes)}
            prior_idx = [i for i, n in enumerate(notes) if n.slug not in new_slugs]
            for s in new_slugs:
                i = idx.get(s)
                out[(scope, s)] = (float(max(float(vecs[i] @ vecs[j])
                                             for j in prior_idx))
                                   if i is not None else 0.0)
    except Exception as e:  # noqa: BLE001 — birth-stability hint; never fail the run
        print(f"[curate] WARN surprise precompute failed: {e}", flush=True)
        for scope, new_slugs in by_scope.items():
            for s in new_slugs:
                out.setdefault((scope, s), 0.0)
    return out


def _set_birth_stability(ctx: FireContext, manifest: CurationManifest,
                         compute_surprise) -> int:
    """Give each newly-created note its initial stability (Phase III). Surprising
    notes (far from everything we already know) are born more durable; the soul's
    permanent core (``kind: identity``/``achievement``) is born at graduation with
    a max importance anchor so it never decays and resists overwrite. Dynamic
    fields already set (e.g. a manual seed like the Engram note) are never clobbered.
    Runs before the index rebuild so the fresh stability is indexed + committed."""
    created = [(n.slug, n.scope) for n in manifest.notes if n.action == "created"]
    if not created:
        return 0
    surprises = compute_surprise(ctx, created)
    n_set = 0
    for slug, scope in created:
        path = _note_dir(ctx, scope) / f"{slug}.md"
        if not path.exists():
            continue
        try:
            note = KnowledgeNote.parse(path.read_text(), expect_slug=slug)
        except CurationSchemaError:
            continue
        if note.stability and note.stability > 0:
            continue  # already carries dynamics (manual seed) — don't clobber
        updates: dict[str, object] = {"last_used": ctx.target.isoformat(), "uses": 0}
        if note.kind in ("identity", "achievement"):
            updates["stability"] = round(dynamics.S_PERM, 1)   # born permanent
            updates["surprise"] = 0.9
            updates["importance"] = 1.0
        else:
            sigma = dynamics.surprise_from_similarity(surprises.get((scope, slug), 0.0))
            updates["stability"] = round(dynamics.initial_stability(sigma), 3)
            updates["surprise"] = round(sigma, 3)
        try:
            path.write_text(set_frontmatter_keys(path.read_text(), updates))
            n_set += 1
        except (CurationSchemaError, OSError) as e:
            print(f"[curate] WARN birth-stability skip {slug}: {e}", flush=True)
    return n_set


def _stamp_supersession_validity(ctx: FireContext,
                                 manifest: CurationManifest) -> int:
    """Belt-and-suspenders for temporal validity (Brick 3): any note this run
    touched that now carries ``superseded_by`` must also carry ``valid_to`` —
    the moment a fact is superseded is the moment it stopped being true, and a
    reversed decision without a ``valid_to`` silently injects as current. The
    skill contract asks the curator to stamp it; this backstop catches misses.
    An existing ``valid_to`` is NEVER overwritten (the curator may know the
    real reversal date better than the filing date). Runs before the index
    rebuild so the stamp is indexed + committed with the run."""
    n_set = 0
    for edit in manifest.notes:
        path = _note_dir(ctx, edit.scope) / f"{edit.slug}.md"
        if not path.exists():
            continue
        try:
            note = KnowledgeNote.parse(path.read_text(), expect_slug=edit.slug)
        except CurationSchemaError:
            continue
        if not note.superseded_by or note.valid_to:
            continue
        try:
            path.write_text(set_frontmatter_keys(
                path.read_text(), {"valid_to": ctx.target.isoformat()}))
            n_set += 1
        except (CurationSchemaError, OSError) as e:
            print(f"[curate] WARN validity stamp skip {edit.slug}: {e}",
                  flush=True)
    return n_set


def _rebuild_indices(ctx: FireContext) -> dict[str, int]:
    """Rebuild the derived project + global indices from the canonical corpora.
    Embeds via the WARM DAEMON when it's up: rebuilds used to load a second
    in-process model while the daemon held the GPU — every one OOM'd, silently,
    freezing production recall on a stale index (2026-07-04..06). In-process is
    now only the daemon-down fallback (the GPU is free then). Lazy imports keep
    torch out of the wrapper's import path (and the tests)."""
    from recall.index import best_embedder, build_index
    emb = best_embedder(alert_degraded=True)
    print(f"[curate] index embeddings via {type(emb).__name__}", flush=True)
    out = {ctx.slug: build_index(ctx.project_knowledge_dir,
                                 ctx.project_index_path, emb)}
    out[config.GLOBAL_SCOPE] = build_index(ctx.global_dir,
                                           ctx.global_index_path, emb)
    return out


def _git_commit_scoped(repo: Path, paths: list[str], target: date,
                       summary: str, label: str) -> str | None:
    """Scoped, best-effort commit on the CURRENT branch (no push). Only the
    given paths are staged. Returns the commit subject, or None if nothing
    changed / not a git repo."""
    if not (repo / ".git").exists():
        return None
    subprocess.run(["git", "-C", str(repo), "add", "--", *paths],
                   check=True, capture_output=True, text=True)
    staged = subprocess.run(["git", "-C", str(repo), "diff", "--cached",
                             "--quiet", "--", *paths])
    if staged.returncode == 0:
        return None
    line = summary.strip().splitlines()[0][:120] if summary.strip() else "curation"
    msg = (f"[curator] {target.isoformat()} ({label}): {line}\n\n"
           f"Automated by recall curate (/curate-memory).")
    subprocess.run(["git", "-C", str(repo), "commit", "-m", msg],
                   check=True, capture_output=True, text=True)
    return msg.splitlines()[0]


def _autocommit(ctx: FireContext, manifest: CurationManifest) -> list[str]:
    """Commit project notes in the project repo and global notes in the global
    repo — two scoped, no-push commits. Global is machine-local (never pushed)."""
    scopes = {n.scope for n in manifest.notes}
    done: list[str] = []
    if "project" in scopes:
        c = _git_commit_scoped(ctx.project_dir,
                               [str(ctx.project_knowledge_dir)], ctx.target,
                               manifest.summary, ctx.slug)
        if c:
            done.append(c)
    if "global" in scopes:
        c = _git_commit_scoped(ctx.global_dir, ["."], ctx.target,
                               manifest.summary, "global")
        if c:
            done.append(c)
    return done


# ---- post-run validation -------------------------------------------------

def validate_manifest_against(
        manifest_path: Path, target_iso: str,
        dir_for_scope: Callable[[str], Path],
) -> tuple[CurationManifest | None, Outcome | None]:
    """Validate a manifest + cross-check every note it references on disk, in the
    dir its scope routes to (via ``dir_for_scope``). Shared by curate and
    reconsolidate so both get identical structural guarantees + error reasons.
    ``(manifest, None)`` on success; ``(None, failed Outcome)`` on any error."""
    if not manifest_path.exists():
        return None, Outcome(kind="failed", reason="no_manifest_written",
                             detail=f"claude exited but {manifest_path.name} "
                                    f"was not created", exit_code=1,
                             alert_priority="urgent")
    try:
        manifest = CurationManifest.from_json(manifest_path.read_text())
    except CurationSchemaError as e:
        return None, Outcome(kind="failed", reason="manifest_schema_error",
                             detail=f"manifest validation failed: {e}",
                             exit_code=1, alert_priority="urgent")
    if manifest.date != target_iso:
        return None, Outcome(kind="failed", reason="manifest_date_mismatch",
                             detail=f"manifest.date={manifest.date!r} but wrapper "
                                    f"expected {target_iso!r}",
                             exit_code=1, alert_priority="urgent")
    for edit in manifest.notes:
        note_path = dir_for_scope(edit.scope) / f"{edit.slug}.md"
        if not note_path.exists():
            return None, Outcome(kind="failed", reason="note_missing",
                                 detail=f"manifest lists {edit.slug!r} "
                                        f"({edit.action}, {edit.scope}) but "
                                        f"{note_path} is absent", exit_code=1,
                                 alert_priority="urgent")
        try:
            KnowledgeNote.parse(note_path.read_text(), expect_slug=edit.slug)
        except CurationSchemaError as e:
            return None, Outcome(kind="failed", reason="note_schema_error",
                                 detail=f"note {edit.slug!r} malformed: {e}",
                                 exit_code=1, alert_priority="urgent")
    return manifest, None


def _validate_manifest(ctx: FireContext
                       ) -> tuple[CurationManifest | None, Outcome | None]:
    """Curate's manifest validation — delegates to the shared validator with the
    curate scope→dir routing."""
    return validate_manifest_against(
        ctx.manifest_path, ctx.target.isoformat(),
        lambda scope: _note_dir(ctx, scope))


# ---- session log writing -------------------------------------------------

def _et_clock(when: datetime | None = None) -> str:
    return (when or datetime.now(timezone.utc)).astimezone(ET).strftime("%H:%M ET")


def _append_session_block(path: Path, target: date, slug: str, outcome: Outcome,
                          ctx: FireContext | None,
                          manifest: CurationManifest | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"# curator ({slug}) — {target.isoformat()}\n")
    when = _et_clock()
    lines: list[str] = ["\n"]
    if outcome.kind == "curated" and manifest is not None and ctx is not None:
        n_new = sum(1 for n in manifest.notes if n.action == "created")
        n_upd = sum(1 for n in manifest.notes if n.action == "updated")
        n_glob = sum(1 for n in manifest.notes if n.scope == "global")
        scope_tag = (f" [session {ctx.state_key[:8]}]"
                     if ctx.state_bucket == "sessions" else "")
        lines.append(f"## {when} — curated {target.isoformat()}{scope_tag}: "
                     f"+{n_new} new / ~{n_upd} updated ({n_glob} global) "
                     f"({ctx.bundle_stats.sessions} sessions, "
                     f"{ctx.bundle_stats.exchanges} turns)\n")
        lines.append(f"\n- Summary: {manifest.summary}\n")
        for n in manifest.notes:
            mark = "+" if n.action == "created" else "~"
            tag = " [global]" if n.scope == "global" else ""
            lines.append(f"- {mark} `{n.slug}`{tag} — {n.title}\n")
    elif outcome.kind == "skipped":
        lines.append(f"## {when} — skipped ({outcome.reason}): {outcome.detail}\n")
    else:
        lines.append(f"## {when} — failed ({outcome.reason}): {outcome.detail}\n")
    path.write_text(path.read_text() + "".join(lines))


# ---- main orchestration --------------------------------------------------

def _alert_on_failure(outcome: Outcome) -> None:
    if outcome.alert_priority is None:
        return
    notify_alert(title=f"[recall curate] {outcome.reason}", body=outcome.detail,
                 priority=outcome.alert_priority)


def _print_outcome(o: Outcome) -> None:
    print(f"[curate] {o.kind}: {o.reason} — {o.detail}", flush=True)


def run(argv: list[str] | None = None, *,
        invoke_claude: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
        rebuild_indices: Callable[[FireContext], dict[str, int]] | None = None,
        autocommit: Callable[[FireContext, CurationManifest], list[str]] | None = None,
        compute_neighbors: Callable[[FireContext], list[dict]] | None = None,
        compute_surprise: Callable[..., dict] | None = None,
        today_et: date | None = None) -> Outcome:
    """Entry point. ``invoke_claude`` / ``rebuild_indices`` / ``autocommit`` /
    ``compute_neighbors`` / ``compute_surprise`` are injectable so tests swap the
    subprocess + the model-loading index build + git + the dedup-neighbor lookup +
    the birth-stability novelty pass; ``today_et`` is injectable so tests drive the
    target date."""
    invoke_claude = invoke_claude or _invoke_claude
    rebuild_indices = rebuild_indices or _rebuild_indices
    autocommit = autocommit or _autocommit
    compute_neighbors = compute_neighbors or _compute_neighbors
    compute_surprise = compute_surprise or _compute_surprise

    args = _parse_args(argv)
    if today_et is None:
        today_et = datetime.now(timezone.utc).astimezone(ET).date()
    try:
        target = date.fromisoformat(args.date) if args.date else today_et
    except ValueError:
        o = Outcome(kind="failed", reason="bad_date",
                    detail=f"--date {args.date!r} is not an ISO date",
                    exit_code=1, alert_priority="urgent")
        _print_outcome(o); _alert_on_failure(o)
        return o

    resolved = _resolve_context(args, today_et, target)
    if isinstance(resolved, Outcome):
        _print_outcome(resolved); _alert_on_failure(resolved)
        if resolved.reason != "already_curated":
            slug = config.project_slug(args.project_dir or Path.cwd())
            _append_session_block(
                config.curation_dir() / slug / "sessions" / f"{target.isoformat()}.md",
                target, slug, resolved, None, None)
        return resolved
    ctx = resolved
    if args.provisional:
        # A provisional bundle announces itself: the curator writes volatile
        # facts at confidence ~0.3 instead of asserting them as settled.
        ctx = replace(ctx, bundle_text=PROVISIONAL_HINT + ctx.bundle_text)
    env = _build_env(ctx)

    if args.dry_run:
        print(f"[curate] DRY-RUN {ctx.slug} {ctx.target}: "
              f"{ctx.bundle_stats.sessions} sessions, "
              f"{ctx.bundle_stats.exchanges} turns, {ctx.bundle_stats.chars} chars",
              flush=True)
        print(f"  project corpus = {ctx.project_knowledge_dir}", flush=True)
        print(f"  global corpus  = {ctx.global_dir}", flush=True)
        print(f"  bundle         = {ctx.bundle_path}", flush=True)
        return Outcome(kind="skipped", reason="dry_run",
                       detail="dry-run only; no LLM call", exit_code=0)

    _materialize_inputs(ctx, compute_neighbors(ctx))

    print(f"[curate] invoking claude for {ctx.slug} {ctx.target.isoformat()} "
          f"({ctx.bundle_stats.sessions} sessions, "
          f"{ctx.bundle_stats.exchanges} turns)", flush=True)
    try:
        cp = invoke_claude(ctx, env, CLAUDE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        o = Outcome(kind="failed", reason="claude_timeout",
                    detail=f"claude did not return within {CLAUDE_TIMEOUT_S}s",
                    exit_code=1, alert_priority="urgent")
        _print_outcome(o); _alert_on_failure(o)
        _append_session_block(ctx.session_log_path, target, ctx.slug, o, ctx, None)
        return o
    except FileNotFoundError as e:
        o = Outcome(kind="failed", reason="claude_bin_missing",
                    detail=f"claude binary not found: {e}", exit_code=1,
                    alert_priority="urgent")
        _print_outcome(o); _alert_on_failure(o)
        _append_session_block(ctx.session_log_path, target, ctx.slug, o, ctx, None)
        return o

    if cp.returncode != 0:
        o = Outcome(kind="failed", reason="claude_nonzero",
                    detail=f"claude exited with code {cp.returncode}",
                    exit_code=1, alert_priority="urgent")
        _print_outcome(o); _alert_on_failure(o)
        _append_session_block(ctx.session_log_path, target, ctx.slug, o, ctx, None)
        return o

    manifest, fail = _validate_manifest(ctx)
    if fail is not None:
        _print_outcome(fail); _alert_on_failure(fail)
        _append_session_block(ctx.session_log_path, target, ctx.slug, fail, ctx, None)
        return fail

    try:
        n_birth = _set_birth_stability(ctx, manifest, compute_surprise)
        if n_birth:
            print(f"[curate] birth-stability set on {n_birth} new note(s)", flush=True)
    except Exception as e:  # noqa: BLE001 — corpus already written, never fatal
        print(f"[curate] WARN birth-stability failed (corpus intact): {e}", flush=True)

    try:
        n_stamp = _stamp_supersession_validity(ctx, manifest)
        if n_stamp:
            print(f"[curate] valid_to stamped on {n_stamp} superseded note(s)",
                  flush=True)
    except Exception as e:  # noqa: BLE001 — corpus already written, never fatal
        print(f"[curate] WARN validity stamp failed (corpus intact): {e}", flush=True)

    if args.incremental:
        # The watermark IS the incremental state; buckets stay untouched so the
        # canonical machinery never mistakes a partial pass for a finished unit.
        # Advancing only here — after validation — means a failed pass never
        # loses tail (at-least-once; dedup absorbs re-runs).
        if ctx.watermark_advance:
            _mark_watermark(ctx.state_file, ctx.state_key, ctx.watermark_advance)
            print(f"[curate] watermark {ctx.state_key} → {ctx.watermark_advance}",
                  flush=True)
    elif args.provisional:
        # Provisional passes never claim the unit: the session may grow after a
        # resume, and the canonical (nightly / final) pass must be able to rerun.
        print(f"[curate] provisional pass — {ctx.state_bucket}:{ctx.state_key} "
              f"left unmarked", flush=True)
    else:
        _mark_curated(ctx.state_file, ctx.state_bucket, ctx.state_key)
    try:
        counts = rebuild_indices(ctx)
        print(f"[curate] indices rebuilt: {counts}", flush=True)
    except Exception as e:  # noqa: BLE001 — derived index: never fails the CURATION
        # ...but a stale index silently blinds production recall — new notes exist
        # yet never surface (it cost two invisible days, 2026-07-04..06) — so this
        # SCREAMS instead of warning into a log nobody reads. Curation still exits
        # 0: the corpus + manifest landed; only the derived index is behind.
        print(f"[curate] ERROR index rebuild failed (corpus intact): {e}", flush=True)
        notify_alert(title="[recall curate] index_rebuild_failed",
                     body=(f"{ctx.slug}: {type(e).__name__}: {e} — production recall "
                           f"is serving a STALE index until a rebuild succeeds "
                           f"(recall build --project "
                           f"{ctx.project_knowledge_dir.parent.parent})"),
                     priority="urgent")

    ok = Outcome(kind="curated", reason="ok",
                 detail=manifest.summary if manifest else "", exit_code=0)
    _print_outcome(ok)
    _append_session_block(ctx.session_log_path, target, ctx.slug, ok, ctx, manifest)

    if args.commit:
        try:
            for c in autocommit(ctx, manifest):
                print(f"[curate] {c}", flush=True)
        except Exception as e:  # noqa: BLE001 — corpus already written, never fatal
            print(f"[curate] WARN auto-commit failed (corpus intact): {e}", flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    return run(argv).exit_code


if __name__ == "__main__":
    sys.exit(main())
