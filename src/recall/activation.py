"""Activation trace — the hippocampal fast store.

The recall hook (``scripts/recall_inject.py``) appends one event per surfaced note
on every prompt; the nightly ``recall consolidate`` folds those events into note
stability (the slow cortical store). The log is **disposable telemetry** under the
data root — append-only JSONL, lock-free, fail-open, torch-free. Nothing here
loads a model or blocks prompt submission.

One log per scope (a project slug, or ``global``) at
``<data_root>/activation/<scope>.jsonl``; each line is
``{"ts": <iso>, "slug": <slug>, "kind": "surfaced"|"cited"}``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from recall import config

ACTIVATION_DIRNAME = "activation"
_KINDS = ("surfaced", "cited")


def activation_dir() -> Path:
    return config.data_root() / ACTIVATION_DIRNAME


def log_path(scope: str) -> Path:
    return activation_dir() / f"{scope}.jsonl"


def _now_iso(ts: datetime | None = None) -> str:
    return (ts or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec="seconds")


def log_surfaced(hits, *, ts: datetime | None = None) -> None:
    """Append one ``surfaced`` event per hit, grouped by corpus/scope. Called by
    the recall hook. FAIL-OPEN: any error is swallowed — recording an activation
    must never block prompt submission. ``hits`` are ``index.Hit`` (``.corpus`` is
    the scope label, ``.slug`` the note)."""
    try:
        stamp = _now_iso(ts)
        by_scope: dict[str, list[str]] = {}
        for h in hits:
            scope = getattr(h, "corpus", "") or ""
            slug = getattr(h, "slug", "") or ""
            if scope and slug:
                by_scope.setdefault(scope, []).append(slug)
        if not by_scope:
            return
        activation_dir().mkdir(parents=True, exist_ok=True)
        for scope, slugs in by_scope.items():
            with log_path(scope).open("a") as f:
                for slug in slugs:
                    f.write(json.dumps({"ts": stamp, "slug": slug, "kind": "surfaced"},
                                       separators=(",", ":")) + "\n")
    except Exception:  # noqa: BLE001 — fail-open; the hook must never raise
        pass


def _read_events_file(path: Path) -> list[dict]:
    """Tolerant JSONL read: skip blank/garbled lines, keep well-formed events."""
    out: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("slug"):
            out.append(ev)
    return out


def read_events(scope: str) -> list[dict]:
    """All currently-pending events for a scope (live log only)."""
    p = log_path(scope)
    return _read_events_file(p) if p.exists() else []


def claim_events(scope: str) -> tuple[list[dict], Path | None]:
    """Atomically take a scope's pending events for consolidation.

    Renames the live log aside (``os.replace`` is atomic, so an in-flight append
    is captured whole; any append *after* the rename lands in a fresh live log and
    is caught next run — no events lost, none double-counted) and folds it into a
    durable ``<scope>.consuming.jsonl`` that also absorbs any leftover from a
    previously failed run. Returns ``(events, consuming_path)`` — ``(…, None)`` when
    nothing is pending. On success the caller calls :func:`discard_claimed`; on
    failure it leaves the consuming file for the next run to retry."""
    live = log_path(scope)
    consuming = activation_dir() / f"{scope}.consuming.jsonl"
    if live.exists():
        tmp = activation_dir() / f"{scope}.claim-tmp.jsonl"
        try:
            os.replace(live, tmp)                       # atomic claim of live content
            with consuming.open("a") as out:
                out.write(tmp.read_text())
            tmp.unlink()
        except OSError:
            pass
    events = _read_events_file(consuming) if consuming.exists() else []
    return events, (consuming if events else None)


def discard_claimed(scope: str) -> None:
    """Drop the consumed log after a successful consolidation."""
    consuming = activation_dir() / f"{scope}.consuming.jsonl"
    try:
        consuming.unlink()
    except OSError:
        pass


def rollup(events: list[dict]) -> dict[str, dict]:
    """Aggregate events per slug: ``{slug: {count, surfaced, cited, last_ts}}``.
    Used by the consolidate fold (one stability bump per note per run — honoring
    the spacing effect — with the gain decided by whether it was cited)."""
    agg: dict[str, dict] = {}
    for ev in events:
        slug = str(ev.get("slug") or "").strip()
        if not slug:
            continue
        a = agg.setdefault(slug, {"count": 0, "surfaced": 0, "cited": 0, "last_ts": ""})
        kind = str(ev.get("kind") or "surfaced")
        if kind not in _KINDS:
            kind = "surfaced"
        a[kind] += 1
        a["count"] += 1
        ts = str(ev.get("ts") or "")
        if ts > a["last_ts"]:
            a["last_ts"] = ts
    return agg
