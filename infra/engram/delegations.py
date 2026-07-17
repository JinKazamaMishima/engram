#!/usr/bin/env python3
"""In-process registry of live cross-provider delegations (aurora m3).

Sub-agents (the Agent tool) and workflows ride the SDK task registry, so the
agents panel already sees them. A grok call is different in kind: it's an
in-process MCP *tool call*, not a task — it never enters that registry, so the
ctrl+t panel and the one-liner were blind to it (the operator opened the panel during a
10-way Grok/Sonnet bench and saw none of the Grok arms). This module is the
missing channel: the grok tool — and any future in-process hand to a
non-Anthropic model — registers a live entry on the way in and finishes it on
the way out, and the TUI renders the snapshot alongside real tasks.

Same process, same event loop as the panel, so the reads are naturally
consistent; the lock is cheap insurance in case a future caller finishes from a
worker thread. Fail-open to the bone — every function swallows its own errors,
because a status-line registry must never be able to break a delegation.
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_LIVE: dict[int, dict] = {}     # id -> {id,label,model,provider,started}
_SEQ = 0                        # monotonic id source
_DONE = 0                       # finished-ok count this session
_FAILED = 0                     # finished-error count this session
_SPEND = 0.0                    # summed cost_usd of finished delegations


def start(label: str, model: str = "", provider: str = "") -> int:
    """Register a live delegation; return an id to hand to finish(). A return of
    0 means the registry hiccuped — finish(0) is a safe no-op, so the caller
    never has to check."""
    global _SEQ
    try:
        with _LOCK:
            _SEQ += 1
            did = _SEQ
            _LIVE[did] = {"id": did, "label": label, "model": model,
                          "provider": provider, "started": time.monotonic()}
            return did
    except Exception:  # noqa: BLE001 — a registry must never break its caller
        return 0


def finish(did: int, *, cost=None, ok: bool = True) -> None:
    """Mark a delegation finished (idempotent; unknown or 0 id → no-op)."""
    global _DONE, _FAILED, _SPEND
    if not did:
        return
    try:
        with _LOCK:
            if _LIVE.pop(did, None) is None:
                return
            if ok:
                _DONE += 1
            else:
                _FAILED += 1
            if cost:
                _SPEND += float(cost)
    except Exception:  # noqa: BLE001
        pass


def snapshot() -> dict:
    """{live: [entry...], done, failed, cost} — `live` newest-last, `cost` the
    session's summed delegation spend (best-effort; grok reports cost per call)."""
    try:
        with _LOCK:
            live = sorted(_LIVE.values(), key=lambda e: e["started"])
            return {"live": [dict(e) for e in live],
                    "done": _DONE, "failed": _FAILED, "cost": _SPEND}
    except Exception:  # noqa: BLE001
        return {"live": [], "done": 0, "failed": 0, "cost": 0.0}


def reset() -> None:
    """Clear the registry — new conversation thread, or a test."""
    global _SEQ, _DONE, _FAILED, _SPEND
    try:
        with _LOCK:
            _LIVE.clear()
            _SEQ = _DONE = _FAILED = 0
            _SPEND = 0.0
    except Exception:  # noqa: BLE001
        pass
