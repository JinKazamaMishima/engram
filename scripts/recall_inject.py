#!/usr/bin/env python3
"""UserPromptSubmit hook — inject the most relevant curated knowledge notes.

Machine-global recall for every Claude Code project. On each prompt: fused
hybrid recall over THIS project's corpus + the shared global/"soul" corpus via
the local indices, with the query vector fetched from the warm recall-embedder
daemon; if the daemon is down it degrades to keyword-only (FTS5). Matched note
titles are injected as ``additionalContext`` (silent model context) so future
reasoning can build on prior conversations, AND a short ``🧠 recalled: ...``
line is shown to the operator via ``systemMessage`` — recall stays visible.

FAIL-OPEN ALWAYS: any error (no index, daemon down, recall not importable, bad
input) prints nothing and exits 0 — a recall hook must never block prompt
submission. All ``recall`` imports are lazy so even an import failure fails open.
Nothing here loads torch: the model lives in the daemon; keyword-only recall
needs no model.

Wire via ~/.claude/settings.json (user-level, applies to every project) — or
just run ``scripts/install_hook.py``, which writes this for you:
  "UserPromptSubmit": [{"hooks": [{"type": "command",
     "command": "<python> <repo>/scripts/recall_inject.py"}]}]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

MIN_PROMPT = 12          # don't recall on trivial prompts ("ok", "thanks")
TOP_K = int(os.environ.get("RECALL_K", "5"))
DAEMON_TIMEOUT = 0.6     # seconds; daemon-down must not stall prompt submission
GLOBAL_SCOPE = "global"  # corpus label for the shared soul (== config.GLOBAL_SCOPE)


def _project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path.cwd())


def _scopes() -> list[tuple[str, Path]]:
    """This project's index + the shared global index (either may be absent)."""
    from recall import config
    slug = config.project_slug(_project_dir())
    return [(slug, config.index_path(slug)),
            (config.GLOBAL_SCOPE, config.index_path(config.GLOBAL_SCOPE))]


def _fetch_query_vector(prompt: str) -> list[float] | None:
    """Query embedding from the warm daemon; None if it's unreachable."""
    host = os.environ.get("RECALL_EMBED_HOST", "127.0.0.1")
    port = os.environ.get("RECALL_EMBED_PORT", "8973")
    data = json.dumps({"text": prompt, "is_query": True}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/embed", data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=DAEMON_TIMEOUT) as resp:
            return json.loads(resp.read()).get("embedding")
    except Exception:  # noqa: BLE001 — daemon down -> keyword-only fallback
        return None


def _format_context(hits) -> str:
    """Model-only block (additionalContext): hits grouped by corpus (this project
    vs the shared soul), each tagged with its kind, so the model reads structure
    instead of a flat list. Pure string formatting over data already on each Hit
    — no extra query, no added latency (the 0.6s budget is the daemon fetch)."""
    lines = ["## Recalled knowledge — curated notes",
             "_Prior reasoning that may be relevant; cite the slug if you use one;"
             " ask /recall to follow [[links]] or read a superseded note's"
             " replacement._"]

    def _section(title: str, group: list) -> None:
        if not group:
            return
        lines.append(f"\n### {title}")
        for h in group:
            tag = f" [{h.kind}]" if h.kind else ""
            hist = (f" — ⏳ HISTORICAL (was true until {h.valid_to})"
                    if getattr(h, "historical", False) else "")
            lines.append(f"- **{h.slug}**{tag} — {h.description}{hist}")

    _section("This project", [h for h in hits if h.corpus != GLOBAL_SCOPE])
    _section("Global / soul", [h for h in hits if h.corpus == GLOBAL_SCOPE])
    return "\n".join(lines)


def _format_system_message(hits) -> str:
    """Short, operator-VISIBLE line (systemMessage)."""
    return "🧠 recalled: " + ", ".join(f"{h.corpus}:{h.slug}" for h in hits)


def recall_hits(prompt: str, scopes, *, query_vector=None, k: int = TOP_K):
    """Testable core: fused hybrid-recall hits, or [] when there's nothing to
    surface (short prompt, no index, no matches)."""
    prompt = (prompt or "").strip()
    if len(prompt) < MIN_PROMPT or not any(Path(db).exists() for _, db in scopes):
        return []
    from recall import index
    return index.search_corpora(scopes, prompt, query_vector=query_vector, k=k)


def main() -> int:
    try:
        raw = sys.stdin.read()
        hook = json.loads(raw) if raw.strip() else {}
        prompt = (hook.get("prompt") or "").strip()
        if len(prompt) < MIN_PROMPT:
            return 0
        qvec = _fetch_query_vector(prompt)
        hits = recall_hits(prompt, _scopes(), query_vector=qvec)
        if hits:
            # Record the activation (the hippocampal trace the nightly consolidate
            # folds into note stability). Fail-open + torch-free: one local append,
            # internally swallowed, so a logging hiccup never blocks the prompt.
            try:
                from recall import activation
                activation.log_surfaced(hits)
            except Exception:  # noqa: BLE001 — recording must never block recall
                pass
            # systemMessage = operator-visible; additionalContext = model context
            # (silent); suppressOutput hides the raw JSON echo.
            print(json.dumps({
                "suppressOutput": True,
                "systemMessage": _format_system_message(hits),
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": _format_context(hits),
                },
            }))
    except Exception:  # noqa: BLE001 — fail-open, never block the prompt
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
