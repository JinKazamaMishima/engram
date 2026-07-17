#!/usr/bin/env python3
"""Recall memory as first-class in-process MCP tools.

The ambient UserPromptSubmit hook PUSHES ~5 note titles into each turn; these
tools let the model PULL — search the fused corpus mid-turn with its own query,
then read full note bodies. They complement, never replace, the hook: same
indices, same warm-daemon embed with keyword-only degrade, same fail-open
discipline (a memory hiccup must never break a turn — errors come back as
``is_error`` content the model can route around). Nothing here loads torch:
the model lives in the daemon; keyword-only recall needs none. All ``recall``
imports are lazy so an uninstalled memory stack degrades to a polite error.

Search hits are logged as activations (the same hippocampal trace the hook
writes) — a model-pulled memory is a genuine retrieval and should reinforce.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

DAEMON_TIMEOUT = 0.6     # seconds; daemon-down must not stall a turn
MAX_NOTE_CHARS = 8000    # protect the context window from a giant note
_SLUG_RE = re.compile(r"[A-Za-z0-9._-]+")

SEARCH_DESC = (
    "Search Engram's curated long-term memory — this project's knowledge corpus plus "
    "the shared global/'soul' corpus — with a free-text query (hybrid keyword + "
    "semantic). Returns ranked notes as '[score] (corpus·kind) slug — description'. "
    "Use it mid-task whenever past decisions, lessons, evals, or project history "
    "would inform the answer; follow up with recall_read_note(slug) for a note's "
    "full reasoning.")
READ_DESC = (
    "Read the full body of one memory note by slug (as returned by recall_search). "
    "Gives the complete prior reasoning — the numbers, mechanics, and the WHY — "
    "not just the one-line description.")
CODE_SEARCH_DESC = (
    "Semantic + keyword search over THIS repository's own source code (the sonar "
    "index) — find code by what it DOES, not just literal tokens. Returns ranked "
    "'path:Lstart-Lend — signature' locators; follow up by Reading that path over "
    "that line range. Reach for it to locate where a behavior lives before editing "
    "or explaining it — it complements grep/glob (which need the exact string) with "
    "meaning-based recall. An empty result or 'no code index' just means fall back "
    "to grep.")

# Code chunks are matched by a prose query, so the note index's 0.40 semantic
# floor (tuned on the note eval) does not transfer; m1 leaves the gate OFF and
# returns the fused top-k. A code-tuned floor is a later milestone.
CODE_SEM_FLOOR = float(os.environ.get("RECALL_CODE_SEM_FLOOR", "0.0"))


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _fetch_query_vector(prompt: str) -> list | None:
    """Query embedding from the warm daemon; None → keyword-only (same recipe as
    the UserPromptSubmit hook in scripts/recall_inject.py)."""
    host = os.environ.get("RECALL_EMBED_HOST", "127.0.0.1")
    port = os.environ.get("RECALL_EMBED_PORT", "8973")
    data = json.dumps({"text": prompt, "is_query": True}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/embed", data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=DAEMON_TIMEOUT) as resp:
            return json.loads(resp.read()).get("embedding")
    except Exception:  # noqa: BLE001 — daemon down → keyword-only fallback
        return None


def build_code_tools(cwd: Path | str) -> list:
    """The ``code_search`` tool bound to ``cwd``'s repo — hybrid recall over the
    sonar code index (``config.code_index_path``). Separate concern from the note
    tools: it searches SOURCE, never the memory corpus, and is a pure accelerator
    (the model still Reads the file it points at). Fail-open like the recall tools;
    a repo with no code index yet returns a polite 'run code-build', not an error
    that breaks the turn."""
    project_dir = Path(cwd)

    def _search_sync(query: str, k: int) -> list | None:
        from recall import config, index
        db = config.code_index_path(config.project_slug(project_dir))
        if not Path(db).exists():
            return None
        conn = index._connect(Path(db), read_only=True)
        try:
            return index.search(conn, query, query_vector=_fetch_query_vector(query),
                                k=k, corpus_label="code", sem_floor=CODE_SEM_FLOOR)
        finally:
            conn.close()

    @tool("code_search", CODE_SEARCH_DESC, {"query": str, "k": int})
    async def code_search(args: dict) -> dict:
        try:
            query = str(args.get("query") or "").strip()
            if not query:
                return _err("code_search: empty query")
            k = max(1, min(int(args.get("k") or 8), 25))
            hits = await asyncio.to_thread(_search_sync, query, k)
            if hits is None:
                return _err("no code index for this repo yet — run: recall code-build")
            if not hits:
                return _text("No matching code found (try grep for an exact token).")
            lines = [f"[{getattr(h, 'score', 0.0):.2f}] {h.slug} — {h.description}"
                     for h in hits]
            return _text("\n".join(lines))
        except Exception as exc:  # noqa: BLE001 — fail open, model routes around
            return _err(f"code index unavailable: {type(exc).__name__}: {exc}")

    return [code_search]


def build_recall_tools(cwd: Path | str) -> list:
    """The ``recall_search`` + ``recall_read_note`` tools + the sonar ``code_search``
    tool, bound to ``cwd``'s project corpus + the shared soul. Shared source for both
    the SDK server (below) and the envoy native loop, so both drive the SAME handlers
    — no drift."""
    project_dir = Path(cwd)

    def _scopes() -> list:
        from recall import config
        slug = config.project_slug(project_dir)
        return [(slug, config.index_path(slug)),
                (config.GLOBAL_SCOPE, config.index_path(config.GLOBAL_SCOPE))]

    def _search_sync(query: str, k: int) -> list:
        from recall import index
        hits = index.search_corpora(_scopes(), query,
                                    query_vector=_fetch_query_vector(query), k=k)
        try:
            from recall import activation
            activation.log_surfaced(hits)      # retrieval reinforces (FSRS trace)
        except Exception:  # noqa: BLE001 — recording must never block recall
            pass
        return hits

    @tool("recall_search", SEARCH_DESC, {"query": str, "k": int})
    async def recall_search(args: dict) -> dict:
        try:
            query = str(args.get("query") or "").strip()
            if not query:
                return _err("recall_search: empty query")
            k = max(1, min(int(args.get("k") or 5), 20))
            hits = await asyncio.to_thread(_search_sync, query, k)
            if not hits:
                return _text("No matching notes in the memory corpus.")
            lines = []
            for h in hits:
                kind = f"·{h.kind}" if getattr(h, "kind", None) else ""
                score = getattr(h, "score", 0.0)
                lines.append(f"[{score:.2f}] ({h.corpus}{kind}) {h.slug} — {h.description}")
            return _text("\n".join(lines))
        except Exception as exc:  # noqa: BLE001 — fail open, model routes around
            return _err(f"memory unavailable: {type(exc).__name__}: {exc}")

    @tool("recall_read_note", READ_DESC, {"slug": str})
    async def recall_read_note(args: dict) -> dict:
        try:
            slug = str(args.get("slug") or "").strip()
            if not _SLUG_RE.fullmatch(slug):
                return _err(f"recall_read_note: bad slug {slug!r}")

            def _read() -> str | None:
                from recall import config
                for base in (config.project_corpus_dir(project_dir),
                             config.global_corpus_dir()):
                    p = Path(base) / f"{slug}.md"
                    if p.exists():
                        return p.read_text()
                return None

            body = await asyncio.to_thread(_read)
            if body is None:
                return _err(f"no note '{slug}' in this project's corpus or the soul")
            if len(body) > MAX_NOTE_CHARS:
                body = body[:MAX_NOTE_CHARS] + "\n…(truncated)"
            return _text(body)
        except Exception as exc:  # noqa: BLE001 — fail open, model routes around
            return _err(f"memory unavailable: {type(exc).__name__}: {exc}")

    return [recall_search, recall_read_note, *build_code_tools(project_dir)]


def build_recall_server(cwd: Path | str):
    """The in-process MCP server ('recall') wrapping build_recall_tools. Returns None
    if the SDK server can't be built — the caller treats memory tools as optional."""
    try:
        return create_sdk_mcp_server(name="recall", version="1.0.0",
                                     tools=build_recall_tools(cwd))
    except Exception:  # noqa: BLE001 — memory tools are strictly optional
        return None
