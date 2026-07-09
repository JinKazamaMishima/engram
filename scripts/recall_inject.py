#!/usr/bin/env python3
"""UserPromptSubmit hook — standing rules (always-on) + relevant curated notes.

Machine-global recall for every Claude Code project. On each prompt, TWO tiers:

1. **Standing rules, every turn, any prompt** — all active ``kind: rule`` notes
   (the shared soul's in any folder + this project's own) are prepended via a
   torch-free, index-free frontmatter scan, so the behavioral/identity channel
   is deterministic: zero-miss, and it survives an index/daemon outage. Rule
   notes are dropped from the retrieved hits below (already present). Skipped
   inside recall's own headless memory-skill runs (same env contract the gate
   exempts) — the curator doesn't take operator directives.
2. **Retrieval** — fused hybrid recall over THIS project's corpus + the shared
   global/"soul" corpus via the local indices, with the query vector fetched
   from the warm recall-embedder daemon; if the daemon is down it degrades to
   keyword-only (FTS5). Matched note titles are injected as
   ``additionalContext`` (silent model context) so future reasoning can build
   on prior conversations, AND a short ``🧠 recalled: ...`` line is shown to
   the operator via ``systemMessage`` — recall stays visible.

A session inside an UNREGISTERED git repo also gets a one-line warning (model
context every turn; operator-visible once per session): sessions there are not
curated into memory until `recall register` — the coverage-gap class of bug
made visible instead of silent.

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
import re
import sys
import urllib.request
from pathlib import Path

MIN_PROMPT = 12          # don't run RETRIEVAL on trivial prompts ("ok", "thanks")
TOP_K = int(os.environ.get("RECALL_K", "5"))
DAEMON_TIMEOUT = 0.6     # seconds; daemon-down must not stall prompt submission
GLOBAL_SCOPE = "global"  # corpus label for the shared soul (== config.GLOBAL_SCOPE)
RULE_KIND = "rule"       # == recall.rules.RULE_KIND (literal: imports stay lazy)
# recall's own headless memory-skill runs export these env contracts — the same
# predicate scripts/recall_gate.py exempts. Rules + the registration warning
# stay out of machine runs: the curator takes its wrapper's contract, not the
# operator's standing directives, and its transcripts stay clean of them.
EXEMPT_ENV_PREFIXES = ("RECALL_CURATE_", "RECALL_DREAM_", "RECALL_RECON")


def _project_dir() -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path.cwd())


def _machine_run() -> bool:
    return any(k.startswith(EXEMPT_ENV_PREFIXES) for k in os.environ)


def _rules_context() -> str | None:
    """Tier 1: the always-on standing-rules block (soul + this project's
    ``kind: rule`` notes). Fail-open: any error means no rules this turn."""
    try:
        from recall import rules
        return rules.rules_context(_project_dir())
    except Exception:  # noqa: BLE001 — the rules tier must never block a prompt
        return None


def _registration_warning(session_id: str) -> tuple[str | None, str | None]:
    """(model_context_line, operator_visible_line) when the session sits in a
    git repo recall doesn't curate. The context line rides every turn (it must
    survive compaction); the visible line fires once per session via a marker
    file so the operator is nudged, not nagged. Fail-open: (None, None)."""
    try:
        from recall import config, registry
        proj = _project_dir().resolve()
        if not (proj / ".git").exists():          # not a repo (worktrees: .git file)
            return None, None
        root = config.data_root().resolve()
        if proj == root or root in proj.parents:  # the data root / soul repo itself
            return None, None
        if proj in registry.list_projects():
            return None, None
        ctx = ("⚠ This repo is NOT registered with recall — sessions here are "
               "never curated into long-term memory. Register it from the repo "
               "root: `recall register`.")
        visible = f"⚠ {proj.name}: not registered with recall — `recall register`"
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", session_id or "nosession")[:64]
        marker = root / "state" / f"regwarn-{sid}"
        if marker.exists():
            return ctx, None
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        except OSError:
            pass  # marker is best-effort; worst case the nudge repeats
        return ctx, visible
    except Exception:  # noqa: BLE001 — a warning must never block a prompt
        return None, None


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


def _drop_rule_hits(hits):
    """Rule notes are the always-on tier — surfacing one as a retrieval hit
    would spend a slot on something already injected above."""
    return [h for h in hits if (getattr(h, "kind", "") or "").lower() != RULE_KIND]


def recall_hits(prompt: str, scopes, *, query_vector=None, k: int = TOP_K):
    """Testable core: fused hybrid-recall hits, or [] when there's nothing to
    surface (short prompt, no index, no matches)."""
    prompt = (prompt or "").strip()
    if len(prompt) < MIN_PROMPT or not any(Path(db).exists() for _, db in scopes):
        return []
    from recall import index
    return _drop_rule_hits(
        index.search_corpora(scopes, prompt, query_vector=query_vector, k=k))


def main() -> int:
    try:
        raw = sys.stdin.read()
        hook = json.loads(raw) if raw.strip() else {}
        prompt = (hook.get("prompt") or "").strip()

        context: list[str] = []   # additionalContext blocks, in reading order
        visible: list[str] = []   # systemMessage fragments

        # Tier 1 — standing rules + registration nudge: every turn, ANY prompt
        # length (identity must hold on a bare "hey"), but never in recall's
        # own machine runs.
        warn_ctx = None
        if not _machine_run():
            rules_block = _rules_context()
            if rules_block:
                context.append(rules_block)
            warn_ctx, warn_visible = _registration_warning(
                str(hook.get("session_id") or ""))
            if warn_visible:
                visible.append(warn_visible)

        # Tier 2 — retrieval, still gated on a non-trivial prompt.
        if len(prompt) >= MIN_PROMPT:
            qvec = _fetch_query_vector(prompt)
            hits = recall_hits(prompt, _scopes(), query_vector=qvec)
            if hits:
                # Record the activation (the hippocampal trace the nightly
                # consolidate folds into note stability). Fail-open + torch-free:
                # one local append, internally swallowed, so a logging hiccup
                # never blocks the prompt. Rules never log — always-on presence
                # is not an activation signal.
                try:
                    from recall import activation
                    activation.log_surfaced(hits)
                except Exception:  # noqa: BLE001 — recording must never block recall
                    pass
                context.append(_format_context(hits))
                visible.insert(0, _format_system_message(hits))
        if warn_ctx:
            context.append(warn_ctx)

        if context:
            # systemMessage = operator-visible; additionalContext = model context
            # (silent); suppressOutput hides the raw JSON echo.
            out = {
                "suppressOutput": True,
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": "\n\n".join(context),
                },
            }
            if visible:
                out["systemMessage"] = " · ".join(visible)
            print(json.dumps(out))
    except Exception:  # noqa: BLE001 — fail-open, never block the prompt
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
