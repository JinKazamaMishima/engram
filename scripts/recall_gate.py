#!/usr/bin/env python3
"""PreToolUse gate — the hot-cache preemption fix's enforcement half.

The fact channel is archived out of session auto-load so recall's per-turn
injection is actually exercised; this hook keeps the archive honest by gating
AMBIENT corpus browsing while leaving deliberate retrieval doors open:

  DENY   Read/Grep/Glob targeting a knowledge corpus (a repo's docs/knowledge,
         the global/soul corpus, the archived file-memory) — silently re-reading
         those files would rebuild the hot cache one tool call at a time and
         make retrieval misses invisible again.
  ALLOW  the recall MCP tools (index-mediated, the sanctioned door), Write/Edit
         (the curator owns note writes), and everything else.
  LOG    Bash commands touching a corpus path — allowed on purpose as the ONE
         deliberate override, and appended to the miss-log: every use means the
         per-turn injection failed to surface something needed. The miss-log is
         the measurement this whole design exists to produce.
  EXEMPT the memory-skill headless runs (curate/dream/reconsolidate export a
         distinctive env contract; hook subprocesses inherit it) — the curator
         must read the corpus it maintains.

FAIL-OPEN: any error exits 0 with no output (allow) — a broken gate must never
block real work. Enforced as a HOOK, never a prompt rule: prompt rules decay,
the harness doesn't.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MISS_LOG = Path(os.environ.get(
    "RECALL_MISS_LOG",
    os.path.expanduser("~/.local/share/recall/miss-log.jsonl")))

# Path fragments that mark a knowledge corpus. Substring match on the tool's
# stated target — deliberately simple; the gate stops ambient habit, not intent.
PROTECTED = (
    "/docs/knowledge",
    "/.local/share/recall/global",
    "/memory/archive",
)

# The memory-skill runs export these; their tool calls must pass untouched.
EXEMPT_ENV_PREFIXES = ("RECALL_CURATE_", "RECALL_DREAM_", "RECALL_RECON")

GATED_TOOLS = {"Read", "Grep", "Glob"}


def _protected(target: str) -> bool:
    return bool(target) and any(seg in target for seg in PROTECTED)


def _targets(tool_input: dict) -> list[str]:
    """Every path-ish string a Read/Grep/Glob call names."""
    out = []
    for key in ("file_path", "path", "pattern"):
        v = tool_input.get(key)
        if isinstance(v, str):
            out.append(v)
    return out


def main() -> int:
    try:
        hook = json.loads(sys.stdin.read() or "{}")
        if any(k.startswith(EXEMPT_ENV_PREFIXES) for k in os.environ):
            return 0
        tool = hook.get("tool_name", "")
        tin = hook.get("tool_input") or {}

        if tool in GATED_TOOLS and any(_protected(t) for t in _targets(tin)):
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Corpus browsing is gated (hot-cache preemption fix): "
                    "facts must arrive via per-turn recall injection. If it "
                    "missed something, use the recall MCP tools "
                    "(recall_search / recall_read_note) — or `Bash cat` the "
                    "file, which is allowed and LOGGED as a retrieval miss."),
            }}))
            return 0

        if tool == "Bash":
            cmd = tin.get("command") or ""
            if _protected(cmd):
                MISS_LOG.parent.mkdir(parents=True, exist_ok=True)
                with MISS_LOG.open("a") as f:
                    f.write(json.dumps({
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "cwd": hook.get("cwd", ""),
                        "cmd": cmd[:500],
                    }) + "\n")
        return 0
    except Exception:  # noqa: BLE001 — fail-open, never block real work
        return 0


if __name__ == "__main__":
    sys.exit(main())
