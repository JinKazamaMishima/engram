#!/usr/bin/env python3
"""Install the recall UserPromptSubmit hook into ~/.claude/settings.json.

Run this YOURSELF — the agent is intentionally blocked from self-modifying its
own Claude Code config, so a human enacts the global hook:

  python scripts/install_hook.py

Idempotent: backs up once to settings.json.pre-recall.bak, preserves every
existing setting, and won't duplicate the hook if it's already present.
Re-runnable. To uninstall, run with --remove (or restore the backup).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOOK_CMD = f"{sys.executable} {REPO / 'scripts' / 'recall_inject.py'}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    remove = "--remove" in argv
    p = Path.home() / ".claude" / "settings.json"
    data = json.loads(p.read_text()) if p.exists() else {}

    bak = p.with_name("settings.json.pre-recall.bak")
    if p.exists() and not bak.exists():
        bak.write_text(p.read_text())

    ups = data.setdefault("hooks", {}).setdefault("UserPromptSubmit", [])

    def has_hook(group: dict) -> bool:
        return any(h.get("command") == HOOK_CMD for h in group.get("hooks", []))

    if remove:
        for group in list(ups):
            group["hooks"] = [h for h in group.get("hooks", [])
                              if h.get("command") != HOOK_CMD]
            if not group.get("hooks"):
                ups.remove(group)
        p.write_text(json.dumps(data, indent=2) + "\n")
        print(f"removed recall hook from {p} (open /hooks or restart to reload)")
        return 0

    if any(has_hook(g) for g in ups):
        print("recall hook already installed; nothing to do.")
        return 0

    ups.append({"hooks": [{"type": "command", "command": HOOK_CMD}]})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")
    print(f"OK — recall hook installed in {p}")
    print(f"backup: {bak}")
    print("now open /hooks once (or restart Claude Code) to load it this session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
