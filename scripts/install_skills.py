#!/usr/bin/env python3
"""Install recall's user-level skills into ~/.claude/skills/ from the repo (the
source of truth). Re-run to update after editing a skill in the repo.

  python scripts/install_skills.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_SKILLS = Path(__file__).resolve().parent.parent / "skills"
DEST = Path.home() / ".claude" / "skills"


def main() -> int:
    if not REPO_SKILLS.is_dir():
        print(f"no skills dir at {REPO_SKILLS}", file=sys.stderr)
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for skill in sorted(REPO_SKILLS.iterdir()):
        if not skill.is_dir():
            continue
        target = DEST / skill.name
        target.mkdir(parents=True, exist_ok=True)
        for f in skill.glob("*"):
            if f.is_file():
                shutil.copy2(f, target / f.name)
                n += 1
        print(f"installed skill: {skill.name} -> {target}")
    print(f"done ({n} files). recall + curate-memory now available in every project.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
