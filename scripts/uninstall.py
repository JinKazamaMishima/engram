#!/usr/bin/env python3
"""Engram uninstaller — cleanly reverse what install.sh + the setup wizard created.

Runs on the SYSTEM Python (stdlib only) so it still works after the virtualenv is
gone. It scans the machine, shows a plan, and asks before removing anything.

Your MEMORY CORPUS (the data root) is treated as sacred: it is NEVER removed
unless you explicitly pass --purge-data (and confirm). Everything else is the
install "footprint" — safe to remove and fully re-created by re-running install.

  ./uninstall.sh                 # interactive: remove the install, KEEP your data
  ./uninstall.sh --dry-run       # show what would be removed, change nothing
  ./uninstall.sh --purge-data    # ALSO remove your memory corpus (irreversible)
  ./uninstall.sh --yes           # no prompts (still keeps data unless --purge-data)
  ./uninstall.sh --keep-venv     # leave the repo's .venv in place
  ./uninstall.sh --no-systemd    # don't touch systemd user units
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()
VENV = REPO / ".venv"
CONFIG_DIR = HOME / ".config" / "engram"
ENV_FILE = CONFIG_DIR / "engram.env"
TELEGRAM_ENV = HOME / ".config" / "recall" / "telegram-agent.env"
SKILLS_DEST = HOME / ".claude" / "skills"
SETTINGS = HOME / ".claude" / "settings.json"
LAUNCHER = HOME / ".local" / "bin" / "recall"
SYSTEMD_USER = HOME / ".config" / "systemd" / "user"
DEFAULT_DATA_ROOT = HOME / ".local" / "share" / "recall"

# Every user unit any Engram tier may have installed. Removing an absent one is a
# no-op, so this list can be generous.
SYSTEMD_UNITS = [
    "recall-embedder.service",
    "recall-curate.service", "recall-curate.timer",
    "recall-reconsolidate.service", "recall-reconsolidate.timer",
    "recall-telegram-engram.service",
    "engram-eye.service", "engram-perceive.service",
]

_TTY = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bold(s): return c("1", s)
def dim(s): return c("2", s)
def red(s): return c("31", s)
def green(s): return c("32", s)
def yellow(s): return c("33", s)


def mark(dry: bool) -> str:
    return dim("  would remove") if dry else green("  ✓ removed")


def confirm(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    try:
        ans = input(f"{prompt} [{d}] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not ans:
        return default
    return ans in ("y", "yes")


def data_root() -> Path:
    """Resolve the memory corpus location the same way recall does: an explicit
    RECALL_DATA_ROOT in the environment, else the one saved in engram.env, else
    the default."""
    env = os.environ.get("RECALL_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("RECALL_DATA_ROOT="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return Path(val).expanduser()
    return DEFAULT_DATA_ROOT


# ---- removal steps: each prints its plan (dry) or acts, and returns whether it
#      found something to do -----------------------------------------------------

def step_launcher(dry: bool) -> bool:
    # Only our own symlink (pointing back into this repo's venv), never a real file.
    if not (LAUNCHER.is_symlink() and str(VENV) in os.readlink(LAUNCHER)):
        return False
    print(f"{mark(dry)}  launcher {LAUNCHER}")
    if not dry:
        LAUNCHER.unlink(missing_ok=True)
    return True


def step_skills(dry: bool) -> bool:
    repo_skills = REPO / "skills"
    names = [p.name for p in repo_skills.iterdir() if p.is_dir()] if repo_skills.is_dir() else []
    present = [n for n in names if (SKILLS_DEST / n).is_dir()]
    if not present:
        return False
    print(f"{mark(dry)}  skills {', '.join(present)}  (in {SKILLS_DEST})")
    if not dry:
        for n in present:
            shutil.rmtree(SKILLS_DEST / n, ignore_errors=True)
    return True


def step_hook(dry: bool) -> bool:
    # Match by the recall_inject.py path, NOT the interpreter — the hook was
    # installed with the venv's python, but we run under system python here.
    if not SETTINGS.exists():
        return False
    try:
        data = json.loads(SETTINGS.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    ups = data.get("hooks", {}).get("UserPromptSubmit", [])
    found = any("recall_inject.py" in h.get("command", "")
                for g in ups for h in g.get("hooks", []))
    if not found:
        return False
    print(f"{mark(dry)}  UserPromptSubmit hook (recall_inject.py) from {SETTINGS}")
    if not dry:
        for g in list(ups):
            g["hooks"] = [h for h in g.get("hooks", [])
                          if "recall_inject.py" not in h.get("command", "")]
            if not g["hooks"]:
                ups.remove(g)
        if not ups:
            data.get("hooks", {}).pop("UserPromptSubmit", None)
        if not data.get("hooks"):
            data.pop("hooks", None)
        SETTINGS.write_text(json.dumps(data, indent=2) + "\n")
    return True


def step_systemd(dry: bool, enabled: bool) -> bool:
    if not enabled:
        return False
    if not (shutil.which("systemctl") and SYSTEMD_USER.is_dir()):
        return False
    present = [u for u in SYSTEMD_UNITS if (SYSTEMD_USER / u).exists()]
    if not present:
        return False
    print(f"{mark(dry)}  systemd user units: {', '.join(present)}")
    if not dry:
        # disable+stop, then remove the unit files, then reload.
        subprocess.run(["systemctl", "--user", "disable", "--now", *present],
                       check=False, capture_output=True)
        for u in present:
            (SYSTEMD_USER / u).unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"],
                       check=False, capture_output=True)
    return True


def _strip_engram_blocks(text: str) -> tuple[str, int]:
    """Remove the wizard's rc additions: each is a `# Engram…` comment line
    followed by exactly one payload line (the env-source or the PATH export)."""
    out: list[str] = []
    removed = 0
    skip_next = False
    for line in text.splitlines():
        if skip_next:
            skip_next = False
            removed += 1
            continue
        s = line.strip()
        if s == "# Engram" or s.startswith("# Engram —") or s.startswith("# Engram -"):
            removed += 1
            skip_next = True
            continue
        out.append(line)
    new = "\n".join(out)
    if text.endswith("\n") and not new.endswith("\n"):
        new += "\n"
    return new, removed


def step_rc(dry: bool) -> bool:
    acted = False
    for rc in (HOME / ".zshrc", HOME / ".bashrc"):
        if not rc.exists():
            continue
        text = rc.read_text()
        new, removed = _strip_engram_blocks(text)
        if removed == 0:
            continue
        acted = True
        print(f"{mark(dry)}  {removed} Engram line(s) from {rc}")
        if not dry:
            rc.with_name(rc.name + ".pre-engram-uninstall.bak").write_text(text)
            rc.write_text(new)
    return acted


def step_config(dry: bool) -> bool:
    acted = False
    if CONFIG_DIR.exists():
        acted = True
        print(f"{mark(dry)}  config {CONFIG_DIR}" + dim("  (may hold an API key)"))
        if not dry:
            shutil.rmtree(CONFIG_DIR, ignore_errors=True)
    if TELEGRAM_ENV.exists():
        acted = True
        print(f"{mark(dry)}  telegram secret {TELEGRAM_ENV}")
        if not dry:
            TELEGRAM_ENV.unlink(missing_ok=True)
            try:
                TELEGRAM_ENV.parent.rmdir()  # only if now empty
            except OSError:
                pass
    return acted


def step_venv(dry: bool) -> bool:
    if not VENV.exists():
        return False
    print(f"{mark(dry)}  virtualenv {VENV}")
    if not dry:
        shutil.rmtree(VENV, ignore_errors=True)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="uninstall", description="Cleanly remove Engram from this machine.")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="show what would be removed, change nothing")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="don't prompt (still keeps your data unless --purge-data)")
    ap.add_argument("--purge-data", action="store_true",
                    help="ALSO delete your memory corpus (RECALL_DATA_ROOT) — irreversible")
    ap.add_argument("--keep-venv", action="store_true",
                    help="leave the repo's .venv in place")
    ap.add_argument("--no-systemd", action="store_true",
                    help="don't touch systemd user units")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    droot = data_root()
    print(bold("\n▓▒░ Engram uninstaller ░▒▓"))
    print(dim(f"repo: {REPO}\n"))

    steps = [
        step_launcher, step_skills, step_hook,
        lambda d: step_systemd(d, not a.no_systemd),
        step_rc, step_config,
        (lambda d: False) if a.keep_venv else step_venv,
    ]

    # Plan (dry pass — prints, changes nothing).
    print(bold("Install footprint to remove:"))
    present = any([s(True) for s in steps])   # list forces all to print
    if not present:
        print(dim("  (none found)"))

    # Data corpus line.
    print(bold("\nYour memory corpus:"))
    have_data = droot.exists()
    if not have_data:
        print(dim(f"  (none at {droot})"))
    elif a.purge_data:
        print(red(f"  ✗ WILL DELETE {droot}  (--purge-data)"))
    else:
        print(green(f"  ✓ KEEP {droot}") + dim("  (pass --purge-data to remove)"))

    if a.dry_run:
        print(dim("\ndry-run: nothing was changed."))
        return 0

    if not present and not (a.purge_data and have_data):
        print("\nNothing to remove.")
        return 0

    # Confirm.
    if not a.yes:
        print()
        if present and not confirm(bold("Remove the install footprint above?"), default=False):
            print(yellow("aborted — nothing changed."))
            return 1

    # Execute footprint.
    print()
    for s in steps:
        s(False)

    # Execute data purge (separate, explicit, loud).
    if a.purge_data and have_data:
        ok = a.yes
        if not ok:
            print(red(f"\n⚠ This permanently deletes your memory corpus at {droot}."))
            ok = confirm(red("Type y to confirm you want it GONE"), default=False)
        if ok:
            shutil.rmtree(droot, ignore_errors=True)
            print(f"{green('  ✓ removed')}  memory corpus {droot}")
        else:
            print(yellow(f"  kept {droot}"))
    elif have_data:
        print(green(f"\nKept your memory corpus: {droot}"))

    # Final notes — things we deliberately DON'T touch.
    print(bold("\nDone.") + " Engram's footprint is removed.")
    notes = [
        f"the code is still here — delete it with:  rm -rf {REPO}",
        "uv, the `claude` CLI, and Python were left installed (general tools).",
    ]
    if SETTINGS.with_name("settings.json.pre-recall.bak").exists():
        notes.append(f"a settings backup remains: {SETTINGS.with_name('settings.json.pre-recall.bak')}")
    if not a.no_systemd and not shutil.which("systemctl"):
        notes.append("no systemd here — if you set up launchd/cron jobs, remove them manually.")
    for n in notes:
        print(dim(f"  • {n}"))
    print(dim("  • open a new shell so the removed PATH/env lines take effect."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
