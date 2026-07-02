"""Opt-in registry of projects the nightly curator visits.

A plain-text file (one absolute project path per line, ``#`` comments) at
``$RECALL_DATA_ROOT/projects.txt`` — trivial to hand-edit. ``recall curate-all``
iterates it; ``recall register`` adds the current (or named) project.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recall import config


def registry_path() -> Path:
    return config.data_root() / "projects.txt"


def list_projects() -> list[Path]:
    p = registry_path()
    if not p.exists():
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = Path(line).expanduser().resolve()
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def register(project_dir: str | Path) -> bool:
    """Append a project if not already present. Returns True if added."""
    d = Path(project_dir).expanduser().resolve()
    if d in list_projects():
        return False
    p = registry_path()
    config.ensure_dirs(p.parent)
    with p.open("a") as f:
        f.write(f"{d}\n")
    return True


def curate_all(argv: list[str] | None = None) -> int:
    """Run ``recall curate`` for every registered project (in-process,
    sequentially). Aggregates exit codes — one project's failure doesn't stop
    the others, but the overall code is non-zero if any failed."""
    from recall import curate
    ap = argparse.ArgumentParser(prog="recall curate-all")
    ap.add_argument("--date", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    projects = list_projects()
    if not projects:
        print(f"[curate-all] no projects registered in {registry_path()} — "
              f"add one with `recall register`.", file=sys.stderr)
        return 0
    rc = 0
    for d in projects:
        sub = ["--project-dir", str(d)]
        if a.date:
            sub += ["--date", a.date]
        if a.force:
            sub.append("--force")
        if a.commit:
            sub.append("--commit")
        if a.dry_run:
            sub.append("--dry-run")
        print(f"\n=== curate {d.name} ({d}) ===", flush=True)
        if curate.run(sub).exit_code != 0:
            rc = 1
    return rc


def curate_sessions_all(argv: list[str] | None = None) -> int:
    """Nightly safety-net sweep: for every registered project, curate each of the
    day's SESSIONS not already done live (the 'sessions' bucket dedups, so no
    double-work), catching abandoned / terminal / bridge-was-down sessions. This
    replaces the date-based curate-all in the nightly cycle; curate-all stays for
    manual date backfills. One session's failure doesn't stop the rest."""
    from datetime import date, datetime, timezone

    from recall import curate
    from recall import transcripts as T
    from recall.curate import ET
    ap = argparse.ArgumentParser(prog="recall curate-sessions-all")
    ap.add_argument("--date", default=None,
                    help="sweep sessions active on/after this ISO date (default: today ET)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        target = (date.fromisoformat(a.date) if a.date
                  else datetime.now(timezone.utc).astimezone(ET).date())
    except ValueError:
        print(f"[curate-sessions-all] bad --date {a.date!r}", file=sys.stderr)
        return 1

    projects = list_projects()
    if not projects:
        print(f"[curate-sessions-all] no projects registered in {registry_path()} — "
              f"add one with `recall register`.", file=sys.stderr)
        return 0

    def _flags(base: list[str]) -> list[str]:
        if a.force:
            base.append("--force")
        if a.commit:
            base.append("--commit")
        if a.dry_run:
            base.append("--dry-run")
        return base

    rc = 0
    for d in projects:
        sessions = T.discover_transcripts(T.project_transcript_dir(d), target)
        print(f"\n=== curate-sessions {d.name}: {len(sessions)} session(s) "
              f"active since {target.isoformat()} ({d}) ===", flush=True)
        for path in sessions:
            sub = _flags(["--session", path.stem, "--project-dir", str(d)])
            if curate.run(sub).exit_code != 0:
                rc = 1
    return rc


def consolidate_all(argv: list[str] | None = None) -> int:
    """Nightly activation fold for the global soul corpus + every registered
    project: bump note stability from the day's recall activations, sync indices,
    scoped-commit. Deterministic + GPU-free; one scope's failure doesn't stop the
    others. The overall code is non-zero if any failed."""
    from recall import consolidate
    ap = argparse.ArgumentParser(prog="recall consolidate-all")
    ap.add_argument("--date", default=None)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    def _flags(base: list[str]) -> list[str]:
        if a.date:
            base += ["--date", a.date]
        if a.commit:
            base.append("--commit")
        if a.dry_run:
            base.append("--dry-run")
        return base

    rc = 0
    print("\n=== consolidate global ===", flush=True)
    if consolidate.run(_flags(["--scope", "global"])).exit_code != 0:
        rc = 1
    for d in list_projects():
        print(f"\n=== consolidate {d.name} ({d}) ===", flush=True)
        if consolidate.run(
                _flags(["--scope", "project", "--project-dir", str(d)])
        ).exit_code != 0:
            rc = 1
    return rc


def dream_all(argv: list[str] | None = None) -> int:
    """Nightly dream pass for the global soul corpus + every registered project:
    recombine the day's memories into quarantined hypotheses and bleed corroborated
    ones into the soul. One scope's failure doesn't stop the others; the overall
    code is non-zero if any failed."""
    from recall import dream
    ap = argparse.ArgumentParser(prog="recall dream-all")
    ap.add_argument("--date", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--counterfactual", action="store_true",
                    help="run the L1 counterfactual (what-if) operator on the global soul")
    a = ap.parse_args(argv)

    def _flags(base: list[str]) -> list[str]:
        if a.date:
            base += ["--date", a.date]
        if a.force:
            base.append("--force")
        if a.commit:
            base.append("--commit")
        if a.dry_run:
            base.append("--dry-run")
        return base

    rc = 0
    # Counterfactual (L1 "what-if") dreaming is scoped to the global SOUL — that's where
    # charged episodes/decisions live; project corpora are mostly reference facts the
    # forkability + charge gates would skip anyway. Widen into the project loop below if
    # that ever changes.
    cf = ["--counterfactual"] if a.counterfactual else []
    print("\n=== dream global ===", flush=True)
    if dream.run(_flags(["--scope", "global"]) + cf).exit_code != 0:
        rc = 1
    for d in list_projects():
        print(f"\n=== dream {d.name} ({d}) ===", flush=True)
        if dream.run(
                _flags(["--scope", "project", "--project-dir", str(d)])
        ).exit_code != 0:
            rc = 1
    return rc


def reconsolidate_all(argv: list[str] | None = None) -> int:
    """Weekly corpus-wide reconsolidation for the global soul corpus + every
    registered project. One scope's failure doesn't stop the others; the overall
    code is non-zero if any failed."""
    from recall import reconsolidate
    ap = argparse.ArgumentParser(prog="recall reconsolidate-all")
    ap.add_argument("--date", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    def _flags(base: list[str]) -> list[str]:
        if a.date:
            base += ["--date", a.date]
        if a.force:
            base.append("--force")
        if a.commit:
            base.append("--commit")
        if a.dry_run:
            base.append("--dry-run")
        return base

    rc = 0
    print("\n=== reconsolidate global ===", flush=True)
    if reconsolidate.run(_flags(["--scope", "global"])).exit_code != 0:
        rc = 1
    for d in list_projects():
        print(f"\n=== reconsolidate {d.name} ({d}) ===", flush=True)
        if reconsolidate.run(
                _flags(["--scope", "project", "--project-dir", str(d)])
        ).exit_code != 0:
            rc = 1
    return rc
