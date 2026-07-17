"""``recall doctor`` — deterministic drift checks for the memory doctrine.

The doctrine gives every class of content exactly one home:

  CLAUDE.md    repo instructions, hand-written; machine writers never touch it
  auto-memory  a frozen stub pointing at recall — no facts, no index lines
  rules        ``kind: rule`` corpus notes, operator-promoted, injected always-on
  corpus       everything learned (facts/episodes/design), curator-written
  soul         the shared global corpus (operator, identity, cross-project)

Enforcement lives in the harness, not in good intentions: the wrappers already
REJECT machine-authored rules at validation time. Doctor is the nightly sweep
for what a reject can't cover — a harness writing facts back into the memory
stub, a machine commit escaping its docs/knowledge lane, an index gone stale
or missing, a rule note silently failing to parse. Read-only, torch-free,
fail-soft per check; findings print to stdout and (``--notify``) page Telegram.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from recall import config, registry, rules

# The canonical auto-memory stub. The migration writes it; doctor flags drift.
# Its text is itself the doctrine's live carrier: the harness force-loads
# MEMORY.md every session, so this is what tells a bare Claude where memory
# actually lives — including when the whole recall stack is down.
STUB_MEMORY_MD = """\
# Memory lives in recall — this file stays a stub

Standing rules (`kind: rule` notes, soul + project) are injected every turn;
facts are retrieved per-turn from the corpus. Do NOT add index lines, facts,
or note files here — write a corpus note instead (`docs/knowledge/` for
project knowledge, the global soul for operator/cross-project; `sources:
[manual]`). A new RULE additionally needs the operator's explicit go
(`kind: rule`). `recall doctor` flags any drift of this file nightly.

- Injection missed something? `recall_search` / `recall_read_note` (MCP), or
  `Bash cat` the note — the override is logged as a retrieval miss, which is
  the miss-detector working: it finds recall's gaps.
- Pre-doctrine archive: `archive/` + `ARCHIVE.md` (the answer key the
  miss-log diffs against).
"""

# Machine writers commit with these subject prefixes ([curator] etc.) and are
# allowed to touch ONLY the project corpus. Anything else in one of their
# commits is an escaped lane — CLAUDE.md above all.
MACHINE_COMMIT_GREPS = (r"^\[curator\]", r"^\[consolidate\]", r"^\[dream\]",
                        r"^\[reconsolidate\]")
MACHINE_LANE = "docs/knowledge/"

# The nightly cycle syncs indices, so >26h of note-newer-than-index lag means
# at least one full cycle failed to pick the edits up.
INDEX_LAG_HOURS = 26
STRAY_LIST_MAX = 5       # findings page Telegram — name a few, count the rest
MISS_LOG_MAX = 5         # flagged archived slugs to name before counting the rest


@dataclass(frozen=True)
class Finding:
    level: str    # "error" (recall is blind/broken) | "warn" (doctrine drift)
    scope: str    # project slug / "global" / registry
    message: str

    def line(self) -> str:
        mark = "✗" if self.level == "error" else "⚠"
        return f"{mark} [{self.scope}] {self.message}"


def _norm(text: str) -> str:
    """Whitespace-insensitive comparison form: trailing space and blank-line
    runs don't count as drift; words do."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln)


def memory_dir_for(project_dir: Path) -> Path:
    """Claude Code's per-project auto-memory dir (cwd flattened, '/' → '-')."""
    flat = str(Path(project_dir).resolve()).replace("/", "-")
    return Path.home() / ".claude" / "projects" / flat / "memory"


def check_memory_stub(mem_dir: Path, scope: str) -> list[Finding]:
    """The native auto-memory must stay a stub: MEMORY.md == STUB_MEMORY_MD
    (whitespace-insensitive) and no stray note files at the memory root —
    facts written there bypass the corpus and recreate the two-store split."""
    out: list[Finding] = []
    mem_md = mem_dir / "MEMORY.md"
    if mem_md.exists():
        try:
            if _norm(mem_md.read_text()) != _norm(STUB_MEMORY_MD):
                out.append(Finding(
                    "warn", scope,
                    f"MEMORY.md drifted from the recall stub ({mem_md}) — "
                    f"facts may be bypassing the corpus; migrate them to a "
                    f"corpus note and restore the stub"))
        except OSError as e:
            out.append(Finding("warn", scope, f"MEMORY.md unreadable: {e}"))
    if mem_dir.is_dir():
        stray = sorted(p.name for p in mem_dir.glob("*.md")
                       if p.name not in ("MEMORY.md", "ARCHIVE.md"))
        if stray:
            shown = ", ".join(stray[:STRAY_LIST_MAX])
            more = f", … +{len(stray) - STRAY_LIST_MAX} more" \
                if len(stray) > STRAY_LIST_MAX else ""
            out.append(Finding(
                "warn", scope,
                f"{len(stray)} native memory note(s) written outside recall: "
                f"{shown}{more} — migrate to corpus notes (archive/ is the "
                f"sanctioned home for pre-doctrine files)"))
    return out


def check_machine_lane(repo: Path, scope: str, *, days: int) -> list[Finding]:
    """No machine commit may touch anything outside docs/knowledge/ — that
    lane is enforced at commit time by construction (scoped `git add`), so a
    hit here means a writer regressed. Fail-soft on any git trouble."""
    if not (repo / ".git").exists():
        return []
    # "@@" sentinel marks commit headers; every other non-blank line is a path
    # (paths can contain spaces, so no format-guessing).
    cmd = ["git", "-C", str(repo), "log", f"--since={days} days ago",
           "--pretty=format:@@%h %s", "--name-only"]
    for g in MACHINE_COMMIT_GREPS:
        cmd += ["--grep", g]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    out: list[Finding] = []
    header = ""
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        if line.startswith("@@"):
            header = line[2:]
            continue
        path = line.strip()
        if not path.startswith(MACHINE_LANE):
            out.append(Finding(
                "error", scope,
                f"machine commit escaped its lane: {header or '?'} touched "
                f"{path!r} (allowed: {MACHINE_LANE}*)"))
    return out


def check_index(corpus_dir: Path, index_path: Path, scope: str) -> list[Finding]:
    """A corpus with notes must have an index, and the index must not predate
    the notes — an index older than the newest note means edits/curation have
    landed that production recall cannot see (the silent-freeze class). A
    dormant project (old notes, old index) is NOT stale: staleness is measured
    against content, not the wall clock."""
    notes = list(Path(corpus_dir).glob("*.md")) if Path(corpus_dir).is_dir() else []
    if not notes:
        return []
    if not index_path.exists():
        return [Finding("error", scope,
                        f"{len(notes)} note(s) but NO index at {index_path} — "
                        f"recall is blind here (run `recall build`)")]
    newest = max(p.stat().st_mtime for p in notes)
    lag_h = (newest - index_path.stat().st_mtime) / 3600
    if lag_h > INDEX_LAG_HOURS:
        return [Finding("warn", scope,
                        f"index is {lag_h:.0f}h older than the newest note — "
                        f"the nightly rebuild may be failing; recall can't see "
                        f"recent edits")]
    return []


def check_rules(corpus_dir: Path, scope: str) -> list[Finding]:
    """A broken rule note is a SILENT rule outage (it just drops out of the
    always-on tier); an over-budget tier silently omits rules. Both must page."""
    active, broken = rules.scan_rules(corpus_dir)
    out: list[Finding] = []
    if broken:
        out.append(Finding(
            "error", scope,
            f"rule note(s) fail to parse — silent rule outage: "
            f"{', '.join(broken)}"))
    size = sum(len(f"- **{n.slug}** — {n.description}") for n in active)
    budget = rules.rules_budget()
    if size > budget:
        out.append(Finding(
            "warn", scope,
            f"{len(active)} rules total {size} chars > {rules.BUDGET_ENV}"
            f"={budget} — some are being omitted from injection"))
    return out


def _archived_slugs() -> dict[str, str]:
    """slug -> scope for every note currently sitting in an ``archive/`` dir
    (global soul + each registered project). Best-effort; empty on any trouble."""
    out: dict[str, str] = {}
    dirs = [(config.GLOBAL_SCOPE, config.archive_dir(config.global_corpus_dir()))]
    for proj in registry.list_projects():
        if proj.exists():
            dirs.append((config.project_slug(proj),
                         config.archive_dir(config.project_corpus_dir(proj))))
    for scope, adir in dirs:
        try:
            if adir.is_dir():
                for p in adir.glob("*.md"):
                    out.setdefault(p.stem, scope)
        except OSError:
            continue
    return out


def check_miss_log(*, days: int) -> list[Finding]:
    """Close the reaper's loop. The reaper (``recall reap``) archives notes recall
    judges cold; the miss-log records when the operator reached PAST injection for
    a note by hand (``Bash cat`` of a corpus path — recall_gate.py logs it). If a
    recently-missed note is one we've since archived, that's a candidate WRONGFUL
    eviction — the note was cold to retrieval yet the operator still wanted it —
    so surface it with the one-liner that undoes it. Fail-soft: any trouble → no
    findings (the miss-log is disposable telemetry, never a gate)."""
    archived = _archived_slugs()
    if not archived:
        return []
    log = config.data_root() / "miss-log.jsonl"
    if not log.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    hit: dict[str, str] = {}   # slug -> scope
    try:
        for line in log.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = str(rec.get("ts", ""))
            if ts:
                try:
                    if datetime.fromisoformat(ts) < cutoff:
                        continue
                except ValueError:
                    pass  # unparseable ts -> keep it (fail-open on relevance)
            cmd = str(rec.get("cmd", ""))
            for slug, scope in archived.items():
                if slug not in hit and slug in cmd:
                    hit[slug] = scope
    except OSError:
        return []
    by_scope: dict[str, list[str]] = {}
    for slug in sorted(hit):
        by_scope.setdefault(hit[slug], []).append(slug)
    out: list[Finding] = []
    for scope, sl in by_scope.items():
        shown = ", ".join(sl[:MISS_LOG_MAX])
        more = (f", … +{len(sl) - MISS_LOG_MAX} more"
                if len(sl) > MISS_LOG_MAX else "")
        undo = "recall reap --restore <slug>" + (
            "" if scope == config.GLOBAL_SCOPE
            else f" --scope project --project-dir <{scope} repo>")
        out.append(Finding(
            "warn", scope,
            f"{len(sl)} archived note(s) reached for by hand within {days}d — "
            f"possible wrongful eviction: {shown}{more} (undo: {undo})"))
    return out


def run_checks(*, days: int = 8, do_notify: bool = False) -> int:
    """All checks over the global soul + every registered project. Exit code:
    0 clean, 1 warns only, 2 any error."""
    findings: list[Finding] = []

    g_dir = config.global_corpus_dir()
    findings += check_index(g_dir, config.index_path(config.GLOBAL_SCOPE),
                            config.GLOBAL_SCOPE)
    findings += check_rules(g_dir, config.GLOBAL_SCOPE)
    findings += check_miss_log(days=days)

    for proj in registry.list_projects():
        slug = config.project_slug(proj)
        if not proj.exists():
            findings.append(Finding("error", slug,
                                    f"registered path is gone: {proj}"))
            continue
        corpus = config.project_corpus_dir(proj)
        findings += check_index(corpus, config.index_path(slug), slug)
        findings += check_rules(corpus, slug)
        findings += check_memory_stub(memory_dir_for(proj), slug)
        findings += check_machine_lane(proj, slug, days=days)

    if not findings:
        print("[doctor] OK — no doctrine drift, indices young, rules clean.")
        return 0
    for f in findings:
        print(f"[doctor] {f.line()}")
    errors = any(f.level == "error" for f in findings)
    if do_notify:
        try:
            from recall.notify import notify_alert
            notify_alert(f"recall doctor: {len(findings)} finding(s)",
                         "\n".join(f.line() for f in findings),
                         priority="high" if errors else "normal")
        except Exception as e:  # noqa: BLE001 — a page failure never masks findings
            print(f"[doctor] WARN notify failed: {e}", file=sys.stderr)
    return 2 if errors else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="recall doctor",
        description="doctrine drift checks (memory stubs, machine-commit "
                    "lanes, indices, rules tier)")
    ap.add_argument("--days", type=int, default=8,
                    help="how far back to scan machine commits (default 8)")
    ap.add_argument("--notify", action="store_true",
                    help="send findings to the Telegram alert channel")
    a = ap.parse_args(argv)
    return run_checks(days=a.days, do_notify=a.notify)
