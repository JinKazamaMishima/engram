"""The always-on standing-rules tier — ``kind: rule`` notes.

Rules are the zero-miss-tolerance channel: behavioral directives (identity,
protocol, workflow) where "hope retrieval surfaces it" is a regression. They
live as ordinary corpus notes — git-versioned, provenance-carrying, findable
by ``recall_search``, reconsolidation-visible — but they are never *retrieval*
targets: the inject hook prepends every active rule to every prompt (soul
rules in any folder, project rules inside that project) and drops them from
the retrieved-hits list, where they would only waste a slot.

Deliberately torch-free and index-free: a plain frontmatter scan over the two
corpus dirs, so the rules channel survives an index/daemon/GPU outage — the
same guarantee the old MEMORY.md hot-load gave, without a second store that
can diverge from the corpus.

Only the operator promotes a rule. The curate/reconsolidate wrappers fail any
run whose manifest touches a ``kind: rule`` note (``validate_manifest_against``)
and the weekly reconsolidation worklist excludes them, so the machine writers
can neither author nor edit the channel that steers them.
"""
from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

from recall import config
from recall.schema import CurationSchemaError, KnowledgeNote

RULE_KIND = "rule"
BUDGET_ENV = "RECALL_RULES_BUDGET_CHARS"
DEFAULT_BUDGET_CHARS = 4000

# Cheap prefilter before the real parse: a kind:rule declaration at line start
# (the frontmatter grammar is flat, so this matches quoted and unquoted). A
# false positive (e.g. the phrase in a body code block) costs one parse; the
# authoritative check is the parsed ``note.kind``.
_KIND_RE = re.compile(r"^kind:\s*['\"]?rule\b", re.MULTILINE | re.IGNORECASE)

_HEADER = (
    "## Standing rules — always on\n"
    "_Operator-promoted directives (`kind: rule` notes: soul + this project), "
    "injected every turn. Precedence: the operator's live word > CLAUDE.md > "
    "these rules > retrieved notes (background). Facts never belong in "
    "CLAUDE.md or auto-memory — write a corpus note instead._")


def scan_rules(corpus_dir: str | Path, *, today: date | None = None
               ) -> tuple[list[KnowledgeNote], list[str]]:
    """All ACTIVE ``kind: rule`` notes in one corpus dir (sorted by slug), plus
    the slugs of rule-looking notes that failed to parse. A broken rule note is
    a silent rule outage — callers must surface it, never swallow it. Active =
    not superseded and the validity window covers today. Missing dir → ([], [])."""
    d = Path(corpus_dir)
    if not d.is_dir():
        return [], []
    iso = (today or date.today()).isoformat()
    active: list[KnowledgeNote] = []
    broken: list[str] = []
    for path in sorted(d.glob("*.md")):
        try:
            text = path.read_text()
        except OSError:
            continue
        if not _KIND_RE.search(text):
            continue
        try:
            note = KnowledgeNote.parse(text, expect_slug=path.stem)
        except CurationSchemaError:
            broken.append(path.stem)
            continue
        if note.kind != RULE_KIND:
            continue  # prefilter false positive — the phrase appeared in a body
        if note.superseded:
            continue
        if note.valid_to and note.valid_to < iso:
            continue  # retired rule
        if note.valid_from and note.valid_from > iso:
            continue  # not yet in force
        active.append(note)
    return active, broken


def rules_budget() -> int:
    try:
        return int(os.environ.get(BUDGET_ENV, "") or DEFAULT_BUDGET_CHARS)
    except ValueError:
        return DEFAULT_BUDGET_CHARS


def rules_context(project_dir: str | Path, *, today: date | None = None,
                  budget: int | None = None) -> str | None:
    """The injectable always-on block for a session in ``project_dir``: the
    soul's rules first (identity core, present in every folder), then the
    project corpus's own. Returns None when there are no rules and no broken
    ones — a repo with no corpus stays silent. Char-budgeted (``budget`` arg >
    $RECALL_RULES_BUDGET_CHARS > default): overflow drops whole rules
    deterministically (soul-first, slug order) and says how many it dropped —
    a silently thinned rule channel would be a regression, not a savings."""
    budget = rules_budget() if budget is None else budget
    sections: list[tuple[str, list[KnowledgeNote]]] = []
    broken: list[str] = []

    soul, b = scan_rules(config.global_corpus_dir(), today=today)
    broken += b
    if soul:
        sections.append(("Global / soul", soul))
    proj, b = scan_rules(config.project_corpus_dir(project_dir), today=today)
    broken += b
    if proj:
        sections.append(("This project", proj))
    if not sections and not broken:
        return None

    lines = [_HEADER]
    used = len(_HEADER)
    omitted = 0
    for title, notes in sections:
        head = f"\n### {title}"
        body = []
        for n in notes:
            line = f"- **{n.slug}** — {n.description}"
            if used + len(head) + len(line) > budget:
                omitted += 1
                continue
            body.append(line)
            used += len(line)
        if body:
            lines.append(head)
            lines.extend(body)
            used += len(head)
    if omitted:
        lines.append(f"\n⚠ {omitted} rule(s) omitted — over {BUDGET_ENV}"
                     f"={budget}; `recall doctor` has the full list.")
    if broken:
        lines.append(f"\n⚠ rule note(s) failed to parse — SILENT RULE OUTAGE: "
                     f"{', '.join(sorted(broken))} — fix the note or run "
                     f"`recall doctor`.")
    return "\n".join(lines)
