"""working_set.py — tier 2 of the continuous-STM stack (Brick 3).

A deterministic, no-LLM ``<working-memory>`` block, RE-DERIVED FROM the immutable
LiveBuffer (+ the recent activation log) every turn and injected into the
model-only prepend. This is the anti-drift mechanism: rather than trust the
closed model's recursive self-summary once its native context compacts, we
re-ground it each turn from source — the raw recent turns (which nothing else
re-injects) plus the notes this conversation has recently surfaced.

Pure string assembly over a handful of small on-disk reads (~1ms), and
FAIL-OPEN: any error returns ``""`` and the turn proceeds exactly as before
Brick 3. Never calls a model. Validity-aware — a note whose ``valid_to`` has
passed is dropped, so a reversed fact never rides in as current.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

WM_TURNS = int(os.environ.get("ENGRAM_WM_TURNS", "12"))
WM_NOTES = int(os.environ.get("ENGRAM_WM_NOTES", "5"))
WM_CHAR_BUDGET = int(os.environ.get("ENGRAM_WM_CHAR_BUDGET", "4000"))
_PER_TURN_CAP = 800   # a single pasted turn can't eat the whole budget

_ROLE = {"user": "OPERATOR", "assistant": "ENGRAM"}


def _recent_slugs(cwd: Path, limit: int) -> list[tuple[str, str]]:
    """(scope, slug) pairs most-recently activated first, deduped, across the
    project + global logs — the notes THIS conversation has been surfacing.
    Fail-open to []."""
    try:
        from recall import activation, config
        scopes = [config.project_slug(cwd), config.GLOBAL_SCOPE]
    except Exception:  # noqa: BLE001
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for scope in scopes:
        try:
            events = activation.read_events(scope)
        except Exception:  # noqa: BLE001
            continue
        # newest last in the log → walk it backwards for most-recent-first
        for ev in reversed(events):
            slug = ev.get("slug")
            if slug and slug not in seen:
                seen.add(slug)
                out.append((scope, slug))
    return out[: limit * 3]   # over-collect; validity/read filtering trims below


def _note_line(cwd: Path, scope: str, slug: str, today: str) -> Optional[str]:
    """One ``- **slug** — description`` line, or None if the note is missing,
    malformed, or historical (``valid_to`` in the past). Fail-open: a read
    error drops just this note, never the block."""
    try:
        from recall import config
        from recall.schema import KnowledgeNote
        d = (config.global_corpus_dir() if scope == config.GLOBAL_SCOPE
             else config.project_corpus_dir(cwd))
        note = KnowledgeNote.parse((Path(d) / f"{slug}.md").read_text(),
                                   expect_slug=slug)
    except Exception:  # noqa: BLE001 — missing/malformed → skip this note only
        return None
    if note.valid_to and note.valid_to < today:
        return None   # reversed fact — never surface it as current here
    return f"- **{slug}** — {note.description}"


def _stamp(row: dict, clock: datetime) -> str:
    """Local-time stamp for a buffer row: '12:08' if same-day as ``clock``,
    'Jul 7 16:21' otherwise — so a conversation spanning days SHOWS it, and
    'earlier in context' can never masquerade as 'earlier today'. Fail-open to
    '' (no stamp) on a missing/malformed ts."""
    try:
        dt = datetime.fromisoformat(str(row["ts"])).astimezone()
        if dt.date() == clock.date():
            return dt.strftime("%H:%M")
        return f"{dt.strftime('%b')} {dt.day} {dt.strftime('%H:%M')}"
    except Exception:  # noqa: BLE001
        return ""


def _turn_line(row: dict, clock: datetime) -> str:
    role = _ROLE.get(row.get("role"), "?")
    text = str(row.get("text") or "").strip().replace("\n", " ")
    if len(text) > _PER_TURN_CAP:
        text = text[:_PER_TURN_CAP].rstrip() + " …"
    ts = _stamp(row, clock)
    tag = f"[{role} @ {ts}]" if ts else f"[{role}]"
    return f"{tag} {text}"


def build_working_memory(buffer, cwd: Path, *, turns: int = WM_TURNS,
                         notes: int = WM_NOTES, budget: int = WM_CHAR_BUDGET,
                         now: Optional[date] = None) -> str:
    """The tier-2 block, or ``""`` (nothing to ground yet, disabled, or any
    error). ``buffer`` is the driver's LiveBuffer; ``cwd`` scopes the notes.

    Budgeting keeps the SIGNAL: the newest turn always survives (drop oldest
    turns first), and the top-activated note always survives (drop lowest
    first) — a hard char cap that degrades gracefully, never a hard failure."""
    try:
        if buffer is None or not getattr(buffer, "enabled", False):
            return ""
        # The wall clock is part of the grounding: a Brick-3 conversation spans
        # days, and without a per-turn NOW the model narrates time from its
        # position in context ("this morning" for yesterday's work). ``clock``
        # drives the NOW header + per-turn stamps; ``now`` (a date) still
        # overrides for the validity check, keeping the old test seam.
        clock = datetime.now().astimezone()
        today = (now or clock.date()).isoformat()

        tail = buffer.tail(turns)          # oldest→newest; excludes the live turn
        turn_lines = [_turn_line(r, clock) for r in tail]

        note_lines: list[str] = []
        for scope, slug in _recent_slugs(Path(cwd), notes):
            line = _note_line(Path(cwd), scope, slug, today)
            if line:
                note_lines.append(line)
            if len(note_lines) >= notes:
                break

        if not turn_lines and not note_lines:
            return ""

        # Assemble under budget. Turns are the irreplaceable half (nothing else
        # re-injects them post-compaction), so they get first claim; notes fill
        # the remainder. Within each, keep the most valuable and drop from the
        # weak end (oldest turn / lowest-activation note).
        header = ("<working-memory>\n"
                  f"NOW: {clock.strftime('%a %Y-%m-%d %H:%M %Z')}. Recent context "
                  "for THIS conversation, re-grounded from source each turn — "
                  "trust it over any summary if they disagree. Turn stamps are "
                  "local time: date-stamped turns are NOT from today.\n")
        footer = "\n</working-memory>"
        room = max(0, budget - len(header) - len(footer))

        turns_block = _fit_section("## Recent turns", turn_lines,
                                   keep="last", room=room)
        room -= len(turns_block)
        notes_block = _fit_section("## Active notes", note_lines,
                                   keep="first", room=max(0, room))

        body = turns_block + notes_block
        if not body.strip():
            return ""
        return header + body + footer
    except Exception:  # noqa: BLE001 — memory is a passenger, never the driver
        return ""


def _fit_section(title: str, lines: list[str], *, keep: str, room: int) -> str:
    """Render ``title`` + ``lines`` within ``room`` chars. ``keep='last'`` drops
    from the FRONT (oldest turns go first, newest survives); ``keep='first'``
    drops from the BACK (top notes survive). Returns "" if nothing fits."""
    if not lines or room <= 0:
        return ""
    kept = list(lines)
    while kept:
        block = "\n" + title + "\n" + "\n".join(kept) + "\n"
        if len(block) <= room:
            return block
        # drop one from the weak end and retry
        kept = kept[1:] if keep == "last" else kept[:-1]
    return ""
