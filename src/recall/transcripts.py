"""Discover and denoise a project's Claude Code conversation transcripts into a
clean user<->assistant bundle the ``curate`` step can mine.

Claude Code persists each session as newline-delimited JSON at
``~/.claude/projects/<encoded-repo-path>/<session-uuid>.jsonl`` — one event
object per line (``user`` / ``assistant`` / ``system`` + harness bookkeeping).
We keep only genuine human<->assistant prose: the gold is the reasoning fleshed
out together, not tool output, not the model's private ``thinking``, and not the
automated headless skill runs that also land in this directory. A session with
no surviving *human* turn is treated as one of those automated runs and dropped.

Pure, stdlib only (``json`` + ``zoneinfo``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DEFAULT_PROJECTS_BASE = Path.home() / ".claude" / "projects"

# Wrapper blocks injected by the harness, not the user — removed whole
# (open tag .. close tag, content included) wherever they appear.
_WRAPPER_SPAN_RE = re.compile(
    r"<(system-reminder|local-command-caveat|local-command-stdout|"
    r"command-name|command-message|command-args)>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# A line of Honcho-injected "additional context" (legacy; recall replaced it).
_HONCHO_LINE_RE = re.compile(r"^\s*\[Honcho Memory.*$", re.MULTILINE)
# Any stray, unpaired wrapper tag left behind.
_STRAY_TAG_RE = re.compile(
    r"</?(system-reminder|local-command-caveat|local-command-stdout|"
    r"command-name|command-message|command-args)>",
    re.IGNORECASE,
)
# A headless run of one of recall's OWN memory skills (`claude -p /curate-memory`,
# `/dream`, `/reconsolidate-memory`) lands in the transcript dir with the harness
# command wrapper as its first human turn, then injects the skill's prose as a
# later user turn — which would otherwise survive denoising and get mined as if it
# were a real conversation (the curator curating itself). The wrapper is the tell.
_SKILL_RUN_RE = re.compile(
    r"<command-name>\s*/?(curate-memory|dream|reconsolidate-memory)\b",
    re.IGNORECASE,
)


def project_transcript_dir(repo_path: str | Path,
                           base: Path = DEFAULT_PROJECTS_BASE) -> Path:
    """Map an absolute repo path to its Claude Code transcript directory.

    ``/home/user/repos/myproject`` -> ``<base>/-home-user-repos-myproject``
    (every ``/`` becomes ``-``, including the leading one)."""
    encoded = str(Path(repo_path).resolve()).replace("/", "-")
    return base / encoded


@dataclass(frozen=True)
class Exchange:
    role: str          # "user" | "assistant"
    text: str
    session_id: str
    ts: datetime       # event timestamp, tz-aware (UTC as stored)


@dataclass(frozen=True)
class BundleStats:
    sessions: int
    exchanges: int
    chars: int


def _strip_noise(text: str) -> str:
    text = _WRAPPER_SPAN_RE.sub("", text)
    text = _HONCHO_LINE_RE.sub("", text)
    text = _STRAY_TAG_RE.sub("", text)
    return text.strip()


def _is_noise_only(text: str) -> bool:
    """A human turn that carries no insight: empty after denoising, or a bare
    slash-command invocation (e.g. a headless ``/some-skill``)."""
    s = text.strip()
    if not s:
        return True
    if s.startswith("/") and len(s.split()) == 1:
        return True  # lone "/some-skill" token, not prose that opens with "/"
    return False


def _content_to_text(content) -> str:
    """Flatten a message ``content`` (str, or list-of-blocks) to plain text,
    keeping only human-facing ``text`` blocks — drop ``tool_use`` /
    ``tool_result`` / ``thinking`` / images."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _is_recall_skill_run(raw_lines: list[str]) -> bool:
    """True if this transcript is a headless run of one of recall's own memory
    skills (``claude -p /curate-memory`` / ``/dream`` / ``/reconsolidate-memory``).
    Such runs land in the project transcript dir and — because the skill prose is
    injected as a later user turn — would otherwise survive denoising and be mined
    as if they were a human conversation (self-curation). The tell is the harness
    command wrapper on the FIRST human turn."""
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or ev.get("type") != "user":
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        return bool(_SKILL_RUN_RE.search(_content_to_text(msg.get("content"))))
    return False


def _parse_iso(raw) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_ts(ev: dict) -> datetime | None:
    return _parse_iso(ev.get("timestamp"))


def _in_window(ts: datetime, since: datetime | None,
               until: datetime | None) -> bool:
    """The incremental-slice window: ``since < ts <= until``. ``since`` is the
    watermark (strictly after — the watermarked exchange itself was already
    curated); ``until`` is the cooled edge (inclusive — it IS the last row the
    caller wants curated)."""
    if since is not None and ts <= since:
        return False
    if until is not None and ts > until:
        return False
    return True


def iter_exchanges(path: Path, target: date | None, tz: ZoneInfo = ET, *,
                   since: datetime | None = None,
                   until: datetime | None = None) -> Iterator[Exchange]:
    """Yield denoised user/assistant exchanges from one transcript. With a
    ``target`` date, keep only events on that day (in ``tz``); with
    ``target is None``, keep every properly-dated exchange — the whole session
    end to end, for session-scoped curation. ``since``/``until`` additionally
    slice by timestamp (``since < ts <= until``) for incremental curation.
    Malformed lines and events are skipped defensively rather than raising."""
    try:
        raw_lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    session_id = path.stem
    if _is_recall_skill_run(raw_lines):
        return  # headless run of recall's own memory skill — machinery, not gold
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or ev.get("type") not in ("user",
                                                               "assistant"):
            continue
        ts = _parse_ts(ev)
        if ts is None:
            continue
        if target is not None and ts.astimezone(tz).date() != target:
            continue
        if not _in_window(ts, since, until):
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        text = _strip_noise(_content_to_text(msg.get("content")))
        if ev["type"] == "user":
            if _is_noise_only(text):
                continue
        elif not text:
            continue
        yield Exchange(role=ev["type"], text=text,
                       session_id=session_id, ts=ts)


def iter_buffer_exchanges(path: Path, *,
                          since: datetime | None = None,
                          until: datetime | None = None) -> Iterator[Exchange]:
    """Yield denoised exchanges from an Engram LiveBuffer JSONL — the harness's
    append-only tier-1 STM, rows ``{"convo_id","seq","ts","role","text"}``.
    The buffer holds only real user/assistant prose (no tool spam, no headless
    skill wrappers, so no ``_is_recall_skill_run`` drop), but the same noise
    discipline applies. A third role, ``perception``, carries gate-verified
    sensor events from the perceiving loop's own buffer (step-5 eviction) —
    already corroboration-filtered at write time, so it passes like assistant
    prose. Rows yield in ``(seq, ts)`` order; a partial line (a crash
    mid-append) or garbled row is skipped, never fatal. ``since``/``until``
    slice ``since < ts <= until`` — the incremental-eviction window."""
    try:
        raw_lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    convo_id = path.stem
    rows: list[tuple[int, datetime, str, str]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail write — tolerate, the next row is intact
        if not isinstance(row, dict) or row.get("role") not in (
                "user", "assistant", "perception"):
            continue
        ts = _parse_iso(row.get("ts"))
        if ts is None or not _in_window(ts, since, until):
            continue
        text = _strip_noise(str(row.get("text") or ""))
        if row["role"] == "user":
            if _is_noise_only(text):
                continue
        elif not text:
            continue
        try:
            seq = int(row.get("seq", 0))
        except (TypeError, ValueError):
            seq = 0
        rows.append((seq, ts, row["role"], text))
    rows.sort(key=lambda r: (r[0], r[1]))
    for seq, ts, role, text in rows:
        yield Exchange(role=role, text=text, session_id=convo_id, ts=ts)


def buffer_last_ts(path: Path, *, until: datetime | None = None
                   ) -> datetime | None:
    """Timestamp of the newest surviving buffer exchange (≤ ``until`` if given)
    — what a successful incremental curation advances the watermark to."""
    last: datetime | None = None
    for ex in iter_buffer_exchanges(path, until=until):
        if last is None or ex.ts > last:
            last = ex.ts
    return last


def transcript_last_ts(path: Path, *, until: datetime | None = None
                       ) -> datetime | None:
    """``buffer_last_ts``'s Claude Code–transcript twin: the newest surviving
    exchange ≤ ``until`` — the watermark value for incremental --session passes."""
    last: datetime | None = None
    for ex in iter_exchanges(path, None, until=until):
        if last is None or ex.ts > last:
            last = ex.ts
    return last


def discover_transcripts(transcript_dir: str | Path, target: date,
                         tz: ZoneInfo = ET) -> list[Path]:
    """``.jsonl`` files that *could* hold target-date events, pre-filtered
    cheaply by mtime — a file with an event on ``target`` must have been
    modified on/after ``target``. Off-date events are filtered out per-event
    later, so this only avoids reading clearly-too-old files."""
    d = Path(transcript_dir)
    if not d.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(d.glob("*.jsonl")):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=tz).date()
        except OSError:
            continue
        if mtime >= target:
            out.append(p)
    return out


def session_transcript_path(transcript_dir: str | Path, session_id: str) -> Path:
    """Path to one session's transcript (``<dir>/<session_id>.jsonl``). The file
    may not exist (an unknown / not-yet-flushed session); callers check."""
    return Path(transcript_dir) / f"{session_id}.jsonl"


def session_date(path: Path, tz: ZoneInfo = ET) -> date | None:
    """The ``tz`` date of a session's last real exchange — the day it should be
    filed under (manifest date + dynamics ``last_used``). ``None`` if the file
    has no surviving human/assistant turn."""
    last: datetime | None = None
    for ex in iter_exchanges(path, None, tz):
        if last is None or ex.ts > last:
            last = ex.ts
    return last.astimezone(tz).date() if last is not None else None


def _render_bundle(by_session: dict[str, list[Exchange]], order: list[str],
                   max_chars_per_msg: int) -> tuple[str, BundleStats]:
    """The one canonical bundle renderer — every reader (Claude Code transcript,
    Engram LiveBuffer) funnels through it so the curator sees identical markdown.
    Drops sessions with no genuine user turn (headless runs → not gold)."""
    chunks: list[str] = []
    n_sessions = n_exchanges = 0
    for sid in order:
        exchanges = sorted(by_session[sid], key=lambda e: e.ts)
        # Genuine signal = a human prompt OR gate-verified perception rows
        # (real-world events). A session with neither is an automated/headless
        # run -> not gold.
        if not any(e.role in ("user", "perception") for e in exchanges):
            continue
        n_sessions += 1
        chunks.append(f"## Session {sid[:8]} — {len(exchanges)} turns")
        for e in exchanges:
            n_exchanges += 1
            body = e.text
            if len(body) > max_chars_per_msg:
                body = body[:max_chars_per_msg].rstrip() + "\n…[truncated]"
            chunks.append(f"\n### {e.role.upper()}\n{body}")
        chunks.append("")

    text = ("\n".join(chunks).strip() + "\n") if chunks else ""
    return text, BundleStats(sessions=n_sessions, exchanges=n_exchanges,
                             chars=len(text))


def build_bundle(paths: list[Path], target: date | None, tz: ZoneInfo = ET, *,
                 max_chars_per_msg: int = 6000,
                 since: datetime | None = None,
                 until: datetime | None = None) -> tuple[str, BundleStats]:
    """Group surviving exchanges by session, drop sessions with no genuine user
    turn (headless skill runs), and render a readable markdown bundle. ``target``
    scopes the exchanges by day; ``target is None`` keeps every dated exchange in
    ``paths`` (used to curate a single session end to end); ``since``/``until``
    slice by timestamp for incremental curation.

    Returns ``(bundle_text, stats)``; ``stats.exchanges == 0`` means there was
    nothing worth curating."""
    by_session: dict[str, list[Exchange]] = {}
    order: list[str] = []
    for p in paths:
        for ex in iter_exchanges(p, target, tz, since=since, until=until):
            if ex.session_id not in by_session:
                by_session[ex.session_id] = []
                order.append(ex.session_id)
            by_session[ex.session_id].append(ex)
    return _render_bundle(by_session, order, max_chars_per_msg)


def build_buffer_bundle(path: Path, *, max_chars_per_msg: int = 6000,
                        since: datetime | None = None,
                        until: datetime | None = None
                        ) -> tuple[str, BundleStats]:
    """Render a bundle from ONE Engram LiveBuffer file — same markdown the curator
    already reads (shared ``_render_bundle``). ``since``/``until`` carve the
    incremental-eviction window; ``stats.exchanges == 0`` means no new tail."""
    exchanges = list(iter_buffer_exchanges(path, since=since, until=until))
    if not exchanges:
        return "", BundleStats(sessions=0, exchanges=0, chars=0)
    return _render_bundle({path.stem: exchanges}, [path.stem],
                          max_chars_per_msg)
