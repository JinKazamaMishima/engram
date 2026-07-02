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


def _parse_ts(ev: dict) -> datetime | None:
    raw = ev.get("timestamp")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def iter_exchanges(path: Path, target: date | None, tz: ZoneInfo = ET
                   ) -> Iterator[Exchange]:
    """Yield denoised user/assistant exchanges from one transcript. With a
    ``target`` date, keep only events on that day (in ``tz``); with
    ``target is None``, keep every properly-dated exchange — the whole session
    end to end, for session-scoped curation. Malformed lines and events are
    skipped defensively rather than raising."""
    try:
        raw_lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    session_id = path.stem
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


def build_bundle(paths: list[Path], target: date | None, tz: ZoneInfo = ET, *,
                 max_chars_per_msg: int = 6000) -> tuple[str, BundleStats]:
    """Group surviving exchanges by session, drop sessions with no genuine user
    turn (headless skill runs), and render a readable markdown bundle. ``target``
    scopes the exchanges by day; ``target is None`` keeps every dated exchange in
    ``paths`` (used to curate a single session end to end).

    Returns ``(bundle_text, stats)``; ``stats.exchanges == 0`` means there was
    nothing worth curating."""
    by_session: dict[str, list[Exchange]] = {}
    order: list[str] = []
    for p in paths:
        for ex in iter_exchanges(p, target, tz):
            if ex.session_id not in by_session:
                by_session[ex.session_id] = []
                order.append(ex.session_id)
            by_session[ex.session_id].append(ex)

    chunks: list[str] = []
    n_sessions = n_exchanges = 0
    for sid in order:
        exchanges = sorted(by_session[sid], key=lambda e: e.ts)
        if not any(e.role == "user" for e in exchanges):
            continue  # automated/headless run, no human prompt -> not gold
        n_sessions += 1
        chunks.append(f"## Session {sid[:8]} — {len(exchanges)} turns")
        for e in exchanges:
            n_exchanges += 1
            body = e.text
            if len(body) > max_chars_per_msg:
                body = body[:max_chars_per_msg].rstrip() + "\n…[truncated]"
            chunks.append(
                f"\n### {'USER' if e.role == 'user' else 'ASSISTANT'}\n{body}")
        chunks.append("")

    text = ("\n".join(chunks).strip() + "\n") if chunks else ""
    return text, BundleStats(sessions=n_sessions, exchanges=n_exchanges,
                             chars=len(text))
