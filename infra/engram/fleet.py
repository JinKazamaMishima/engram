"""Engram fleet — parallel sibling sessions across repos, steered from one TUI.

A fleet member is a full ``AgentSDKDriver`` pointed at ANOTHER folder. That one
fact buys the whole design: the member resumes that folder's own session thread
(``SessionStore`` keys by cwd), writes that conversation's LiveBuffer (Brick 3:
buffer → working set → eviction-as-curation), and its per-turn recall injection
loads THAT repo's standing rules + corpus through the user-scope hooks — so
every member is memory-native, never an amnesiac worker. Coordination happens
through the operator, the repos, and recall itself; members don't message each
other (that's agent-teams territory, deliberately out of scope here).

The operator drives via ``/fleet`` in the TUI: spawn a member per repo, watch
the panel, peek at a transcript, steer one with a message, kill it. Each member
holds its folder's :class:`core.LaunchLock`, so a fleet member and a real
``engram`` opened in the same folder can never interleave one session.

Model-agnostic + UI-free on purpose: everything here consumes the
:class:`core.ModelDriver` Event contract and is unit-testable with a fake
driver; ``app.py`` owns the rendering.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from core import LOCK_DIR, PERSONA, Event, LaunchLock

# Appended to the member's persona so it knows what it is (and stays crisp —
# its output is read through a peek panel, not a full transcript).
MEMBER_NOTE = (
    "\n\nYou are running as Engram fleet member '{name}', working in {cwd} — a "
    "parallel session the operator steers from the operator's main Engram terminal. Work "
    "autonomously on what you're given, verify before claiming, and end each "
    "turn with a concise statement of what you did and found; he reads you "
    "through a compact panel, so lead with the result.")

STATUS_ICONS = {"starting": "◌", "idle": "✦", "working": "⏳",
                "dead": "✗", "killed": "✗"}
TRANSCRIPT_MAX = 600      # rendered chunks kept per member (view joins a tail)


def _default_factory(*, cwd: Path, name: str):
    from core import AgentSDKDriver
    return AgentSDKDriver(
        cwd=cwd,
        persona=PERSONA + MEMBER_NOTE.format(name=name, cwd=cwd))


def render_member_event(ev: Event) -> str:
    """One Event → the member-transcript chunk it contributes ('' = skip).
    Text chunks pass through raw (the peek view joins them); tool/recall events
    become their own compact lines; ephemeral status + panel-feed events skip."""
    if ev.kind == "text":
        return ev.text
    if ev.kind == "tool":
        return f"\n· {ev.text}\n"
    if ev.kind == "recall":
        return f"\n◆ recall: {ev.text or '(none)'}\n" if ev.text is not None else ""
    return ""


def _last_line(chunks) -> str:
    """The most recent non-empty line across the joined chunk tail — the panel's
    one-glance 'what is it doing right now'."""
    tail = "".join(list(chunks)[-40:])
    for line in reversed(tail.strip().splitlines()):
        line = line.strip().lstrip(">·* ").strip()
        if line:
            return line[:72]
    return ""


@dataclass
class FleetMember:
    name: str
    cwd: Path
    driver: object
    status: str = "starting"          # starting | idle | working | dead | killed
    transcript: deque = field(default_factory=lambda: deque(maxlen=TRANSCRIPT_MAX))
    inbox: "asyncio.Queue[str]" = field(default_factory=asyncio.Queue)
    pending: int = 0                  # queued messages not yet started
    turns: int = 0
    error: str = ""
    task: Optional[asyncio.Task] = None
    lock: Optional[LaunchLock] = None

    @property
    def last(self) -> str:
        return self._last if self._last else _last_line(self.transcript)

    _last: str = ""

    def tail(self, chars: int = 3000) -> str:
        return "".join(self.transcript)[-chars:]


class Fleet:
    """The member registry + per-member turn runners. ``on_change`` fires on any
    state transition the panel should repaint for (fail-soft: a broken callback
    never breaks a member)."""

    def __init__(self, *, driver_factory: Callable = _default_factory,
                 lock_root: Path = LOCK_DIR,
                 on_change: Optional[Callable[[], None]] = None) -> None:
        self.members: dict[str, FleetMember] = {}
        self._factory = driver_factory
        self._lock_root = lock_root
        self._on_change = on_change or (lambda: None)

    def _changed(self) -> None:
        try:
            self._on_change()
        except Exception:  # noqa: BLE001 — panel repaint must never kill a member
            pass

    # ---- lifecycle --------------------------------------------------------

    def spawn(self, path: str | Path, task: str = "",
              name: str = "") -> tuple[Optional[FleetMember], str]:
        """Create + start a member for ``path``; returns (member, message) —
        member None when refused (missing folder, live engram already there)."""
        cwd = Path(path).expanduser().resolve()
        if not cwd.is_dir():
            return None, f"no such folder: {cwd}"
        if any(m.cwd == cwd and m.status not in ("dead", "killed")
               for m in self.members.values()):
            return None, f"a fleet member already works in {cwd.name}"
        name = name or cwd.name
        base, n = name, 2
        while name in self.members:
            name, n = f"{base}-{n}", n + 1
        lock = LaunchLock(cwd, root=self._lock_root)
        owner = lock.acquire()
        if owner is not None:
            return None, (f"{cwd.name} is already driven by a live engram "
                          f"(pid {owner}) — steer that one instead")
        try:
            driver = self._factory(cwd=cwd, name=name)
        except Exception as e:  # noqa: BLE001 — a bad spawn must not leak the lock
            lock.release()
            return None, f"driver failed to start: {type(e).__name__}: {e}"
        member = FleetMember(name=name, cwd=cwd, driver=driver, lock=lock)
        self.members[name] = member
        if task.strip():
            member.inbox.put_nowait(task.strip())
            member.pending += 1
        member.task = asyncio.get_running_loop().create_task(self._run(member))
        self._changed()
        resumed = " (resumed that folder's thread)" if getattr(
            driver, "resumed", False) else ""
        return member, f"⚑ {name} spawned in {cwd}{resumed}"

    async def _run(self, m: FleetMember) -> None:
        """The member's whole life: connect, then serve the inbox one turn at a
        time, draining background stragglers between turns. Errors mark the
        member dead but never propagate into the TUI loop."""
        try:
            await m.driver.connect()
            m.status = "idle"
            self._changed()
            while True:
                text = await m.inbox.get()
                m.pending = max(0, m.pending - 1)
                m.status, m.turns = "working", m.turns + 1
                m.transcript.append(f"\n\n❯ {text}\n\n")
                self._changed()
                async for ev in m.driver.query(text):
                    chunk = render_member_event(ev)
                    if chunk:
                        m.transcript.append(chunk)
                    if ev.kind in ("tool", "status"):
                        m._last = (ev.text or "")[:72]
                        self._changed()
                # Paint late background results (sub-agents, workflows) while
                # nothing else is queued — a queued steer takes priority.
                while (getattr(m.driver, "has_background_tasks", False)
                       and m.inbox.empty()):
                    async for ev in m.driver.drain_background():
                        chunk = render_member_event(ev)
                        if chunk:
                            m.transcript.append(chunk)
                m._last = ""
                m.status = "idle"
                self._changed()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — a member dies alone, quietly
            m.status = "dead"
            m.error = f"{type(e).__name__}: {e}"
            self._changed()

    def send(self, name: str, text: str) -> str:
        m = self.members.get(name)
        if m is None:
            return f"no fleet member '{name}' — /fleet lists them"
        if m.status in ("dead", "killed"):
            return f"{name} is {m.status}" + (f" ({m.error})" if m.error else "")
        m.inbox.put_nowait(text)
        m.pending += 1
        self._changed()
        return (f"→ {name}: queued (working through "
                f"{m.pending} message{'s' if m.pending != 1 else ''})"
                if m.status == "working" else f"→ {name}: sent")

    async def kill(self, name: str) -> str:
        m = self.members.get(name)
        if m is None:
            return f"no fleet member '{name}'"
        if m.task is not None:
            m.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await m.task
        with contextlib.suppress(Exception):
            await m.driver.disconnect()
        if m.lock is not None:
            m.lock.release()
        m.status = "killed"
        del self.members[name]           # frees the name; transcript goes with it
        self._changed()
        return f"✗ {name} stopped (its session file persists — respawn resumes it)"

    async def shutdown(self) -> None:
        """Kill everyone (TUI teardown) — locks released, no lingering drivers."""
        for name in list(self.members):
            await self.kill(name)

    # ---- views ------------------------------------------------------------

    def rows(self) -> list[dict]:
        return [{"name": m.name, "dir": m.cwd.name, "status": m.status,
                 "pending": m.pending, "turns": m.turns, "last": m.last,
                 "error": m.error}
                for m in self.members.values()]


def render_fleet_line(rows: list[dict]) -> str:
    """The one-line fleet strip above the prompt ('' when no members). Pure —
    unit-testable without Textual."""
    if not rows:
        return ""
    bits = []
    for r in rows:
        icon = STATUS_ICONS.get(r["status"], "·")
        bit = f"⚑ {r['name']} {icon}"
        if r["status"] == "working" and r.get("last"):
            bit += f" {r['last'][:34]}"
        if r.get("pending"):
            bit += f" (+{r['pending']} queued)"
        if r["status"] == "dead" and r.get("error"):
            bit += f" {r['error'][:40]}"
        bits.append(bit)
    return "   ".join(bits)
