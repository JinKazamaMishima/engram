#!/usr/bin/env python3
"""Engram core — the model-driver layer of Engram's OWN harness.

Engram runs on your Claude Pro/Max **subscription** (never an API key) via the
Claude Agent SDK, which spawns the logged-in `claude` CLI. This module is the
shared core every Engram front-end (terminal TUI, the Telegram bridge, the future
perceiving loop) sits on: a thin ``ModelDriver`` seam + the Agent-SDK driver +
the Engram persona.

The seam is the point. A future local self-hosted model can drop in here as a
second ``ModelDriver`` (an OpenAI-compatible client), and every front-end keeps
working unchanged. The durable layers — persona, memory wiring, tools, the
front-ends — are model-agnostic; only the driver swaps.
Today there is exactly one driver: ``AgentSDKDriver``.

Memory + skills come FOR FREE: running with ``setting_sources=["project"]`` and
``cwd`` = the recall repo loads CLAUDE.md, the skills, and the recall hook — so
every turn auto-injects relevant curated memory, exactly like the terminal /
Telegram Engram. (Ported from the proven ``infra/telegram/agent_bridge.py``.)
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

# --- auth: subscription vs API key (BEFORE the SDK can spawn `claude`) --------
# The SDK spawns `claude`, which prefers ANTHROPIC_API_KEY over the subscription
# login when a key is present. By default Engram respects whatever you configured
# — an API key if one is set, otherwise your `claude` subscription login. Set
# ENGRAM_FORCE_SUBSCRIPTION=1 to always bill to a Pro/Max subscription by stripping
# any key first (handy if a key is in your environment but you want to use Pro/Max).
_STRIPPED_API_KEY = False
if os.environ.get("ENGRAM_FORCE_SUBSCRIPTION", "").lower() in ("1", "true", "yes"):
    _STRIPPED_API_KEY = os.environ.pop("ANTHROPIC_API_KEY", None) is not None

from claude_agent_sdk import (  # noqa: E402 — after the env strip, on purpose
    TERMINAL_TASK_STATUSES,
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    query as _sdk_query,
)

REPO = Path(os.environ.get("RECALL_REPO") or Path(__file__).resolve().parents[2])
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
ENGRAM_MODEL = os.environ.get("ENGRAM_MODEL", "opus[1m]")
ENGRAM_EFFORT = os.environ.get("ENGRAM_EFFORT", "max")   # max always; downgrade via /effort
# Like Claude Code: Engram operates in the directory it was launched from (override
# with ENGRAM_CWD). The recall memory hook is global (~/.claude/settings.json), so
# loading "user" settings (below) means memory follows Engram into any folder.
ENGRAM_CWD = Path(os.environ.get("ENGRAM_CWD") or Path.cwd())

# Where Engram persists its resumable session ids — one file per launch directory,
# so a fresh `engram` resumes the conversation from the SAME folder (the way Claude
# Code keys sessions by project dir). Under the shared recall data root.
DATA_ROOT = Path(os.environ.get("RECALL_DATA_ROOT",
                                os.path.expanduser("~/.local/share/recall")))
SESSION_DIR = Path(os.environ.get("ENGRAM_SESSION_DIR", str(DATA_ROOT / "engram" / "sessions")))
# Per-launch-directory advisory lock (one file per cwd) so two `engram` in the SAME folder
# don't both resume + drive one session id and interleave their turns into a corrupt thread.
LOCK_DIR = Path(os.environ.get("ENGRAM_LOCK_DIR", str(DATA_ROOT / "engram" / "locks")))

# Reasoning effort levels, low→high, matching Claude Code's /effort.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Permission modes Engram cycles between with shift+tab, like Claude Code. "Regular" =
# bypassPermissions (Engram acts freely; the persona is the only guardrail). "Plan" =
# the SDK's read-only plan mode (investigate + propose, make no changes). The SDK also
# accepts default/acceptEdits/dontAsk/auto, but those need an interactive permission UI
# this harness doesn't have — so the front-end toggle stays between these two.
REGULAR_MODE = "bypassPermissions"
PLAN_MODE = "plan"
PERMISSION_MODES = (
    "default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto")

# The two CLI-native tools that need the operator IN the loop: ExitPlanMode (present a
# plan, wait for approval) and AskUserQuestion (offer options, wait for a pick). They are
# invisible to the SDK except through the `can_use_tool` permission channel — the CLI
# routes BOTH through it even under bypassPermissions (verified), while ordinary tools
# auto-allow without a round-trip. So the driver intercepts exactly these, hands them to a
# front-end `on_interaction` handler, and renders them richly instead of as a bare status
# blip (we also suppress their tool-use markers in the stream, since the card IS the render).
_INTERACTIVE_TOOLS = {"ExitPlanMode", "AskUserQuestion"}

# When the model auto-delegates, the CLI's Agent tool runs the sub-agent ASYNC: its
# progress/completion arrive as Task* messages AFTER the parent turn's ResultMessage.
# We keep reading the stream until they finish; if a pending sub-agent goes fully silent
# this long (seconds), we stop waiting and detach it rather than hang the turn.
SUBAGENT_IDLE_TIMEOUT = float(os.environ.get("ENGRAM_SUBAGENT_IDLE_TIMEOUT", "180"))

PERSONA = (
    "You are Engram, a persistent-memory assistant running in the user's terminal. "
    "Your defining trait is long-term memory: a curated knowledge corpus that a "
    "retrieval hook auto-injects into every turn, so you carry context across "
    "sessions instead of starting cold. You also load the working directory's full "
    "project brain — its CLAUDE.md and skills. This is a terminal: full markdown and "
    "real depth are welcome when useful, but stay crisp. Before any consequential or "
    "hard-to-reverse action (writing to the memory corpus, state-changing shell "
    "commands, git commits or pushes, anything outward-facing), propose it and wait "
    "for the user's explicit go. This is Engram's default persona — point "
    "ENGRAM_PERSONA_FILE at your own to give it a different character."
)
_PERSONA_FILE = os.environ.get("ENGRAM_PERSONA_FILE", "")
if _PERSONA_FILE:
    try:
        PERSONA = Path(_PERSONA_FILE).read_text().strip()
    except OSError:
        pass

# Engram's sub-agent roster, passed to the SDK as ``agents=`` so the model can
# delegate via the Task tool (auto) and the /agent command can target one by name.
# We define them explicitly because the CLI does NOT expose its built-in agent types
# to an SDK-driven session (verified: a bare session reports zero subagents) — so to
# "mirror Claude Code's agents" we recreate the three common ones. Keys ARE the
# ``subagent_type`` names (kept matching app.py's SUBAGENTS); ``model="inherit"`` runs
# each on whatever model Engram is currently using.
SUBAGENT_DEFS = {
    "Explore": AgentDefinition(
        description=("Read-only search agent for broad, fan-out exploration. Locates code, "
                     "maps how something works, finds all usages — reads and reports, never edits."),
        prompt=("You are a read-only exploration sub-agent. Search broadly and efficiently to "
                "answer the question you are given, using Read, Grep, Glob, and read-only Bash "
                "(ls/find/rg). Never modify anything. Return a concise, well-organized findings "
                "report that LEADS with the answer, then cites concrete evidence as path:line."),
        tools=["Read", "Grep", "Glob", "Bash"],
        model="inherit",
        background=False,   # run synchronously so the result returns within the turn
    ),
    "Plan": AgentDefinition(
        description=("Software-architect sub-agent that designs an implementation plan without "
                     "writing code — step-by-step plans, critical files, trade-offs."),
        prompt=("You are a planning sub-agent. Investigate the relevant code (read-only) and "
                "design a clear, step-by-step implementation plan for the task. Identify the "
                "critical files to change and call out trade-offs and risks. Do NOT edit files — "
                "output the plan only."),
        tools=["Read", "Grep", "Glob"],
        model="inherit",
        background=False,
    ),
    "general-purpose": AgentDefinition(
        description=("General-purpose sub-agent for researching complex questions and executing "
                     "multi-step tasks end-to-end, including making edits when asked."),
        prompt=("You are a general-purpose sub-agent. Handle the task end-to-end: investigate as "
                "needed and carry it through to completion, using whatever tools are appropriate. "
                "Be autonomous and return a concise summary of what you did and found."),
        model="inherit",
        background=False,
    ),
}


@dataclass
class Event:
    """One streamed unit of a turn. ``kind``: 'text' | 'tool'."""
    kind: str
    text: str


# --- presentation helpers (model-agnostic; shared by every front-end) ---------

def _tool_label(block) -> str:
    """UI label for a tool-use block. A sub-agent delegation (the ``Agent`` tool —
    older builds call it ``Task``) surfaces its target sub-agent + description, so the
    status line shows WHO Engram handed the work to — ``Agent→Explore: find all callers``
    — instead of a bare ``Agent``."""
    name = str(getattr(block, "name", "") or "tool")
    if name in ("Agent", "Task"):
        inp = getattr(block, "input", None) or {}
        sub = inp.get("subagent_type") or inp.get("subagentType") or "?"
        desc = (inp.get("description") or "").strip()
        return f"{name}→{sub}" + (f": {desc}" if desc else "")
    return name


def _ctx_bar(pct: float, width: int = 24) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    fill = int(round(width * pct / 100.0))
    return "█" * fill + "░" * (width - fill)


def render_context_md(u: dict) -> str:
    """Format the SDK's ``get_context_usage()`` payload (a ``ContextUsageResponse``)
    as compact markdown — the data Claude Code's own ``/context`` shows. Every field
    is read defensively (``.get``), so a partial/older payload still renders."""
    if not u:
        return "**Context** — no usage data available."
    total = int(u.get("totalTokens") or 0)
    raw = int(u.get("rawMaxTokens") or u.get("maxTokens") or 0)
    pct = u.get("percentage")
    if pct is None:
        pct = (100.0 * total / raw) if raw else 0.0
    model = u.get("model") or "?"
    if raw:
        head = (f"### Context — `{model}`\n"
                f"`{_ctx_bar(pct)}`  {total:,} / {raw:,} tokens · {pct:.0f}%")
    else:
        head = f"### Context — `{model}`\n{total:,} tokens"
    lines = [head, ""]
    # Drop the "Free space" pseudo-category — the header bar already shows used vs.
    # free, so listing the empty remainder (often ~98%) just buries the real consumers.
    cats = [c for c in (u.get("categories") or [])
            if c.get("tokens") and "free" not in str(c.get("name", "")).lower()]
    if cats:
        lines += ["| category | tokens | % |", "|---|--:|--:|"]
        for c in sorted(cats, key=lambda c: c.get("tokens", 0), reverse=True):
            t = int(c.get("tokens") or 0)
            lines.append(f"| {c.get('name', '?')} | {t:,} | "
                         + (f"{100.0 * t / raw:.0f}%" if raw else "") + " |")
        lines.append("")
    extra = []
    if u.get("memoryFiles"):
        extra.append(f"memory files: {len(u['memoryFiles'])}")
    if u.get("mcpTools"):
        extra.append(f"MCP tools: {len(u['mcpTools'])}")
    agents = [a for a in (u.get("agents") or []) if isinstance(a, dict)]
    if agents:
        names = [n for n in (str(a.get("name") or "").strip() for a in agents) if n]
        extra.append(f"sub-agents ({len(agents)}): {', '.join(names[:8])}"
                     if names else f"sub-agents: {len(agents)}")
    if extra:
        lines.append("  ·  ".join(extra))
    if u.get("isAutoCompactEnabled"):
        eff = int(u.get("maxTokens") or 0)
        note = "auto-compact on"
        if eff and raw and eff != raw:
            note += f" · usable window {eff:,} of {raw:,}"
        lines += ["", f"_{note}_"]
    return "\n".join(lines)


class ModelDriver:
    """The interface seam. A driver connects to a model backend and streams a
    turn's :class:`Event` objects while preserving conversation state across turns.

    Subclasses implement ``connect`` / ``disconnect`` / ``reset`` and an async
    generator ``query(text) -> Event``. A future ``LocalModelDriver`` (OpenAI-
    compatible, for the self-hosted endgame) implements this same surface, so no
    front-end changes when the backend swaps."""

    # Optional front-end seam for interactive tools (plan approval, option questions):
    # ``async on_interaction(request: dict) -> dict``. The front-end (the TUI) sets it;
    # left None the driver falls back to safe headless defaults (the perceiving loop,
    # the --simple REPL, tests). Model-agnostic: a LocalModelDriver would drive the same
    # handler from its own tool-call surface.
    on_interaction = None

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def reset(self) -> None: ...
    async def set_effort(self, level: str) -> None: ...
    async def set_model(self, name: str) -> None: ...
    async def set_permission_mode(self, mode: str) -> None: ...
    async def interrupt(self) -> None: ...

    async def get_context_usage(self) -> dict:
        """Context-window usage breakdown for the live session. Empty dict if the
        backend can't report it (a future ``LocalModelDriver`` may estimate its own)."""
        return {}

    async def run_subagent(self, name: str, task: str) -> AsyncIterator[Event]:
        """Run a named sub-agent as an isolated one-off, yielding its Events. Default:
        unsupported (yields nothing); ``AgentSDKDriver`` implements it."""
        return
        yield  # unreachable — marks this as an async generator
    # async def query(self, text: str) -> AsyncIterator[Event]:  (subclass provides)


class SessionStore:
    """Per-launch-directory persistence of the resumable session id, so a fresh
    ``engram`` process resumes the conversation from the same folder — keyed by cwd,
    the way Claude Code organizes sessions by project dir. One tiny file per cwd
    under :data:`SESSION_DIR`. Model-agnostic: it stores an opaque thread id, so a
    future ``LocalModelDriver`` reuses it unchanged. (Same idea as the Telegram
    bridge's ``_save_session`` / ``_load_session``, generalized to be per-cwd.)"""

    def __init__(self, root: Path = SESSION_DIR) -> None:
        self.root = root

    def _path_for(self, cwd: Path) -> Path:
        # Flatten the absolute path to a safe filename: '/home/user/project'
        # -> '-home-user-project', matching Claude Code's project-dir scheme.
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", str(Path(cwd).resolve()))
        return self.root / (safe or "root")

    def load(self, cwd: Path) -> Optional[str]:
        try:
            return self._path_for(cwd).read_text().strip() or None
        except OSError:
            return None

    def save(self, cwd: Path, sid: Optional[str]) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path_for(cwd)
            if sid:
                path.write_text(sid)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            pass  # persistence is best-effort; a fresh thread is an acceptable fallback


def _pid_is_live_engram(pid: int) -> bool:
    """True if ``pid`` is a running process that looks like an Engram launcher. Liveness via
    ``kill(pid, 0)``; on Linux we then confirm it's actually an engram via ``/proc/<pid>/cmdline``
    to shrug off pid-reuse (a crashed engram's pid recycled by an unrelated process). No procfs
    (non-Linux) falls back to trusting the liveness probe. Errs toward NOT-live, so a stale
    lock never wedges a folder shut."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False                      # no such process → stale lock
    except PermissionError:
        return True                       # alive, just not ours → still a live holder
    except OSError:
        return False
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True                       # no procfs (non-Linux) → trust kill(0)
    return b"engram" in raw.lower()       # our launcher's path/name contains "engram"


class LaunchLock:
    """A per-launch-directory advisory lock: one file per cwd holding the owner pid, so two
    ``engram`` processes don't drive the SAME resumable session from one folder — which would
    interleave both clients' turns into a single thread and corrupt it. A stale file (dead
    pid, or a pid that is no longer an engram) is reclaimed, so a crash — SIGHUP on a dropped
    SSH — never permanently locks a folder out. Advisory, not a hard mutex: any IO failure
    fails OPEN (the launch proceeds), because guarding the session is a nicety, never a
    reason to keep the user out of their own terminal."""

    def __init__(self, cwd: Path, root: Path = LOCK_DIR) -> None:
        self.cwd = Path(cwd).resolve()
        self.root = root
        safe = re.sub(r"[^A-Za-z0-9._-]", "-", str(self.cwd))
        self.path = root / (safe or "root")
        self._held = False

    def _live_owner(self) -> Optional[int]:
        """The pid of a LIVE engram already holding this folder, or None (free / stale)."""
        try:
            pid = int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None
        if pid <= 0 or not _pid_is_live_engram(pid):
            return None
        return pid

    def acquire(self) -> Optional[int]:
        """Take the lock. Returns None on success (folder was free, or its lock was stale
        and reclaimed); returns the pid of the live holder if another engram already owns
        this folder — the caller should refuse to start."""
        owner = self._live_owner()
        if owner is not None and owner != os.getpid():
            return owner
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(os.getpid()))
            self._held = True
        except OSError:
            pass                          # fail open — never block a launch on lock IO
        return None

    def release(self) -> None:
        """Drop the lock, but only if we still own the file (don't clobber a process that
        already reclaimed it as stale)."""
        if not self._held:
            return
        try:
            if self.path.read_text().strip() == str(os.getpid()):
                self.path.unlink(missing_ok=True)
        except OSError:
            pass
        self._held = False


# Sentinel so callers can pass ``store=None`` to disable persistence (e.g. tests),
# while the default constructs a real per-cwd store.
_DEFAULT_STORE = object()


class AgentSDKDriver(ModelDriver):
    """Engram on the Claude Agent SDK + the Pro/Max subscription (no API key). Holds
    ONE persistent ``ClaudeSDKClient`` so the conversation has memory across turns;
    tracks the session id for resume, persists it per-cwd via a :class:`SessionStore`
    so a *fresh process* resumes the last conversation in that folder, with a
    one-shot fresh fallback if a resume goes stale."""

    def __init__(self, *, cwd: Path = ENGRAM_CWD, model: str = ENGRAM_MODEL,
                 effort: str = ENGRAM_EFFORT, persona: str = PERSONA,
                 cli_path: str = CLAUDE_BIN,
                 permission_mode: str = REGULAR_MODE,
                 setting_sources: Optional[list[str]] = None,
                 agents: Optional[dict] = None,
                 store=_DEFAULT_STORE) -> None:
        self.cwd = cwd
        self.model = model
        self.effort = effort
        self.persona = persona
        # Plan ↔ regular (shift+tab). Stored on the driver so it survives the
        # set_model / set_effort reconnects and reset(); _options() reads it on connect.
        self.permission_mode = permission_mode
        self.cli_path = cli_path
        # Sub-agent roster for Task-tool delegation; default = Engram's built-in trio,
        # pass {} to disable. (See SUBAGENT_DEFS for why we define them explicitly.)
        self.agents = SUBAGENT_DEFS if agents is None else agents
        # ["user", "project", "local"] = Claude Code's own default: load the global
        # ~/.claude (settings + the recall memory hook + global skills) AND the
        # launched folder's project/local config. So Engram has that folder's brain
        # AND memory everywhere — exactly like opening Claude Code there.
        self.setting_sources = setting_sources or ["user", "project", "local"]
        self._store: Optional[SessionStore] = (
            SessionStore() if store is _DEFAULT_STORE else store)
        self._client: Optional[ClaudeSDKClient] = None
        # Resume the last conversation from THIS folder, if one was saved.
        self.session_id: Optional[str] = (
            self._store.load(self.cwd) if self._store else None)
        self.resumed: bool = self.session_id is not None  # for the front-end to announce
        self.actual_model: Optional[str] = None   # what the SDK reports it's REALLY using
        self._stderr: list[str] = []
        # Set by the front-end (app.py) to render plan-approval / option-question UI; see
        # ModelDriver.on_interaction. None → headless defaults in _can_use_tool.
        self.on_interaction = None

    def _options(self) -> ClaudeAgentOptions:
        opts = dict(
            system_prompt={"type": "preset", "preset": "claude_code",
                           "append": self.persona},
            setting_sources=self.setting_sources,
            permission_mode=self.permission_mode,
            cwd=str(self.cwd),
            cli_path=self.cli_path,
            resume=self.session_id,
            agents=self.agents,            # sub-agent roster → Task-tool delegation
            can_use_tool=self._can_use_tool,   # interactive-tool interception (plan / questions)
            stderr=self._stderr.append,
        )
        if self.effort:
            opts["effort"] = self.effort
        if self.model:
            opts["model"] = self.model
        return ClaudeAgentOptions(**opts)

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = ClaudeSDKClient(options=self._options())
        await self._client.connect()

    async def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.disconnect()
        finally:
            self._client = None

    def reset(self) -> None:
        """Drop the session id (and its saved file) so the next turn — and the next
        launch in this folder — starts a fresh thread."""
        self.session_id = None
        self.resumed = False
        if self._store:
            self._store.save(self.cwd, None)

    async def set_effort(self, level: str) -> None:
        """Change reasoning effort (low→max). Drops the warm client so the next
        turn reconnects with the new effort; the session id is kept, so the
        conversation continues — just at a different effort."""
        if level not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}")
        self.effort = level
        await self.disconnect()

    async def set_model(self, name: str) -> None:
        """Change the model; reconnects on the next turn (session kept)."""
        self.model = name
        await self.disconnect()

    async def set_permission_mode(self, mode: str) -> None:
        """Switch permission mode (regular ↔ plan). Recycles the warm client — drops it
        so the NEXT turn reconnects in the new mode, with the session id kept so the
        conversation continues (exactly like set_model / set_effort). We reconnect rather
        than fire a live control request because the live switch does NOT release *plan*
        mode mid-session: it leaves you wedged in plan, unable to act, with no way out
        from the UI. The mode only governs the next turn anyway, so a clean reconnect
        costs nothing but a fresh subprocess and is the only reliable way out of plan."""
        if mode not in PERMISSION_MODES:
            raise ValueError(f"permission_mode must be one of {PERMISSION_MODES}")
        self.permission_mode = mode
        await self.disconnect()

    async def _can_use_tool(self, name: str, tool_input: dict, ctx) -> object:
        """Permission callback — the only channel the CLI-native interactive tools reach.
        Ordinary tools never arrive here under bypassPermissions (auto-allowed), so this
        fires only for ExitPlanMode / AskUserQuestion (verified empirically). We hand each
        to the front-end ``on_interaction`` handler and translate its verdict into the wire
        result: PermissionResultAllow = run the tool (plan approved → the turn proceeds to
        implement, and the CLI persistently leaves plan mode); PermissionResultDeny(message)
        feeds ``message`` back to the model as the tool result — which is how BOTH a rejected
        plan (revise, keep planning) and an answered question (the chosen option / typed
        answer) are delivered. With no handler wired we fall back to safe headless defaults
        so the perceiving loop / --simple REPL / tests never wedge."""
        tool_input = tool_input or {}
        handler = self.on_interaction
        if name == "ExitPlanMode":
            if handler is None:
                return PermissionResultAllow()          # headless: accept the plan, proceed
            decision = await handler({"kind": "plan", "plan": tool_input.get("plan") or ""})
            if decision.get("approved"):
                # Approving persistently exits plan mode at the CLI level (verified), so just
                # sync our stored field — the indicator follows and no reconnect is needed.
                self.permission_mode = REGULAR_MODE
                return PermissionResultAllow()
            return PermissionResultDeny(
                message=decision.get("message") or "Keep planning — don't implement yet.")
        if name == "AskUserQuestion":
            if handler is None:
                return PermissionResultDeny(message=(
                    "No interactive UI is available here; proceed with your best judgment "
                    "and state the assumption you made."))
            decision = await handler({"kind": "question",
                                      "questions": tool_input.get("questions") or []})
            return PermissionResultDeny(
                message=decision.get("message")
                or "The user didn't choose; proceed with your best judgment.")
        return PermissionResultAllow()   # any other tool that reaches here → allow

    async def interrupt(self) -> None:
        """Stop the in-flight turn now: ask the live CLI to stop generating (the SDK
        interrupt control request). The streaming ``_stream`` then sees the turn end and
        the front-end's worker cleans up normally; the warm client stays connected, so
        the next turn continues this same conversation. No-op if nothing is connected —
        the Telegram bridge's /cancel uses this same call."""
        if self._client is not None:
            await self._client.interrupt()

    async def get_context_usage(self) -> dict:
        """Live context-window breakdown for this session — the same data Claude
        Code's own ``/context`` shows (a ``ContextUsageResponse``). ``connect()`` is
        idempotent, so this also works before the first turn (baseline context).
        Raises if the connected CLI doesn't support the control request — callers
        surface that as 'context unavailable' rather than crashing."""
        await self.connect()
        assert self._client is not None
        return dict(await self._client.get_context_usage())

    async def run_subagent(self, name: str, task: str) -> AsyncIterator[Event]:
        """Run a named sub-agent as a one-off, ISOLATED, synchronous sub-query — its
        own fresh CLI subprocess + context (no shared session), streaming its events
        live. We do this rather than the CLI's built-in Agent tool because that tool
        runs sub-agents asynchronously (fire-and-forget), so their result would never
        return within Engram's turn. The agent's roster entry supplies its system prompt
        + tools; an unknown name falls back to general-purpose."""
        spec = self.agents.get(name) or self.agents.get("general-purpose")
        append = spec.prompt if spec else f"You are the {name} sub-agent."
        opts = dict(
            system_prompt={"type": "preset", "preset": "claude_code", "append": append},
            setting_sources=self.setting_sources,
            permission_mode="bypassPermissions",
            cwd=str(self.cwd),
            cli_path=self.cli_path,
            stderr=self._stderr.append,
        )
        if spec and spec.tools:
            opts["allowed_tools"] = list(spec.tools)
        if self.effort:
            opts["effort"] = self.effort
        if self.model:
            opts["model"] = self.model
        async for msg in _sdk_query(prompt=task, options=ClaudeAgentOptions(**opts)):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield Event("text", block.text)
                    elif getattr(block, "name", None):
                        yield Event("tool", _tool_label(block))

    async def query(self, text: str) -> AsyncIterator[Event]:
        """Run one turn, streaming text + tool-use events. Reconnects on a stale
        resume once before giving up."""
        try:
            await self.connect()
            async for ev in self._stream(text):
                yield ev
        except Exception:
            # A stale resume is the common failure: drop the dead pointer, reconnect
            # fresh, and TELL the front-end — silently swapping in a fresh thread while
            # the UI still says "resumed your last conversation" was the gap (the Telegram
            # bridge surfaces this; the TUI didn't).
            if self.session_id is not None:
                self.session_id = None
                self.resumed = False
                if self._store:
                    self._store.save(self.cwd, None)   # clear the pointer to a dead session
                await self.disconnect()
                await self.connect()
                yield Event("text", "\n\n> ⚠ *couldn't resume the previous thread "
                                    "— started a fresh one*\n\n")
                async for ev in self._stream(text):
                    yield ev
            else:
                raise

    async def _stream(self, text: str) -> AsyncIterator[Event]:
        assert self._client is not None
        await self._client.query(text)
        # We read the raw message stream (not receive_response) so we can keep going PAST
        # the parent turn's ResultMessage while a background sub-agent is still running —
        # the CLI's Agent tool is async (fire-and-forget) and its progress + completion
        # arrive as Task* messages after that result (see SUBAGENT_IDLE_TIMEOUT). We stop
        # on a ResultMessage that lands with NO sub-agent pending (a normal turn stops at
        # the very first one — unchanged). `parent_tool_use_id` tells a sub-agent's OWN
        # messages (skipped — surfaced as the Task* markers below) from top-level ones, so
        # the reply shows the main agent's narration + its final synthesis, not the
        # sub-agent's raw monologue.
        pending: set[str] = set()
        names: dict[str, str] = {}                       # task_id -> subagent_type
        it = self._client.receive_messages().__aiter__()
        while True:
            try:
                msg = await (asyncio.wait_for(it.__anext__(), SUBAGENT_IDLE_TIMEOUT)
                             if pending else it.__anext__())
            except asyncio.TimeoutError:                 # a pending sub-agent went silent
                who = ", ".join(names.get(t, "sub-agent") for t in pending)
                yield Event("text", f"\n\n> ⏳ *{who} still running — detached; "
                                    f"its result wasn't captured this turn*\n\n")
                return
            except StopAsyncIteration:
                return

            if isinstance(msg, AssistantMessage):
                if getattr(msg, "parent_tool_use_id", None):
                    continue                             # sub-agent-internal → Task* markers
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield Event("text", block.text)
                    elif getattr(block, "name", None):   # ToolUseBlock-ish
                        if block.name in _INTERACTIVE_TOOLS:
                            continue                     # rendered as a card via _can_use_tool
                        yield Event("tool", _tool_label(block))
            elif isinstance(msg, TaskStartedMessage):
                name = (getattr(msg, "data", {}) or {}).get("subagent_type") or "sub-agent"
                names[msg.task_id] = name
                pending.add(msg.task_id)
                yield Event("text", f"\n\n> 🛰 *delegated to **{name}** — {msg.description}*\n\n")
            elif isinstance(msg, TaskProgressMessage):
                bits = f"🛰 {names.get(msg.task_id, 'sub-agent')}"
                if getattr(msg, "last_tool_name", None):
                    bits += f" · {msg.last_tool_name}"
                tot = (getattr(msg, "usage", None) or {}).get("total_tokens")
                if tot:
                    bits += f" · {tot:,} tok"
                yield Event("status", bits)              # ephemeral — status line only
            elif isinstance(msg, (TaskNotificationMessage, TaskUpdatedMessage)):
                status = getattr(msg, "status", None)
                if status in TERMINAL_TASK_STATUSES and msg.task_id in pending:
                    pending.discard(msg.task_id)
                    name = names.get(msg.task_id, "sub-agent")
                    summ = getattr(msg, "summary", None) or f"{name} {status}"
                    mark = "✓" if status == "completed" else "✗"
                    yield Event("text", f"\n\n> {mark} *{summ}*\n\n")
            elif isinstance(msg, SystemMessage):
                data = getattr(msg, "data", {}) or {}
                self.session_id = self.session_id or data.get("session_id")
                self.actual_model = data.get("model") or self.actual_model
                # Flush the id the INSTANT the session exists — not only at the closing
                # ResultMessage below. A turn cut off before its Result (VPN drop / SIGHUP)
                # would otherwise orphan a live server-side session; this matters most for
                # the FIRST turn in a fresh folder, when nothing is on disk yet to resume.
                if self._store and self.session_id:
                    self._store.save(self.cwd, self.session_id)
            elif isinstance(msg, ResultMessage):
                self.session_id = getattr(msg, "session_id", None) or self.session_id
                self.actual_model = getattr(msg, "model", None) or self.actual_model
                # Persist the (authoritative) session id so the next `engram` in this folder
                # resumes here. Stop only when no sub-agent is still running.
                if self._store and self.session_id:
                    self._store.save(self.cwd, self.session_id)
                if not pending:
                    return

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr[-6:])
