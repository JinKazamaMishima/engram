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
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from buffer import LiveBuffer  # noqa: E402 — sibling module (infra/engram)

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
    HookEventMessage,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    UserMessage,
    query as _sdk_query,
)

REPO = Path(os.environ.get("RECALL_REPO") or Path(__file__).resolve().parents[2])
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
ENGRAM_MODEL = os.environ.get("ENGRAM_MODEL", "opus[1m]")
ENGRAM_EFFORT = os.environ.get("ENGRAM_EFFORT", "max")   # max always; downgrade via /effort
# The SDK's stdio transport caps ONE stream-json line at 1 MiB by default
# (claude_agent_sdk .../transport/subprocess_cli.py _DEFAULT_MAX_BUFFER_SIZE). A single
# tool result carrying a base64 image (e.g. Read on a screenshot) can exceed that in one
# message and crash the whole transport — "JSON message exceeded maximum buffer size of
# 1048576 bytes" — killing the session (seen on a 765 KB full-page PNG → ~1.02 MB base64).
# Raise the ceiling generously so realistic image/tool payloads pass; env-tunable
# (ENGRAM_CLI_MAX_BUFFER_MB) for the rare giant read.
CLI_MAX_BUFFER_SIZE = int(float(os.environ.get("ENGRAM_CLI_MAX_BUFFER_MB", "64")) * 1024 * 1024)
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
# LiveBuffer (Brick 3 tier 1): append-only raw-conversation JSONLs, one per convo id,
# the immutable source the working set re-derives from and eviction curates from.
# ENGRAM_BUFFER=0 is the master cutout — no rows written, working set + eviction go
# quiet with it (nothing to derive from), and the nightly transcript sweep still
# covers curation exactly as before Brick 3.
BUFFER_ON = os.environ.get("ENGRAM_BUFFER", "1") != "0"
BUFFER_DIR = Path(os.environ.get("ENGRAM_BUFFER_DIR", str(DATA_ROOT / "engram" / "buffer")))
# Eviction-is-curation (Brick 3 A6): once enough of the buffer has COOLED out of
# the working-set window, a detached incremental `curate --buffer` folds the
# cooled tail into the LTM corpus and advances the per-convo watermark — off the
# hot path, never blocking a turn. ENGRAM_EVICT=0 disables (buffer + working set
# stay; only mid-session curation stops — the nightly sweep still covers it).
EVICT_ON = os.environ.get("ENGRAM_EVICT", "1") != "0"
EVICT_CHARS = int(os.environ.get("ENGRAM_EVICT_CHARS", "40000"))
# The hot window kept OUT of eviction — the last N turns still live in the
# working set, so curating them mid-flight would freeze a still-unfolding thread.
# Mirrors working_set.WM_TURNS (same env), so the two edges stay coupled.
EVICT_HOT_TURNS = int(os.environ.get("ENGRAM_WM_TURNS", "12"))

# Reasoning effort levels, low→high, matching Claude Code's /effort.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Known model families, so the UI can tell a fallback apart from the primary even
# though one side is an alias ("fable", "opus[1m]") and the other the resolved id
# the SDK reports back ("claude-fable-5", "claude-opus-4-8"). Order matters only if
# an id ever contained two — it won't.
_MODEL_FAMILIES = ("opus", "sonnet", "haiku", "fable")


def _model_family(name: Optional[str]) -> Optional[str]:
    """The family token in a model alias OR a resolved id ('opus'/'sonnet'/
    'haiku'/'fable'), else None — lets configured-vs-actual be compared across the
    alias/id boundary without a lookup table."""
    if not name:
        return None
    low = str(name).lower()
    return next((f for f in _MODEL_FAMILIES if f in low), None)

# Permission modes Engram cycles between with shift+tab, like Claude Code. "Regular" =
# bypassPermissions (Engram acts freely; the persona is the only guardrail). "Plan" =
# the SDK's read-only plan mode (investigate + propose, make no changes). The SDK also
# accepts default/acceptEdits/dontAsk/auto, but those need an interactive permission UI
# this harness doesn't have — so the front-end toggle stays between these two.
REGULAR_MODE = "bypassPermissions"
PLAN_MODE = "plan"
PERMISSION_MODES = (
    "default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto")
# The operator-facing cycle (shift+tab): act freely → plan only. There is deliberately
# NO "ask" (permission-prompt) mode — Engram acts when it judges an action right, and
# the persona is the only guardrail.
MODE_CYCLE = (REGULAR_MODE, PLAN_MODE)
_MODE_ALIASES = {"regular": REGULAR_MODE, "plan": PLAN_MODE}
# The daily default (bypass — the persona is the guardrail);
# ENGRAM_DEFAULT_MODE=plan|regular overrides per launch.
DEFAULT_MODE = _MODE_ALIASES.get(
    os.environ.get("ENGRAM_DEFAULT_MODE", "regular"), REGULAR_MODE)

# The two CLI-native tools that need the operator IN the loop: ExitPlanMode (present a
# plan, wait for approval) and AskUserQuestion (offer options, wait for a pick). They are
# invisible to the SDK except through the `can_use_tool` permission channel — the CLI
# routes BOTH through it even under bypassPermissions (verified), while ordinary tools
# auto-allow without a round-trip. So the driver intercepts exactly these, hands them to a
# front-end `on_interaction` handler, and renders them richly instead of as a bare status
# blip (we also suppress their tool-use markers in the stream, since the card IS the render).
_INTERACTIVE_TOOLS = {"ExitPlanMode", "AskUserQuestion"}

# Harness-injected spans inside a user prompt (memory reminders, task pings) —
# stripped when building a checkpoint preview so it shows what was TYPED. The
# `|$` arm tolerates an unclosed tag (a truncated echo must not eat the prompt).
_INJECTED_RE = re.compile(r"<(system-reminder|task-notification)>.*?(</\1>|$)", re.S)

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
    """One streamed unit of a turn. ``kind``: 'text' | 'tool' | 'status' | 'recall'.
    ``data`` is an optional structured payload for richer kinds a front-end may
    render beyond the text (e.g. 'todos'/'task' panels) — None for the plain
    kinds, so every existing constructor and consumer keeps working. Model-
    agnostic: a future LocalModelDriver emits the same shapes. 'recall' carries the
    per-turn memory provenance (which notes the inject hook surfaced; '' = it ran
    and surfaced none) so the front-end can make memory visible turn by turn."""
    kind: str
    text: str
    data: Optional[dict] = None


# --- presentation helpers (model-agnostic; shared by every front-end) ---------

def _recall_line(msg) -> Optional[str]:
    """The operator-visible recall provenance from a UserPromptSubmit hook_response
    (``include_hook_events`` streams them): the ``systemMessage`` the inject hook
    printed, minus its ``🧠 recalled:`` prefix — i.e. the ``corpus:slug`` list that
    fed this turn. Returns '' when the hook ran but surfaced nothing (a real
    zero-hit — the miss-detector cares about the difference), and None when the
    event isn't a UserPromptSubmit hook_response at all. Fail-soft: a hook whose
    output isn't the expected JSON reads as '' (ran, nothing to show)."""
    if (getattr(msg, "subtype", "") != "hook_response"
            or getattr(msg, "hook_event_name", "") != "UserPromptSubmit"):
        return None
    out = ((getattr(msg, "data", {}) or {}).get("output") or "").strip()
    if not out:
        return ""
    try:
        sysmsg = str(json.loads(out).get("systemMessage") or "")
    except Exception:  # noqa: BLE001 — someone else's hook / non-JSON output
        return ""
    return sysmsg.removeprefix("🧠 recalled:").strip()


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
    if name == "Workflow":
        script = str((getattr(block, "input", None) or {}).get("script") or "")
        m = re.search(r"name:\s*['\"]([A-Za-z0-9_-]+)", script)
        return f"Workflow→{m.group(1)}" if m else "Workflow"
    return name


def workflow_snapshot(wp: list) -> dict:
    """Collapse one ``task_progress`` message's ``workflow_progress`` list into the
    panel/detail snapshot: ordered phases, each with its agents' label/state/model.
    The runtime re-sends the FULL tree on every heartbeat, so each snapshot
    REPLACES the previous one (point-in-time, like the task-registry copies).
    Shape (all keys always present): {phases: [{title, agents: [{label, state,
    model}]}], done, total, phase} — ``phase`` is where work currently is (the
    last streamed agent's phase)."""
    phases: list[dict] = []
    by_title: dict[str, dict] = {}
    cur_phase = ""
    for e in wp or []:
        if e.get("type") == "workflow_phase":
            title = e.get("title") or f"phase {e.get('index', '?')}"
            if title not in by_title:
                p = {"title": title, "agents": []}
                phases.append(p)
                by_title[title] = p
        elif e.get("type") == "workflow_agent":
            title = e.get("phaseTitle") or "…"
            p = by_title.get(title)
            if p is None:
                p = {"title": title, "agents": []}
                phases.append(p)
                by_title[title] = p
            p["agents"].append({
                "label": e.get("label") or f"agent {e.get('index', '?')}",
                "state": e.get("state") or "…",
                "model": _model_family(e.get("model")) or ""})
            cur_phase = title
    agents = [a for p in phases for a in p["agents"]]
    return {"phases": phases,
            "done": sum(1 for a in agents if a["state"] == "done"),
            "total": len(agents),
            "phase": cur_phase or (phases[-1]["title"] if phases else "")}


def session_curation_cmd(sid: str, cwd: Path, provisional: bool = False) -> list[str]:
    """Argv that curates ONE session into its project corpus + the soul — the same
    command the Telegram bridge fires on /new · /end (brick 2), resolved from this
    venv (RECALL_BIN overrides). ``provisional`` marks a pass over a session that
    is NOT over (PreCompact mid-session, or a terminal close that will RESUME
    later): the recall CLI grows --provisional + reconcile with Brick 3 — until
    then that spawn exits non-zero and is swallowed, deliberately, so a live
    session is never marked fully-curated in the idempotency bucket while it is
    still growing. The nightly sweep remains the real writer meanwhile."""
    recall_bin = os.environ.get("RECALL_BIN") or str(
        Path(sys.executable).with_name("recall"))
    cmd = [recall_bin, "curate", "--session", sid, "--project-dir", str(cwd)]
    if provisional:
        cmd.append("--provisional")
    cmd.append("--commit")
    return cmd


def spawn_session_curate(sid: Optional[str], cwd: Path,
                         provisional: bool = False) -> None:
    """Detached fire-and-forget curation for shutdown paths — the SDK has NO
    SessionEnd hook (verified: 10 events only), so session-end lives in the
    front-ends' teardown. Plain Popen, not asyncio: the loop may be tearing down.
    Best-effort by design; a child killed with the terminal is harmless
    (idempotent sessions bucket + the nightly safety-net sweep)."""
    if not sid:
        return
    try:
        subprocess.Popen(session_curation_cmd(sid, cwd, provisional),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception:  # noqa: BLE001 — curation must never break teardown
        pass


def buffer_curation_cmd(buffer_path: Path, cwd: Path, *,
                        until: Optional[str] = None,
                        provisional: bool = True) -> list[str]:
    """Argv for an INCREMENTAL curate over an Engram LiveBuffer (Brick 3): folds the
    tail after this convo's watermark into the corpus and advances it. ``until``
    caps the slice at the cooled edge so still-hot turns stay uncurated; omit it
    (shutdown) to flush the whole tail. Always ``--provisional`` — a live buffer
    is a possibly-still-open conversation."""
    recall_bin = os.environ.get("RECALL_BIN") or str(
        Path(sys.executable).with_name("recall"))
    cmd = [recall_bin, "curate", "--buffer", str(buffer_path), "--incremental",
           "--project-dir", str(cwd)]
    if until:
        cmd += ["--until", until]
    if provisional:
        cmd.append("--provisional")
    cmd.append("--commit")
    return cmd


def spawn_buffer_curate(buffer_path: Path, cwd: Path, *,
                        until: Optional[str] = None,
                        provisional: bool = True):
    """Detached `curate --buffer` (Brick 3 eviction). Returns the Popen so a
    caller can wait on the PID directly (never trust a completion ping —
    dont-trust-detached-background-completion-notifications), or None on failure.
    start_new_session so it outlives Engram quitting mid-eviction."""
    try:
        return subprocess.Popen(
            buffer_curation_cmd(buffer_path, cwd, until=until,
                                provisional=provisional),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
    except Exception:  # noqa: BLE001 — eviction must never break a turn/teardown
        return None


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

    # True while background sub-agents (or their unread follow-up turns) are still
    # out — the front-end keeps an idle drain running so late results PAINT when
    # they land, and gates client-recycling commands that would kill them.
    has_background_tasks: bool = False

    async def drain_background(self) -> AsyncIterator[Event]:
        """Yield Events that arrive AFTER a turn has ended (background sub-agent
        completions + the model's follow-up turns). Default: nothing to drain."""
        return
        yield  # unreachable — marks this as an async generator

    def list_checkpoints(self) -> list[dict]:
        """File-rewind anchors — ``{"uuid", "preview", "ts"}`` per user prompt — if
        the backend supports file checkpointing. Default: none."""
        return []

    async def rewind_to(self, uuid: str) -> None:
        """Restore files to their state just before the given prompt."""
        raise NotImplementedError("this driver has no file checkpointing")

    def list_sessions(self, limit: int = 9) -> list[dict]:
        """This folder's resumable sessions (newest first). Default: none."""
        return []

    async def resume_session(self, sid: str) -> None:
        raise NotImplementedError("this driver has no session switching")

    async def fork(self) -> None:
        raise NotImplementedError("this driver has no session forking")

    async def run_subagent(self, name: str, task: str) -> AsyncIterator[Event]:
        """Run a named sub-agent as an isolated one-off, yielding its Events. Default:
        unsupported (yields nothing); ``AgentSDKDriver`` implements it."""
        return
        yield  # unreachable — marks this as an async generator
    # async def query(self, text: str, *, prepend: str = "") -> AsyncIterator[Event]:
    #   (subclass provides; `prepend` is model-only context — never logged as the
    #    operator's raw text)


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
                 permission_mode: str = DEFAULT_MODE,
                 setting_sources: Optional[list[str]] = None,
                 agents: Optional[dict] = None,
                 store=_DEFAULT_STORE,
                 buffer_dir=None) -> None:
        self.cwd = cwd
        self.model = model
        self.effort = effort
        self.persona = persona
        # Plan ↔ regular (shift+tab). Stored on the driver so it survives the
        # set_model / set_effort reconnects and reset(); _options() reads it on connect.
        self.permission_mode = permission_mode
        # The mode the operator was in when they ENTERED plan mode — what approving a
        # plan must land back in. None until plan mode is entered this process.
        self._pre_plan_mode: Optional[str] = None
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
        # LiveBuffer (Brick 3 tier 1). buffer_dir: None = auto (on ONLY for the
        # production case — the real default SessionStore, an actual thread-of-
        # record convo), False = forced off, Path = forced on there. store=None
        # (perceiving mind, transient) and custom stores (tests with tmp dirs)
        # never buffer implicitly — tests opt in with buffer_dir=Path(tmp), so
        # no test can pollute the real buffer dir. (Not an identity-sentinel
        # default on purpose: test_permissions importlib.reload()s this module,
        # and a def-time sentinel stops `is`-matching its reloaded self.)
        if buffer_dir is None:
            buffer_dir = (BUFFER_DIR if (BUFFER_ON and store is _DEFAULT_STORE)
                          else None)
        elif buffer_dir is False:
            buffer_dir = None
        # Until the SDK mints a session id mid-turn-1, rows key on a provisional
        # launch id; _sync_buf_convo renames the file the moment the id exists.
        self._launch_id = "launch-" + uuid.uuid4().hex[:12]
        self._buf_convo_id: str = self.session_id or self._launch_id
        self._buffer = LiveBuffer(buffer_dir, lambda: self._buf_convo_id)
        self._buffer.reseed()          # resumed convo: continue seq, never restart
        self._fork_buf_copy = False    # armed by fork(); consumed at the new sid
        self._evicting = False         # one detached curate at a time (A6)
        self.actual_model: Optional[str] = None   # what the SDK reports it's REALLY using
        self.fallback_model: Optional[str] = None  # configured fallback (set in _options)
        self._stderr: list[str] = []
        # Set by the front-end (app.py) to render plan-approval / option-question UI; see
        # ModelDriver.on_interaction. None → headless defaults in _can_use_tool.
        self.on_interaction = None
        # Task bookkeeping that OUTLIVES a single turn. A sub-agent launched with
        # run_in_background (the Agent tool's own flag) finishes AFTER its turn by
        # design: its terminal Task message and the model's notification-driven
        # follow-up turn arrive later — read by the idle drain (drain_background)
        # or by whichever stream is open when they land.
        self._task_names: dict[str, str] = {}   # task_id -> subagent_type label
        self._bg_tasks: set[str] = set()         # background task_ids still running
        self._bg_owed: int = 0                    # notified completions whose follow-up turn is unread
        # Session task registry for the front-end panel: task_id -> {name, desc,
        # status, background, tokens, last_tool}. Survives across turns (finished
        # sub-agents stay listed); cleared with the thread.
        self.tasks: dict[str, dict] = {}
        # File-rewind anchors, one per user prompt: {"uuid", "preview", "ts"}.
        # Populated from the replayed UserMessage echoes (see _note_checkpoint);
        # /rewind lists them and rewind_to() restores the files.
        self.checkpoints: list[dict] = []
        self._fork_next = False       # next connect resumes WITH fork_session=True
        # In-process MCP: recall memory as first-class tools the model can PULL
        # with (the ambient hook only pushes titles). Built once and reused across
        # our disconnect→reconnect recycles; strictly optional (fail-open).
        self.mcp_servers: dict = {}
        if os.environ.get("ENGRAM_RECALL_TOOLS", "1") != "0":
            try:
                from memory_tools import build_recall_server
                srv = build_recall_server(self.cwd)
                if srv is not None:
                    self.mcp_servers["recall"] = srv
            except Exception:  # noqa: BLE001 — memory tools must never block launch
                pass

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
            hooks=self._hooks(),           # lifecycle hooks (PreCompact → provisional curation)
            stderr=self._stderr.append,
            # Raise the stdio line ceiling so one base64 image in a tool result can't
            # overflow the transport and crash the session (see CLI_MAX_BUFFER_SIZE).
            max_buffer_size=CLI_MAX_BUFFER_SIZE,
            # Stream hook lifecycle events: the recall inject hook's response carries
            # WHICH notes it surfaced, and _stream turns that into the per-turn
            # provenance line (Event 'recall'). Absence of the event is itself signal
            # — the front-end renders "hook silent", the injection-outage tell.
            include_hook_events=True,
        )
        if self.effort:
            opts["effort"] = self.effort
        if self.model:
            opts["model"] = self.model
        # Model rotation on overload: when the primary hits a rate-limit window or an
        # overloaded/unavailable error, the CLI retries the SAME turn on the fallback
        # instead of dying — which also kept the stale-resume fallback in query() from
        # ever misreading an overload as a dead session and resetting the thread.
        # Default Opus 4.8 — a real capability floor, not Sonnet.
        # ENGRAM_FALLBACK_MODEL="" disables; a SAME-FAMILY fallback is skipped (pointless,
        # and the CLI rejects a fallback equal to the primary) so opus-primary sessions
        # don't "fall back" to Opus. self.fallback_model names it for the UI.
        fallback = os.environ.get("ENGRAM_FALLBACK_MODEL", "claude-opus-4-8")
        if (fallback and fallback != self.model
                and _model_family(fallback) != _model_family(self.model)):
            opts["fallback_model"] = fallback
            self.fallback_model = fallback
        else:
            self.fallback_model = None
        if os.environ.get("ENGRAM_CHECKPOINTS", "1") != "0":
            # File checkpoints (the CLI shadow-copies before each edit, one anchor per
            # user prompt) + replay-user-messages so the stream echoes UserMessages
            # WITH the `uuid` rewind_files() targets. ENGRAM_CHECKPOINTS=0 is the kill
            # switch if the replay flag ever destabilizes the stream.
            opts["enable_file_checkpointing"] = True
            opts["extra_args"] = {"replay-user-messages": None}
        if self.mcp_servers:
            opts["mcp_servers"] = self.mcp_servers
            # Pre-allow the memory tools (allowed_tools ADDS allow rules; it does not
            # restrict other tools).
            opts["allowed_tools"] = ["mcp__recall__recall_search",
                                     "mcp__recall__recall_read_note"]
        if self._fork_next:
            # Resume + fork_session: the CLI mints a NEW session id branched off
            # the resumed thread; the original stays untouched (see fork()).
            opts["fork_session"] = True
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
            # The CLI subprocess is gone — background sub-agents died with it and
            # no follow-up turn can arrive on this client. (The front-end gates the
            # disconnecting commands while has_background_tasks, so reaching here
            # with live ones is deliberate, e.g. quitting the app.)
            self._task_names.clear()
            self._bg_tasks.clear()
            self._bg_owed = 0

    def reset(self) -> None:
        """Drop the session id (and its saved file) so the next turn — and the next
        launch in this folder — starts a fresh thread."""
        self.session_id = None
        self.resumed = False
        self.checkpoints.clear()      # rewind anchors belong to the old thread
        self.tasks.clear()            # so does the task panel's registry
        if self._store:
            self._store.save(self.cwd, None)
        # A fresh thread is a fresh conversation: new provisional buffer id (the
        # old file stays where it is — its thread is still reachable via /sessions).
        self._launch_id = "launch-" + uuid.uuid4().hex[:12]
        self._buf_convo_id = self._launch_id
        self._buffer.reseed()

    # ---- LiveBuffer identity (Brick 3) -------------------------------------

    def _sync_buf_convo(self) -> None:
        """Make the buffer file follow the SDK session identity, the moment the
        id is learned (same instant the SessionStore flushes). turn-1: rename
        launch→sid (merge if a resumed file already exists). fork: COPY the
        parent buffer to the new sid — the branched context genuinely contains
        those turns — leaving the parent's file intact. Fail-open throughout."""
        sid = self.session_id
        if not sid or sid == self._buf_convo_id:
            return
        self._buffer.migrate(self._buf_convo_id, sid, copy=self._fork_buf_copy)
        self._fork_buf_copy = False
        self._buf_convo_id = sid
        self._buffer.reseed()

    def _revert_buf_convo(self) -> None:
        """Stale resume: the server-side session is dead but the human
        conversation continues — park the buffer under the launch id so the
        pre-retry rows and the retry share one file; the fresh sid reclaims it
        via _sync_buf_convo."""
        if self._buf_convo_id != self._launch_id:
            self._buffer.migrate(self._buf_convo_id, self._launch_id)
            self._buf_convo_id = self._launch_id
            self._buffer.reseed()

    # ---- eviction-is-curation (Brick 3 A6) ---------------------------------

    def _evict_watermark(self) -> str:
        """This convo's curation watermark, read fresh each time. '' when
        never curated. Fail-open '' (see buffer.read_buffer_watermark)."""
        from buffer import read_buffer_watermark
        return read_buffer_watermark(self.cwd, self._buf_convo_id)

    def _cooled_edge(self) -> Optional[tuple[str, int]]:
        """(until_ts, cooled_char_count) for the buffer range that has cooled OUT
        of the working-set window since the watermark — the range eviction may
        curate. None if nothing has cooled past the hot window yet. The hot
        window (last EVICT_HOT_TURNS rows) is always excluded so a still-live
        thread is never frozen mid-flight."""
        after = self._buffer.tail_after(self._evict_watermark())
        if len(after) <= EVICT_HOT_TURNS:
            return None
        cooled = after[:-EVICT_HOT_TURNS] if EVICT_HOT_TURNS > 0 else after
        if not cooled:
            return None
        until = str(cooled[-1].get("ts") or "")
        if not until:
            return None
        chars = sum(len(str(r.get("text") or "")) for r in cooled)
        return until, chars

    def _spawn_eviction(self, until: Optional[str]) -> None:
        """Fire one detached `curate --buffer` and reap its PID. Guarded so only
        one runs at a time; the reaper clears the guard on exit. ``until=None``
        = full flush (shutdown). Fail-open."""
        try:
            if self._evicting or not self._buffer.enabled:
                return
            path = self._buffer.path()
            if path is None:
                return
            proc = spawn_buffer_curate(path, self.cwd, until=until,
                                       provisional=True)
            if proc is None:
                return
            self._evicting = True
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._reap_eviction(proc))
            except RuntimeError:
                # No running loop (teardown): the detached curate still runs; we
                # just can't reap here. Release the guard for a later turn.
                self._evicting = False
        except Exception:  # noqa: BLE001 — memory is a passenger, never the driver
            self._evicting = False

    async def _reap_eviction(self, proc) -> None:
        """Wait on the curate PID directly (NOT a completion ping) and release
        the guard on exit — success OR crash. If Engram quits first the detached
        curate keeps running and the nightly sweep reconciles."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, proc.wait)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._evicting = False

    def _maybe_evict(self) -> None:
        """Size-gated eviction at the turn boundary: if the COOLED tail (out of
        the working-set window) has crossed ENGRAM_EVICT_CHARS since the watermark,
        curate just that range. Never blocks the turn; only fires when material
        has genuinely cooled. Fail-open."""
        try:
            if not EVICT_ON or self._evicting or not self._buffer.enabled:
                return
            edge = self._cooled_edge()
            if edge is None or edge[1] < EVICT_CHARS:
                return
            self._spawn_eviction(edge[0])
        except Exception:  # noqa: BLE001
            pass

    def evict_on_shutdown(self) -> None:
        """Teardown flush (front-ends call this): with a buffer, full-flush it
        (no --until — there is no 'later' in this process; the nightly confirm
        pass covers late reversals). Without one, fall back to the transcript
        --session pass. A store-less driver (perceiving mind) never curates."""
        try:
            if self._buffer.enabled and self._buffer.path() is not None:
                spawn_buffer_curate(self._buffer.path(), self.cwd,
                                    provisional=True)   # until=None → full flush
            elif self._store is not None:
                spawn_session_curate(self.session_id, self.cwd, provisional=True)
        except Exception:  # noqa: BLE001 — teardown must never fail on curation
            pass

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
        if mode == PLAN_MODE and self.permission_mode != PLAN_MODE:
            self._pre_plan_mode = self.permission_mode   # restored on plan approval
        elif mode != PLAN_MODE:
            self._pre_plan_mode = None                   # manual exit consumes it
        self.permission_mode = mode
        await self.disconnect()

    @property
    def plan_restore_target(self) -> str:
        """Where approving a plan lands: the mode the operator was in before entering
        plan, else the launch default — never plan itself."""
        target = self._pre_plan_mode or DEFAULT_MODE
        return target if target != PLAN_MODE else REGULAR_MODE

    def _hooks(self) -> Optional[dict]:
        """SDK hook registrations.

        - PreCompact fires a PROVISIONAL curation pass: the context is about to be
          summarized away, so capture the session's durable insight NOW rather than
          lose the tail to compaction — Brick 3's harness seam (inert until the
          recall CLI grows --provisional). ENGRAM_CURATE_ON_COMPACT=0 disables."""
        out: dict = {}
        if os.environ.get("ENGRAM_CURATE_ON_COMPACT", "1") != "0":
            out["PreCompact"] = [HookMatcher(hooks=[self._on_precompact], timeout=30)]
        return out or None

    async def _on_precompact(self, hook_input, tool_use_id, ctx) -> dict:
        """Never blocks or vetoes compaction — detached spawn, empty verdict.
        Compaction is the one event that would orphan the cooling band, so we
        curate it NOW regardless of the size gate (the hot window survives via
        working-memory re-injection from the immutable buffer, so we still
        exclude it — cooled edge, not full flush). With a LiveBuffer that's the
        buffer path; otherwise the legacy transcript --session pass. A store-less
        (perceiving) driver never curates — camera greetings are not gold."""
        try:
            if self._store is None:
                return {}
            if self._buffer.enabled and self._buffer.path() is not None:
                edge = self._cooled_edge()
                if edge is not None:
                    self._spawn_eviction(edge[0])   # ignore the size gate
            else:
                sid = (hook_input or {}).get("session_id") or self.session_id
                spawn_session_curate(sid, self.cwd, provisional=True)
        except Exception:  # noqa: BLE001 — must never block compaction
            pass
        return {}

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
                # Approving releases plan mode at the CLI level — but into the CLI's
                # own "default" mode, NOT the bypass mode the operator acts in. Restore
                # the pre-plan mode LIVE so the implementation turn runs free (legal here —
                # the forbidden live switch only releases *plan*, which the CLI just did).
                target = self.plan_restore_target
                self.permission_mode = target
                self._pre_plan_mode = None
                if self._client is not None:
                    try:
                        await self._client.set_permission_mode(target)
                    except Exception:  # noqa: BLE001 — older CLI: field is synced,
                        pass           # the next reconnect applies it
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
        # Any OTHER tool reaching this callback (not plan/question): auto-allow. There
        # is no permission prompt — Engram acts and the persona is the guardrail. Under
        # bypassPermissions ordinary tools never even reach here (auto-allowed upstream);
        # this is the safe default for the rare call the CLI still routes through
        # can_use_tool.
        return PermissionResultAllow()

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

    @property
    def active_fallback(self) -> Optional[str]:
        """The model actually serving THIS session when it differs in FAMILY from
        the configured primary — i.e. the CLI silently rotated to the fallback on an
        overload. None while running on the primary (or when actual/primary can't be
        told apart). The UI surfaces this so a rotation to Opus isn't invisible."""
        prim, act = _model_family(self.model), _model_family(self.actual_model)
        return self.actual_model if (prim and act and prim != act) else None

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
            max_buffer_size=CLI_MAX_BUFFER_SIZE,   # base64 images must not overflow the stdio line (see _options)
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

    async def query(self, text: str, *, prepend: str = "") -> AsyncIterator[Event]:
        """Run one turn, streaming text + tool-use events. Reconnects on a stale
        resume once before giving up.

        THE buffer invariant (Brick 3): the LiveBuffer logs the RAW ``text``;
        ``prepend`` (working-memory block + markers, model-only) is sent to the
        SDK but NEVER enters the buffer. The working set is re-derived FROM the
        buffer each turn — logging it back in would feed the derivation its own
        output and compound drift. The assistant row accumulates every text
        event this generator yields (both streams on the retry path, partials
        on timeout) and lands in the ``finally`` — which also runs on
        ``GeneratorExit`` when the front-end abandons the turn (ESC), so an
        interrupted reply is still captured as what the operator actually saw."""
        # Turn boundary: finished sub-agents belong to past turns — drop them so
        # the panel snapshots (and the registry) only ever carry live work. The
        # transcript's inline ✓/✗ lines are the durable record.
        self.tasks = {k: v for k, v in self.tasks.items()
                      if v.get("status") not in TERMINAL_TASK_STATUSES}
        self._buffer.append("user", text)
        sdk_text = prepend + text
        reply: list[str] = []
        try:
            try:
                await self.connect()
                async for ev in self._stream(sdk_text):
                    if ev.kind == "text":
                        reply.append(ev.text)
                    yield ev
            except GeneratorExit:
                raise
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
                    # The server-side session died but the human conversation continues:
                    # park the buffer under the launch id; the retry's fresh sid reclaims
                    # it in _sync_buf_convo — one continuous file across the SDK break.
                    self._revert_buf_convo()
                    await self.disconnect()
                    await self.connect()
                    yield Event("text", "\n\n> ⚠ *couldn't resume the previous thread "
                                        "— started a fresh one*\n\n")
                    async for ev in self._stream(sdk_text):
                        if ev.kind == "text":
                            reply.append(ev.text)
                        yield ev
                else:
                    raise
        finally:
            if reply:
                self._buffer.append("assistant", "".join(reply))
            self._maybe_evict()   # off-hot-path: only fires once the tail cools

    async def _stream(self, text: str) -> AsyncIterator[Event]:
        assert self._client is not None
        await self._client.query(text)
        # We read the RAW message stream (not receive_response) because a delegating turn
        # spans MULTIPLE ResultMessages. The live CLI ordering (verified empirically) is:
        #   tool-use(Agent) → TaskStarted → [sub-agent runs INLINE + TaskUpdated=completed]
        #   → PARENT ResultMessage → TaskNotification → model re-invoked → synthesis → Result
        # The real answer lands in the notification-driven turns AFTER the parent Result, so
        # we must not stop there. `pending` therefore tracks tasks whose async NOTIFICATION
        # hasn't arrived yet — NOT the earlier inline TaskUpdated=completed, which fires
        # before the parent Result and (if we cleared on it) would stop the stream before the
        # sub-agent's result comes back. We stop at the first Result with `pending` empty —
        # the TRUE final one. A normal (no-delegation) turn keeps `pending` empty throughout
        # and stops at its first Result, unchanged. `parent_tool_use_id` marks a sub-agent's
        # OWN messages (skipped — surfaced as the Task* markers) vs. top-level narration.
        pending: set[str] = set()
        bg_launches: set[str] = set()    # tool_use_ids of Agent calls with run_in_background
        bg_started = False               # did THIS turn launch background tasks?
        saw_task = False                                 # did this turn delegate at all?
        it = self._client.receive_messages().__aiter__()
        while True:
            try:
                # Once we've delegated, NEVER block un-timed — a wedged sub-agent or a
                # missing final Result must not hang the turn. A normal (no-task) turn
                # still blocks for its single Result.
                msg = await (asyncio.wait_for(it.__anext__(), SUBAGENT_IDLE_TIMEOUT)
                             if (pending or saw_task) else it.__anext__())
            except asyncio.TimeoutError:                 # silence after delegating
                if pending:
                    # A sync delegation went quiet past the timeout. Release the turn
                    # (never hang the prompt) but keep tracking the stragglers as
                    # BACKGROUND tasks — the idle drain (or the next open stream)
                    # paints their results when they land, so nothing is lost.
                    self._bg_tasks.update(pending)
                    who = ", ".join(self._task_names.get(t, "sub-agent") for t in pending)
                    yield Event("text", f"\n\n> ⏳ *{who} is taking a while — releasing "
                                        f"the turn; the result streams in when it lands*\n\n")
                return
            except StopAsyncIteration:
                return

            if isinstance(msg, UserMessage):
                self._note_checkpoint(msg)               # replayed echo → rewind anchor
            elif isinstance(msg, AssistantMessage):
                if getattr(msg, "parent_tool_use_id", None):
                    continue                             # sub-agent-internal → Task* markers
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield Event("text", block.text)
                    elif getattr(block, "name", None):   # ToolUseBlock-ish
                        if block.name in _INTERACTIVE_TOOLS:
                            continue                     # rendered as a card via _can_use_tool
                        if block.name == "TodoWrite":
                            # The panel IS the render — no status blip for it.
                            yield Event("todos", "", data={
                                "todos": (getattr(block, "input", None) or {}
                                          ).get("todos") or []})
                            continue
                        if (block.name in ("Agent", "Task")
                                and (getattr(block, "input", None) or {}).get("run_in_background")
                                and getattr(block, "id", None)):
                            # Its TaskStarted links back here via tool_use_id.
                            bg_launches.add(block.id)
                        yield Event("tool", _tool_label(block))
            elif isinstance(msg, TaskStartedMessage):
                data = getattr(msg, "data", {}) or {}
                is_wf = data.get("task_type") == "local_workflow"
                name = (f"⚙ {data.get('workflow_name') or 'workflow'}" if is_wf
                        else data.get("subagent_type") or "sub-agent")
                self._task_names[msg.task_id] = name
                saw_task = True
                if is_wf:
                    # A dynamic workflow runs OUTSIDE the conversation and outlives
                    # the turn by design — the model's own turn ends with "running,
                    # I'll report back", and the completion notification re-invokes
                    # it. Track it like a background sub-agent so the prompt stays
                    # responsive for the whole run; the idle drain (or whichever
                    # stream is open) paints progress and the final report.
                    self._bg_tasks.add(msg.task_id)
                    bg_started = True
                    yield Event("text",
                                f"\n\n> ⚙ *workflow "
                                f"**{data.get('workflow_name') or 'unnamed'}** "
                                f"launched — {msg.description}*\n\n")
                elif getattr(msg, "tool_use_id", None) in bg_launches:
                    # Launched with run_in_background: it OUTLIVES this turn by design,
                    # so it must NOT hold the turn open via `pending` (that hold + its
                    # 180s idle timeout was the "detached; result wasn't captured"
                    # noise). Track it on the driver instead — the turn ends at its
                    # own Result and the idle drain paints the result when it lands.
                    self._bg_tasks.add(msg.task_id)
                    bg_started = True
                    yield Event("text", f"\n\n> 🛰 *delegated to **{name}** (background) — "
                                        f"{msg.description}*\n\n")
                else:
                    pending.add(msg.task_id)
                    yield Event("text", f"\n\n> 🛰 *delegated to **{name}** — "
                                        f"{msg.description}*\n\n")
                yield self._task_upd(msg.task_id, name=name, desc=msg.description,
                                     status="running",
                                     background=msg.task_id in self._bg_tasks,
                                     workflow=True if is_wf else None)
            elif isinstance(msg, TaskProgressMessage):
                name = self._task_names.get(msg.task_id, "sub-agent")
                tot = (getattr(msg, "usage", None) or {}).get("total_tokens")
                wp = (getattr(msg, "data", {}) or {}).get("workflow_progress")
                if wp:
                    # Workflow heartbeat: the full phase/agent tree rides every
                    # progress message — snapshot it for the panel + /workflows.
                    snap = workflow_snapshot(wp)
                    bits = (f"{name} · {snap['phase']}"
                            f" · {snap['done']}/{snap['total']} agents")
                    if tot:
                        bits += f" · {tot:,} tok"
                    yield Event("status", bits)          # ephemeral — status line only
                    yield self._task_upd(msg.task_id, tokens=tot, wf=snap)
                    continue
                bits = f"🛰 {name}"
                if getattr(msg, "last_tool_name", None):
                    bits += f" · {msg.last_tool_name}"
                if tot:
                    bits += f" · {tot:,} tok"
                yield Event("status", bits)              # ephemeral — status line only
                yield self._task_upd(msg.task_id, tokens=tot,
                                     last_tool=getattr(msg, "last_tool_name", None))
            elif isinstance(msg, TaskNotificationMessage):
                # The async completion ping — the signal that this sub-agent's result has
                # been folded back in (it re-invokes the model). Clear `pending` HERE, not
                # on the earlier inline TaskUpdated, so the stream keeps reading past the
                # parent Result until every delegated result has actually landed.
                status = getattr(msg, "status", None)
                done = self._bg_finish(msg.task_id, status, getattr(msg, "summary", None))
                if done is not None:
                    # A background task's notification also re-invokes the model, so one
                    # follow-up turn is now owed on the stream (drain_background reads it).
                    self._bg_owed += 1
                    yield self._task_upd(msg.task_id, status=status)
                    yield done
                elif status in TERMINAL_TASK_STATUSES and msg.task_id in pending:
                    pending.discard(msg.task_id)
                    name = self._task_names.get(msg.task_id, "sub-agent")
                    summ = getattr(msg, "summary", None) or f"{name} {status}"
                    mark = "✓" if status == "completed" else "✗"
                    yield self._task_upd(msg.task_id, status=status)
                    yield Event("text", f"\n\n> {mark} *{summ}*\n\n")
            elif isinstance(msg, TaskUpdatedMessage):
                # Inline lifecycle patch. A terminal "completed" here fires BEFORE the parent
                # Result and BEFORE the notification-driven re-invocation, so we must NOT
                # clear `pending` on it (doing so was the bug: the turn stopped before the
                # answer came back). We DO clear on a non-completed terminal state (failed /
                # killed / stopped) — a dead-end with no re-invocation to wait for — so a
                # failed delegation ends promptly instead of idling out to the timeout.
                # Background tasks follow the same asymmetry: a completed one always
                # notifies (handled above), but a killed one may report ONLY here with the
                # notification suppressed (per the SDK docs) — clear it or it leaks forever.
                status = getattr(msg, "status", None)
                done = (self._bg_finish(msg.task_id, status)
                        if status != "completed" else None)
                if done is not None:
                    yield self._task_upd(msg.task_id, status=status)
                    yield done            # no follow-up turn owed without a notification
                elif (status in TERMINAL_TASK_STATUSES and status != "completed"
                        and msg.task_id in pending):
                    pending.discard(msg.task_id)
                    name = self._task_names.get(msg.task_id, "sub-agent")
                    yield self._task_upd(msg.task_id, status=status)
                    yield Event("text", f"\n\n> ✗ *{name} {status}*\n\n")
            elif isinstance(msg, HookEventMessage):
                # Subclass of SystemMessage — must be matched BEFORE it. The only one
                # rendered is the recall inject hook's response: the front-end shows
                # which memory notes fed this turn (memory made visible per turn).
                line = _recall_line(msg)
                if line is not None:
                    yield Event("recall", line)
            elif isinstance(msg, SystemMessage):
                data = getattr(msg, "data", {}) or {}
                if self._fork_next and data.get("session_id"):
                    # Forking: the init carries the NEW branched id — take it
                    # UNCONDITIONALLY (the `or` below would keep the old one and
                    # silently break the fork).
                    self._fork_next = False
                    self.session_id = data["session_id"]
                self.session_id = self.session_id or data.get("session_id")
                self.actual_model = data.get("model") or self.actual_model
                # Flush the id the INSTANT the session exists — not only at the closing
                # ResultMessage below. A turn cut off before its Result (VPN drop / SIGHUP)
                # would otherwise orphan a live server-side session; this matters most for
                # the FIRST turn in a fresh folder, when nothing is on disk yet to resume.
                if self._store and self.session_id:
                    self._store.save(self.cwd, self.session_id)
                self._sync_buf_convo()   # buffer file follows the id, same instant
            elif isinstance(msg, ResultMessage):
                self.session_id = getattr(msg, "session_id", None) or self.session_id
                self.actual_model = getattr(msg, "model", None) or self.actual_model
                self._sync_buf_convo()
                # Persist the (authoritative) session id so the next `engram` in this folder
                # resumes here. Stop only once every delegated task has NOTIFIED (pending
                # empty) — the true final Result, not the parent one mid-delegation.
                # Background tasks never hold the turn; the idle drain owns them next.
                if self._store and self.session_id:
                    self._store.save(self.cwd, self.session_id)
                if not pending:
                    if bg_started and self._bg_tasks:
                        n = len(self._bg_tasks)
                        yield Event("text", f"\n\n> 🛰 *{n} background agent"
                                            f"{'s' if n > 1 else ''} still working*\n\n")
                    return

    def _note_checkpoint(self, msg: UserMessage) -> None:
        """Record a file-rewind anchor from a replayed UserMessage echo (its ``uuid``
        is the target ``rewind_files()`` takes). Skips tool-result echoes and
        sub-agent-internal ones, dedups by uuid (a stale-resume retry re-streams),
        and strips harness injections (system reminders, task pings) from the
        preview so it shows what the operator actually TYPED — an injection-only
        prompt records nothing."""
        if (not getattr(msg, "uuid", None) or getattr(msg, "parent_tool_use_id", None)
                or getattr(msg, "tool_use_result", None) is not None):
            return
        if any(c["uuid"] == msg.uuid for c in self.checkpoints):
            return
        content = msg.content
        text = content if isinstance(content, str) else " ".join(
            b.text for b in content if isinstance(b, TextBlock))
        text = _INJECTED_RE.sub("", text)
        lines = [ln for ln in text.splitlines()
                 if ln.strip() and not ln.lstrip().startswith("[identity]")]
        preview = " ".join(" ".join(lines).split())
        if not preview:
            return
        self.checkpoints.append({"uuid": msg.uuid, "preview": preview[:80],
                                 "ts": time.time()})

    def list_checkpoints(self) -> list[dict]:
        return list(self.checkpoints)

    async def rewind_to(self, uuid: str) -> None:
        """Restore FILES to their state just before the given user prompt. Files
        only — the conversation is untouched (the SDK's rewind semantics)."""
        await self.connect()
        await self._client.rewind_files(uuid)

    def list_sessions(self, limit: int = 9) -> list[dict]:
        """This folder's resumable sessions, newest first — read from Claude
        Code's own transcript dir (the same source the resume recap uses). Each:
        ``{"sid", "mtime", "preview", "current"}``. Fail-open to [] — the picker
        then simply reports none found. Previews come through recall's denoiser;
        a session with no real user prompt falls back to its id."""
        try:
            from recall.transcripts import iter_exchanges, project_transcript_dir
            tdir = project_transcript_dir(self.cwd)
            files = sorted(tdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
                           reverse=True)[:max(1, limit)]
        except Exception:  # noqa: BLE001 — recall absent / dir missing → empty
            return []
        out = []
        for p in files:
            preview = ""
            try:
                for ex in iter_exchanges(p, None):
                    if ex.role == "user" and ex.text.strip():
                        preview = " ".join(ex.text.split())[:80]
                        break
            except Exception:  # noqa: BLE001 — unreadable transcript → id only
                pass
            out.append({"sid": p.stem, "mtime": p.stat().st_mtime,
                        "preview": preview or p.stem[:8],
                        "current": p.stem == self.session_id})
        return out

    async def resume_session(self, sid: str) -> None:
        """Switch to another of this folder's sessions: recycle the client, point
        ``resume=`` at ``sid``, persist the choice. Picking a dead session is
        covered by the stale-resume fallback in query()."""
        await self.disconnect()
        self.session_id = sid
        self.resumed = True
        if self._store:
            self._store.save(self.cwd, sid)
        # Continue that conversation's own buffer (the current one stays put —
        # its thread remains reachable via /sessions).
        self._buf_convo_id = sid
        self._buffer.reseed()

    async def fork(self) -> None:
        """Branch this conversation: the NEXT turn resumes the current session
        with ``fork_session=True``, so the CLI mints a NEW session id and the
        original thread stays untouched (reachable again via /sessions)."""
        self._fork_next = True
        self._fork_buf_copy = True    # seed the branch's buffer with the parent's
        await self.disconnect()

    def _task_upd(self, task_id: str, **fields) -> Event:
        """Update the session task registry and return the refreshed ``task``
        Event (front-ends re-render their panel from ``data["tasks"]``; the
        human narration still flows as text/status events)."""
        entry = self.tasks.setdefault(
            task_id, {"name": self._task_names.get(task_id, "sub-agent"),
                      "desc": "", "status": "running"})
        entry.update({k: v for k, v in fields.items() if v is not None})
        # Copy the entries: each Event must be a point-in-time snapshot, not an
        # alias of the mutable registry (later updates would rewrite history).
        return Event("task", "", data={"tasks": [dict(e) for e in self.tasks.values()]})

    def _bg_finish(self, task_id: str, status, summary: Optional[str] = None) -> Optional[Event]:
        """If ``task_id`` is a tracked background task reaching a terminal status, stop
        tracking it and return its ✓/✗ marker Event (else None). Shared by the
        notification and task-updated paths — a background task may terminate via
        EITHER (the CLI sometimes suppresses the notification for a killed task)."""
        if task_id not in self._bg_tasks or status not in TERMINAL_TASK_STATUSES:
            return None
        self._bg_tasks.discard(task_id)
        name = self._task_names.get(task_id, "sub-agent")
        mark = "✓" if status == "completed" else "✗"
        return Event("text", f"\n\n> {mark} *{summary or f'{name} {status}'}*\n\n")

    @property
    def has_background_tasks(self) -> bool:
        return bool(self._bg_tasks) or self._bg_owed > 0

    async def drain_background(self) -> AsyncIterator[Event]:
        """Keep reading the live stream AFTER a turn has ended, while background
        sub-agents (or their follow-up turns) are still owed — so results PAINT the
        moment they land instead of sitting buffered until the next typed message.
        The front-end runs this while idle and CANCELS it when a typed turn starts
        (single reader on the stream — see app.py's exclusive worker group). No
        per-message timeout: awaiting a silent stream costs nothing, and the exits
        are cancellation, disconnect (StopAsyncIteration), or nothing left owed."""
        if self._client is None or not self.has_background_tasks:
            return
        bg_launches: set[str] = set()
        it = self._client.receive_messages().__aiter__()
        while True:
            try:
                msg = await it.__anext__()
            except StopAsyncIteration:           # client dropped (quit / disconnect)
                return
            if isinstance(msg, UserMessage):
                self._note_checkpoint(msg)       # follow-up turns get anchors too
            elif isinstance(msg, AssistantMessage):
                if getattr(msg, "parent_tool_use_id", None):
                    continue                     # sub-agent-internal narration
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield Event("text", block.text)
                    elif getattr(block, "name", None):
                        if block.name in _INTERACTIVE_TOOLS:
                            continue
                        if block.name == "TodoWrite":
                            yield Event("todos", "", data={
                                "todos": (getattr(block, "input", None) or {}
                                          ).get("todos") or []})
                            continue
                        if (block.name in ("Agent", "Task")
                                and (getattr(block, "input", None) or {}).get("run_in_background")
                                and getattr(block, "id", None)):
                            bg_launches.add(block.id)
                        yield Event("tool", _tool_label(block))
            elif isinstance(msg, TaskStartedMessage):
                # A follow-up turn may delegate AGAIN (a sub-agent or a whole
                # workflow). The drain has no turn to hold open, so sync or
                # background, track it as background and keep listening.
                data = getattr(msg, "data", {}) or {}
                is_wf = data.get("task_type") == "local_workflow"
                name = (f"⚙ {data.get('workflow_name') or 'workflow'}" if is_wf
                        else data.get("subagent_type") or "sub-agent")
                self._task_names[msg.task_id] = name
                self._bg_tasks.add(msg.task_id)
                bg = getattr(msg, "tool_use_id", None) in bg_launches
                mark = "⚙ *workflow" if is_wf else "🛰 *delegated to"
                yield Event("text", f"\n\n> {mark} **{name.removeprefix('⚙ ')}**"
                                    f"{' (background)' if bg and not is_wf else ''}"
                                    f" — {msg.description}*\n\n")
                yield self._task_upd(msg.task_id, name=name, desc=msg.description,
                                     status="running", background=bg,
                                     workflow=True if is_wf else None)
            elif isinstance(msg, TaskProgressMessage):
                name = self._task_names.get(msg.task_id, "sub-agent")
                tot = (getattr(msg, "usage", None) or {}).get("total_tokens")
                wp = (getattr(msg, "data", {}) or {}).get("workflow_progress")
                if wp:
                    snap = workflow_snapshot(wp)
                    bits = (f"{name} · {snap['phase']}"
                            f" · {snap['done']}/{snap['total']} agents")
                    if tot:
                        bits += f" · {tot:,} tok"
                    yield Event("status", bits)
                    yield self._task_upd(msg.task_id, tokens=tot, wf=snap)
                    continue
                bits = f"🛰 {name}"
                if getattr(msg, "last_tool_name", None):
                    bits += f" · {msg.last_tool_name}"
                if tot:
                    bits += f" · {tot:,} tok"
                yield Event("status", bits)
                yield self._task_upd(msg.task_id, tokens=tot,
                                     last_tool=getattr(msg, "last_tool_name", None))
            elif isinstance(msg, TaskNotificationMessage):
                status = getattr(msg, "status", None)
                done = self._bg_finish(msg.task_id, status,
                                       getattr(msg, "summary", None))
                if done is not None:
                    self._bg_owed += 1           # its follow-up turn is now owed
                    yield self._task_upd(msg.task_id, status=status)
                    yield done
            elif isinstance(msg, TaskUpdatedMessage):
                # Same asymmetry as _stream: "completed" terminals arrive via the
                # notification (above); a killed/failed task may report ONLY here.
                status = getattr(msg, "status", None)
                done = (self._bg_finish(msg.task_id, status)
                        if status != "completed" else None)
                if done is not None:
                    yield self._task_upd(msg.task_id, status=status)
                    yield done
            elif isinstance(msg, SystemMessage):
                data = getattr(msg, "data", {}) or {}
                self.session_id = self.session_id or data.get("session_id")
            elif isinstance(msg, ResultMessage):
                # End of one follow-up turn. (A follow-up that delegated a SYNC
                # sub-agent emits a mid-delegation parent Result too — decrementing
                # on it is fine, since its own notification re-adds the debt and
                # `_bg_tasks` keeps the drain alive meanwhile.)
                self.session_id = getattr(msg, "session_id", None) or self.session_id
                if self._store and self.session_id:
                    self._store.save(self.cwd, self.session_id)
                if self._bg_owed > 0:
                    self._bg_owed -= 1
                if not self.has_background_tasks:
                    return

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr[-6:])
