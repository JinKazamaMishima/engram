#!/usr/bin/env python3
"""Engram — the full terminal TUI, on Textual (MIT). Engram's home.

Full-screen, a scrollable markdown conversation, incremental streaming, a
multi-line prompt, drag-drop / clipboard image attachments, a command palette,
copy-a-reply, and a bespoke "engram" star theme. Reuses the subscription-backed
core (``infra/engram/core.py``) — the UI talks only to a ``ModelDriver``, so when
the local model lands we swap the driver and this whole TUI is unchanged.

    .venv/bin/python infra/engram/app.py        # or: ./infra/engram/engram
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attach import grab_clipboard_image, is_image, parse_dropped_paths  # noqa: E402
from core import (  # noqa: E402
    _STRIPPED_API_KEY,
    EFFORT_LEVELS,
    ENGRAM_CWD,
    PLAN_MODE,
    REGULAR_MODE,
    AgentSDKDriver,
    LaunchLock,
    ModelDriver,
    _model_family,
    render_context_md,
)
from rich.markup import escape  # noqa: E402
from textual import events, work  # noqa: E402
from textual.app import App, ComposeResult, SystemCommand  # noqa: E402
from textual.containers import VerticalScroll  # noqa: E402
from textual.message import Message  # noqa: E402
from textual.theme import Theme  # noqa: E402
from textual.widgets import Footer, Markdown, OptionList, Static, TextArea  # noqa: E402
from textual.widgets.option_list import Option  # noqa: E402

# On reattach, replay the tail of the resumed conversation so a fresh TUI isn't blank —
# you can SEE where you left off (the tmux-like bit). Trailing logical turns (adjacent
# same-role fragments collapsed into one round), each truncated for a compact recap.
# OPT-IN via ENGRAM_RESUME_RECAP=1 (it surfaces prior prose from the local transcript).
RESUME_RECAP_TURNS = 6
RESUME_RECAP_CHARS = 400

# Slash commands, each (name, one-line help) — the source for the homegrown
# dropdown (type "/" → ↑/↓-navigable menu, like Claude Code).
SLASH_CMDS = [
    ("/new", "start a fresh thread"),
    ("/effort", "set reasoning effort  (low|medium|high|xhigh|max)"),
    ("/model", "switch model  (e.g. opus[1m], sonnet)"),
    ("/mode", "toggle plan ↔ regular  (or shift+tab)"),
    ("/ultracode", "toggle multi-agent workflow orchestration"),
    ("/workflows", "workflow runs this session — phases + agents"),
    ("/fleet", "parallel Engram sessions across repos  (/fleet <path> [task])"),
    ("/agent", "delegate to a sub-agent  (e.g. /agent Explore <task>)"),
    ("/context", "show context-window usage"),
    ("/rewind", "restore files to before an earlier message"),
    ("/sessions", "resume another of this folder's sessions"),
    ("/fork", "branch this conversation (original kept)"),
    ("/export", "save this conversation as markdown"),
    ("/copy", "copy my last reply to the clipboard  (or ctrl+y)"),
    ("/status", "show session · model · effort"),
    ("/paste", "attach a clipboard image"),
    ("/exit", "quit Engram"),
]
# Commands that take an argument: selecting one completes the text and waits (for
# /effort it then offers the levels; for /agent, the sub-agent names); the rest
# submit immediately on select.
ARG_CMDS = {"/effort", "/model", "/agent", "/fleet"}
# State-changing commands — they touch the driver / warm client, so they must not
# run while a reply is streaming (mid-turn they're blocked, never queued). /context
# is read-only but pokes the warm client (a control request), so it's gated too.
STATE_CMDS = ("/new", "/effort", "/model", "/paste", "/context", "/rewind",
              "/sessions", "/fork", "/export")

# Sub-agents Engram can delegate to via /agent (and that the model auto-invokes via the
# Task tool). These mirror Claude Code's built-ins — confirm the exact names the CLI
# exposes with /context's `agents` list, and adjust here if they differ.
SUBAGENTS = ("Explore", "Plan", "general-purpose")

# Corpus label of the shared soul as it appears on the inject hook's wire format
# (recall_inject._format_system_message; == recall config.GLOBAL_SCOPE).
GLOBAL_SCOPE = "global"


def render_recall_line(line: str | None) -> str:
    """Markup for the per-turn memory-provenance line. ``line`` is the inject hook's
    ``corpus:slug`` list ('' = the hook ran and surfaced nothing — an honest zero;
    None = it never fired — the injection-outage tell). Soul notes keep a ``soul:``
    prefix; project notes drop theirs (this project is the default context). Long
    lists cap at 3 slugs + a count. Pure (unit-testable without Textual)."""
    if line is None:
        body = "[dim]silent — no injection this turn[/dim]"
    elif not line:
        body = "no notes"
    else:
        names = []
        for entry in (e.strip() for e in line.split(",")):
            if not entry:
                continue
            corpus, _, slug = entry.partition(":")
            names.append(f"soul:{slug}" if corpus == GLOBAL_SCOPE else slug or corpus)
        shown = ", ".join(names[:3]) + (f"  +{len(names) - 3}" if len(names) > 3 else "")
        body = f"{len(names)} note{'s' if len(names) != 1 else ''} · {escape(shown)}"
    return f"◆ recall · {body}"


def render_tasks_line(todos: list, tasks: list) -> str:
    """The one-line task panel: todo progress + the active step, then each LIVE
    sub-agent with its state. Finished agents don't stack up — the transcript
    already carries their ✓/✗ summary lines inline, so terminal entries collapse
    into compact counters (and the whole panel empties when nothing is live).
    Pure (unit-testable without Textual); empty string when nothing to show."""
    parts = []
    if todos:
        done = sum(1 for t in todos if t.get("status") == "completed")
        line = f"☑ {done}/{len(todos)}"
        cur = next((t for t in todos if t.get("status") == "in_progress"), None)
        if cur:
            line += f"  ▶ {cur.get('activeForm') or cur.get('content') or ''}"
        parts.append(line)
    n_done = n_dead = 0
    for t in tasks:
        status = t.get("status")
        if status == "completed":
            n_done += 1
        elif status in ("failed", "stopped", "killed"):
            n_dead += 1
        elif t.get("workflow"):
            # A dynamic-workflow run: show where it is in its phase/agent tree
            # (the wf snapshot rides every progress heartbeat — see
            # core.workflow_snapshot); /workflows expands the full tree.
            w = t.get("wf") or {}
            bit = f"⚙ {t.get('name', 'workflow').removeprefix('⚙ ')} ⏳"
            if w.get("total"):
                bit += f" {w.get('phase', '')} {w['done']}/{w['total']}"
            if t.get("tokens"):
                bit += f" {int(t['tokens']) // 1000}k"
            parts.append(bit)
        else:
            bit = f"🛰 {t.get('name', 'sub-agent')} ⏳"
            if t.get("tokens"):
                bit += f" {int(t['tokens']) // 1000}k"
            parts.append(bit)
    if n_done:
        parts.append(f"🛰 ✓ {n_done} done")
    if n_dead:
        parts.append(f"🛰 ✗ {n_dead} failed")
    return "   ".join(parts)


def _age(ts) -> str:
    """Compact relative-age suffix for a checkpoint row ('' when unknown)."""
    if not ts:
        return ""
    m = int(max(0.0, time.time() - float(ts)) // 60)
    if m == 0:
        return "   · just now"
    if m < 60:
        return f"   · {m}m ago"
    return f"   · {m // 60}h {m % 60}m ago"

# Models offered in the /model dropdown. Free-form still works — set_model passes any
# string straight to the CLI; this is discoverability only. Aliases the CLI accepts;
# opus[1m] = the 1M-context window. (name, one-line description) like SLASH_CMDS.
MODELS = (
    ("opus[1m]", "Opus 4.8 · 1M context (default)"),
    ("opus",     "Opus 4.8 · 200K"),
    ("sonnet",   "Sonnet 4.6"),
    ("fable",    "Fable 5"),
    ("haiku",    "Haiku 4.5 · fastest"),
)

# Sticky-header palette (Rich-markup hex, matching ENGRAM_THEME — Static markup can't
# see Textual's $accent vars). The logo is a pixel gem (half-block "pixels") — the
# star Engram is named for. Later this header area can swap to a pixel-rendered face.
LOGO_C = "#67E8F9"   # cyan — the star's glint
NAME_C = "#E8ECF8"   # starlight
SUB_C = "#8593B8"    # muted
ENGRAM_LOGO = ("█   █", " █ █ ", "  █  ")

# Twinkling starfield in the header's right field — fixed star positions, brightness
# flickers each tick (a gentle ~0.7s timer). Movement, kept subtle.
STAR_GLYPHS = ("·", "✦", "✧", "⋆", "˖")
STAR_DIM = "#5A6C96"
STAR_LIT = "#C7D6FF"
STAR_CYAN = "#67E8F9"

# What opens the home: true facts about the engram — the memory trace the project is
# named for. One is chosen at random each launch.
ENGRAM_EPIGRAPHS = (
    "engram (n.) — the physical trace a memory leaves behind.",
    "the mark that outlasts the moment it was made.",
    "coined in 1904 for the idea that every memory is written into matter.",
    "to find where a memory lives is to go looking for the self.",
    "not the recollection, but the change it leaves in you.",
    "what persists when the moment is gone.",
)

# Injected each typed turn while /ultracode is on — mirrors Claude Code's standing
# "ultracode" opt-in so the CLI's Workflow tool treats multi-agent orchestration as the
# default for substantive work. Prepended to the turn like the identity marker.
ULTRACODE_REMINDER = (
    "<system-reminder>\n"
    "Ultracode is on for the session — multi-agent Workflow orchestration is a standing "
    "opt-in. For every substantive task, prefer authoring and running a Workflow "
    "(decompose → fan out in parallel → adversarially verify → synthesize) over solving "
    "inline; go solo only for trivial or conversational turns. Favor the most thorough, "
    "correct result; token cost is not the constraint. Stays on until the user runs "
    "/ultracode off.\n"
    "</system-reminder>\n\n"
)

# A deep night-sky palette: blue-white starlight text on near-black, with a single
# cyan glint as the accent — memory as points of light held in the dark.
ENGRAM_THEME = Theme(
    name="engram",
    primary="#9FB9FF", secondary="#C4B5FD", accent="#67E8F9",
    foreground="#E8ECF8", background="#0A0E1A", surface="#121829", panel="#1B2340",
    success="#86EFAC", warning="#FBBF24", error="#FB7185",
    dark=True, variables={"input-cursor-background": "#67E8F9"},
)


def _seam(prev: str, nxt: str) -> str:
    """Newlines to insert between two streamed markdown blocks so they don't run
    together on one line (a block ending in ':' gluing onto the next). Counts the break
    already present at the seam — prev's trailing newlines + nxt's leading newlines —
    and tops it up to a blank line, so it leaves core.py's already-\\n\\n-padded
    sub-agent markers untouched and never triple-breaks. Each Event('text') is a
    COMPLETE block (no token-level streaming), so a paragraph break between blocks is
    always the right boundary — a single flowing paragraph arrives as one block."""
    if not prev:
        return ""
    have = (len(prev) - len(prev.rstrip("\n"))) + (len(nxt) - len(nxt.lstrip("\n")))
    return "\n" * max(0, 2 - have)


class UserMsg(Static):
    """One operator turn — a single accent stripe sets it apart from Engram's reply."""


class PromptArea(TextArea):
    """Multi-line prompt. Enter submits; Ctrl+J / Shift+Enter insert a newline
    (terminals can't always tell Shift+Enter from Enter, so Ctrl+J is the reliable
    newline). Drag-dropped / pasted file PATHS are caught and attached."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        app = self.app
        # When the slash-command dropdown is open it owns ↑/↓/Enter/Tab/Esc; every
        # other key falls through to normal editing (and re-filters the menu).
        if getattr(app, "_menu_open", False):
            handler = {
                "down": lambda: app._menu_move(1),    # type: ignore[attr-defined]
                "up": lambda: app._menu_move(-1),     # type: ignore[attr-defined]
                "enter": app._accept_menu,            # type: ignore[attr-defined]
                "tab": app._accept_menu,              # type: ignore[attr-defined]
                "escape": app._hide_menu,             # type: ignore[attr-defined]
            }.get(event.key)
            if handler is not None:
                event.prevent_default()
                event.stop()
                handler()
                return
        # An interaction card is open (plan approval / option question): ↑/↓ move the
        # highlight; Enter with an EMPTY prompt picks it; Enter with text submits that text
        # as the free-text answer/feedback; Esc cancels. Every OTHER key falls through to
        # normal editing, so "type your own answer / just chat" works without leaving the card.
        if getattr(app, "_interact_open", False):
            if event.key in ("down", "up"):
                event.prevent_default()
                event.stop()
                app._interact_move(1 if event.key == "down" else -1)   # type: ignore[attr-defined]
                return
            if event.key == "enter":
                event.prevent_default()
                event.stop()
                if self.text.strip():
                    self.post_message(self.Submitted(self.text))       # → free-text answer
                else:
                    app._interact_accept_highlight()                   # type: ignore[attr-defined]
                return
            if event.key == "escape":
                event.prevent_default()
                event.stop()
                app._interact_cancel()                                 # type: ignore[attr-defined]
                return
        if event.key == "escape" and getattr(app, "_busy", False):
            event.prevent_default()                # ESC mid-reply → stop Engram now
            event.stop()
            await app.action_interrupt()           # type: ignore[attr-defined]
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key == "shift+tab":              # cycle plan ↔ regular, like Claude Code
            event.prevent_default()
            event.stop()
            await app.action_cycle_mode()         # type: ignore[attr-defined]
            return
        if event.key == "ctrl+c":                 # copy the selection — NEVER quit (quit = Ctrl+Q)
            event.prevent_default()
            event.stop()
            app.action_copy_selection()           # type: ignore[attr-defined]
            return
        await super()._on_key(event)

    async def on_paste(self, event: events.Paste) -> None:
        paths = parse_dropped_paths(event.text)
        if paths:
            event.prevent_default()
            event.stop()
            self.app.attach_files(paths)  # type: ignore[attr-defined]


class EngramApp(App):
    CSS = """
    Screen { background: $background; }
    #vhead { dock: top; height: 3; padding: 0 2; background: $panel; }
    #convo { height: 1fr; padding: 1 2 0 2; scrollbar-size-vertical: 1; }
    #welcome { color: $accent; text-style: italic; padding: 0 1; margin: 0 0 1 0; }
    UserMsg {
        color: $primary; text-style: bold;
        margin: 1 0; padding: 0 0 0 1;
        border-left: thick $accent 55%;
    }
    Markdown { margin: 0 0 1 0; padding: 0 1; }
    #status { height: 1; color: $secondary; text-style: italic; padding: 0 2; }
    #chips { height: auto; color: $accent; padding: 0 2; }
    #queued { height: auto; color: $warning; text-style: italic; padding: 0 2; }
    #fleet { height: auto; color: $accent; padding: 0 2; }
    #tasks { height: auto; color: $secondary; padding: 0 2; }
    #cmdmenu {
        display: none;
        margin: 0 2; height: auto; max-height: 8;
        background: $panel; border: round $accent 60%;
        scrollbar-size-vertical: 1;
    }
    #cmdmenu > .option-list--option { padding: 0 1; color: $foreground; }
    #cmdmenu > .option-list--option-highlighted {
        background: $accent 25%; color: $foreground; text-style: bold;
    }
    .interact {
        height: auto; max-height: 12; margin: 0 0 1 0;
        background: $panel; border: round $accent 60%;
    }
    .interact > .option-list--option { padding: 0 1; color: $foreground; }
    .interact > .option-list--option-highlighted {
        background: $accent 25%; color: $foreground; text-style: bold;
    }
    .plancard { border: round $secondary 50%; margin: 0 0 1 0; padding: 0 1; }
    .interact-hint { color: $secondary; text-style: italic; padding: 0 1; margin: 0 0 1 0; }
    .recall-line { color: $secondary; padding: 0 1; margin: 0 0 1 0; }
    #perception {
        display: none;                 /* shown only when ENGRAM_PERCEIVE is on */
        height: auto; margin: 0 2; padding: 0 1;
        background: $panel; border: round $accent 45%;
    }
    PromptArea {
        margin: 0 2 1 2; height: auto; max-height: 12;
        background: $surface; color: $foreground; border: round $accent 35%;
    }
    PromptArea:focus { border: round $accent; }
    """
    BINDINGS = [
        ("ctrl+c", "copy_selection", "copy"),
        ("ctrl+q", "quit", "quit"),
        ("ctrl+n", "new_thread", "new"),
        ("ctrl+v", "paste_image", "paste image"),
        ("ctrl+y", "copy_reply", "copy reply"),
    ]

    def __init__(self, driver: ModelDriver | None = None) -> None:
        super().__init__()
        self.driver: ModelDriver = driver or AgentSDKDriver()
        self._busy = False
        self._attachments: list = []     # pending file paths for the next turn
        self._last_reply = ""            # for copy-a-reply
        self._menu_open = False          # slash-command dropdown visible?
        self._queue: list[tuple[str, list]] = []   # type-ahead: msgs typed while busy
        self._pending_mode: str | None = None      # plan/regular armed via shift+tab mid-reply
        self._ultracode = False                     # /ultracode: standing workflow-orchestration opt-in
        self._rewind_note = ""                      # model-only heads-up after a /rewind
        self._fallback_shown = False                # one-time notice when the model rotates
        self._todos: list = []                      # last TodoWrite list (persists across turns)
        self._tasks_snapshot: list = []             # last sub-agent registry snapshot
        self._fleet = None                          # Fleet (lazy — created on first /fleet)
        self._perception = None                     # PerceptionBridge (opt-in: ENGRAM_PERCEIVE=1)
        # Interactive tools (plan approval · option questions). The driver calls
        # self._handle_interaction through its on_interaction seam; a live card parks the
        # turn until you pick or type. _cur_* hold the active streaming Markdown so an
        # interaction can BREAK it — text after the card renders below it, never glued above.
        self._interact: dict | None = None          # {future, list} while a card is open
        self._interact_open = False
        # The SDK spawns each permission callback as its OWN task, so parallel
        # AskUserQuestion calls (the model often sends one per question) would race
        # for the single _interact slot — cards stack, only the last one answers,
        # the rest wedge. The lock serializes them: ask → answer → next.
        self._interact_lock = asyncio.Lock()
        self._recall_shown = False                  # provenance line rendered this turn?
        self._cur_md = None
        self._cur_stream = None
        self._cur_last = ""
        try:
            self.driver.on_interaction = self._handle_interaction
        except Exception:  # noqa: BLE001 — a driver may forbid the attr set; degrade to no-UI
            pass

    def compose(self) -> ComposeResult:
        yield Static(id="vhead")
        yield VerticalScroll(id="convo")
        yield Static("", id="status")
        yield Static("", id="chips")
        yield Static("", id="queued")
        yield Static("", id="fleet")        # live fleet-member strip (⚑ per repo)
        yield Static("", id="tasks")        # live todo + sub-agent panel
        yield OptionList(id="cmdmenu")
        yield Static("", id="perception")   # live senses HUD (shown when ENGRAM_PERCEIVE on)
        yield PromptArea(id="prompt", soft_wrap=True, tab_behavior="focus")
        yield Footer()

    async def on_mount(self) -> None:
        self.register_theme(ENGRAM_THEME)
        self.theme = "engram"
        self._render_header()
        self.set_interval(0.7, self._render_header)   # twinkle
        prompt = self.query_one("#prompt", PromptArea)
        prompt.border_title = "message"
        await self._add(Static("✦ Engram — " + random.choice(ENGRAM_EPIGRAPHS), id="welcome"))
        prompt.focus()
        # If the driver resumed a saved session for this folder, replay its tail so the
        # screen isn't blank on reattach, then say so.
        if getattr(self.driver, "resumed", False):
            await self._render_resumed_history()
            await self._add(Static("[dim]· · ·  resumed your last conversation here "
                                   "— /new for a fresh thread  · · ·[/dim]"))
            self._status("resumed last session  ·  /new for fresh")
        else:
            self._status("ready")
        if os.environ.get("ENGRAM_PERCEIVE"):
            self._start_perception()

    async def _render_resumed_history(self) -> None:
        """Replay the last few turns of a resumed session so reattaching to a fresh TUI
        shows where you left off (otherwise the screen is blank but for the 'resumed'
        note — the thing that makes a dropped-VPN reconnect feel like tmux). Reads Claude
        Code's own session transcript for this cwd through recall's denoiser. Best-effort:
        any failure — recall not importable, transcript missing or not yet flushed —
        silently skips the recap; it must never block the home from opening.

        OFF by default: it renders prior conversation prose (from the local transcript), so
        it's opt-in via ``ENGRAM_RESUME_RECAP=1``. The plain 'resumed last session' note
        still shows either way — only the prose recap is gated."""
        if not os.environ.get("ENGRAM_RESUME_RECAP"):
            return
        sid = getattr(self.driver, "session_id", None)
        cwd = getattr(self.driver, "cwd", None)
        if not sid or not cwd:
            return
        try:
            from recall.transcripts import (
                iter_exchanges,
                project_transcript_dir,
                session_transcript_path,
            )
            path = session_transcript_path(project_transcript_dir(cwd), sid)
            if not path.exists():
                return
            exchanges = list(iter_exchanges(path, None))
        except Exception:  # noqa: BLE001 — the recap is a nicety, never a blocker
            return
        if not exchanges:
            return
        # One logical turn can span several same-role events (assistant text split by tool
        # calls); collapse adjacent same-role fragments so the recap reads as real rounds
        # (a user prompt + its reply) rather than a wall of one turn's fragments.
        merged: list[list] = []
        for ex in exchanges:
            if merged and merged[-1][0] == ex.role:
                merged[-1][1] += " " + ex.text
            else:
                merged.append([ex.role, ex.text])
        await self._add(Static("[dim]──  earlier in this thread  "
                               "──────────────────────[/dim]"))
        for role, text in merged[-RESUME_RECAP_TURNS:]:
            body = " ".join(text.split())
            if len(body) > RESUME_RECAP_CHARS:
                body = body[:RESUME_RECAP_CHARS].rstrip() + "…"
            prefix = "❯ " if role == "user" else ""
            await self._add(Static(f"[dim]{prefix}{escape(body)}[/dim]"))
        await self._add(Static("[dim]──  now  "
                               "───────────────────────────────────[/dim]"))

    # ---- command palette ----
    def get_system_commands(self, screen):
        yield from super().get_system_commands(screen)
        yield SystemCommand("Engram: New thread", "Clear context and start fresh",
                            lambda: self.run_worker(self._reset_thread()))
        yield SystemCommand("Engram: Copy last reply", "Copy Engram's last message",
                            self.action_copy_reply)
        yield SystemCommand("Engram: Attach clipboard image", "Paste a screenshot",
                            lambda: self.run_worker(self.action_paste_image()))
        yield SystemCommand("Engram: Context usage", "Show context-window usage",
                            lambda: self.run_worker(self._show_context()))
        for lvl in EFFORT_LEVELS:
            yield SystemCommand(f"Engram: Effort → {lvl}", f"Set reasoning effort to {lvl}",
                                lambda level=lvl: self.run_worker(self._apply_effort(level)))

    # ---- helpers ----
    def _status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _subtitle(self) -> str:
        model = getattr(self.driver, "model", "?")
        effort = getattr(self.driver, "effort", "")
        # A live fallback rotation is loud in the header: "fable → opus ⚠ fallback".
        # The 0.7s twinkle re-renders this, so it appears within a blink of the switch.
        fb = getattr(self.driver, "active_fallback", None)
        if fb:
            model = f"{model} → {_model_family(fb) or fb} ⚠ fallback"
        mode = " · ⏸ plan" if self._effective_mode() == PLAN_MODE else ""
        ultra = " · ⚡ ultracode" if self._ultracode else ""
        tail = " · subscription" if _STRIPPED_API_KEY else ""
        base = f"{model} · {effort}" if effort else f"{model}"
        return f"{base}{mode}{ultra}{tail}"

    def _render_header(self) -> None:
        try:                                     # a twinkle tick can race mount/teardown
            head = self.query_one("#vhead", Static)
        except Exception:  # noqa: BLE001 — header not mounted (yet / anymore); skip
            return
        a, b, c = ENGRAM_LOGO
        sub = self._subtitle()
        left = [
            (a, f"[{LOGO_C}]{a}[/]"),
            (f"{b}  E N G R A M", f"[{LOGO_C}]{b}[/]  [b {NAME_C}]E N G R A M[/]"),
            (f"{c}  {sub}", f"[{LOGO_C}]{c}[/]  [{SUB_C}]{sub}[/]"),
        ]
        width = head.content_size.width
        start = max(len(p) for p, _ in left) + 4
        lines = [markup + " " * (start - len(plain))
                 + (self._starline(width - start, i) if width > start + 1 else "")
                 for i, (plain, markup) in enumerate(left)]
        head.update("\n".join(lines))

    def _starline(self, width: int, seed: int) -> str:
        """One row of the header starfield: positions/glyphs fixed (seeded), each
        star's brightness flickers per call → twinkle without drifting."""
        if width <= 1:
            return ""
        rng = random.Random(seed * 9973 + 7)
        cols = set(rng.sample(range(width), min(max(2, width // 6), width)))
        glyph = {col: rng.choice(STAR_GLYPHS) for col in cols}
        out = []
        for col in range(width):
            if col not in cols:
                out.append(" ")
                continue
            r = random.random()
            if r < 0.45:
                out.append(" ")                                       # dark
            elif r < 0.80:
                out.append(f"[{STAR_DIM}]{glyph[col]}[/]")            # dim
            elif r < 0.94:
                out.append(f"[{STAR_LIT}]{glyph[col]}[/]")            # bright
            else:
                out.append(f"[{STAR_CYAN}]✦[/]")                      # rare cyan sparkle
        return "".join(out)

    def _status_line(self) -> str:
        d = self.driver
        actual = getattr(d, "actual_model", None)
        fb = getattr(d, "active_fallback", None)
        cfg_fb = getattr(d, "fallback_model", None)
        return (f"model={getattr(d, 'model', '?')}"
                + (f"  (SDK reports: {actual})" if actual else "")
                + (f"  ⚠ ON FALLBACK: {fb}" if fb
                   else (f" · fallback={cfg_fb}" if cfg_fb else ""))
                + f" · effort={getattr(d, 'effort', '?')}"
                + f" · session={getattr(d, 'session_id', None) or 'fresh'}")

    def _scroll(self) -> None:
        # Force the convo region to re-layout+repaint each event, THEN pin to the
        # bottom. scroll_end alone is a no-op once already at the bottom, so streamed
        # MarkdownStream writes + freshly-mounted widgets sat un-composited until the
        # next input triggered a re-layout — the "reply only shows on my next message"
        # bug. The header kept painting only because its 0.7s timer marks it dirty.
        convo = self.query_one("#convo", VerticalScroll)
        convo.refresh(layout=True)
        convo.scroll_end(animate=False)

    async def _add(self, widget) -> None:
        await self.query_one("#convo", VerticalScroll).mount(widget)
        self._scroll()

    async def _reset_thread(self) -> None:
        await self.driver.disconnect()
        self.driver.reset()
        self._todos = []
        self._tasks_snapshot = []
        self._render_tasks()
        await self._add(Static("[dim]· · ·  new thread  · · ·[/dim]"))

    async def _apply_effort(self, level: str) -> None:
        await self.driver.set_effort(level)
        self._render_header()
        self._status(f"effort → {level}  (applies to your next message)")

    # ---- plan ↔ regular mode (shift+tab, like Claude Code · or /mode) ----
    def _effective_mode(self) -> str:
        """The mode shown to the user: a pending (armed-mid-reply) mode if one is queued,
        else the driver's live mode."""
        return self._pending_mode or getattr(self.driver, "permission_mode", REGULAR_MODE)

    async def action_cycle_mode(self) -> None:
        """Flip plan ↔ regular. Bound to shift+tab; also reachable via /mode."""
        target = REGULAR_MODE if self._effective_mode() == PLAN_MODE else PLAN_MODE
        await self._set_mode(target)

    async def _set_mode(self, target: str) -> None:
        """Apply a permission mode. When idle, recycle the client so the next turn
        reconnects in the new mode; mid-reply we must NOT drop the warm client — so we
        ARM it and apply the moment the turn ends (it governs the next turn, exactly like
        Claude Code)."""
        if self._busy or getattr(self.driver, "has_background_tasks", False):
            # Mid-reply OR background agents out: applying now would recycle (busy)
            # or kill (background) the warm client — ARM it; it applies at the next
            # quiet turn end and governs the turn after, exactly like Claude Code.
            self._pending_mode = target
            self._render_header()
            self._status(self._mode_msg(target) + "  ·  applies after this reply")
            return
        try:
            await self.driver.set_permission_mode(target)
        except Exception as exc:  # noqa: BLE001 — older driver / control unsupported
            self._status(f"mode unchanged — {type(exc).__name__}")
            return
        self._pending_mode = None
        self._render_header()
        self._status(self._mode_msg(target))

    @staticmethod
    def _mode_msg(mode: str) -> str:
        if mode == PLAN_MODE:
            return "⏸ plan mode — Engram investigates and proposes, makes no changes"
        return "▶ regular mode — Engram acts"

    # ---- attachments (drag-drop a file · ctrl+v a clipboard image · /paste) ----
    def attach_files(self, paths) -> None:
        for p in paths:
            p = Path(p)
            if p not in self._attachments:
                self._attachments.append(p)
        self._render_chips()
        self._status(f"📎 {len(self._attachments)} attached — sent with your next message")

    def _render_chips(self) -> None:
        chips = "  ".join(("🖼 " if is_image(p) else "📎 ") + escape(p.name)
                          for p in self._attachments)
        self.query_one("#chips", Static).update(chips)

    # ---- actions ----
    async def action_new_thread(self) -> None:
        if self._busy or getattr(self.driver, "has_background_tasks", False):
            return                    # same guard as /new — don't orphan live agents
        await self._reset_thread()

    async def action_paste_image(self) -> None:
        self._status("checking clipboard…")
        path = await grab_clipboard_image()
        if path:
            self.attach_files([path])
        else:
            self._status("no image in the clipboard (drop a file, or copy a screenshot first)")

    def copy_to_clipboard(self, text: str) -> None:
        """EVERY copy in the app lands here — ours (Ctrl+Y, /copy) AND Textual's own:
        the Screen binds ctrl+c → screen.copy_text for drag-selections, which calls
        this and nothing else. Base Textual only emits OSC52, which terminals like
        GNOME/VTE silently gate — the historic "I copied but got nothing" bug — so
        chase it with a real clipboard tool (Wayland/X11/macOS) and always confirm
        on the status line (silent success is indistinguishable from silent failure)."""
        super().copy_to_clipboard(text)                   # OSC52 (works over SSH)
        if not text:
            return
        for tool in (["wl-copy"], ["xclip", "-selection", "clipboard"],
                     ["xsel", "--clipboard", "--input"], ["pbcopy"]):
            try:
                if subprocess.run(tool, input=text.encode(), timeout=5).returncode == 0:
                    break
            except (FileNotFoundError, OSError, subprocess.SubprocessError):
                continue
        self._status(f"📋 copied {len(text)} chars to the clipboard")

    def _copy_text(self, text: str) -> None:
        self.copy_to_clipboard(text)

    def action_copy_selection(self) -> None:
        """Ctrl+C — copy the current selection: text selected INSIDE the prompt box
        first, else the screen drag-selection. Never quits (quit is Ctrl+Q; Ctrl+Y
        copies the last reply). Reached when the prompt has focus (its _on_key routes
        ctrl+c here) or when nothing is selected (Screen's own ctrl+c SkipActions)."""
        sel = ""
        try:
            sel = self.query_one("#prompt", PromptArea).selected_text
        except Exception:  # noqa: BLE001 — selection is best-effort
            pass
        if not sel:
            try:
                sel = self.screen.get_selected_text() or ""
            except Exception:   # noqa: BLE001
                sel = ""
        if sel:
            self._copy_text(sel)
        else:
            self._status("nothing selected — drag to select, then Ctrl+C  ·  "
                         "Ctrl+Y = last reply  ·  Ctrl+Q = quit")

    def action_copy_reply(self) -> None:
        if not self._last_reply:
            self._status("nothing to copy yet")
            return
        self._copy_text(self._last_reply)
        self._status("📋 copied Engram's last reply")

    async def action_interrupt(self) -> None:
        """ESC while a reply streams → stop Engram now. We send the SDK interrupt; the
        in-flight turn then ends and its worker's `finally` restores the prompt and
        leaves the partial reply on screen. No-op when idle. (Graceful by design — no
        worker cancel — so cleanup always runs and the warm client stays good for the
        next turn.)"""
        if not self._busy:
            return
        self._status("✋ stopping…")
        try:
            await self.driver.interrupt()
        except Exception:  # noqa: BLE001 — never let stop() itself crash the home
            pass

    # ---- homegrown slash-command dropdown (type "/" → ↑/↓ menu, like Claude Code) ----
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._refresh_menu(event.text_area.text)

    def _menu_items(self, text: str) -> list[tuple[str, str]]:
        """(insert_value, display_label) pairs for the current prompt text — or []
        to dismiss. Shapes: command names, /effort levels, /agent names, or nothing."""
        if not text.startswith("/") or "\n" in text:
            return []
        # /effort <level> — offer the reasoning levels, filtered by what's typed.
        m = re.match(r"^/effort\s+(.*)$", text)
        if m:
            partial = m.group(1).strip()
            return [(f"/effort {lvl}", f"/effort {lvl}")
                    for lvl in EFFORT_LEVELS if lvl.startswith(partial)]
        # /agent <name> — offer the sub-agent names while typing the NAME (one token
        # after /agent); once a task follows (another space) the regex stops matching,
        # so the menu clears and won't cover the task. The trailing space in the value
        # marks "name completed, awaiting task" (see _choose_command).
        m = re.match(r"^/agent\s+(\S*)$", text)
        if m:
            partial = m.group(1).lower()
            return [(f"/agent {name} ", f"/agent {name}")
                    for name in SUBAGENTS if name.lower().startswith(partial)]
        # /model <name> — offer the known models, filtered by what's typed. Free-form still
        # works: an unmatched string just submits and is passed straight to the CLI.
        m = re.match(r"^/model\s+(\S*)$", text)
        if m:
            partial = m.group(1).lower()
            return [(f"/model {name}", f"/model {name}   {desc}")
                    for name, desc in MODELS if name.lower().startswith(partial)]
        # Any other "/cmd <arg>" (free text after a non-completing command) — stop.
        if re.match(r"^/\S+\s", text):
            return []
        # Typing a command name — filter the command list by prefix.
        token = text.strip()
        return [(cmd, f"{cmd}   {desc}") for cmd, desc in SLASH_CMDS if cmd.startswith(token)]

    def _refresh_menu(self, text: str) -> None:
        if self._interact_open:            # a card owns the prompt — don't pop the slash menu
            self._hide_menu()
            return
        menu = self.query_one("#cmdmenu", OptionList)
        items = self._menu_items(text)
        if not items:
            self._hide_menu()
            return
        menu.clear_options()
        menu.add_options([Option(label, id=value) for value, label in items])
        menu.highlighted = 0
        menu.display = True
        self._menu_open = True

    def _hide_menu(self) -> None:
        if self._menu_open:
            self.query_one("#cmdmenu", OptionList).display = False
            self._menu_open = False

    def _menu_move(self, delta: int) -> None:
        menu = self.query_one("#cmdmenu", OptionList)
        n = menu.option_count
        if n:
            menu.highlighted = ((menu.highlighted or 0) + delta) % n

    def _accept_menu(self) -> None:
        menu = self.query_one("#cmdmenu", OptionList)
        if menu.highlighted is None:
            return
        opt = menu.get_option_at_index(menu.highlighted)
        if opt.id:
            self._choose_command(opt.id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if (self._interact is not None and event.option_list is self._interact.get("list")
                and event.option.id):                    # mouse-click on an interaction choice
            event.stop()
            self._resolve_interact({"kind": "option", "id": event.option.id})
            return
        if event.option_list.id == "cmdmenu" and event.option.id:
            event.stop()
            self._choose_command(event.option.id)

    def _choose_command(self, value: str) -> None:
        """Act on a selected menu entry. Arg-taking command names (/effort, /model,
        /agent) complete the text and wait (re-opening the menu for /effort's levels
        and /agent's names); a chosen /agent NAME also completes-and-waits for the
        task; every other entry — arg-less commands, /effort levels — submits."""
        prompt = self.query_one("#prompt", PromptArea)
        if value in ARG_CMDS:                      # bare "/effort" / "/model" / "/agent"
            prompt.load_text(value + " ")
            prompt.move_cursor(prompt.document.end)
            prompt.focus()
            self._refresh_menu(prompt.text)        # /effort,/agent,/model → show options
            return
        # Selected a sub-agent NAME ("/agent Explore ") — complete it and wait for the
        # task; never submit a task-less /agent.
        if re.match(r"^/agent\s+\S+\s*$", value):
            prompt.load_text(value if value.endswith(" ") else value + " ")
            prompt.move_cursor(prompt.document.end)
            prompt.focus()
            self._hide_menu()
            return
        self._hide_menu()
        prompt.load_text("")
        self.post_message(PromptArea.Submitted(value))

    async def on_unmount(self) -> None:
        if self._perception is not None:
            try:
                self._perception.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._fleet is not None:
            # Members disconnect + release their folder locks; each member's own
            # evict_on_shutdown isn't fired here — their buffers are covered by
            # the nightly sweep, same as a terminal that closed without /end.
            try:
                await self._fleet.shutdown()
            except Exception:  # noqa: BLE001 — teardown must never wedge quit
                pass
        try:
            await self.driver.disconnect()
        except Exception:  # noqa: BLE001
            pass
        # Terminal session-end curation seam (the SDK has no SessionEnd hook).
        # PROVISIONAL full-flush of the LiveBuffer (Brick 3) — folds the whole
        # un-evicted tail into LTM, advancing the watermark; the nightly confirm
        # pass reconciles late reversals. Falls back to the transcript --session
        # pass when there's no buffer. Never marks the session fully-curated.
        try:
            evict = getattr(self.driver, "evict_on_shutdown", None)
            if evict is not None:
                evict()
        except Exception:  # noqa: BLE001 — teardown must never fail on curation
            pass

    # ---- perception (opt-in: ENGRAM_PERCEIVE=1) — camera senses wired into THIS chat ----
    def _start_perception(self) -> None:
        """Boot the PerceptionBridge so Engram's camera senses (face-ID + the eye) feed this
        session's live HUD and its per-prompt identity marker — vision only, no audio.
        Booted on a worker thread — opening the camera must not freeze the UI — with a lazy
        import so a normal launch never pulls cv2/onnxruntime."""
        def _boot() -> None:
            try:
                sys.path.insert(0, os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "perceive"))
                from bridge import PerceptionBridge
                pb = PerceptionBridge()
                pb.start()
                self._perception = pb
                self.call_from_thread(self._perception_ready)
            except Exception as exc:  # noqa: BLE001 — perception must never kill the TUI
                self.call_from_thread(self._status, f"perception off — {type(exc).__name__}")
        threading.Thread(target=_boot, name="perception-boot", daemon=True).start()

    def _perception_ready(self) -> None:
        """On the app thread once the bridge is up: reveal the HUD + start polling it."""
        self._status(self._perception.status)
        self.query_one("#perception", Static).display = True
        self.set_interval(0.3, self._render_perception)

    def _render_perception(self) -> None:
        """The live senses card above the prompt — who's in frame, the engagement gate, and
        the eye's latest scene reading. Vision only (sound disabled); polled a few times/sec,
        cheap (attribute reads)."""
        pb = self._perception
        if pb is None:
            return
        snap = pb.snapshot()
        card = self.query_one("#perception", Static)
        if not snap["ok"]:
            card.update(f"[red]● senses off — {escape(str(snap['error']))}[/red]")
            return
        if snap["faces"]:
            who = "  ".join(
                (f"[green]{escape(n)}[/green] {s:.2f}" if n == self._perceive_target(pb)
                 else f"[yellow]{escape(n)}[/yellow] {s:.2f}") for n, s in snap["faces"])
        else:
            who = "[dim]nobody in frame[/dim]"
        state = snap["state"]
        scol = {"engaged": "green", "passive": "yellow", "idle": "dim"}.get(state, "white")
        scene = snap.get("scene")
        if scene:
            s = scene if len(scene) <= 72 else scene[:71] + "…"
            eye_line = f"\n[dim]eye:[/dim] {escape(s)}"
        else:
            eye_line = ""
        card.update(f"👁 {who}   ·   [{scol}]{state}[/{scol}]{eye_line}")

    @staticmethod
    def _perceive_target(pb) -> str:
        return getattr(pb, "target", None) or os.environ.get("ENGRAM_USER") or "operator"

    @staticmethod
    def _identity_note(target: str, snap: "dict | None") -> str:
        """Pure: build the per-prompt identity marker from a perception snapshot (or '').

        Three cases, by who the camera sees at the keyboard: the target (confirm), some
        OTHER/unknown face (warn — don't assume it's the target, withhold private context),
        or nobody (unverified — the target may be off-camera, so proceed but hold consequential
        actions). Informational, never a hard block: a false-negative must not lock the target
        out. Kept pure so it's unit-testable without a camera/TUI."""
        if not snap or not snap.get("ok"):
            return ""
        faces = snap.get("faces") or []
        present = set(snap.get("present") or [])
        tgt_cos = max((s for n, s in faces if n == target), default=0.0)
        if target in present or tgt_cos > 0:
            cos = f" (face match {tgt_cos:.2f})" if tgt_cos > 0 else ""
            return (f"[identity] Camera confirms the operator is {target}{cos}. "
                    f"Proceed normally; no need to acknowledge this line.\n\n")
        others = [(n or "unknown", s) for n, s in faces if n != target]
        if others:
            who = ", ".join(f"{n} {s:.2f}" for n, s in others)
            return (f"[identity] ⚠ The person at the keyboard is NOT {target} — the "
                    f"camera sees: {who}. Do not assume you are speaking with {target}: "
                    f"withhold their private context, and verify who you are talking "
                    f"to before any private or consequential action.\n\n")
        return (f"[identity] No face is visible at the keyboard, so this message's sender "
                f"is unverified ({target} may be off-camera). Keep helping, but don't take "
                f"irreversible or private actions on unverified identity alone. No need to "
                f"acknowledge this line.\n\n")

    def _ultracode_marker(self) -> str:
        """The standing ultracode opt-in, prepended to a typed turn while /ultracode is on
        (empty otherwise). Model-only, like the identity marker; typed turns only."""
        return ULTRACODE_REMINDER if self._ultracode else ""

    def _working_memory_marker(self) -> str:
        """The Brick-3 tier-2 block: the conversation's recent raw turns + hot
        notes, re-derived from THIS driver's LiveBuffer (never the SDK's drifting
        self-summary). Model-only, prepended to `prompt`; the buffer logs the raw
        `prompt`, so this derived block never feeds back into its own source.
        ENGRAM_WORKING_MEMORY=0 turns it off; fail-open '' otherwise (like the
        identity marker), so a memory hiccup never touches the turn."""
        if os.environ.get("ENGRAM_WORKING_MEMORY", "1") == "0":
            return ""
        buf = getattr(self.driver, "_buffer", None)
        if buf is None:
            return ""
        try:
            from working_set import build_working_memory
            block = build_working_memory(buf, getattr(self.driver, "cwd", ENGRAM_CWD))
        except Exception:  # noqa: BLE001 — passenger, never the driver
            return ""
        return (block + "\n\n") if block else ""

    def _identity_marker(self) -> str:
        """Gather the live snapshot and build the marker; '' when perception is off, so
        normal launches (ENGRAM_PERCEIVE unset) and the Telegram bridge (no camera, single
        access) are untouched. Never raises — identity must not break a turn."""
        pb = self._perception
        if pb is None:
            return ""
        try:
            snap = pb.snapshot()
        except Exception:   # noqa: BLE001 — identity is advisory; a bad read can't kill the turn
            return ""
        return self._identity_note(self._perceive_target(pb), snap)

    # ---- submit ----
    async def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        text = event.value.strip()
        self.query_one("#prompt", PromptArea).load_text("")
        if not text and not self._attachments:
            return
        # Always safe, even mid-reply.
        if text in ("/exit", "/quit", "/q"):
            self.exit()
            return
        # A plan/question card is open → this submit IS the free-text answer / plan feedback
        # (the "add my own / just chat" path), not a new turn. Resolve the card and stop.
        if self._interact_open and self._interact is not None:
            self._resolve_interact({"kind": "text", "text": text})
            return
        if text == "/status":
            self._status(self._status_line())
            return
        if text == "/copy":            # read-only, safe mid-reply (copies the PREVIOUS reply)
            self.action_copy_reply()
            return
        # /mode — toggle, or set explicitly. Like shift+tab it recycles the client when
        # idle (the new mode governs the next turn) and ARMS mid-reply instead of
        # disrupting it — so unlike the other recycling cmds it's safe mid-reply (not a
        # STATE_CMD).
        if text == "/mode" or text.startswith("/mode "):
            arg = text[len("/mode"):].strip().lower()
            if arg in ("", "toggle"):
                await self.action_cycle_mode()
            elif arg in ("plan", "p"):
                await self._set_mode(PLAN_MODE)
            elif arg in ("regular", "reg", "normal", "default", "run", "r"):
                await self._set_mode(REGULAR_MODE)
            else:
                self._status("usage: /mode [plan|regular]   (bare /mode toggles)")
            return
        # /ultracode — standing opt-in to multi-agent workflow orchestration. Safe mid-reply
        # (only flips a flag + re-renders; governs the NEXT turn's prompt), so it sits with
        # the other always-safe toggles, not in STATE_CMDS.
        if text == "/ultracode" or text.startswith("/ultracode "):
            arg = text[len("/ultracode"):].strip().lower()
            if arg in ("", "toggle"):
                self._ultracode = not self._ultracode
            elif arg in ("on", "yes", "1"):
                self._ultracode = True
            elif arg in ("off", "no", "0"):
                self._ultracode = False
            else:
                self._status("usage: /ultracode [on|off]   (bare /ultracode toggles)")
                return
            self._render_header()
            self._status("ultracode ⚡ on — Engram orchestrates substantive work with workflows"
                         if self._ultracode else "ultracode off")
            return
        # State-changing slash commands can't run while a turn is in flight (they'd
        # disrupt the warm client) — block them, don't queue; otherwise dispatch.
        # The client-RECYCLING ones are also blocked while background agents are out:
        # dropping the warm client would kill them mid-run.
        if any(text == c or text.startswith(c + " ") for c in STATE_CMDS):
            cmd = text.split()[0]
            if self._busy:
                self._status(f"busy — {cmd} runs once the reply finishes")
            elif (cmd in ("/new", "/effort", "/model", "/rewind", "/sessions", "/fork")
                    and getattr(self.driver, "has_background_tasks", False)):
                self._status(f"🛰 background agents still working — {cmd} would drop "
                             "them; try again when they finish")
            else:
                await self._handle_command(text)
            return
        # /agent <name> <task> — a sub-agent delegation. Validate up-front, then run it
        # through the SAME queue/stream machinery as a normal message (no attachments).
        # The Task-tool rewrite happens at dispatch, so the convo still shows the tidy
        # "/agent …" line, not the forcing boilerplate.
        if text == "/agent" or text.startswith("/agent "):
            if self._parse_agent(text) is None:
                self._status("usage: /agent <name> <task>   "
                             "e.g. /agent Explore find every caller of AgentSDKDriver")
                return
            if self._busy:
                self._queue.append((text, []))
                self._render_queue()
                self._status("⏳ queued — the sub-agent runs when the reply finishes")
            else:
                await self._dispatch(text, [])
            return
        # A normal message. Bind any pending attachments to THIS message and clear.
        attachments = self._attachments
        self._attachments = []
        self._render_chips()
        # Type-ahead, like Claude Code: if a reply is still streaming, queue the
        # message and pick it up the moment the turn ends — never drop it.
        if self._busy:
            self._queue.append((text, attachments))
            self._render_queue()
            self._status("⏳ queued — Engram reads it when the current reply finishes")
            return
        await self._dispatch(text, attachments)

    async def _handle_command(self, text: str) -> None:
        """A state-changing / warm-client slash command (/new · /effort · /model ·
        /paste · /context)."""
        if text == "/new":
            await self._reset_thread()
        elif text.startswith("/effort"):
            parts = text.split(maxsplit=1)
            lvl = parts[1].strip() if len(parts) > 1 else ""
            if lvl in EFFORT_LEVELS:
                await self._apply_effort(lvl)
            else:
                self._status("usage: /effort " + "|".join(EFFORT_LEVELS)
                             + f"   (now: {getattr(self.driver, 'effort', '?')})")
        elif text.startswith("/model"):
            parts = text.split(maxsplit=1)
            name = parts[1].strip() if len(parts) > 1 else ""
            if name:
                await self.driver.set_model(name)
                self._render_header()
                self._status(f"model → {name}  (applies to your next message)")
            else:
                self._status(f"usage: /model <name>   (now: {getattr(self.driver, 'model', '?')})")
        elif text == "/paste":
            await self.action_paste_image()
        elif text == "/context":
            await self._show_context()
        elif text == "/rewind":
            await self._show_rewind()
        elif text == "/sessions":
            await self._show_sessions()
        elif text == "/fork":
            await self._do_fork()
        elif text == "/export":
            await self._do_export()
        elif text == "/workflows":
            await self._show_workflows()
        elif text.startswith("/fleet"):
            await self._handle_fleet(text[len("/fleet"):].strip())

    async def _show_rewind(self) -> None:
        """List this session's file checkpoints (one per typed prompt, newest first)
        and restore the working tree to just before the picked one. Files only —
        the conversation is untouched (the SDK's rewind semantics) — so a model-only
        note is armed for the next turn telling Engram the tree moved under it."""
        if self._busy:
            self._status("busy — /rewind runs once the reply finishes")
            return
        cps = list(getattr(self.driver, "list_checkpoints", lambda: [])())
        if not cps:
            self._status("no file checkpoints yet this session — they start with your next message")
            return
        recent = list(reversed(cps[-9:]))          # newest first, capped for the card
        await self._add(Static("[b]▌ Rewind files to just before…[/b]"))
        choices = [(c["uuid"], f"{i}. ❯ {c['preview']}{_age(c.get('ts'))}")
                   for i, c in enumerate(recent, 1)]
        choice = await self._await_choice(
            choices, hint="↑/↓ + Enter to restore files  ·  Esc to cancel")
        if choice.get("kind") != "option":
            await self._add(Static("[dim]rewind cancelled[/dim]"))
            self._status("ready")
            return
        picked = next(c for c in recent if c["uuid"] == choice["id"])
        try:
            await self.driver.rewind_to(picked["uuid"])
        except Exception as exc:  # noqa: BLE001 — surface, never crash the home
            await self._add(Static(
                f"[red]rewind failed — {type(exc).__name__}: {escape(str(exc))}[/red]"))
            tail = getattr(self.driver, "stderr_tail", "")
            if tail:
                await self._add(Static(f"[red dim]{escape(tail)}[/red dim]"))
            self._status("ready")
            return
        await self._add(Static(
            f"[#86EFAC]⏪ files restored to just before: ❯ {escape(picked['preview'])}[/]"))
        self._rewind_note = (
            "[system] The working tree was just REWOUND to its state before the prompt "
            f"\"{picked['preview']}\" — edits made after that point are undone on disk; "
            "re-read any file you rely on before acting.\n\n")
        self._status("⏪ rewound — Engram is told on your next message")

    async def _show_sessions(self) -> None:
        """Pick one of this folder's recent sessions and resume it — per-cwd,
        like Claude Code's session picker."""
        if self._busy:
            self._status("busy — /sessions runs once the reply finishes")
            return
        self._status("listing sessions…")
        sessions = list(getattr(self.driver, "list_sessions", lambda: [])())
        if not sessions:
            self._status("no sessions found for this folder")
            return
        await self._add(Static("[b]▌ Resume a session in this folder…[/b]"))
        choices = []
        for i, s in enumerate(sessions, 1):
            mark = "   · current" if s.get("current") else ""
            choices.append((s["sid"], f"{i}. ❯ {s['preview']}{_age(s.get('mtime'))}{mark}"))
        choice = await self._await_choice(
            choices, hint="↑/↓ + Enter to resume  ·  Esc to cancel")
        if choice.get("kind") != "option":
            await self._add(Static("[dim]cancelled[/dim]"))
            self._status("ready")
            return
        await self.driver.resume_session(choice["id"])
        await self._add(Static(f"[#86EFAC]↺ resumed session {escape(choice['id'][:8])} — "
                               "your next message continues that thread[/]"))
        self._status("↺ resumed — next message continues that thread")

    async def _do_fork(self) -> None:
        """Branch the conversation: next message starts a NEW session id resumed
        from here; the original thread stays untouched (see /sessions)."""
        if self._busy:
            self._status("busy — /fork runs once the reply finishes")
            return
        if not getattr(self.driver, "session_id", None):
            self._status("nothing to fork yet — this thread has no session")
            return
        await self.driver.fork()
        await self._add(Static("[#C4B5FD]⑂ forked — your next message starts a new "
                               "branch; the original thread is kept (see /sessions)[/]"))
        self._status("⑂ forked — next message begins the branch")

    async def _do_export(self) -> None:
        """Write this session's conversation (denoised) to a markdown file in the
        launch folder."""
        sid = getattr(self.driver, "session_id", None)
        cwd = Path(getattr(self.driver, "cwd", ENGRAM_CWD))
        if not sid:
            self._status("nothing to export yet — this thread has no session")
            return
        try:
            from recall.transcripts import (
                iter_exchanges,
                project_transcript_dir,
                session_transcript_path,
            )
            path = session_transcript_path(project_transcript_dir(cwd), sid)
            exchanges = list(iter_exchanges(path, None))
        except Exception as exc:  # noqa: BLE001 — surface, never crash the home
            self._status(f"export failed: {type(exc).__name__}")
            return
        if not exchanges:
            self._status("nothing to export yet — the transcript is still empty")
            return
        stamp = time.strftime("%Y%m%d-%H%M")
        out = cwd / f"engram-session-{sid[:8]}-{stamp}.md"
        lines = [f"# Engram session {sid[:8]} — exported {time.strftime('%Y-%m-%d %H:%M')}", ""]
        for ex in exchanges:
            who = "**❯ you**" if ex.role == "user" else "**Engram**"
            lines += [f"{who}:", "", ex.text.strip(), ""]
        try:
            out.write_text("\n".join(lines))
        except OSError as exc:
            self._status(f"export failed: {exc}")
            return
        await self._add(Static(f"[#86EFAC]⇩ exported → {escape(str(out))}[/]"))
        self._status("exported")

    async def _show_context(self) -> None:
        """Render the context-window breakdown into the scrollback. Gated on busy so
        the command-palette path can't poke the warm client mid-stream (the typed
        /context is already gated via STATE_CMDS)."""
        if self._busy:
            self._status("busy — /context runs once the reply finishes")
            return
        self._status("reading context…")
        try:
            usage = await self.driver.get_context_usage()
        except Exception as exc:  # noqa: BLE001 — older CLI / control unsupported
            self._status(f"context unavailable: {type(exc).__name__}")
            tail = getattr(self.driver, "stderr_tail", "")
            if tail:
                await self._add(Static(f"[red dim]{escape(tail)}[/red dim]"))
            return
        await self._add(Markdown(render_context_md(usage)))
        self._status("ready")

    def _parse_agent(self, text: str) -> "tuple[str, str] | None":
        """Parse '/agent <name> <task>' → (name, task); None if not well-formed.
        Dispatch runs it as an isolated synchronous sub-query (driver.run_subagent),
        not the CLI's async Agent tool, so the result returns within the turn."""
        m = re.match(r"^/agent\s+(\S+)\s+(.+)$", text.strip(), re.DOTALL)
        if not m:
            return None
        name, task = m.group(1), m.group(2).strip()
        return (name, task) if task else None

    async def _dispatch(self, text: str, attachments: list) -> None:
        """Show the operator turn and run it. /agent runs the named sub-agent as an
        isolated, synchronous sub-query (its result streams back in-turn); attachments
        become Read-tool file refs (images render visually), like the Telegram bridge."""
        agent = self._parse_agent(text)
        if agent is not None:
            await self._add(UserMsg(f"❯ {escape(text)}"))
            self._run_turn("", agent=agent)
            return
        if attachments:
            refs = "\n".join(f"[attached file — open with your Read tool] {p}"
                             for p in attachments)
            body = text or "I've attached the file(s) above — take a look."
            prompt = f"{refs}\n\n{body}"
            names = ", ".join(p.name for p in attachments)
            shown = (f"{text}  " if text else "") + f"[📎 {names}]"
        else:
            prompt = shown = text
        await self._add(UserMsg(f"❯ {escape(shown)}"))
        # Build the model-ONLY prepend (never logged as the operator's raw text —
        # the LiveBuffer logs `prompt`, the driver sends prepend+prompt): the
        # working-memory block (standing conversation context, re-grounded from
        # the buffer) → the one-shot rewind note (if files were just restored) →
        # the standing ultracode reminder (if /ultracode on) → the live face-ID
        # verdict (if ENGRAM_PERCEIVE on). Order = standing memory → tree state →
        # mode → who. All model-only, none in `shown`.
        note, self._rewind_note = self._rewind_note, ""
        prepend = (self._working_memory_marker() + note
                   + self._ultracode_marker() + self._identity_marker())
        self._run_turn(prompt, prepend=prepend)

    def _render_tasks(self) -> None:
        """Refresh the sticky todo/sub-agent panel above the prompt."""
        try:
            self.query_one("#tasks", Static).update(
                escape(render_tasks_line(self._todos, self._tasks_snapshot)))
        except Exception:  # noqa: BLE001 — panel not mounted (teardown); skip
            pass

    async def _show_workflows(self) -> None:
        """Expand this session's workflow runs — each phase with its agents'
        states, from the latest progress snapshot (core.workflow_snapshot)."""
        runs = [t for t in self._tasks_snapshot if t.get("workflow")]
        if not runs:
            await self._add(Static(
                "[dim]no workflow runs this session — /ultracode on (or say "
                "“use a workflow”) and give me something big[/dim]"))
            return
        lines = ["**Workflow runs this session**"]
        for t in runs:
            mark = {"running": "⏳", "completed": "✓"}.get(
                t.get("status"), f"✗ {t.get('status', '?')}")
            head = (f"- **{str(t.get('name', '?')).removeprefix('⚙ ')}** {mark}"
                    f" — {t.get('desc', '')}")
            if t.get("tokens"):
                head += f" · {int(t['tokens']) // 1000}k tok"
            lines.append(head)
            for p in (t.get("wf") or {}).get("phases", []):
                done = sum(1 for a in p["agents"] if a["state"] == "done")
                lines.append(f"  - {p['title']} ({done}/{len(p['agents'])})")
                for a in p["agents"][:10]:
                    m = {"done": "✓", "failed": "✗"}.get(a["state"], "⏳")
                    row = f"    - {m} {a['label']}"
                    if a.get("model"):
                        row += f" · {a['model']}"
                    lines.append(row)
                if len(p["agents"]) > 10:
                    lines.append(f"    - … +{len(p['agents']) - 10} more")
        await self._add(Markdown("\n".join(lines)))

    # ---- fleet: parallel Engram sessions across repos (/fleet) ----------------

    def _get_fleet(self):
        if self._fleet is None:
            from fleet import Fleet
            self._fleet = Fleet(on_change=self._render_fleet)
        return self._fleet

    def _render_fleet(self) -> None:
        """Refresh the ⚑ fleet strip above the prompt (fail-soft on teardown)."""
        try:
            from fleet import render_fleet_line
            rows = self._fleet.rows() if self._fleet else []
            self.query_one("#fleet", Static).update(
                escape(render_fleet_line(rows)))
        except Exception:  # noqa: BLE001 — panel not mounted (teardown); skip
            pass

    async def _handle_fleet(self, args: str) -> None:
        """/fleet — spawn/list/view/msg/kill parallel Engram sessions:
        ``/fleet <path> [task…]`` spawn · ``/fleet`` list · ``/fleet view <name>``
        peek · ``/fleet msg <name> <text>`` steer · ``/fleet kill <name>``."""
        fleet = self._get_fleet()
        parts = args.split(maxsplit=1)
        sub = parts[0] if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        if not sub:
            rows = fleet.rows()
            if not rows:
                await self._add(Static(
                    "[dim]no fleet members — /fleet <path> [task] spawns a "
                    "parallel Engram session in that repo (its own session, memory, and "
                    "recall injection)[/dim]"))
                return
            lines = ["**Fleet**"]
            for r in rows:
                lines.append(f"- **{r['name']}** ({r['dir']}) — {r['status']}"
                             + (f", {r['pending']} queued" if r['pending'] else "")
                             + (f" · {r['last']}" if r.get("last") else "")
                             + (f" · {r['error']}" if r.get("error") else ""))
            await self._add(Markdown("\n".join(lines)))
        elif sub == "kill" and rest:
            self._status(await fleet.kill(rest.split()[0]))
        elif sub == "view" and rest:
            name = rest.split()[0]
            m = fleet.members.get(name)
            if m is None:
                self._status(f"no fleet member '{name}'")
                return
            await self._add(Static(f"[b]▌ fleet · {escape(name)} — {m.status} "
                                   f"({escape(str(m.cwd))})[/b]"))
            await self._add(Markdown(m.tail() or "*(nothing yet)*",
                                     classes="plancard"))
        elif sub == "msg" and rest:
            bits = rest.split(maxsplit=1)
            if len(bits) < 2:
                self._status("usage: /fleet msg <name> <message>")
                return
            self._status(fleet.send(bits[0], bits[1]))
        else:
            # `/fleet <path> [task…]` — spawn. The path is the first token.
            member, note = fleet.spawn(sub, task=rest)
            if member is None:
                self._status(note)
            else:
                await self._add(Static(f"[dim]{escape(note)}[/dim]"))
                self._render_fleet()

    def _render_queue(self) -> None:
        """The pending type-ahead strip above the prompt — previews of queued msgs."""
        widget = self.query_one("#queued", Static)
        if not self._queue:
            widget.update("")
            return
        def preview(item) -> str:
            t, atts = item
            s = " ".join(t.split())
            if len(s) > 46:
                s = s[:45] + "…"
            if atts:
                s = (s + " " if s else "") + f"📎{len(atts)}"
            return s or "(attachment)"
        widget.update("   ".join("⏳ " + escape(preview(it)) for it in self._queue))

    async def _drain_queue(self) -> None:
        """Send the oldest queued message, once the turn that scheduled this drain
        has fully unwound (we're invoked via call_after_refresh, so _run_turn's
        exclusive worker is already gone and re-launching it is safe)."""
        if self._busy or not self._queue:
            return
        text, attachments = self._queue.pop(0)
        self._render_queue()
        await self._dispatch(text, attachments)

    # ---- interactive tools: plan approval + option questions (driver.on_interaction) ----
    async def _handle_interaction(self, req: dict) -> dict:
        """Driver seam (core.AgentSDKDriver._can_use_tool): render a plan or an option
        question inline and BLOCK the turn until you decide, returning the verdict the
        driver maps onto the SDK wire result. Runs on the app's event loop (the SDK spawns
        the permission callback there), so it mounts widgets and awaits a Future the UI
        resolves. Never raises — a broken card must not wedge the turn; it degrades to a
        sensible default (approve nothing / no preference). Serialized: the SDK runs each
        permission callback as its own task, so parallel AskUserQuestion calls arrive
        CONCURRENTLY — without the lock they'd all mount at once and clobber the single
        _interact slot (stacked cards, only the last answerable, the rest wedged)."""
        try:
            async with self._interact_lock:
                await self._break_stream()  # finalize pre-card text so the card lands below it
                if req.get("kind") == "plan":
                    return await self._interact_plan(req.get("plan") or "")
                return await self._interact_question(req.get("questions") or [])
        except Exception:  # noqa: BLE001 — identity of the failure doesn't matter; don't wedge
            return {"approved": False, "message": "(the interaction UI failed; use your judgment)"}
        finally:
            self._status("✦ thinking…")

    async def _interact_plan(self, plan_md: str) -> dict:
        """Render the proposed plan as real Markdown (the rendering fix) and offer
        approve / keep-planning. Approve → leave plan mode and let the turn implement;
        typed feedback → keep planning with that steer."""
        await self._add(Static("[b]▌ Engram proposes a plan[/b]"))
        await self._add(Markdown(plan_md or "*(empty plan)*", classes="plancard"))
        choice = await self._await_choice(
            [("approve", "✅  Approve — leave plan mode and implement this now"),
             ("keep",    "✎  Keep planning — refine it (or type feedback below)")],
            hint="↑/↓ + Enter to choose  ·  or type feedback + Enter to keep planning")
        if choice.get("kind") == "option" and choice.get("id") == "approve":
            # Approving lands back in the PRE-plan mode (the driver restores it live once
            # this handler returns); mirror the same target here for the indicator and drop
            # any armed pending mode so nothing re-enters plan.
            self.driver.permission_mode = getattr(
                self.driver, "plan_restore_target", REGULAR_MODE)
            self._pending_mode = None
            self._render_header()
            await self._add(Static("[#86EFAC]✓ approved — leaving plan mode, implementing…[/]"))
            return {"approved": True}
        feedback = choice.get("text", "").strip() if choice.get("kind") == "text" else ""
        await self._add(Static(
            f"[#C4B5FD]✎ keep planning{': ' + escape(feedback) if feedback else ''}[/]"))
        return {"approved": False,
                "message": feedback or "Keep planning — don't implement yet; refine the plan."}

    async def _interact_question(self, questions: list) -> dict:
        """Ask each question in turn (usually one), collecting a picked option or a typed
        answer, then hand the combined answer to the model. 'Add my own' and 'just chat' are
        the free-text path — you type instead of picking."""
        records: list[str] = []
        for q in (questions or [{}]):
            header = (q.get("header") or "").strip()
            qtext = (q.get("question") or "").strip()
            opts = q.get("options") or []
            title = (f"**{escape(header)}** — {escape(qtext)}" if header
                     else f"**{escape(qtext) or 'Engram asks:'}**")
            await self._add(Markdown(title, classes="plancard"))
            choices = []
            for i, o in enumerate(opts):
                label = str(o.get("label", "")).strip() or f"option {i + 1}"
                desc = str(o.get("description", "")).strip()
                choices.append((f"opt{i}", label + (f"   —   {desc}" if desc else "")))
            choice = await self._await_choice(
                choices, hint="↑/↓ + Enter to pick  ·  or type your own answer / chat + Enter")
            if choice.get("kind") == "option" and str(choice.get("id", "")).startswith("opt"):
                idx = int(choice["id"][3:])
                ans = str(opts[idx].get("label", "")).strip() or f"option {idx + 1}"
            else:
                ans = choice.get("text", "").strip() or "(no preference — you decide)"
            records.append(f"{header or qtext or 'answer'}: {ans}")
            await self._add(Static(f"[#67E8F9]❯ {escape(ans)}[/]"))
        return {"message": "The user answered your question(s):\n" + "\n".join(records)}

    async def _await_choice(self, choices: list, hint: str = "") -> dict:
        """Mount an inline chooser and await your decision: a picked option
        ({'kind':'option','id':...}), free text ({'kind':'text','text':...}), or a cancel
        ({'kind':'cancel'}). The prompt keeps focus so typing works instead of picking."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        ol = OptionList(classes="interact")
        await self._add(ol)
        if choices:
            ol.add_options([Option(label, id=cid) for cid, label in choices])
            ol.highlighted = 0
        if hint:
            await self._add(Static(hint, classes="interact-hint"))
            self._status(hint)
        # Keep the prompt focused so ↑/↓/Enter route through PromptArea._on_key AND typing
        # a free-text answer works — the chooser is driven from there, never focused itself.
        try:
            self.query_one("#prompt", PromptArea).focus()
        except Exception:  # noqa: BLE001
            pass
        self._interact = {"future": fut, "list": ol}
        self._interact_open = True
        try:
            return await fut
        finally:
            self._interact_open = False
            self._interact = None
            try:
                ol.disabled = True             # freeze it in the transcript as a record
            except Exception:  # noqa: BLE001
                pass

    def _resolve_interact(self, result: dict) -> None:
        st = self._interact
        if st is not None and not st["future"].done():
            st["future"].set_result(result)

    def _interact_move(self, delta: int) -> None:
        st = self._interact
        if st is None:
            return
        ol = st["list"]
        if ol.option_count:
            ol.highlighted = ((ol.highlighted or 0) + delta) % ol.option_count

    def _interact_accept_highlight(self) -> None:
        st = self._interact
        if st is None:
            return
        ol = st["list"]
        if not ol.option_count or ol.highlighted is None:
            return                             # no options → the operator must type an answer
        opt = ol.get_option_at_index(ol.highlighted)
        if opt.id:
            self._resolve_interact({"kind": "option", "id": opt.id})

    def _interact_cancel(self) -> None:
        self._resolve_interact({"kind": "cancel"})

    # ---- per-turn recall provenance (memory made visible) ----
    async def _add_recall_line(self, line: str | None) -> None:
        """One dim line under the prompt showing which memory notes fed this turn —
        so it's legible when Engram is grounded in a real note vs winging it, and a
        zero/silent turn is the retrieval miss-detector."""
        self._recall_shown = True
        await self._add(Static(render_recall_line(line), classes="recall-line"))

    # ---- streaming Markdown, segmentable so a mid-turn card renders in order ----
    async def _ensure_stream(self) -> None:
        """Open a fresh streaming Markdown block if none is active (lazy — so a turn that
        opens with a tool or a card doesn't leave an empty bubble above it)."""
        if self._cur_stream is None:
            md = Markdown()
            await self._add(md)
            self._cur_md = md
            self._cur_stream = Markdown.get_stream(md)
            self._cur_last = ""

    async def _break_stream(self) -> None:
        """Finalize the active streaming block so whatever mounts next — an interaction
        card, then more text — lands BELOW it in order rather than glued into it."""
        if self._cur_stream is not None:
            try:
                await self._cur_stream.stop()
            except Exception:  # noqa: BLE001 — stopping a spent stream must never crash a turn
                pass
            self._cur_stream = None
            self._cur_md = None
            self._cur_last = ""

    # ---- the turn (background worker; incremental Markdown streaming) ----
    @work(exclusive=True)
    async def _run_turn(self, text: str, agent: "tuple | None" = None,
                        prepend: str = "") -> None:
        self._busy = True
        self._cur_md = None
        self._cur_stream = None
        self._cur_last = ""
        # Sub-agent turns run their own CLI (no hook events here) — suppress the
        # provenance line for them rather than falsely reporting "silent".
        self._recall_shown = agent is not None
        acc: list[str] = []
        tools: list[str] = []
        wrote_any = False
        self._status(f"✦ {agent[0]} working…" if agent else "✦ thinking…")
        try:
            source = (self.driver.run_subagent(*agent) if agent
                      else self.driver.query(text, prepend=prepend))
            async for ev in source:
                if ev.kind == "recall":
                    if not self._recall_shown:      # first response wins (one hook wired)
                        await self._add_recall_line(ev.text)
                elif ev.kind == "text":
                    if not self._recall_shown:      # text arrived, no hook event ever did
                        await self._add_recall_line(None)
                    await self._ensure_stream()     # (re)open a block; a card may have broken it
                    sep = _seam(self._cur_last, ev.text)   # paragraph break so blocks don't glue
                    if sep:
                        acc.append(sep)
                        await self._cur_stream.write(sep)
                    acc.append(ev.text)
                    await self._cur_stream.write(ev.text)
                    self._cur_last = ev.text
                    wrote_any = True
                elif ev.kind == "tool":
                    if ev.text not in tools:
                        tools.append(ev.text)
                    self._status(f"⚙ {ev.text}…")
                elif ev.kind == "status":          # ephemeral (sub-agent progress)
                    self._status(f"⚙ {ev.text}…")
                elif ev.kind == "todos":
                    self._todos = (ev.data or {}).get("todos") or []
                    self._render_tasks()
                elif ev.kind == "task":
                    self._tasks_snapshot = (ev.data or {}).get("tasks") or []
                    self._render_tasks()
                self._scroll()
        except Exception as exc:  # noqa: BLE001 — surface, never crash the home
            await self._ensure_stream()
            await self._cur_stream.write(f"\n\n**error:** `{type(exc).__name__}: {exc}`")
            tail = getattr(self.driver, "stderr_tail", "")
            if tail:
                await self._add(Static(f"[red dim]{escape(tail)}[/red dim]"))
        finally:
            await self._break_stream()
            self._busy = False
            bg_live = getattr(self.driver, "has_background_tasks", False)
            # A mode armed via shift+tab mid-reply applies now, governing the next turn —
            # unless background agents are still out: applying recycles the warm client,
            # which would kill them. It stays armed and applies at the next quiet turn end.
            if self._pending_mode is not None and not bg_live:
                target, self._pending_mode = self._pending_mode, None
                try:
                    await self.driver.set_permission_mode(target)
                except Exception:  # noqa: BLE001 — never let a mode flip break the turn
                    pass
                self._render_header()
            self._last_reply = "".join(acc).strip()
            try:                                   # all best-effort: a quit/teardown mid-turn
                # Loud, once-per-rotation notice when the CLI silently drops to the
                # fallback (overload). Re-arms when it rotates back, so a later switch
                # announces again. The header marker (via _subtitle) stays up meanwhile.
                fb = getattr(self.driver, "active_fallback", None)
                if fb and not self._fallback_shown:
                    self._fallback_shown = True
                    await self._add(Static(
                        f"[yellow]⚠ primary model unavailable — this turn ran on the "
                        f"fallback ([b]{escape(str(fb))}[/b]). The conversation "
                        f"continues; it rotates back automatically when the primary "
                        f"recovers.[/yellow]"))
                elif not fb:
                    self._fallback_shown = False
                if not wrote_any and not self._last_reply:   # removes #convo before this runs
                    await self._add(Static("[dim]*(no text in reply)*[/dim]"))
                self._status("🛰 background agents working — results stream in here" if bg_live
                             else "ready" + (f"   ·   ⚙ {', '.join(tools)}" if tools else ""))
                if not bg_live:
                    # The turn is over: finished sub-agents live in the transcript,
                    # not the panel (todos persist — they're session state).
                    self._tasks_snapshot = []
                    self._render_tasks()
                self._scroll()
            except Exception:  # noqa: BLE001 — convo/status gone (app closing); nothing to show
                pass
            # Type-ahead: if messages were queued mid-reply, send the next one now.
            # Deferred to after-refresh so this (exclusive) worker fully exits first.
            if self._queue:
                self.call_after_refresh(self._drain_queue)
            elif bg_live:
                # Nothing queued but background agents are out: keep listening while
                # idle so their results PAINT when they land (prompt stays unlocked;
                # a typed turn cancels this exclusive-group worker and takes over).
                self.call_after_refresh(self._drain_bg)

    @work(exclusive=True)
    async def _drain_bg(self) -> None:
        """Idle listener: paints background sub-agent completions and the model's
        follow-up turns the moment they land, with the prompt UNLOCKED throughout.
        Same exclusive worker group as _run_turn, so typing cancels this instantly
        and the typed turn (reading the same stream) takes over seamlessly."""
        source = getattr(self.driver, "drain_background", None)
        if source is None:
            return
        self._cur_md = None
        self._cur_stream = None
        self._cur_last = ""
        try:
            async for ev in source():
                if ev.kind == "text":
                    await self._ensure_stream()
                    sep = _seam(self._cur_last, ev.text)
                    if sep:
                        await self._cur_stream.write(sep)
                    await self._cur_stream.write(ev.text)
                    self._cur_last = ev.text
                elif ev.kind in ("tool", "status"):
                    self._status(f"⚙ {ev.text}…")
                elif ev.kind == "todos":
                    self._todos = (ev.data or {}).get("todos") or []
                    self._render_tasks()
                elif ev.kind == "task":
                    self._tasks_snapshot = (ev.data or {}).get("tasks") or []
                    self._render_tasks()
                self._scroll()
        except Exception:  # noqa: BLE001 — best-effort; the next typed turn re-reads
            pass
        finally:
            try:
                await self._break_stream()
                if not getattr(self.driver, "has_background_tasks", False):
                    self._status("ready")
                    self._tasks_snapshot = []      # last background agent landed —
                    self._render_tasks()           # the panel's job here is done
                    self._scroll()
            except Exception:  # noqa: BLE001 — teardown / cancelled mid-await
                pass


def main() -> int:
    # One Engram per folder: refuse to start a second LIVE instance in the same cwd — both
    # would resume + write the same session id and interleave into one corrupt thread. A
    # stale lock from a crashed run is auto-reclaimed, so this only bites a genuine
    # double-launch; any lock IO error fails open (the launch proceeds).
    lock = LaunchLock(ENGRAM_CWD)
    owner = lock.acquire()
    if owner is not None:
        sys.stderr.write(
            f"\nEngram is already running in this folder (pid {owner}).\n"
            f"Use that terminal, or close it first.\n"
            f"If you're sure it's gone, remove the stale lock:  rm {lock.path}\n\n")
        return 1
    try:
        EngramApp().run()
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
