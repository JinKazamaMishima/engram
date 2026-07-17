#!/usr/bin/env python3
"""broker m4 — the GrokDriver: Engram *running on* Grok 4.5.

m3's envoy proved the agentic loop as a one-shot, read-only, UNSUPERVISED subagent.
The GrokDriver is the same loop grown into a persistent, OPERATOR-DRIVEN conversation
driver: it implements ``core.ModelDriver`` so the TUI (and, once wired, the phone) can
run Engram itself on Grok 4.5 when Anthropic limits bite — same tools, same memory, same
Event stream, just a non-Anthropic backend. No proxy: a direct xAI ``chat/completions``
tool-calling loop on m1's ``xai_common``.

Two things separate it from the envoy:
  * PERSISTENT history — ``query`` appends to ``self.messages`` and each call continues
    the same conversation (envoy was task-in / result-out).
  * FULL tools — because THE OPERATOR is in the loop (supervised), the driver adds write_file /
    edit_file / bash on top of the envoy's read/research set. (The envoy stays read-only:
    it runs unattended.)

recall is fully wired: the read tools + the ``recall_inject`` preamble are reused verbatim
from the envoy tool layer, so both drivers pull ONE source. Fail-open throughout: a bad
turn yields an error Event and leaves the conversation intact.

v1 scope (see PLAN-broker-m3.md): non-streaming (one text Event per model turn — the phone
sends whole messages anyway; token streaming is a later milestone); in-memory history only
(no cross-restart resume / checkpoints / compaction — those are SDK-provided for Claude and
land as parity later); interactive tools (plan / questions) not intercepted (the supervised
surfaces run bypass). WIRING into the TUI/bridge is m5.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import envoy
import xai_common
from core import EFFORT_LEVELS, PERSONA, Event, ModelDriver
from grok_agent import MODEL, map_effort

MAX_STEPS = 12           # tool-call rounds in a single turn before we force a synthesis
TIMEOUT = 180.0          # per xAI call
TURN_WALL_CLOCK = 600.0  # whole-turn budget (seconds)
MAX_BASH_CHARS = 30_000  # cap bash output into context
# Grok's context window — the meter's denominator. xAI doesn't return it in the
# usage block, so we carry it here, env-tunable because it moves with the model.
# NOTE: confirm the live figure against xAI's grok-4.5 spec; 256k is the safe floor
# (Grok 4's window), chosen over an optimistic guess so the gauge never under-reads
# how full the window is.
CONTEXT_WINDOW = int(os.environ.get("ENGRAM_GROK_CONTEXT_WINDOW") or 256_000)


# --- full (read + write) toolset: the envoy read layer + operator-supervised writes ---
async def _write_file(args: dict, cwd: Path) -> str:
    p = envoy._within(str(args.get("path") or ""), cwd)
    if p is None:
        return "[denied] path outside the workspace"
    content = str(args.get("content") or "")

    def _do() -> str:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[wrote] {args.get('path')} ({len(content)} chars)"
    return await asyncio.to_thread(_do)


async def _edit_file(args: dict, cwd: Path) -> str:
    p = envoy._within(str(args.get("path") or ""), cwd)
    if p is None:
        return "[denied] path outside the workspace"
    old, new = str(args.get("old") or ""), str(args.get("new") or "")
    if not old:
        return "[error] edit_file: empty 'old' string"
    replace_all = bool(args.get("replace_all"))

    def _do() -> str:
        if not p.exists():
            return f"[not found] {args.get('path')}"
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return "[error] 'old' string not found"
        if count > 1 and not replace_all:
            return f"[error] 'old' string is not unique ({count} matches); pass replace_all"
        p.write_text(text.replace(old, new), encoding="utf-8")
        return f"[edited] {args.get('path')} ({count} replacement{'s' if count != 1 else ''})"
    return await asyncio.to_thread(_do)


async def _bash(args: dict, cwd: Path) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        return "[error] bash: empty command"

    def _do() -> str:
        try:
            proc = subprocess.run(command, shell=True, capture_output=True, text=True,
                                  timeout=120, cwd=str(cwd))
        except subprocess.TimeoutExpired:
            return "[error] bash: timed out (120s)"
        except Exception as e:  # noqa: BLE001
            return f"[error] bash: {type(e).__name__}: {e}"
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        out = out.strip() or f"(no output, exit {proc.returncode})"
        if len(out) > MAX_BASH_CHARS:
            out = out[:MAX_BASH_CHARS] + "\n…(truncated)"
        return out
    return await asyncio.to_thread(_do)


_WRITE_DESC = "Write a UTF-8 text file in the workspace (creating parents). Args: path, content."
_EDIT_DESC = ("Replace a string in a workspace file. Args: path; old (must match exactly and, "
              "unless replace_all=true, be unique); new; replace_all (optional bool).")
_BASH_DESC = ("Run a shell command in the workspace and return combined stdout/stderr. Args: "
              "command. You are operator-supervised — use it as you would a terminal.")

_WRITE_NATIVE = [
    {"name": "write_file", "fn": _write_file, "description": _WRITE_DESC,
     "schema": {"type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]}},
    {"name": "edit_file", "fn": _edit_file, "description": _EDIT_DESC,
     "schema": {"type": "object",
                "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                               "new": {"type": "string"}, "replace_all": {"type": "boolean"}},
                "required": ["path", "old", "new"]}},
    {"name": "bash", "fn": _bash, "description": _BASH_DESC,
     "schema": {"type": "object", "properties": {"command": {"type": "string"}},
                "required": ["command"]}},
]


def _full_toolset(cwd: Path) -> tuple[list, dict]:
    """The envoy's read/research layer (read_file/grep/glob + recall + x_search — reused
    verbatim, ONE source) plus the operator-supervised write tools."""
    specs, dispatch = envoy._build_toolset(cwd)
    for w in _WRITE_NATIVE:
        specs.append({"type": "function", "function": {
            "name": w["name"], "description": w["description"], "parameters": w["schema"]}})
        dispatch[w["name"]] = (lambda f: (lambda a: f(a, cwd)))(w["fn"])
    return specs, dispatch


_PERSONA_HEAD = (
    "You are Engram, running on Grok 4.5 (xAI) instead of your usual Claude backend — the operator "
    "switched you here, deliberately. You are the SAME Engram: same name, same relationship, "
    "same standing rules and memory. You have full tools — read_file, grep, glob, write_file, "
    "edit_file, bash, plus recall (your memory) and x_search — and you act on your own "
    "judgment, holding only for the gates the operator set (show diffs before commits; hold before "
    "irreversible or outward-facing actions). Investigate with tools before acting; never "
    "guess when a tool can settle it. Standing rules and relevant memory are injected below "
    "each turn."
)


class GrokDriver(ModelDriver):
    """Engram on Grok 4.5 via a native xAI tool-calling loop. Implements the ModelDriver
    surface the TUI already consumes, so switching the backend needs no front-end change.
    Persistent in-memory history; full (read+write) tools; recall injected per turn."""

    def __init__(self, *, cwd: Path | str = ".", effort: str = "low",
                 model: str = MODEL, persona: str = PERSONA) -> None:
        self.cwd = Path(cwd).resolve()
        self.effort = effort
        self.model = model
        self.persona = persona
        self.actual_model: Optional[str] = None
        self.fallback_model: Optional[str] = None
        self.on_interaction = None
        self._messages: list[dict] = []
        self._specs: list = []
        self._dispatch: dict = {}
        self._last_usage: dict = {}   # aurora m4: last xAI usage block, for the meter

    # --- lifecycle -----------------------------------------------------------
    async def connect(self) -> None:
        if not self._specs:
            self._specs, self._dispatch = await asyncio.to_thread(_full_toolset, self.cwd)

    async def disconnect(self) -> None:
        return

    def reset(self) -> None:
        """/new — drop the conversation (the tool layer is stateless, keep it)."""
        self._messages = []

    async def set_effort(self, level: str) -> None:
        if level in EFFORT_LEVELS:
            self.effort = level

    async def set_model(self, name: str) -> None:
        self.model = name or self.model

    async def get_context_usage(self) -> dict:
        """aurora m4: normalize xAI's usage into the shape the TUI meter reads, so
        the gauge works on Grok exactly as on Claude. ``prompt_tokens`` of the LAST
        response is the tokens that went into the model = the live context fill; the
        window is our own constant (xAI doesn't report it). ``isAutoCompactEnabled``
        is False and TRUTHFUL — Grok has no compaction or auto-compact net, so a
        filling window is a hard wall and the meter must say so. Empty until the
        first response (fail-open → ``{}``, the gauge stays blank)."""
        used = int((self._last_usage or {}).get("prompt_tokens") or 0)
        if used <= 0:
            return {}
        return {
            "totalTokens": used,
            "rawMaxTokens": CONTEXT_WINDOW,
            "maxTokens": CONTEXT_WINDOW,
            "percentage": 100.0 * used / CONTEXT_WINDOW,
            "model": self.actual_model or self.model,
            "isAutoCompactEnabled": False,
        }

    def _system(self) -> str:
        return f"{_PERSONA_HEAD}\n\n{self.persona}" if self.persona else _PERSONA_HEAD

    # --- the turn ------------------------------------------------------------
    async def query(self, text: str, *, prepend: str = "") -> AsyncIterator[Event]:
        """One operator turn: inject memory, then loop tool-call → result until Grok
        answers, streaming Events. ``prepend`` is model-only context (never the operator's
        raw text). Never raises — a transport/parse failure yields an error Event and
        leaves history consistent."""
        key = xai_common.load_key()
        if not key:
            yield Event("text", "⚠ Grok unavailable: no xAI key "
                                "(set XAI_API_KEY or ~/.config/recall/xai.env)")
            return
        await self.connect()
        if not self._messages:
            self._messages.append({"role": "system", "content": self._system()})

        inject = await asyncio.to_thread(envoy._recall_inject, text, self.cwd)
        if inject:
            yield Event("recall", "")          # ran; provenance detail is a later milestone
            self._messages.append({"role": "system", "content": f"[memory]\n{inject}"})
        self._messages.append({"role": "user", "content": (prepend + text) if prepend else text})

        start = time.monotonic()
        for _ in range(MAX_STEPS):
            if time.monotonic() - start > TURN_WALL_CLOCK:
                yield Event("text", "\n\n> ⏳ *turn exceeded its time budget*")
                return
            payload = {"model": self.model, "messages": self._messages, "tools": self._specs,
                       "tool_choice": "auto", "reasoning_effort": map_effort(self.effort)}
            resp, err = await asyncio.to_thread(
                xai_common.post_json, "chat/completions", payload, key, TIMEOUT)
            if err:
                yield Event("text", f"\n\n> ⚠ *Grok error: {err}*")
                return
            try:
                msg = resp["choices"][0]["message"]
                self.actual_model = resp.get("model") or self.actual_model
                # aurora m4: prompt_tokens of this response = the context that went
                # in = the live window fill the meter reads. Keep the last non-empty.
                self._last_usage = resp.get("usage") or self._last_usage
            except Exception as e:  # noqa: BLE001 — malformed payload → fail-open
                yield Event("text", f"\n\n> ⚠ *Grok parse error: {type(e).__name__}: {e}*")
                return

            self._messages.append(msg)
            content = (msg.get("content") or "").strip()
            if content:
                yield Event("text", content)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                yield Event("tool", name)
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except Exception:  # noqa: BLE001 — bad tool args → empty, tool defaults
                    args = {}
                runner = self._dispatch.get(name)
                result = (await runner(args)) if runner else f"[error] unknown tool {name!r}"
                self._messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                                       "content": result})

        yield Event("text", "\n\n> ⏳ *reached the step limit for this turn*")
