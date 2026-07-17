#!/usr/bin/env python3
"""broker m3 — the ENVOY: an autonomous Grok 4.5 research subagent.

Where m1 (``grok_task``) answered one prompt with no tools and m2 exposed it as the
one-shot ``grok`` tool, the envoy runs a *full agentic loop* on Grok 4.5: Engram (on
Claude) delegates a self-contained subtask and the envoy investigates with a
READ-ONLY toolset — file reads, code search, Engram's memory — looping tool-call →
result → tool-call until it can answer, then returns a synthesized result.

The point is to spare Anthropic's daily limits on fan-out: Opus/Fable stay the brain
and orchestrate envoys (choosing ``effort`` per task), while the grunt research runs
off-meter on xAI. It is also the seed of m4's GrokDriver — the same loop, wrapped for
a persistent operator-driven conversation.

Native, not a proxy: a direct xAI ``chat/completions`` tool-calling loop on m1's
``xai_common`` plumbing (key/HTTP/cost) and ``grok_agent.map_effort``. Read-only and
unsupervised by design — no Edit/Write/Bash — because an envoy runs with no human in
the loop (operator-driven full tool use is m4). Fail-open throughout: every failure
comes back as text the orchestrator routes around, never an exception.

recall is fully wired: the SAME ``recall_search`` / ``recall_read_note`` handlers the
SDK path uses (via ``build_recall_tools``) plus the ``recall_inject`` hook run as a
loop preamble, so the envoy reasons with Engram's memory. ``x_search`` (live X/web) is
wired the same way when a key is present.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import xai_common
from claude_agent_sdk import create_sdk_mcp_server, tool
from grok_agent import MODEL, map_effort

MAX_STEPS = 8            # tool-call rounds before we force a synthesis
MAX_READ_CHARS = 50_000  # cap a single file read into context
MAX_HITS = 200           # cap grep/glob result lines
TIMEOUT = 180.0          # per xAI call
WALL_CLOCK = 420.0       # whole-envoy budget (seconds)


# --- recall inject (loop preamble) ------------------------------------------
_INJECT = os.environ.get("ENGRAM_BRIDGE_INJECT_HOOK") or str(
    Path(__file__).resolve().parents[2] / "scripts" / "recall_inject.py")


def _recall_inject(task: str, cwd: Path) -> str:
    """Retrieved-memory block for this task via the canonical recall_inject hook (the
    same logic the terminal + phone use). Best-effort: any hiccup → empty string."""
    if not _INJECT or not Path(_INJECT).exists():
        return ""
    payload = json.dumps({"prompt": task, "cwd": str(cwd), "session_id": "envoy",
                          "hook_event_name": "UserPromptSubmit"})
    try:
        proc = subprocess.run(
            [sys.executable, _INJECT], input=payload, capture_output=True,
            text=True, timeout=30, cwd=str(cwd),
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(cwd)})
        out = (proc.stdout or "").strip()
        if not out:
            return ""
        data = json.loads(out)
        return (data.get("hookSpecificOutput") or {}).get("additionalContext") or ""
    except Exception:  # noqa: BLE001 — inject must never block the envoy
        return ""


# --- JSON-Schema conversion for the in-process tools ------------------------
_PY_JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
            dict: "object", list: "array"}


def _json_schema(shorthand: dict) -> dict:
    """SDK shorthand ({"query": str, "k": int}) → an OpenAI JSON-Schema object.
    Permissive (nothing required): the handlers already default every field, so a
    model that omits one still gets a graceful answer, not an API rejection."""
    props = {name: {"type": _PY_JSON.get(typ, "string")}
             for name, typ in (shorthand or {}).items()}
    return {"type": "object", "properties": props}


def _mcp_text(result: dict) -> str:
    """Flatten an in-process MCP tool result ({"content": [{text}], is_error?}) to
    plain text for a tool-result message."""
    parts = [b.get("text", "") for b in (result.get("content") or [])
             if isinstance(b, dict) and b.get("type") == "text"]
    body = "\n".join(p for p in parts if p).strip() or "(no output)"
    return f"[error] {body}" if result.get("is_error") else body


# --- native read-only executors ---------------------------------------------
def _within(path: str, cwd: Path) -> Optional[Path]:
    """Resolve ``path`` (relative to cwd) and confirm it stays inside the workspace.
    None => outside → denied (an unsupervised envoy reads only its workspace)."""
    try:
        p = Path(path).resolve() if os.path.isabs(path) else (cwd / path).resolve()
        p.relative_to(cwd.resolve())
        return p
    except Exception:  # noqa: BLE001
        return None


async def _read_file(args: dict, cwd: Path) -> str:
    p = _within(str(args.get("path") or ""), cwd)
    if p is None:
        return "[denied] path outside the workspace"

    def _do() -> str:
        if not p.exists() or not p.is_file():
            return f"[not found] {args.get('path')}"
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "[skipped] binary or non-UTF-8 file"
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n…(truncated)"
        return text
    return await asyncio.to_thread(_do)


async def _grep(args: dict, cwd: Path) -> str:
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return "[error] grep: empty pattern"
    root = _within(str(args.get("path") or "."), cwd)
    if root is None:
        return "[denied] path outside the workspace"

    def _do() -> str:
        cmd = ["rg", "--line-number", "--no-heading", "--color=never",
               "--max-count", "50", "-e", pattern, "--", str(root)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=30, cwd=str(cwd))
        except FileNotFoundError:
            return "[unavailable] ripgrep (rg) not installed"
        except Exception as e:  # noqa: BLE001
            return f"[error] grep: {type(e).__name__}: {e}"
        lines = (proc.stdout or "").splitlines()
        if not lines:
            return "(no matches)"
        base = str(cwd.resolve()) + os.sep
        shown = [ln.replace(base, "") for ln in lines[:MAX_HITS]]
        extra = f"\n…(+{len(lines) - len(shown)} more)" if len(lines) > len(shown) else ""
        return "\n".join(shown) + extra
    return await asyncio.to_thread(_do)


async def _glob(args: dict, cwd: Path) -> str:
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        return "[error] glob: empty pattern"

    def _do() -> str:
        try:
            matches = sorted(str(p.relative_to(cwd)) for p in cwd.glob(pattern)
                             if p.is_file())
        except Exception as e:  # noqa: BLE001
            return f"[error] glob: {type(e).__name__}: {e}"
        if not matches:
            return "(no matches)"
        shown = matches[:MAX_HITS]
        extra = f"\n…(+{len(matches) - len(shown)} more)" if len(matches) > len(shown) else ""
        return "\n".join(shown) + extra
    return await asyncio.to_thread(_do)


_READ_DESC = ("Read a UTF-8 text file inside the workspace. Args: path (relative to "
              "the workspace root, or an absolute path within it).")
_GREP_DESC = ("Search file contents with a regular expression (ripgrep). Args: pattern "
              "(regex); path (optional subdir/file, default = whole workspace). Returns "
              "file:line:match lines.")
_GLOB_DESC = ("List workspace files matching a glob, e.g. '**/*.py' or "
              "'infra/**/test_*.py'. Args: pattern.")

_NATIVE = [
    {"name": "read_file", "fn": _read_file, "description": _READ_DESC,
     "schema": {"type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"]}},
    {"name": "grep", "fn": _grep, "description": _GREP_DESC,
     "schema": {"type": "object",
                "properties": {"pattern": {"type": "string"},
                               "path": {"type": "string"}},
                "required": ["pattern"]}},
    {"name": "glob", "fn": _glob, "description": _GLOB_DESC,
     "schema": {"type": "object", "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"]}},
]


# --- toolset assembly (native + in-process, one registry) -------------------
def _inprocess_tools(cwd: Path) -> list:
    """recall (+ x_search if a key is present) SdkMcpTool objects — best-effort, so a
    missing corpus or key just drops that tool, never breaks the envoy."""
    tools: list = []
    try:
        from memory_tools import build_recall_tools
        tools += build_recall_tools(cwd)
    except Exception:  # noqa: BLE001
        pass
    try:
        from x_search import build_x_search_tools
        tools += build_x_search_tools()
    except Exception:  # noqa: BLE001
        pass
    return tools


async def _run_mcp(handler, args: dict) -> str:
    try:
        return _mcp_text(await handler(args))
    except Exception as e:  # noqa: BLE001 — a tool failure is a result, not a crash
        return f"[error] {type(e).__name__}: {e}"


def _build_toolset(cwd: Path) -> tuple[list, dict]:
    """(openai_tool_specs, dispatch{name -> async (args)->str}). The in-process
    recall/x_search tools reuse the SAME handlers as the SDK path; the native
    read/search executors are bound to cwd."""
    specs: list = []
    dispatch: dict[str, Callable[[dict], Awaitable[str]]] = {}

    for ex in _NATIVE:
        specs.append({"type": "function", "function": {
            "name": ex["name"], "description": ex["description"],
            "parameters": ex["schema"]}})
        dispatch[ex["name"]] = (lambda f: (lambda args: f(args, cwd)))(ex["fn"])

    for t in _inprocess_tools(cwd):
        specs.append({"type": "function", "function": {
            "name": t.name, "description": t.description,
            "parameters": _json_schema(t.input_schema)}})
        dispatch[t.name] = (lambda h: (lambda args: _run_mcp(h, args)))(t.handler)
    return specs, dispatch


# --- the envoy loop ----------------------------------------------------------
@dataclass
class EnvoyResult:
    """Outcome of one envoy run. ``error`` set => everything else best-effort."""
    text: Optional[str] = None
    cost_usd: float = 0.0
    model: Optional[str] = None
    steps: int = 0
    truncated: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


_SYSTEM = (
    "You are Engram's envoy — an autonomous research agent running on Grok 4.5, dispatched "
    "to complete ONE self-contained subtask and report a synthesized result to Engram (who "
    "runs on Claude and delegated to you). You have READ-ONLY tools: read_file, grep, glob "
    "over the workspace; recall_search / recall_read_note for Engram's curated memory; and, "
    "when available, x_search for live X/web. You cannot edit files, run commands, or ask "
    "questions — you have no interactive channel, so make reasonable assumptions and STATE "
    "them. Investigate with the tools before answering; do not guess when a tool can settle "
    "it. Return findings and conclusions, not a narration of your steps."
)


async def run_envoy(task: str, *, effort: str = "low", cwd: Path | str = ".",
                    max_steps: int = MAX_STEPS) -> EnvoyResult:
    """Run one delegated subtask on Grok 4.5 with the read-only research toolset,
    looping tool-call → result until Grok answers (or a step/time budget trips).
    Never raises — a missing key, network error, or step-limit come back on the
    :class:`EnvoyResult`."""
    key = xai_common.load_key()
    if not key:
        return EnvoyResult(error="no xAI key (set XAI_API_KEY or ~/.config/recall/xai.env)")
    task = (task or "").strip()
    if not task:
        return EnvoyResult(error="empty task")

    cwd = Path(cwd).resolve()
    specs, dispatch = _build_toolset(cwd)
    inject = _recall_inject(task, cwd)
    system = _SYSTEM + (f"\n\n--- Engram's memory (relevant to this task) ---\n{inject}"
                        if inject else "")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]

    total_cost = 0.0
    model = MODEL
    start = time.monotonic()
    for step in range(1, max_steps + 1):
        if time.monotonic() - start > WALL_CLOCK:
            return EnvoyResult(text="(envoy exceeded its time budget before concluding)",
                               cost_usd=total_cost, model=model, steps=step - 1,
                               truncated=True)
        payload = {"model": MODEL, "messages": messages, "tools": specs,
                   "tool_choice": "auto", "reasoning_effort": map_effort(effort)}
        resp, err = await asyncio.to_thread(
            xai_common.post_json, "chat/completions", payload, key, TIMEOUT)
        if err:
            return EnvoyResult(error=err, cost_usd=total_cost, steps=step - 1, model=model)
        try:
            choice = resp["choices"][0]
            msg = choice["message"]
            model = resp.get("model") or model
            total_cost += xai_common.usd_from_ticks(
                (resp.get("usage") or {}).get("cost_in_usd_ticks")) or 0.0
        except Exception as e:  # noqa: BLE001 — malformed payload → fail-open
            return EnvoyResult(error=f"parse: {type(e).__name__}: {e}",
                               cost_usd=total_cost, steps=step - 1, model=model)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return EnvoyResult(text=(msg.get("content") or "").strip(),
                               cost_usd=total_cost, model=model, steps=step)

        messages.append(msg)
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except Exception:  # noqa: BLE001 — bad tool args → empty, tool defaults
                args = {}
            runner = dispatch.get(name)
            content = (await runner(args)) if runner else f"[error] unknown tool {name!r}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": content})

    return EnvoyResult(text="(envoy reached its step limit before concluding)",
                       cost_usd=total_cost, model=model, steps=max_steps, truncated=True)


# --- in-process MCP tool: Engram dispatches an envoy ---------------------------
ENVOY_DESC = (
    "Dispatch an autonomous research ENVOY that runs on Grok 4.5 (a non-Anthropic model) "
    "to complete ONE self-contained subtask and report back — sparing Anthropic limits on "
    "fan-out. Unlike `grok` (a one-shot opinion, no tools), the envoy LOOPS with a READ-ONLY "
    "toolset: file reads, code search (grep/glob), Engram's memory (recall), and live X/web "
    "(x_search). Reach for it to investigate, research, or bulk-analyze — 'map how X works "
    "across the code', 'research Y and summarize', 'find every call site of Z'. It cannot "
    "edit files or run commands. Pass ALL the context it needs (it sees only your task). "
    "'effort' picks the reasoning budget (low|medium|high, default low; raise it for hard "
    "investigations). Costs real money; the result is one delegated worker's output — weigh "
    "it, don't obey it."
)


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _format(result: EnvoyResult) -> str:
    """The envoy's answer plus a model/steps/cost footer — same footer shape as the
    other Grok tools, so a downstream reader sees provenance and spend at a glance."""
    body = (result.text or "").strip() or "(envoy returned no text)"
    bits = [f"model {result.model or MODEL}",
            f"{result.steps} step{'s' if result.steps != 1 else ''}",
            f"~${result.cost_usd:.3f}"]
    if result.truncated:
        bits.append("truncated")
    return f"{body}\n\n[envoy · {' · '.join(bits)}]"


def build_envoy_server(cwd: Path | str = ".", *, require_key: bool = True):
    """The in-process MCP server ('envoy') exposing the ``envoy`` delegation tool.
    Returns None when no xAI key is resolvable (nothing to expose) or the SDK server
    can't be built — the caller treats envoy delegation as strictly optional, exactly
    like grok / x_search / image_gen."""
    if require_key and xai_common.load_key() is None:
        return None

    @tool("envoy", ENVOY_DESC, {"task": str, "effort": str})
    async def envoy(args: dict) -> dict:
        try:
            task = str(args.get("task") or "").strip()
            if not task:
                return _err("envoy: empty task")
            effort = str(args.get("effort") or "low").strip() or "low"
            result = await run_envoy(task, effort=effort, cwd=cwd)
            if not result.ok:
                return _err(f"envoy unavailable: {result.error}")
            return _text(_format(result))
        except Exception as exc:  # noqa: BLE001 — fail open, model routes around
            return _err(f"envoy unavailable: {type(exc).__name__}: {exc}")

    try:
        return create_sdk_mcp_server(name="envoy", version="1.0.0", tools=[envoy])
    except Exception:  # noqa: BLE001 — envoy delegation is strictly optional
        return None
