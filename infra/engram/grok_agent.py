#!/usr/bin/env python3
"""Native Grok worker — Engram's first non-Anthropic hand.

A tiny, torch-free text worker that runs one prompt on Grok 4.5 at a chosen
reasoning effort and returns text (or a schema-validated object). No SDK, no
translation proxy: a *direct* xAI ``chat/completions`` call. That directness is
the point — this is the seed of the driver Engram will eventually run its OWN
model (a local model, served OpenAI-compatible) on, with zero Anthropic in the loop.
Build it against a real frontier model now; reuse it for a local model later.

Fail-open like every Grok tool; shared key/HTTP/cost plumbing lives in
``xai_common`` so the key logic + ``cost_in_usd_ticks`` convention stay in one
place. ``build_grok_server`` exposes ``grok_task`` as the in-process ``grok``
MCP tool, wired for BOTH the TUI and the phone through ``engram_mcp`` — Engram's
first tool that hands a subtask to a non-Anthropic model.

Effort maps Engram's ladder -> xAI's ``reasoning_effort``::

    low -> low   medium -> medium   high|xhigh|max -> high

(Grok 4.5 exposes only low|medium|high; Engram's hotter rungs collapse to high.)
Reasoning models reject ``presence_penalty`` / ``frequency_penalty`` / ``stop``,
so we never send them.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import delegations
import xai_common
from claude_agent_sdk import create_sdk_mcp_server, tool

MODEL = "grok-4.5"

# Engram effort ladder (low|medium|high|xhigh|max) -> xAI reasoning_effort.
_EFFORT_MAP = {
    "low": "low", "medium": "medium", "high": "high",
    "xhigh": "high", "max": "high",
}


def map_effort(level: Optional[str]) -> str:
    """Collapse a Engram effort level onto Grok's low|medium|high."""
    return _EFFORT_MAP.get((level or "low").lower(), "low")


@dataclass
class GrokResult:
    """Outcome of one Grok call. ``error`` set => everything else best-effort."""
    text: Optional[str] = None
    data: Any = None                 # parsed object when a schema was supplied
    cost_usd: Optional[float] = None
    model: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def grok_task(prompt: str, *, effort: str = "low",
              system: Optional[str] = None,
              schema: Optional[dict] = None,
              max_tokens: Optional[int] = None,
              timeout: float = 300.0) -> GrokResult:
    """Run one prompt on Grok 4.5 and return a :class:`GrokResult`.

    ``effort`` picks the reasoning budget (grunt=low, expert=high). ``schema``
    (a JSON Schema dict) forces structured output and parses it into
    ``result.data``. Never raises — a missing key, network error, or malformed
    response comes back as ``result.error``.
    """
    key = xai_common.load_key()
    if not key:
        return GrokResult(error="no xAI key (set XAI_API_KEY or ~/.config/recall/xai.env)")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "reasoning_effort": map_effort(effort),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "out", "strict": True, "schema": schema},
        }

    resp, err = xai_common.post_json("chat/completions", payload, key, timeout)
    if err:
        return GrokResult(error=err)

    try:
        message = resp["choices"][0]["message"]
        text = message.get("content")
        data = json.loads(text) if (schema is not None and text) else None
        usage = resp.get("usage") or {}
        cost = xai_common.usd_from_ticks(usage.get("cost_in_usd_ticks"))
        return GrokResult(text=text, data=data, cost_usd=cost,
                          model=resp.get("model"))
    except Exception as e:  # noqa: BLE001 — malformed payload → fail-open
        return GrokResult(error=f"parse: {type(e).__name__}: {e}")


# --- in-process MCP tool: Engram hands a subtask to Grok -----------------------

GROK_DESC = (
    "Delegate a self-contained subtask to Grok 4.5 (xAI) — a NON-Anthropic model — "
    "and get its text back. Reach for it when a second, independent model helps: a "
    "cross-model sanity check or second opinion, a cheap bulk transform you'd rather "
    "keep off your own reasoning, or a fresh take unanchored to this conversation. "
    "One-shot: Grok sees only the prompt you send — no memory, no tools, no follow-up, "
    "so pass all the context it needs. Optional 'effort' picks the reasoning budget "
    "(low|medium|high, default low; raise it for genuinely hard problems). Each call "
    "costs real money, and the result is ONE model's output — weigh it, don't obey it.")


def _text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _format(result: GrokResult) -> str:
    """Grok's answer plus a model/cost footer — the same shape as the other Grok
    tools, so a downstream reader sees provenance and spend at a glance."""
    body = (result.text or "").strip() or "(Grok returned no text)"
    bits = [f"model {result.model or MODEL}"]
    if result.cost_usd is not None:
        bits.append(f"~${result.cost_usd:.3f}")
    return f"{body}\n\n[grok · {' · '.join(bits)}]"


def build_grok_server(*, require_key: bool = True):
    """The in-process MCP server ('grok') exposing a single ``grok`` tool that runs
    one prompt on Grok 4.5. Returns None when no xAI key is resolvable (nothing to
    expose) or the SDK server can't be built — the caller treats grok delegation as
    strictly optional, exactly like x_search and image_gen."""
    if require_key and xai_common.load_key() is None:
        return None

    @tool("grok", GROK_DESC, {"prompt": str, "effort": str})
    async def grok(args: dict) -> dict:
        try:
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return _err("grok: empty prompt")
            effort = str(args.get("effort") or "low").strip() or "low"
        except Exception as exc:  # noqa: BLE001 — fail open, model routes around
            return _err(f"grok unavailable: {type(exc).__name__}: {exc}")
        # aurora m3: announce the delegation so the TUI panel + one-liner can show
        # this Grok call live — it's a tool call, not an SDK task, so it wouldn't
        # otherwise surface anywhere. finish() in the `finally` is idempotent.
        did = delegations.start(f"grok·{effort}", MODEL, "xai")
        ok = False
        cost = None
        try:
            result = await asyncio.to_thread(grok_task, prompt, effort=effort)
            ok = result.ok
            cost = result.cost_usd
            if not ok:
                return _err(f"grok unavailable: {result.error}")
            return _text(_format(result))
        except Exception as exc:  # noqa: BLE001 — fail open, model routes around
            return _err(f"grok unavailable: {type(exc).__name__}: {exc}")
        finally:
            delegations.finish(did, cost=cost, ok=ok)

    try:
        return create_sdk_mcp_server(name="grok", version="1.0.0", tools=[grok])
    except Exception:  # noqa: BLE001 — grok delegation is strictly optional
        return None
