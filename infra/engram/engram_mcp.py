#!/usr/bin/env python3
"""Single source of truth for Engram's in-process MCP servers.

Both entry points wire tools through here — the terminal driver (``core.py``) and
the Telegram bridge (``agent_bridge.py``) — so the two can never drift again. (They
did: ``x_search`` was wired in core but the bridge builds its own options and never
got it, so the phone silently lacked the tool.)

Each tool is env-gated and strictly optional: a build that returns None or raises
is skipped, never fatal. The tool modules live beside this file in ``infra/engram``;
callers must have that directory on ``sys.path`` (core.py is already there; the
bridge inserts it).
"""
from __future__ import annotations

import os

# MCP tool names to pre-allow per server, for ask-mode permission (the bridge runs
# bypassPermissions and ignores these; the TUI's ask-mode needs them so a memory
# lookup / X read / image gen never throws a permission card).
_ALLOW = {
    "recall": ["mcp__recall__recall_search", "mcp__recall__recall_read_note"],
    "x_search": ["mcp__x_search__x_search"],
    "image_gen": ["mcp__image_gen__image_generate"],
    "grok": ["mcp__grok__grok"],
    "envoy": ["mcp__envoy__envoy"],
}


def build_servers(cwd) -> dict:
    """Build every enabled in-process MCP server, keyed by name. Order is stable
    (recall, x_search, image_gen, grok, envoy). Any tool whose env gate is '0', whose
    key is absent, or whose build fails is simply omitted."""
    servers: dict = {}

    if os.environ.get("ENGRAM_RECALL_TOOLS", "1") != "0":
        try:
            from memory_tools import build_recall_server
            srv = build_recall_server(cwd)
            if srv is not None:
                servers["recall"] = srv
        except Exception:  # noqa: BLE001 — memory tools must never block launch
            pass

    if os.environ.get("ENGRAM_X_SEARCH", "1") != "0":
        try:
            from x_search import build_x_search_server
            srv = build_x_search_server()
            if srv is not None:
                servers["x_search"] = srv
        except Exception:  # noqa: BLE001 — X search must never block launch
            pass

    if os.environ.get("ENGRAM_IMAGE_GEN", "1") != "0":
        try:
            from image_gen import build_image_gen_server
            srv = build_image_gen_server()
            if srv is not None:
                servers["image_gen"] = srv
        except Exception:  # noqa: BLE001 — image gen must never block launch
            pass

    if os.environ.get("ENGRAM_GROK", "1") != "0":
        try:
            from grok_agent import build_grok_server
            srv = build_grok_server()
            if srv is not None:
                servers["grok"] = srv
        except Exception:  # noqa: BLE001 — grok delegation must never block launch
            pass

    if os.environ.get("ENGRAM_ENVOY", "1") != "0":
        try:
            from envoy import build_envoy_server
            srv = build_envoy_server(cwd)
            if srv is not None:
                servers["envoy"] = srv
        except Exception:  # noqa: BLE001 — envoy delegation must never block launch
            pass

    return servers


def allowed_tool_names(servers) -> list:
    """Pre-allow list for the servers actually registered (ask-mode only)."""
    names: list = []
    for key in servers:
        names += _ALLOW.get(key, [])
    return names
