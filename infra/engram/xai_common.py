#!/usr/bin/env python3
"""Shared plumbing for Engram's in-process xAI/Grok tools (x_search, image_gen).

Kept tiny and torch-free — key resolution, the API base, a JSON POST with uniform
error capture, and the cost-tick decode. Every Grok tool imports from here so the
key logic and the ``cost_in_usd_ticks`` convention live in exactly one place.

Cost note: xAI reports spend as ``cost_in_usd_ticks`` in NANO-USD — divide by 1e9
for dollars (verified against a real search that billed 237811000 ticks = $0.238
and an image that billed 200000000 = $0.20)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENV_FILE = "~/.config/recall/xai.env"
TICKS_PER_USD = 1_000_000_000     # cost_in_usd_ticks are nano-USD


def load_key() -> str | None:
    """xAI key from the environment first, else parsed from the env file
    (``XAI_ENV_FILE`` or ~/.config/recall/xai.env). Reading the file at call-time
    means a tool works the instant the key lands — no relaunch, no systemd
    EnvironmentFile dependency."""
    key = os.environ.get("XAI_API_KEY")
    if key and key.strip():
        return key.strip()
    path = Path(os.path.expanduser(os.environ.get("XAI_ENV_FILE") or DEFAULT_ENV_FILE))
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if line.startswith("XAI_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    except Exception:  # noqa: BLE001 — no file / unreadable → no key
        pass
    return None


def api_base() -> str:
    return os.environ.get("XAI_API_BASE", "https://api.x.ai/v1").rstrip("/")


def usd_from_ticks(ticks) -> float | None:
    return ticks / TICKS_PER_USD if isinstance(ticks, (int, float)) else None


def post_json(path: str, payload: dict, key: str,
              timeout: float) -> tuple[dict | None, str | None]:
    """POST JSON to ``{api_base}/{path}`` with bearer auth. Returns (response,
    None) or (None, error_message) — never raises, so callers stay fail-open."""
    url = f"{api_base()}/{path.lstrip('/')}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
            msg = body.get("error") or body
        except Exception:  # noqa: BLE001
            msg = f"HTTP {e.code}"
        return None, f"xAI API error: {msg}"
    except Exception as e:  # noqa: BLE001 — network/timeout/parse
        return None, f"{type(e).__name__}: {e}"
