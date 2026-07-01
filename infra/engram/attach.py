"""Attachment capture for the Engram TUI.

Two kinds of "paste", per the polish research:
  • a dropped/pasted file PATH — arrives as text (bracketed paste / drag-drop);
    parsed here from a Textual ``events.Paste``.
  • a clipboard IMAGE (a screenshot, no file on disk) — NOT delivered to a terminal
    app, so we shell out to the platform clipboard tool on a keypress and write the
    bytes to a temp PNG.

Both converge on a local file path, which the agent's ``Read`` tool renders
(images visually) — the same "path → Read" flow Claude Code and the recall
Telegram bridge use. Linux/Wayland-first (this box has ``wl-paste``); X11/macOS
handled too. Pure stdlib + the system clipboard tool — no new Python deps.
"""
from __future__ import annotations

import asyncio
import os
import platform
import shlex
import time
import urllib.parse
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
CLIP_CACHE = Path("~/.cache/engram/clips").expanduser()


def parse_dropped_paths(text: str) -> list[Path]:
    """Existing file paths inside a paste/drop. Handles single/double quotes,
    backslash-escaped spaces (shlex), ``file://`` URIs (percent-decoded), and
    multi-file drops. Returns only paths that actually exist as files."""
    text = text.strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError:           # unbalanced quotes — fall back to naive split
        tokens = text.split()
    out: list[Path] = []
    for tok in tokens:
        if tok.startswith("file://"):
            tok = urllib.parse.unquote(urllib.parse.urlparse(tok).path)
        p = Path(tok).expanduser()
        if p.is_file() and p not in out:
            out.append(p)
    return out


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


async def _run(*cmd: str) -> bytes | None:
    """Run a command, return stdout bytes (None on any failure / missing tool)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        return out if proc.returncode == 0 else None
    except (FileNotFoundError, OSError):
        return None


async def grab_clipboard_image(now: str | None = None) -> Path | None:
    """Pull an image off the clipboard into a temp PNG; ``None`` if there isn't
    one (or no clipboard tool). Wayland → ``wl-paste``, X11 → ``xclip``, macOS →
    ``pngpaste``. Async so the UI never blocks."""
    system = platform.system()
    data: bytes | None = None
    if system == "Linux" and os.environ.get("WAYLAND_DISPLAY"):
        types = await _run("wl-paste", "--list-types")
        if not types or b"image/" not in types:
            return None
        data = await _run("wl-paste", "--type", "image/png")
    elif system == "Linux":
        targets = await _run("xclip", "-selection", "clipboard", "-t", "TARGETS", "-o")
        if not targets or b"image/png" not in targets:
            return None
        data = await _run("xclip", "-selection", "clipboard", "-t", "image/png", "-o")
    elif system == "Darwin":
        data = await _run("pngpaste", "-")
    if not data:
        return None
    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    stamp = now or time.strftime("%Y%m%d-%H%M%S")
    out = CLIP_CACHE / f"clip-{stamp}.png"
    out.write_bytes(data)
    return out
