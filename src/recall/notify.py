"""Best-effort failure notifier for unattended recall jobs (nightly curation).

Silent no-op unless ``RECALL_TELEGRAM_TOKEN`` + ``RECALL_TELEGRAM_CHAT_ID`` are
set: missing creds never raise, so a job's success path never depends on
notifications being configured.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request


def notify_alert(title: str, body: str, *, priority: str = "high") -> bool:
    """POST a short alert to Telegram if creds are configured; else no-op.
    Returns True iff a message was actually sent. Never raises."""
    token = os.environ.get("RECALL_TELEGRAM_TOKEN")
    chat = os.environ.get("RECALL_TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    mark = "⚠️ " if priority in ("high", "urgent") else ""
    payload = json.dumps({"chat_id": chat, "text": f"{mark}{title}\n{body}"}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True
    except Exception:  # noqa: BLE001 — notification is best-effort, never fatal
        return False


# --- markdown -> Telegram HTML --------------------------------------------
# The conversational bridge (Engram over Telegram) composes replies in markdown;
# this renders the subset to Telegram HTML so they read richly on the phone. Never
# raises on odd input — the bridge's _send_one falls back to plain text if Telegram
# rejects it.

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")
_CODE_RE = re.compile(r"`([^`]+)`")
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_ITAL_STAR_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_ITAL_US_RE = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_SPOILER_RE = re.compile(r"\|\|(.+?)\|\|")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_md(text: str) -> str:
    """Inline-level markdown -> HTML for a single line. Inline code is stashed
    BEFORE escaping so identifiers like ``recall_inject.py`` keep their underscores
    (otherwise the italic rule would eat them)."""
    codes: list[str] = []

    def _stash(m: "re.Match[str]") -> str:
        codes.append(m.group(1))
        return f"\x00C{len(codes) - 1}\x00"

    text = _CODE_RE.sub(_stash, text)
    text = _esc(text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)          # before single-* italic
    text = _SPOILER_RE.sub(r"<tg-spoiler>\1</tg-spoiler>", text)
    text = _ITAL_STAR_RE.sub(r"<i>\1</i>", text)
    text = _ITAL_US_RE.sub(r"<i>\1</i>", text)
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)
    for i, c in enumerate(codes):
        text = text.replace(f"\x00C{i}\x00", f"<code>{_esc(c)}</code>")
    return text


def _md_to_telegram_html(md: str) -> str:
    """Convert the markdown subset Engram emits into Telegram HTML: headers -> bold
    (H1 also underlined), ``**bold**``, ``_italic_``/``*x*``, `` `code` `` ->
    monospace, ```` ```fenced``` ```` -> ``<pre>``, ``-`` bullets -> •, ``---`` ->
    divider, ``[t](url)`` links, ``>`` blockquotes, ``||spoiler||``. Everything
    else is HTML-escaped and passed through. Never raises on odd input."""
    fences: list[str] = []

    def _stash_fence(m: "re.Match[str]") -> str:
        fences.append(m.group(1))
        return f"\x00F{len(fences) - 1}\x00"

    md = _FENCE_RE.sub(_stash_fence, md)
    out: list[str] = []
    quote: list[str] = []

    def _flush_quote() -> None:
        if quote:
            out.append("<blockquote>" + "\n".join(quote) + "</blockquote>")
            quote.clear()

    for raw in md.split("\n"):
        if raw.startswith(">"):
            quote.append(_inline_md(raw[1:].lstrip()))
            continue
        _flush_quote()
        m = _HEADER_RE.match(raw)
        if m:
            txt = _inline_md(m.group(2))
            out.append(f"<b><u>{txt}</u></b>" if len(m.group(1)) == 1
                       else f"<b>{txt}</b>")
            continue
        if _HR_RE.match(raw):
            out.append("──────────")
            continue
        m = _BULLET_RE.match(raw)
        if m:
            out.append(f"{m.group(1)}• {_inline_md(m.group(2))}")
            continue
        out.append(_inline_md(raw))
    _flush_quote()

    result = "\n".join(out)
    for i, block in enumerate(fences):
        result = result.replace(f"\x00F{i}\x00", f"<pre>{_esc(block)}</pre>")
    return result
