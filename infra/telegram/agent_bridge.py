#!/usr/bin/env python3
"""recall Telegram *conversational* bridge — "Engram" (Claude Agent SDK).

Lets you talk to Engram about the `recall` repo from your phone. Holds ONE
persistent ``ClaudeSDKClient`` so the conversation has memory across messages;
each ``client.query()`` continues the same session, a new client is a new thread.

It loads the recall project brain (CLAUDE.md + skills) via
``setting_sources=["project"]`` and wires the recall-injection hook EXPLICITLY
into the SDK options — project scope excludes the user-level
~/.claude/settings.json, where a terminal install typically registers that
hook, so without this wiring bridge sessions would silently get no corpus
recall on their turns. Runs on the
Claude **subscription** (no API key — see the env strip below); supports ``/new`` /
``/end`` plus auto-resume of the last thread across a service restart.

One bot, one operator chat id (the allowlist is the load-bearing safety), its
own token + state dir.

Config (env, normally from ~/.config/recall/telegram-agent.env via systemd):
  RECALL_TELEGRAM_AGENT_TOKEN    — bot token from @BotFather (required)
  RECALL_TELEGRAM_AGENT_CHAT_ID  — numeric operator chat id (required, the allowlist)
  RECALL_TELEGRAM_API            — override API base (default https://api.telegram.org)
  RECALL_REPO                    — repo to cwd into (default: the repo root)
  RECALL_DATA_ROOT               — recall data root; state under $.../telegram-agent/
  CLAUDE_BIN                     — claude CLI path (default: found on PATH)
  RECALL_AGENT_IDLE_SECS         — release the warm client after N idle secs (default 1800)
  RECALL_AGENT_EFFORT            — reasoning effort (default max; low|medium|high|xhigh|max)
  RECALL_AGENT_MODEL             — initial model (default opus[1m]); switch live with /model
  RECALL_INJECT_HOOK             — recall-injection hook script (default
                                   <repo>/scripts/recall_inject.py; empty disables)

Built-in commands (no model call): /new /end /cancel /lock /unlock /status /ping /model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# The sibling engine modules (LiveBuffer, the driver's tier-1 gate) live in
# infra/engram; putting it on the path keeps this bridge and the terminal
# driver on ONE wiring path so they can't drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra" / "engram"))

# --- auth: subscription vs API key -------------------------------------------
# The SDK spawns `claude`, which prefers ANTHROPIC_API_KEY over the subscription
# login when present. By default the bridge respects whatever you configured; set
# ENGRAM_FORCE_SUBSCRIPTION=1 to always bill to a Pro/Max subscription by stripping
# any key first (e.g. if a business key is in the environment).
_STRIPPED_API_KEY = False
if os.environ.get("ENGRAM_FORCE_SUBSCRIPTION", "").lower() in ("1", "true", "yes"):
    _STRIPPED_API_KEY = os.environ.pop("ANTHROPIC_API_KEY", None) is not None

from claude_agent_sdk import (  # noqa: E402  (after the env strip on purpose)
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

from buffer import LiveBuffer  # noqa: E402 — sibling infra/engram module, shared with the terminal driver
from core import BUFFER_DIR, BUFFER_ON  # noqa: E402 — tier-1 gate/dir single source (never fork it)

TOKEN = os.environ.get("RECALL_TELEGRAM_AGENT_TOKEN", "")
CHAT_ID_RAW = os.environ.get("RECALL_TELEGRAM_AGENT_CHAT_ID", "")
API_BASE = os.environ.get("RECALL_TELEGRAM_API", "https://api.telegram.org").rstrip("/")
REPO = Path(os.environ.get("RECALL_REPO") or Path(__file__).resolve().parents[2])
DATA_ROOT = Path(os.environ.get("RECALL_DATA_ROOT",
                                os.path.expanduser("~/.local/share/recall")))
CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
# The recall CLI used to curate an ended session in the background (brick 2).
# Default: the console script beside this interpreter (the bridge's own venv).
RECALL_BIN = os.environ.get("RECALL_BIN") or str(Path(sys.executable).with_name("recall"))
# The recall-injection hook: the SAME script a terminal install's UserPromptSubmit
# hook runs, wired explicitly into the SDK options because
# setting_sources=["project"] excludes the user settings where that hook is
# registered. Empty -> disabled.
RECALL_INJECT = os.environ.get("RECALL_INJECT_HOOK",
                               str(REPO / "scripts" / "recall_inject.py"))
INJECT_TIMEOUT = 15        # generous; the script's own daemon budget is 0.6s
IDLE_SECS = int(os.environ.get("RECALL_AGENT_IDLE_SECS", "1800"))


def _csv_env(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


# --- instance config (defaults are the recall Engram bot) ----------------------
AGENT_CWD = Path(os.environ.get("RECALL_AGENT_CWD", str(REPO)))
STATE_DIR = Path(os.environ.get("RECALL_AGENT_STATE_DIR", str(DATA_ROOT / "telegram-agent")))
BOT_LABEL = os.environ.get("RECALL_AGENT_LABEL", "engram")
# Which CLAUDE.md / skills / settings to load. Default: the full recall brain.
# An EMPTY value -> None -> a clean assistant with NO project access.
SETTING_SOURCES: Optional[list[str]] = _csv_env("RECALL_AGENT_SETTING_SOURCES", "project") or None
# Tool whitelist / blacklist. Empty whitelist -> all tools (default).
ALLOWED_TOOLS: Optional[list[str]] = _csv_env("RECALL_AGENT_ALLOWED_TOOLS", "") or None
DISALLOWED_TOOLS: list[str] = _csv_env("RECALL_AGENT_DISALLOWED_TOOLS", "")
# Reasoning effort + model per turn. We load setting_sources=["project"], which
# EXCLUDES the user ~/.claude/settings.json — so without pinning these here,
# phone-Engram would silently fall back to the CLI defaults instead of the operator's
# configured effort/model. Default to sensible values: max effort + Opus 1M context.
AGENT_EFFORT = os.environ.get("RECALL_AGENT_EFFORT", "max")        # low|medium|high|xhigh|max
AGENT_MODEL = os.environ.get("RECALL_AGENT_MODEL", "opus[1m]")  # /model swaps this live

# Models offered by /model for discoverability; free-form still works (the name is
# passed straight to the CLI on the next reconnect). Mirrors the TUI's list so the
# phone and terminal switch between the same aliases. opus[1m] = the 1M window.
MODELS = (
    ("opus[1m]", "Opus 4.8 · 1M context (default)"),
    ("opus",     "Opus 4.8 · 200K"),
    ("sonnet",   "Sonnet 4.6"),
    ("fable",    "Fable 5"),
    ("haiku",    "Haiku 4.5 · fastest"),
)

OFFSET_FILE = STATE_DIR / "last_update_id"
LOCK_FILE = STATE_DIR / "bridge.lock"
LOG_FILE = STATE_DIR / "messages.log"
SESSION_FILE = STATE_DIR / "session_id"
INBOX_DIR = STATE_DIR / "inbox"   # downloaded photos/documents from inbound msgs

POLL_TIMEOUT = 50          # seconds for Telegram long-poll
HTTP_BUDGET = POLL_TIMEOUT + 15
MAX_MSG_LEN = 4096
AUDIT_SNIPPET_LEN = 240
TURN_TIMEOUT = 1200        # hard cap on a single model turn (20 min)
COALESCE_WINDOW = 1.0      # debounce: batch a burst of msgs into ONE turn

PERSONA = (
    "You are Engram, a persistent-memory assistant, reachable here over Telegram. "
    "You are the same assistant that runs in the terminal, now reachable by message. "
    "You are in the `recall` repo: the machine-local memory system that holds the "
    "curated knowledge corpus. It loads the full project brain (CLAUDE.md + skills), "
    "and a retrieval hook auto-injects relevant memory into each message. Read freely: "
    "the code, the corpus, logs, and the nightly curate/consolidate/dream output. Keep "
    "replies concise and conversational unless asked for depth — the user is on a "
    "phone, so a few short paragraphs, not long markdown dumps. Before any "
    "consequential or hard-to-reverse action (writing to the memory corpus, "
    "state-changing shell commands, git commits or pushes, anything outward-facing), "
    "propose it and wait for the user's explicit approval, then do it. You retain the "
    "full project rules from CLAUDE.md. Override this default persona via "
    "RECALL_AGENT_PERSONA_FILE."
)
# An alternate instance may override the persona with its own file.
_PERSONA_FILE = os.environ.get("RECALL_AGENT_PERSONA_FILE", "")
if _PERSONA_FILE:
    try:
        PERSONA = Path(_PERSONA_FILE).read_text().strip()
    except Exception as exc:  # noqa: BLE001
        print(f"persona file {_PERSONA_FILE!r} unreadable ({exc}); using default",
              file=sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("recall-telegram-engram")

BOT_USERNAME = ""

# --- conversation state ------------------------------------------------------
_client: Optional[ClaudeSDKClient] = None
_session_id: Optional[str] = None

# LiveBuffer (tier 1 of the continuous-STM stack) — bridge parity with the
# terminal driver. Phone conversations were invisible to tier 1 (and to
# everything re-derived from it: the working set, eviction, the cogito
# instrument) because only the driver buffered and the bridge holds a raw SDK
# client. Same gate + dir as core, same provisional-launch-id -> migrate-on-sid
# semantics, appending the RAW exchanged text at the audit seams (the log-raw /
# inject-derived invariant). Deliberately NO eviction here: bridge sessions
# already curate per-session on /new + /end — eviction stays driver-side.
_buf_launch_id: str = "launch-" + uuid.uuid4().hex[:12]
_buf_convo_id: str = _buf_launch_id
_buffer = LiveBuffer(BUFFER_DIR if BUFFER_ON else None, lambda: _buf_convo_id)


def _buf_rekey(sid: Optional[str]) -> None:
    """Follow the conversation's identity on disk: mint (launch->sid) and
    resume (saved sid at boot) MIGRATE the file; end-of-conversation
    (sid->None on /new,/end) mints a FRESH launch id and leaves the finished
    conversation's file where it lies. Always reseeds so seq stays strictly
    ordered across restarts and renames. Fail-open throughout (LiveBuffer)."""
    global _buf_convo_id, _buf_launch_id
    if sid:
        if sid != _buf_convo_id:
            _buffer.migrate(_buf_convo_id, sid)
            _buf_convo_id = sid
    else:
        _buf_launch_id = "launch-" + uuid.uuid4().hex[:12]
        _buf_convo_id = _buf_launch_id
    _buffer.reseed()


_current_model: str = AGENT_MODEL   # mutable; /model swaps it, _build_options reads it on reconnect
_turn_task: Optional[asyncio.Task] = None
_last_activity = 0.0
_stderr_ring: list[str] = []
_pending: list[dict] = []
_drain_task: Optional[asyncio.Task] = None
_queued_ack_sent = False    # one ack per busy period — never spam a multi-part send


# ---------------------------------------------------------------------------
# Telegram HTTP (stdlib only; called via asyncio.to_thread from the loop)
# ---------------------------------------------------------------------------

def _telegram_post(method: str, params: dict, timeout: float = 15.0) -> dict:
    url = f"{API_BASE}/bot{TOKEN}/{method}"
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# Reuse recall's markdown->Telegram-HTML renderer so replies get bold/headers/
# code/spoiler/blockquote treatment. Guarded: if the import fails, replies just go
# out as plain text.
try:
    from recall.notify import _md_to_telegram_html as _render_md_html
except Exception:  # noqa: BLE001 — never let a render-path import break the bridge
    _render_md_html = None


def _telegram_get(method: str, params: dict, timeout: float) -> dict:
    qs = urllib.parse.urlencode(params)
    url = f"{API_BASE}/bot{TOKEN}/{method}?{qs}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _telegram_file_path(file_id: str) -> Optional[str]:
    resp = _telegram_get("getFile", {"file_id": file_id}, timeout=15)
    if not resp.get("ok"):
        return None
    return (resp.get("result") or {}).get("file_path")


def _download_telegram_file(file_path: str, dest: Path) -> None:
    url = f"{API_BASE}/file/bot{TOKEN}/{file_path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


# Headroom so the "[i/N] " label can never push a chunk over Telegram's cap (it
# counts length in UTF-16 code units).
PREFIX_RESERVE = 16


def _u16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _hard_wrap(s: str, budget: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for ch in s:
        if cur and _u16_len(cur) + _u16_len(ch) > budget:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _split_for_telegram(text: str, budget: int) -> list[str]:
    """Split into <=budget UTF-16-unit chunks, preferring paragraph then line
    boundaries so sections stay intact; hard-wrap only when a line won't fit."""
    chunks: list[str] = []
    cur = ""
    for para in text.split("\n\n"):
        block = para if not cur else "\n\n" + para
        if _u16_len(cur) + _u16_len(block) <= budget:
            cur += block
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        if _u16_len(para) <= budget:
            cur = para
            continue
        for line in para.split("\n"):
            piece = line if not cur else "\n" + line
            if _u16_len(cur) + _u16_len(piece) <= budget:
                cur += piece
                continue
            if cur:
                chunks.append(cur)
                cur = ""
            if _u16_len(line) <= budget:
                cur = line
            else:
                parts = _hard_wrap(line, budget)
                chunks.extend(parts[:-1])
                cur = parts[-1] if parts else ""
    if cur:
        chunks.append(cur)
    return chunks or [""]


async def send(text: str, chat_id: Optional[int] = None) -> None:
    """Send text to the chat, splitting on Telegram's 4096 UTF-16-unit cap."""
    if not text:
        return
    target = chat_id if chat_id is not None else int(CHAT_ID_RAW)
    chunks = _split_for_telegram(text, MAX_MSG_LEN - PREFIX_RESERVE)
    for i, chunk in enumerate(chunks, start=1):
        prefix = f"[{i}/{len(chunks)}] " if len(chunks) > 1 else ""
        try:
            await _send_one(target, prefix + chunk)
        except Exception as exc:  # noqa: BLE001
            log.error("sendMessage failed (chunk %d/%d): %s", i, len(chunks), exc)


async def _send_one(target: int, text: str) -> None:
    """POST one message rendered as Telegram HTML; fall back to plain text if the
    HTML is rejected (e.g. a chunk split mid-entity) so nothing is dropped."""
    if _render_md_html is not None:
        try:
            html = _render_md_html(text)
        except Exception:  # noqa: BLE001 — bad markdown shouldn't lose the message
            html = None
        if html is not None:
            try:
                await asyncio.to_thread(
                    _telegram_post, "sendMessage",
                    {"chat_id": target, "text": html, "parse_mode": "HTML"}, 15,
                )
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("HTML send rejected, retrying plain: %s", exc)
    await asyncio.to_thread(
        _telegram_post, "sendMessage", {"chat_id": target, "text": text}, 15,
    )


async def send_typing() -> None:
    try:
        await asyncio.to_thread(
            _telegram_post, "sendChatAction",
            {"chat_id": int(CHAT_ID_RAW), "action": "typing"}, 10,
        )
    except Exception:  # noqa: BLE001
        pass


def audit(direction: str, text: str) -> None:
    snippet = text.replace("\n", " ")[:AUDIT_SNIPPET_LEN]
    line = f"{datetime.now(timezone.utc).isoformat()} {direction} {snippet}\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Offset + session persistence
# ---------------------------------------------------------------------------

def read_offset() -> int:
    try:
        return int(OFFSET_FILE.read_text().strip())
    except Exception:  # noqa: BLE001
        return 0


def write_offset(off: int) -> None:
    try:
        OFFSET_FILE.write_text(str(off))
    except Exception as exc:  # noqa: BLE001
        log.error("failed to persist offset: %s", exc)


def seed_offset_if_missing() -> int:
    """On first run, jump past any backlog so we don't replay old messages."""
    if OFFSET_FILE.exists():
        return read_offset()
    try:
        r = _telegram_get("getUpdates", {"offset": -1, "limit": 1, "timeout": 0}, timeout=10)
        updates = r.get("result") or []
        if updates:
            off = updates[-1]["update_id"]
            write_offset(off)
            log.info("seeded offset to %d (skipped backlog)", off)
            return off
    except Exception as exc:  # noqa: BLE001
        log.warning("seed_offset failed (continuing with 0): %s", exc)
    return 0


def _save_session(sid: Optional[str]) -> None:
    global _session_id
    _session_id = sid
    _buf_rekey(sid)   # tier-1 follows the conversation identity (fail-open)
    try:
        if sid:
            SESSION_FILE.write_text(sid)
        else:
            SESSION_FILE.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        log.error("failed to persist session id: %s", exc)


def _load_session() -> Optional[str]:
    try:
        s = SESSION_FILE.read_text().strip()
        return s or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Per-session curation trigger (brick 2) — curate an ended session in the
# background so memory is fresh the same day, without blocking the user.
# ---------------------------------------------------------------------------

def _session_curation_cmd(sid: str) -> list[str]:
    """Argv that curates ONE ended session into its project corpus + the soul,
    reusing the recall CLI from the bridge's own venv against AGENT_CWD. Pure, so
    it can be unit-tested without spawning anything."""
    return [RECALL_BIN, "curate", "--session", sid,
            "--project-dir", str(AGENT_CWD), "--commit"]


async def _reap_curation(sid: str, proc) -> None:
    try:
        rc = await proc.wait()
        log.info("session-curation for %s exited rc=%s", sid[:8], rc)
    except Exception as exc:  # noqa: BLE001
        log.warning("session-curation reap error for %s: %s", sid[:8], exc)


async def _fire_session_curation(sid: Optional[str]) -> None:
    """Fire-and-forget: curate the just-ended session in the background so the
    user is never blocked. Idempotent (the 'sessions' bucket) and re-swept nightly,
    so a spawn failure — or the child being killed on a bridge restart — is
    harmless: the session is simply picked up by the nightly safety-net sweep."""
    if not sid:
        return
    cmd = _session_curation_cmd(sid)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:  # noqa: BLE001 — never let curation break session teardown
        log.warning("session-curation spawn failed for %s: %s", sid[:8], exc)
        return
    log.info("session-curation fired for %s (pid %s)", sid[:8], proc.pid)
    asyncio.create_task(_reap_curation(sid, proc))


# ---------------------------------------------------------------------------
# Claude Agent SDK conversation manager
# ---------------------------------------------------------------------------

def _collect_stderr(line: str) -> None:
    _stderr_ring.append(line)
    if len(_stderr_ring) > 80:
        del _stderr_ring[:-80]


async def _recall_inject_hook(input_data, tool_use_id, context):  # noqa: ARG001
    """UserPromptSubmit hook: run the same recall_inject.py a terminal install
    uses and pass its JSON through (additionalContext -> silent model context).
    The script is already fail-open; this wrapper fail-opens too ({} = inject
    nothing), so a recall hiccup can never block a phone turn. The
    operator-visible systemMessage has no Telegram surface — log it instead so
    injection stays observable."""
    if not RECALL_INJECT:
        return {}
    try:
        env = dict(os.environ, CLAUDE_PROJECT_DIR=str(AGENT_CWD))
        proc = await asyncio.create_subprocess_exec(
            sys.executable, RECALL_INJECT,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=env)
        out, _ = await asyncio.wait_for(
            proc.communicate(json.dumps(input_data).encode()),
            timeout=INJECT_TIMEOUT)
        payload = json.loads(out) if out.strip() else {}
    except Exception as exc:  # noqa: BLE001 — recall must never block a turn
        log.warning("recall inject hook failed (fail-open): %s", exc)
        return {}
    if payload.get("systemMessage"):
        log.info("%s", payload["systemMessage"])
    return payload


def _build_options(resume_id: Optional[str]) -> ClaudeAgentOptions:
    opts = dict(
        system_prompt={"type": "preset", "preset": "claude_code", "append": PERSONA},
        setting_sources=SETTING_SOURCES,
        permission_mode="bypassPermissions",  # async channel: no human to click approve
        cwd=str(AGENT_CWD),
        cli_path=CLAUDE_BIN,
        resume=resume_id,
        stderr=_collect_stderr,
        # One base64 image in a tool result (e.g. Read on a Telegram photo) must
        # not overflow the SDK's 1 MiB stdio line and kill the bridge — same
        # crash class the harness fixed in core.py (CLI_MAX_BUFFER_SIZE; 64 MiB
        # default, ENGRAM_CLI_MAX_BUFFER_MB tunes).
        max_buffer_size=int(float(os.environ.get(
            "ENGRAM_CLI_MAX_BUFFER_MB", "64")) * 1024 * 1024),
    )
    if RECALL_INJECT:
        opts["hooks"] = {"UserPromptSubmit": [HookMatcher(hooks=[_recall_inject_hook])]}
    if AGENT_EFFORT:
        opts["effort"] = AGENT_EFFORT
    if _current_model:
        opts["model"] = _current_model
    if ALLOWED_TOOLS:
        opts["allowed_tools"] = ALLOWED_TOOLS
    if DISALLOWED_TOOLS:
        opts["disallowed_tools"] = DISALLOWED_TOOLS
    return ClaudeAgentOptions(**opts)


async def _ensure_client() -> None:
    global _client
    if _client is not None:
        return
    _client = ClaudeSDKClient(options=_build_options(_session_id))
    await _client.connect()
    log.info("client connected (resume=%s)", _session_id or "fresh")


async def _disconnect_client() -> None:
    global _client
    if _client is None:
        return
    try:
        await _client.disconnect()
    except Exception as exc:  # noqa: BLE001
        log.warning("client disconnect error: %s", exc)
    finally:
        _client = None


async def _query_once(text: str) -> str:
    assert _client is not None
    parts: list[str] = []
    sid: Optional[str] = None
    await _client.query(text)
    async for msg in _client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
        elif isinstance(msg, SystemMessage):
            data = getattr(msg, "data", {}) or {}
            sid = sid or data.get("session_id")
        elif isinstance(msg, ResultMessage):
            sid = getattr(msg, "session_id", None) or sid
    if sid:
        _save_session(sid)
    return "".join(parts).strip()


async def run_turn(text: str) -> str:
    """Run a turn, with a one-shot fresh fallback if a resume went stale."""
    resuming = _client is None and _session_id is not None
    await _ensure_client()
    try:
        return await _query_once(text)
    except Exception as exc:  # noqa: BLE001
        if resuming:
            log.warning("resume turn failed (%s); retrying as a fresh session", exc)
            await _disconnect_client()
            _save_session(None)
            await _ensure_client()
            out = await _query_once(text)
            return ("(couldn't resume the previous thread — started a fresh one)\n\n" + out)
        raise


async def _set_model(name: str) -> None:
    """Switch the model for subsequent turns. Recycles the warm client so the next
    turn reconnects with the new model, keeping the session id so the SAME
    conversation continues — mirrors core.AgentSDKDriver.set_model. (The recycle is
    the mechanism: a live model swap on a connected client isn't reliable, so we drop
    the client and let run_turn's resume path rebuild it under the new model.)"""
    global _current_model
    _current_model = name
    await _disconnect_client()
    await send(f"🔀 model → **{name}** — same conversation continues "
               "(resumes on your next message)")


# ---------------------------------------------------------------------------
# Command + message handling
# ---------------------------------------------------------------------------

def _busy() -> bool:
    return _turn_task is not None and not _turn_task.done()


def _clear_pending() -> None:
    global _pending, _drain_task, _queued_ack_sent
    _pending = []
    _queued_ack_sent = False
    if _drain_task is not None and not _drain_task.done():
        _drain_task.cancel()
    _drain_task = None


def _arm_drain() -> None:
    global _drain_task
    if _drain_task is not None and not _drain_task.done():
        _drain_task.cancel()
    _drain_task = asyncio.create_task(_drain_after_delay())


async def _drain_after_delay() -> None:
    try:
        await asyncio.sleep(COALESCE_WINDOW)
        while _busy():            # let any in-flight turn finish first
            await asyncio.sleep(0.3)
    except asyncio.CancelledError:
        return
    await _flush_pending()


async def _flush_pending() -> None:
    """Combine all buffered fragments into one prompt and start a single turn."""
    global _turn_task, _pending, _queued_ack_sent
    if not _pending or _busy():
        return
    frags, _pending = _pending, []
    _queued_ack_sent = False          # next busy period gets its own (single) ack
    texts = [f["text"] for f in frags if f["text"]]
    paths = [p for f in frags for p in f["paths"]]
    prompt = "\n\n".join(texts)
    if paths:
        listing = "\n".join(f"- {p}" for p in paths)
        note = (
            f"[The operator sent {len(paths)} image/file(s) over Telegram across "
            f"{len(frags)} message(s), now saved locally. Use your Read tool to open "
            "each path below (Read renders images visually), then respond:]\n" + listing
        )
        prompt = f"{prompt}\n\n{note}" if prompt else note
    log.info("flushing %d fragment(s), %d file(s)", len(frags), len(paths))
    _turn_task = asyncio.create_task(_process_turn(prompt))


def _matches_cmd(text: str, name: str) -> bool:
    if text == f"/{name}":
        return True
    if BOT_USERNAME and text == f"/{name}@{BOT_USERNAME}":
        return True
    return False


def _cmd_arg(text: str, name: str) -> Optional[str]:
    """Like _matches_cmd but for a command that takes an argument: return the arg
    string (possibly '') if `text` invokes /name (optionally @bot), else None.
    '/model' -> '', '/model fable' -> 'fable', '/status' -> None."""
    heads = [f"/{name}"]
    if BOT_USERNAME:
        heads.append(f"/{name}@{BOT_USERNAME}")
    for head in heads:
        if text == head:
            return ""
        if text.startswith(head + " "):
            return text[len(head) + 1:].strip()
    return None


async def _process_turn(text: str) -> None:
    global _last_activity
    _last_activity = time.monotonic()
    await send_typing()
    try:
        reply = await asyncio.wait_for(run_turn(text), timeout=TURN_TIMEOUT)
    except asyncio.CancelledError:
        await send("✋ cancelled")
        raise
    except asyncio.TimeoutError:
        await send("⏱️ that turn ran past the time limit — try /cancel then resend, or /new")
        return
    except Exception as exc:  # noqa: BLE001
        log.exception("turn failed: %s", exc)
        tail = "\n".join(_stderr_ring[-6:])
        await send(f"⚠️ error: {type(exc).__name__}: {exc}" + (f"\n\n{tail}" if tail else ""))
        return
    finally:
        _last_activity = time.monotonic()
    audit("OUT", reply)
    _buffer.append("assistant", reply)
    await send(reply or "(no text in reply)")


async def _ingest_attachments(msg: dict) -> list[str]:
    """Download any photo or document on this message into INBOX_DIR."""
    targets: list[tuple[str, str, Optional[str]]] = []
    photos = msg.get("photo") or []
    if photos:
        big = photos[-1]
        targets.append((big.get("file_id"), big.get("file_unique_id"), None))
    doc = msg.get("document")
    if doc and doc.get("file_id"):
        targets.append((doc["file_id"], doc.get("file_unique_id"), doc.get("file_name")))

    paths: list[str] = []
    for file_id, unique_id, name in targets:
        if not file_id:
            continue
        try:
            fpath = await asyncio.to_thread(_telegram_file_path, file_id)
            if not fpath:
                log.warning("getFile returned no path for %s", file_id)
                continue
            suffix = Path(name or fpath).suffix or ".bin"
            dest = INBOX_DIR / f"{unique_id or file_id}{suffix}"
            await asyncio.to_thread(_download_telegram_file, fpath, dest)
            paths.append(str(dest))
            log.info("downloaded attachment -> %s", dest)
        except Exception as exc:  # noqa: BLE001 — one bad file shouldn't drop the msg
            log.warning("attachment download failed (%s): %s", file_id, exc)
    return paths


async def handle_message(update: dict) -> None:
    global _turn_task
    msg = update.get("message") or update.get("edited_message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    text = (msg.get("text") or "").strip()
    caption = (msg.get("caption") or "").strip()
    has_media = bool(msg.get("photo") or msg.get("document"))
    if chat_id is None:
        return
    if chat_id != int(CHAT_ID_RAW):
        log.warning("reject chat_id=%r (not allowed)", chat_id)
        audit("REJECT", f"chat_id={chat_id} text={text[:80]}")
        return
    if not text and not has_media:
        return

    audit("IN", text or f"[media] {caption}")
    _buffer.append("user", text or f"[media] {caption}")
    log.info("inbound: %r media=%s", (text or caption)[:160], has_media)

    if _matches_cmd(text, "start") or _matches_cmd(text, "help"):
        await send(
            "👋 Hey, it's Engram. What are we thinking about today?\n\n"
            "  /new    — end this conversation and start fresh\n"
            "  /end    — end this conversation\n"
            "  /cancel — stop the current reply\n"
            "  /status — locked? busy? model? active session\n"
            "  /model  — switch model (opus[1m] · opus · fable · …)\n"
            "  /lock /unlock — pause/resume inbound\n"
            "  /ping   — health check"
        )
        return
    if _matches_cmd(text, "ping"):
        await send("pong")
        return
    if _matches_cmd(text, "status"):
        await send(
            f"recall chat ok | locked={LOCK_FILE.exists()} | busy={_busy()} | "
            f"model={_current_model} | "
            f"session={'yes' if (_session_id or _client) else 'none'} | "
            f"{BOT_LABEL} | cwd={AGENT_CWD}"
        )
        return
    model_arg = _cmd_arg(text, "model")
    if model_arg is not None:
        if not model_arg:
            listing = "\n".join(f"  `{n}` — {d}" for n, d in MODELS)
            await send(f"current model: **{_current_model}**\n\n{listing}\n\n"
                       "usage: /model <name>  — e.g. `/model fable` or `/model opus[1m]`")
            return
        if _busy():
            await send("⏳ a reply's in flight — /cancel first, then switch model")
            return
        await _set_model(model_arg)
        return
    if _matches_cmd(text, "lock"):
        LOCK_FILE.write_text(datetime.now(timezone.utc).isoformat())
        await send("🔒 LOCKED — incoming messages ignored until /unlock")
        return
    if _matches_cmd(text, "unlock"):
        LOCK_FILE.unlink(missing_ok=True)
        await send("🔓 UNLOCKED")
        return
    if _matches_cmd(text, "cancel"):
        had_pending = bool(_pending)
        _clear_pending()
        if _busy():
            try:
                if _client is not None:
                    await _client.interrupt()
            except Exception as exc:  # noqa: BLE001
                log.warning("interrupt error: %s", exc)
            if _turn_task is not None:
                _turn_task.cancel()
            await send("✋ cancelling…")
        elif had_pending:
            await send("✋ dropped the queued message(s)")
        else:
            await send("(nothing running)")
        return
    if _matches_cmd(text, "new") or _matches_cmd(text, "end"):
        ending_sid = _session_id          # capture the ending session before we clear it
        _clear_pending()
        if _busy() and _turn_task is not None:
            _turn_task.cancel()
        await _disconnect_client()
        _save_session(None)
        await _fire_session_curation(ending_sid)   # brick 2: curate it in the background
        await send("🆕 new conversation — context cleared. Go ahead."
                   if _matches_cmd(text, "new")
                   else "👋 conversation ended. Send anything to start a fresh one.")
        return

    if LOCK_FILE.exists():
        await send("🔒 bridge is locked — /unlock first")
        return

    if not _busy():
        await send_typing()
    paths = await _ingest_attachments(msg) if has_media else []
    frag_text = text or caption
    if not frag_text and not paths:
        await send("⚠️ I got an attachment but couldn't download it — try resending.")
        return
    _pending.append({"text": frag_text, "paths": paths})
    if _busy():
        # The one-message-behind confusion (operator report, 2026-07-07): a message that lands
        # mid-turn waits SILENTLY for the whole in-flight turn (minutes on a build
        # turn), then the old turn's reply arrives looking like the answer to the
        # new message. Say so, once — and don't show "typing…" for a reply that
        # isn't about this message.
        global _queued_ack_sent
        if not _queued_ack_sent:
            _queued_ack_sent = True
            await send("⏳ still writing the reply to your previous message — "
                       "got this one, it's queued next (the reply above/next is "
                       "for the earlier message).")
    _arm_drain()


# ---------------------------------------------------------------------------
# Idle watcher — release the warm client after inactivity (resume on next msg)
# ---------------------------------------------------------------------------

async def _idle_watch() -> None:
    while True:
        await asyncio.sleep(60)
        if _client is not None and not _busy() and IDLE_SECS > 0:
            if time.monotonic() - _last_activity > IDLE_SECS:
                log.info("idle %ds — releasing warm client (session kept for resume)", IDLE_SECS)
                await _disconnect_client()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def fetch_bot_username() -> str:
    try:
        r = _telegram_get("getMe", {}, timeout=10)
        return (r.get("result") or {}).get("username", "") or ""
    except Exception as exc:  # noqa: BLE001
        log.warning("getMe failed (continuing without bot_username): %s", exc)
        return ""


async def main() -> int:
    global BOT_USERNAME, _session_id, _last_activity

    if not TOKEN or not CHAT_ID_RAW:
        log.error("RECALL_TELEGRAM_AGENT_TOKEN and RECALL_TELEGRAM_AGENT_CHAT_ID "
                  "must both be set")
        return 2
    try:
        int(CHAT_ID_RAW)
    except ValueError:
        log.error("RECALL_TELEGRAM_AGENT_CHAT_ID must be numeric, got %r", CHAT_ID_RAW)
        return 2

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    if _STRIPPED_API_KEY:
        log.info("stripped ANTHROPIC_API_KEY from env — forcing subscription auth")

    BOT_USERNAME = await asyncio.to_thread(fetch_bot_username)
    _session_id = _load_session()
    _buf_rekey(_session_id)   # boot resume: tier-1 continues under the saved sid
    _last_activity = time.monotonic()
    log.info("agent bridge starting; label=%s bot=@%s allowed_chat=%s cwd=%s state=%s "
             "setting_sources=%s allowed_tools=%s resume=%s",
             BOT_LABEL, BOT_USERNAME, CHAT_ID_RAW, AGENT_CWD, STATE_DIR,
             SETTING_SOURCES, ALLOWED_TOOLS, _session_id or "none")

    offset = await asyncio.to_thread(seed_offset_if_missing)
    try:
        await send(f"✅ Engram online — @{BOT_USERNAME} ready. Just talk to me. "
                   f"(/new /end /status /model /cancel /lock)")
    except Exception:  # noqa: BLE001
        pass

    asyncio.create_task(_idle_watch())

    while True:
        try:
            r = await asyncio.to_thread(
                _telegram_get, "getUpdates",
                {"offset": offset + 1, "timeout": POLL_TIMEOUT,
                 "allowed_updates": json.dumps(["message"])},
                HTTP_BUDGET,
            )
            for upd in r.get("result", []):
                offset = upd["update_id"]
                try:
                    await handle_message(upd)
                except Exception as exc:  # noqa: BLE001
                    log.exception("handle failed: %s", exc)
                write_offset(offset)
        except asyncio.CancelledError:
            log.info("cancelled; exit")
            return 0
        except Exception as exc:  # noqa: BLE001
            log.error("poll loop error: %s; sleeping 5s", exc)
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
