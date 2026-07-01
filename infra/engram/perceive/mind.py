#!/usr/bin/env python3
"""Engram's mind, woken by perception — the bridge from the gate to ``core.py``.

Step 3 of the perceiving loop: when the engagement gate (``loop.py``) decides
the operator is present, this wakes Engram's actual mind — the SAME ``core.AgentSDKDriver``
the terminal TUI and the Telegram bridge run on — so the loop becomes a THIRD
front-end on the one brain, driven by what the camera sees instead of typed input.

The seam is ``PerceivingMind.on_event``, which you hand to the loop as its
``on_event`` callback. The loop ticks synchronously (OpenCV); the mind runs its
own asyncio event loop on a background thread, and ``on_event`` (called from the
loop thread) schedules a wake onto it via ``run_coroutine_threadsafe``. So the
camera never blocks on the model, and the model never blocks the camera.

Discipline that keeps an always-on camera from spamming a real model:
  * wake only on ENGAGE, once per presence episode (reset when they leave);
  * a cooldown so flickering in/out of frame can't re-greet repeatedly;
  * a busy-guard so a second turn never starts while one is still streaming
    (the persistent SDK client holds one conversation — concurrent queries would
    corrupt the stream);
  * its OWN non-persisted session (``store=None``), so camera greetings never
    land in the terminal Engram thread, and low effort for a snappy greeting.

``on_reply(text)`` surfaces the greeting text; today it prints.
"""
from __future__ import annotations

import asyncio
import getpass
import os
import sys
import threading
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for core

# Re-frames the terminal-flavored base persona for embodied perception. Kept
# conversation-only on purpose: this fires autonomously, so the first cut greets
# and does not act. (A later step can hand the perceiving front-end a sandboxed,
# Read-only tool set if Engram should ever act on what it sees.)
PERCEPTION_NOTE = (
    "\n\n--- Right now you are NOT in the terminal. You perceive the person present "
    "through your eye: a camera, with a face model that recognizes who is present and a "
    "vision model that reads the scene. This turn was triggered by your perceiving loop — "
    "by SEEING them come into view — not by anything they typed. Respond with a brief, "
    "natural greeting (one or two sentences). Do NOT use tools or take actions; just greet "
    "them."
)


class PerceivingMind:
    def __init__(self, driver=None, *, target: Optional[str] = None, eye_enabled: bool = True,
                 greet_cooldown: float = 30.0, effort: str = "low",
                 model: Optional[str] = None,
                 on_reply: Optional[Callable[[str], None]] = None,
                 gate: Optional[Callable[[], bool]] = None) -> None:
        self.target = target or os.environ.get("ENGRAM_USER") or getpass.getuser()
        self.eye_enabled = eye_enabled
        self.greet_cooldown = greet_cooldown
        self.on_reply = on_reply
        # gate(): the mind only ACTS (greets / replies) when this returns True. Used to
        # bind autonomy to "the harness is open AND it sees the operator" — so Engram never
        # fires a turn unattended. None = always act (e.g. supervised dev runs).
        self.gate = gate
        if driver is None:
            from core import PERSONA, AgentSDKDriver  # lazy: keeps the gate SDK-free
            kwargs = dict(persona=PERSONA + PERCEPTION_NOTE, effort=effort, store=None)
            if model:
                kwargs["model"] = model
            driver = AgentSDKDriver(**kwargs)
        self.driver = driver
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._busy = False
        self._greeted_episode = False
        self._last_greet = 0.0

    # ---- lifecycle: an asyncio loop + a pre-warmed model client on a bg thread
    def start(self, connect_timeout: float = 30.0) -> "PerceivingMind":
        self._thread = threading.Thread(target=self._run, name="engram-mind", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=connect_timeout):
            print("  (mind: model client did not connect in time — greetings disabled)")
        return self

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self.driver.connect())
            self._ready.set()
            self._loop.run_forever()
        except Exception as exc:   # noqa: BLE001 — surface, don't crash the process
            print(f"  (mind thread error: {type(exc).__name__}: {exc})")
        finally:
            self._ready.set()

    def stop(self, drain_timeout: float = 20.0) -> None:
        # let any in-flight greeting finish so we don't cut Engram off mid-sentence
        deadline = time.time() + drain_timeout
        while self._busy and time.time() < deadline:
            time.sleep(0.1)
        if self._loop is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(self.driver.disconnect(), self._loop)
                fut.result(timeout=5)
            except Exception:   # noqa: BLE001
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def busy(self) -> bool:
        return self._busy

    # ---- the seam: hand THIS to PerceiveLoop(on_event=...) --------------------
    def on_event(self, ev) -> None:
        """Called from the perceiving-loop thread for every gate decision."""
        if ev.kind == "engage":
            self._greeted_episode = False            # a new presence episode
            if not self.eye_enabled:
                self._wake(reading=None)             # no eye to wait for — greet now
        elif ev.kind == "eye" and not self._greeted_episode:
            self._wake(reading=ev.data.get("reading"))   # greet on the first scene read
        elif ev.kind in ("idle", "passive"):
            self._greeted_episode = False            # episode over; allow a fresh greet

    def _wake(self, reading: Optional[str]) -> None:
        now = time.time()
        if self._loop is None or not self._ready.is_set():
            return
        if self.gate is not None and not self.gate():    # dormant: harness closed / not seen
            return
        if self._busy or self._greeted_episode or (now - self._last_greet) < self.greet_cooldown:
            return
        self._greeted_episode = True                 # guard set synchronously (loop thread)
        self._last_greet = now
        asyncio.run_coroutine_threadsafe(self._turn(self._prompt(reading)), self._loop)

    def _prompt(self, reading: Optional[str]) -> str:
        if reading:
            return (f"[perception] {self.target} just came into view. Through your eye you "
                    f"see: \"{reading}\". Greet them.")
        return f"[perception] {self.target} just came into view. Greet them."

    async def _turn(self, prompt: str) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            print("  ◇ Engram is waking …")
            parts: list[str] = []
            async for ev in self.driver.query(prompt):
                if ev.kind == "text":
                    parts.append(ev.text)
            reply = "".join(parts).strip()
            if reply:
                print(f"  ◆ Engram → {reply}")
                if self.on_reply is not None:
                    self.on_reply(reply)
        except Exception as exc:   # noqa: BLE001 — a failed turn must not kill perception
            print(f"  (mind error: {type(exc).__name__}: {exc})")
        finally:
            self._busy = False
