#!/usr/bin/env python3
"""PerceptMemory — step 5 of the perceiving loop: perception events become memory.

Until now the loop's events lived only in a ``deque(maxlen=500)`` — working
memory that died with the process. This module gives perception the SAME
three-tier stack conversations got in Brick 3: gate-worthy events append to an
immutable LiveBuffer JSONL (tier 1, one file per LOCAL day), and once enough of
the tail has cooled past the hot window a detached ``recall curate --buffer``
folds it into the LTM corpus and advances the per-file watermark (tier 3).
Rows carry ``role: "perception"`` — the engine renders them as their own
``### PERCEPTION`` blocks, never masquerading as human turns.

THE GATE IS THE POINT (clean-perception doctrine): a VLM's free-text is the
confabulation vector, and surprise buys permanence downstream — so only
corroboration-STABLE eye readings are eviction-eligible (this finally consumes
the ``data['stable']`` signal B.2 built for exactly this step), and only when
the corroborated scene actually CHANGED, so an hour of "desk, laptop" is one
row, not 450. Presence transitions (engage/passive/idle) ride the ArcFace
gate and persist as-is; directed speech (``heard``) is wake-word-gated
upstream; ``overheard``/``ambient``/unknown kinds are dropped — fail-closed
on novelty.

Wiring: ``loop = PerceiveLoop(..., on_event=percept.wrap(mind.on_event))``.
Everything here is FAIL-OPEN and cheap — perception memory is a passenger,
never the driver; a broken disk must not blind the camera.

Cutouts: ``ENGRAM_PERCEPT=0`` disables persistence entirely; ``ENGRAM_EVICT=0``
keeps the buffer but stops mid-flight curation (the file remains for a manual
``recall curate --buffer`` sweep).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for buffer

from buffer import LiveBuffer, read_buffer_watermark  # noqa: E402

DATA_ROOT = Path(os.environ.get("RECALL_DATA_ROOT",
                                os.path.expanduser("~/.local/share/recall")))
PERCEPT_ON = os.environ.get("ENGRAM_PERCEPT", "1") != "0"
PERCEPT_DIR = Path(os.environ.get("ENGRAM_PERCEPT_DIR",
                                  str(DATA_ROOT / "engram" / "percept")))
# Perception rows are short (~100 chars); 6000 cooled chars ≈ half a day of
# transitions + scene changes — big enough to be worth a curator pass.
EVICT_CHARS = int(os.environ.get("ENGRAM_PERCEPT_EVICT_CHARS", "6000"))
# The newest rows stay out of eviction — a presence episode still unfolding
# shouldn't be frozen into LTM mid-flight.
HOT_ROWS = int(os.environ.get("ENGRAM_PERCEPT_HOT_ROWS", "6"))
EVICT_ON = os.environ.get("ENGRAM_EVICT", "1") != "0"
REPO = Path(os.environ.get("RECALL_REPO") or Path(__file__).resolve().parents[3])

# Event kinds that persist as-is: rare, transition-shaped, already gated
# upstream (ArcFace presence / wake word). Everything else needs the eye rule.
_PERSIST_KINDS = frozenset({"start", "stop", "engage", "passive", "idle",
                            "heard"})


class PerceptMemory:
    """Listener that persists gate-worthy perception events and evicts the
    cooled tail into curation. Duck-types the loop's ``Event`` (reads only
    ``.t``/``.kind``/``.detail``/``.data``) so importing this never drags in
    cv2. Thread-safe: the vision loop and the hearing gate fire ``on_event``
    from different threads, so one lock serializes gate→append→evict."""

    def __init__(self, dir_: Optional[Path] = None, *,
                 enabled: Optional[bool] = None,
                 evict_chars: int = EVICT_CHARS, hot_rows: int = HOT_ROWS,
                 evict_on: Optional[bool] = None, cwd: Path = REPO,
                 spawn: Optional[Callable] = None) -> None:
        self.enabled = PERCEPT_ON if enabled is None else enabled
        self.evict_chars = evict_chars
        self.hot_rows = hot_rows
        self.evict_on = EVICT_ON if evict_on is None else evict_on
        self.cwd = cwd
        self._spawn_fn = spawn
        self._day: Optional[str] = None      # local-date key of the open file
        self._last_terms: Optional[list] = None  # last PERSISTED stable scene
        self._proc = None                    # one detached curate at a time
        self._lock = threading.Lock()        # vision + hearing threads
        self._buffer = LiveBuffer(
            (Path(dir_) if dir_ is not None else PERCEPT_DIR)
            if self.enabled else None,
            lambda: f"percept-{self._day}")

    # ---- the seam: hand wrap(mind.on_event) to PerceiveLoop/Hearing ---------

    def wrap(self, next_cb: Optional[Callable] = None) -> Callable:
        """Compose: persist first (fail-open), then forward to ``next_cb``
        (the mind) — so a memory hiccup can never eat a greeting."""
        def _cb(ev) -> None:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001 — passenger, never the driver
                pass
            if next_cb is not None:
                next_cb(ev)
        return _cb

    def on_event(self, ev) -> None:
        """Gate → append → (rollover) → size-gated eviction. Fail-open."""
        if not self.enabled or not self._buffer.enabled:
            return
        try:
            with self._lock:
                self._roll_day(ev)
                row = self._gate(ev)
                if row is not None:
                    text, extra = row
                    self._buffer.append("perception", text, extra)
                    self._maybe_evict()
        except Exception:  # noqa: BLE001
            pass

    # ---- gate (clean-perception: only corroborated reality persists) --------

    def _gate(self, ev) -> Optional[tuple[str, dict]]:
        kind = getattr(ev, "kind", None)
        detail = str(getattr(ev, "detail", "") or "")
        data = getattr(ev, "data", None) or {}
        if kind == "eye":
            if not data.get("stable"):
                return None                  # unconfirmed read → never memory
            terms = sorted(data.get("corroborated") or [])
            if terms == self._last_terms:
                return None                  # same stable scene → one row only
            self._last_terms = terms
        elif kind not in _PERSIST_KINDS:
            return None                      # fail-closed on unknown kinds
        return (f"[{kind}] {detail}"[:2000],
                {"kind": kind, "data": data})

    # ---- day files (perception is continuous; days are the natural unit) ----

    def _roll_day(self, ev) -> None:
        key = time.strftime("%Y-%m-%d",
                            time.localtime(getattr(ev, "t", None) or time.time()))
        if self._day is None:
            self._day = key
            self._buffer.reseed()
            return
        if key == self._day:
            return
        old_path = self._buffer.path()       # still keyed to the closing day
        self._day = key
        self._buffer.reseed()
        self._last_terms = None              # a new day corroborates afresh
        if old_path is not None:
            self._spawn(old_path, until=None)   # flush yesterday's whole tail

    # ---- eviction-is-curation (same shape as core's A6, percept-tuned) ------

    def _maybe_evict(self) -> None:
        if not self.evict_on:
            return
        if self._proc is not None and self._proc.poll() is None:
            return                            # one detached curate at a time
        path = self._buffer.path()
        if path is None:
            return
        mark = read_buffer_watermark(self.cwd, path.stem)
        after = self._buffer.tail_after(mark)
        if len(after) <= self.hot_rows:
            return
        cooled = after[:-self.hot_rows] if self.hot_rows > 0 else after
        chars = sum(len(str(r.get("text") or "")) for r in cooled)
        until = str(cooled[-1].get("ts") or "")
        if chars < self.evict_chars or not until:
            return
        self._spawn(path, until=until)

    def flush(self) -> None:
        """Teardown flush (the service's finally block): fold the whole
        un-curated tail — there is no 'later' in this process. Cheap when the
        tail is empty (the curate CLI skips on no_new_exchanges)."""
        try:
            if not self.enabled or not self.evict_on:
                return
            path = self._buffer.path()
            if path is not None and self._day is not None:
                self._spawn(path, until=None)
        except Exception:  # noqa: BLE001
            pass

    def _spawn(self, path: Path, *, until: Optional[str]) -> None:
        try:
            fn = self._spawn_fn
            if fn is None:
                from core import spawn_buffer_curate  # lazy: SDK-heavy module
                fn = spawn_buffer_curate
            proc = fn(path, self.cwd, until=until, provisional=True)
            if proc is not None:
                self._proc = proc
        except Exception:  # noqa: BLE001 — eviction must never break the loop
            pass
