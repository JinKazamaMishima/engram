#!/usr/bin/env python3
"""Engram's perceiving loop — the engagement gate (roadmap track 2, step 2).

Built on the Sensorium (ONE capture). Each tick runs the CHEAP always-on sense —
face-ID — over the latest frame to decide WHO is present, then a small state
machine gates what Engram does:

    nobody                 -> idle     (Engram rests)
    the known person (me)  -> engage   (greet + read the scene with the eye)
    anyone else / stranger -> passive  (logged once: "not engaging")

This is the stated payoff: "recognize if it's me ... decide if you want to
help or not." The expensive eye (SmolVLM) is THROTTLED — it runs on the engage
transition and then every few seconds, never every frame. Every decision is
appended to a bounded in-memory event log = Engram's working memory, the substrate
a later step evicts into recall (the nightly-dream long-term store).

The mind is not wired yet: the optional ``on_event`` callback is the seam where a
later step drives ``core.AgentSDKDriver`` so this loop becomes a THIRD front-end
on the same Engram brain as the TUI and Telegram — woken by perception, not typing.

    # watch the gate decide for 30s, headless (walk in / out of frame):
    .venv/bin/python infra/engram/perceive/loop.py --seconds 30
    .venv/bin/python infra/engram/perceive/loop.py --gui          # with a window
    .venv/bin/python infra/engram/perceive/loop.py --no-eye       # gate only, no VLM
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # for `import sensorium`
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "eye"))  # sibling senses
from eye import Eye  # noqa: E402
from face import AMBER, FONT, GREEN, FaceID, draw_faces  # noqa: E402
from sensorium import Sensorium  # noqa: E402

# When Engram engages, this is what the eye is asked — a greet-context scene read,
# not "describe everything" (the small VLM answers a specific question far better).
# Prompt discipline (clean-perception plan, Track B.4): tell the VLM to describe only
# what it can clearly see and to admit uncertainty instead of guessing. This is a
# FLOOR, not the main defense — VLMs are miscalibrated, so the corroboration filter
# below (B.2) is what actually keeps a one-off misread ("head-on-hand" → "coffee cup")
# out of memory. See [[engram-clean-perception-doctrine]].
ENGAGE_PROMPT = ("Look at this webcam frame. Describe ONLY what you can clearly see — "
                 "who is here and what they are actually doing. If you are unsure what "
                 "an object or action is, say so instead of guessing. One or two sentences.")

# Temporal-corroboration filter (Track B.2). The eye's free text is an *impression*,
# not an observation; a real object persists across the throttled ~8s reads while a
# misread does not. We tag each reading with the scene terms that RECUR across the
# recent reads (≥ CORROBORATE_MIN of the last CORROBORATE_WINDOW) and a ``stable``
# flag. Nothing is promoted to long-term memory HERE — that is step-5 eviction
# (``percept.PerceptMemory``), whose gate requires this signal BEFORE recall's
# surprise scoring, so a hallucinated (hence surprising) claim can't buy permanence.
CORROBORATE_WINDOW = 4     # look back over the last N eye reads (~30s at the 8s cadence)
CORROBORATE_MIN = 3        # a scene term must recur in ≥ this many of them to be trusted

# Generic / prompt-echo words carry no scene evidence — corroborating them would mark
# every read "stable". We only track the specific, confabulation-prone content terms
# (objects, actions) whose recurrence is real evidence.
_STOP = {
    "the", "and", "are", "for", "with", "this", "that", "they", "them", "their",
    "have", "has", "had", "was", "were", "who", "what", "here", "there", "into",
    "from", "near", "seem", "seems", "look", "looks", "looking", "appear",
    "appears", "one", "two", "sentence", "sentences", "briefly", "webcam", "frame",
    "image", "photo", "picture", "person", "people", "someone", "man", "woman",
    "being", "doing", "front", "side", "behind", "wearing", "clearly", "unsure",
    "actually", "something", "sitting", "you", "your", "can", "see",
    "his", "her", "him", "hers", "its",
}


def _content_terms(text: str) -> set[str]:
    """Salient scene words in a reading — lowercase alpha tokens (≥3 chars) minus the
    generic/prompt-echo stopwords. These are what the corroboration filter counts."""
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


@dataclass
class Event:
    """One decision of the loop = one unit of working memory. ``detail`` is the
    human-readable log line; ``data`` is the structured payload listeners read
    (the mind in step 3, recall eviction in step 5) instead of parsing strings."""
    t: float
    kind: str          # start | engage | passive | idle | eye | stop
    detail: str
    data: dict = field(default_factory=dict)

    def stamp(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.t))


class PerceiveLoop:
    """The cheap-tier gate: face-ID every tick -> presence -> a 3-state machine
    (idle / engaged / passive); the expensive eye runs only while engaged, throttled.

    ``on_event(Event)`` fires for every logged decision — the seam a later step
    uses to wake ``core.AgentSDKDriver`` (drive the real mind from perception)."""

    def __init__(self, sensorium: Sensorium, faceid: FaceID, eye: Eye, *,
                 target: str = os.environ.get("ENGRAM_USER") or getpass.getuser(), eye_interval: float = 8.0, hold: float = 1.5,
                 use_eye: bool = True, on_event: Optional[Callable[[Event], None]] = None,
                 log_maxlen: int = 500) -> None:
        self.sensorium = sensorium
        self.faceid = faceid
        self.eye = eye
        self.target = target
        self.eye_interval = eye_interval
        self.hold = hold                       # presence debounce window (seconds)
        self.on_event = on_event
        self.events: deque[Event] = deque(maxlen=log_maxlen)   # working memory
        self.state = "idle"
        self.present: set[str] = set()           # who's in frame now (debounced) — for the HUD
        self.faces: list = []                    # last identified Face list (name + cos) — for the HUD
        self._last_seen: dict[str, float] = {}   # name -> last detection time
        self._last_eye = 0.0
        self._last_reading: Optional[str] = None
        # rolling window of recent readings' content-term sets, for corroboration (B.2)
        self._reading_terms: deque[set[str]] = deque(maxlen=CORROBORATE_WINDOW)
        # the eye is optional and degrades: only attempt it if asked AND a backend
        # answers, so the gate still works headless with no llama-server.
        self._eye_ok = use_eye and eye.health()

    # ---- presence (debounced so a single missed detection doesn't churn) -----
    def _present(self, faces, now: float) -> set[str]:
        for f in faces:
            self._last_seen[f.name] = now
        return {n for n, t in self._last_seen.items() if now - t <= self.hold}

    # ---- event log = working memory ------------------------------------------
    def _emit(self, kind: str, detail: str, data: Optional[dict] = None) -> Event:
        ev = Event(time.time(), kind, detail, data or {})
        self.events.append(ev)
        print(f"  {ev.stamp()}  {kind:<8} {detail}")
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception as exc:   # noqa: BLE001 — a bad listener must not kill perception
                print(f"  (on_event raised {type(exc).__name__}: {exc})")
        return ev

    # ---- one tick: cheap sense -> gate -> (throttled) expensive sense ---------
    def tick(self):
        """Process the latest frame. Returns ``(frame, faces)`` for an optional GUI."""
        seq, frame = self.sensorium.latest_with_seq()
        if frame is None:
            return None, []
        now = time.time()
        faces = self.faceid.identify(frame)
        present = self._present(faces, now)
        self.present, self.faces = present, faces   # publish for the HUD
        target_here = self.target in present
        others = present - {self.target}
        desired = "engaged" if target_here else ("passive" if others else "idle")
        if desired != self.state:
            self._transition(desired, others)
        if self.state == "engaged" and self._eye_ok and now - self._last_eye >= self.eye_interval:
            self._look()
        return frame, faces

    def _transition(self, desired: str, others: set[str]) -> None:
        if desired == "engaged":
            self._emit("engage", f"{self.target} is here — engaging", {"person": self.target})
            self._last_eye = 0.0                     # force an eye read on the next tick
        elif desired == "passive":
            who = ", ".join(sorted(others))
            self._emit("passive", f"{who} present — known but not {self.target}; staying passive"
                       if who != "unknown" else "unknown person — staying passive",
                       {"others": sorted(others)})
        else:
            self._emit("idle", "frame is clear — resting")
        self.state = desired

    def _look(self) -> None:
        jpeg = self.sensorium.latest_jpeg()
        if jpeg is None:
            return
        r = self.eye.look(jpeg, prompt=ENGAGE_PROMPT)
        self._last_eye = time.time()
        if r.ok:
            self._last_reading = r.text
            corroborated, stable = self._corroborate(r.text)
            mark = (f"[✓ {', '.join(corroborated)}]" if stable
                    else "[unconfirmed — single read]")
            self._emit("eye", f"({r.latency:.1f}s) {r.text}  {mark}",
                       {"reading": r.text, "latency": r.latency,
                        "corroborated": corroborated, "stable": stable})
        else:
            self._emit("eye", "[eye unavailable — pausing the VLM]", {"reading": None})
            self._eye_ok = self.eye.health()         # re-check; stop trying if still down

    def _corroborate(self, text: str) -> tuple[list[str], bool]:
        """Update the rolling window with this reading and return the scene terms that
        RECUR across ≥ CORROBORATE_MIN of the last CORROBORATE_WINDOW reads, plus a
        ``stable`` flag. A real object persists across reads; a one-off misread does
        not — so only ``stable`` readings are eviction-eligible (the step-5 gate,
        ``percept.PerceptMemory._gate``, consumes ``data['corroborated']``/
        ``data['stable']``). This only produces the signal; percept persists it."""
        terms = _content_terms(text)
        self._reading_terms.append(terms)
        counts: Counter = Counter()
        for s in self._reading_terms:                # each read contributes a term once
            counts.update(s)
        corroborated = sorted(t for t in terms if counts[t] >= CORROBORATE_MIN)
        return corroborated, bool(corroborated)

    # ---- run loops -----------------------------------------------------------
    def run(self, seconds: Optional[float] = None, tick_hz: float = 4.0) -> int:
        period = 1.0 / tick_hz
        deadline = None if seconds is None else time.time() + seconds
        self._emit("start", f"perceiving — target={self.target}, "
                            f"eye={'on' if self._eye_ok else 'off'}", {"eye": self._eye_ok})
        try:
            while deadline is None or time.time() < deadline:
                self.tick()
                time.sleep(period)
        except KeyboardInterrupt:
            pass
        self._emit("stop", f"{len(self.events)} events held in working memory",
                   {"events": len(self.events)})
        return 0

    def run_gui(self, seconds: Optional[float] = None) -> int:
        win = "Engram - perceiving"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"GUI unavailable ({exc}). Run without --gui.")
            return 1
        deadline = None if seconds is None else time.time() + seconds
        self._emit("start", f"perceiving (gui) — target={self.target}, "
                            f"eye={'on' if self._eye_ok else 'off'}", {"eye": self._eye_ok})
        while deadline is None or time.time() < deadline:
            frame, faces = self.tick()
            if frame is not None:
                draw_faces(frame, faces)
                color = {"engaged": GREEN, "passive": AMBER}.get(self.state, (180, 147, 133))
                cv2.putText(frame, f"state: {self.state}", (12, 30), FONT, 0.8, color, 2, cv2.LINE_AA)
                if self._last_reading and self.state == "engaged":
                    cv2.putText(frame, self._last_reading[:80], (12, 60), FONT, 0.55,
                                (248, 236, 232), 1, cv2.LINE_AA)
                cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        self._emit("stop", f"{len(self.events)} events held in working memory",
                   {"events": len(self.events)})
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram perceiving loop — the engagement gate")
    ap.add_argument("--device", type=int, default=0, help="V4L2 camera index (Brio = 0)")
    ap.add_argument("--person", default=os.environ.get("ENGRAM_USER") or getpass.getuser(), help="who to engage with (the 'is it me' gate)")
    ap.add_argument("--seconds", type=float, default=None,
                    help="run for N seconds then exit (default: until Ctrl-C)")
    ap.add_argument("--eye-interval", type=float, default=8.0,
                    help="seconds between eye reads while engaged")
    ap.add_argument("--hold", type=float, default=1.5, help="presence debounce window (s)")
    ap.add_argument("--threshold", type=float, default=None, help="face match threshold")
    ap.add_argument("--no-eye", action="store_true", help="gate only; never call the VLM eye")
    ap.add_argument("--gui", action="store_true", help="show a window (face boxes + state)")
    args = ap.parse_args()

    fid = FaceID(threshold=args.threshold) if args.threshold is not None else FaceID()
    if not fid.gallery:
        print("⚠ face gallery empty — run `infra/engram/eye/face.py enroll YourName` first; "
              "everyone reads as 'unknown' so the gate will only ever go passive/idle.")
    eye = Eye()
    with Sensorium(device=args.device) as sns:
        if not sns.wait_first(3.0):
            print("✗ no camera frame — is /dev/video0 free?")
            return 1
        loop = PerceiveLoop(sns, fid, eye, target=args.person,
                            eye_interval=args.eye_interval, hold=args.hold,
                            use_eye=not args.no_eye)
        return loop.run_gui(seconds=args.seconds) if args.gui else loop.run(seconds=args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
