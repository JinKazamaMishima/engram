#!/usr/bin/env python3
"""Engram's hearing — the audio engagement gate (the ear's loop).

The audio twin of the camera gate in ``loop.py``, cheap→expensive:

    energy-VAD (endpointing) → Whisper (ear.py) → "addressed to Engram?" → ``heard``

The gate that solves the office-with-a-guest problem is the **wake word**: an utterance
is fed to the mind only if it says "Engram …" (or ``--look-to-talk`` is on AND you're
engaged on camera — opt-in, for when you're alone). Everything else is logged as
``overheard`` and dropped, so a guest's chatter / a phone call never reaches the mind.

Speaker-ID (``voice.py``) runs as a LOGGED HINT, not a gate. On a shared far-field
office mic two similar voices overlap too much (guest ~0.83 vs owner ~0.85) to gate on
reliably — hard-gating would both admit the guest and drop the real owner. The model + recipe
are correct (clean distinct speakers separate at ~0.45); the wall is far-field + short
utterances. FUTURE UPGRADE: a close-talk/headset mic, a bigger speaker model, or AS-norm
cohort score-normalization → then promote voice-ID back to a hard gate. See
[[engram-ear-speaker-id]].

Runs its own thread off a :class:`Microphone`, emitting :class:`loop.Event` objects
via ``on_event`` — so it plugs into the SAME mind as the camera loop.

    .venv/bin/python infra/engram/perceive/hearing.py    # standalone (wake-word "Engram …")
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
import threading
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ear import Ear  # noqa: E402
from loop import Event  # noqa: E402 — reuse the event shape + log
from mic import Microphone  # noqa: E402
from voice import VoiceID  # noqa: E402

# Strips a leading wake word so "Engram, what's the time" → "what's the time".
WAKE_RE = re.compile(r"^\W*(hey\s+|ok(ay)?\s+)?engram\b[\s,.:;!?-]*", re.I)


class Hearing:
    def __init__(self, voiceid: VoiceID, ear: Ear, *, target: str = os.environ.get("ENGRAM_USER") or getpass.getuser(),
                 mic: Optional[Microphone] = None,
                 is_engaged: Optional[Callable[[], bool]] = None,
                 on_event: Optional[Callable[[Event], None]] = None,
                 look_to_talk: bool = False,
                 active: Optional[Callable[[], bool]] = None,
                 speech_gate: float = 0.02, hangover: float = 0.7,
                 min_utter: float = 0.5, max_utter: float = 15.0) -> None:
        self.voiceid = voiceid
        self.ear = ear
        self.target = target
        self.mic = mic or Microphone()
        self._own_mic = mic is None
        self.is_engaged = is_engaged or (lambda: False)
        # look-to-talk (accept any engaged-on-camera speech, no wake word) is OPT-IN: safe
        # only when alone. With others around, leave it off — the wake word is the gate.
        self.look_to_talk = look_to_talk
        # active(): only transcribe when this returns True (harness open AND the owner on camera).
        # Skips Whisper on ambient office speech when Engram isn't in use — privacy + CPU.
        self.active = active
        self.on_event = on_event
        self.speech_gate = speech_gate      # RMS onset gate (tune to the room)
        self.hangover = hangover            # silence (s) that ends an utterance
        self.min_utter = min_utter
        self.max_utter = max_utter
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _emit(self, kind: str, detail: str, data: Optional[dict] = None) -> Event:
        ev = Event(time.time(), kind, detail, data or {})
        print(f"  {ev.stamp()}  {kind:<9} {detail}")
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception as exc:   # noqa: BLE001 — a bad listener must not kill hearing
                print(f"  (on_event raised {type(exc).__name__}: {exc})")
        return ev

    def start(self) -> "Hearing":
        if self._thread is not None:
            return self
        if self._own_mic:
            self.mic.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="engram-hearing", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._own_mic:
            self.mic.stop()

    # ---- endpointing: detect one utterance (speech onset → speech → silence) --
    def _loop(self) -> None:
        in_speech, t_start, t_voice = False, 0.0, 0.0
        while not self._stop.is_set():
            now = time.time()
            lvl = self.mic.rms(0.15)
            if lvl >= self.speech_gate:
                if not in_speech:
                    in_speech, t_start = True, now - 0.25      # small pre-roll
                t_voice = now
            elif in_speech and (now - t_voice) >= self.hangover:
                in_speech = False
                dur = min(t_voice - t_start + 0.3, self.max_utter)
                if dur >= self.min_utter:
                    try:
                        self._process(self.mic.recent(dur))
                    except Exception as exc:   # noqa: BLE001
                        print(f"  (hearing error: {type(exc).__name__}: {exc})")
            self._stop.wait(0.1)

    # ---- gate one utterance: speech? → transcribe → directed at Engram? ---------
    def _process(self, utter) -> None:
        if self.active is not None and not self.active():   # harness closed / not seen
            return                                           # don't even transcribe
        spk = self.voiceid.identify(utter)
        if spk.name in ("silence", "too-short"):
            return
        # NOTE: speaker-ID is a LOGGED HINT, not a hard gate. On this far-field mic two
        # similar voices overlap badly (guest ~0.83 vs owner ~0.85), so gating on it would
        # both let the guest through AND drop the real owner. The WAKE WORD is the reliable gate.
        # FUTURE UPGRADE: a close-talk mic / bigger model / AS-norm → make voice-ID the gate.
        tr = self.ear.transcribe(utter)
        text = tr.text.strip()
        if not text:
            return
        voice = spk.name if spk.name != "unknown" else f"?{spk.score:.2f}"
        wake = bool(WAKE_RE.match(text)) or ("engram" in text.lower())
        if wake:
            clean = WAKE_RE.sub("", text, count=1).strip() or text
            self._emit("heard", f"“{clean}”  (wake-word · voice~{voice})",
                       {"text": clean, "raw": text, "via": "wake", "voice": spk.name, "cos": spk.score})
        elif self.look_to_talk and self.is_engaged():
            self._emit("heard", f"“{text}”  (look-to-talk · voice~{voice})",
                       {"text": text, "raw": text, "via": "gaze", "voice": spk.name, "cos": spk.score})
        else:
            self._emit("overheard", f"(voice~{voice}) “{text[:60]}…” — no 'Engram', not ingesting",
                       {"text": text, "voice": spk.name, "cos": spk.score})


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram hearing — audio gate (standalone, wake-word only)")
    ap.add_argument("--person", default=os.environ.get("ENGRAM_USER") or getpass.getuser())
    ap.add_argument("--threshold", type=float, default=None, help="voice match threshold")
    ap.add_argument("--ear-model", default="base.en")
    ap.add_argument("--speech-gate", type=float, default=0.02)
    args = ap.parse_args()
    vid = VoiceID(threshold=args.threshold) if args.threshold is not None else VoiceID()
    if args.person not in vid.gallery:
        print(f"⚠ '{args.person}' not enrolled — run `voice.py enroll {args.person}` first.")
        return 1
    ear = Ear(model=args.ear_model)
    print(f"loading Whisper '{args.ear_model}' …; speak (say 'Engram …' to address me). Ctrl-C to stop.")
    ear.ready()
    h = Hearing(vid, ear, target=args.person, speech_gate=args.speech_gate)
    try:
        h.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        h.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
