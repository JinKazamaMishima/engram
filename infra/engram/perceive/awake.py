#!/usr/bin/env python3
"""Engram awake — the perceiving loop wired to the real mind, SEEING and HEARING.

The assembled perceiving front-end: the Sensorium (one camera capture) feeds the
engagement gate (``loop.py``) while a Microphone feeds the hearing gate
(``hearing.py``); both wake Engram's actual mind (``mind.py`` → ``core.AgentSDKDriver``)
— the SAME brain as the terminal TUI and the Telegram bridge, now driven by
perception. Sit down and Engram greets you by name, aware of what its eye sees; speak
to it — look at it, or say "Engram …" — and it replies to what its ear heard. Guests,
phone calls, and passers-by are gated out by face- and voice-ID and never reach the
mind.

    .venv/bin/python infra/engram/perceive/awake.py              # see + hear (headless)
    .venv/bin/python infra/engram/perceive/awake.py --gui        # + the live window
    .venv/bin/python infra/engram/perceive/awake.py --no-ears    # camera only
    .venv/bin/python infra/engram/perceive/awake.py --no-eye     # senses, but greet blind

Needs the SmolVLM eye backend (eye/README.md) for scene-aware greetings, your voice
enrolled (`voice.py enroll YourName`), and the camera face gallery for the gates.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_PROC = "infra/engram/app.py"   # the Engram TUI; autonomy is gated to "this is running"


def harness_open(pattern: str = HARNESS_PROC) -> bool:
    """Is the Engram harness (the TUI) open in a terminal? Autonomy is bound to this AND a
    present face, so Engram never fires a turn unattended. Scans /proc precisely — a PYTHON
    process with the harness script as an actual argv arg — so a shell command that merely
    mentions the path (or this awake.py service itself) can't false-trigger the gate."""
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    args = f.read().split(b"\0")
            except OSError:
                continue
            if not args or not args[0]:
                continue
            if "python" not in os.path.basename(args[0].decode("utf-8", "replace")).lower():
                continue
            if any(pattern in a.decode("utf-8", "replace") for a in args[1:]):
                return True
        return False
    except Exception:   # noqa: BLE001
        return False
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "eye"))
from ear import Ear  # noqa: E402
from eye import Eye  # noqa: E402
from face import FaceID  # noqa: E402
from hearing import Hearing  # noqa: E402
from loop import PerceiveLoop  # noqa: E402
from mind import PerceivingMind  # noqa: E402
from percept import PerceptMemory  # noqa: E402
from sensorium import Sensorium  # noqa: E402
from voice import VoiceID  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram awake — perceiving loop wired to the mind")
    ap.add_argument("--device", type=int, default=0, help="V4L2 camera index (Brio = 0)")
    ap.add_argument("--person", default=os.environ.get("ENGRAM_USER") or getpass.getuser(), help="who to engage with / listen for")
    ap.add_argument("--seconds", type=float, default=None,
                    help="run for N seconds then exit (default: until Ctrl-C)")
    ap.add_argument("--eye-interval", type=float, default=6.0, help="seconds between eye reads while engaged")
    ap.add_argument("--hold", type=float, default=1.5, help="presence debounce window (s)")
    ap.add_argument("--threshold", type=float, default=None, help="FACE match threshold")
    ap.add_argument("--voice-threshold", type=float, default=None, help="VOICE match threshold")
    ap.add_argument("--ear-model", default="base.en", help="faster-whisper model")
    ap.add_argument("--speech-gate", type=float, default=0.02, help="mic RMS onset gate (tune to room)")
    ap.add_argument("--no-eye", action="store_true", help="gate only; greet blind (no VLM)")
    ap.add_argument("--no-ears", action="store_true", help="camera only (no mic / speaker-ID / Whisper)")
    ap.add_argument("--look-to-talk", action="store_true",
                    help="also accept engaged-on-camera speech without the 'Engram' wake word "
                         "(only safe when you're alone — off by default)")
    ap.add_argument("--gui", action="store_true", help="show the live window")
    ap.add_argument("--greet-cooldown", type=float, default=30.0, help="min seconds between greetings")
    ap.add_argument("--no-harness-gate", action="store_true",
                    help="don't require the Engram TUI to be open (by default Engram only ACTS when "
                         "the harness is open AND it sees you — keeps the always-on service unattended-safe)")
    ap.add_argument("--mind-effort", default="low", help="reasoning effort for replies (low = snappy)")
    ap.add_argument("--mind-model", default=None, help="override the mind's model")
    args = ap.parse_args()

    # --- the eye (camera) gate prerequisites ---
    fid = FaceID(threshold=args.threshold) if args.threshold is not None else FaceID()
    if not fid.gallery:
        print("⚠ face gallery empty — run `infra/engram/eye/face.py enroll YourName` first.")
        return 1
    eye = Eye()
    use_eye = not args.no_eye
    if use_eye and not eye.health():
        print("⚠ no SmolVLM backend — greeting blind (start it per eye/README.md for scene-aware greetings).")
        use_eye = False

    # --- the ear (audio) gate prerequisites ---
    hearing = None
    voiceid = ear = None
    if not args.no_ears:
        voiceid = VoiceID(threshold=args.voice_threshold) if args.voice_threshold is not None else VoiceID()
        if args.person not in voiceid.gallery:
            print(f"⚠ '{args.person}' not enrolled by voice — run `voice.py enroll {args.person}`; "
                  "running camera-only for now.")
            voiceid = None
        else:
            ear = Ear(model=args.ear_model)
            threading.Thread(target=ear.ready, daemon=True).start()   # pre-warm Whisper

    print("waking Engram's mind (connecting to the model) …")
    mind = PerceivingMind(target=args.person, eye_enabled=use_eye,
                          greet_cooldown=args.greet_cooldown, effort=args.mind_effort,
                          model=args.mind_model).start()
    # Step 5: gate-worthy events persist to the percept LiveBuffer and evict
    # into curation — perception grows long-term memory (ENGRAM_PERCEPT=0 to cut).
    percept = PerceptMemory()
    on_ev = percept.wrap(mind.on_event)

    with Sensorium(device=args.device) as sns:
        if not sns.wait_first(3.0):
            print("✗ no camera frame — is /dev/video0 free?")
            mind.stop()
            return 1
        loop = PerceiveLoop(sns, fid, eye, target=args.person, eye_interval=args.eye_interval,
                            hold=args.hold, use_eye=use_eye, on_event=on_ev)
        # AUTONOMY GATE: Engram only ACTS (greet / reply) and only TRANSCRIBES when it currently
        # SEES the operator (engaged on camera) AND — unless --no-harness-gate — the harness TUI is open.
        # Binds every autonomous turn to "the operator is present and actively running Engram."
        if args.no_harness_gate:
            def gate() -> bool:
                return loop.state == "engaged"
        else:
            def gate() -> bool:
                return harness_open() and loop.state == "engaged"
        mind.gate = gate
        print(f"  autonomy gate: {'face present' if args.no_harness_gate else 'harness-open AND face present'}"
              f"  ·  harness now {'OPEN' if harness_open() else 'CLOSED'}")
        if voiceid is not None and ear is not None:
            hearing = Hearing(voiceid, ear, target=args.person, on_event=on_ev,
                              is_engaged=lambda: loop.state == "engaged", active=gate,
                              look_to_talk=args.look_to_talk, speech_gate=args.speech_gate).start()
            mode = "say 'Engram …' or just talk (look-to-talk)" if args.look_to_talk else "say 'Engram …' to be heard"
            print(f"  ears on — {mode}. (voice-ID is a logged hint; wake word is the gate.)")
        try:
            if args.gui:
                loop.run_gui(seconds=args.seconds)
            else:
                loop.run(seconds=args.seconds)
        finally:
            if hearing is not None:
                hearing.stop()
            mind.stop()
            percept.flush()   # fold the day's un-curated tail before we go dark
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
