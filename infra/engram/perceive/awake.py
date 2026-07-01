#!/usr/bin/env python3
"""Engram awake — the perceiving loop wired to the real mind, SEEING.

The assembled perceiving front-end: the Sensorium (one camera capture) feeds the
engagement gate (``loop.py``), which wakes Engram's actual mind (``mind.py`` →
``core.AgentSDKDriver``) — the SAME brain as the terminal TUI and the Telegram bridge,
now driven by perception. Sit down and Engram greets you by name, aware of what its eye
sees; other people are gated out by face-ID and never reach the mind. (Audio input is
intentionally not built yet — vision only.)

    .venv/bin/python infra/engram/perceive/awake.py              # see (headless)
    .venv/bin/python infra/engram/perceive/awake.py --gui        # + the live window
    .venv/bin/python infra/engram/perceive/awake.py --no-eye     # gate only; greet blind

Needs the SmolVLM eye backend (eye/README.md) for scene-aware greetings and the camera
face gallery for the gate.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys

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
from eye import Eye  # noqa: E402
from face import FaceID  # noqa: E402
from loop import PerceiveLoop  # noqa: E402
from mind import PerceivingMind  # noqa: E402
from sensorium import Sensorium  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram awake — perceiving loop wired to the mind")
    ap.add_argument("--device", type=int, default=0, help="V4L2 camera index (usually 0)")
    ap.add_argument("--person", default=getpass.getuser(), help="who to engage with")
    ap.add_argument("--seconds", type=float, default=None,
                    help="run for N seconds then exit (default: until Ctrl-C)")
    ap.add_argument("--eye-interval", type=float, default=6.0, help="seconds between eye reads while engaged")
    ap.add_argument("--hold", type=float, default=1.5, help="presence debounce window (s)")
    ap.add_argument("--threshold", type=float, default=None, help="FACE match threshold")
    ap.add_argument("--no-eye", action="store_true", help="gate only; greet blind (no VLM)")
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

    print("waking Engram's mind (connecting to the model) …")
    mind = PerceivingMind(target=args.person, eye_enabled=use_eye,
                          greet_cooldown=args.greet_cooldown, effort=args.mind_effort,
                          model=args.mind_model).start()

    with Sensorium(device=args.device) as sns:
        if not sns.wait_first(3.0):
            print("✗ no camera frame — is /dev/video0 free?")
            mind.stop()
            return 1
        loop = PerceiveLoop(sns, fid, eye, target=args.person, eye_interval=args.eye_interval,
                            hold=args.hold, use_eye=use_eye, on_event=mind.on_event)
        # AUTONOMY GATE: Engram only ACTS (greets) when it currently SEES the user (engaged on
        # camera) AND — unless --no-harness-gate — the harness TUI is open. Binds every
        # autonomous turn to "the user is present and actively running Engram."
        if args.no_harness_gate:
            gate = lambda: loop.state == "engaged"                       # noqa: E731
        else:
            gate = lambda: harness_open() and loop.state == "engaged"    # noqa: E731
        mind.gate = gate
        print(f"  autonomy gate: {'face present' if args.no_harness_gate else 'harness-open AND face present'}"
              f"  ·  harness now {'OPEN' if harness_open() else 'CLOSED'}")
        try:
            if args.gui:
                loop.run_gui(seconds=args.seconds)
            else:
                loop.run(seconds=args.seconds)
        finally:
            mind.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
