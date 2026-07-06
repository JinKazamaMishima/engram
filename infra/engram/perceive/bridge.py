#!/usr/bin/env python3
"""PerceptionBridge — wire the senses INTO a running front-end (the TUI), so seeing
and hearing the operator drive the SAME conversation they type into, not a separate session.

This is the "interact here" path. Unlike ``awake.py`` (a standalone service with its
own mind + journal), the bridge does only SENSING + GATING and hands an actionable
prompt to a callback — the TUI feeds that to its own driver and renders Engram's reply
inline. So perception and typing are one Engram, one chat.

In-harness UX choice: **voice-driven, no auto-greet.** While the operator works they're on camera
continuously, so greeting every glance would spam the chat. Instead vision is the GATE
(only listen when it sees him) and their voice — "Engram …" — drives a turn. The latest
scene reading rides along so "Engram, what do you see?" works.

    from bridge import PerceptionBridge
    pb = PerceptionBridge(on_act=lambda prompt, marker: app.post(...)).start()
    ...
    pb.stop()
"""
from __future__ import annotations

import getpass
import os
import sys
import threading
import time
from typing import Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "eye"))


class PerceptionBridge:
    def __init__(self, on_act: Callable[[str, str], None], *, target: str = os.environ.get("ENGRAM_USER") or getpass.getuser(),
                 device: int = 0, voice_threshold: Optional[float] = None,
                 ear_model: str = "base.en", speech_gate: float = 0.02,
                 tick_hz: float = 4.0) -> None:
        self.on_act = on_act
        self.target = target
        self.device = device
        self.voice_threshold = voice_threshold
        self.ear_model = ear_model
        self.speech_gate = speech_gate
        self.tick_hz = tick_hz
        self._sns = self._loop = self._hearing = None
        self._percept = None
        self._stop = threading.Event()
        self._tick_thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.last_voice = None    # (name, cos, t) — last speaker the ear identified, for the HUD
        self.last_heard = None    # (text, t)      — last thing it transcribed, for the HUD

    def start(self) -> "PerceptionBridge":
        """Bring up camera + face + eye. (The ear — mic + voice-ID + Whisper — is DISABLED
        for now; see the SOUND DISABLED block below.) Heavy imports are lazy, so a normal
        TUI launch (perception off) never loads cv2 / onnxruntime / whisper."""
        try:
            from eye import Eye  # noqa: E402
            from face import FaceID  # noqa: E402
            from loop import PerceiveLoop  # noqa: E402
            from sensorium import Sensorium  # noqa: E402
            # --- SOUND DISABLED (2026-06-30) ---------------------------------------------
            # The ear is off until voice-ID is reliable. On the far-field Brio it was locking
            # onto background NOISE, not voices — so every speaker matched (the noise floor,
            # not the speaker, dominated the embedding). Vision + face only for now. To
            # re-enable: uncomment the imports + the ear bring-up below, restore the audio
            # rows in app.py::_render_perception, and the audio fields in snapshot(). See
            # [[engram-ear-speaker-id]].
            # from hearing import Hearing             # noqa: E402
            # from voice import VoiceID               # noqa: E402
            # from ear import Ear                     # noqa: E402

            fid = FaceID()
            if self.target not in fid.gallery:
                self.error = f"face '{self.target}' not enrolled (eye/face.py enroll {self.target})"
                return self
            eye = Eye()
            use_eye = eye.health()
            self._sns = Sensorium(device=self.device).start()
            if not self._sns.wait_first(3.0):
                self.error = "no camera frame (is /dev/video0 free?)"
                self.stop()
                return self
            # Step 5: even as a passive HUD, gate-worthy events (presence
            # transitions, corroborated scene changes) persist + evict to LTM.
            from percept import PerceptMemory  # noqa: E402
            self._percept = PerceptMemory()
            # The loop maintains the face/engagement gate + a throttled scene reading. With
            # the ear off, nothing drives a turn — perception is a passive vision-only HUD.
            self._loop = PerceiveLoop(self._sns, fid, eye, target=self.target,
                                      use_eye=use_eye,
                                      on_event=self._percept.wrap(None))

            # --- SOUND DISABLED: the ear bring-up (mic + voice-ID + Whisper) --------------
            # engaged = lambda: self._loop.state == "engaged"   # noqa: E731 — the gate
            # vid = (VoiceID(threshold=self.voice_threshold)
            #        if self.voice_threshold is not None else VoiceID())
            # ear = Ear(model=self.ear_model)
            # threading.Thread(target=ear.ready, daemon=True).start()    # pre-warm Whisper
            # self._hearing = Hearing(vid, ear, target=self.target, on_event=self._on_event,
            #                         is_engaged=engaged, active=engaged,
            #                         speech_gate=self.speech_gate)

            self._stop.clear()
            self._tick_thread = threading.Thread(target=self._tick_loop, name="bridge-vision",
                                                 daemon=True)
            self._tick_thread.start()
            # self._hearing.start()                  # SOUND DISABLED
        except Exception as exc:   # noqa: BLE001 — never take the TUI down with us
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop()
        return self

    def _tick_loop(self) -> None:
        """Drive the camera gate + scene reading (PerceiveLoop.tick) ourselves, so we can
        stop cleanly — no auto-greet, just keep `loop.state` + `_last_reading` current."""
        period = 1.0 / self.tick_hz
        while not self._stop.is_set():
            try:
                self._loop.tick()
            except Exception:   # noqa: BLE001
                pass
            self._stop.wait(period)

    def _on_event(self, ev) -> None:
        """From the hearing gate. Any speech event updates the HUD telemetry; only a
        directed utterance ('Engram …' while seen) actually drives a turn."""
        if ev.kind in ("heard", "overheard", "ambient"):
            self.last_voice = (ev.data.get("voice", "unknown"), ev.data.get("cos", 0.0), time.time())
        if ev.kind != "heard":
            return
        text = (ev.data.get("text") or "").strip()
        if not text:
            return
        self.last_heard = (text, time.time())
        scene = getattr(self._loop, "_last_reading", None)
        prompt = (f"[perception] {self.target} said this aloud to you"
                  + (f' (through your camera eye you currently see: "{scene}")' if scene else "")
                  + f': "{text}". Reply to him briefly and naturally, as a spoken reply.')
        self.on_act(prompt, text)

    @property
    def status(self) -> str:
        if self.error:
            return f"perception off — {self.error}"
        return "perceiving — vision only (sound disabled)"

    def snapshot(self) -> dict:
        """Live telemetry for the HUD — present faces, the engagement gate state, and the
        eye's latest scene reading. Cheap (attribute reads), safe to poll a few times/sec.
        Audio telemetry (mic level, last speaker, last heard) is omitted while SOUND is
        DISABLED — see start()."""
        loop = self._loop
        return {
            "ok": self.error is None,
            "error": self.error,
            "present": sorted(loop.present) if loop else [],
            "faces": [(f.name, f.score) for f in (loop.faces if loop else [])],
            "state": loop.state if loop else "booting",
            "scene": getattr(loop, "_last_reading", None) if loop else None,
        }

    def stop(self) -> None:
        self._stop.set()
        if self._percept is not None:
            try:
                self._percept.flush()   # fold the un-curated tail
            except Exception:   # noqa: BLE001
                pass
            self._percept = None
        if self._hearing is not None:
            try:
                self._hearing.stop()
            except Exception:   # noqa: BLE001
                pass
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=2)
            self._tick_thread = None
        if self._sns is not None:
            try:
                self._sns.stop()
            except Exception:   # noqa: BLE001
                pass
            self._sns = None
