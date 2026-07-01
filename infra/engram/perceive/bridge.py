#!/usr/bin/env python3
"""PerceptionBridge — wire the camera senses INTO a running front-end (the TUI), so the
eye and face-ID feed the SAME session the user types into, not a separate one.

This is the "interact here" path. Unlike ``awake.py`` (a standalone service with its own
mind + greeting), the bridge does only SENSING — it maintains the face/engagement gate and
a throttled scene reading, and exposes them via :meth:`snapshot` for the TUI's live senses
card and its per-prompt identity marker. Vision only: who is at the keyboard, and what the
eye currently sees. (Audio input is intentionally not built yet.)

    from bridge import PerceptionBridge
    pb = PerceptionBridge().start()
    ...
    snap = pb.snapshot()
    pb.stop()
"""
from __future__ import annotations

import getpass
import os
import sys
import threading
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "eye"))


class PerceptionBridge:
    def __init__(self, *, target: str = os.environ.get("ENGRAM_USER") or getpass.getuser(),
                 device: int = 0, tick_hz: float = 4.0) -> None:
        self.target = target
        self.device = device
        self.tick_hz = tick_hz
        self._sns = self._loop = None
        self._stop = threading.Event()
        self._tick_thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> "PerceptionBridge":
        """Bring up camera + face + eye. Heavy imports are lazy, so a normal TUI launch
        (perception off) never loads cv2 / onnxruntime."""
        try:
            from eye import Eye  # noqa: E402
            from face import FaceID  # noqa: E402
            from loop import PerceiveLoop  # noqa: E402
            from sensorium import Sensorium  # noqa: E402

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
            # The loop maintains the face/engagement gate + a throttled scene reading;
            # perception is a passive vision-only HUD (nothing here drives a turn).
            self._loop = PerceiveLoop(self._sns, fid, eye, target=self.target,
                                      use_eye=use_eye, on_event=lambda ev: None)
            self._stop.clear()
            self._tick_thread = threading.Thread(target=self._tick_loop, name="bridge-vision",
                                                 daemon=True)
            self._tick_thread.start()
        except Exception as exc:   # noqa: BLE001 — never take the TUI down with us
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop()
        return self

    def _tick_loop(self) -> None:
        """Drive the camera gate + scene reading (PerceiveLoop.tick) ourselves, so we can
        stop cleanly — keep ``loop.state`` + ``_last_reading`` current."""
        period = 1.0 / self.tick_hz
        while not self._stop.is_set():
            try:
                self._loop.tick()
            except Exception:   # noqa: BLE001
                pass
            self._stop.wait(period)

    @property
    def status(self) -> str:
        if self.error:
            return f"perception off — {self.error}"
        return "perceiving — vision"

    def snapshot(self) -> dict:
        """Live telemetry for the HUD — present faces, the engagement gate state, and the
        eye's latest scene reading. Cheap (attribute reads), safe to poll a few times/sec."""
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
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=2)
            self._tick_thread = None
        if self._sns is not None:
            try:
                self._sns.stop()
            except Exception:   # noqa: BLE001
                pass
            self._sns = None
