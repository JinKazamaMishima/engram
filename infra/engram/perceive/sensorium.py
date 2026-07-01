#!/usr/bin/env python3
"""Engram's Sensorium — ONE capture source that fans frames to every sense.

V4L2 is single-consumer: only one open handle can hold ``/dev/video0`` at a time,
so the eye-bench and the face sense — each of which opened the camera directly —
could never run at once (we hit this live: face-verify failed because the eye
window already held the camera). The Sensorium is the fix and the foundation of
the perceiving loop: a single background thread owns the capture device, grabs
frames continuously, and publishes the latest one under a lock. Every sense reads
through the ONE Sensorium instead of touching ``/dev/video0``:

    eye  (SmolVLM) ─┐
    face (ArcFace) ─┴─ sensorium.latest() / .latest_jpeg() / .latest_with_seq()

It is display-free and thread-safe, so it drops into a headless service (systemd)
later exactly like the Telegram bridge. The open code (MJPG, 1280x720, CAP_V4L2)
is the same as ``eye/bench.py`` and ``eye/face.py:open_cam`` — now owned in one
place so those callers can be pointed here.

    # the whole proof of step 1: face-ID AND the eye, off ONE capture, at once
    .venv/bin/python infra/engram/perceive/sensorium.py            # selftest
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2


class Sensorium:
    """One background thread owns the camera; every sense reads the latest frame.

    Use as a context manager (``with Sensorium() as sns:``) or call
    :meth:`start` / :meth:`stop` directly. Thread-safe: :meth:`latest` and friends
    hand back a *copy* under a lock, so consumers can hold a frame while the grab
    thread moves on."""

    def __init__(self, device: int = 0, width: int = 1280, height: int = 720) -> None:
        self.device = device
        self.width = width
        self.height = height
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None             # latest BGR frame (numpy), guarded by _lock
        self._seq = 0                  # monotonic frame counter; 0 = nothing yet
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps = 0.0

    # ---- lifecycle -----------------------------------------------------------
    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera /dev/video{self.device} — is it free? "
                "(V4L2 is single-consumer; close any eye-bench / face window first)")
        return cap

    def start(self, warmup: float = 2.0) -> "Sensorium":
        """Open the device and start grabbing. Blocks up to ``warmup`` seconds for
        the first good frame so callers don't immediately see ``None``."""
        if self._thread is not None:
            return self
        self._cap = self._open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._grab_loop, name="sensorium",
                                        daemon=True)
        self._thread.start()
        self.wait_first(timeout=warmup)
        return self

    def _grab_loop(self) -> None:
        n, t0 = 0, time.time()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._stop.wait(0.02)      # camera hiccup — back off, keep owning it
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
            n += 1
            if n >= 30:                    # refresh the fps estimate every ~30 frames
                now = time.time()
                self._fps = n / (now - t0) if now > t0 else 0.0
                n, t0 = 0, now

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ---- frame access (every sense reads through here) -----------------------
    def latest(self):
        """A copy of the most recent BGR frame, or ``None`` if nothing captured yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def latest_with_seq(self):
        """``(seq, frame_copy)``. ``seq`` is monotonic (``0``/``None`` = nothing yet),
        so a consumer can skip a frame it has already processed."""
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._seq, self._frame.copy()

    def latest_jpeg(self, quality: int = 90):
        """The latest frame JPEG-encoded (``bytes``) for the eye/VLM, or ``None``."""
        frame = self.latest()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def wait_first(self, timeout: float = 2.0) -> bool:
        """Block until the first frame is captured (or ``timeout``). ``True`` if one
        arrived."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.02)
        with self._lock:
            return self._frame is not None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- context manager -----------------------------------------------------
    def __enter__(self) -> "Sensorium":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# --------------------------------- selftest ----------------------------------
def _selftest(device: int, rounds: int) -> int:
    """The proof of step 1: run face-ID AND the eye off ONE Sensorium at the same
    time — the exact thing the two direct-open callers could never do together."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eye"))
    from eye import Eye  # noqa: E402 — sibling sense modules
    from face import FaceID  # noqa: E402

    print(f"opening Sensorium on /dev/video{device} …")
    with Sensorium(device=device) as sns:
        if not sns.wait_first(3.0):
            print("✗ no frame captured — is the camera busy or disconnected?")
            return 1
        print("✓ capturing")
        fid = FaceID()
        eye = Eye()
        backend = eye.health()
        print(f"  face gallery : {', '.join(fid.gallery) or '(empty)'}")
        print(f"  eye backend  : {'up' if backend else 'DOWN — readings will be ok=False'}")
        print(f"  — {rounds} rounds, BOTH senses reading the ONE capture —")
        for i in range(rounds):
            seq, frame = sns.latest_with_seq()
            faces = fid.identify(frame)
            who = ", ".join(f"{f.name}({f.score:.2f})" for f in faces) or "nobody"
            jpeg = sns.latest_jpeg()
            r = eye.look(jpeg, prompt="In one sentence, what is in this frame?") if jpeg else None
            if r and r.ok:
                eye_txt = r.text
            elif backend:
                eye_txt = "[eye error]"
            else:
                eye_txt = "[eye backend down]"
            print(f"  [{i+1}/{rounds}] seq={seq:>4}  face=[{who}]  eye={eye_txt[:78]}")
            time.sleep(0.3)
        fps = sns.fps
    print(f"✓ selftest done — ONE capture (~{fps:.0f} fps) fanned to face-ID + eye, "
          "no device contention.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Engram Sensorium — prove ONE capture fans to every sense")
    ap.add_argument("--device", type=int, default=0, help="V4L2 camera index (usually 0)")
    ap.add_argument("--rounds", type=int, default=5, help="selftest rounds")
    args = ap.parse_args()
    return _selftest(args.device, args.rounds)


if __name__ == "__main__":
    raise SystemExit(main())
