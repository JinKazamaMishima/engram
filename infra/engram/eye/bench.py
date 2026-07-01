#!/usr/bin/env python3
"""Engram's eye-bench — a live window: the webcam feed with SmolVLM's running read.

The video plays at camera framerate; a background thread sends the current frame
to the local VLM (:class:`Eye`) as fast as it can and overlays the latest reading
+ latency. This is the human-watchable proof that objects (and, soon, faces) are
recognized correctly — and the first draft of the eye in the perceiving loop.

    # 1) start the VLM backend (once):
    ~/.local/opt/llama.cpp/llama-b9840/llama-server \
        -hf ggml-org/SmolVLM-500M-Instruct-GGUF --host 127.0.0.1 --port 8080
    # 2) watch:
    .venv/bin/python infra/engram/eye/bench.py              # live window
    .venv/bin/python infra/engram/eye/bench.py --headless 5 # 5 captions, no GUI (for CI/SSH)

Keys in the window:  p = cycle prompt   s = snapshot   q / Esc = quit

FACE-ID SEAM: the caption worker already holds each decoded frame. A future
``face.py`` (detect -> embed -> match an enrolled gallery) consumes the SAME
frames to answer "is this the user?" and gate whether Engram engages — the milestone.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye import DEFAULT_PROMPT, DEFAULT_SERVER, Eye  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (249, 232, 103)    # BGR of #67E8F9 — Engram's accent
WHITE = (248, 236, 232)   # BGR of #E8ECF8
DIM = (180, 147, 133)

# Prompt presets, cycled with `p`. The small VLM answers a specific question far
# better than "describe everything"; "face" is the bridge to the face-ID milestone.
PROMPTS = [
    ("scene", DEFAULT_PROMPT),
    ("one-line", "In one sentence, what is happening in this webcam frame?"),
    ("objects", "List the distinct objects you can see, comma-separated."),
    ("face", "Describe the face of the person closest to the camera: hair, facial "
             "hair, glasses, expression. If there is no person, say 'no face'."),
]


def wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out or [""]


class EyeBench:
    def __init__(self, device: int = 0, server: str = DEFAULT_SERVER,
                 width: int = 1280, height: int = 720) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.eye = Eye(server=server)
        self._lock = threading.Lock()
        self._frame = None              # latest BGR frame (numpy)
        self._reading = None            # latest Reading
        self._stop = threading.Event()
        self._prompt_idx = 0

    @property
    def prompt(self) -> str:
        return PROMPTS[self._prompt_idx][1]

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera device {self.device} (/dev/video{self.device})")
        return cap

    # ---- background captioner: read latest frame -> VLM -> store reading ----
    def _caption_worker(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is None:
                self._stop.wait(0.05)
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            reading = self.eye.look(buf.tobytes(), prompt=self.prompt)
            with self._lock:
                self._reading = reading

    # ---- overlay ----
    def _bar(self, frame, x, y, w, h, alpha) -> None:
        y0, y1 = max(0, y), max(0, y) + h
        sub = frame[y0:y1, x:x + w]
        if sub.size == 0:
            return
        overlay = sub.copy()
        overlay[:] = (20, 15, 10)
        cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)

    def _draw(self, frame, reading, fps) -> None:
        h, w = frame.shape[:2]
        label = PROMPTS[self._prompt_idx][0]
        lat = f"{reading.latency:.1f}s" if reading else "…"
        ok = "" if (reading is None or reading.ok) else "  ⚠ no backend"
        self._bar(frame, 0, 0, w, 30, 0.55)
        cv2.putText(frame, f"Engram eye  ·  SmolVLM-500M  ·  {fps:4.1f} fps  ·  {lat}  ·  "
                           f"prompt[{label}]{ok}", (10, 20), FONT, 0.5, CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, "p:prompt  s:snap  q:quit", (w - 230, 20), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        text = reading.text if reading else "warming up — first reading on the way…"
        lines = wrap(text, max(30, w // 12))
        box_h = 16 + 24 * len(lines)
        self._bar(frame, 0, h - box_h, w, box_h, 0.6)
        y = h - box_h + 26
        for ln in lines:
            cv2.putText(frame, ln, (12, y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
            y += 24

    def _snapshot(self, frame) -> str:
        path = f"/tmp/engram_eye_{int(time.time())}.jpg"
        cv2.imwrite(path, frame)
        return path

    # ---- live window ----
    def run_window(self) -> int:
        cap = self._open()
        threading.Thread(target=self._caption_worker, daemon=True).start()
        win = "Engram - eye bench"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            self._stop.set()
            cap.release()
            print(f"GUI unavailable ({exc}). Run with --headless N instead.")
            return 1
        fps_t, n, fps = time.time(), 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with self._lock:
                self._frame = frame
                reading = self._reading
            n += 1
            if n >= 10:
                now = time.time()
                fps = n / (now - fps_t)
                fps_t, n = now, 0
            self._draw(frame, reading, fps)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                self._prompt_idx = (self._prompt_idx + 1) % len(PROMPTS)
            elif key == ord("s"):
                print("snapshot:", self._snapshot(frame))
        self._stop.set()
        cap.release()
        cv2.destroyAllWindows()
        return 0

    # ---- headless (verify the pipeline without a window: CI / SSH) ----
    def run_headless(self, count: int) -> int:
        cap = self._open()
        print(f"camera /dev/video{self.device} open · server {self.eye.server} · "
              f"healthy={self.eye.health()}")
        for i in range(count):
            frame = None
            for _ in range(12):                  # discard stale/dark warm-up frames
                ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[{i+1}/{count}] camera read failed")
                continue
            _, buf = cv2.imencode(".jpg", frame)
            r = self.eye.look(buf.tobytes(), prompt=self.prompt)
            print(f"[{i+1}/{count}] ({r.latency:.1f}s ok={r.ok}) {r.text}")
        cap.release()
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram eye bench — live VLM read of the webcam")
    ap.add_argument("--device", type=int, default=0, help="V4L2 camera index (usually 0)")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="llama-server base URL")
    ap.add_argument("--headless", type=int, metavar="N",
                    help="capture N frames, print captions, exit (no GUI)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    bench = EyeBench(device=args.device, server=args.server,
                     width=args.width, height=args.height)
    if not bench.eye.health():
        print(f"⚠ no llama-server at {args.server} — start the SmolVLM backend first "
              "(see infra/engram/eye/README.md).")
    return bench.run_headless(args.headless) if args.headless else bench.run_window()


if __name__ == "__main__":
    raise SystemExit(main())
