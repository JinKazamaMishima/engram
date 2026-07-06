#!/usr/bin/env python3
"""Engram's microphone — continuous capture into a ring buffer (the ear's feed).

The audio twin of the Sensorium's camera grab: one background thread owns the mic
(via ``arecord``, so zero Python audio deps — same spirit as the eye using a
prebuilt binary) and streams 16 kHz mono PCM into a fixed ring buffer. The senses
read the recent window through here:

    speaker-ID (voice.py) ─┐
    Whisper    (ear.py)   ─┴─ mic.recent(seconds)  /  mic.rms(seconds)

16 kHz mono S16 is exactly what Whisper and the ECAPA speaker model want, so no
resampling downstream. The cheap always-on tier is an energy gate (``rms``); real
VAD and the speaker-ID gate sit on top, so Whisper only ever runs on gated,
short, owner-only segments.

    .venv/bin/python infra/engram/perceive/mic.py            # 4s capture + level meter
"""
from __future__ import annotations

import argparse
import subprocess
import threading
import time
import wave

import numpy as np

DEFAULT_DEVICE = "plughw:2,0"   # Brio 301 (card 2); plughw lets ALSA convert to 16k/mono
RATE = 16000


class Microphone:
    def __init__(self, device: str = DEFAULT_DEVICE, rate: int = RATE,
                 ring_seconds: float = 30.0) -> None:
        self.device = device
        self.rate = rate
        self._cap = int(rate * ring_seconds)
        self._ring = np.zeros(self._cap, dtype=np.int16)
        self._w = 0            # next write index (wraps)
        self._n = 0            # total samples ever written
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._err: str | None = None

    # ---- lifecycle -----------------------------------------------------------
    def start(self, warmup: float = 0.5) -> "Microphone":
        if self._thread is not None:
            return self
        cmd = ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
               "-c", "1", "-t", "raw", "-q"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stop.clear()
        self._thread = threading.Thread(target=self._read_loop, name="engram-mic", daemon=True)
        self._thread.start()
        time.sleep(warmup)
        if self._err:
            raise RuntimeError(f"arecord failed on '{self.device}': {self._err}")
        return self

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        chunk_bytes = self.rate * 2 // 10        # ~100 ms of S16 mono
        while not self._stop.is_set():
            data = self._proc.stdout.read(chunk_bytes)
            if not data:
                # arecord died — capture its complaint (busy device, bad name, …)
                err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
                if err.strip():
                    self._err = err.strip().splitlines()[-1]
                break
            self._write(np.frombuffer(data, dtype=np.int16))

    def _write(self, chunk: np.ndarray) -> None:
        m = len(chunk)
        if m == 0:
            return
        if m >= self._cap:
            chunk, m = chunk[-self._cap:], self._cap
        with self._lock:
            end = self._w + m
            if end <= self._cap:
                self._ring[self._w:end] = chunk
            else:
                first = self._cap - self._w
                self._ring[self._w:] = chunk[:first]
                self._ring[:m - first] = chunk[first:]
            self._w = end % self._cap
            self._n += m

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    # ---- read access (every audio sense reads through here) ------------------
    def recent(self, seconds: float) -> np.ndarray:
        """The last ``seconds`` of audio as float32 in [-1, 1] (mono, 16 kHz)."""
        k = min(int(seconds * self.rate), self._cap)
        with self._lock:
            k = min(k, self._n)
            if k == 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._w - k) % self._cap
            if start + k <= self._cap:
                out = self._ring[start:start + k].copy()
            else:
                out = np.concatenate([self._ring[start:], self._ring[:(start + k) % self._cap]])
        return out.astype(np.float32) / 32768.0

    def rms(self, seconds: float = 0.5) -> float:
        """Energy of the last ``seconds`` — the cheap always-on speech/silence gate."""
        x = self.recent(seconds)
        return float(np.sqrt(np.mean(x * x))) if x.size else 0.0

    @property
    def samples_seen(self) -> int:
        return self._n

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "Microphone":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def write_wav(path: str, audio: np.ndarray, rate: int = RATE) -> None:
    """Save float32 [-1,1] mono audio to a 16-bit WAV (stdlib; no soundfile dep)."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram microphone — capture + level meter")
    ap.add_argument("--device", default=DEFAULT_DEVICE, help="ALSA device (Brio = plughw:2,0)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--wav", default="/tmp/engram_mic.wav", help="where to dump the capture")
    args = ap.parse_args()
    print(f"capturing {args.seconds:.0f}s from '{args.device}' … (speak!)")
    with Microphone(device=args.device) as mic:
        t_end = time.time() + args.seconds
        peak = 0.0
        while time.time() < t_end:
            lvl = mic.rms(0.3)
            peak = max(peak, lvl)
            bars = int(min(lvl * 4000, 40))
            print(f"  level |{'█' * bars:<40}| rms={lvl:.4f}", end="\r", flush=True)
            time.sleep(0.1)
        audio = mic.recent(args.seconds)
    print()
    print(f"✓ captured {audio.size} samples ({audio.size / RATE:.1f}s), peak rms={peak:.4f}")
    write_wav(args.wav, audio)
    print(f"  wrote {args.wav}  (play: aplay {args.wav})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
