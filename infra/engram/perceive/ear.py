#!/usr/bin/env python3
"""Engram's ear — local speech-to-text (the mid-tier audio sense).

Hand it a window of 16 kHz mono float32 audio; it returns a :class:`Transcript`
(the text + how long it took). The audio twin of ``eye/eye.py``: no mic, no VAD,
no gating — those live in ``mic.py`` / ``voice.py`` / the loop — so this same class
drops into the perceiving loop as the EXPENSIVE sense that the cheap always-on
tiers (energy-VAD + speaker-ID) gate. Whisper only ever runs on a short, voiced,
owner-only segment.

Backed by faster-whisper (CTranslate2): LOCAL, CPU, no API key, and torch-free, so
it never perturbs recall's hard-pinned ML stack. The model is swappable in one line
(a bigger Whisper, or whisper.cpp) — the seam, as with the eye, is the point.

    .venv/bin/python infra/engram/perceive/ear.py               # record 5s, transcribe
    .venv/bin/python infra/engram/perceive/ear.py --wav f.wav   # transcribe a file
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mic import RATE, Microphone  # noqa: E402

DEFAULT_MODEL = os.environ.get("ENGRAM_EAR_MODEL", "base.en")   # ~140MB; small.en = more accurate


@dataclass
class Transcript:
    text: str
    latency: float           # seconds for the transcription
    ok: bool = True


class Ear:
    """Thin client over a local Whisper (faster-whisper / CTranslate2). The model is
    loaded lazily on first use (and downloaded once), so importing is cheap."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu",
                 compute_type: str = "int8", language: str = "en") -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _ensure(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device=self.device,
                                       compute_type=self.compute_type)
        return self._model

    def ready(self) -> bool:
        try:
            self._ensure()
            return True
        except Exception:   # noqa: BLE001
            return False

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> Transcript:
        """Transcribe 16 kHz mono float32 audio (the gated, voiced segment)."""
        t = time.time()
        try:
            model = self._ensure()
            segments, _ = model.transcribe(
                audio.astype(np.float32), language=self.language, beam_size=1,
                vad_filter=False,                 # already gated upstream; don't double-VAD
                condition_on_previous_text=False,  # independent utterances
                initial_prompt=prompt)
            text = " ".join(s.text.strip() for s in segments).strip()
            return Transcript(text, time.time() - t, ok=True)
        except Exception as exc:   # noqa: BLE001 — never crash the perceiving loop
            return Transcript(f"[ear error: {type(exc).__name__}: {exc}]",
                              time.time() - t, ok=False)


def _read_wav(path: str) -> np.ndarray:
    w = wave.open(path, "rb")
    raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram ear — local Whisper transcription")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model (base.en, small.en, …)")
    ap.add_argument("--seconds", type=float, default=5.0, help="record N seconds from the mic")
    ap.add_argument("--wav", help="transcribe this WAV instead of recording")
    args = ap.parse_args()
    ear = Ear(model=args.model)
    print(f"loading Whisper '{args.model}' (first run downloads it) …")
    if not ear.ready():
        print("✗ could not load the model")
        return 1
    if args.wav:
        audio = _read_wav(args.wav)
        print(f"transcribing {args.wav} ({audio.size / RATE:.1f}s) …")
    else:
        print(f"recording {args.seconds:.0f}s — speak now …")
        with Microphone() as mic:
            time.sleep(args.seconds)
            audio = mic.recent(args.seconds)
    r = ear.transcribe(audio)
    print(f"\n  ({r.latency:.1f}s, ok={r.ok})  “{r.text}”")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
