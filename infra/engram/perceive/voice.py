#!/usr/bin/env python3
"""Engram's voice sense — recognize WHOSE voice it is, to gate what the ear ingests.

The audio twin of ``eye/face.py``. SmolVLM *describes* a face; YuNet+SFace
*identify* it. Here Whisper *transcribes* speech; this *identifies the speaker* —
"is this the owner talking, or someone else / a phone call / a passer-by?" — so the ear
only ever feeds the owner's speech to the mind. Without this gate an office mic would
pollute the conversation with everyone in earshot; with it, every other voice is
"unknown" and dropped (or logged as ambient, never ingested).

Pipeline: **fbank** (kaldi-native-fbank, 80-dim) → **embed** (WeSpeaker CAM++
ONNX, 512-d, via onnxruntime — torch-free) → **match** a small enrolled gallery
by cosine. Same shape as face.py: an enrolled ``gallery.npz`` of L2-normalized
embeddings, cosine threshold, enroll/id/watch/gallery/forget. The recognizer is a
seam — swap a bigger speaker model later without changing callers.

    .venv/bin/python infra/engram/perceive/voice.py enroll YourName # teach Engram your voice
    .venv/bin/python infra/engram/perceive/voice.py watch         # live: who's speaking
    .venv/bin/python infra/engram/perceive/voice.py id --headless 5
    .venv/bin/python infra/engram/perceive/voice.py gallery

Biometric data stays LOCAL: only voice embeddings (not audio) are stored, under
~/.local/share/recall/engram/voices/gallery.npz. `forget <name>` deletes a person.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mic import RATE, Microphone  # noqa: E402 — the ear's capture feed

VOICE_DIR = Path(os.environ.get(
    "ENGRAM_VOICE_DIR", os.path.expanduser("~/.local/share/recall/engram/voices")))
MODELS_DIR = VOICE_DIR / "models"
GALLERY_PATH = VOICE_DIR / "gallery.npz"
MODEL_PATH = MODELS_DIR / "wespeaker_en_voxceleb_campplus.onnx"
CALIB_LOG = VOICE_DIR / "calibrate.log"          # calibrate tees its report here (readable)
CALIB_NPZ = VOICE_DIR / "calibration.npz"        # …and saves the embeddings for offline re-analysis
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx")  # sic: tag typo

# CAM++ same-speaker cosine sits well above this; different speakers fall below.
# Strict-ish for an "is this the owner" gate — a false accept (a guest waved in) is
# worse than a false reject (ask the owner to say a bit more). Tune with --threshold.
DEFAULT_THRESHOLD = 0.55    # calibrated on a clean two-speaker set: other ~0.42 vs owner 0.65-0.96 (centroid)
SPEECH_RMS = 0.012          # absolute silence floor (below this the window is just room tone)
MIN_VOICED_SEC = 0.8        # need ~0.8s of actual voiced speech to embed a reliable identity
VAD_RATIO = 2.5             # a 25ms frame is "voiced" if its energy > VAD_RATIO x noise floor


def ensure_model() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
        print(f"fetching speaker model (28MB) → {MODEL_PATH} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@dataclass
class Speaker:
    name: str            # matched gallery name, or "unknown"
    score: float         # best cosine similarity to the gallery
    rms: float           # energy of the analyzed window (for the caller's VAD)


class VoiceID:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, model_path: Path = MODEL_PATH) -> None:
        ensure_model()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2            # ear runs on CPU; keep it polite
        self.session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.gallery = self._load_gallery()    # name -> (N, 512) float32, L2-normalized

    # ---- gallery persistence (identical shape to face.py) --------------------
    def _load_gallery(self) -> dict:
        if GALLERY_PATH.exists():
            data = np.load(GALLERY_PATH)
            return {k: data[k] for k in data.files}
        return {}

    def _save_gallery(self) -> None:
        GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GALLERY_PATH, **self.gallery)

    def forget(self, name: str) -> bool:
        if name in self.gallery:
            del self.gallery[name]
            self._save_gallery()
            return True
        return False

    # ---- core: fbank -> embed -> match ---------------------------------------
    def _fbank(self, samples: np.ndarray) -> np.ndarray:
        """80-dim kaldi fbank for the WeSpeaker CAM++ ONNX, using the recipe the model's own
        ONNX metadata dictates: samples in INT16 SCALE (×32768 — the model says
        normalize_samples=0), dither 0, and NO external cepstral mean-norm (the export bakes
        feature-norm into the graph; adding CMN destroys speaker identity). Validated on known
        speakers at matched length: same-speaker ~0.82 vs different ~0.45 (margin +0.37);
        feeding [-1,1] instead collapsed it to length-dominated noise (margin +0.08), and CMN
        collapsed it further."""
        wav = samples.astype(np.float32) * 32768.0
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = float(RATE)
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = False
        opts.mel_opts.num_bins = 80
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(float(RATE), wav)
        fb.input_finished()
        return np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)

    def _voiced_seconds(self, x: np.ndarray, frame_ms: int = 25, hop_ms: int = 10) -> float:
        """Total voiced duration (s) — used ONLY to gate whether there's enough speech to
        embed. We deliberately do NOT trim to it: this embedding is length/framing-sensitive
        (trimming 0.3s swung same-speaker cosine 0.87→0.51), so we embed the FULL window for
        stability and just keep silence-only windows from reaching the model."""
        fl = int(RATE * frame_ms / 1000)
        hop = int(RATE * hop_ms / 1000)
        if x.size < fl:
            return 0.0
        frames = np.lib.stride_tricks.sliding_window_view(x, fl)[::hop]
        e = np.sqrt(np.mean(frames * frames, axis=1))
        floor = np.percentile(e, 20)                       # noise floor of this window
        thr = max(SPEECH_RMS, floor * VAD_RATIO)
        return int(np.count_nonzero(e > thr)) * hop / RATE

    def _embed(self, audio: np.ndarray):
        feats = self._fbank(audio)
        if feats.shape[0] < 50:
            return None
        embs = self.session.run(["embs"], {"feats": feats[None]})[0][0].astype(np.float32)
        n = float(np.linalg.norm(embs))
        return embs / n if n else embs

    def embed(self, samples: np.ndarray):
        """A unit-norm 512-d speaker embedding of this window, or None if it doesn't hold
        ~MIN_VOICED_SEC of actual speech (pure silence / room-tone is gated out)."""
        if self._voiced_seconds(samples) < MIN_VOICED_SEC:
            return None
        return self._embed(samples)

    def _match(self, emb) -> tuple[str, float]:
        # Match against each person's CENTROID (mean voiceprint), not the max over
        # samples: max-matching lets one broad/odd enrollment sample over-accept an
        # impostor, whereas the centroid is the robust, stricter summary of a voice.
        best_name, best = "unknown", -1.0
        for name, embs in self.gallery.items():
            c = embs.mean(0)
            c /= np.linalg.norm(c) + 1e-9
            s = float(c @ emb)
            if s > best:
                best, best_name = s, name
        return (best_name if best >= self.threshold else "unknown"), max(best, 0.0)

    def identify(self, samples: np.ndarray) -> Speaker:
        """Who is speaking in this audio window? (one dominant speaker assumed)."""
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        if self._voiced_seconds(samples) < MIN_VOICED_SEC:
            return Speaker("silence" if rms < SPEECH_RMS else "too-short", 0.0, rms)
        emb = self._embed(samples)
        if emb is None:
            return Speaker("too-short", 0.0, rms)
        name, score = self._match(emb)
        return Speaker(name, score, rms)

    def is_speaker(self, samples: np.ndarray, name: str) -> bool:
        """The ear's gate: is `name` the one speaking in this window?"""
        return self.identify(samples).name == name

    # ---- enrollment ----------------------------------------------------------
    def enroll(self, name: str, embeddings: list) -> int:
        arr = np.stack(embeddings).astype(np.float32)
        if name in self.gallery:
            arr = np.vstack([self.gallery[name], arr])
        self.gallery[name] = arr
        self._save_gallery()
        return arr.shape[0]


# --------------------------------- commands ----------------------------------
def cmd_enroll(args) -> int:
    vid = VoiceID(threshold=args.threshold)
    print(f"Enrolling '{args.name}'. Talk naturally for ~{args.seconds:.0f}s — read "
          "something aloud, vary your tone. (Only YOU should be speaking.)")
    embs, win = [], args.window
    with Microphone() as mic:
        time.sleep(0.3)
        t_end = time.time() + args.seconds
        last = 0.0
        while time.time() < t_end:
            now = time.time()
            if now - last >= win * 0.6:                 # overlapping windows
                seg = mic.recent(win)
                e = vid.embed(seg)
                rms = mic.rms(win)
                if e is not None and rms >= SPEECH_RMS:
                    embs.append(e)
                    last = now
                    print(f"  captured sample {len(embs)}  (rms={rms:.3f})")
                elif rms < SPEECH_RMS:
                    print("  …quiet — keep talking", end="\r", flush=True)
            time.sleep(0.15)
    if len(embs) < 3:
        print(f"✗ only {len(embs)} usable speech samples — too quiet/short. Speak up, try again.")
        return 1
    # Outlier-reject: keep samples that agree with the consensus voice, so one stray
    # capture (a cough, a passing voice) can't broaden the gallery into false accepts.
    arr = np.stack(embs)
    centroid = arr.mean(0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sims = arr @ centroid
    kept = [e for e, s in zip(embs, sims) if s >= 0.45]
    dropped = len(embs) - len(kept)
    if len(kept) < 3:
        kept = embs                                   # consensus too loose — keep all, warn
        print("  ⚠ samples disagree a lot — enroll in a quieter spot for a tighter voiceprint.")
    elif dropped:
        print(f"  dropped {dropped} outlier sample(s) (cos<0.45 to your voice consensus)")
    karr = np.stack(kept)
    if len(karr) >= 2:
        cs = [float(karr[i] @ karr[j]) for i in range(len(karr)) for j in range(i + 1, len(karr))]
        q = float(np.mean(cs))
        verdict = "clean" if q > 0.55 else ("usable" if q > 0.4 else "NOISY — re-enroll closer/quieter")
        print(f"  voiceprint consistency: intra-cosine mean {q:.2f}, min {min(cs):.2f}  → {verdict}")
    total = vid.enroll(args.name, kept)
    print(f"✓ enrolled '{args.name}' — {len(kept)} samples ({total} total). Saved to {GALLERY_PATH}")
    print("  next: `voice.py watch` — read your cosine vs the other person's; we set --threshold between them.")
    return 0


def cmd_watch(args) -> int:
    vid = VoiceID(threshold=args.threshold)
    if not vid.gallery:
        print("⚠ gallery empty — run `voice.py enroll <name>` first (all will be 'unknown').")
    print(f"Listening. Enrolled: {', '.join(vid.gallery) or '(none)'}.  Ctrl-C to quit.")
    try:
        with Microphone() as mic:
            time.sleep(0.4)
            while True:
                spk = vid.identify(mic.recent(args.window))
                bars = int(min(spk.rms * 4000, 30))
                tag = spk.name if spk.name not in ("silence", "too-short") else f"({spk.name})"
                print(f"  |{'█' * bars:<30}| {tag:<10} cos={spk.score:.2f}", end="\r", flush=True)
                time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


def cmd_id(args) -> int:
    vid = VoiceID(threshold=args.threshold)
    print(f"enrolled: {', '.join(vid.gallery) or '(none)'}  threshold={vid.threshold}")
    with Microphone() as mic:
        time.sleep(0.4)
        for i in range(args.headless or 1):
            time.sleep(args.window)
            spk = vid.identify(mic.recent(args.window))
            print(f"  [{i+1}] {spk.name:>10}  cos={spk.score:.3f}  rms={spk.rms:.3f}")
    return 0


def cmd_calibrate(args) -> int:
    """Measure the real separation between the enrolled person and an impostor
    (a second speaker), so the threshold is set from DATA, not guessed. Captures both voices
    live and reports cosine under max-match AND centroid matching + a recommendation."""
    vid = VoiceID()
    if args.person not in vid.gallery:
        print(f"⚠ '{args.person}' not enrolled — run `voice.py enroll {args.person}` first.")
        return 1
    garr = vid.gallery[args.person]
    cen = garr.mean(0)
    cen /= np.linalg.norm(cen) + 1e-9

    def capture(label: str) -> np.ndarray | None:
        input(f"\n[{label}] press Enter, then talk continuously ~{args.seconds:.0f}s…")
        embs, last = [], 0.0
        with Microphone() as mic:
            time.sleep(0.3)
            t_end = time.time() + args.seconds
            while time.time() < t_end:
                now = time.time()
                if now - last >= args.window * 0.6:
                    e = vid.embed(mic.recent(args.window))
                    if e is not None:
                        embs.append(e); last = now
                        print(f"  captured {len(embs)}", end="\r", flush=True)
                time.sleep(0.15)
        print()
        return np.stack(embs) if embs else None

    self_e = capture(f"{args.person} (you) speaking")
    imp_e = capture("the OTHER person speaking")
    if self_e is None or imp_e is None:
        print("✗ not enough voiced speech captured — speak up / closer, try again.")
        return 1

    def sc(E):
        return (garr @ E.T).max(0), E @ cen          # per-sample: max-match, centroid
    smx, sct = sc(self_e)
    imx, ict = sc(imp_e)
    lines = [
        f"calibration {time.strftime('%Y-%m-%d %H:%M:%S')}  person={args.person}  "
        f"(self n={len(self_e)}, impostor n={len(imp_e)})",
        f"  {args.person} self : max-match {smx.mean():.2f} [{smx.min():.2f}-{smx.max():.2f}]   "
        f"centroid {sct.mean():.2f} [{sct.min():.2f}-{sct.max():.2f}]",
        f"  impostor   : max-match {imx.mean():.2f} [{imx.min():.2f}-{imx.max():.2f}]   "
        f"centroid {ict.mean():.2f} [{ict.min():.2f}-{ict.max():.2f}]",
    ]
    best = None
    for label, s, i in (("max-match", smx, imx), ("centroid", sct, ict)):
        margin = float(s.min() - i.max())
        thr = (float(s.min()) + float(i.max())) / 2
        if margin > 0:
            lines.append(f"  [{label}] CLEAN gap -> threshold ~{thr:.2f} (margin {margin:+.2f})")
            if best is None or margin > best[1]:
                best = (label, margin, thr)
        else:
            lines.append(f"  [{label}] OVERLAP -> impostor max {i.max():.2f} >= your min {s.min():.2f}")
    lines.append(f"  => best: {best[0]} matching, threshold ~{best[2]:.2f}" if best
                 else "  => voices overlap under both matchers; need AS-norm cohort score-norm")
    report = "\n".join(lines)
    print("\n" + report)
    # Persist so the result can be read + re-analyzed without recapturing (the input
    # box can't paste). Embeddings let me try AS-norm / other thresholds offline.
    CALIB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CALIB_NPZ, self_emb=self_e, impostor_emb=imp_e, gallery=garr)
    with open(CALIB_LOG, "a") as f:
        f.write(report + "\n\n")
    print(f"\n  saved report -> {CALIB_LOG}")
    print(f"  saved embeddings -> {CALIB_NPZ}   (I can read both)")
    return 0


def cmd_gallery(args) -> int:
    vid = VoiceID()
    if not vid.gallery:
        print("gallery empty.")
        return 0
    for name, embs in vid.gallery.items():
        print(f"  {name}: {embs.shape[0]} samples, dim={embs.shape[1]}")
    return 0


def cmd_forget(args) -> int:
    vid = VoiceID()
    print(f"forgot '{args.name}'" if vid.forget(args.name) else f"'{args.name}' not enrolled")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram voice sense — recognize who is speaking")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="cosine match threshold (higher = stricter)")
    ap.add_argument("--window", type=float, default=3.0, help="analysis window seconds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll", help="teach Engram a voice")
    e.add_argument("name")
    e.add_argument("--seconds", type=float, default=15.0)
    e.set_defaults(func=cmd_enroll)
    sub.add_parser("watch", help="live: who is speaking").set_defaults(func=cmd_watch)
    i = sub.add_parser("id", help="identify the speaker in N windows")
    i.add_argument("--headless", type=int, default=1, metavar="N")
    i.set_defaults(func=cmd_id)
    cal = sub.add_parser("calibrate", help="measure you vs an impostor; recommend a threshold")
    cal.add_argument("--person", default=os.environ.get("ENGRAM_USER") or getpass.getuser(), help="the enrolled person to test against")
    cal.add_argument("--seconds", type=float, default=8.0)
    cal.set_defaults(func=cmd_calibrate)
    sub.add_parser("gallery", help="list enrolled voices").set_defaults(func=cmd_gallery)
    fo = sub.add_parser("forget", help="delete an enrolled voice")
    fo.add_argument("name")
    fo.set_defaults(func=cmd_forget)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
