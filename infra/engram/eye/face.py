#!/usr/bin/env python3
"""Engram's face sense — recognize WHO is in the chair, to gate whether Engram engages.

Pipeline: **detect** (YuNet → box + 5 landmarks) → **align** (5-point similarity
warp to the ArcFace canonical 112×112) → **embed** (ArcFace MobileFaceNet
``w600k_mbf``, 512-d, via onnxruntime — a face → unit vector) → **match** against a
small enrolled gallery by cosine similarity. SmolVLM *describes* a face ("a person with
glasses"); this *identifies* it ("that's the user"). Built on the same OpenCV frames the
eye bench decodes — no new heavy deps beyond onnxruntime, just two small ONNX
models (auto-downloaded once).

The recognizer is a **seam**: the SFace 128-d embedder was swapped for ArcFace
``w600k_mbf`` 512-d on 2026-07-01 (clean-perception plan, Track A) — *callers did
not move*: ``sensorium``/``loop``/``awake``/``bridge`` still call
``FaceID().identify()/.is_present()``. The two embedding spaces are NOT comparable,
so ``gallery.npz`` is **stamped with the recognizer id** and a stamp mismatch fails
CLOSED (old 128-d enrollments are ignored, forcing a one-time re-enroll) — 128-d and
512-d can never silently mix.

    # one-time: teach Engram your face (look at the camera, turn your head a little)
    .venv/bin/python infra/engram/eye/face.py enroll YourName
    # live recognition window (green = known, amber = unknown)
    .venv/bin/python infra/engram/eye/face.py watch
    # headless check (who is in frame, no GUI)
    .venv/bin/python infra/engram/eye/face.py id --headless 5
    .venv/bin/python infra/engram/eye/face.py gallery        # who is enrolled
    # verify the two footguns without a camera (align crop + positive control):
    .venv/bin/python infra/engram/eye/face.py selftest --save /tmp/engram_align
    # set the operating threshold from data (you vs a negative face set):
    .venv/bin/python infra/engram/eye/face.py calibrate --capture 12 --neg-dir /path/to/not-you

Biometric data stays LOCAL: only face embeddings (not images) are stored, under
~/.local/share/recall/engram/faces/gallery.npz on this machine. `forget <name>`
deletes a person.
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

import cv2
import numpy as np
import onnxruntime as ort

FACE_DIR = Path(os.environ.get(
    "ENGRAM_FACE_DIR", os.path.expanduser("~/.local/share/recall/engram/faces")))
MODELS_DIR = FACE_DIR / "models"
GALLERY_PATH = FACE_DIR / "gallery.npz"

# The active recognizer id. Stamped into gallery.npz; a mismatch fails closed (see
# _load_gallery). Bump this whenever the embedder / its embedding space changes.
RECOGNIZER = "arcface_w600k_mbf"

# YuNet detects + gives 5 landmarks; ArcFace w600k_mbf (insightface *buffalo_s*
# recognizer) turns an aligned 112×112 face into a 512-d embedding. Both small,
# CPU-fast, auto-fetched once. The w600k_mbf single-file ONNX comes from the immich
# mirror (buffalo_s recognition head), verified 13.6MB, in=[N,3,112,112] out=[1,512].
MODEL_URLS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "w600k_mbf.onnx": "https://huggingface.co/immich-app/buffalo_s/resolve/main/"
                      "recognition/model.onnx",
}
RECOG_MODEL = "w600k_mbf.onnx"

# The ArcFace canonical destination landmarks (insightface standard) in the 112×112
# aligned frame, ordered image-left→right: [left-eye, right-eye, nose, left-mouth,
# right-mouth]. YuNet emits landmarks in the SAME image-left-first order
# (right-eye = the person's, i.e. image-left; then left-eye, nose, right-mouth,
# left-mouth), so YuNet's row[4:14] maps index-for-index onto this. THE footgun:
# if the aligned crop comes out mirrored/tilted, swap the two eye rows (and the two
# mouth rows). Never trust a cosine before the dumped crop looks like an upright face.
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# ArcFace cosine lives on a DIFFERENT scale than SFace (the old 0.40 was SFace-
# specific; this 0.40 is re-validated for ArcFace below, not inherited). Eval-gated
# per the plan, two independent checks on 2026-07-01:
#   • a 3-identity public positive control: same-identity 0.728 vs cross-identity <0.07.
#   • live calibrate (--capture 12 vs a 3-identity negative set): self 0.822–0.965,
#     impostor −0.05–0.07, margin +0.75, centroid crossover ~0.445.
# The gate matches by MAX cosine over samples (not the centroid the crossover uses), so
# we keep 0.40 — comfortably inside the ~0.75-wide gap and leaning strict: a false accept
# (stranger waved in) is worse than a false reject (ask the user to lean in). Re-tune per
# operator with `calibrate`.
DEFAULT_THRESHOLD = 0.40


def ensure_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        p = MODELS_DIR / name
        if not p.exists() or p.stat().st_size < 5000:
            print(f"fetching {name} …")
            urllib.request.urlretrieve(url, p)


@dataclass
class Face:
    name: str            # matched gallery name, or "unknown"
    score: float         # best cosine similarity to the gallery
    box: tuple           # (x, y, w, h)
    det: float           # detector confidence


class FaceID:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 det_score: float = 0.8, input_size=(320, 320)) -> None:
        ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            str(MODELS_DIR / "yunet.onnx"), "", input_size, score_threshold=det_score)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2                # perception runs on CPU; stay polite
        self.session = ort.InferenceSession(
            str(MODELS_DIR / RECOG_MODEL), sess_options=so,
            providers=["CPUExecutionProvider"])
        self._in = self.session.get_inputs()[0].name    # "input.1"
        self._out = self.session.get_outputs()[0].name  # "516"
        self.threshold = threshold
        self.gallery_model: str | None = None            # recognizer that WROTE the gallery
        self.gallery = self._load_gallery()   # name -> (N, 512) float32, L2-normalized
        if self.gallery_model is not None and self.gallery_model != RECOGNIZER:
            print(f"⚠ face gallery was built with '{self.gallery_model}' but the active "
                  f"recognizer is '{RECOGNIZER}' — ignoring the old enrollment (fail "
                  f"closed). Re-enroll: `face.py enroll <name>`.", file=sys.stderr)

    # ---- gallery persistence (model-stamped) ---------------------------------
    def _load_gallery(self) -> dict:
        """Load only if the stamp matches the active recognizer; else fail closed
        (return empty) so incompatible-dimension vectors can never reach _match."""
        if not GALLERY_PATH.exists():
            return {}
        data = np.load(GALLERY_PATH)
        files = list(data.files)
        # Legacy galleries (pre-swap SFace) have no stamp → treat as 'sface'.
        self.gallery_model = str(data["__model__"]) if "__model__" in files else "sface"
        if self.gallery_model != RECOGNIZER:
            return {}
        return {k: data[k] for k in files if k != "__model__"}

    def _save_gallery(self) -> None:
        GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(GALLERY_PATH, __model__=np.array(RECOGNIZER), **self.gallery)
        self.gallery_model = RECOGNIZER

    def forget(self, name: str) -> bool:
        if name in self.gallery:
            del self.gallery[name]
            self._save_gallery()
            return True
        return False

    # ---- core: detect -> align -> embed -> match -----------------------------
    def detect(self, frame) -> np.ndarray:
        """Return YuNet rows: (N, 15) = x,y,w,h, 5 landmarks(10), score."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def _align(self, frame, row) -> np.ndarray:
        """5-point similarity warp of the face onto the 112×112 ArcFace frame.

        Uses YuNet's landmarks (row[4:14]) and estimateAffinePartial2D — a 4-DOF
        similarity (rotation + uniform scale + translation, NO shear), exactly what
        ArcFace alignment wants. Falls back to a plain box-crop resize if the
        landmark solve degenerates, so embed never crashes on a bad detection."""
        lmk = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        M, _ = cv2.estimateAffinePartial2D(lmk, ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            x, y, w, h = (int(v) for v in row[:4])
            x, y = max(x, 0), max(y, 0)
            crop = frame[y:y + max(h, 1), x:x + max(w, 1)]
            if crop.size == 0:
                return np.zeros((112, 112, 3), np.uint8)
            return cv2.resize(crop, (112, 112))
        return cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)

    def embed(self, frame, row) -> np.ndarray:
        """Align by landmarks, run ArcFace, return an L2-normalized 512-d embedding.

        Preprocessing (the second footgun — get the normalization exactly right): map
        BGR→RGB and pixels [0,255]→[-1,1] exactly as insightface does, via
        blobFromImage(scale=1/127.5, mean=127.5, swapRB=True) → NCHW. A wrong
        mean/scale still yields plausible-looking cosines while quietly rotting —
        verify with the positive control (`selftest`) before trusting the gallery."""
        aligned = self._align(frame, row)
        blob = cv2.dnn.blobFromImage(
            aligned, scalefactor=1.0 / 127.5, size=(112, 112),
            mean=(127.5, 127.5, 127.5), swapRB=True)
        feat = self.session.run([self._out], {self._in: blob})[0].flatten().astype(np.float32)
        n = float(np.linalg.norm(feat))
        return feat / n if n else feat

    def _match(self, emb) -> tuple[str, float]:
        best_name, best = "unknown", -1.0
        for name, embs in self.gallery.items():
            s = float((embs @ emb).max())     # max cosine over that person's samples
            if s > best:
                best, best_name = s, name
        return (best_name if best >= self.threshold else "unknown"), max(best, 0.0)

    def identify(self, frame) -> list[Face]:
        out = []
        for row in self.detect(frame):
            name, score = self._match(self.embed(frame, row))
            x, y, w, h = row[:4].astype(int)
            out.append(Face(name, score, (int(x), int(y), int(w), int(h)), float(row[-1])))
        return out

    def is_present(self, frame, name: str) -> bool:
        """The engagement gate: is `name` one of the faces in this frame?"""
        return any(f.name == name for f in self.identify(frame))

    # ---- enrollment: collect embeddings of one person ------------------------
    def enroll(self, name: str, embeddings: list) -> int:
        arr = np.stack(embeddings).astype(np.float32)
        if name in self.gallery:
            arr = np.vstack([self.gallery[name], arr])
        self.gallery[name] = arr
        self._save_gallery()
        return arr.shape[0]


# ------------------------------- camera helpers -------------------------------
def open_cam(device: int, width=1280, height=720):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera /dev/video{device}")
    return cap


GREEN = (172, 239, 134)   # BGR of #86EFAC — known
AMBER = (36, 191, 251)    # BGR of #FBBF24 — unknown
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_faces(frame, faces: list[Face]) -> None:
    for f in faces:
        x, y, w, h = f.box
        color = GREEN if f.name != "unknown" else AMBER
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{f.name} {f.score:.2f}" if f.name != "unknown" else f"unknown {f.score:.2f}"
        cv2.rectangle(frame, (x, y - 22), (x + max(90, 9 * len(label)), y), color, -1)
        cv2.putText(frame, label, (x + 4, y - 6), FONT, 0.55, (10, 15, 20), 1, cv2.LINE_AA)


# --------------------------------- commands ----------------------------------
def cmd_enroll(args) -> int:
    fid = FaceID(threshold=args.threshold)
    cap = open_cam(args.device)
    target, samples = args.samples, []
    gui = not args.headless
    win = "Engram - enroll"
    if gui:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            gui = False
    print(f"Enrolling '{args.name}'. Look at the camera; turn your head slowly. "
          f"Collecting {target} samples…")
    last = 0.0
    while len(samples) < target:
        ok, frame = cap.read()
        if not ok:
            continue
        faces = fid.detect(frame)
        now = time.time()
        status = ""
        if len(faces) == 1 and now - last > 0.25:
            samples.append(fid.embed(frame, faces[0]))
            last = now
            status = f"captured {len(samples)}/{target}"
            print(" ", status)
        elif len(faces) == 0:
            status = "no face — move into view"
        elif len(faces) > 1:
            status = "multiple faces — only you in frame, please"
        if gui:
            for row in faces:
                x, y, w, h = row[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
            cv2.putText(frame, f"enroll {args.name}: {len(samples)}/{target}  {status}",
                        (12, 30), FONT, 0.7, GREEN, 2, cv2.LINE_AA)
            cv2.imshow(win, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                print("aborted."); cap.release(); cv2.destroyAllWindows(); return 1
    cap.release()
    if gui:
        cv2.destroyAllWindows()
    total = fid.enroll(args.name, samples)
    print(f"✓ enrolled '{args.name}' — {len(samples)} new samples ({total} total). "
          f"Saved to {GALLERY_PATH}  [{RECOGNIZER}]")
    return 0


def cmd_watch(args) -> int:
    fid = FaceID(threshold=args.threshold)
    if not fid.gallery:
        print("⚠ gallery is empty — run `face.py enroll <name>` first "
              "(everyone will show as 'unknown').")
    cap = open_cam(args.device)
    win = "Engram - face"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        cap.release()
        print(f"GUI unavailable ({exc}). Use `id --headless N`.")
        return 1
    print(f"Watching. Enrolled: {', '.join(fid.gallery) or '(none)'}.  q/Esc to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = fid.identify(frame)
        draw_faces(frame, faces)
        who = ", ".join(sorted({f.name for f in faces})) or "nobody"
        cv2.putText(frame, f"Engram sees: {who}", (12, 30), FONT, 0.7, GREEN, 2, cv2.LINE_AA)
        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def cmd_id(args) -> int:
    fid = FaceID(threshold=args.threshold)
    cap = open_cam(args.device)
    print(f"enrolled: {', '.join(fid.gallery) or '(none)'}  threshold={fid.threshold}")
    for i in range(args.headless or 1):
        frame = None
        for _ in range(12):
            ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{i+1}] read failed"); continue
        faces = fid.identify(frame)
        if not faces:
            print(f"[{i+1}] no face")
        for f in faces:
            print(f"[{i+1}] {f.name:>10}  cos={f.score:.3f}  box={f.box}  det={f.det:.2f}")
    cap.release()
    return 0


def cmd_gallery(args) -> int:
    fid = FaceID()
    stamp = fid.gallery_model or "(none)"
    if not fid.gallery:
        extra = "" if stamp in ("(none)", RECOGNIZER) else f"  (stale '{stamp}' ignored — re-enroll)"
        print(f"gallery empty.  recognizer={RECOGNIZER}{extra}")
        return 0
    print(f"recognizer={RECOGNIZER}  (gallery stamp: {stamp})")
    for name, embs in fid.gallery.items():
        print(f"  {name}: {embs.shape[0]} samples, dim={embs.shape[1]}")
    return 0


def cmd_forget(args) -> int:
    fid = FaceID()
    print(f"forgot '{args.name}'" if fid.forget(args.name) else f"'{args.name}' not enrolled")
    return 0


# ---- verification: the two footguns (alignment + preprocessing) --------------
def _embed_image_file(fid: FaceID, path: str):
    """Detect the largest face in an image file and return (embedding, aligned_crop)
    or (None, None). Used by selftest/calibrate — no camera needed."""
    img = cv2.imread(path)
    if img is None:
        return None, None
    faces = fid.detect(img)
    if len(faces) == 0:
        return None, None
    row = max(faces, key=lambda r: r[2] * r[3])   # largest box
    return fid.embed(img, row), fid._align(img, row)


def cmd_selftest(args) -> int:
    """Prove the ONNX path (shape/norm/self-cosine) and, if images are given, run the
    positive control: same-identity cosine must be HIGH, cross-identity LOW. Dumps the
    aligned crops so the alignment footgun can be eyeballed (read the PNGs)."""
    fid = FaceID(threshold=args.threshold)
    print(f"recognizer={RECOGNIZER}  in={fid._in} out={fid._out}  threshold={fid.threshold}")

    # 1) plumbing on a synthetic input — no camera, no faces
    dummy = np.full((112, 112, 3), 127, np.uint8)
    blob = cv2.dnn.blobFromImage(dummy, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    feat = fid.session.run([fid._out], {fid._in: blob})[0].flatten().astype(np.float32)
    n = float(np.linalg.norm(feat))
    unit = feat / n if n else feat
    ok_shape = feat.shape == (512,)
    ok_self = abs(float(unit @ unit) - 1.0) < 1e-4
    print(f"  plumbing: out-dim={feat.shape[0]} (want 512: {'✓' if ok_shape else '✗'})  "
          f"raw-norm={n:.3f}  self-cosine={float(unit @ unit):.4f} "
          f"({'✓' if ok_self else '✗'})")

    # 2) positive control on real faces, if provided
    imgs = args.image or []
    if len(imgs) >= 2:
        embs, names = [], []
        for p in imgs:
            e, crop = _embed_image_file(fid, p)
            tag = Path(p).stem
            if e is None:
                print(f"  ✗ no face detected in {p}"); continue
            embs.append(e); names.append(tag)
            if args.save:
                out = f"{args.save}_{tag}.png"
                cv2.imwrite(out, crop)
                print(f"  aligned crop → {out}  (open it: the face must be upright & centered)")
        print("  pairwise cosine (same identity → high, different → low):")
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                print(f"    {names[i]:>10} · {names[j]:<10} = {float(embs[i] @ embs[j]):+.3f}")
    elif args.save:
        print("  (pass ≥2 --image files to run the positive control + dump aligned crops)")
    return 0 if (ok_shape and ok_self) else 1


def cmd_calibrate(args) -> int:
    """Set the operating threshold from DATA, not a guess (plan step A.6). Positives =
    live camera capture of the target (or --pos-dir of images); negatives = --neg-dir
    of not-target faces (LFW subset / coworkers). Reports the self vs impostor cosine
    distributions against the target's centroid and the crossover threshold."""
    fid = FaceID()
    pos_embs: list = []
    if args.pos_dir:
        for p in sorted(Path(args.pos_dir).glob("*")):
            e, _ = _embed_image_file(fid, str(p))
            if e is not None:
                pos_embs.append(e)
        print(f"positives: {len(pos_embs)} from {args.pos_dir}")
    else:
        n = args.capture or 12
        cap = open_cam(args.device)
        print(f"Capturing {n} positive frames of '{args.person}'. Look at the camera; "
              "turn your head a little.")
        last = 0.0
        while len(pos_embs) < n:
            ok, frame = cap.read()
            if not ok:
                continue
            faces = fid.detect(frame)
            now = time.time()
            if len(faces) == 1 and now - last > 0.25:
                pos_embs.append(fid.embed(frame, faces[0])); last = now
                print(f"  captured {len(pos_embs)}/{n}")
        cap.release()
    if len(pos_embs) < 3:
        print("✗ not enough positive faces (need ≥3)."); return 1

    neg_embs: list = []
    if args.neg_dir:
        for p in sorted(Path(args.neg_dir).glob("*")):
            e, _ = _embed_image_file(fid, str(p))
            if e is not None:
                neg_embs.append(e)
    if len(neg_embs) < 1:
        print("✗ need a --neg-dir with ≥1 not-target face (e.g. an LFW subset)."); return 1

    pos = np.stack(pos_embs); neg = np.stack(neg_embs)
    cen = pos.mean(0); cen /= np.linalg.norm(cen) + 1e-9
    sc = pos @ cen        # positive-to-own-centroid
    ic = neg @ cen        # impostor-to-target-centroid
    margin = float(sc.min() - ic.max())
    thr = (float(sc.min()) + float(ic.max())) / 2
    print(f"\ncalibrate {time.strftime('%Y-%m-%d %H:%M:%S')}  person={args.person}  "
          f"(pos n={len(pos)}, neg n={len(neg)})")
    print(f"  self     : {sc.mean():.3f} [{sc.min():.3f}–{sc.max():.3f}]")
    print(f"  impostor : {ic.mean():.3f} [{ic.min():.3f}–{ic.max():.3f}]")
    if margin > 0:
        print(f"  CLEAN gap → set --threshold ~{thr:.2f}  (margin {margin:+.3f})")
    else:
        print(f"  OVERLAP → impostor max {ic.max():.3f} ≥ your min {sc.min():.3f}; "
              "capture more/cleaner frames or a bigger neg set.")
    log = FACE_DIR / "calibrate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {RECOGNIZER} person={args.person} "
                f"self[{sc.min():.3f}-{sc.max():.3f}] imp[{ic.min():.3f}-{ic.max():.3f}] "
                f"-> thr~{thr:.3f} margin{margin:+.3f}\n")
    print(f"  logged → {log}")
    return 0


def cmd_align(args) -> int:
    """Dump the aligned 112×112 crop for a single image (or one camera frame) so the
    alignment footgun can be verified visually. `selftest --image A B --save P` is the
    fuller check; this is the quick one-shot."""
    fid = FaceID()
    if args.image:
        e, crop = _embed_image_file(fid, args.image)
        if crop is None:
            print(f"✗ no face in {args.image}"); return 1
    else:
        cap = open_cam(args.device)
        frame = None
        for _ in range(12):
            ok, frame = cap.read()
        cap.release()
        faces = fid.detect(frame) if frame is not None else []
        if len(faces) == 0:
            print("✗ no face in camera frame"); return 1
        crop = fid._align(frame, max(faces, key=lambda r: r[2] * r[3]))
    cv2.imwrite(args.save, crop)
    print(f"aligned crop → {args.save}  (must be an upright, centred face)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Engram face sense — recognize who is in the chair")
    ap.add_argument("--device", type=int, default=0, help="camera index (usually 0)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="cosine match threshold (higher = stricter)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll", help="teach Engram a face")
    e.add_argument("name")
    e.add_argument("--samples", type=int, default=12)
    e.add_argument("--headless", action="store_true", help="no window (just capture)")
    e.set_defaults(func=cmd_enroll)
    w = sub.add_parser("watch", help="live recognition window")
    w.set_defaults(func=cmd_watch)
    i = sub.add_parser("id", help="identify faces in N frames")
    i.add_argument("--headless", type=int, default=1, metavar="N")
    i.set_defaults(func=cmd_id)
    sub.add_parser("gallery", help="list enrolled people").set_defaults(func=cmd_gallery)
    fo = sub.add_parser("forget", help="delete an enrolled person")
    fo.add_argument("name")
    fo.set_defaults(func=cmd_forget)
    st = sub.add_parser("selftest", help="verify ONNX plumbing + (with --image) positive control")
    st.add_argument("--image", nargs="*", help="face image files (≥2 for the positive control)")
    st.add_argument("--save", help="prefix to dump aligned crops to (e.g. /tmp/engram_align)")
    st.set_defaults(func=cmd_selftest)
    cal = sub.add_parser("calibrate", help="set the threshold from data (positives vs a negative set)")
    cal.add_argument("--person", default=getpass.getuser())
    cal.add_argument("--capture", type=int, default=12, help="live positive frames to grab")
    cal.add_argument("--pos-dir", help="use images here as positives instead of the camera")
    cal.add_argument("--neg-dir", help="folder of not-target faces (required)")
    cal.set_defaults(func=cmd_calibrate)
    al = sub.add_parser("align", help="dump one aligned crop (verify the alignment footgun)")
    al.add_argument("--image", help="image file (default: one camera frame)")
    al.add_argument("--save", default="/tmp/engram_align.png")
    al.set_defaults(func=cmd_align)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
