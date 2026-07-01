# Engram's eye

Local vision for Engram. A small VLM (SmolVLM) served by llama.cpp reads the
webcam; the bench is a live window so you can *see* that objects (and, next,
faces) are recognized correctly. This is the first draft of the eye in the
perceiving loop — `eye.py` is display-free on purpose so it drops straight in.

## 1. Start the backend (once)

SmolVLM-500M on `llama-server`, localhost only (point `LLAMA` at wherever you
unpacked llama.cpp):

```bash
LLAMA=/path/to/llama.cpp            # dir containing the llama-server binary
LD_LIBRARY_PATH=$LLAMA $LLAMA/llama-server \
    -hf ggml-org/SmolVLM-500M-Instruct-GGUF --host 127.0.0.1 --port 8080
```

Runs on CPU (~0.6–1.8 s/frame, plenty for a bench). For real-time later, the
Vulkan prebuilt (`llama-*-bin-ubuntu-vulkan-x64`) offloads to a GPU with no CUDA
toolkit needed — as long as `libvulkan` is present.

## 2. Watch

```bash
.venv/bin/python infra/engram/eye/bench.py              # live window
.venv/bin/python infra/engram/eye/bench.py --headless 5 # 5 captions to stdout, no GUI
```

Window keys: `p` cycle prompt · `s` snapshot · `q`/`Esc` quit.

## Pieces

- **`eye.py`** — `Eye.look(jpeg) -> Reading`. Pure VLM client (no camera/display),
  the reusable seam; point it at a bigger VLM by changing the server/model.
- **`bench.py`** — webcam capture (OpenCV) + live overlay window + a `--headless`
  mode for verification over SSH/CI.

## Face identification — `face.py` (shipped)

Recognize *who* is in the chair, not just "a face": **YuNet** detect → **SFace**
embed (128-d) → cosine-match a small enrolled gallery. Zero new deps — OpenCV
built-ins + two small auto-downloaded ONNX models, on the same frames the eye
decodes. SmolVLM *describes* faces; this *identifies* them.

```bash
.venv/bin/python infra/engram/eye/face.py enroll YourName   # teach it your face
.venv/bin/python infra/engram/eye/face.py watch             # live: green=known, amber=unknown
.venv/bin/python infra/engram/eye/face.py id --headless 5
.venv/bin/python infra/engram/eye/face.py gallery           # who's enrolled
.venv/bin/python infra/engram/eye/face.py forget YourName
```

`FaceID().is_present(frame, "YourName")` is the **engagement-gate** hook for the
perceiving loop — let *who* is present decide whether Engram greets. Default cosine
threshold 0.40 (strict on purpose); tune with `--threshold`. Privacy: only
embeddings (not images) are stored, locally at
`~/.local/share/recall/engram/faces/gallery.npz`.

## Next: one capture source

Both `bench.py` and `face.py` open `/dev/video0` directly, so only one runs at a
time (V4L2 is single-consumer). The perceiving loop will own a single capture and
fan frames to both the eye (VLM) and the face sense.
