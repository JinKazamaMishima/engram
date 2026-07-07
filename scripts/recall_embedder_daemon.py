#!/usr/bin/env python3
"""Tiny localhost embedder daemon for the recall hook.

Keeps ``Qwen3-Embedding-0.6B`` (bf16) warm on the GPU so the ``UserPromptSubmit``
recall hook gets a query vector in ~tens of ms instead of paying a multi-second
model load on every prompt. Stdlib HTTP only. If the daemon is down, the hook degrades
to keyword-only (FTS5) recall, so this service is an optimization, never a hard
dependency. One daemon serves every project on the machine.

Endpoints (127.0.0.1 only):
  POST /embed   {"text": "...", "is_query": true}      -> {"embedding": [...], "dim": N}
  POST /embed   {"texts": [...], "is_query": true}     -> {"embeddings": [[...]], "dim": N}
  POST /rerank  {"query": "...", "passages": [...]}    -> {"scores": [...]}
  GET  /healthz  REAL control, not a liveness ping: runs one tiny encode and
                 reports the serving device -> 200 {"ok": true, "dim": N,
                 "device": "cuda:0", "warm_device": "cuda:0", "probe_ms": f};
                 503 ok:false when the model left its warm-time device or the
                 probe fails (2026-07-07: the daemon fell off the GPU and served
                 CPU-slow for hours while a liveness-only healthz stayed green).

The reranker (Qwen3-Reranker-0.6B, bf16) is loaded lazily on the first /rerank
call — startup only warms the embedder, so machines that never rerank never pay
for the reranker's VRAM.

Run via systemd --user (infra/systemd/recall-embedder.service) or directly:
  .venv/bin/python scripts/recall_embedder_daemon.py
Config: RECALL_EMBED_HOST (default 127.0.0.1), RECALL_EMBED_PORT (8973).
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from recall.index import CrossEncoderReranker, SentenceTransformerEmbedder

HOST = os.environ.get("RECALL_EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("RECALL_EMBED_PORT", "8973"))
EMBED_MAX_TEXTS = 256   # per-request batch cap — clients chunk above this

_embedder: SentenceTransformerEmbedder | None = None
_reranker: CrossEncoderReranker | None = None
_warm_device: str | None = None   # device the model warmed on; drift = degraded


def _model_device(emb: SentenceTransformerEmbedder) -> str:
    try:
        return str(emb._model.device)  # noqa: SLF001 — our own class
    except Exception:  # noqa: BLE001
        try:
            return str(next(iter(emb._model.parameters())).device)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return "unknown"


def _get_embedder() -> SentenceTransformerEmbedder:
    global _embedder, _warm_device
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
        _warm_device = _model_device(_embedder)
    return _embedder


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:  # lazy: only loaded when /rerank is first called
        _reranker = CrossEncoderReranker()
    return _reranker


class _Handler(BaseHTTPRequestHandler):
    def _reply(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._reply(404, {"error": "not found"})
            return
        # A REAL control, not a liveness ping: one tiny encode + the serving
        # device. The 2026-07-07 outage had the model silently off its warm GPU,
        # serving CPU-slow, while liveness-healthz stayed green (the 4th vacuous
        # detector). A failing probe or a device drift IS the unhealthy signal.
        try:
            emb = _get_embedder()
            t0 = time.monotonic()
            emb.embed(["healthz probe"], is_query=True)
            ms = round((time.monotonic() - t0) * 1000, 1)
            device = _model_device(emb)
            ok = device == _warm_device
            self._reply(200 if ok else 503,
                        {"ok": ok, "dim": emb.dim, "device": device,
                         "warm_device": _warm_device, "probe_ms": ms})
        except Exception as e:  # noqa: BLE001 — a failing probe IS the signal
            self._reply(503, {"ok": False, "error": str(e)})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/embed", "/rerank"):
            self._reply(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/embed":
                # Batch shape ({"texts": [...]}) or the legacy single-text shape
                # ({"text": "..."}) — the hook still posts the latter per turn.
                texts = req.get("texts")
                single = texts is None
                if single:
                    t = req.get("text", "")
                    texts = [t] if isinstance(t, str) else []
                if (not isinstance(texts, list) or not texts
                        or len(texts) > EMBED_MAX_TEXTS
                        or not all(isinstance(t, str) and t.strip() for t in texts)):
                    self._reply(400, {"error": "non-empty 'text' or 'texts' "
                                      f"(1..{EMBED_MAX_TEXTS} strings) required"})
                    return
                is_q = bool(req.get("is_query", True))
                try:
                    vecs = _get_embedder().embed(list(texts), is_query=is_q)
                except Exception as e:  # noqa: BLE001
                    if "out of memory" not in str(e).lower() or len(texts) == 1:
                        raise
                    # A batch of LONG passages can spike attention VRAM past the
                    # card even for the daemon itself (encode pads the batch to
                    # its longest text). Shed the spike: free the cache, embed
                    # one text at a time — slow beats a 500 stranding a rebuild.
                    print(f"[recall-embedder] batch of {len(texts)} OOMed; "
                          "retrying per-text", file=sys.stderr, flush=True)
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001 — cache purge is best-effort
                        pass
                    vecs = [_get_embedder().embed([t], is_query=is_q)[0]
                            for t in texts]
                if single:
                    self._reply(200, {"embedding": vecs[0], "dim": len(vecs[0])})
                else:
                    self._reply(200, {"embeddings": vecs, "dim": len(vecs[0])})
            else:  # /rerank
                query = req.get("query", "")
                passages = req.get("passages", [])
                if not isinstance(query, str) or not isinstance(passages, list):
                    self._reply(400, {"error": "'query' str + 'passages' list required"})
                    return
                scores = _get_reranker().score(query, [str(p) for p in passages])
                self._reply(200, {"scores": scores})
        except Exception as e:  # noqa: BLE001
            # Log BEFORE replying: a silent 500 (body-only error) made the
            # first batch-OOM undiagnosable from the journal (2026-07-07).
            import traceback
            print(f"[recall-embedder] {self.path} failed: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            self._reply(500, {"error": str(e)})

    def log_message(self, *_args) -> None:  # silence per-request logging
        pass


def main() -> int:
    _get_embedder()  # warm the model before accepting traffic
    srv = ThreadingHTTPServer((HOST, PORT), _Handler)
    print(f"[recall-embedder] ready on {HOST}:{PORT} (dim={_get_embedder().dim})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
