#!/usr/bin/env python3
"""Tiny localhost embedder daemon for the recall hook.

Keeps ``Qwen3-Embedding-0.6B`` (bf16) warm on the GPU so the ``UserPromptSubmit``
recall hook gets a query vector in ~tens of ms instead of paying a multi-second
model load on every prompt. Stdlib HTTP only. If the daemon is down, the hook degrades
to keyword-only (FTS5) recall, so this service is an optimization, never a hard
dependency. One daemon serves every project on the machine.

Endpoints (127.0.0.1 only):
  POST /embed   {"text": "...", "is_query": true}      -> {"embedding": [...], "dim": N}
  POST /rerank  {"query": "...", "passages": [...]}    -> {"scores": [...]}
  GET  /healthz                                         -> {"ok": true, "dim": N}

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from recall.index import CrossEncoderReranker, SentenceTransformerEmbedder

HOST = os.environ.get("RECALL_EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("RECALL_EMBED_PORT", "8973"))

_embedder: SentenceTransformerEmbedder | None = None
_reranker: CrossEncoderReranker | None = None


def _get_embedder() -> SentenceTransformerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
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
        if self.path == "/healthz":
            self._reply(200, {"ok": True, "dim": _get_embedder().dim})
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/embed", "/rerank"):
            self._reply(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/embed":
                text = req.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    self._reply(400, {"error": "non-empty 'text' required"})
                    return
                vec = _get_embedder().embed(
                    [text], is_query=bool(req.get("is_query", True)))[0]
                self._reply(200, {"embedding": vec, "dim": len(vec)})
            else:  # /rerank
                query = req.get("query", "")
                passages = req.get("passages", [])
                if not isinstance(query, str) or not isinstance(passages, list):
                    self._reply(400, {"error": "'query' str + 'passages' list required"})
                    return
                scores = _get_reranker().score(query, [str(p) for p in passages])
                self._reply(200, {"scores": scores})
        except Exception as e:  # noqa: BLE001
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
