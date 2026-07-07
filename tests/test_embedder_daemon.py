"""DaemonEmbedder client + best_embedder factory — the single-GPU-owner seam.

No real models and no real daemon: a stdlib fake server impersonates
``scripts/recall_embedder_daemon.py`` on an ephemeral port. What matters here:
batching (one POST per BATCH texts, not per text), the degraded-daemon path
(503 healthz WITH a diagnostic body -> fallback + urgent alert), and the
hermetic RECALL_NO_DAEMON switch conftest relies on.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from recall import index


class FakeEmbedder:
    dim = 4

    def embed(self, texts, *, is_query=False):
        return [[0.5] * self.dim for _ in texts]


def _fake_daemon(payloads: list, healthz: dict,
                 embed_error: dict | None = None) -> ThreadingHTTPServer:
    """A stdlib HTTP server impersonating the embedder daemon; /embed request
    bodies are recorded into ``payloads``. healthz mirrors the real daemon:
    200 when ok, 503 (with the same diagnostic body) when degraded.
    ``embed_error`` makes every /embed answer 500 with that body."""

    class H(BaseHTTPRequestHandler):
        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            self._reply(200 if healthz.get("ok") else 503, healthz)

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", "0") or "0")
            req = json.loads(self.rfile.read(n) or b"{}")
            payloads.append(req)
            if embed_error is not None:
                self._reply(500, embed_error)
                return
            texts = req.get("texts") or [req.get("text", "")]
            self._reply(200, {"embeddings": [[0.5] * 4 for _ in texts], "dim": 4})

        def log_message(self, *_args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_daemon_embedder_batches_per_request():
    payloads: list = []
    srv = _fake_daemon(payloads, {"ok": True, "dim": 4})
    try:
        emb = index.DaemonEmbedder(host="127.0.0.1", port=srv.server_address[1])
        assert emb.dim == 4
        texts = [f"note {i}" for i in range(index.DaemonEmbedder.BATCH + 3)]
        out = emb.embed(texts)
    finally:
        srv.shutdown()
    assert len(out) == len(texts)
    embeds = [p for p in payloads if "texts" in p]
    assert len(embeds) == 2  # BATCH texts + a 3-text remainder -> exactly two POSTs
    assert embeds[0]["texts"] == texts[:index.DaemonEmbedder.BATCH]
    assert embeds[1]["texts"] == texts[index.DaemonEmbedder.BATCH:]


def test_daemon_embedder_surfaces_500_body():
    # The daemon's diagnostic body (e.g. "CUDA out of memory") must reach the
    # caller — a bare HTTPError hid the first live batch-OOM entirely.
    import pytest
    srv = _fake_daemon([], {"ok": True, "dim": 4},
                       embed_error={"error": "CUDA out of memory"})
    try:
        emb = index.DaemonEmbedder(host="127.0.0.1", port=srv.server_address[1])
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            emb.embed(["a"])
    finally:
        srv.shutdown()


def test_best_embedder_honors_no_daemon_env(monkeypatch):
    sentinel = FakeEmbedder()
    monkeypatch.setattr(index, "SentenceTransformerEmbedder", lambda: sentinel)
    monkeypatch.setenv("RECALL_NO_DAEMON", "1")
    assert index.best_embedder() is sentinel


def test_best_embedder_uses_daemon_when_healthy(monkeypatch):
    payloads: list = []
    srv = _fake_daemon(payloads, {"ok": True, "dim": 4})
    try:
        monkeypatch.delenv("RECALL_NO_DAEMON", raising=False)
        monkeypatch.setenv("RECALL_EMBED_PORT", str(srv.server_address[1]))
        emb = index.best_embedder()
        assert isinstance(emb, index.DaemonEmbedder)
        assert emb.embed(["hello"]) == [[0.5] * 4]
    finally:
        srv.shutdown()


def test_best_embedder_falls_back_and_alerts_on_degraded(monkeypatch, capsys):
    # Degraded = daemon UP but healthz not ok (e.g. fell off its warm device).
    # Expect: in-process fallback + ONE urgent alert (alert_degraded=True) with
    # the diagnostic body surfaced from the 503.
    srv = _fake_daemon([], {"ok": False, "device": "cpu", "warm_device": "cuda:0"})
    alerts: list = []
    try:
        monkeypatch.delenv("RECALL_NO_DAEMON", raising=False)
        monkeypatch.setenv("RECALL_EMBED_PORT", str(srv.server_address[1]))
        sentinel = FakeEmbedder()
        monkeypatch.setattr(index, "SentenceTransformerEmbedder", lambda: sentinel)
        from recall import notify
        monkeypatch.setattr(
            notify, "notify_alert",
            lambda title, body, **kw: alerts.append((title, body)) or True)
        emb = index.best_embedder(alert_degraded=True)
    finally:
        srv.shutdown()
    assert emb is sentinel
    assert [t for t, _ in alerts] == ["recall embedder DEGRADED"]
    assert "cuda:0" in alerts[0][1]          # the 503 body reached the alert
    assert "daemon unavailable" in capsys.readouterr().err


def test_best_embedder_daemon_down_no_alert(monkeypatch):
    # Down (connection refused) is NORMAL on daemon-less boxes — fall back
    # silently to in-process, never page.
    srv = _fake_daemon([], {"ok": True, "dim": 4})
    port = srv.server_address[1]
    srv.shutdown()
    srv.server_close()   # port now closed -> connection refused
    alerts: list = []
    monkeypatch.delenv("RECALL_NO_DAEMON", raising=False)
    monkeypatch.setenv("RECALL_EMBED_PORT", str(port))
    sentinel = FakeEmbedder()
    monkeypatch.setattr(index, "SentenceTransformerEmbedder", lambda: sentinel)
    from recall import notify
    monkeypatch.setattr(notify, "notify_alert",
                        lambda *a, **kw: alerts.append(a) or True)
    assert index.best_embedder(alert_degraded=True) is sentinel
    assert alerts == []
