"""Shared test bootstrap.

RECALL_NO_DAEMON: the suite must be hermetic. On a dev box the warm embedder
daemon is usually up on :8973, and ``best_embedder()`` would silently route
test embeddings to the LIVE daemon — real vectors where tests expect
deterministic fakes, and test traffic hitting a production service. Force the
in-process path; individual tests that exercise the daemon path spin their own
fake server and clear this var via monkeypatch.
"""
import os

os.environ.setdefault("RECALL_NO_DAEMON", "1")
