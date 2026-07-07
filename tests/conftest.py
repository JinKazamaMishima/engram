"""Shared test bootstrap.

RECALL_NO_DAEMON: the suite must be hermetic. On a dev box the warm embedder
daemon is usually up on :8973, and ``best_embedder()`` would silently route
test embeddings to the LIVE daemon — real vectors where tests expect
deterministic fakes, and test traffic hitting a production service. Force the
in-process path; individual tests that exercise the daemon path spin their own
fake server and clear this var via monkeypatch.

Telegram creds are POPPED for the same reason: ``notify_alert`` fires on env
creds, and a shell that carries them (the bridge session — it IS the Telegram
service — or any sourced telegram.env) would page the operator's PHONE with
every failure-path fixture the suite exercises (found 2026-07-07: a pytest run
from the bridge spammed a dozen real ⚠ alerts). Tests that exercise sending
set their own fake creds via monkeypatch.
"""
import os

os.environ.setdefault("RECALL_NO_DAEMON", "1")
os.environ.pop("RECALL_TELEGRAM_TOKEN", None)
os.environ.pop("RECALL_TELEGRAM_CHAT_ID", None)
