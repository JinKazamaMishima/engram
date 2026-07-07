"""The suite must never touch live services — these canaries fail if the
conftest guards are ever removed. Found the hard way 2026-07-07: a pytest run
from the bridge shell (which carries the live Telegram creds) paged the operator's
phone with a dozen real ⚠ alerts from curate/dream/reconsolidate failure-path
fixtures."""
import os


def test_no_live_telegram_creds_in_suite_env():
    assert not os.environ.get("RECALL_TELEGRAM_TOKEN"), \
        "conftest must pop RECALL_TELEGRAM_TOKEN — tests may not page the operator"
    assert not os.environ.get("RECALL_TELEGRAM_CHAT_ID"), \
        "conftest must pop RECALL_TELEGRAM_CHAT_ID — tests may not page the operator"


def test_no_live_embedder_daemon_in_suite_env():
    assert os.environ.get("RECALL_NO_DAEMON") == "1", \
        "conftest must force the in-process embedder (no test traffic to :8973)"
