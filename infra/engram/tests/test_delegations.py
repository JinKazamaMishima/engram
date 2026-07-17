#!/usr/bin/env python3
"""Unit tests for the aurora m3 in-process delegation registry — the channel
that makes a grok tool call (not an SDK task) visible to the TUI panel.

    .venv/bin/python infra/engram/tests/test_delegations.py
"""
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

import delegations  # noqa: E402


def test_start_finish_snapshot():
    delegations.reset()
    assert delegations.snapshot() == {"live": [], "done": 0, "failed": 0,
                                      "cost": 0.0}
    a = delegations.start("grok·low", "grok-4.5", "xai")
    b = delegations.start("grok·high", "grok-4.5", "xai")
    assert a and b and a != b, "ids are non-zero and distinct"
    snap = delegations.snapshot()
    assert [e["label"] for e in snap["live"]] == ["grok·low", "grok·high"]
    assert snap["done"] == 0 and snap["failed"] == 0
    delegations.finish(a, cost=0.02, ok=True)
    snap = delegations.snapshot()
    assert [e["label"] for e in snap["live"]] == ["grok·high"], "finished one drops"
    assert snap["done"] == 1 and abs(snap["cost"] - 0.02) < 1e-9
    delegations.finish(b, ok=False)                      # error → failed tally
    snap = delegations.snapshot()
    assert snap["live"] == [] and snap["done"] == 1 and snap["failed"] == 1
    print("✓ start/finish/snapshot: live list, done/failed tallies, summed cost")


def test_finish_idempotent_and_safe_noops():
    delegations.reset()
    a = delegations.start("x")
    delegations.finish(a, cost=0.01, ok=True)
    delegations.finish(a, cost=0.01, ok=True)            # second finish: no-op
    delegations.finish(0)                                 # sentinel id: no-op
    delegations.finish(999)                               # unknown id: no-op
    snap = delegations.snapshot()
    assert snap["done"] == 1 and snap["failed"] == 0 and snap["live"] == []
    assert abs(snap["cost"] - 0.01) < 1e-9, "cost counted once, not twice"
    delegations.reset()
    assert delegations.snapshot()["cost"] == 0.0
    print("✓ finish is idempotent; 0 / unknown id are safe no-ops; reset clears")


def main() -> None:
    test_start_finish_snapshot()
    test_finish_idempotent_and_safe_noops()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
