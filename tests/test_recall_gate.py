"""Tests for the PreToolUse gate hook's pure helpers — no hook I/O, no model.
scripts/recall_gate.py lives outside the package, so load it by path (mirrors
tests/test_recall_inject.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "recall_gate",
    Path(__file__).resolve().parent.parent / "scripts" / "recall_gate.py")
recall_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recall_gate)


def test_reach_kind_defaults_to_miss():
    # An unmarked reach is a genuine retrieval miss — the signal the log exists for.
    assert recall_gate._reach_kind(
        "cat /repo/docs/knowledge/archive/n.md") == "miss"


def test_reach_kind_reads_investigate_marker():
    assert recall_gate._reach_kind(
        "RECALL_REACH=investigate cat /repo/docs/knowledge/archive/n.md"
    ) == "investigate"


def test_reach_kind_stops_at_non_letter():
    # The token is letters only, so a path/flag right after it can't bleed in.
    assert recall_gate._reach_kind("RECALL_REACH=investigate; ls") == "investigate"
    assert recall_gate._reach_kind("RECALL_REACH= cat n.md") == "miss"


def test_protected_matches_corpus_paths_only():
    assert recall_gate._protected("cat /repo/docs/knowledge/n.md")
    assert recall_gate._protected("cat ~/.local/share/recall/global/n.md")
    assert not recall_gate._protected("cat /repo/src/recall/index.py")
