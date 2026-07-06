"""Tests for the UserPromptSubmit recall hook's pure helpers — no model, no
daemon, no network (the functions take Hits / prompts directly). First coverage
for scripts/recall_inject.py, which lives outside the package."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from recall.index import Hit

_spec = importlib.util.spec_from_file_location(
    "recall_inject",
    Path(__file__).resolve().parent.parent / "scripts" / "recall_inject.py")
recall_inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recall_inject)


def test_format_context_groups_by_corpus_and_kind():
    hits = [Hit("g1-overnight", "g1 needs an overnight hold", "s", 0.9, "myproject"),
            Hit("owner-vision", "owner sets the vision", "s", 0.8, "global", "identity")]
    out = recall_inject._format_context(hits)
    assert "### This project" in out and "### Global / soul" in out
    assert "g1-overnight" in out and "owner-vision" in out
    assert "[identity]" in out                              # kind tag rendered
    assert out.index("g1-overnight") < out.index("owner-vision")  # project first


def test_format_context_omits_empty_section():
    out = recall_inject._format_context(
        [Hit("only-proj", "d", "s", 0.5, "myproject")])
    assert "### This project" in out and "### Global / soul" not in out


def test_recall_hits_short_prompt_is_empty():
    # fail-open guard: a trivial prompt surfaces nothing, no index touched
    assert recall_inject.recall_hits("ok", [("p", Path("/nope.sqlite"))]) == []


def test_recall_hits_missing_index_is_empty():
    assert recall_inject.recall_hits(
        "a real question about something",
        [("p", Path("/does/not/exist.sqlite"))]) == []


def test_format_system_message():
    msg = recall_inject._format_system_message(
        [Hit("a", "d", "s", 0.5, "myproject")])
    assert msg.startswith("🧠 recalled:") and "myproject:a" in msg


def test_format_context_labels_historical():
    hits = [Hit("old-deploy-plan", "deploy via GH Pages", "s", 0.9, "proj",
                valid_to="2020-01-02"),
            Hit("new-deploy-plan", "deploy via Cloudflare", "s", 0.8, "proj")]
    out = recall_inject._format_context(hits)
    assert "⏳ HISTORICAL (was true until 2020-01-02)" in out
    # The label rides the historical line only.
    current_line = [ln for ln in out.splitlines() if "new-deploy-plan" in ln][0]
    assert "HISTORICAL" not in current_line


def test_format_context_future_valid_to_not_historical():
    # A fact scheduled to expire is still true today — no label yet.
    hits = [Hit("expiring", "true until the far future", "s", 0.9, "proj",
                valid_to="2099-01-01")]
    assert "HISTORICAL" not in recall_inject._format_context(hits)
