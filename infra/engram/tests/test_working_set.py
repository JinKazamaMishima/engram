#!/usr/bin/env python3
"""Headless tests for the working-set builder (Brick 3 tier 2): deterministic
assembly from the LiveBuffer + activation log, validity filtering, budget
truncation, and the fail-open contract.

    .venv/bin/python infra/engram/tests/test_working_set.py
"""
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)
REPO = os.path.abspath(os.path.join(ENGRAM, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from buffer import LiveBuffer  # noqa: E402
import working_set as ws  # noqa: E402

# Activation/validity note-surfacing lives in the recall ENGINE (separate mirror track);
# skip the note tests when this repo's engine predates Brick-3 (the buffer/render tests
# are engine-independent). Probe a concrete Brick-3 engine symbol as the gate.
from recall import transcripts as _T  # noqa: E402
_ENGINE_HAS_BRICK3 = hasattr(_T, "iter_buffer_exchanges")


def _buf(dir_, rows):
    holder = {"id": "conv-1"}
    b = LiveBuffer(Path(dir_), lambda: holder["id"])
    for role, text in rows:
        b.append(role, text)
    return b


def test_empty_when_no_buffer():
    # Isolated data root so no real activation log bleeds notes into the block.
    with tempfile.TemporaryDirectory() as data:
        os.environ["RECALL_DATA_ROOT"] = str(data)
        assert ws.build_working_memory(None, Path(".")) == ""
        with tempfile.TemporaryDirectory() as d:
            off = LiveBuffer(None, lambda: "x")           # disabled buffer
            assert ws.build_working_memory(off, Path(d)) == ""
            empty = LiveBuffer(Path(d), lambda: "conv")   # enabled but no rows
            # no turns, no activation → nothing to ground
            assert ws.build_working_memory(empty, Path(d)) == ""
    print("✓ empty/disabled buffer → '' (nothing to ground, fail-open)")


def test_recent_turns_rendered_newest_survives():
    with tempfile.TemporaryDirectory() as d:
        b = _buf(d, [("user", "first question"),
                     ("assistant", "first answer"),
                     ("user", "second question"),
                     ("assistant", "second answer")])
        block = ws.build_working_memory(b, Path(d), notes=0)
        assert "<working-memory>" in block and "</working-memory>" in block
        assert "## Recent turns" in block
        assert "[OPERATOR] second question" in block
        assert "[ENGRAM] second answer" in block
        # roles are relabeled, not raw
        assert "[user]" not in block and "assistant]" not in block.split("ENGRAM")[0]
    print("✓ recent turns rendered with OPERATOR/ENGRAM labels, newest present")


def test_turns_window_and_per_turn_cap():
    with tempfile.TemporaryDirectory() as d:
        rows = [("user", f"msg {i}") for i in range(20)]
        b = _buf(d, rows)
        block = ws.build_working_memory(b, Path(d), turns=3, notes=0)
        assert "msg 19" in block and "msg 16" not in block   # only last 3
        # a huge single turn is capped, block stays bounded
        b2 = _buf(d + "/x" if False else d, [])
    with tempfile.TemporaryDirectory() as d2:
        big = _buf(d2, [("user", "x" * 5000)])
        block = ws.build_working_memory(big, Path(d2), notes=0)
        assert "…" in block and len(block) < 2000
    print("✓ turns window honored; a giant single turn is capped, not unbounded")


def test_budget_truncation_drops_oldest_turns_first():
    with tempfile.TemporaryDirectory() as d:
        rows = [("user", f"turn number {i} with some padding text") for i in range(12)]
        b = _buf(d, rows)
        block = ws.build_working_memory(b, Path(d), notes=0, budget=400)
        assert len(block) <= 400
        assert "turn number 11" in block          # newest survives …
        assert "turn number 0" not in block        # … oldest dropped first
    print("✓ budget truncation keeps the newest turn, drops oldest first")


def _setup_corpus(tmp, notes):
    """Write notes + activation events into an isolated RECALL_DATA_ROOT and a
    project corpus; return (cwd, restore_env)."""
    os.environ["RECALL_DATA_ROOT"] = str(Path(tmp) / "data")
    os.environ.pop("RECALL_GLOBAL_DIR", None)
    from recall import config
    cwd = Path(tmp) / "proj"
    kd = config.project_corpus_dir(cwd)
    kd.mkdir(parents=True, exist_ok=True)
    slug_scope = []
    for slug, desc, valid_to in notes:
        extra = f"valid_to: {valid_to}\n" if valid_to else ""
        (kd / f"{slug}.md").write_text(
            f"---\nname: {slug}\ndescription: {desc}\n{extra}---\nBody of {slug}.\n")
        slug_scope.append((config.project_slug(cwd), slug))
    from recall import activation
    for scope, slug in slug_scope:
        activation.log_surfaced(
            [type("H", (), {"corpus": scope, "slug": slug})()])
    return cwd


def test_notes_from_activation_validity_filtered():
    if not _ENGINE_HAS_BRICK3:
        print("↷ SKIP active-notes/validity — recall engine lacks Brick-3")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _setup_corpus(tmp, [
            ("live-note", "a currently-true fact", ""),
            ("dead-note", "a reversed fact", "2020-01-01")])   # historical
        with tempfile.TemporaryDirectory() as bd:
            b = _buf(bd, [("user", "hi")])
            block = ws.build_working_memory(b, cwd, now=date(2026, 7, 4))
        assert "## Active notes" in block
        assert "live-note" in block and "a currently-true fact" in block
        assert "dead-note" not in block          # valid_to in the past → dropped
    print("✓ active notes pulled from activation; historical (valid_to<today) dropped")


def test_notes_budget_keeps_top_activated():
    if not _ENGINE_HAS_BRICK3:
        print("↷ SKIP note-budget — recall engine lacks Brick-3")
        return
    with tempfile.TemporaryDirectory() as tmp:
        cwd = _setup_corpus(tmp, [
            ("note-a", "first surfaced fact", ""),
            ("note-b", "second surfaced fact", ""),
            ("note-c", "third surfaced fact", "")])
        with tempfile.TemporaryDirectory() as bd:
            b = _buf(bd, [("user", "hi")])
            # budget with room for header + turn + exactly ONE note
            block = ws.build_working_memory(b, cwd, budget=250, now=date(2026, 7, 4))
        assert len(block) <= 250
        assert "note-c" in block                 # last activated = highest priority
        assert "note-a" not in block             # weakest dropped first
    print("✓ note budget keeps the top-activated note, drops the weakest")


def test_fail_open_on_broken_corpus():
    # activation points at a slug whose file is malformed → that note is skipped,
    # the block still builds from the turns.
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["RECALL_DATA_ROOT"] = str(Path(tmp) / "data")
        from recall import config, activation
        cwd = Path(tmp) / "proj"
        kd = config.project_corpus_dir(cwd)
        kd.mkdir(parents=True, exist_ok=True)
        (kd / "broken.md").write_text("not valid frontmatter at all")
        activation.log_surfaced(
            [type("H", (), {"corpus": config.project_slug(cwd), "slug": "broken"})()])
        with tempfile.TemporaryDirectory() as bd:
            b = _buf(bd, [("user", "still works")])
            block = ws.build_working_memory(b, cwd, now=date(2026, 7, 4))
        assert "still works" in block and "broken" not in block
    print("✓ malformed note skipped; block still builds (fail-open per-note)")


def main() -> int:
    test_empty_when_no_buffer()
    test_recent_turns_rendered_newest_survives()
    test_turns_window_and_per_turn_cap()
    test_budget_truncation_drops_oldest_turns_first()
    test_notes_from_activation_validity_filtered()
    test_notes_budget_keeps_top_activated()
    test_fail_open_on_broken_corpus()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
