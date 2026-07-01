"""Tests for recall.activation — the hippocampal JSONL trace (append + claim +
rollup). No model, no network; the data root is redirected to tmp."""
from __future__ import annotations

from datetime import datetime, timezone

from recall import activation
from recall.index import Hit


def _hit(slug, corpus):
    return Hit(slug=slug, description="d", snippet="s", score=0.5, corpus=corpus)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)


def test_log_surfaced_groups_by_scope(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    activation.log_surfaced([_hit("a", "proj"), _hit("b", "proj"),
                             _hit("x", "global")])
    assert len(activation.read_events("proj")) == 2
    assert len(activation.read_events("global")) == 1
    assert {e["slug"] for e in activation.read_events("proj")} == {"a", "b"}
    assert all(e["kind"] == "surfaced" for e in activation.read_events("proj"))


def test_log_surfaced_skips_blank_scope_or_slug(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    activation.log_surfaced([_hit("a", ""), _hit("", "proj")])
    assert activation.read_events("proj") == []
    assert activation.read_events("") == []


def test_log_surfaced_is_fail_open_on_bad_input(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # objects missing .corpus/.slug must not raise (the hook can never break)
    activation.log_surfaced([object()])
    assert activation.read_events("proj") == []


def test_claim_events_is_atomic_and_resets_live_log(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    activation.log_surfaced([_hit("a", "proj"), _hit("a", "proj"),
                             _hit("b", "proj")])
    events, claimed = activation.claim_events("proj")
    assert len(events) == 3 and claimed is not None
    # live log is now empty; a fresh surfacing starts a new log
    assert activation.read_events("proj") == []
    activation.log_surfaced([_hit("c", "proj")])
    assert {e["slug"] for e in activation.read_events("proj")} == {"c"}
    # and the claimed-but-not-discarded events are re-claimed (failed-run safety)
    again, _ = activation.claim_events("proj")
    assert {e["slug"] for e in again} == {"a", "b", "c"}


def test_discard_claimed_drops_the_consumed_log(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    activation.log_surfaced([_hit("a", "proj")])
    activation.claim_events("proj")
    activation.discard_claimed("proj")
    events, claimed = activation.claim_events("proj")
    assert events == [] and claimed is None


def test_claim_empty_scope_is_empty(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    events, claimed = activation.claim_events("never-used")
    assert events == [] and claimed is None


def test_rollup_counts_and_grades():
    ts = datetime(2026, 6, 24, tzinfo=timezone.utc).isoformat()
    later = datetime(2026, 6, 24, 1, tzinfo=timezone.utc).isoformat()
    events = [
        {"slug": "a", "kind": "surfaced", "ts": ts},
        {"slug": "a", "kind": "cited", "ts": later},
        {"slug": "b", "kind": "surfaced", "ts": ts},
        {"slug": "", "kind": "surfaced", "ts": ts},      # dropped
    ]
    agg = activation.rollup(events)
    assert set(agg) == {"a", "b"}
    assert agg["a"] == {"count": 2, "surfaced": 1, "cited": 1, "last_ts": later}
    assert agg["b"]["count"] == 1
