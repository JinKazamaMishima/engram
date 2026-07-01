"""Tests for recall.consolidate — the deterministic activation fold. Index sync
and git are injected, so these run with no model and no repo. Decay/reinforcement
math lives in test_dynamics; here we test the orchestration: events -> frontmatter
stability, log lifecycle, citation gain, and the surgical-writer guarantee that the
curator's quoted description survives untouched."""
from __future__ import annotations

from datetime import date

from recall import activation, config, consolidate
from recall import dynamics as D
from recall.index import Hit
from recall.schema import KnowledgeNote

TARGET = date(2026, 6, 24)
_QUOTED_DESC = 'description: "ORCL 2026-06-08: the −7.76% print was actually +2%"'


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)


def _write_note(corpus_dir, slug, *, last_updated="2026-05-25", **fm):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {slug}", _QUOTED_DESC, f"last_updated: {last_updated}"]
    lines += [f"{k}: {v}" for k, v in fm.items()]
    lines += ["---", "The durable insight, with the why.", ""]
    (corpus_dir / f"{slug}.md").write_text("\n".join(lines))
    return corpus_dir / f"{slug}.md"


def _run_global(monkeypatch, **kw):
    synced = {}

    def sync(db_path, rows):
        synced["rows"] = rows
        return len(rows)

    out = consolidate.run(["--scope", "global"],
                          sync_index=sync, autocommit=lambda ctx: None,
                          today_et=TARGET, **kw)
    return out, synced


def _note(path) -> KnowledgeNote:
    return KnowledgeNote.parse(path.read_text(), expect_slug=path.stem)


def test_consolidate_reinforces_and_persists(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    note_path = _write_note(config.global_corpus_dir(), "owner-values-x")
    activation.log_surfaced([Hit("owner-values-x", "d", "s", 0.5, "global")])

    out, synced = _run_global(monkeypatch)
    assert out.kind == "consolidated", out

    n = _note(note_path)
    assert n.stability > D.S_DEFAULT          # bootstrapped from default, then grew
    assert n.last_used == TARGET.isoformat()
    assert n.uses == 1
    # the curator's hand-quoted description is preserved byte-for-byte
    assert _QUOTED_DESC in note_path.read_text()
    # the index sync got the same row
    assert synced["rows"] == [("owner-values-x", round(n.stability, 3),
                               TARGET.isoformat(), 1)]
    # the activation log was consumed (idempotent: nothing left to claim)
    events, claimed = activation.claim_events("global")
    assert events == [] and claimed is None


def test_consolidate_no_activations_skips(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _write_note(config.global_corpus_dir(), "lonely-note")
    out, _ = _run_global(monkeypatch)
    assert out.kind == "skipped" and out.reason == "no_activations"


def test_consolidate_citation_beats_surfacing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    gdir = config.global_corpus_dir()
    a = _write_note(gdir, "note-a")
    b = _write_note(gdir, "note-b")
    activation.log_surfaced([Hit("note-a", "d", "s", 0.5, "global"),
                             Hit("note-b", "d", "s", 0.5, "global")])
    # inject citation detection: only note-a was actually used in conversation
    out, _ = _run_global(monkeypatch,
                         detect_cited=lambda ctx, cands: {"note-a"} & cands)
    assert out.kind == "consolidated"
    assert _note(a).stability > _note(b).stability   # cited reinforced harder


def test_consolidate_skips_surfaced_note_not_in_corpus(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _write_note(config.global_corpus_dir(), "still-here")
    # a slug that was surfaced but has since been removed from the corpus
    activation.log_surfaced([Hit("still-here", "d", "s", 0.5, "global"),
                             Hit("deleted-note", "d", "s", 0.5, "global")])
    out, synced = _run_global(monkeypatch)
    assert out.kind == "consolidated"
    assert [r[0] for r in synced["rows"]] == ["still-here"]   # ghost slug ignored


def test_consolidate_bad_date(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = consolidate.run(["--scope", "global", "--date", "nonsense"],
                          today_et=TARGET)
    assert out.kind == "failed" and out.reason == "bad_date"
