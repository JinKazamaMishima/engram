"""Tests for recall.reap — ebb, the active-forgetting pass. Index rebuild and git
are injected, so these run with no embedder and no repo. The FSRS math lives in
test_dynamics; here we test the orchestration: which notes are selected (cold vs
superseded vs each exemption), the reversible archive-move, --restore, the
coldest-first cap, dry-run, and that archived notes are never re-examined."""
from __future__ import annotations

from datetime import date

from recall import config, reap

TARGET = date(2026, 6, 24)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    # Neutralize any host-level threshold overrides so the defaults are tested.
    for k in ("RECALL_REAP_DORMANT_DAYS", "RECALL_REAP_R_FLOOR",
              "RECALL_REAP_USES_MAX", "RECALL_REAP_IMPORTANCE_FLOOR",
              "RECALL_REAP_BOOTSTRAP_S", "RECALL_REAP_MAX_PER_RUN"):
        monkeypatch.delenv(k, raising=False)


def _write_note(corpus_dir, slug, *, body="the durable insight, with the why.",
                **fm):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {slug}", f'description: "{slug} description"']
    lines += [f"{k}: {v}" for k, v in fm.items()]
    lines += ["---", body, ""]
    p = corpus_dir / f"{slug}.md"
    p.write_text("\n".join(lines))
    return p


def _run(argv, **kw):
    return reap.run(argv, rebuild_index=lambda ctx: 0,
                    autocommit=lambda ctx, s: None, today_et=TARGET, **kw)


def _archive(corpus_dir, slug):
    return config.archive_dir(corpus_dir) / f"{slug}.md"


# ---- cold + superseded selection ------------------------------------------

def test_cold_note_archived(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "old-cold", last_updated="2025-12-01")   # ~205d dormant
    out = _run(["--scope", "global", "--commit"])
    assert out.kind == "reaped", out
    assert not p.exists()                       # moved out of the live corpus
    arch = _archive(g, "old-cold")
    assert arch.exists()
    text = arch.read_text()
    assert "archived_reason: cold" in text and "archived_on: 2026-06-24" in text


def test_superseded_note_archived_even_if_recent(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "replaced", last_updated="2026-06-20",
                    superseded="true", superseded_by="new-note")
    out = _run(["--scope", "global"])
    assert out.kind == "reaped"
    assert not p.exists()
    assert "archived_reason: superseded" in _archive(g, "replaced").read_text()


def test_fresh_note_kept(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "fresh", last_updated="2026-06-20")   # 4d dormant
    out = _run(["--scope", "global"])
    assert out.kind == "skipped" and out.reason == "nothing_cold"
    assert p.exists()


def test_used_note_kept_despite_age(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "popular", last_updated="2025-12-01", uses="5")
    out = _run(["--scope", "global"])          # uses(5) > REAP_USES_MAX(2)
    assert out.reason == "nothing_cold" and p.exists()


# ---- the three never-reap exemptions --------------------------------------

def test_rule_note_exempt(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "rule-x", last_updated="2025-12-01", kind="rule")
    assert _run(["--scope", "global"]).reason == "nothing_cold"
    assert p.exists()


def test_permanent_note_exempt(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "perma", last_updated="2025-12-01", stability="400")
    assert _run(["--scope", "global"]).reason == "nothing_cold"
    assert p.exists()


def test_importance_anchor_exempt(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "identity", last_updated="2025-12-01", importance="1.0")
    assert _run(["--scope", "global"]).reason == "nothing_cold"
    assert p.exists()


def test_surprising_note_protected_by_bootstrap(tmp_path, monkeypatch):
    # A never-activated but SURPRISING note keeps its flashbulb S0 (~15d), so its
    # retrievability stays above the floor — it is not "cold".
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "aha", last_updated="2025-12-01", surprise="1.0")
    assert _run(["--scope", "global"]).reason == "nothing_cold"
    assert p.exists()


# ---- dry-run / cap / restore / idempotency --------------------------------

def test_dry_run_moves_nothing(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "old-cold", last_updated="2025-12-01")
    out = _run(["--scope", "global", "--dry-run"])
    assert out.kind == "skipped" and out.reason == "dry_run"
    assert p.exists()                                   # untouched
    assert not _archive(g, "old-cold").exists()
    assert "would archive" in capsys.readouterr().out


def test_max_per_run_keeps_coldest(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    older = _write_note(g, "older", last_updated="2025-10-01")   # colder
    newer = _write_note(g, "newer", last_updated="2026-01-15")   # cold but warmer
    out = _run(["--scope", "global", "--max-per-run", "1"])
    assert out.kind == "reaped"
    assert not older.exists() and newer.exists()        # cap took the coldest
    assert _archive(g, "older").exists()


def test_restore_round_trip(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    p = _write_note(g, "old-cold", last_updated="2025-12-01")
    _run(["--scope", "global"])
    assert not p.exists() and _archive(g, "old-cold").exists()
    out = _run(["--scope", "global", "--restore", "old-cold"])
    assert out.kind == "restored"
    assert p.exists() and not _archive(g, "old-cold").exists()
    assert "restored_on: 2026-06-24" in p.read_text()


def test_restore_missing_slug_skips(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _write_note(config.global_corpus_dir(), "present", last_updated="2026-06-20")
    out = _run(["--scope", "global", "--restore", "ghost"])
    assert out.kind == "skipped" and out.reason == "not_archived"


def test_archived_notes_not_re_reaped(tmp_path, monkeypatch):
    # The archive/ subdir is under the corpus dir but the loader globs
    # non-recursively, so a second pass sees an empty live corpus.
    _setup(tmp_path, monkeypatch)
    g = config.global_corpus_dir()
    _write_note(g, "old-cold", last_updated="2025-12-01")
    assert _run(["--scope", "global"]).kind == "reaped"
    out = _run(["--scope", "global"])
    assert out.kind == "skipped" and out.reason == "nothing_cold"


# ---- guards ---------------------------------------------------------------

def test_no_corpus_skips(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)                       # global dir never created
    out = _run(["--scope", "global"])
    assert out.kind == "skipped" and out.reason == "no_corpus"


def test_bad_date_fails(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    out = reap.run(["--scope", "global", "--date", "nope"], today_et=TARGET)
    assert out.kind == "failed" and out.reason == "bad_date"
