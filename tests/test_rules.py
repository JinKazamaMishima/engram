"""kind:rule scanning + the always-on standing-rules block (torch-free, no
index, no daemon — plain tmp corpora)."""
from __future__ import annotations

from datetime import date

from recall import rules

TODAY = date(2026, 7, 9)


def _note(d, slug, desc, *, kind="rule", extra=""):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{slug}.md").write_text(
        f'---\nname: {slug}\ndescription: "{desc}"\nkind: {kind}\n{extra}---\n'
        f"Body with the why.\n")


def test_scan_picks_only_active_rules(tmp_path):
    _note(tmp_path, "rule-a", "do the thing")
    _note(tmp_path, "plain-fact", "a normal fact", kind="lesson")
    _note(tmp_path, "rule-retired", "was true", extra="valid_to: 2026-01-01\n")
    _note(tmp_path, "rule-future", "not yet", extra="valid_from: 2099-01-01\n")
    _note(tmp_path, "rule-superseded", "replaced", extra="superseded: true\n")
    active, broken = rules.scan_rules(tmp_path, today=TODAY)
    assert [n.slug for n in active] == ["rule-a"]
    assert broken == []


def test_scan_body_mention_is_not_a_rule(tmp_path):
    # The cheap prefilter can hit a body line; the parsed kind decides.
    (tmp_path / "doc.md").write_text(
        '---\nname: doc\ndescription: "how to promote a rule"\n---\n'
        "Add this frontmatter line:\nkind: rule\nand get the operator's go.\n")
    assert rules.scan_rules(tmp_path, today=TODAY) == ([], [])


def test_scan_reports_broken_rule_note(tmp_path):
    # A rule note that fails to parse is a SILENT rule outage — surfaced, never
    # swallowed.
    (tmp_path / "rule-broken.md").write_text(
        "---\nname: rule-broken\nkind: rule\n---\nno description field\n")
    active, broken = rules.scan_rules(tmp_path, today=TODAY)
    assert active == [] and broken == ["rule-broken"]


def test_scan_missing_dir_is_empty(tmp_path):
    assert rules.scan_rules(tmp_path / "nope", today=TODAY) == ([], [])


def test_rules_context_sections_order_and_precedence(tmp_path, monkeypatch):
    soul = tmp_path / "soul"
    proj = tmp_path / "proj"
    _note(soul, "rule-identity", "your name is ada")
    _note(proj / "docs" / "knowledge", "rule-ship", "also ship to the mirror")
    monkeypatch.setenv("RECALL_GLOBAL_DIR", str(soul))
    out = rules.rules_context(proj, today=TODAY)
    assert out.startswith("## Standing rules")
    assert "Precedence" in out                      # the doctrine line rides along
    assert "rule-identity" in out and "rule-ship" in out
    assert out.index("Global / soul") < out.index("This project")  # soul first


def test_rules_context_soul_only_in_plain_folder(tmp_path, monkeypatch):
    # A repo with no corpus still gets the soul's rules — identity follows
    # the operator into any folder.
    soul = tmp_path / "soul"
    _note(soul, "rule-identity", "your name is ada")
    monkeypatch.setenv("RECALL_GLOBAL_DIR", str(soul))
    out = rules.rules_context(tmp_path / "random-clone", today=TODAY)
    assert "rule-identity" in out and "This project" not in out


def test_rules_context_none_when_no_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_GLOBAL_DIR", str(tmp_path / "soul"))
    assert rules.rules_context(tmp_path / "proj", today=TODAY) is None


def test_rules_context_budget_drops_whole_rules_and_says_so(tmp_path, monkeypatch):
    soul = tmp_path / "soul"
    for i in range(5):
        _note(soul, f"rule-{i}", "x" * 80)
    monkeypatch.setenv("RECALL_GLOBAL_DIR", str(soul))
    out = rules.rules_context(tmp_path / "proj", today=TODAY,
                              budget=len(rules._HEADER) + 130)
    assert "rule-0" in out                 # deterministic: slug order survives
    assert "omitted" in out                # the thinning is announced, not silent
    assert "rule-4" not in out


def test_rules_context_broken_note_warning_rides_along(tmp_path, monkeypatch):
    soul = tmp_path / "soul"
    soul.mkdir()
    (soul / "rule-broken.md").write_text(
        "---\nname: rule-broken\nkind: rule\n---\nno description\n")
    monkeypatch.setenv("RECALL_GLOBAL_DIR", str(soul))
    out = rules.rules_context(tmp_path / "proj", today=TODAY)
    assert "SILENT RULE OUTAGE" in out and "rule-broken" in out
