"""Unit tests for recall.schema — the curator's output contracts. Every
rejection path is exercised: a malformed manifest or note must fail the fire,
never be silently accepted into the corpus."""
from __future__ import annotations

import json

import pytest

from recall.schema import (
    CurationManifest,
    CurationSchemaError,
    KnowledgeNote,
    NoteEdit,
    set_frontmatter_keys,
    sha256_str,
)

# ---- CurationManifest ----------------------------------------------------

def _manifest(**over) -> dict:
    d = {
        "schema_version": 1,
        "date": "2026-06-01",
        "summary": "captured the index-reconstitution flow mechanic",
        "notes": [{"slug": "index-reconstitution-flows",
                   "action": "created", "title": "rebalance flows"}],
    }
    d.update(over)
    return d


def test_manifest_roundtrip():
    m = CurationManifest.from_json(json.dumps(_manifest()))
    assert m.date == "2026-06-01"
    assert len(m.notes) == 1
    assert m.notes[0].slug == "index-reconstitution-flows"
    assert m.notes[0].action == "created"
    assert CurationManifest.from_json(m.to_json()).to_dict() == m.to_dict()


def test_manifest_empty_notes_is_valid_no_insight_day():
    m = CurationManifest.from_json(json.dumps(
        _manifest(notes=[], summary="ops/debugging only; nothing durable")))
    assert m.notes == []
    assert m.summary.startswith("ops/debugging")


@pytest.mark.parametrize("over, msg", [
    ({"date": ""}, "date is required"),
    ({"summary": ""}, "summary is required"),
    ({"schema_version": 2}, "unsupported schema_version"),
    ({"schema_version": "x"}, "bad schema_version"),
    ({"notes": "notalist"}, "notes must be a list"),
])
def test_manifest_rejections(over, msg):
    with pytest.raises(CurationSchemaError, match=msg):
        CurationManifest.from_json(json.dumps(_manifest(**over)))


def test_manifest_invalid_json():
    with pytest.raises(CurationSchemaError, match="not valid JSON"):
        CurationManifest.from_json("{nope")


def test_manifest_duplicate_slug():
    dup = _manifest(notes=[
        {"slug": "a-b", "action": "created", "title": "x"},
        {"slug": "a-b", "action": "updated", "title": "y"},
    ])
    with pytest.raises(CurationSchemaError, match="duplicate note slug"):
        CurationManifest.from_json(json.dumps(dup))


@pytest.mark.parametrize("note, msg", [
    ({"slug": "Bad Slug", "action": "created", "title": "t"}, "kebab-case"),
    ({"slug": "ok", "action": "deleted", "title": "t"}, "action"),
    ({"slug": "ok", "action": "created", "title": ""}, "title is empty"),
    ({"slug": "ok", "action": "created"}, "title is empty"),
    ({"action": "created", "title": "t"}, "missing"),
])
def test_note_edit_rejections(note, msg):
    with pytest.raises(CurationSchemaError, match=msg):
        NoteEdit.from_dict(note)


def test_note_edit_scope_default_and_global():
    assert NoteEdit.from_dict(
        {"slug": "a", "action": "created", "title": "t"}).scope == "project"
    g = NoteEdit.from_dict({"slug": "owner-identity", "action": "created",
                            "title": "who owner is", "scope": "Global"})
    assert g.scope == "global"  # case-normalized


def test_note_edit_bad_scope_rejected():
    with pytest.raises(CurationSchemaError, match="scope"):
        NoteEdit.from_dict({"slug": "a", "action": "created", "title": "t",
                            "scope": "personal"})


# ---- KnowledgeNote -------------------------------------------------------

GOOD_NOTE = """\
---
name: index-reconstitution-flows
description: index reconstitution forces passive rebalancing at the close
tags: [index, sp500, rebalance]
sources: [2026-06-01]
---
Passive funds tracking the index must trade to match membership changes at the
close on the effective date. We tilt around announced reconstitutions.
"""


def test_knowledge_note_parse_ok():
    n = KnowledgeNote.parse(GOOD_NOTE, expect_slug="index-reconstitution-flows")
    assert n.slug == "index-reconstitution-flows"
    assert n.description.startswith("index reconstitution")
    assert n.tags == ("index", "sp500", "rebalance")
    assert n.sources == ("2026-06-01",)
    assert "Passive funds" in n.body
    assert n.kind == ""  # ordinary domain note: no kind


def test_knowledge_note_scalar_tags():
    note = GOOD_NOTE.replace("tags: [index, sp500, rebalance]", "tags: index")
    assert KnowledgeNote.parse(note).tags == ("index",)


def test_knowledge_note_optional_kind():
    """The global/soul corpus tags notes with a kind; parsing keeps it
    (lowercased). Absent -> ''."""
    note = GOOD_NOTE.replace("sources: [2026-06-01]",
                             "sources: [2026-06-01]\nkind: Identity")
    assert KnowledgeNote.parse(note).kind == "identity"


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: s.replace("---\n", "", 1), "missing YAML frontmatter"),
    (lambda s: s.replace("name: index-reconstitution-flows",
                         "name: Bad Name"), "not kebab-case"),
    (lambda s: s.replace("description: index reconstitution forces passive "
                         "rebalancing at the close", "description:"),
     "description' is required"),
    (lambda s: s.split("---\n", 2)[0] + "---\n" + s.split("---\n", 2)[1]
     + "---\n", "body is empty"),
])
def test_knowledge_note_rejections(mutate, msg):
    with pytest.raises(CurationSchemaError, match=msg):
        KnowledgeNote.parse(mutate(GOOD_NOTE))


def test_knowledge_note_slug_mismatch():
    with pytest.raises(CurationSchemaError, match="!= filename slug"):
        KnowledgeNote.parse(GOOD_NOTE, expect_slug="some-other-slug")


def test_sha256_str_stable():
    assert sha256_str("abc") == sha256_str("abc")
    assert sha256_str("abc") != sha256_str("abd")
    assert len(sha256_str("abc")) == 64


def test_knowledge_note_dates_and_supersede_parsed():
    note = GOOD_NOTE.replace(
        "sources: [2026-06-01]",
        "sources: [2026-06-01]\nfirst_seen: 2026-05-20\n"
        "last_updated: 2026-06-01\nsuperseded: true\nsuperseded_by: newer-note")
    n = KnowledgeNote.parse(note)
    assert n.first_seen == "2026-05-20" and n.last_updated == "2026-06-01"
    assert n.superseded is True and n.superseded_by == "newer-note"


def test_knowledge_note_supersede_defaults_backward_compatible():
    n = KnowledgeNote.parse(GOOD_NOTE)  # GOOD_NOTE has none of the new keys
    assert n.superseded is False and n.superseded_by == ""
    assert n.first_seen == "" and n.last_updated == ""


def test_knowledge_note_quoted_false_is_not_true():
    note = GOOD_NOTE.replace("sources: [2026-06-01]",
                             'sources: [2026-06-01]\nsuperseded: "false"')
    assert KnowledgeNote.parse(note).superseded is False


def test_knowledge_note_bad_superseded_by_rejected():
    note = GOOD_NOTE.replace("sources: [2026-06-01]",
                             "sources: [2026-06-01]\nsuperseded_by: Not Kebab")
    with pytest.raises(CurationSchemaError, match="kebab-case"):
        KnowledgeNote.parse(note)


# ---- lenient frontmatter fallback (unquoted-colon prose) -----------------

def test_knowledge_note_description_with_internal_colon_parses():
    """The reported nightly failure: an unquoted description with an internal
    colon-space, which PyYAML reads as a nested map and rejects. The lenient
    fallback takes the value verbatim; flow-list tags/sources still parse."""
    note = GOOD_NOTE.replace(
        "description: index reconstitution forces passive rebalancing at the close",
        "description: ORCL 2026-06-08 example: the '-7.76%' print was actually +2% up")
    n = KnowledgeNote.parse(note, expect_slug="index-reconstitution-flows")
    assert n.description == (
        "ORCL 2026-06-08 example: the '-7.76%' print was actually +2% up")
    assert n.tags == ("index", "sp500", "rebalance")
    assert n.sources == ("2026-06-01",)
    assert "Passive funds" in n.body


def test_knowledge_note_quoted_description_with_colon_roundtrips():
    """A properly double-quoted description with a colon parses via the strict
    YAML path (no fallback), with the quotes stripped."""
    note = GOOD_NOTE.replace(
        "description: index reconstitution forces passive rebalancing at the close",
        'description: "ORCL 2026-06-08: actually +2% up, not -7.76%"')
    assert KnowledgeNote.parse(note).description == (
        "ORCL 2026-06-08: actually +2% up, not -7.76%")


def test_loose_fallback_preserves_semantic_rejections():
    """Even when the fallback kicks in (colon in description), a bad slug still
    fails loud — the fallback forgives serialization, never semantics."""
    note = GOOD_NOTE.replace(
        "name: index-reconstitution-flows", "name: Bad Name").replace(
        "description: index reconstitution forces passive rebalancing at the close",
        "description: has an internal colon: here")
    with pytest.raises(CurationSchemaError, match="kebab-case"):
        KnowledgeNote.parse(note)


def test_loose_fallback_missing_description_still_raises():
    """A fallback dict missing a required field still fails loud."""
    note = ("---\nname: foo-bar\nextra: a: b: c\n---\nbody here\n")
    with pytest.raises(CurationSchemaError, match="description' is required"):
        KnowledgeNote.parse(note)


def test_loose_fallback_non_keyvalue_line_raises():
    """A genuinely broken frontmatter line (no 'key: value') still raises on the
    fallback path."""
    note = ("---\nname: foo-bar\ndescription: has a colon: here\n"
            "a bare prose line with no key\n---\nbody here\n")
    with pytest.raises(CurationSchemaError, match="not 'key: value'"):
        KnowledgeNote.parse(note)


# ---- dynamic-memory fields (stability / last_used / uses / surprise) ------

def test_knowledge_note_dynamic_fields_parsed():
    note = GOOD_NOTE.replace(
        "sources: [2026-06-01]",
        "sources: [2026-06-01]\nstability: 12.5\nlast_used: 2026-06-20\n"
        "uses: 7\nsurprise: 0.8\nimportance: 1.0")
    n = KnowledgeNote.parse(note)
    assert n.stability == 12.5 and n.last_used == "2026-06-20"
    assert n.uses == 7 and n.surprise == 0.8 and n.importance == 1.0


def test_knowledge_note_dynamic_fields_default_backward_compatible():
    n = KnowledgeNote.parse(GOOD_NOTE)  # none of the dynamic keys present
    assert n.stability == 0.0 and n.last_used == "" and n.uses == 0
    assert n.surprise == -1.0 and n.importance == 0.0   # -1 == "unset" sentinel


def test_knowledge_note_garbled_dynamic_scalar_falls_back():
    note = GOOD_NOTE.replace("sources: [2026-06-01]",
                             "sources: [2026-06-01]\nstability: not-a-number")
    assert KnowledgeNote.parse(note).stability == 0.0   # garbled -> default


# ---- set_frontmatter_keys (surgical scalar writer) -----------------------

def test_set_frontmatter_keys_updates_in_place_and_inserts():
    updated = set_frontmatter_keys(GOOD_NOTE, {"stability": 5.0, "uses": 3})
    n = KnowledgeNote.parse(updated)
    assert n.stability == 5.0 and n.uses == 3
    # body and other fields untouched
    assert "Passive funds tracking the index" in updated
    assert n.tags == ("index", "sp500", "rebalance")


def test_set_frontmatter_keys_preserves_quoted_description_byte_for_byte():
    note = GOOD_NOTE.replace(
        "description: index reconstitution forces passive rebalancing at the close",
        'description: "ORCL: actually +2%, not −7.76% — recompute"')
    out = set_frontmatter_keys(note, {"stability": 9.0, "last_used": "2026-06-24"})
    assert 'description: "ORCL: actually +2%, not −7.76% — recompute"' in out
    n = KnowledgeNote.parse(out)
    assert n.stability == 9.0 and n.last_used == "2026-06-24"
    assert n.description == "ORCL: actually +2%, not −7.76% — recompute"


def test_set_frontmatter_keys_replaces_existing_key_once():
    note = GOOD_NOTE.replace("sources: [2026-06-01]",
                             "sources: [2026-06-01]\nstability: 2.0")
    out = set_frontmatter_keys(note, {"stability": 8.5})
    assert out.count("stability:") == 1
    assert KnowledgeNote.parse(out).stability == 8.5


def test_set_frontmatter_keys_rejects_missing_frontmatter():
    with pytest.raises(CurationSchemaError, match="frontmatter"):
        set_frontmatter_keys("no frontmatter here\n", {"stability": 1.0})


def test_set_frontmatter_keys_noop_on_empty_updates():
    assert set_frontmatter_keys(GOOD_NOTE, {}) == GOOD_NOTE
