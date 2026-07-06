"""Unit tests for recall.transcripts — transcript discovery and denoising. The
denoiser decides what counts as 'gold', so it gets the most coverage:
harness-noise stripping, tz date boundaries, and dropping headless skill-run
sessions (no human turn)."""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from recall import transcripts as T

ET = ZoneInfo("America/New_York")
TARGET = date(2026, 6, 1)


# ---- helpers -------------------------------------------------------------

def _user(text, ts="2026-06-01T16:00:00.000Z"):
    return {"type": "user", "message": {"role": "user", "content": text},
            "timestamp": ts}


def _assistant(blocks, ts="2026-06-01T16:01:00.000Z"):
    return {"type": "assistant",
            "message": {"role": "assistant", "content": blocks},
            "timestamp": ts}


def _text(s):
    return {"type": "text", "text": s}


def _jsonl(tmp_path, name, events):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def _set_mtime(path, d: date):
    ts = datetime(d.year, d.month, d.day, 12, 0, tzinfo=ET).timestamp()
    os.utime(path, (ts, ts))


# ---- project_transcript_dir ----------------------------------------------

def test_project_dir_encoding(tmp_path):
    out = T.project_transcript_dir("/home/user/repos/myproject", base=tmp_path)
    assert out == tmp_path / "-home-user-repos-myproject"


# ---- denoising -----------------------------------------------------------

def test_strip_system_reminder_span():
    raw = "real question\n<system-reminder>injected\nmultiline</system-reminder>\nmore"
    assert T._strip_noise(raw) == "real question\n\nmore"


def test_strip_honcho_line():
    raw = "[Honcho Memory for Ada]: Relevant conclusions: blah\nactual text"
    assert T._strip_noise(raw) == "actual text"


def test_strip_command_and_caveat_spans():
    raw = ("<local-command-caveat>Caveat: ...</local-command-caveat>"
           "<command-name>/clear</command-name>"
           "<command-message>clear</command-message>")
    assert T._strip_noise(raw) == ""


def test_is_noise_only_bare_slash_command():
    assert T._is_noise_only("/deviation-trader")
    assert T._is_noise_only("   ")
    assert not T._is_noise_only("/deviation-trader is wrong because the rope...")
    assert not T._is_noise_only("what is the index reconstitution effect?")


def test_content_to_text_variants():
    assert T._content_to_text("plain string") == "plain string"
    blocks = [_text("hello"), {"type": "tool_use", "name": "Bash"},
              {"type": "thinking", "thinking": "secret"}, _text("world")]
    assert T._content_to_text(blocks) == "hello\nworld"
    assert T._content_to_text([{"type": "tool_result", "content": "x"}]) == ""


# ---- iter_exchanges ------------------------------------------------------

def test_iter_keeps_real_exchange_drops_noise(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [
        _user("why does index reconstitution move prices?"),
        _assistant([{"type": "thinking", "thinking": "ponder"}]),  # no text
        _assistant([_text("Passive funds must rebalance at the close.")]),
        _user("/clear"),  # bare command -> noise
    ])
    got = list(T.iter_exchanges(p, TARGET, ET))
    assert [(e.role, e.text) for e in got] == [
        ("user", "why does index reconstitution move prices?"),
        ("assistant", "Passive funds must rebalance at the close."),
    ]


def test_iter_filters_by_et_date(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [
        _user("on 06-01 ET", ts="2026-06-01T16:00:00Z"),       # 12:00 EDT 06-01
        _user("late 05-31 ET", ts="2026-06-01T02:00:00Z"),     # 22:00 EDT 05-31
        _user("early 06-01 ET", ts="2026-06-02T03:00:00Z"),    # 23:00 EDT 06-01
    ])
    texts = [e.text for e in T.iter_exchanges(p, TARGET, ET)]
    assert texts == ["on 06-01 ET", "early 06-01 ET"]


def test_iter_skips_malformed_lines(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("{not json\n" + json.dumps(_user("real")) + "\n\n")
    assert [e.text for e in T.iter_exchanges(p, TARGET, ET)] == ["real"]


# ---- discover_transcripts ------------------------------------------------

def test_discover_mtime_prefilter(tmp_path):
    old = _jsonl(tmp_path, "old.jsonl", [_user("x")])
    fresh = _jsonl(tmp_path, "fresh.jsonl", [_user("y")])
    _set_mtime(old, date(2026, 5, 20))
    _set_mtime(fresh, date(2026, 6, 1))
    found = T.discover_transcripts(tmp_path, TARGET, ET)
    assert old not in found and fresh in found


def test_discover_missing_dir(tmp_path):
    assert T.discover_transcripts(tmp_path / "nope", TARGET, ET) == []


# ---- build_bundle --------------------------------------------------------

def test_bundle_drops_headless_session(tmp_path):
    """A session with assistant turns but no surviving human turn is a headless
    skill run, not a conversation — drop it."""
    human = _jsonl(tmp_path, "human.jsonl",
                   [_user("real question"), _assistant([_text("real answer")])])
    headless = _jsonl(tmp_path, "headless.jsonl",
                      [_user("/shorts-expert"),  # bare command -> noise
                       _assistant([_text("report written to disk")])])
    text, stats = T.build_bundle([human, headless], TARGET, ET)
    assert stats.sessions == 1
    assert stats.exchanges == 2
    assert "real question" in text and "real answer" in text
    assert "report written to disk" not in text


def test_bundle_truncates_long_message(tmp_path):
    long = "A" * 100
    p = _jsonl(tmp_path, "s.jsonl",
               [_user("q"), _assistant([_text(long)])])
    text, _ = T.build_bundle([p], TARGET, ET, max_chars_per_msg=20)
    assert "…[truncated]" in text
    assert "A" * 100 not in text


def test_bundle_empty_when_nothing(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [_user("/clear")])
    text, stats = T.build_bundle([p], TARGET, ET)
    assert text == ""
    assert stats.exchanges == 0 and stats.sessions == 0


# ---- session-scoped helpers (brick 1) ------------------------------------

def test_iter_all_dates_when_target_none(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [
        _user("day one", ts="2026-06-01T16:00:00Z"),
        _user("day two", ts="2026-06-02T16:00:00Z"),
    ])
    assert [e.text for e in T.iter_exchanges(p, None, ET)] == ["day one", "day two"]


def test_build_bundle_all_dates_when_target_none(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [
        _user("q1", ts="2026-06-01T16:00:00Z"),
        _assistant([_text("a1")], ts="2026-06-01T16:01:00Z"),
        _user("q2", ts="2026-06-02T16:00:00Z"),
    ])
    text, stats = T.build_bundle([p], None, ET)
    assert stats.sessions == 1 and stats.exchanges == 3
    assert "q1" in text and "q2" in text


def test_session_transcript_path():
    assert (T.session_transcript_path("/x/dir", "abc-123")
            == Path("/x/dir/abc-123.jsonl"))


def test_session_date_is_last_activity(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [
        _user("first", ts="2026-06-01T16:00:00Z"),
        _assistant([_text("reply")], ts="2026-06-02T18:00:00Z"),  # 14:00 EDT 06-02
    ])
    assert T.session_date(p, ET) == date(2026, 6, 2)


def test_session_date_none_when_no_human_turn(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [_user("/clear")])   # noise only
    assert T.session_date(p, ET) is None


# ---- drop recall's own headless skill runs (brick 2.5) -------------------

def _skill_run(tmp_path, name):
    """A `claude -p /<name>` headless transcript: command wrapper first, then the
    injected skill prose as a user turn (which would otherwise survive denoising)."""
    return _jsonl(tmp_path, f"{name}.jsonl", [
        {"type": "user",
         "message": {"role": "user",
                     "content": f"<command-message>{name}</command-message>\n"
                                f"<command-name>/{name}</command-name>"},
         "timestamp": "2026-06-01T16:00:00Z"},
        {"type": "user",   # skill expansion injected as a user turn — real text
         "message": {"role": "user",
                     "content": f"Base directory for this skill: /x/skills/{name}\n\n"
                                "SKILL_PROSE_MARKER — you are the memory curator."},
         "timestamp": "2026-06-01T16:00:05Z"},
        _assistant([_text("Done. Curation complete.")], ts="2026-06-01T16:02:00Z"),
    ])


def test_iter_drops_headless_recall_skill_runs(tmp_path):
    for name in ("curate-memory", "dream", "reconsolidate-memory"):
        p = _skill_run(tmp_path, name)
        assert list(T.iter_exchanges(p, None, ET)) == [], name   # whole session dropped
        assert T.build_bundle([p], None, ET)[1].exchanges == 0, name
        assert T.session_date(p, ET) is None, name


def test_bundle_drops_skill_run_keeps_real(tmp_path):
    skill = _skill_run(tmp_path, "curate-memory")
    real = _jsonl(tmp_path, "real.jsonl",
                  [_user("genuine question"), _assistant([_text("genuine answer")])])
    text, stats = T.build_bundle([skill, real], None, ET)
    assert stats.sessions == 1 and "genuine question" in text
    assert "SKILL_PROSE_MARKER" not in text          # the skill's own prose is never mined


def test_keeps_human_session_opening_with_other_slash_command(tmp_path):
    # a human session that OPENS with some non-memory slash command must NOT be dropped
    p = _jsonl(tmp_path, "s.jsonl", [
        {"type": "user",
         "message": {"role": "user",
                     "content": "<command-message>code-review</command-message>\n"
                                "<command-name>/code-review</command-name>"},
         "timestamp": "2026-06-01T16:00:00Z"},
        _user("actually let's discuss the auth bug", ts="2026-06-01T16:01:00Z"),
        _assistant([_text("Sure — here's the issue.")], ts="2026-06-01T16:02:00Z"),
    ])
    texts = [e.text for e in T.iter_exchanges(p, None, ET)]
    assert "actually let's discuss the auth bug" in texts   # kept, not over-dropped


# ---- since/until slicing (incremental curation) ----------------------------

def _utc(h, m=0):
    from datetime import timezone
    return datetime(2026, 6, 1, h, m, tzinfo=timezone.utc)


def test_iter_exchanges_since_until_window(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [
        _user("first", ts="2026-06-01T10:00:00Z"),
        _assistant([_text("one")], ts="2026-06-01T11:00:00Z"),
        _user("second", ts="2026-06-01T12:00:00Z"),
        _assistant([_text("two")], ts="2026-06-01T13:00:00Z"),
    ])
    # since is EXCLUSIVE (the watermarked exchange was already curated) …
    texts = [e.text for e in T.iter_exchanges(p, None, ET, since=_utc(11))]
    assert texts == ["second", "two"]
    # … until is INCLUSIVE (it IS the cooled edge the caller wants curated).
    texts = [e.text for e in T.iter_exchanges(p, None, ET,
                                              since=_utc(10), until=_utc(12))]
    assert texts == ["one", "second"]
    # empty window → nothing.
    assert list(T.iter_exchanges(p, None, ET, since=_utc(13))) == []


def test_build_bundle_threads_since(tmp_path):
    p = _jsonl(tmp_path, "s.jsonl", [
        _user("old", ts="2026-06-01T10:00:00Z"),
        _user("new", ts="2026-06-01T12:00:00Z"),
        _assistant([_text("reply")], ts="2026-06-01T12:01:00Z"),
    ])
    text, stats = T.build_bundle([p], None, ET, since=_utc(11))
    assert "new" in text and "old" not in text
    assert stats.exchanges == 2


# ---- Engram LiveBuffer reader -------------------------------------------------

def _buffer_row(seq, role, text, ts):
    return json.dumps({"convo_id": "conv-1", "seq": seq, "ts": ts,
                       "role": role, "text": text})


def _buffer_file(tmp_path, rows, name="conv-1.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(rows) + "\n")
    return p


def test_iter_buffer_exchanges_roundtrip_and_order(tmp_path):
    p = _buffer_file(tmp_path, [
        _buffer_row(2, "assistant", "the reply", "2026-06-01T16:01:00+00:00"),
        _buffer_row(1, "user", "the question", "2026-06-01T16:00:00+00:00"),
    ])
    exs = list(T.iter_buffer_exchanges(p))
    assert [(e.role, e.text) for e in exs] == [
        ("user", "the question"), ("assistant", "the reply")]   # (seq, ts) order
    assert all(e.session_id == "conv-1" for e in exs)


def test_iter_buffer_exchanges_tolerates_garbage(tmp_path):
    p = _buffer_file(tmp_path, [
        _buffer_row(1, "user", "good row", "2026-06-01T16:00:00+00:00"),
        '{"convo_id": "conv-1", "seq": 2, "ts": "2026-06-01T16:0',  # torn write
        "not json at all",
        _buffer_row(3, "assistant", "still fine", "2026-06-01T16:02:00+00:00"),
        _buffer_row(4, "assistant", "", "2026-06-01T16:03:00+00:00"),  # empty
        _buffer_row(5, "user", "/lone-slash", "2026-06-01T16:04:00+00:00"),
    ])
    exs = list(T.iter_buffer_exchanges(p))
    assert [e.text for e in exs] == ["good row", "still fine"]


def test_iter_buffer_exchanges_window_and_noise(tmp_path):
    p = _buffer_file(tmp_path, [
        _buffer_row(1, "user", "old", "2026-06-01T10:00:00+00:00"),
        _buffer_row(2, "user",
                    "<system-reminder>x</system-reminder>real ask",
                    "2026-06-01T12:00:00+00:00"),
    ])
    exs = list(T.iter_buffer_exchanges(p, since=_utc(11)))
    assert [e.text for e in exs] == ["real ask"]    # sliced + denoised


def test_build_buffer_bundle_matches_canonical_render(tmp_path):
    # The buffer path must produce byte-identical markdown to the transcript
    # path for the same conversation — one renderer, one curator contract.
    buf = _buffer_file(tmp_path, [
        _buffer_row(1, "user", "why X?", "2026-06-01T16:00:00+00:00"),
        _buffer_row(2, "assistant", "because Y.", "2026-06-01T16:01:00+00:00"),
    ])
    tr = _jsonl(tmp_path, "conv-1t.jsonl", [
        _user("why X?", ts="2026-06-01T16:00:00Z"),
        _assistant([_text("because Y.")], ts="2026-06-01T16:01:00Z"),
    ])
    btext, bstats = T.build_buffer_bundle(buf)
    ttext, tstats = T.build_bundle([tr], None, ET)
    assert btext.replace("conv-1t", "conv-1") == ttext.replace("conv-1t", "conv-1")
    assert (bstats.sessions, bstats.exchanges) == (tstats.sessions, tstats.exchanges)


def test_build_buffer_bundle_empty_and_assistant_only(tmp_path):
    empty = _buffer_file(tmp_path, [], name="empty.jsonl")
    text, stats = T.build_buffer_bundle(empty)
    assert text == "" and stats.exchanges == 0
    solo = _buffer_file(tmp_path, [
        _buffer_row(1, "assistant", "greeting into the void",
                    "2026-06-01T16:00:00+00:00")], name="solo.jsonl")
    text, stats = T.build_buffer_bundle(solo)
    assert stats.exchanges == 0     # no human turn -> not gold (same rule as transcripts)


def test_buffer_last_ts(tmp_path):
    p = _buffer_file(tmp_path, [
        _buffer_row(1, "user", "a", "2026-06-01T10:00:00+00:00"),
        _buffer_row(2, "assistant", "b", "2026-06-01T11:00:00+00:00"),
    ])
    assert T.buffer_last_ts(p) == _utc(11)
    assert T.buffer_last_ts(p, until=_utc(10, 30)) == _utc(10)
    assert T.buffer_last_ts(tmp_path / "nope.jsonl") is None


def test_iter_buffer_exchanges_perception_role(tmp_path):
    # Perception rows (the perceiving loop's step-5 buffer) yield like
    # assistant prose; unknown roles stay dropped; extra provenance keys in
    # the row are tolerated; the watermark advances over perception rows.
    rows = [
        _buffer_row(1, "perception", "[engage] Ada is here — engaging",
                    "2026-06-01T10:00:00+00:00"),
        _buffer_row(2, "tool", "never a memory", "2026-06-01T10:30:00+00:00"),
        json.dumps({"convo_id": "conv-1", "seq": 3,
                    "ts": "2026-06-01T11:00:00+00:00", "role": "perception",
                    "text": "[eye] a desk with a laptop  [✓ desk, laptop]",
                    "kind": "eye", "data": {"stable": True}}),
    ]
    p = _buffer_file(tmp_path, rows, name="percept-2026-06-01.jsonl")
    exs = list(T.iter_buffer_exchanges(p))
    assert [(e.role, e.text[:6]) for e in exs] == [
        ("perception", "[engag"), ("perception", "[eye] ")]
    assert T.buffer_last_ts(p) == _utc(11)


def test_build_buffer_bundle_perception_only_is_gold(tmp_path):
    # A perception-only buffer is genuine signal (gate-verified real-world
    # events), NOT a headless run — it must render, as ### PERCEPTION blocks.
    p = _buffer_file(tmp_path, [
        _buffer_row(1, "perception", "[engage] Ada is here — engaging",
                    "2026-06-01T10:00:00+00:00"),
        _buffer_row(2, "perception", "[idle] frame is clear — resting",
                    "2026-06-01T18:00:00+00:00"),
    ], name="percept-2026-06-01.jsonl")
    text, stats = T.build_buffer_bundle(p)
    assert stats.exchanges == 2
    assert "### PERCEPTION" in text and "[engage] Ada" in text
