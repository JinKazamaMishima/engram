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
