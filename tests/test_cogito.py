"""Tests for recall.cogito — the self-reference stance instrument (Cogito m1).
Hermetic: the judge server and the notifier are injected; the LiveBuffer is a
tmp fixture; no llama, no network, no Telegram."""
from __future__ import annotations

import json
from datetime import date

from recall import cogito, config

TARGET = date(2026, 6, 24)


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    monkeypatch.delenv("RECALL_COGITO_SELF_NAMES_FILE", raising=False)


def _buffer_row(seq, role, text, *, ts=f"{TARGET.isoformat()}T15:00:00+00:00"):
    return json.dumps({"convo_id": "test-convo", "seq": seq, "ts": ts,
                       "role": role, "text": text})


def _write_buffer(rows, name="test-convo"):
    buf_dir = config.engram_buffer_dir()
    buf_dir.mkdir(parents=True, exist_ok=True)
    (buf_dir / f"{name}.jsonl").write_text("\n".join(rows) + "\n")


class _FakeJudge:
    """Injectable stand-in for SpawnedJudge: canned verdicts, no subprocess."""

    def __init__(self, verdicts=None, up=True):
        self.verdicts = verdicts or {}
        self.up = up
        self.started = 0
        self.stopped = 0

    def start(self, timeout_s=0):
        self.started += 1
        return self.up

    def ask(self, sentence):
        for frag, verdict in self.verdicts.items():
            if frag in sentence:
                return verdict
        return "OTHER"

    def stop(self):
        self.stopped += 1


def _run(monkeypatch, *, judge=None, notify=None, argv=None):
    return cogito.run(argv or [],
                      judge_factory=(lambda: judge) if judge else None,
                      notify=notify or (lambda **kw: True),
                      today_et=TARGET)


# ---- prefilter + classify --------------------------------------------------

def test_prefilter_routes(monkeypatch):
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova", "the system"))
    assert cogito.prefilter("I'll commit it after the tests pass.") == "first"
    assert cogito.prefilter("Soy Nova, la asistente del operador.") == "first"   # soy wins
    assert cogito.prefilter("Nova's palate rated this highly.") == "name"
    assert cogito.prefilter("The system retries the request.") == "name"
    assert cogito.prefilter("The operator pushed the fix last night.") is None
    assert cogito.prefilter("Terminal-Bench scores dropped.") is None


def test_prefilter_skips_slug_and_code_spans(monkeypatch):
    """m3: a self-name inside a ``[[slug]]`` link or ``inline code`` names a note
    or systemd unit, not the assistant -- it must not reach the judge (the FP the
    first clean gold set exposed in its out-of-window tail)."""
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova", "the system"))
    assert cogito.prefilter("Realizes the owned-body tier of `nova-two-tier-own-body`.") is None
    assert cogito.prefilter("Adding it means a `nova-bridge.service` instance.") is None
    assert cogito.prefilter("Sourced from [[nova-two-tier-own-body-rent-ceiling]] mainly.") is None
    assert cogito.matched_name("Grounded in `nova-bridge` config.") is None
    # a genuine prose mention outside any span still routes to the judge
    assert cogito.prefilter("Emphatically not Nova's brain yet.") == "name"


def test_prefilter_ignores_outline_enumerator_i(monkeypatch):
    """m3: a bare capital 'I' opening an outline item is the Roman numeral for
    part I, not the pronoun (report row 42) -- so it is not first person, while a
    real pronoun (sentence-final or later in the line) is untouched."""
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    assert cogito.prefilter("- **I — Anatomy, layer by layer:** residual stream, MoE.") is None
    assert cogito.prefilter("I. Overview of the training loop.") is None
    assert cogito.prefilter("That call is up to you and I.") == "first"
    assert cogito.prefilter("I'll commit it after the tests pass.") == "first"
    assert cogito.prefilter("- **I — Anatomy**, which I wrote last night.") == "first"


def test_classify_regex_owns_first_judge_owns_referent(monkeypatch):
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    judge = _FakeJudge({"palate": "SPEAKER", "star": "OTHER"})
    assert cogito.classify("My palate scored it 0.7.", judge.ask) == ("first", "regex")
    assert cogito.classify("Nova's palate rated this highly.", judge.ask) == ("third", "judge")
    assert cogito.classify("Nova is the name of a star explosion.", judge.ask) == ("none", "judge")
    assert cogito.classify("Nothing about anyone here.", judge.ask) is None
    # no judge available -> the hard case is recorded honestly, never guessed
    assert cogito.classify("Nova's palate rated this highly.", None) == ("unjudged", "unjudged")


# ---- run: tracing + idempotency ---------------------------------------------

def test_run_traces_assistant_turns_only_and_is_idempotent(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    _write_buffer([
        _buffer_row(1, "user", "I want you to fix it."),          # user 'I' ignored
        _buffer_row(2, "assistant",
                    "I'll commit it tonight. Nova's palate liked it. The tests are green."),
    ])
    judge = _FakeJudge({"palate": "SPEAKER"})
    out = _run(monkeypatch, judge=judge)
    assert out.kind == "traced", out
    rows = [json.loads(x) for x in cogito.trace_path().read_text().splitlines()]
    assert [(r["stance"], r["via"]) for r in rows] == [
        ("first", "regex"), ("third", "judge")]                    # green-tests line skipped
    assert judge.started == 1 and judge.stopped == 1               # spawned once, torn down
    out2 = _run(monkeypatch, judge=_FakeJudge())
    assert "0 new record(s)" in out2.detail                        # idempotent re-run


def test_run_no_name_candidates_never_spawns_judge(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    _write_buffer([_buffer_row(1, "assistant", "I'll handle the rebuild myself.")])
    judge = _FakeJudge()
    out = _run(monkeypatch, judge=judge)
    assert out.kind == "traced" and judge.started == 0             # regex-only night


def test_run_judge_down_records_unjudged(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    _write_buffer([_buffer_row(1, "assistant", "Nova's palate liked the conjecture.")])
    out = _run(monkeypatch, judge=_FakeJudge(up=False))
    assert out.kind == "traced"
    row = json.loads(cogito.trace_path().read_text().splitlines()[0])
    assert row["stance"] == "unjudged" and row["via"] == "unjudged"


def test_stale_buffer_is_named_not_silent(tmp_path, monkeypatch):
    """A day with zero in-window rows while the buffer's newest row predates the
    window is a DEAD-source signal (buffer_stale), never a quiet 'traced 0'."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    _write_buffer([_buffer_row(1, "assistant", "I'll commit it tonight.",
                               ts="2026-06-20T15:00:00+00:00")])   # 4 days before TARGET
    out = _run(monkeypatch)
    assert out.kind == "skipped" and out.reason == "buffer_stale"
    assert "2026-06-20" in out.detail


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    _write_buffer([_buffer_row(1, "assistant", "I'll commit it tonight.")])
    out = _run(monkeypatch, argv=["--dry-run"])
    assert out.kind == "skipped" and out.reason == "dry_run"
    assert not cogito.trace_path().exists()


# ---- the 100-gate: one calibration report, ever ------------------------------

def test_calibration_report_triggers_once_at_threshold(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    monkeypatch.setattr(cogito, "REPORT_AT", 3)
    pings = []

    def notify(**kw):
        pings.append(kw)
        return True

    _write_buffer([_buffer_row(1, "assistant", "I'll commit it. My tests pass.")])
    out1 = _run(monkeypatch, notify=notify)                        # 2 records, under gate
    assert "REPORT" not in out1.detail and not pings
    _write_buffer([_buffer_row(9, "assistant", "I've mirrored it already.",
                               ts=f"{TARGET.isoformat()}T16:00:00+00:00")],
                  name="second-convo")
    out2 = _run(monkeypatch, notify=notify)                        # crosses 3
    assert "REPORT" in out2.detail and len(pings) == 1
    report = cogito.cogito_dir() / "calibration-3.md"
    assert report.exists()
    text = report.read_text()
    assert "first 3 self-reference records" in text and "gold:" in text
    _write_buffer([_buffer_row(20, "assistant", "I re-ran it once more.",
                               ts=f"{TARGET.isoformat()}T17:00:00+00:00")],
                  name="third-convo")
    _run(monkeypatch, notify=notify)                               # latch: no second report
    assert len(pings) == 1


# ---- integrity guards: anti-contamination armor ------------------------------

def test_degenerate_self_names_flags_stopwords_and_shorties():
    assert cogito.degenerate_self_names(("the",)) == ["the"]        # the 2026-07 bug
    assert cogito.degenerate_self_names(("Nova", "I", "  ")) == ["I", "  "]
    assert cogito.degenerate_self_names(("Nova", "the system", "the assistant")) == []


def test_matched_name_returns_the_hit(monkeypatch):
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova", "the system"))
    assert cogito.matched_name("The system retried it.").lower() == "the system"
    assert cogito.matched_name("The operator shipped it.") is None


def test_run_refuses_degenerate_self_name(tmp_path, monkeypatch):
    """A bare-stopword self-name (the 2026-07 'the' contamination) must abort the
    run loudly before any tracing, never silently generate noise."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova", "the"))
    _write_buffer([_buffer_row(1, "assistant", "Easy yes on the instinct.")])
    pings = []
    out = _run(monkeypatch, notify=lambda **kw: pings.append(kw) or True)
    assert out.kind == "failed" and out.reason == "degenerate_self_name"
    assert "the" in out.detail
    assert pings and pings[0]["priority"] == "high"                # paged loudly
    assert not cogito.trace_path().exists()                        # nothing traced


def test_run_match_rate_gate_refuses_saturating_name(tmp_path, monkeypatch):
    """Backstop: a non-stopword self-name that still matches too much prose is
    degenerate whatever its source; refuse before spawning the judge or writing."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    monkeypatch.setattr(cogito, "_GATE_MIN_SENTS", 4)              # small sample for test
    _write_buffer([_buffer_row(1, "assistant",
                               "Nova ran. Nova ran. Sam ran. Lee ran.")])   # 50% > 20%
    judge = _FakeJudge({"Nova": "SPEAKER"})
    pings = []
    out = _run(monkeypatch, judge=judge,
               notify=lambda **kw: pings.append(kw) or True)
    assert out.kind == "failed" and out.reason == "self_name_too_common"
    assert judge.started == 0                                      # bailed before the judge
    assert pings and pings[0]["priority"] == "high"
    assert not cogito.trace_path().exists()


def test_run_match_rate_gate_abstains_on_small_sample(tmp_path, monkeypatch):
    """Below the sample floor a rate is too noisy to judge -- the gate abstains so
    a real self-name on a quiet day is never falsely flagged."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))          # floor stays at 50
    _write_buffer([_buffer_row(1, "assistant", "Nova shipped it. Nova tested it.")])
    out = _run(monkeypatch, judge=_FakeJudge({"Nova": "SPEAKER"}))
    assert out.kind == "traced"                                   # 2 sentences < floor


def test_name_route_rows_record_which_name_matched(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(cogito, "SELF_NAMES", ("Nova",))
    _write_buffer([_buffer_row(1, "assistant",
                               "I'll commit it. Nova's palate liked it.")])
    _run(monkeypatch, judge=_FakeJudge({"palate": "SPEAKER"}))
    rows = [json.loads(x) for x in cogito.trace_path().read_text().splitlines()]
    regex_row = next(r for r in rows if r["via"] == "regex")
    judge_row = next(r for r in rows if r["via"] == "judge")
    assert "matched" not in regex_row                             # first-person: no name
    assert judge_row["matched"].lower() == "nova"                 # name route: records hit


# ---- layer 4: self-names from a JSON file (the robust channel) ---------------

def test_self_names_from_json_file_survive_spaces(tmp_path, monkeypatch):
    """A space-containing name in a JSON array loads intact -- the channel that
    can't be truncated the way the unquoted systemd env value was."""
    f = tmp_path / "self_names.json"
    f.write_text(json.dumps(["Nova", "the system", "the assistant"]))
    monkeypatch.setenv("RECALL_COGITO_SELF_NAMES_FILE", str(f))
    monkeypatch.delenv("RECALL_COGITO_SELF_NAMES", raising=False)
    assert cogito.load_self_names() == ("Nova", "the system", "the assistant")
    assert cogito.self_names_file_problem() is None


def test_self_names_file_takes_precedence_over_comma_env(tmp_path, monkeypatch):
    f = tmp_path / "self_names.json"
    f.write_text(json.dumps(["Nova"]))
    monkeypatch.setenv("RECALL_COGITO_SELF_NAMES_FILE", str(f))
    monkeypatch.setenv("RECALL_COGITO_SELF_NAMES", "should,be,ignored")
    assert cogito.load_self_names() == ("Nova",)


def test_broken_self_names_file_is_a_loud_problem(tmp_path, monkeypatch):
    bad = tmp_path / "broken.json"
    bad.write_text("{not json")
    monkeypatch.setenv("RECALL_COGITO_SELF_NAMES_FILE", str(bad))
    assert cogito.self_names_file_problem() is not None            # malformed
    monkeypatch.setenv("RECALL_COGITO_SELF_NAMES_FILE", str(tmp_path / "nope.json"))
    assert cogito.self_names_file_problem() is not None            # absent
    monkeypatch.delenv("RECALL_COGITO_SELF_NAMES_FILE", raising=False)
    assert cogito.self_names_file_problem() is None                # unset: fine


def test_run_refuses_when_configured_self_names_file_is_broken(tmp_path, monkeypatch):
    """A configured-but-broken source fails loud rather than silently running on
    fallback names that drop the operator's real self-names."""
    _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("RECALL_COGITO_SELF_NAMES_FILE", str(tmp_path / "missing.json"))
    _write_buffer([_buffer_row(1, "assistant", "Nova shipped it.")])
    pings = []
    out = _run(monkeypatch, notify=lambda **kw: pings.append(kw) or True)
    assert out.kind == "failed" and out.reason == "self_names_file"
    assert pings and pings[0]["priority"] == "high"
    assert not cogito.trace_path().exists()
