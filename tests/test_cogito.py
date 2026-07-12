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
