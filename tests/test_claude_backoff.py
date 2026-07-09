"""Unit tests for run_claude_with_backoff — the transient-overload retry that
keeps a 529 capacity blip from failing the nightly memory cycle and paging the
operator, while still failing fast on real faults (bad flag, auth, missing bin).
"""
from __future__ import annotations

import subprocess
import types

from recall import curate, dream, reconsolidate
from recall.curate import _TRANSIENT_RE, run_claude_with_backoff


def _cp(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeRunner:
    """Returns a scripted sequence of CompletedProcess results, one per call."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._results.pop(0)


def _run(runner, **kw):
    slept: list[float] = []
    cp = run_claude_with_backoff(
        ["claude", "-p", "/curate-memory"],
        env={}, cwd="/tmp", timeout=10,
        runner=runner, sleep=slept.append, jitter=lambda: 0.0, **kw)
    return cp, slept


def test_retries_transient_then_succeeds():
    runner = _FakeRunner([
        _cp(1, stdout=b"API Error: 529 Overloaded. try again in a moment"),
        _cp(0, stdout=b"Curation complete."),
    ])
    cp, slept = _run(runner)
    assert cp.returncode == 0
    assert runner.calls == 2
    assert len(slept) == 1          # backed off exactly once


def test_no_retry_on_real_failure():
    runner = _FakeRunner([_cp(1, stderr=b"error: unknown option '--bogus'")])
    cp, slept = _run(runner)
    assert cp.returncode == 1
    assert runner.calls == 1        # failed fast, no retry
    assert slept == []


def test_exhausts_attempts_on_persistent_overload():
    runner = _FakeRunner([_cp(1, stdout=b"529 Overloaded") for _ in range(5)])
    cp, slept = _run(runner, attempts=3)
    assert cp.returncode == 1
    assert runner.calls == 3        # attempts cap honored
    assert len(slept) == 2          # slept between the 3 attempts


def test_backoff_is_exponential():
    runner = _FakeRunner([_cp(1, stdout=b"overloaded") for _ in range(4)])
    _, slept = _run(runner, attempts=4, base_delay=5.0)
    # jitter pinned to 0 -> pure exponential: 5, 10, 20 (last attempt doesn't sleep)
    assert slept == [5.0, 10.0, 20.0]


def test_transient_regex_matches():
    for s in ("API Error: 529 Overloaded", "HTTP 429", "rate limit exceeded",
              "503 Service Unavailable", "overloaded_error"):
        assert _TRANSIENT_RE.search(s), f"should match: {s}"
    for s in ("unknown option --bogus", "authentication_error", "exit code 2"):
        assert not _TRANSIENT_RE.search(s), f"should NOT match: {s}"


# --- model/effort argv contract ------------------------------------------------
# A typo in a model or effort string bricks the whole nightly job (and retry can't
# save a bad model), so pin the intended defaults AND assert the argv wires them.

def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _capture_argv(monkeypatch, module) -> dict:
    box: dict = {}

    def fake(argv, **kw):
        box["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(module, "run_claude_with_backoff", fake)
    return box


def test_curate_argv_pins_sonnet5_xhigh(monkeypatch):
    assert curate.CURATE_MODEL == "claude-sonnet-5"
    assert curate.CURATE_EFFORT == "xhigh"
    box = _capture_argv(monkeypatch, curate)
    curate._invoke_claude(types.SimpleNamespace(project_dir="/tmp"), {}, 10)
    assert _flag(box["argv"], "--model") == curate.CURATE_MODEL
    assert _flag(box["argv"], "--effort") == curate.CURATE_EFFORT


def test_reconsolidate_argv_pins_opus48_1m_xhigh(monkeypatch):
    assert reconsolidate.RECON_MODEL == "claude-opus-4-8[1m]"
    assert reconsolidate.RECON_EFFORT == "xhigh"
    box = _capture_argv(monkeypatch, reconsolidate)
    reconsolidate._invoke_claude(types.SimpleNamespace(repo="/tmp"), {}, 10)
    assert _flag(box["argv"], "--model") == reconsolidate.RECON_MODEL
    assert _flag(box["argv"], "--effort") == reconsolidate.RECON_EFFORT


def test_dream_argv_pins_opus48_1m_xhigh(monkeypatch):
    assert dream.DREAM_MODEL == "claude-opus-4-8[1m]"
    assert dream.DREAM_EFFORT == "xhigh"
    box = _capture_argv(monkeypatch, dream)
    dream._invoke_claude(types.SimpleNamespace(repo="/tmp"), {}, 10)
    assert _flag(box["argv"], "--model") == dream.DREAM_MODEL
    assert _flag(box["argv"], "--effort") == dream.DREAM_EFFORT
