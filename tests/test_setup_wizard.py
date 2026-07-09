"""Unit tests for the setup wizard's non-interactive plumbing.

The wizard itself is an interactive TUI; these tests cover the pure helpers
that the auth-step rework introduced — auth-status probing, API-key
validation, previous-answer reload, --yes prompt bypass, and the final
verification checks.
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    "setup_wizard", Path(__file__).parent.parent / "scripts" / "setup_wizard.py")
wizard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wizard)


# ---- claude_auth_status -------------------------------------------------------

def _fake_run(stdout: str, returncode: int = 0):
    def fake(cmd, **kwargs):
        return SimpleNamespace(stdout=stdout, returncode=returncode)
    return fake


def test_auth_status_logged_in(monkeypatch):
    payload = {"loggedIn": True, "email": "a@b.c", "subscriptionType": "max"}
    monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps(payload)))
    assert wizard.claude_auth_status("/usr/bin/claude") == payload


def test_auth_status_logged_out_nonzero_rc_still_parses(monkeypatch):
    # `claude auth status --json` exits 1 when logged out but still emits JSON.
    payload = {"loggedIn": False, "authMethod": "none"}
    monkeypatch.setattr(subprocess, "run", _fake_run(json.dumps(payload), returncode=1))
    assert wizard.claude_auth_status("claude") == payload


def test_auth_status_none_when_no_cli():
    assert wizard.claude_auth_status(None) is None


def test_auth_status_none_on_old_cli_garbage(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("error: unknown command 'auth'"))
    assert wizard.claude_auth_status("claude") is None


def test_auth_status_none_on_exception(monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 30)
    monkeypatch.setattr(subprocess, "run", boom)
    assert wizard.claude_auth_status("claude") is None


def test_describe_login():
    assert wizard.describe_login({"email": "a@b.c", "subscriptionType": "max"}) \
        == "a@b.c (max subscription)"
    assert wizard.describe_login({}) == "your Anthropic account"


# ---- validate_api_key ---------------------------------------------------------

class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_validate_api_key_ok(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=0: _FakeResponse(b"{}"))
    assert wizard.validate_api_key("sk-ant-good") == (True, "")


def test_validate_api_key_rejected(monkeypatch):
    def raise_401(req, timeout=0):
        raise urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", raise_401)
    ok, why = wizard.validate_api_key("sk-ant-bad")
    assert ok is False and "401" in why


def test_validate_api_key_indeterminate_on_server_error(monkeypatch):
    def raise_529(req, timeout=0):
        raise urllib.error.HTTPError("url", 529, "Overloaded", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", raise_529)
    ok, _ = wizard.validate_api_key("sk-ant-x")
    assert ok is None


def test_validate_api_key_indeterminate_offline(monkeypatch):
    def raise_url(req, timeout=0):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(urllib.request, "urlopen", raise_url)
    ok, _ = wizard.validate_api_key("sk-ant-x")
    assert ok is None


# ---- load_prev_env ------------------------------------------------------------

def test_load_prev_env_parses_and_skips_noise(tmp_path, monkeypatch):
    env_file = tmp_path / "engram.env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "RECALL_DATA_ROOT=/data/x\n"
        "ENGRAM_USER=alex\n"
        "malformed line without equals\n"
        "ANTHROPIC_API_KEY=sk-ant-123=with=equals\n")
    monkeypatch.setattr(wizard, "ENV_FILE", env_file)
    prev = wizard.load_prev_env()
    assert prev == {
        "RECALL_DATA_ROOT": "/data/x",
        "ENGRAM_USER": "alex",
        "ANTHROPIC_API_KEY": "sk-ant-123=with=equals",
    }


def test_load_prev_env_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(wizard, "ENV_FILE", tmp_path / "nope.env")
    assert wizard.load_prev_env() == {}


# ---- --yes prompt bypass --------------------------------------------------------

def test_assume_yes_returns_defaults_without_prompting(monkeypatch):
    monkeypatch.setattr(wizard, "ASSUME_YES", True)
    # If any of these tried to prompt, reading stdin under pytest would fail.
    assert wizard.confirm("install?", default=True) is True
    assert wizard.confirm("install?", default=False) is False
    assert wizard.ask("dir", default="/tmp/x") == "/tmp/x"
    assert wizard.ask_choice("auth", ["subscription", "apikey"], "subscription") \
        == "subscription"


def test_assume_yes_subscription_skips_interactive_login(monkeypatch):
    monkeypatch.setattr(wizard, "ASSUME_YES", True)
    calls = []
    monkeypatch.setattr(wizard, "run", lambda *a, **k: calls.append(a) or 0)
    env: dict = {}
    facts = {"claude": "/usr/bin/claude", "auth": {"loggedIn": False}}
    wizard.auth_subscription(env, facts)
    assert env["ENGRAM_FORCE_SUBSCRIPTION"] == "1"
    assert calls == []  # never launched `claude auth login` (or the TUI) unattended


# ---- final verification checks --------------------------------------------------

def test_hook_and_skills_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert wizard.hook_installed() is False
    assert wizard.skills_installed() is False
    claude_dir = tmp_path / ".claude"
    (claude_dir / "skills" / "recall").mkdir(parents=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": "python x/recall_inject.py"}]}]}}))
    assert wizard.hook_installed() is True
    assert wizard.skills_installed() is True
