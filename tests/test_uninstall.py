"""Tests for scripts/uninstall.py — the clean-uninstall flow.

Fully sandboxed: uninstall.py derives the repo from its own __file__ and the home
from Path.home(), so we run a COPY of it inside a throwaway fake repo (with a fake
.venv) under a throwaway fake $HOME, as a subprocess. Nothing real is ever touched
— in particular no real virtualenv can be deleted. Always --no-systemd so the test
never pokes the runner's systemd.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

UNINSTALL_SRC = Path(__file__).resolve().parents[1] / "scripts" / "uninstall.py"
SKILL_NAMES = ("recall", "curate-memory", "dream", "reconsolidate-memory")


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "uninstall.py").write_bytes(UNINSTALL_SRC.read_bytes())
    for name in SKILL_NAMES:
        (repo / "skills" / name).mkdir(parents=True)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "recall").write_text("#!/bin/sh\n")
    return repo


def _make_home(tmp_path: Path, repo: Path) -> tuple[Path, Path]:
    home = tmp_path / "home"
    for name in SKILL_NAMES:
        (home / ".claude" / "skills" / name).mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(json.dumps({
        "model": "opus",
        "theme": "dark",
        "hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command",
                        "command": f"{repo}/.venv/bin/python {repo}/scripts/recall_inject.py"}]}]},
    }))
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".local" / "bin" / "recall").symlink_to(repo / ".venv" / "bin" / "recall")
    data = home / ".local" / "share" / "recall"
    (data / "global").mkdir(parents=True)
    (data / "global" / "x.md").write_text("a note")
    (home / ".config" / "engram").mkdir(parents=True)
    (home / ".config" / "engram" / "engram.env").write_text(f"RECALL_DATA_ROOT={data}\n")
    (home / ".config" / "recall").mkdir(parents=True)
    (home / ".config" / "recall" / "telegram-agent.env").write_text("RECALL_TELEGRAM_AGENT_TOKEN=secret\n")
    (home / ".zshrc").write_text(
        "export EDITOR=vim\n"
        "alias ll='ls -la'\n\n"
        "# Engram\n"
        "[ -f /x/engram.env ] && set -a && . /x/engram.env && set +a\n\n"
        "# Engram — recall launcher on PATH\n"
        'export PATH="$HOME/.local/bin:$PATH"\n\n'
        "export FOO=bar\n")
    sysd = home / ".config" / "systemd" / "user"
    sysd.mkdir(parents=True)
    (sysd / "recall-curate.timer").write_text("")
    return home, data


def _run(repo: Path, home: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=str(home), NO_COLOR="1")
    env.pop("RECALL_DATA_ROOT", None)  # force resolution via engram.env
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "uninstall.py"), *args],
        env=env, cwd=str(repo), capture_output=True, text=True, timeout=60)


@pytest.fixture()
def sandbox(tmp_path):
    repo = _make_repo(tmp_path)
    home, data = _make_home(tmp_path, repo)
    return repo, home, data


def test_dry_run_changes_nothing(sandbox):
    repo, home, data = sandbox
    r = _run(repo, home, "--dry-run")
    assert r.returncode == 0
    assert "would remove" in r.stdout
    # untouched
    assert (home / ".claude" / "skills" / "recall").is_dir()
    assert (home / ".local" / "bin" / "recall").exists()
    assert (repo / ".venv").is_dir()
    assert data.is_dir()


def test_removes_footprint_but_keeps_data(sandbox):
    repo, home, data = sandbox
    r = _run(repo, home, "--yes", "--no-systemd")
    assert r.returncode == 0
    assert not (home / ".local" / "bin" / "recall").exists()          # launcher
    assert not (home / ".claude" / "skills" / "recall").exists()      # skills
    assert not (home / ".config" / "engram").exists()                 # config
    assert not (home / ".config" / "recall" / "telegram-agent.env").exists()
    assert not (repo / ".venv").exists()                              # venv
    assert data.is_dir()                                              # DATA KEPT
    assert (data / "global" / "x.md").exists()


def test_hook_removed_other_settings_preserved(sandbox):
    repo, home, _ = sandbox
    _run(repo, home, "--yes", "--no-systemd")
    settings = json.loads((home / ".claude" / "settings.json").read_text())
    hooks = settings.get("hooks", {}).get("UserPromptSubmit", [])
    assert not any("recall_inject.py" in h.get("command", "")
                   for g in hooks for h in g.get("hooks", []))
    assert settings["model"] == "opus" and settings["theme"] == "dark"


def test_rc_engram_blocks_stripped_user_lines_kept(sandbox):
    repo, home, _ = sandbox
    _run(repo, home, "--yes", "--no-systemd")
    rc = (home / ".zshrc").read_text()
    assert "# Engram" not in rc
    assert ".local/bin" not in rc          # the launcher PATH line is gone
    assert "export EDITOR=vim" in rc       # user lines survive
    assert "export FOO=bar" in rc
    assert (home / ".zshrc.pre-engram-uninstall.bak").exists()


def test_no_systemd_leaves_units(sandbox):
    repo, home, _ = sandbox
    _run(repo, home, "--yes", "--no-systemd")
    assert (home / ".config" / "systemd" / "user" / "recall-curate.timer").exists()


def test_purge_data_removes_corpus(sandbox):
    repo, home, data = sandbox
    r = _run(repo, home, "--yes", "--purge-data", "--no-systemd")
    assert r.returncode == 0
    assert not data.exists()


def test_idempotent_second_run_finds_nothing(sandbox):
    repo, home, _ = sandbox
    _run(repo, home, "--yes", "--no-systemd")
    r = _run(repo, home, "--yes", "--no-systemd")
    assert r.returncode == 0
    assert "nothing to remove" in r.stdout.lower()
