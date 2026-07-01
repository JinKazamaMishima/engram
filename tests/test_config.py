"""Unit tests for recall.config — scope/path resolution. Env overrides keep
these hermetic (no dependence on the real ~/.local/share/recall)."""
from __future__ import annotations

from pathlib import Path

from recall import config


def test_project_slug_basename_sanitized():
    assert config.project_slug("/home/user/repos/myproject") == "myproject"
    assert config.project_slug("/home/user/repos/myproject-v2") == "myproject-v2"
    assert config.project_slug("/x/My Cool Game!!") == "my-cool-game"
    assert config.project_slug("/x/___") == "project"  # degenerate -> fallback


def test_data_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("RECALL_GLOBAL_DIR", raising=False)
    assert config.data_root() == tmp_path
    assert config.index_dir() == tmp_path / "index"
    assert config.global_corpus_dir() == tmp_path / "global"
    assert config.index_path("myproject") == tmp_path / "index" / "myproject.sqlite"
    assert config.index_path(config.GLOBAL_SCOPE) == tmp_path / "index" / "global.sqlite"


def test_global_dir_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("RECALL_GLOBAL_DIR", str(tmp_path / "soul"))
    assert config.global_corpus_dir() == tmp_path / "soul"


def test_project_corpus_dir_convention():
    p = Path("/home/user/repos/myproject")
    assert config.project_corpus_dir(p) == p / "docs" / "knowledge"


def test_ensure_dirs_creates(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    config.ensure_dirs(target)
    assert target.is_dir()
