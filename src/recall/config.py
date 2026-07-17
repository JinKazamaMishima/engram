"""Filesystem conventions for the machine-local recall install.

The markdown corpora are the source of truth and live in human places: a
project's ``docs/knowledge/`` and the shared global/"soul" corpus. Everything
under the recall *data root* is derived and disposable — the sqlite indices and
the curator's bundles/manifests/state — and can be deleted and rebuilt anytime.

Overridable via env so tests and alternate machines never hardcode a home:
  RECALL_DATA_ROOT   derived data + indices   (default ~/.local/share/recall)
  RECALL_GLOBAL_DIR  the shared "soul" corpus (default <data_root>/global)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_DATA_ROOT = Path.home() / ".local" / "share" / "recall"
PROJECT_CORPUS_RELPATH = Path("docs") / "knowledge"
GLOBAL_SCOPE = "global"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def data_root() -> Path:
    """Root of all derived recall data (indices, bundles, manifests, state).
    Never raises if absent — recall is read-mostly and must degrade, not crash;
    writers call :func:`ensure_dirs` before writing."""
    return Path(os.environ.get("RECALL_DATA_ROOT") or DEFAULT_DATA_ROOT).expanduser()


def global_corpus_dir() -> Path:
    """The shared, machine-local "soul" corpus every project also recalls from."""
    override = os.environ.get("RECALL_GLOBAL_DIR")
    return Path(override).expanduser() if override else data_root() / GLOBAL_SCOPE


def index_dir() -> Path:
    """Directory holding one derived sqlite index per scope."""
    return data_root() / "index"


def curation_dir() -> Path:
    """Where the curator writes per-day bundles/manifests + idempotency state."""
    return data_root() / "curation"


def engram_buffer_dir() -> Path:
    """Where the Engram harness appends its per-conversation LiveBuffer JSONLs
    (``<convo_id>.jsonl``) — tier-1 STM, read by ``curate --buffer``."""
    return data_root() / "engram" / "buffer"


def subconscious_dir(scope: str) -> Path:
    """Quarantined staging for the nightly dream pass — hypothesis notes recombined
    from the day's memory. Deliberately OUTSIDE the corpus and NEVER indexed into
    live recall: an unverified dream must not surface as curated fact. Material
    only reaches the soul through the bleed membrane (corroboration-gated). One
    dir per scope (a project slug, or ``global``)."""
    return data_root() / "subconscious" / scope


def project_slug(project_dir: str | Path) -> str:
    """Stable, human-readable scope id from a project directory's name.

    ``/home/user/repos/myproject`` -> ``myproject``; sanitized to ``[a-z0-9-]``.
    Note: keyed on the basename, so two projects with the same directory name in
    different parents would collide — acceptable on a single dev box; revisit
    with a path hash if it ever bites."""
    name = Path(project_dir).resolve().name.lower()
    slug = _SLUG_RE.sub("-", name).strip("-")
    return slug or "project"


def index_path(scope: str) -> Path:
    """Derived index DB for a scope id (a project slug, or ``global``)."""
    return index_dir() / f"{scope}.sqlite"


def project_corpus_dir(project_dir: str | Path) -> Path:
    """Convention for a project's own corpus: ``<project>/docs/knowledge``."""
    return Path(project_dir).resolve() / PROJECT_CORPUS_RELPATH


def archive_dir(corpus_dir: str | Path) -> Path:
    """The reversible graveyard for a corpus: ``<corpus>/archive``. The reaper
    (``recall reap``) MOVES cold/superseded notes here. Because the note loaders
    glob non-recursively (``glob("*.md")``), an archived note drops out of the
    rebuilt index automatically while staying in the same repo — still on disk,
    still ``cat``-able (so the miss-log can catch a wrong eviction), one
    ``reap --restore`` from coming back."""
    return Path(corpus_dir) / "archive"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
