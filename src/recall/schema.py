"""Contracts the ``curate`` step must emit (data, never code), plus the on-disk
knowledge-note format the curate wrapper validates after the skill runs.

Structural validation only: a typed error, frozen dataclasses, ``from_dict`` /
``from_json`` staticmethods, a provenance hash. A malformed manifest or note is
a *failed* fire (the wrapper alerts and writes nothing to the corpus), never a
silent accept: a wrong corpus entry is worse than a missing one.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import yaml

SCHEMA_VERSION = 1
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ACTIONS = ("created", "updated")
_SCOPES = ("project", "global")


class CurationSchemaError(ValueError):
    """A manifest or knowledge note is structurally malformed (missing/wrong
    fields, bad slug, empty body). Distinct from a clean no-insight day, which
    is a *valid* manifest with ``notes: []`` and an explanatory ``summary``."""


def sha256_str(s: str) -> str:
    """Hex SHA-256 — pins exactly what produced a record (the bundle, the
    SKILL.md)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NoteEdit:
    """One note the curator created or updated this run. ``scope`` routes it:
    ``project`` → the project's ``docs/knowledge`` corpus (default); ``global``
    → the shared machine-local "soul" corpus."""

    slug: str
    action: str   # "created" | "updated"
    title: str    # the note's one-line description (drives the session log)
    scope: str = "project"   # "project" | "global"

    @staticmethod
    def from_dict(d: dict) -> "NoteEdit":
        if not isinstance(d, dict):
            raise CurationSchemaError(f"note entry not an object: {d!r}")
        try:
            slug = str(d["slug"]).strip()
            action = str(d["action"]).strip()
        except KeyError as e:
            raise CurationSchemaError(
                f"note entry missing {e}: {d!r}") from e
        title = str(d.get("title", "")).strip()
        scope = str(d.get("scope", "project")).strip().lower() or "project"
        if not SLUG_RE.match(slug):
            raise CurationSchemaError(
                f"note slug {slug!r} is not kebab-case [a-z0-9-]")
        if action not in _ACTIONS:
            raise CurationSchemaError(
                f"note {slug}: action {action!r} not in {_ACTIONS}")
        if scope not in _SCOPES:
            raise CurationSchemaError(
                f"note {slug}: scope {scope!r} not in {_SCOPES}")
        if not title:
            raise CurationSchemaError(f"note {slug}: title is empty")
        return NoteEdit(slug=slug, action=action, title=title, scope=scope)

    def to_dict(self) -> dict:
        return {"slug": self.slug, "action": self.action,
                "title": self.title, "scope": self.scope}


@dataclass(frozen=True)
class CurationManifest:
    """The single JSON file the skill writes describing what it did. The
    wrapper validates it, cross-checks every referenced note on disk, and turns
    ``summary`` into the session-log line."""

    date: str          # ISO date curated; must equal the wrapper's target
    summary: str       # one line for the session log; required even if empty day
    notes: list[NoteEdit] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def from_dict(d: dict) -> "CurationManifest":
        if not isinstance(d, dict):
            raise CurationSchemaError("manifest is not a JSON object")
        ver = d.get("schema_version", SCHEMA_VERSION)
        try:
            ver = int(ver)
        except (TypeError, ValueError) as e:
            raise CurationSchemaError(f"bad schema_version {ver!r}") from e
        if ver != SCHEMA_VERSION:
            raise CurationSchemaError(
                f"unsupported schema_version {ver} (expected {SCHEMA_VERSION})")

        date_s = str(d.get("date", "")).strip()
        if not date_s:
            raise CurationSchemaError("manifest.date is required")

        summary = str(d.get("summary", "")).strip()
        if not summary:
            raise CurationSchemaError(
                "manifest.summary is required and non-empty (it is the "
                "session-log line — even a no-insight day must say so)")

        raw_notes = d.get("notes", [])
        if not isinstance(raw_notes, list):
            raise CurationSchemaError("manifest.notes must be a list")
        notes = [NoteEdit.from_dict(x) for x in raw_notes]
        seen: set[str] = set()
        for n in notes:
            if n.slug in seen:
                raise CurationSchemaError(f"duplicate note slug {n.slug!r}")
            seen.add(n.slug)

        return CurationManifest(date=date_s, summary=summary, notes=notes,
                                schema_version=ver)

    @staticmethod
    def from_json(s: str) -> "CurationManifest":
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise CurationSchemaError(
                f"manifest is not valid JSON: {e}") from e
        return CurationManifest.from_dict(obj)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "date": self.date,
            "summary": self.summary,
            "notes": [n.to_dict() for n in self.notes],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class KnowledgeNote:
    """A parsed, validated knowledge note (YAML frontmatter + markdown body).
    Required frontmatter: ``name`` (kebab, == filename stem) and a non-empty
    ``description``; the body must be non-empty. ``kind`` is optional and free
    (e.g. ``identity`` / ``achievement`` for the global "soul" corpus; empty for
    ordinary domain notes). ``first_seen``/``last_updated`` (ISO dates) and
    ``superseded``/``superseded_by`` are optional and default empty/False — they
    feed recency ranking and bi-temporal supersession; notes without them parse
    unchanged (backward-compatible).

    Dynamic-memory fields (all optional, written by ``recall consolidate`` /
    curate, see docs/dynamic-memory.md): ``stability`` (DSR ``S`` in days — the
    decay rate), ``last_used`` (last *activation*, ISO date — distinct from
    ``last_updated`` which is the last *content* edit), ``uses`` (cumulative
    activation count), ``surprise`` (novelty σ∈[0,1] at encoding), ``importance``
    (EWC edit-resistance anchor for identity notes). A note without any of them
    parses and ranks exactly as before.

    Temporal-validity fields (all optional, Brick 3): ``valid_from`` (ISO date
    the fact became true; empty ≡ ``first_seen``), ``valid_to`` (ISO date it
    stopped being true; empty ≡ still true — a set ``valid_to`` renders as
    HISTORICAL at injection, it never changes ranking), ``confidence``
    (certainty ∈[0,1] of the *current* truth-value — distinct from ``surprise``
    which is novelty at encoding; provisional mid-session facts land ~0.3 and
    are raised or superseded by later passes)."""

    slug: str
    description: str
    body: str
    tags: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    kind: str = ""
    first_seen: str = ""
    last_updated: str = ""
    superseded: bool = False
    superseded_by: str = ""
    stability: float = 0.0
    last_used: str = ""
    uses: int = 0
    surprise: float = -1.0   # -1 == unset (distinct from a real σ of 0.0)
    importance: float = 0.0
    valid_from: str = ""
    valid_to: str = ""
    confidence: float = -1.0   # -1 == unset (never asserted, vs a real 0.0)

    @staticmethod
    def parse(text: str, *, expect_slug: str | None = None) -> "KnowledgeNote":
        fm, body = _split_frontmatter(text)
        name = str(fm.get("name") or "").strip()
        if not SLUG_RE.match(name):
            raise CurationSchemaError(
                f"note frontmatter 'name' {name!r} missing or not kebab-case")
        if expect_slug is not None and name != expect_slug:
            raise CurationSchemaError(
                f"note 'name' {name!r} != filename slug {expect_slug!r}")
        description = str(fm.get("description") or "").strip()
        if not description:
            raise CurationSchemaError(
                f"note {name}: frontmatter 'description' is required")
        if not body.strip():
            raise CurationSchemaError(f"note {name}: body is empty")
        superseded_by = str(fm.get("superseded_by") or "").strip()
        if superseded_by and not SLUG_RE.match(superseded_by):
            raise CurationSchemaError(
                f"note {name}: superseded_by {superseded_by!r} is not kebab-case")
        valid_from = str(fm.get("valid_from") or "").strip()
        valid_to = str(fm.get("valid_to") or "").strip()
        if valid_from and valid_to and valid_to < valid_from:
            # ISO dates compare lexicographically; a window that ends before it
            # starts is curator error, not a representable state.
            raise CurationSchemaError(
                f"note {name}: valid_to {valid_to!r} predates "
                f"valid_from {valid_from!r}")
        confidence = _as_float(fm.get("confidence"), -1.0)
        # Negative ≡ unset sentinel; anything else clamps into [0, 1].
        confidence = -1.0 if confidence < 0 else min(confidence, 1.0)
        return KnowledgeNote(
            slug=name, description=description, body=body.strip(),
            tags=_str_tuple(fm.get("tags")),
            sources=_str_tuple(fm.get("sources")),
            kind=str(fm.get("kind") or "").strip().lower(),
            first_seen=str(fm.get("first_seen") or "").strip(),
            last_updated=str(fm.get("last_updated") or "").strip(),
            superseded=_as_bool(fm.get("superseded")),
            superseded_by=superseded_by,
            stability=_as_float(fm.get("stability"), 0.0),
            last_used=str(fm.get("last_used") or "").strip(),
            uses=_as_int(fm.get("uses"), 0),
            surprise=_as_float(fm.get("surprise"), -1.0),
            importance=_as_float(fm.get("importance"), 0.0),
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
        )


def _split_frontmatter(text: str) -> tuple[dict, str]:
    t = text.lstrip("﻿")
    if not t.startswith("---"):
        raise CurationSchemaError("note missing YAML frontmatter (--- block)")
    parts = re.split(r"^---\s*$", t, maxsplit=2, flags=re.MULTILINE)
    if len(parts) < 3:
        raise CurationSchemaError("note frontmatter not terminated by '---'")
    block = parts[1]
    try:
        fm = yaml.safe_load(block) or {}
    except yaml.YAMLError:
        # The curator writes free-text prose (description) that routinely holds
        # ':', '#', '%', em-dashes and quotes. Unquoted, PyYAML reads 'a: b: c'
        # as a nested map (ScannerError) and would fail the whole nightly run.
        # The frontmatter grammar is tiny and flat, so on a *strict* failure
        # fall back to a line parser that takes scalar values verbatim. The
        # semantic checks in KnowledgeNote.parse still run on the result, so a
        # genuinely malformed note (no name/description, bad slug) still fails
        # loud — only a serialization slip is forgiven.
        fm = _loose_frontmatter(block)
    if not isinstance(fm, dict):
        raise CurationSchemaError("note frontmatter is not a mapping")
    return fm, parts[2]


def _loose_frontmatter(block: str) -> dict:
    """Forgiving fallback for the flat ``key: value`` / ``key: [a, b]``
    frontmatter the curator emits, invoked ONLY when strict ``yaml.safe_load``
    raises. Blank and ``#``-comment lines are skipped; every other line must be
    ``key: value`` (else the note is genuinely malformed and we raise). Scalar
    values are taken verbatim, so an internal ``:``/``#``/``%``/quote survives
    untouched."""
    fm: dict = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition(":")
        key = key.strip()
        if not sep or not key or any(c.isspace() for c in key):
            raise CurationSchemaError(
                f"note frontmatter line is not 'key: value': {raw!r}")
        fm[key] = _loose_value(val.strip())
    return fm


def _loose_value(val: str):
    """A frontmatter scalar or single-line flow list, parsed leniently. An
    already-quoted scalar is unwrapped; a closed ``[...]`` flow list is split on
    commas (inner quotes stripped); anything else is returned verbatim as a
    string (so ``a: b`` prose or an unterminated ``[…`` survives intact)."""
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        return ([x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
                if inner else [])
    return val


def _str_tuple(v) -> tuple[str, ...]:
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip() for x in v if str(x).strip())
    return (str(v).strip(),) if v is not None and str(v).strip() else ()


def _as_bool(v) -> bool:
    """Coerce a frontmatter value to bool. YAML usually yields a real bool, but a
    quoted ``"false"``/``"no"``/``"0"``/``""`` must NOT read as True."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("true", "yes", "1", "on")


def _as_float(v, default: float) -> float:
    """Coerce a frontmatter scalar to float; ``default`` on absent/garbled."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _as_int(v, default: int) -> int:
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return default


_FM_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_-]+)(\s*):")


def set_frontmatter_keys(text: str, updates: dict[str, object]) -> str:
    """Surgically set flat *scalar* frontmatter keys, preserving every other line
    byte-for-byte — so the curator's hand-quoted ``description`` (and any other
    field) is left untouched. ONLY for simple scalar values (numbers, ISO dates):
    values are written verbatim with no quoting, so never pass prose here. A key
    present in the block is replaced in place; a new key is inserted just before
    the closing ``---``. The body is never touched. Raises if the frontmatter is
    malformed (no opening/closing ``---``).

    This is the write-side counterpart to the read-side tolerant parser: the
    consolidate fold uses it to bump ``stability``/``last_used``/``uses`` without
    round-tripping (and possibly reformatting) the whole YAML block."""
    if not updates:
        return text
    t = text.lstrip("﻿")
    if not t.startswith("---"):
        raise CurationSchemaError("note missing YAML frontmatter (--- block)")
    lines = t.splitlines(keepends=True)
    close = None
    for i in range(1, len(lines)):
        if lines[i].rstrip("\r\n").strip() == "---":
            close = i
            break
    if close is None:
        raise CurationSchemaError("note frontmatter not terminated by '---'")
    remaining = {str(k): v for k, v in updates.items()}
    for i in range(1, close):
        m = _FM_KEY_RE.match(lines[i])
        if not m:
            continue
        key = m.group(2)
        if key in remaining:
            indent = m.group(1)
            newline = "\n" if lines[i].endswith("\n") else ""
            lines[i] = f"{indent}{key}: {remaining.pop(key)}{newline}"
    if remaining:
        insert = [f"{k}: {v}\n" for k, v in remaining.items()]
        lines[close:close] = insert
    return "".join(lines)
