#!/usr/bin/env python3
"""sonar m1 — a disposable semantic + keyword index over a repository's SOURCE.

Sibling to :mod:`recall.index` (the note corpus) but a SEPARATE database per repo
(``config.code_index_path``): code chunks never enter the note corpus, its eval,
or the injection tier. This index exists only to be *pulled* on demand via the
``code_search`` tool, so the harness can serve the right file+line range to the
model instead of it grepping blind. The files are the source of truth; the index
is rebuilt wholesale (atomic swap), exactly like the note index — delete it and
it regenerates.

m1 is Python-only. Files come from ``git ls-files`` (so .gitignore / .venv /
build dirs are skipped for free), and each is split by the stdlib ``ast`` into
module-preamble / function / class-header / method chunks — no tree-sitter, no
new dependency. Embedding reuses the SAME warm daemon as the note index (via the
:mod:`recall.index` machinery), so no second model loads on the contended GPU.
Multi-language chunking (tree-sitter) and incremental per-file rebuilds are later
milestones.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from recall import index
from recall.schema import sha256_str

# Chunk text handed to the embedder is capped so one very large function can't
# spike the daemon's padded batch; the FULL body is still stored for display/read.
EMBED_CHAR_CAP = 4000

# Only used by the non-git fallback walk — ``git ls-files`` already excludes these.
_SKIP_DIRS = frozenset({".git", ".venv", "venv", "env", "__pycache__",
                        "node_modules", "build", "dist", ".mypy_cache",
                        ".pytest_cache", ".ruff_cache", ".tox"})


@dataclass(frozen=True)
class CodeChunk:
    slug: str          # "relpath:Lstart-Lend" — a Read-able locator AND the unique key
    description: str   # signature (+ first docstring line) — the search-result label
    body: str          # the source slice
    symbol: str        # dotted symbol name (Class.method / module relpath), for FTS + debug


def iter_source_files(repo: str | Path) -> list[tuple[str, Path]]:
    """``(relpath, abspath)`` for every tracked ``*.py`` file via ``git ls-files``
    (so .gitignore, .venv, build dirs are excluded for free). Falls back to a
    filtered recursive walk when ``repo`` is not a git repo / git is absent.
    Sorted, for deterministic builds."""
    repo = Path(repo).resolve()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z", "--", "*.py"],
            capture_output=True, check=True).stdout
        rels = sorted(f.decode() for f in out.split(b"\x00") if f)
        return [(r, repo / r) for r in rels]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        walked: list[tuple[str, Path]] = []
        for p in sorted(repo.rglob("*.py")):
            rel = p.relative_to(repo)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            walked.append((str(rel), p))
        return walked


def _span(node: ast.AST) -> tuple[int, int]:
    """1-based ``(start, end)`` line span INCLUDING any decorators (whose lines
    sit above ``node.lineno``)."""
    start = node.lineno
    for dec in getattr(node, "decorator_list", None) or []:
        start = min(start, dec.lineno)
    return start, getattr(node, "end_lineno", node.lineno)


def _describe(node, lines: list[str], class_name: str = "") -> tuple[str, str]:
    """``(dotted symbol, one-line description)`` for a def/class node — its source
    signature line plus the first line of its docstring when present."""
    sig = (lines[node.lineno - 1].strip()
           if 0 < node.lineno <= len(lines) else node.name)
    name = f"{class_name}.{node.name}" if class_name else node.name
    doc = ast.get_docstring(node)
    if doc:
        first = doc.strip().splitlines()[0].strip()
        if first:
            return name, f"{sig}  — {first}"
    return name, sig


def chunk_file(relpath: str, source: str) -> list[CodeChunk]:
    """Split one Python source into module-preamble / function / class-header /
    method chunks. Returns ``[]`` for an unparseable file (syntax error mid-edit)
    — the caller logs and skips it, never failing the build."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    def emit(node, class_name: str = "") -> None:
        start, end = _span(node)
        symbol, desc = _describe(node, lines, class_name)
        chunks.append(CodeChunk(slug=f"{relpath}:L{start}-{end}", description=desc,
                                body="\n".join(lines[start - 1:end]), symbol=symbol))

    first_def: int | None = None
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        start, _ = _span(node)
        first_def = start if first_def is None else min(first_def, start)
        if isinstance(node, ast.ClassDef):
            methods = [s for s in node.body
                       if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            cstart, cend = _span(node)
            # Class-header chunk = the class line up to just before its first method
            # (its docstring + class-level attrs); each method becomes its own chunk,
            # so a big class never collapses into one blob and methods stay findable.
            header_end = max(cstart, min((_span(m)[0] for m in methods),
                                         default=cend + 1) - 1)
            symbol, desc = _describe(node, lines)
            chunks.append(CodeChunk(slug=f"{relpath}:L{cstart}-{header_end}",
                                    description=desc,
                                    body="\n".join(lines[cstart - 1:header_end]),
                                    symbol=symbol))
            for meth in methods:
                emit(meth, class_name=node.name)
        else:
            emit(node)

    # Module preamble (imports, module docstring, top-level constants): the code
    # before the first def/class, or the whole file when it defines none.
    mod_end = (first_def - 1) if first_def else len(lines)
    preamble = "\n".join(lines[:mod_end]).strip()
    if preamble:
        doc = ast.get_docstring(tree)
        desc = f"module {relpath}"
        if doc:
            desc += f" — {doc.strip().splitlines()[0].strip()}"
        chunks.insert(0, CodeChunk(slug=f"{relpath}:L1-{max(1, mod_end)}",
                                   description=desc, body=preamble, symbol=relpath))
    return chunks


def build_code_index(repo_dir: str | Path, db_path: str | Path, embedder) -> int:
    """Full rebuild of the code index into a temp DB, then atomic-swap into place.
    Returns the number of chunks indexed. Reuses the note index's schema +
    connection helpers verbatim — a code chunk is stored as a note-shaped row
    (slug=locator, description=signature, body=source, symbol→tags) so
    ``index.search`` works over it UNCHANGED. ``embedder=None`` builds a
    keyword-only index (FTS5 + an empty vec table)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".building")
    if tmp.exists():
        tmp.unlink()

    chunks: list[CodeChunk] = []
    for relpath, abspath in iter_source_files(repo_dir):
        try:
            source = Path(abspath).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        file_chunks = chunk_file(relpath, source)
        if not file_chunks and source.strip():
            print(f"[code-index] skip unparseable {relpath}", file=sys.stderr)
        chunks.extend(file_chunks)

    conn = index._connect(tmp)
    try:
        index._create_schema(
            conn, embedder.dim if embedder is not None else index.EMBED_DIM)
        if chunks:
            vecs = (embedder.embed([f"{c.description}\n\n{c.body}"[:EMBED_CHAR_CAP]
                                    for c in chunks])
                    if embedder is not None else [None] * len(chunks))
            for i, (chunk, vec) in enumerate(zip(chunks, vecs), start=1):
                conn.execute(
                    "INSERT INTO notes(id,slug,description,body,tags,sources,kind,"
                    "sha,last_updated,sources_count,stability,last_used,uses,"
                    "valid_to) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, chunk.slug, chunk.description, chunk.body, chunk.symbol,
                     "code", "code", sha256_str(chunk.body), "", 0, 0.0, "", 0, ""))
                conn.execute(
                    "INSERT INTO notes_fts(rowid,slug,description,body,tags)"
                    " VALUES (?,?,?,?,?)",
                    (i, chunk.slug, chunk.description, chunk.body, chunk.symbol))
                if vec is not None:
                    conn.execute(
                        "INSERT INTO vec_notes(note_id,embedding) VALUES (?,?)",
                        (i, sqlite_vec.serialize_float32(vec)))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, db_path)
    return len(chunks)
