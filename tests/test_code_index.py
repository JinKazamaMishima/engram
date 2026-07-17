"""Unit tests for recall.code_index — the sonar code index (ast chunking + the
reused sqlite-vec/FTS5 machinery). The same tiny deterministic bag-of-words
embedder as test_index stands in for the real model, so build + search are
exercised with no model download and no daemon."""
from __future__ import annotations

import hashlib
import math

from recall import code_index, index


class FakeEmbedder:
    """Deterministic, normalized bag-of-words vectors — semantic similarity ≈
    word overlap, enough to exercise vec KNN ranking."""
    dim = 16

    def embed(self, texts, *, is_query=False):
        import re
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


# ---- chunking ------------------------------------------------------------

_SRC = "\n".join([
    '"""Module doc."""',          # 1
    "import os",                  # 2
    "",                           # 3
    "CONST = 1",                  # 4
    "",                           # 5
    "def top_level(a, b):",       # 6
    '    """Add two."""',         # 7
    "    return a + b",           # 8
    "",                           # 9
    "class Widget:",              # 10
    '    """A widget."""',        # 11
    "",                           # 12
    '    kind = "w"',             # 13
    "",                           # 14
    "    def __init__(self, x):",  # 15
    "        self.x = x",         # 16
    "",                           # 17
    "    def render(self):",      # 18
    "        return self.x",      # 19
])


def test_chunk_file_splits_symbols():
    chunks = code_index.chunk_file("top.py", _SRC)
    by_symbol = {c.symbol: c for c in chunks}
    # module preamble + top function + class header + two methods
    assert set(by_symbol) == {
        "top.py", "top_level", "Widget", "Widget.__init__", "Widget.render"}
    assert by_symbol["top.py"].slug == "top.py:L1-5"
    assert by_symbol["top.py"].description == "module top.py — Module doc."
    assert by_symbol["top_level"].slug == "top.py:L6-8"
    assert by_symbol["top_level"].description == "def top_level(a, b):  — Add two."
    # class header stops before the first method (docstring + class attrs only)
    assert by_symbol["Widget"].slug == "top.py:L10-14"
    assert "def __init__" not in by_symbol["Widget"].body
    assert by_symbol["Widget.__init__"].slug == "top.py:L15-16"
    assert by_symbol["Widget.render"].slug == "top.py:L18-19"


def test_chunk_file_includes_decorator_lines():
    src = "\n".join([
        "@decorator",          # 1
        "def wrapped():",      # 2
        "    return 1",        # 3
    ])
    chunks = code_index.chunk_file("d.py", src)
    fn = next(c for c in chunks if c.symbol == "wrapped")
    assert fn.slug == "d.py:L1-3"          # span starts at the decorator
    assert fn.body.startswith("@decorator")


def test_chunk_file_skips_unparseable():
    assert code_index.chunk_file("bad.py", "def (:\n    pass\n") == []


def test_chunk_file_module_only():
    chunks = code_index.chunk_file("consts.py", "import os\nX = 1\n")
    assert len(chunks) == 1
    assert chunks[0].symbol == "consts.py"


# ---- source discovery ----------------------------------------------------

def test_iter_source_files_skips_junk_dirs(tmp_path):
    (tmp_path / "real.py").write_text("x = 1\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("y = 2\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("z = 3\n")
    rels = {r for r, _ in code_index.iter_source_files(tmp_path)}
    assert rels == {"real.py", "pkg/mod.py"}       # .venv excluded, subdir kept


# ---- build + search (end to end) -----------------------------------------

def _write_repo(tmp_path):
    (tmp_path / "buffer.py").write_text(
        "def evict_oldest(ring):\n"
        '    """Drop the oldest entry from the ring buffer when full."""\n'
        "    return ring[1:]\n")
    (tmp_path / "colors.py").write_text(
        "def paint_palette(theme):\n"
        '    """Return the dim night palette for the tui."""\n'
        "    return theme\n")
    return tmp_path


def test_build_and_search_code_index(tmp_path):
    _write_repo(tmp_path)
    db = tmp_path / "code.sqlite"
    emb = FakeEmbedder()
    n = code_index.build_code_index(tmp_path, db, emb)
    assert n == 2                                   # one function chunk per file

    conn = index._connect(db, read_only=True)
    try:
        qv = emb.embed(["evict oldest ring buffer"], is_query=True)[0]
        hits = index.search(conn, "evict oldest ring buffer", query_vector=qv,
                            k=3, corpus_label="code", sem_floor=0.0)
        assert hits and hits[0].slug.startswith("buffer.py:L")
        assert hits[0].corpus == "code"
    finally:
        conn.close()


def test_keyword_only_code_index(tmp_path):
    _write_repo(tmp_path)
    db = tmp_path / "code.sqlite"
    assert code_index.build_code_index(tmp_path, db, None) == 2   # empty vec table
    conn = index._connect(db, read_only=True)
    try:
        hits = index.search(conn, "palette", query_vector=None, k=3, sem_floor=0.0)
        assert hits and hits[0].slug.startswith("colors.py:L")
    finally:
        conn.close()


def test_rebuild_is_full_and_atomic(tmp_path):
    _write_repo(tmp_path)
    db = tmp_path / "code.sqlite"
    emb = FakeEmbedder()
    assert code_index.build_code_index(tmp_path, db, emb) == 2
    (tmp_path / "extra.py").write_text("def added():\n    return 0\n")
    assert code_index.build_code_index(tmp_path, db, emb) == 3   # picked up, swapped


# ---- nightly registry sweep ----------------------------------------------

def test_code_build_all_iterates_registry(tmp_path, monkeypatch):
    """The nightly `code-build-all` builds a code.db for every registered repo,
    at config.code_index_path. embedder=None keeps it hermetic (keyword-only)."""
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    from recall import config, registry
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / "app.py").write_text("def handler():\n    return 1\n")
    assert registry.register(repo) is True

    rc = registry.code_build_all([], embedder=None)      # injected: no model load
    assert rc == 0

    db = config.code_index_path(config.project_slug(repo))
    assert db.exists()
    conn = index._connect(db, read_only=True)
    try:
        hits = index.search(conn, "handler", query_vector=None, k=3, sem_floor=0.0)
        assert hits and hits[0].slug.startswith("app.py:L")
    finally:
        conn.close()


def test_code_build_all_empty_registry_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DATA_ROOT", str(tmp_path / "data"))
    from recall import registry
    assert registry.code_build_all([], embedder=None) == 0   # nothing registered
