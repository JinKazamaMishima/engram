"""Unit tests for recall.index — the sqlite-vec + FTS5 hybrid recall index. A
tiny deterministic bag-of-words embedder stands in for bge-small so the plumbing
(build, vec KNN, FTS5, RRF, keyword-only fallback, provenance) is tested with no
model download."""
from __future__ import annotations

import hashlib
import math
import re

from recall import index


class FakeEmbedder:
    """Deterministic, normalized bag-of-words vectors — semantic similarity ≈
    word overlap, enough to exercise vec KNN ranking."""
    dim = 16

    def embed(self, texts, *, is_query=False):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _note(d, slug, desc, body, tags="t", kind=None):
    extra = f"kind: {kind}\n" if kind else ""
    (d / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: {desc}\ntags: [{tags}]\n{extra}---\n{body}\n")


# ---- build + hybrid query ------------------------------------------------

def test_build_and_hybrid_query(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "index-reconstitution", "reconstitution forces rebalance",
          "Passive funds rebalance at the close on the reconstitution "
          "effective date.")
    _note(kd, "borrow-fees", "hard to borrow fees",
          "Short borrow fees spike when a name is hard to borrow.")
    db = tmp_path / "i.sqlite"
    emb = FakeEmbedder()
    assert index.build_index(kd, db, emb) == 2

    conn = index._connect(db, read_only=True)
    try:
        qv = emb.embed(["reconstitution rebalance passive"], is_query=True)[0]
        hits = index.search(conn, "reconstitution rebalance passive",
                            query_vector=qv, k=2, corpus_label="proj")
        assert hits[0].slug == "index-reconstitution"
        assert hits[0].snippet              # body snippet populated
        assert hits[0].corpus == "proj"     # provenance stamped
        # Keyword-only fallback (daemon-down path) still ranks correctly.
        kw = index.search(conn, "borrow fees", query_vector=None, k=2)
        assert kw and kw[0].slug == "borrow-fees"
    finally:
        conn.close()


def test_kind_is_indexed_and_returned(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "owner-likes-the-hard-route", "values the complicated route",
          "Prefers to learn by doing the harder thing.", kind="identity")
    db = tmp_path / "i.sqlite"
    assert index.build_index(kd, db, FakeEmbedder()) == 1
    conn = index._connect(db, read_only=True)
    try:
        hits = index.search(conn, "complicated route learn", k=1)
        assert hits and hits[0].kind == "identity"
    finally:
        conn.close()


def test_skips_readme_and_malformed(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "good", "a real note", "body text here")
    (kd / "README.md").write_text("# readme\nnot a note\n")
    (kd / "bad.md").write_text("no frontmatter at all\n")
    assert index.build_index(kd, tmp_path / "i.sqlite", FakeEmbedder()) == 1


def test_rebuild_is_full_and_atomic(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "a", "da", "body a")
    db = tmp_path / "i.sqlite"
    index.build_index(kd, db, FakeEmbedder())
    _note(kd, "b", "db", "body b")
    assert index.build_index(kd, db, FakeEmbedder()) == 2
    conn = index._connect(db, read_only=True)
    try:
        slugs = {r[0] for r in conn.execute("SELECT slug FROM notes")}
    finally:
        conn.close()
    assert slugs == {"a", "b"}


def test_build_empty_corpus(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    db = tmp_path / "i.sqlite"
    assert index.build_index(kd, db, FakeEmbedder()) == 0
    conn = index._connect(db, read_only=True)
    try:
        assert index.search(conn, "anything", k=3) == []
    finally:
        conn.close()


# ---- helpers -------------------------------------------------------------

def test_fts_match_builder():
    assert index._fts_match("   !!!  ") is None
    assert index._fts_match("a in to") is None  # all <= 2 chars
    assert index._fts_match("Index Reconstitution") == '"index" OR "reconstitution"'
    assert index._fts_match("alpha alpha beta") == '"alpha" OR "beta"'  # dedup


def test_fts_match_strips_stopwords():
    # function words dropped; content words survive
    assert (index._fts_match("how does the index reconstitution work")
            == '"index" OR "reconstitution" OR "work"')
    assert index._fts_match("what are the") is None  # all stopwords


def test_rrf_rewards_agreement():
    fused = dict(index._rrf([["a", "b"], ["b", "c"]], 3))
    assert max(fused, key=fused.get) == "b"  # appears in both arms
    assert set(fused) == {"a", "b", "c"}


# ---- multi-corpus fusion (project + global) ------------------------------

def test_search_corpora_fuses_with_provenance(tmp_path):
    proj = tmp_path / "proj"; proj.mkdir()
    glob = tmp_path / "glob"; glob.mkdir()
    _note(proj, "g1-overnight", "g1 needs overnight hold",
          "The edge is a passive overnight hold; intraday flattening destroys it.")
    _note(glob, "owner-hard-route", "owner values the complicated route",
          "Prefers learning by doing the harder thing.", kind="identity")
    pdb, gdb = tmp_path / "proj.sqlite", tmp_path / "glob.sqlite"
    emb = FakeEmbedder()
    index.build_index(proj, pdb, emb)
    index.build_index(glob, gdb, emb)
    qv = emb.embed(["overnight hold passive edge"], is_query=True)[0]
    hits = index.search_corpora([("myproject", pdb), ("global", gdb)],
                                "overnight hold passive edge",
                                query_vector=qv, k=5)
    assert hits[0].slug == "g1-overnight" and hits[0].corpus == "myproject"
    assert {h.corpus for h in hits} <= {"myproject", "global"}


def test_search_corpora_skips_missing_index(tmp_path):
    """A brand-new project (no global index yet, or vice versa) still recalls
    from whatever scopes DO exist."""
    proj = tmp_path / "proj"; proj.mkdir()
    _note(proj, "only-note", "the only note", "body content about widgets")
    pdb = tmp_path / "proj.sqlite"
    index.build_index(proj, pdb, FakeEmbedder())
    hits = index.search_corpora([("global", tmp_path / "nope.sqlite"),
                                 ("proj", pdb)], "widgets", k=3)
    assert hits and hits[0].corpus == "proj"


# ---- reranking -----------------------------------------------------------

def test_rerank_hits_reorders_and_preserves_fields():
    hits = [
        index.Hit("a", "da", "sa", 0.10, "proj"),
        index.Hit("b", "db", "sb", 0.20, "proj"),
        index.Hit("c", "dc", "sc", 0.30, "global", "identity"),
    ]
    def scorer(_q, passages):  # prefer 'c' (dc) then 'a' (da) then 'b'
        return [9.0 if "dc" in p else (5.0 if "da" in p else 1.0) for p in passages]
    out = index.rerank_hits(scorer, "q", hits, k=2)
    assert [h.slug for h in out] == ["c", "a"]
    assert out[0].kind == "identity" and out[0].corpus == "global"  # preserved
    assert out[0].score == 9.0  # hit score becomes the rerank score


def test_rerank_hits_degrades_on_bad_scorer():
    hits = [index.Hit("a", "d", "s", 0.5, "proj"),
            index.Hit("b", "d", "s", 0.4, "proj")]
    # length mismatch -> fused order
    assert index.rerank_hits(lambda q, p: [1.0], "q", hits, 5) == hits[:5]
    # raising scorer -> fused order
    def boom(_q, _p):
        raise RuntimeError("model down")
    assert index.rerank_hits(boom, "q", hits, 5) == hits[:5]
    assert index.rerank_hits(lambda q, p: [], "q", [], 5) == []


# ---- wikilink graph + structural/temporal ranking ------------------------

def _dnote(d, slug, desc, body, lu):
    (d / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: {desc}\ntags: [t]\n"
        f"last_updated: {lu}\nsources: [{lu}]\n---\n{body}\n")


def test_wikilinks_build_links_table(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "alpha", "alpha note", "See [[beta]] and [[ghost]] and [[alpha]].")
    _note(kd, "beta", "beta note", "Back to [[alpha]].")
    db = tmp_path / "i.sqlite"
    index.build_index(kd, db, FakeEmbedder())
    conn = index._connect(db, read_only=True)
    try:
        links = {(r[0], r[1])
                 for r in conn.execute("SELECT from_slug, to_slug FROM links")}
    finally:
        conn.close()
    assert ("alpha", "beta") in links and ("beta", "alpha") in links
    assert ("alpha", "ghost") not in links   # dangling target dropped
    assert ("alpha", "alpha") not in links    # self-link dropped


def test_link_expansion_surfaces_linked_neighbor(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    # "alpha" matches the query and links to "beta"; "beta" shares NO query terms
    # so on its own it ranks below the lexical "widget" distractors.
    _note(kd, "alpha", "alpha widget overview", "Alpha widgets; detail in [[beta]].")
    _note(kd, "beta", "sprocket torque spec", "Sprocket internals and torque tables.")
    for i in range(5):
        _note(kd, f"d{i}", f"widget note {i}", f"widget commentary number {i}")
    db = tmp_path / "i.sqlite"; emb = FakeEmbedder()
    index.build_index(kd, db, emb)
    conn = index._connect(db, read_only=True)
    try:
        qv = emb.embed(["alpha widget"], is_query=True)[0]
        wl = [h.slug for h in index.search(conn, "alpha widget", query_vector=qv,
                                           k=3, link_decay=1.0, link_seed=5,
                                           sem_floor=0.0)]
        nl = [h.slug for h in index.search(conn, "alpha widget", query_vector=qv,
                                           k=3, link_decay=0.0)]
        assert "beta" in wl          # injected as alpha's 1-hop neighbor
        assert "beta" not in nl       # on its own terms it misses the top-k
    finally:
        conn.close()


def test_ppr_rank_reaches_multi_hop_by_proximity(tmp_path):
    # Chain a-b-c-d (+ isolated z). Seeding PPR at "a" reaches the multi-hop nodes
    # that flat 1-hop expansion can't, ranked by graph proximity to the seed; an
    # unconnected note never surfaces. Positive control for the PPR implementation.
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "a", "node a", "Links to [[b]].")
    _note(kd, "b", "node b", "Links to [[c]].")
    _note(kd, "c", "node c", "Links to [[d]].")
    _note(kd, "d", "node d", "A leaf node.")
    _note(kd, "z", "node z", "Isolated, links to nothing.")
    db = tmp_path / "i.sqlite"
    index.build_index(kd, db, FakeEmbedder())
    conn = index._connect(db, read_only=True)
    try:
        ranked = index._ppr_rank(conn, [("a", 1.0)])
        assert {"b", "c", "d"} <= set(ranked)          # multi-hop reach
        assert "z" not in ranked                        # unconnected never surfaces
        # closer to the seed ⇒ more PPR mass ⇒ earlier in the ranking
        assert ranked.index("b") < ranked.index("c") < ranked.index("d")
    finally:
        conn.close()


def test_ppr_arm_wired_into_search_lifts_connected_note(tmp_path):
    # "alpha" matches the query and links to "beta"→"gamma" (gamma is 2 hops out,
    # no query terms). "delta" is disconnected and equally query-irrelevant. The
    # PPR arm, wired into search(), gives the graph-connected 2-hop note mass that
    # the disconnected one never gets — so gamma outranks delta. And 1-hop
    # expansion provably can't even reach gamma.
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "alpha", "alpha widget overview", "Alpha widgets; see [[beta]].")
    _note(kd, "beta", "sprocket torque spec", "Sprocket internals; see [[gamma]].")
    _note(kd, "gamma", "flange annealing notes", "Flange annealing temperatures.")
    _note(kd, "delta", "carburetor jet sizing", "Carburetor jet sizing chart.")
    db = tmp_path / "i.sqlite"; emb = FakeEmbedder()
    index.build_index(kd, db, emb)
    conn = index._connect(db, read_only=True)
    try:
        # 1-hop expansion reaches beta but never the 2-hop gamma:
        assert "beta" in index._link_neighbors(conn, ["alpha"])
        assert "gamma" not in index._link_neighbors(conn, ["alpha"])
        qv = emb.embed(["alpha widget"], is_query=True)[0]
        ppr = [h.slug for h in index.search(conn, "alpha widget", query_vector=qv,
                                            k=4, link_decay=0.0, ppr_decay=1.0,
                                            sem_floor=0.0)]
        # the graph-connected 2-hop note outranks the disconnected, equally-
        # irrelevant one — only PPR's spreading activation can do this.
        assert "gamma" in ppr and ppr.index("gamma") < ppr.index("delta")
    finally:
        conn.close()


def test_recency_blend_prefers_fresher(tmp_path):
    from datetime import date
    kd = tmp_path / "k"; kd.mkdir()
    # identical content (tied relevance), different last_updated
    _dnote(kd, "old-take", "the take on widgets", "Widgets explained.", "2026-01-01")
    _dnote(kd, "new-take", "the take on widgets", "Widgets explained.", "2026-06-01")
    db = tmp_path / "i.sqlite"; emb = FakeEmbedder()
    index.build_index(kd, db, emb)
    conn = index._connect(db, read_only=True)
    try:
        qv = emb.embed(["the take on widgets"], is_query=True)[0]
        on = index.search(conn, "the take on widgets", query_vector=qv, k=2,
                          w_recency=1.0, half_life_days=30.0,
                          now=date(2026, 6, 8), link_decay=0.0)
        assert on[0].slug == "new-take"   # recency breaks the tie toward fresher
    finally:
        conn.close()


# ---- dynamic columns (stability / last_used / uses) ----------------------

def _dyn_row(db, slug):
    conn = index._connect(db, read_only=True)
    try:
        return conn.execute(
            "SELECT stability, last_used, uses FROM notes WHERE slug=?",
            (slug,)).fetchone()
    finally:
        conn.close()


def test_build_defaults_dynamic_columns(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "note-a", "alpha desc", "Alpha body.")
    db = tmp_path / "i.sqlite"
    index.build_index(kd, db, FakeEmbedder())
    assert _dyn_row(db, "note-a") == (0.0, "", 0)


def test_build_reads_dynamic_frontmatter(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    (kd / "n.md").write_text("---\nname: n\ndescription: d\nstability: 7.5\n"
                             "last_used: 2026-06-20\nuses: 3\n---\nBody text.\n")
    db = tmp_path / "i.sqlite"
    index.build_index(kd, db, FakeEmbedder())
    assert _dyn_row(db, "n") == (7.5, "2026-06-20", 3)


def test_update_dynamics_syncs_without_reembedding(tmp_path):
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "note-a", "alpha desc", "Alpha body.")
    db = tmp_path / "i.sqlite"
    index.build_index(kd, db, FakeEmbedder())
    assert index.update_dynamics(db, [("note-a", 12.5, "2026-06-24", 4)]) == 1
    assert _dyn_row(db, "note-a") == (12.5, "2026-06-24", 4)


def test_update_dynamics_missing_db_is_noop(tmp_path):
    assert index.update_dynamics(tmp_path / "nope.sqlite",
                                 [("x", 1.0, "2026-01-01", 1)]) == 0


def test_retention_blend_prefers_reinforced_recent(tmp_path):
    from datetime import date
    kd = tmp_path / "k"; kd.mkdir()
    # identical content (tied relevance), different use-history
    (kd / "stale.md").write_text(
        "---\nname: stale\ndescription: the take on widgets\nstability: 2.0\n"
        "last_used: 2026-01-01\n---\nWidgets explained here.\n")
    (kd / "reinforced.md").write_text(
        "---\nname: reinforced\ndescription: the take on widgets\nstability: 60.0\n"
        "last_used: 2026-06-07\n---\nWidgets explained here.\n")
    db = tmp_path / "i.sqlite"; emb = FakeEmbedder()
    index.build_index(kd, db, emb)
    conn = index._connect(db, read_only=True)
    try:
        qv = emb.embed(["the take on widgets"], is_query=True)[0]
        # retention OFF (default): relevance is tied, no use-signal applied
        on = index.search(conn, "the take on widgets", query_vector=qv, k=2,
                          w_retention=1.0, now=date(2026, 6, 8), link_decay=0.0)
        # recently-used + high-stability note wins on the FSRS retrievability term
        assert on[0].slug == "reinforced"
    finally:
        conn.close()


def test_retention_ignored_on_pre_dynamics_index(tmp_path):
    # An index whose notes table predates the dynamic columns must not error when
    # retention is requested — it degrades to no retention term.
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "a", "alpha", "Alpha body here.")
    db = tmp_path / "i.sqlite"
    index.build_index(kd, db, FakeEmbedder())
    conn = index._connect(db)
    try:
        conn.execute("ALTER TABLE notes DROP COLUMN stability")
        conn.execute("ALTER TABLE notes DROP COLUMN last_used")
        conn.commit()
        assert not index._has_dynamic_cols(conn)
        hits = index.search(conn, "alpha", k=2, w_retention=1.0)  # must not raise
        assert hits and hits[0].slug == "a"
    finally:
        conn.close()


# ---- temporal validity: HISTORICAL label, never ranking (Brick 3) ----------

def _vnote(d, slug, desc, body, valid_to=""):
    extra = f"valid_to: {valid_to}\n" if valid_to else ""
    (d / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: {desc}\ntags: [t]\n{extra}---\n{body}\n")


def test_valid_to_threads_through_all_three_hit_factories(tmp_path):
    # THE highest-risk Brick-3 bug: a Hit factory that drops valid_to silently
    # re-presents a reversed fact as current. Pin all three constructors.
    kd = tmp_path / "k"; kd.mkdir()
    _vnote(kd, "old-plan", "the deploy plan", "Deploy plan via GH Pages.",
           valid_to="2020-01-02")
    _vnote(kd, "new-plan", "the new deploy plan", "Deploy via Cloudflare.")
    db = tmp_path / "i.sqlite"
    emb = FakeEmbedder()
    assert index.build_index(kd, db, emb) == 2

    conn = index._connect(db, read_only=True)
    try:
        qv = emb.embed(["deploy plan"], is_query=True)[0]
        hits = index.search(conn, "deploy plan", query_vector=qv, k=2)  # factory 1
    finally:
        conn.close()
    by_slug = {h.slug: h for h in hits}
    assert by_slug["old-plan"].valid_to == "2020-01-02"
    assert by_slug["old-plan"].historical is True
    assert by_slug["new-plan"].valid_to == "" and not by_slug["new-plan"].historical

    fused = index.search_corpora([("proj", db)], "deploy plan",           # factory 2
                                 query_vector=qv, k=2)
    assert {h.slug: h.valid_to for h in fused}["old-plan"] == "2020-01-02"

    rer = index.rerank_hits(lambda q, ps: [0.9] * len(fused), "deploy plan",
                            fused, k=2)                                   # factory 3
    assert {h.slug: h.valid_to for h in rer}["old-plan"] == "2020-01-02"


def test_validity_never_changes_ranking(tmp_path):
    # Identical corpora, one with valid_to stamped — scores must be identical.
    emb = FakeEmbedder()
    scores = {}
    for label, stamp in (("plain", ""), ("stamped", "2020-01-02")):
        kd = tmp_path / f"k-{label}"; kd.mkdir()
        _vnote(kd, "fact-a", "how the cache works", "Prefix cache with 5m TTL.",
               valid_to=stamp)
        _vnote(kd, "fact-b", "how eviction works", "Cooled tail is curated.")
        db = tmp_path / f"{label}.sqlite"
        index.build_index(kd, db, emb)
        conn = index._connect(db, read_only=True)
        try:
            qv = emb.embed(["cache TTL"], is_query=True)[0]
            scores[label] = [(h.slug, round(h.score, 9)) for h in
                             index.search(conn, "cache TTL", query_vector=qv, k=2)]
        finally:
            conn.close()
    assert scores["plain"] == scores["stamped"]   # label-only, zero score impact


def test_pre_validity_index_degrades_to_no_label(tmp_path):
    # A legacy index built before the valid_to column existed: searches must
    # succeed with valid_to="" (no label), never error.
    import sqlite3 as _sq

    import sqlite_vec as _sv
    db = tmp_path / "legacy.sqlite"
    conn = _sq.connect(db)
    conn.enable_load_extension(True); _sv.load(conn); conn.enable_load_extension(False)
    conn.execute("""
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL, body TEXT NOT NULL,
            tags TEXT, sources TEXT, kind TEXT, sha TEXT NOT NULL,
            last_updated TEXT, sources_count INTEGER DEFAULT 0,
            stability REAL DEFAULT 0, last_used TEXT, uses INTEGER DEFAULT 0)""")
    conn.execute("CREATE VIRTUAL TABLE notes_fts USING "
                 "fts5(slug UNINDEXED, description, body, tags)")
    conn.execute("CREATE VIRTUAL TABLE vec_notes USING "
                 "vec0(note_id INTEGER PRIMARY KEY, embedding float[16])")
    conn.execute("CREATE TABLE links (from_slug TEXT NOT NULL, to_slug TEXT NOT NULL)")
    conn.execute("INSERT INTO notes(id,slug,description,body,tags,sources,kind,sha)"
                 " VALUES (1,'legacy-note','an old note','Legacy body.','t','','','x')")
    conn.execute("INSERT INTO notes_fts(rowid,slug,description,body,tags)"
                 " VALUES (1,'legacy-note','an old note','Legacy body.','t')")
    conn.commit(); conn.close()

    conn = index._connect(db, read_only=True)
    try:
        assert index._has_validity_cols(conn) is False
        hits = index.search(conn, "legacy note", query_vector=None, k=1)
    finally:
        conn.close()
    assert hits and hits[0].valid_to == "" and hits[0].historical is False


# ---- DaemonEmbedder (warm-daemon embeddings for index rebuilds) -----------

class _FakeResp:
    def __init__(self, obj):
        import json as _j
        self._b = _j.dumps(obj).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_daemon_embedder_probes_health_and_embeds_passages(monkeypatch):
    import json as _j
    calls = []

    def fake_urlopen(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        calls.append(url)
        if url.endswith("/healthz"):
            return _FakeResp({"ok": True, "dim": 3})
        body = _j.loads(req.data)
        assert body["is_query"] is False        # index rebuilds embed PASSAGES bare
        assert body["texts"] == ["alpha", "beta"]   # batched: one POST, not per-text
        return _FakeResp({"embeddings": [[1.0, 0.0, 0.0]] * len(body["texts"]),
                          "dim": 3})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    from recall.index import DaemonEmbedder
    emb = DaemonEmbedder()
    assert emb.dim == 3                          # dim from /healthz, not hardcoded
    vecs = emb.embed(["alpha", "beta"])
    assert vecs == [[1.0, 0.0, 0.0]] * 2
    assert len(calls) == 2                       # 1 health + 1 batched embed


def test_daemon_embedder_raises_when_daemon_down(monkeypatch):
    # Constructor must raise so _rebuild_indices falls back to the in-process
    # embedder (daemon down == GPU free); a silent empty embedder would write a
    # broken index.
    def dead(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", dead)
    import pytest as _pytest

    from recall.index import DaemonEmbedder
    with _pytest.raises(Exception):
        DaemonEmbedder()


# ---- T2 evidence floor ----------------------------------------------------

def _floor_corpus(tmp_path):
    """Two semantically disjoint notes + an index built with FakeEmbedder."""
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "gpu-daemon-oom", "embedder daemon gpu oom retry",
          "The embedder daemon retries per-text on gpu oom and logs the 500.")
    _note(kd, "kosher-supplier-feeds", "supermarket supplier feeds sync",
          "Supplier feeds sync nightly into the supermarket inventory.")
    db = tmp_path / "i.sqlite"
    emb = FakeEmbedder()
    assert index.build_index(kd, db, emb) == 2
    return db, emb


def test_floor_drops_far_note_and_k_is_a_max(tmp_path):
    db, emb = _floor_corpus(tmp_path)
    conn = index._connect(db, read_only=True)
    try:
        q = "embedder daemon gpu oom retry"
        qv = emb.embed([q], is_query=True)[0]
        # floor off: both notes fill the quota (current behavior)
        both = index.search(conn, q, query_vector=qv, k=5, sem_floor=0.0)
        assert len(both) == 2
        # floor on: the far note dies; k becomes a max, not a quota
        floored = index.search(conn, q, query_vector=qv, k=5,
                               sem_floor=0.35, kw_floor=-999.0)
        assert [h.slug for h in floored] == ["gpu-daemon-oom"]
        assert floored[0].cos >= 0.35          # evidence stamped on the Hit
    finally:
        conn.close()


def test_floor_can_return_zero_hits(tmp_path):
    db, emb = _floor_corpus(tmp_path)
    conn = index._connect(db, read_only=True)
    try:
        qv = emb.embed(["totally unrelated cooking recipe"], is_query=True)[0]
        hits = index.search(conn, "totally unrelated cooking recipe",
                            query_vector=qv, k=5,
                            sem_floor=0.99, kw_floor=-999.0)
        assert hits == []                      # nothing vouched -> inject nothing
    finally:
        conn.close()


def test_floor_keyword_rescue(tmp_path):
    db, emb = _floor_corpus(tmp_path)
    conn = index._connect(db, read_only=True)
    try:
        q = "gpu daemon"
        qv = emb.embed([q], is_query=True)[0]
        # sem floor set impossibly high, but ANY keyword match vouches
        # (kw_floor=0: bm25 is negative for every match) -> FTS-matched note
        # survives via the rescue arm; the unmatched note still dies.
        hits = index.search(conn, q, query_vector=qv, k=5,
                            sem_floor=0.99, kw_floor=0.0)
        assert [h.slug for h in hits] == ["gpu-daemon-oom"]
        assert hits[0].bm25 < 0                # keyword evidence stamped
    finally:
        conn.close()


def test_floor_judges_graph_neighbors_by_own_cosine(tmp_path):
    # A linked neighbor enters via the graph arm with NO direct match evidence;
    # the universal cosine judges it on its own distance to the query.
    kd = tmp_path / "k"; kd.mkdir()
    _note(kd, "gpu-daemon-oom", "embedder daemon gpu oom retry",
          "Daemon retries per-text on gpu oom. See [[kosher-supplier-feeds]].")
    _note(kd, "kosher-supplier-feeds", "supermarket supplier feeds sync",
          "Supplier feeds sync nightly into the supermarket inventory.")
    db = tmp_path / "i.sqlite"
    emb = FakeEmbedder()
    assert index.build_index(kd, db, emb) == 2
    conn = index._connect(db, read_only=True)
    try:
        q = "embedder daemon gpu oom retry"
        qv = emb.embed([q], is_query=True)[0]
        # link arm on, floor off: the neighbor rides in on the graph arm
        linked = index.search(conn, q, query_vector=qv, k=5,
                              link_decay=0.5, link_seed=5, sem_floor=0.0)
        assert {h.slug for h in linked} == {"gpu-daemon-oom",
                                            "kosher-supplier-feeds"}
        # link arm on, floor on: the semantically-far neighbor dies anyway
        floored = index.search(conn, q, query_vector=qv, k=5,
                               link_decay=0.5, link_seed=5,
                               sem_floor=0.35, kw_floor=-999.0)
        assert [h.slug for h in floored] == ["gpu-daemon-oom"]
    finally:
        conn.close()


def test_floor_skipped_in_keyword_only_mode(tmp_path):
    # Daemon-down fallback: no query vector -> the gate must NOT apply
    # (degraded recall must not also go mute).
    db, _emb = _floor_corpus(tmp_path)
    conn = index._connect(db, read_only=True)
    try:
        hits = index.search(conn, "supplier feeds", query_vector=None, k=5,
                            sem_floor=0.99, kw_floor=-999.0)
        assert hits and hits[0].slug == "kosher-supplier-feeds"
    finally:
        conn.close()


def test_floor_flows_through_search_corpora(tmp_path):
    db, emb = _floor_corpus(tmp_path)
    q = "embedder daemon gpu oom retry"
    qv = emb.embed([q], is_query=True)[0]
    scopes = [("proj", db)]
    both = index.search_corpora(scopes, q, query_vector=qv, k=5, sem_floor=0.0)
    assert len(both) == 2
    floored = index.search_corpora(scopes, q, query_vector=qv, k=5,
                                   sem_floor=0.35, kw_floor=-999.0)
    assert [h.slug for h in floored] == ["gpu-daemon-oom"]
    assert floored[0].cos > 0                  # stamps survive cross-scope fusion
