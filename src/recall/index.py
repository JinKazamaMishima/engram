"""Derived hybrid recall index over a ``docs/knowledge``-style markdown corpus.

Keyword (SQLite FTS5 / BM25) + semantic (sqlite-vec KNN) search fused with
Reciprocal Rank Fusion, in one single-file DB per scope. The markdown is the
source of truth; this index is **disposable** — fully rebuilt from the notes
(atomic swap), so it can be deleted and regenerated anytime.

Embeddings come from a local model (``Qwen3-Embedding-0.6B``, bf16 on the GPU,
Matryoshka-truncated to 512 dims) — zero external API at recall. The embedder is
injected so the FTS5/vec/RRF plumbing is unit-tested with a tiny deterministic
fake (no model download).
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
import struct
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import sqlite_vec

from recall import dynamics
from recall.schema import CurationSchemaError, KnowledgeNote, sha256_str


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


RRF_K = int(_env_float("RECALL_RRF_K", 60))   # Reciprocal Rank Fusion constant
DEFAULT_POOL = 50    # candidates pulled per arm before fusion

# --- structural + temporal ranking knobs (env-overridable; per-call override too) ---
# Wikilink 1-hop expansion pulls neighbors of the top-LINK_SEED fused hits in as
# extra candidates at LINK_DECAY × the top score (so a linked note holding the
# real answer can surface + be reranked). OFF by default (0.0): on the 42-case eval
# it LOST recall — R@1 0.881→0.857, R@k 0.893→0.869, nDCG 0.847→0.822, MRR 0.897→0.865
# — by over-promoting well-connected hub notes (independently confirmed in the soul
# corpus, where it demoted the correct top hit). PPR (below) is the successor to try.
# The recency+salience+retention blend (Generative-Agents style) is likewise OFF
# (weights 0) until the eval earns it.
LINK_DECAY = _env_float("RECALL_LINK_DECAY", 0.0)
LINK_SEED = int(_env_float("RECALL_LINK_SEED", 10))
W_RECENCY = _env_float("RECALL_W_RECENCY", 0.0)
W_SALIENCE = _env_float("RECALL_W_SALIENCE", 0.0)
# Retention = FSRS retrievability R(t,S) keyed on *use* (last_used + stability),
# the principled successor to the write-time recency term: a note used recently
# and reinforced often decays slowly and ranks up; a stale one fades. OFF by
# default — a ranking change must be earned on the eval, like recency/salience.
W_RETENTION = _env_float("RECALL_W_RETENTION", 0.0)
HALF_LIFE_DAYS = _env_float("RECALL_HALF_LIFE_DAYS", 180.0)
SALIENCE_CAP = _env_float("RECALL_SALIENCE_CAP", 8.0)
# Personalized-PageRank graph arm (HippoRAG-style spreading activation over the
# [[wikilink]] graph): the principled successor to the flat 1-hop expansion — it
# reaches MULTI-hop neighbors and weights every node by graph proximity to the
# query's seed hits (restart-seeded by the fused kw+dense scores), instead of
# dumping in every 1-hop neighbor at one flat weight. A down-weighted RRF arm,
# like the link arm. OFF by default (decay 0) until the eval earns it, exactly
# like recency/retention; when on, it is meant to REPLACE the link arm (set
# RECALL_LINK_DECAY=0). Tiny graphs → cheap dense power iteration, no new deps.
PPR_DECAY = _env_float("RECALL_PPR_DECAY", 0.0)
PPR_ALPHA = _env_float("RECALL_PPR_ALPHA", 0.15)   # teleport/restart probability
PPR_ITERS = int(_env_float("RECALL_PPR_ITERS", 40))
PPR_SEED = int(_env_float("RECALL_PPR_SEED", 10))  # restart only from the top-N hits
# --- T2 evidence floor: relevance gate on ABSOLUTE arm evidence ---------------
# RRF + min-max normalized scores are rank-RELATIVE (the top hit is 1.0 by
# construction, junk or not), so the floor judges candidates on absolute
# evidence instead: true cos(query, note), computed for EVERY fused candidate —
# whatever arm surfaced it, so graph-arm neighbors are judged fairly. k becomes
# a MAX, not a quota: weak candidates drop, 0..k notes inject. Hybrid-path
# only: the keyword-only fallback (daemon down) stays unfiltered — degraded
# recall must not also go mute. SEM_FLOOR=0 disables the gate entirely.
# 0.40 is DOUBLY eval-earned (2026-07-08): (a) blind set (floor_sweep.json) —
# positives 82.7%→84.0% (junk dropping lets targets into top-5), junk notes on
# negatives −30%; (b) a 96-real-prompt stress replay (floor_stress_replay.py) —
# at 0.45 the 0.41–0.45 band killed ~15 LOAD-BEARING notes (brick3 on "how does
# context work now?", warm-daemon on "check systems after restart", pr-gate on
# "commit and mirror"), at 0.40 zero flagged casualties while conversational
# turns ("commit please") and off-domain chat correctly mute to zero injection.
# 0.50+ is the blind-set cliff (situational −3, oblique −5). The 0.40–0.45 band
# is where junk and gold OVERLAP: prefer the conservative edge, leave the tail
# to PPR/T3. Retune BOTH evals after any embedder swap.
SEM_FLOOR = _env_float("RECALL_SEM_FLOOR", 0.40)
# Keyword-rescue floor (bm25 ≤ KW_FLOOR vouches a candidate past the sem floor).
# OFF by default and experimental: FTS5 bm25 is ADDITIVE over OR-ed terms, so a
# many-token junk query out-magnitudes the exact-identifier match this was meant
# to rescue (blind negatives ran −4.4..−21.6 — no separating threshold exists).
# Hit.bm25 still carries the evidence for observability.
KW_FLOOR = _env_float("RECALL_KW_FLOOR", -1e9)
_WIKILINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9-]*)\]\]")

# Function words stripped from FTS5 MATCH queries — they carry no retrieval
# signal and only add noise to the OR-ed keyword arm. Tokens ≤2 chars are
# already dropped, so only the longer function words need listing here.
_STOPWORDS = frozenset({
    "the", "and", "but", "for", "are", "was", "were", "been", "being",
    "does", "did", "done", "has", "have", "had", "how", "what", "when",
    "where", "why", "who", "whom", "which", "that", "this", "these", "those",
    "with", "without", "from", "into", "onto", "over", "under", "about",
    "its", "your", "you", "our", "they", "their", "them", "his", "her",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "not", "yes", "than", "too", "very", "just", "any", "all", "out",
})


# ---- embedding -----------------------------------------------------------

# Dimension of the sqlite-vec embedding column, and the fallback the keyword-only
# build uses to create a valid (empty) vec table when no ML stack is installed —
# a later rebuild WITH models then needs no schema change.
EMBED_DIM = 512


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """``Qwen3-Embedding-0.6B`` truncated to 512 dims (Matryoshka), normalized, on
    CUDA if available. Qwen3 applies a retrieval *instruction* on the query side
    (its built-in ``query`` prompt); passages are embedded bare. Lazy import so
    importing this module stays cheap."""

    MODEL = "Qwen/Qwen3-Embedding-0.6B"
    TRUNCATE_DIM = EMBED_DIM  # Matryoshka 1024 -> 512: keeps the sqlite-vec index lean
    QUERY_PROMPT = "query"    # the model's built-in query-instruction prompt

    def __init__(self, model_name: str = MODEL, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # bf16 on GPU halves VRAM (the 0.6B embedder + 0.6B reranker must share a
        # contended ~10GB card) and is ~lossless for normalized retrieval vectors.
        model_kwargs = {"torch_dtype": torch.bfloat16} if dev == "cuda" else {}
        self._model = SentenceTransformer(
            model_name, truncate_dim=self.TRUNCATE_DIM, device=dev,
            model_kwargs=model_kwargs)
        get_dim = (getattr(self._model, "get_embedding_dimension", None)
                   or self._model.get_sentence_embedding_dimension)
        self.dim = get_dim()

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        # Query side gets Qwen3's instruction prompt; passages are embedded bare.
        # The model defaults its 'query' prompt to ALL encodes, so passages must
        # explicitly pass prompt="" to suppress it.
        kw = {"prompt_name": self.QUERY_PROMPT} if is_query else {"prompt": ""}
        vecs = self._model.encode(texts, normalize_embeddings=True,
                                  convert_to_numpy=True, batch_size=16, **kw)
        return [v.tolist() for v in vecs]


class DaemonEmbedder:
    """Embeddings served by the warm localhost daemon
    (``scripts/recall_embedder_daemon.py``) instead of a fresh in-process model.
    The daemon is the machine's SINGLE embedding authority: loading a second
    0.6B model beside it is the CUDA OOM that silently froze the index for two
    days (2026-07-04..06) and starved it again 2026-07-07. Duck-types
    ``SentenceTransformerEmbedder`` (.dim + .embed); posts texts in batches of
    ``BATCH`` per request. Raises at construction when the daemon is down OR
    degraded (healthz probes real inference + device now) — callers fall back
    to the in-process embedder (the GPU is free then)."""

    BATCH = 8         # texts per POST — the daemon encodes a request in one padded
                      # batch, so long notes spike VRAM with the batch size; 8 keeps
                      # the spike modest (the daemon also retries per-text on OOM)
    TIMEOUT = 120.0   # generous: a contended GPU should yield a LATE answer, not a
                      # timeout that strands a half-built index (2026-07-07)

    def __init__(self, host: str | None = None, port: str | int | None = None,
                 timeout: float | None = None):
        import json
        import urllib.request
        self._base = "http://{}:{}".format(
            host or os.environ.get("RECALL_EMBED_HOST", "127.0.0.1"),
            port or os.environ.get("RECALL_EMBED_PORT", "8973"))
        self._timeout = float(timeout if timeout is not None else
                              os.environ.get("RECALL_EMBED_TIMEOUT", self.TIMEOUT))
        import urllib.error
        try:
            with urllib.request.urlopen(f"{self._base}/healthz", timeout=5.0) as r:
                health = json.loads(r.read())
        except urllib.error.HTTPError as e:
            # A degraded daemon answers 503 WITH a diagnostic body (device drift,
            # probe failure) — surface it instead of a bare HTTP error, so callers
            # can tell degraded (daemon up, needs a restart) from down.
            try:
                health = json.loads(e.read())
            except Exception:  # noqa: BLE001 — non-JSON error body
                health = {"ok": False, "error": f"HTTP {e.code}"}
        if not health.get("ok") or not health.get("dim"):
            raise RuntimeError(f"embedder daemon unhealthy: {health}")
        self.dim = int(health["dim"])

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        import json
        import urllib.error
        import urllib.request
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            chunk = list(texts[i:i + self.BATCH])
            req = urllib.request.Request(
                f"{self._base}/embed",
                data=json.dumps({"texts": chunk, "is_query": is_query}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    out.extend(json.loads(r.read())["embeddings"])
            except urllib.error.HTTPError as e:
                # Surface the daemon's diagnostic body — a bare "HTTP 500" hid
                # the real cause (CUDA OOM) the first time this path fired.
                try:
                    detail = json.loads(e.read()).get("error", "")
                except Exception:  # noqa: BLE001 — non-JSON error body
                    detail = ""
                raise RuntimeError(
                    f"daemon /embed failed (HTTP {e.code}): {detail}") from e
        return out


def best_embedder(*, alert_degraded: bool = False
                  ) -> DaemonEmbedder | SentenceTransformerEmbedder:
    """Daemon-first embedder factory — the ONE way pipeline code should obtain an
    embedder. The warm daemon is the machine's single embedding authority; every
    in-process load beside it risks the OOM that took the index down twice
    (2026-07-04..06 silent, 2026-07-07 starved). In-process only when the daemon
    is unavailable — which is exactly when the GPU is free.

    ``alert_degraded=True``: scream on the alert channel when the daemon is UP
    but unhealthy (e.g. serving off its warm device) — unattended callers (the
    nightly cycle) want that page; interactive CLI calls don't.
    ``RECALL_NO_DAEMON=1`` skips the daemon entirely (hermetic tests must never
    talk to a live daemon). Raises ImportError when neither the daemon nor the
    local ML stack exists — callers keep their keyword-only / best-effort paths."""
    if not os.environ.get("RECALL_NO_DAEMON"):
        try:
            return DaemonEmbedder()
        except Exception as e:  # noqa: BLE001 — any daemon trouble -> in-process
            if alert_degraded and "daemon unhealthy" in str(e):
                try:
                    from recall.notify import notify_alert
                    notify_alert(
                        "recall embedder DEGRADED",
                        f"{e}\nDaemon is up but failed its healthz inference/device "
                        "probe — restart: systemctl --user restart recall-embedder",
                        priority="urgent")
                except Exception:  # noqa: BLE001 — alerting is best-effort
                    pass
            print(f"[recall] embedder daemon unavailable ({e}) — in-process model",
                  file=sys.stderr)
    return SentenceTransformerEmbedder()


class CrossEncoderReranker:
    """``Qwen3-Reranker-0.6B`` — a long-context (32K) cross-encoder that scores
    (query, passage) pairs jointly, far more precise than bi-encoder cosine for
    the final ordering and (unlike bge-reranker-base's 512-token window) able to
    read a full note body. Optional second stage over the fused top-N. Lazy
    import so this module stays cheap; the daemon keeps it warm."""

    MODEL = "Qwen/Qwen3-Reranker-0.6B"

    def __init__(self, model_name: str = MODEL, device: str | None = None):
        import torch
        from sentence_transformers import CrossEncoder
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = {"torch_dtype": torch.bfloat16} if dev == "cuda" else {}
        self._model = CrossEncoder(model_name, device=dev,
                                   model_kwargs=model_kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict([(query, p) for p in passages],
                                     batch_size=16)
        return [float(s) for s in scores]


# ---- result type ---------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    slug: str
    description: str
    snippet: str
    score: float
    corpus: str = ""   # provenance label set by the caller (e.g. "myproject", "global")
    kind: str = ""     # note kind (e.g. "identity"/"achievement"); "" for domain notes
    body: str = ""     # full note body — fed to the reranker (snippet is display-only)
    valid_to: str = ""  # ISO date the fact STOPPED being true; "" = still true
    cos: float = 0.0   # absolute semantic evidence, cos(query, note); 0.0 = unknown
    bm25: float = 0.0  # FTS5 bm25 of the keyword match (negative = better); 0.0 = no match

    @property
    def historical(self) -> bool:
        """True when this fact has an expired validity window — rendered as a
        ⏳ HISTORICAL label at injection. NEVER a ranking input: a reversed
        decision must still surface (so the model knows the history), just not
        read as current."""
        return bool(self.valid_to) and self.valid_to < date.today().isoformat()


# ---- DB helpers ----------------------------------------------------------

def _connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _create_schema(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute("""
        CREATE TABLE notes (
            id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL, body TEXT NOT NULL,
            tags TEXT, sources TEXT, kind TEXT, sha TEXT NOT NULL,
            last_updated TEXT, sources_count INTEGER DEFAULT 0,
            stability REAL DEFAULT 0, last_used TEXT, uses INTEGER DEFAULT 0,
            valid_to TEXT DEFAULT '')""")
    conn.execute("CREATE VIRTUAL TABLE notes_fts USING "
                 "fts5(slug UNINDEXED, description, body, tags)")
    conn.execute(f"CREATE VIRTUAL TABLE vec_notes USING "
                 f"vec0(note_id INTEGER PRIMARY KEY, embedding float[{dim}])")
    # Derived [[wikilink]] graph (filtered to existing slugs at build time) — the
    # fuel for 1-hop neighbor expansion at recall.
    conn.execute("CREATE TABLE links (from_slug TEXT NOT NULL, "
                 "to_slug TEXT NOT NULL)")
    conn.execute("CREATE INDEX idx_links_from ON links(from_slug)")
    conn.execute("CREATE INDEX idx_links_to ON links(to_slug)")


# ---- build ---------------------------------------------------------------

def _extract_links(body: str, known: set[str], *, exclude: str = "") -> list[str]:
    """``[[slug]]`` targets in a note body that actually exist in the corpus
    (dangling + self links dropped), de-duplicated in first-seen order."""
    out: list[str] = []
    for tgt in dict.fromkeys(_WIKILINK_RE.findall(body)):
        if tgt in known and tgt != exclude:
            out.append(tgt)
    return out


def _load_notes(knowledge_dir: Path) -> list[tuple[KnowledgeNote, str]]:
    """Parse every note (skipping README and anything malformed) → (note, sha)."""
    out: list[tuple[KnowledgeNote, str]] = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text()
        try:
            note = KnowledgeNote.parse(text, expect_slug=path.stem)
        except CurationSchemaError as e:
            print(f"[index] skip malformed note {path.name}: {e}",
                  file=sys.stderr)
            continue
        out.append((note, sha256_str(text)))
    return out


def build_index(knowledge_dir: Path, db_path: Path,
                embedder: Embedder | None) -> int:
    """Full rebuild into a temp DB, then atomic-swap into place. Returns the
    number of notes indexed. ``embedder=None`` builds a keyword-only index (FTS5
    + link graph + an empty vec table) — for machines without the local ML stack;
    semantic search then needs a later rebuild WITH an embedder."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".building")
    if tmp.exists():
        tmp.unlink()

    notes = _load_notes(knowledge_dir)
    conn = _connect(tmp)
    try:
        _create_schema(conn, embedder.dim if embedder is not None else EMBED_DIM)
        if notes:
            known = {n.slug for n, _ in notes}
            vecs = (embedder.embed([f"{n.description}\n\n{n.body}"
                                    for n, _ in notes])
                    if embedder is not None else [None] * len(notes))
            for i, ((note, sha), vec) in enumerate(zip(notes, vecs), start=1):
                tags = " ".join(note.tags)
                conn.execute(
                    "INSERT INTO notes(id,slug,description,body,tags,sources,kind,"
                    "sha,last_updated,sources_count,stability,last_used,uses,"
                    "valid_to) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, note.slug, note.description, note.body, tags,
                     " ".join(note.sources), note.kind, sha,
                     note.last_updated, len(note.sources),
                     note.stability, note.last_used, note.uses, note.valid_to))
                conn.execute(
                    "INSERT INTO notes_fts(rowid,slug,description,body,tags)"
                    " VALUES (?,?,?,?,?)",
                    (i, note.slug, note.description, note.body, tags))
                if vec is not None:
                    conn.execute(
                        "INSERT INTO vec_notes(note_id,embedding) VALUES (?,?)",
                        (i, sqlite_vec.serialize_float32(vec)))
                for tgt in _extract_links(note.body, known, exclude=note.slug):
                    conn.execute("INSERT INTO links(from_slug,to_slug) "
                                 "VALUES (?,?)", (note.slug, tgt))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, db_path)
    return len(notes)


def update_dynamics(db_path: Path, rows: list[tuple[str, float, str, int]]) -> int:
    """Cheap in-place sync of the dynamic columns (``stability``/``last_used``/
    ``uses``) into an existing index, WITHOUT re-embedding — the consolidate fold
    changes only these scalars, never note text/vectors, so the GPU never spins.
    ``rows`` is ``[(slug, stability, last_used, uses), …]``. Best-effort: a missing
    DB, or a pre-dynamics index whose columns don't exist yet, degrades to a no-op
    (the next full ``recall build`` picks the values up from frontmatter). Returns
    the number of rows updated."""
    db_path = Path(db_path)
    if not db_path.exists() or not rows:
        return 0
    conn = _connect(db_path)
    try:
        n = 0
        for slug, stability, last_used, uses in rows:
            try:
                cur = conn.execute(
                    "UPDATE notes SET stability=?, last_used=?, uses=? WHERE slug=?",
                    (float(stability), last_used or "", int(uses), slug))
            except sqlite3.Error:
                return n  # pre-dynamics index (no such column); next build fixes it
            n += cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()


# ---- search --------------------------------------------------------------

def _fts_match(text: str) -> str | None:
    """Safe FTS5 MATCH expression: alnum tokens >2 chars, minus stopwords, OR-ed
    and quoted. Stopword removal keeps the OR-query on content words; stemming was
    tried (porter) and dropped — it regressed the eval, the dense arm already
    covers morphology."""
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 2 and t not in _STOPWORDS]
    if not toks:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks[:24]))


def _fts_ranked(conn: sqlite3.Connection, text: str,
                pool: int) -> list[tuple[str, float]]:
    """Keyword arm: (slug, bm25) best-first. The absolute bm25 (negative =
    better) rides along as floor evidence instead of being discarded."""
    match = _fts_match(text)
    if not match:
        return []
    rows = conn.execute(
        "SELECT slug, bm25(notes_fts) FROM notes_fts WHERE notes_fts MATCH ? "
        "ORDER BY bm25(notes_fts) LIMIT ?", (match, pool)).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _vec_ranked(conn: sqlite3.Connection, query_vec: list[float],
                pool: int) -> list[str]:
    try:
        rows = conn.execute(
            "SELECT n.slug FROM vec_notes v JOIN notes n ON n.id = v.note_id "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (sqlite_vec.serialize_float32(query_vec), pool)).fetchall()
    except sqlite3.Error:
        # Dimension mismatch (e.g. a stale daemon serving the old embedding size
        # after a model swap) -> degrade to keyword-only, never fail the query.
        return []
    return [r[0] for r in rows]


def _semantic_evidence(conn: sqlite3.Connection, query_vector: list[float],
                       slugs: list[str]) -> dict[str, float]:
    """True cos(query, note) for every candidate slug, straight from the stored
    embeddings — one batched fetch, no model call. This is the floor's absolute
    evidence: it covers candidates from ANY arm (keyword, vec, link, PPR), so a
    graph neighbor is judged by its real distance to the query, not by the arm
    that surfaced it. Missing embeddings (keyword-only index rows) are absent
    from the result — the caller treats absent as not-vouched. Fail-open: any
    error returns {} and the caller must then skip flooring, never the query."""
    if not slugs:
        return {}
    try:
        qn = math.sqrt(sum(x * x for x in query_vector)) or 1.0
        ph = ",".join("?" * len(slugs))
        rows = conn.execute(
            f"SELECT n.slug, v.embedding FROM vec_notes v "
            f"JOIN notes n ON n.id = v.note_id WHERE n.slug IN ({ph})",
            slugs).fetchall()
        out: dict[str, float] = {}
        for slug, blob in rows:
            vec = struct.unpack(f"<{len(blob) // 4}f", blob)
            if len(vec) != len(query_vector):
                return {}
            dot = sum(a * b for a, b in zip(query_vector, vec))
            vn = math.sqrt(sum(x * x for x in vec)) or 1.0
            out[slug] = dot / (qn * vn)
        return out
    except (sqlite3.Error, struct.error):
        return {}


def _rrf(ranked_lists: list[list[str]], k_out: int, *,
         rrf_k: int = RRF_K,
         weights: list[float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Smaller ``rrf_k`` gives the top ranks more pull
    (60 is the classic default; for short fused lists ~20-40 is often better).
    ``weights`` (per ranked-list) lets one arm count more than another."""
    scores: dict[str, float] = {}
    for i, lst in enumerate(ranked_lists):
        w = weights[i] if weights is not None and i < len(weights) else 1.0
        for rank, slug in enumerate(lst):
            scores[slug] = scores.get(slug, 0.0) + w / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k_out]


def _link_neighbors(conn: sqlite3.Connection, slugs: list[str]) -> list[str]:
    """1-hop neighbors (either direction) of ``slugs`` from the links table.
    Guarded: an index built before the links table existed degrades to no
    expansion rather than erroring the query."""
    if not slugs:
        return []
    qs = ",".join("?" * len(slugs))
    try:
        rows = conn.execute(
            f"SELECT to_slug FROM links WHERE from_slug IN ({qs}) "
            f"UNION SELECT from_slug FROM links WHERE to_slug IN ({qs})",
            (*slugs, *slugs)).fetchall()
    except sqlite3.Error:
        return []
    return [r[0] for r in rows]


def _ppr_rank(conn: sqlite3.Connection, seed_scores: list[tuple[str, float]], *,
              alpha: float = PPR_ALPHA, iters: int = PPR_ITERS,
              seed_top: int = PPR_SEED, pool: int = DEFAULT_POOL) -> list[str]:
    """Personalized PageRank over the (symmetric) ``[[wikilink]]`` graph, the
    restart distribution seeded by the TOP ``seed_top`` fused kw+dense hits
    weighted by their scores. Returns slugs ranked by stationary mass — multi-hop
    reachable notes surface (unlike 1-hop expansion) and are weighted by graph
    proximity to the seeds. Best-effort: no edges / no model / a degenerate seed →
    ``[]`` (the caller just gets no graph arm). Tiny corpora → a dense power
    iteration is trivially cheap, so no ANN/graph dependency is pulled in.

    Restarting from only the top query-relevant hits is what makes this a
    *personalized* PageRank (HippoRAG-style): seeding from the whole candidate
    pool dilutes the personalization and collapses toward generic centrality.

    Links are treated as UNDIRECTED (an edge means the two notes are associated
    either way — same symmetry the 1-hop ``_link_neighbors`` UNION already uses)."""
    seed_scores = [(s, sc) for s, sc in seed_scores[:max(1, seed_top)]]
    if not seed_scores:
        return []
    try:
        rows = conn.execute("SELECT from_slug, to_slug FROM links").fetchall()
    except sqlite3.Error:
        return []
    edges = [(a, b) for a, b in rows if a and b and a != b]
    seeds = [s for s, _ in seed_scores]
    nodes = list(dict.fromkeys([n for e in edges for n in e] + seeds))
    if len(nodes) < 2:
        return []
    try:
        import numpy as np
    except ImportError:
        return []
    idx = {s: i for i, s in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype="float32")
    for a, b in edges:
        i, j = idx[a], idx[b]
        adj[i, j] = adj[j, i] = 1.0          # symmetric
    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0                       # isolated seed nodes: keep restart mass
    walk = adj / deg                          # row-stochastic transition
    restart = np.zeros(n, dtype="float32")
    for s, sc in seed_scores:
        if s in idx:
            restart[idx[s]] = max(0.0, float(sc))
    total = restart.sum()
    if total <= 0:
        return []
    restart /= total
    p = restart.copy()
    for _ in range(max(1, iters)):
        nxt = alpha * restart + (1.0 - alpha) * (walk.T @ p)
        if float(np.abs(nxt - p).sum()) < 1e-6:
            p = nxt
            break
        p = nxt
    order = np.argsort(-p)
    return [nodes[i] for i in order[:pool] if p[i] > 0.0]


def _recency(last_updated: str, today: date, half_life_days: float) -> float:
    """Exponential-decay recency in [0,1]: ~1.0 today, 0.5 at one half-life. A
    blank or unparseable date -> 0.0 (no boost, no penalty)."""
    if not last_updated or half_life_days <= 0:
        return 0.0
    try:
        d = date.fromisoformat(last_updated[:10])
    except ValueError:
        return 0.0
    age = max(0, (today - d).days)
    return 0.5 ** (age / half_life_days)


def _salience(sources_count: int) -> float:
    """Reinforcement salience in [0,1]: log-scaled count of contributing days,
    capped at ``SALIENCE_CAP``."""
    if SALIENCE_CAP <= 0:
        return 0.0
    return min(1.0, math.log1p(max(0, sources_count)) / math.log1p(SALIENCE_CAP))


def _retention(stability: float, last_used: str, last_updated: str,
               today: date) -> float:
    """FSRS retrievability R(t,S) in [0,1], keyed on *use*: age since the note's
    last activation (``last_used``, falling back to the last content edit), and
    its accumulated ``stability``. A note never consolidated (stability ≤ 0) or
    with no usable date contributes 0 — no boost, no penalty — so only notes that
    have actually been used carry a retention signal. A graduated note is floored
    by ``dynamics.effective_retrievability`` so it never fully fades."""
    if stability <= 0:
        return 0.0
    ref = last_used or last_updated
    if not ref:
        return 0.0
    try:
        d = date.fromisoformat(ref[:10])
    except ValueError:
        return 0.0
    age = max(0, (today - d).days)
    return dynamics.effective_retrievability(age, stability)


def _has_dynamic_cols(conn: sqlite3.Connection) -> bool:
    """Whether this index carries the Phase-I dynamic columns. A pre-dynamics
    index (built before they existed) degrades to no retention term rather than
    erroring the query — it picks them up on the next full ``recall build``."""
    try:
        have = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
    except sqlite3.Error:
        return False
    return {"stability", "last_used"} <= have


def _has_validity_cols(conn: sqlite3.Connection) -> bool:
    """Whether this index carries the temporal-validity column (Brick 3). A
    pre-validity index degrades to no HISTORICAL label rather than erroring —
    it picks the column up on the next full rebuild."""
    try:
        have = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
    except sqlite3.Error:
        return False
    return "valid_to" in have


def search(conn: sqlite3.Connection, query_text: str, *,
           query_vector: list[float] | None = None, k: int = 5,
           pool: int = DEFAULT_POOL, corpus_label: str = "",
           rrf_k: int = RRF_K, arm_weights: tuple[float, float] | None = None,
           link_decay: float = LINK_DECAY, link_seed: int = LINK_SEED,
           ppr_decay: float = PPR_DECAY,
           w_recency: float = W_RECENCY, w_salience: float = W_SALIENCE,
           w_retention: float = W_RETENTION, half_life_days: float = HALF_LIFE_DAYS,
           sem_floor: float = SEM_FLOOR, kw_floor: float = KW_FLOOR,
           now: date | None = None) -> list[Hit]:
    """Hybrid recall over ONE index. Keyword (FTS5) always; semantic (vec) added
    when a ``query_vector`` is supplied; fused with RRF. With no vector it
    degrades to keyword-only — the recall hook's fallback when the daemon is down.

    T2 evidence floor (hybrid path only): after fusion, a candidate survives iff
    its ABSOLUTE evidence vouches — cos(query, note) ≥ ``sem_floor`` or a strong
    keyword match (bm25 ≤ ``kw_floor``). Applied before the top-k slice, so k is
    a MAX, not a quota (0..k hits return). ``sem_floor=0`` disables the gate.

    Then: (1) ``[[wikilink]]`` 1-hop expansion injects neighbors of the top
    ``link_seed`` fused hits at ``link_decay`` × the top score; (2) relevance is
    min-max normalized to [0,1] and blended with recency + salience + retention
    (Generative-Agents style; all OFF by default). Retention is FSRS R(t,S) keyed
    on use (``last_used`` + ``stability``) — the principled successor to recency.
    ``corpus_label`` stamps provenance; ``now`` is injectable for tests."""
    fts = _fts_ranked(conn, query_text, pool)
    bm: dict[str, float] = dict(fts)
    arms: list[list[str]] = [[s for s, _ in fts]]
    arm_w = [arm_weights[0] if arm_weights else 1.0]
    if query_vector is not None:
        arms.append(_vec_ranked(conn, query_vector, pool))
        arm_w.append(arm_weights[1] if arm_weights else 1.0)
    # Graph expansion as down-weighted RRF arm(s), seeded from the keyword+vec
    # fusion so a linked note holding the answer competes alongside the lexical/
    # semantic hits (RRF additivity lifts it, where a flat score injection got
    # capped below RRF's compressed top scores). Two flavors:
    #   • 1-hop link expansion — flat: every neighbor of the top seeds at one weight.
    #   • PPR spreading-activation — multi-hop, weighted by graph proximity to the
    #     seeds; the principled successor (run it with link_decay=0 to replace).
    # Hybrid-only: in the keyword-only fallback (daemon down) thin lexical scores
    # let a graph arm over-promote neighbors and crater precision, so both skip there.
    if query_vector is not None and (
            (link_decay > 0 and link_seed > 0) or ppr_decay > 0):
        seed_fused = _rrf(arms, pool, rrf_k=rrf_k, weights=arm_w)
        if link_decay > 0 and link_seed > 0:
            seeds = [s for s, _ in seed_fused[:link_seed]]
            nbrs = list(dict.fromkeys(_link_neighbors(conn, seeds)))
            if nbrs:
                arms.append(nbrs)
                arm_w.append(link_decay)
        if ppr_decay > 0:
            ppr_ranked = _ppr_rank(conn, seed_fused, pool=pool)
            if ppr_ranked:
                arms.append(ppr_ranked)
                arm_w.append(ppr_decay)
    fused = _rrf(arms, pool, rrf_k=rrf_k, weights=arm_w)
    if not fused:
        return []
    # Universal semantic evidence for every fused candidate (any arm) — one
    # batched fetch; also stamped onto Hits (observability + the stress replay).
    cos: dict[str, float] = {}
    if query_vector is not None:
        cos = _semantic_evidence(conn, query_vector, [s for s, _ in fused])
        if sem_floor > 0 and cos:  # fail-open: no evidence map -> no flooring
            fused = [(s, sc) for s, sc in fused
                     if cos.get(s, -1.0) >= sem_floor
                     or (s in bm and bm[s] <= kw_floor)]  # absent = never vouches
            if not fused:
                return []
    base: dict[str, float] = dict(fused)
    mn = min(base.values())
    span = (max(base.values()) - mn) or 1.0
    blend = w_recency > 0 or w_salience > 0 or w_retention > 0
    dyn = w_retention > 0 and _has_dynamic_cols(conn)
    validity = _has_validity_cols(conn)
    cols = "description, body, kind, last_updated, sources_count"
    if dyn:
        cols += ", stability, last_used"
    if validity:
        # Render-time label ONLY — valid_to never enters the score math.
        cols += ", valid_to"
    today = now or date.today()
    scored: list[tuple[float, str, str, str, str, str]] = []
    for slug, rel_raw in base.items():
        row = conn.execute(
            f"SELECT {cols} FROM notes WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            continue
        desc, body, kind, last_updated, scount = row[:5]
        stability, last_used = (row[5], row[6]) if dyn else (0.0, "")
        valid_to = (row[-1] or "") if validity else ""
        final = (rel_raw - mn) / span
        if blend:
            final += (w_recency * _recency(last_updated or "", today, half_life_days)
                      + w_salience * _salience(scount or 0)
                      + w_retention * _retention(stability or 0.0, last_used or "",
                                                 last_updated or "", today))
        scored.append((final, slug, desc, body or "", kind or "", valid_to))
    scored.sort(key=lambda t: t[0], reverse=True)
    hits: list[Hit] = []
    for final, slug, desc, body, kind, valid_to in scored[:k]:
        full = body.strip()
        snippet = full.replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "…"
        hits.append(Hit(slug=slug, description=desc, snippet=snippet,
                        score=final, corpus=corpus_label, kind=kind, body=full,
                        valid_to=valid_to,
                        cos=cos.get(slug, 0.0), bm25=bm.get(slug, 0.0)))
    return hits


def search_corpora(scopes: list[tuple[str, Path]], query_text: str, *,
                   query_vector: list[float] | None = None, k: int = 5,
                   pool: int = DEFAULT_POOL, rrf_k: int = RRF_K,
                   arm_weights: tuple[float, float] | None = None,
                   link_decay: float = LINK_DECAY, link_seed: int = LINK_SEED,
                   ppr_decay: float = PPR_DECAY,
                   w_recency: float = W_RECENCY, w_salience: float = W_SALIENCE,
                   w_retention: float = W_RETENTION,
                   half_life_days: float = HALF_LIFE_DAYS,
                   sem_floor: float = SEM_FLOOR, kw_floor: float = KW_FLOOR,
                   now: date | None = None) -> list[Hit]:
    """Fused hybrid recall across MULTIPLE indices — e.g. a project corpus plus
    the shared global/"soul" corpus. Each scope is ``(label, db_path)``; a
    missing index is skipped (so a brand-new project still recalls from global).
    Per-scope hybrid hits are fused across scopes with a second RRF over their
    within-scope rank, so a note strong in its own corpus competes fairly.
    Provenance (``Hit.corpus``) is preserved and the same slug in two scopes
    stays distinct (keyed on ``(label, slug)``)."""
    per_scope: dict[tuple[str, str], Hit] = {}
    ranked_lists: list[list[tuple[str, str]]] = []
    for label, db in scopes:
        if not Path(db).exists():
            continue
        conn = _connect(Path(db), read_only=True)
        try:
            hits = search(conn, query_text, query_vector=query_vector,
                          k=pool, pool=pool, corpus_label=label,
                          rrf_k=rrf_k, arm_weights=arm_weights,
                          link_decay=link_decay, link_seed=link_seed,
                          ppr_decay=ppr_decay,
                          w_recency=w_recency, w_salience=w_salience,
                          w_retention=w_retention,
                          half_life_days=half_life_days,
                          sem_floor=sem_floor, kw_floor=kw_floor, now=now)
        finally:
            conn.close()
        ranked_lists.append([(label, h.slug) for h in hits])
        for h in hits:
            per_scope[(label, h.slug)] = h
    out: list[Hit] = []
    for key, score in _rrf(ranked_lists, k, rrf_k=rrf_k):
        h = per_scope[key]
        out.append(Hit(slug=h.slug, description=h.description, snippet=h.snippet,
                       score=score, corpus=h.corpus, kind=h.kind, body=h.body,
                       valid_to=h.valid_to, cos=h.cos, bm25=h.bm25))
    return out


def rerank_hits(scorer, query_text: str, hits: list[Hit], k: int) -> list[Hit]:
    """Reorder ``hits`` by a cross-encoder relevance score, keep the top-k.
    ``scorer`` is a callable ``(query, [passages]) -> [float]`` — injected so the
    daemon path and a unit-test fake both fit. Each hit's score becomes its
    rerank score. A scorer failure or length mismatch degrades to fused order
    (rerank is a refinement, never a hard dependency)."""
    if not hits:
        return []
    passages = [f"{h.description}\n\n{h.body or h.snippet}" for h in hits]
    try:
        scores = scorer(query_text, passages)
    except Exception:  # noqa: BLE001 — optional refinement; degrade gracefully
        return hits[:k]
    if len(scores) != len(hits):
        return hits[:k]
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [
        Hit(slug=hits[i].slug, description=hits[i].description,
            snippet=hits[i].snippet, score=float(scores[i]),
            corpus=hits[i].corpus, kind=hits[i].kind, body=hits[i].body,
            valid_to=hits[i].valid_to)
        for i in order[:k]
    ]
