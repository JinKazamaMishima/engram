"""``recall`` — command-line entry point.

  recall build  [--project DIR | --global] [--corpus DIR] [--db PATH]
  recall query  TEXT [--project DIR | --global] [-k N] [--no-vec] [--db PATH]
  recall paths  [--project DIR]

``build`` and ``query`` load the local embedding model; ``--no-vec`` skips it
for keyword-only recall. Scope resolution: ``--global`` targets the shared soul
corpus; otherwise the project (``--project``, default cwd) and its
``docs/knowledge`` corpus. ``--corpus`` / ``--db`` override either explicitly.

Multi-corpus *fused* query (project + global together) lands in Phase B; for now
``query`` searches the single resolved scope.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recall import config


def _rerank_scorer():
    """A ``(query, passages) -> [score]`` callable that prefers the warm
    recall-embedder daemon's /rerank endpoint (fast, model already loaded) and
    falls back to loading the cross-encoder locally if the daemon is down."""
    import json
    import os
    import urllib.request
    host = os.environ.get("RECALL_EMBED_HOST", "127.0.0.1")
    port = os.environ.get("RECALL_EMBED_PORT", "8973")
    state: dict = {}

    def scorer(query: str, passages: list[str]) -> list[float]:
        data = json.dumps({"query": query, "passages": passages}).encode()
        req = urllib.request.Request(
            f"http://{host}:{port}/rerank", data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["scores"]
        except Exception:  # noqa: BLE001 — daemon down: load locally (once)
            if "local" not in state:
                from recall.index import CrossEncoderReranker
                state["local"] = CrossEncoderReranker()
            return state["local"].score(query, passages)

    return scorer


def _resolve_scope(args) -> tuple[str, Path, Path]:
    """(label, corpus_dir, db_path) for the requested scope, honoring overrides."""
    if getattr(args, "global_scope", False):
        label = config.GLOBAL_SCOPE
        corpus = config.global_corpus_dir()
        db = config.index_path(config.GLOBAL_SCOPE)
    else:
        project = Path(args.project).resolve() if args.project else Path.cwd()
        label = config.project_slug(project)
        corpus = config.project_corpus_dir(project)
        db = config.index_path(label)
    if getattr(args, "corpus", None):
        corpus = Path(args.corpus).resolve()
    if getattr(args, "db", None):
        db = Path(args.db).resolve()
    return label, corpus, db


def _cmd_build(args) -> int:
    from recall.index import build_index
    label, corpus, db = _resolve_scope(args)
    if not corpus.is_dir():
        # The shared soul corpus legitimately starts empty on a fresh install;
        # create it so the first `recall build --global` succeeds (0 notes).
        if getattr(args, "global_scope", False):
            corpus.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[recall] corpus dir does not exist: {corpus}", file=sys.stderr)
            return 1
    # Only load the (heavy) embedding model when there's something to embed AND the
    # local ML stack is present — otherwise build a keyword-only index so recall
    # works with zero external deps (semantic needs a later rebuild with models).
    has_notes = any(p.name.lower() != "readme.md" for p in corpus.glob("*.md"))
    embedder = None
    if has_notes:
        try:
            from recall.index import SentenceTransformerEmbedder
            embedder = SentenceTransformerEmbedder()
        except ImportError:
            print("[recall] semantic models not installed — building a keyword-only "
                  "index (FTS5). Install the ML extras for semantic search.",
                  file=sys.stderr)
    n = build_index(corpus, db, embedder)
    mode = "keyword+semantic" if embedder is not None else "keyword-only"
    print(f"[recall] indexed {n} notes from {corpus} -> {db}  "
          f"(scope: {label}, {mode})")
    return 0


def _query_scopes(args) -> list[tuple[str, Path]]:
    """Index scopes to fuse for a query. Default: this project + global. ``--db``
    pins a single index (Phase-A style); ``--global`` restricts to the soul
    corpus; ``--no-global`` restricts to the project."""
    if getattr(args, "db", None):
        label, _corpus, db = _resolve_scope(args)
        return [(label, db)]
    if getattr(args, "global_scope", False):
        return [(config.GLOBAL_SCOPE, config.index_path(config.GLOBAL_SCOPE))]
    project = Path(args.project).resolve() if args.project else Path.cwd()
    slug = config.project_slug(project)
    scopes = [(slug, config.index_path(slug))]
    if not getattr(args, "no_global", False):
        scopes.append((config.GLOBAL_SCOPE, config.index_path(config.GLOBAL_SCOPE)))
    return scopes


def _cmd_query(args) -> int:
    from recall import index
    scopes = _query_scopes(args)
    if not any(Path(db).exists() for _, db in scopes):
        where = ", ".join(f"{lbl} ({db})" for lbl, db in scopes)
        print(f"[recall] no index yet for: {where} — run `recall build` first.",
              file=sys.stderr)
        return 1
    qvec = None
    if not args.no_vec:
        try:
            from recall.index import SentenceTransformerEmbedder
            qvec = SentenceTransformerEmbedder().embed([args.text], is_query=True)[0]
        except ImportError:
            print("[recall] semantic models not installed — keyword-only search "
                  "(pass --no-vec to silence this).", file=sys.stderr)
    rerank = getattr(args, "rerank", False)
    pool = max(args.k, 40) if rerank else args.k
    hits = index.search_corpora(scopes, args.text, query_vector=qvec, k=pool)
    if rerank:
        hits = index.rerank_hits(_rerank_scorer(), args.text, hits, args.k)
    for i, h in enumerate(hits, 1):
        tag = f" [{h.kind}]" if h.kind else ""
        print(f"{i}. [{h.score:.4f}] ({h.corpus}){tag} {h.slug} — {h.description}")
        print(f"     {h.snippet}")
    if not hits:
        print("(no matches)")
    return 0


def _cmd_paths(args) -> int:
    project = Path(args.project).resolve() if args.project else Path.cwd()
    slug = config.project_slug(project)
    print(f"data_root           = {config.data_root()}")
    print(f"project             = {project}")
    print(f"project slug        = {slug}")
    print(f"project corpus      = {config.project_corpus_dir(project)}")
    print(f"project index       = {config.index_path(slug)}")
    print(f"global corpus       = {config.global_corpus_dir()}")
    print(f"global index        = {config.index_path(config.GLOBAL_SCOPE)}")
    return 0


def _add_scope_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", default=None,
                   help="project dir (default: cwd)")
    p.add_argument("--global", dest="global_scope", action="store_true",
                   help="target the shared global/soul corpus instead of a project")
    p.add_argument("--corpus", default=None, help="override the corpus dir")
    p.add_argument("--db", default=None, help="override the index DB path")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recall",
                                description="machine-local hybrid knowledge recall")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="rebuild the index from a corpus")
    _add_scope_flags(b)
    b.set_defaults(func=_cmd_build)

    q = sub.add_parser("query", help="hybrid search (loads the local model)")
    q.add_argument("text")
    _add_scope_flags(q)
    q.add_argument("-k", type=int, default=5)
    q.add_argument("--no-vec", action="store_true",
                   help="keyword-only (skip the model)")
    q.add_argument("--no-global", action="store_true",
                   help="search only this project (skip the global/soul corpus)")
    q.add_argument("--rerank", action="store_true",
                   help="cross-encoder rerank the fused top-N (loads the reranker)")
    q.set_defaults(func=_cmd_query)

    pa = sub.add_parser("paths", help="print resolved corpus/index paths")
    pa.add_argument("--project", default=None)
    pa.set_defaults(func=_cmd_paths)

    r = sub.add_parser("register",
                       help="add a project to the nightly curate registry")
    r.add_argument("project", nargs="?", default=None,
                   help="project dir (default: cwd)")
    r.set_defaults(func=_cmd_register)

    pr = sub.add_parser("projects", help="list registered projects")
    pr.set_defaults(func=_cmd_projects)

    si = sub.add_parser("similar",
                        help="nearest existing notes to a text blob (dedup helper)")
    si.add_argument("text")
    _add_scope_flags(si)
    si.add_argument("-k", type=int, default=5)
    si.add_argument("--no-global", action="store_true")
    si.set_defaults(func=_cmd_similar)

    ev = sub.add_parser("eval", help="measure recall quality over a cases file")
    ev.add_argument("cases", help="YAML/JSON: list of {query, expect: [slugs]}")
    _add_scope_flags(ev)
    ev.add_argument("-k", type=int, default=5)
    ev.add_argument("--no-vec", action="store_true")
    ev.add_argument("--no-global", action="store_true")
    ev.add_argument("--rerank", action="store_true")
    ev.add_argument("--compare", action="store_true",
                    help="A/B keyword vs hybrid vs hybrid+rerank on the same cases")
    ev.add_argument("--abstain-threshold", type=float, default=0.0,
                    help="abstention case passes iff no hit scores ≥ this "
                         "(tune with --rerank; cross-encoder scores are calibrated)")
    ev.add_argument("--rrf-k", type=int, default=None,
                    help="override the RRF fusion constant (default 60)")
    ev.add_argument("--kw-weight", type=float, default=None,
                    help="weight for the keyword arm in fusion (default 1.0)")
    ev.add_argument("--vec-weight", type=float, default=None,
                    help="weight for the vector arm in fusion (default 1.0)")
    ev.add_argument("--no-links", action="store_true",
                    help="disable [[wikilink]] 1-hop expansion (link_decay=0)")
    ev.add_argument("--ppr-decay", type=float, default=None,
                    help="enable the PPR spreading-activation graph arm at this RRF "
                         "weight (default 0=off; pair with --no-links to replace 1-hop)")
    ev.add_argument("--recency-w", type=float, default=None,
                    help="recency weight in the score blend (default 0 = off)")
    ev.add_argument("--salience-w", type=float, default=None,
                    help="salience weight in the score blend (default 0 = off)")
    ev.add_argument("--retention-w", type=float, default=None,
                    help="retention weight (FSRS R(t,S) keyed on use; default 0 = off)")
    ev.add_argument("--half-life", type=float, default=None,
                    help="recency half-life in days (default 180)")
    ev.add_argument("--sweep", default=None,
                    help="sweep one knob: rrf_k|recency_w|salience_w=v1,v2,... "
                         "(runs the chosen config once per value)")
    ev.add_argument("--verbose", action="store_true", help="per-case breakdown")
    ev.set_defaults(func=_cmd_eval)
    return p


def _cmd_similar(args) -> int:
    """Nearest existing notes to a blob of text (passage-side embedding) — for
    pre-create dedup: 'is there already a note like this?'"""
    from recall import index
    from recall.index import SentenceTransformerEmbedder
    scopes = _query_scopes(args)
    if not any(Path(db).exists() for _, db in scopes):
        print("[recall] no index yet — run `recall build` first.", file=sys.stderr)
        return 1
    vec = SentenceTransformerEmbedder().embed([args.text], is_query=False)[0]
    hits = index.search_corpora(scopes, args.text, query_vector=vec, k=args.k)
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h.score:.4f}] ({h.corpus}) {h.slug} — {h.description}")
    if not hits:
        print("(no similar notes)")
    return 0


def _cmd_eval(args) -> int:
    from recall import eval as ev
    cases = ev.load_cases(args.cases)
    scopes = _query_scopes(args)
    if not any(Path(db).exists() for _, db in scopes):
        print("[recall] no index yet — run `recall build` first.", file=sys.stderr)
        return 1
    if args.compare:
        configs = [("keyword", False, False), ("hybrid", True, False),
                   ("hybrid+rerank", True, True)]
    else:
        configs = [("result", not args.no_vec, args.rerank)]

    aw = None
    if args.kw_weight is not None or args.vec_weight is not None:
        aw = (args.kw_weight if args.kw_weight is not None else 1.0,
              args.vec_weight if args.vec_weight is not None else 1.0)

    tune: dict = {}
    if aw is not None:
        tune["arm_weights"] = aw
    if args.rrf_k is not None:
        tune["rrf_k"] = args.rrf_k
    if args.no_links:
        tune["link_decay"] = 0.0
    if args.ppr_decay is not None:
        tune["ppr_decay"] = args.ppr_decay
    if args.recency_w is not None:
        tune["w_recency"] = args.recency_w
    if args.salience_w is not None:
        tune["w_salience"] = args.salience_w
    if args.retention_w is not None:
        tune["w_retention"] = args.retention_w
    if args.half_life is not None:
        tune["half_life_days"] = args.half_life

    sweep_key = sweep_vals = None
    if args.sweep:
        sweep_key, _, vals = args.sweep.partition("=")
        sweep_key = sweep_key.strip()
        if (sweep_key not in ("rrf_k", "recency_w", "salience_w", "retention_w",
                              "ppr_decay") or not vals.strip()):
            print("[recall] --sweep supports "
                  "rrf_k|recency_w|salience_w|retention_w|ppr_decay=v1,v2,...",
                  file=sys.stderr)
            return 1
        sweep_vals = [float(v) for v in vals.split(",") if v.strip()]

    embedder = {"e": None}
    reranker = {"r": None}

    def _qfn():
        if embedder["e"] is None:
            from recall.index import SentenceTransformerEmbedder
            embedder["e"] = SentenceTransformerEmbedder()
        emb = embedder["e"]
        return lambda t: emb.embed([t], is_query=True)[0]

    def _rfn():
        if reranker["r"] is None:
            reranker["r"] = _rerank_scorer()
        return reranker["r"]

    _SWEEP_KW = {"rrf_k": "rrf_k", "recency_w": "w_recency",
                 "salience_w": "w_salience", "retention_w": "w_retention",
                 "ppr_decay": "ppr_decay"}
    rows = []
    if sweep_vals is not None:
        use_vec, use_rr = not args.no_vec, args.rerank
        for sv in sweep_vals:
            kw = dict(tune)
            kw[_SWEEP_KW[sweep_key]] = int(sv) if sweep_key == "rrf_k" else sv
            res = ev.evaluate(cases, scopes, k=args.k,
                              query_vector_fn=_qfn() if use_vec else None,
                              reranker=_rfn() if use_rr else None,
                              abstain_threshold=args.abstain_threshold, **kw)
            rows.append((f"{sweep_key}={sv:g}", res))
    else:
        for name, use_vec, use_rr in configs:
            res = ev.evaluate(cases, scopes, k=args.k,
                              query_vector_fn=_qfn() if use_vec else None,
                              reranker=_rfn() if use_rr else None,
                              abstain_threshold=args.abstain_threshold, **tune)
            rows.append((name, res))

    r0 = rows[0][1]
    print(f"cases: {r0['n']} (pos {r0['n_positive']}, abstain {r0['n_abstain']})"
          f"   k={args.k}  abstain_threshold={args.abstain_threshold}")
    print(f"{'config':16}{'R@1':>7}{'R@k':>7}{'nDCG':>7}{'MRR':>7}{'abstain':>9}")
    for name, res in rows:
        print(f"{name:16}{res['recall_at_1']:>7.3f}{res['recall_at_k']:>7.3f}"
              f"{res['ndcg_at_k']:>7.3f}{res['mrr']:>7.3f}{res['abstain_rate']:>9.3f}")
    if args.verbose:
        for name, res in rows:
            print(f"\n[{name}]")
            for c in res["per_case"]:
                if not c["expect"]:
                    mark = "✓" if c.get("abstain", 0.0) > 0 else "✗"
                    tag = f"abstain top={c['top_score']:.3f}"
                else:
                    mark = "✓" if c["recall"] > 0 else "✗"
                    tag = (f"r={c['recall']:.2f} r1={c['recall1']:.2f} "
                           f"ndcg={c['ndcg']:.2f}")
                cat = f"[{c['category']}] " if c.get("category") else ""
                print(f"  {mark} {tag}  {cat}{c['query'][:44]!r} -> {c['got'][:3]}")
    return 0


def _cmd_register(args) -> int:
    from recall import registry
    d = Path(args.project).resolve() if args.project else Path.cwd()
    added = registry.register(d)
    print(f"[recall] {'registered' if added else 'already registered'}: {d}")
    print(f"[recall] registry: {registry.registry_path()}")
    return 0


def _cmd_projects(args) -> int:
    from recall import registry
    projs = registry.list_projects()
    if not projs:
        print(f"(no projects registered; {registry.registry_path()})")
        return 0
    for d in projs:
        print(d)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Subcommands that own their own arg parsing get the raw remainder.
    if argv and argv[0] == "curate":
        from recall import curate
        return curate.main(argv[1:])
    if argv and argv[0] == "curate-all":
        from recall import registry
        return registry.curate_all(argv[1:])
    if argv and argv[0] == "consolidate":
        from recall import consolidate
        return consolidate.main(argv[1:])
    if argv and argv[0] == "consolidate-all":
        from recall import registry
        return registry.consolidate_all(argv[1:])
    if argv and argv[0] == "dream":
        from recall import dream
        return dream.main(argv[1:])
    if argv and argv[0] == "dream-all":
        from recall import registry
        return registry.dream_all(argv[1:])
    if argv and argv[0] == "reconsolidate":
        from recall import reconsolidate
        return reconsolidate.main(argv[1:])
    if argv and argv[0] == "reconsolidate-all":
        from recall import registry
        return registry.reconsolidate_all(argv[1:])
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
