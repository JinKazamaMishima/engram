"""Eval harness — measure recall quality over query→expected-slug cases.

A cases file (YAML or JSON) holds a list of ``{query, expect: [slugs], category}``.
We run each query through the fused index (optionally reranked) and report, over
the POSITIVE cases (non-empty ``expect``): **recall@k** (fraction of expected
slugs surfaced in the top-k), **recall@1**, **MRR**, and **nDCG@k** (binary gain).

A case with ``expect: []`` is an ABSTENTION case — it passes iff no hit clears
``abstain_threshold`` (an out-of-domain query should surface nothing confident).
Abstention is scored separately as ``abstain_rate`` and NEVER inflates the
positive metrics. The scorer/embedder are injected so the metric logic is
unit-tested with fakes — and so the CLI can A/B configs on the same cases.
Numbers, not vibes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import yaml

from recall import index as I


def load_cases(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text())
    raw = data["cases"] if isinstance(data, dict) and "cases" in data else data
    if not isinstance(raw, list):
        raise ValueError("cases file must be a list (or {cases: [...]})")
    cases = []
    for c in raw:
        if not isinstance(c, dict) or "query" not in c:
            raise ValueError(f"each case needs a 'query': {c!r}")
        cases.append({"query": str(c["query"]),
                      "expect": [str(s) for s in c.get("expect", [])],
                      "category": str(c.get("category", ""))})
    return cases


def _ndcg(got: list[str], expect: set[str], k: int) -> float:
    """Binary-gain nDCG@k: gain 1 for an expected slug at its rank; the ideal
    ranking packs all expected slugs at the top."""
    dcg = sum(1.0 / math.log2(i + 2)
              for i, s in enumerate(got[:k]) if s in expect)
    ideal_n = min(len(expect), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg else 0.0


def evaluate(cases: list[dict], scopes: list[tuple[str, Path]], *, k: int = 5,
             query_vector_fn: Callable[[str], list[float]] | None = None,
             reranker: Callable[[str, list[str]], list[float]] | None = None,
             rerank_pool: int = 40, abstain_threshold: float = 0.0,
             rrf_k: int | None = None,
             arm_weights: tuple[float, float] | None = None,
             link_decay: float | None = None, ppr_decay: float | None = None,
             w_recency: float | None = None,
             w_salience: float | None = None, w_retention: float | None = None,
             half_life_days: float | None = None,
             sem_floor: float | None = None, kw_floor: float | None = None,
             now=None) -> dict:
    """Run cases against the scopes; return aggregate + per-case metrics.
    ``query_vector_fn`` None → keyword-only. ``reranker`` set → fuse a deeper
    pool then cross-encoder-rerank to k. Positive cases (non-empty ``expect``)
    drive recall@k / recall@1 / MRR / nDCG@k; ``expect: []`` cases are abstention
    cases scored by ``abstain_rate`` (pass = nothing scored ≥ ``abstain_threshold``).
    ``rrf_k``/``arm_weights``/``link_decay``/``w_recency``/``w_salience``/
    ``w_retention``/``half_life_days``/``now`` tune fusion + structural/temporal
    ranking and thread into ``search_corpora`` (each forwarded only when set)."""
    sc_kw: dict = {}
    for _name, _val in (("rrf_k", rrf_k), ("arm_weights", arm_weights),
                        ("link_decay", link_decay), ("ppr_decay", ppr_decay),
                        ("w_recency", w_recency),
                        ("w_salience", w_salience), ("w_retention", w_retention),
                        ("half_life_days", half_life_days),
                        ("sem_floor", sem_floor), ("kw_floor", kw_floor),
                        ("now", now)):
        if _val is not None:
            sc_kw[_name] = _val
    rec_k: list[float] = []
    rec_1: list[float] = []
    rrs: list[float] = []
    ndcgs: list[float] = []
    abstains: list[float] = []
    per: list[dict] = []
    for c in cases:
        qvec = query_vector_fn(c["query"]) if query_vector_fn else None
        pool_k = max(k, rerank_pool) if reranker else k
        hits = I.search_corpora(scopes, c["query"], query_vector=qvec,
                                k=pool_k, **sc_kw)
        hits = (I.rerank_hits(reranker, c["query"], hits, k) if reranker
                else hits[:k])
        got = [h.slug for h in hits]
        top_score = round(hits[0].score, 4) if hits else 0.0
        expect = set(c["expect"])
        cat = c.get("category", "")
        if not expect:  # abstention case — pass iff nothing clears the threshold
            ok = 1.0 if all(h.score < abstain_threshold for h in hits) else 0.0
            abstains.append(ok)
            per.append({"query": c["query"], "category": cat, "abstain": ok,
                        "top_score": top_score, "got": got, "expect": []})
            continue
        recall = len([s for s in expect if s in got]) / len(expect)
        recall1 = 1.0 if got and got[0] in expect else 0.0
        rr = next((1.0 / rank for rank, s in enumerate(got, 1) if s in expect),
                  0.0)
        ndcg = _ndcg(got, expect, k)
        rec_k.append(recall)
        rec_1.append(recall1)
        rrs.append(rr)
        ndcgs.append(ndcg)
        per.append({"query": c["query"], "category": cat, "recall": recall,
                    "recall1": recall1, "rr": rr, "ndcg": ndcg,
                    "top_score": top_score, "got": got,
                    "expect": sorted(expect)})

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {"n": len(cases), "k": k,
            "n_positive": len(rec_k), "n_abstain": len(abstains),
            "recall_at_k": _avg(rec_k), "recall_at_1": _avg(rec_1),
            "mrr": _avg(rrs), "ndcg_at_k": _avg(ndcgs),
            "abstain_rate": _avg(abstains), "per_case": per}
