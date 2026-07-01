"""Tests for recall.eval — the metric logic, exercised with a fake embedder and
a fake reranker (no model download). Builds a tiny real index so search_corpora
runs end-to-end."""
from __future__ import annotations

import hashlib
import math
import re

from recall import eval as ev
from recall import index


class FakeEmbedder:
    dim = 16

    def embed(self, texts, *, is_query=False):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def _note(d, slug, desc, body):
    (d / f"{slug}.md").write_text(
        f"---\nname: {slug}\ndescription: {desc}\ntags: [t]\n---\n{body}\n")


def _index(tmp_path):
    kd = tmp_path / "k"
    kd.mkdir()
    _note(kd, "alpha", "about alpha widgets", "alpha widget mechanics and why")
    _note(kd, "beta", "about beta gadgets", "beta gadget mechanics and why")
    db = tmp_path / "i.sqlite"
    emb = FakeEmbedder()
    index.build_index(kd, db, emb)
    return db, emb


def test_evaluate_hybrid_metrics(tmp_path):
    db, emb = _index(tmp_path)
    cases = [{"query": "alpha widgets", "expect": ["alpha"]},
             {"query": "beta gadgets", "expect": ["beta"]}]
    res = ev.evaluate(cases, [("proj", db)], k=2,
                      query_vector_fn=lambda t: emb.embed([t], is_query=True)[0])
    assert res["n"] == 2
    assert res["recall_at_k"] == 1.0
    assert res["mrr"] > 0.0


def test_evaluate_rerank_path(tmp_path):
    db, emb = _index(tmp_path)
    cases = [{"query": "alpha", "expect": ["alpha"]}]

    def scorer(_q, passages):
        return [9.0 if "alpha" in p else 0.0 for p in passages]
    res = ev.evaluate(cases, [("proj", db)], k=1,
                      query_vector_fn=lambda t: emb.embed([t], is_query=True)[0],
                      reranker=scorer)
    assert res["recall_at_k"] == 1.0
    assert res["per_case"][0]["got"][0] == "alpha"


def test_evaluate_keyword_only(tmp_path):
    db, _emb = _index(tmp_path)
    res = ev.evaluate([{"query": "beta gadgets", "expect": ["beta"]}],
                      [("proj", db)], k=2)  # no query_vector_fn -> keyword-only
    assert res["recall_at_k"] == 1.0


def test_load_cases(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("cases:\n  - query: why X\n    expect: [x-note]\n"
                 "  - query: why Y\n    expect: [y-note, y2]\n")
    cases = ev.load_cases(f)
    assert len(cases) == 2
    assert cases[0]["query"] == "why X" and cases[0]["expect"] == ["x-note"]
    assert cases[1]["expect"] == ["y-note", "y2"]


def test_ndcg_and_recall_at_1(tmp_path):
    db, emb = _index(tmp_path)
    res = ev.evaluate([{"query": "alpha widgets", "expect": ["alpha"]}],
                      [("proj", db)], k=2,
                      query_vector_fn=lambda t: emb.embed([t], is_query=True)[0])
    assert res["recall_at_1"] == 1.0
    assert 0.0 < res["ndcg_at_k"] <= 1.0
    assert res["n_positive"] == 1 and res["n_abstain"] == 0


def test_abstention_case_scoring(tmp_path):
    db, emb = _index(tmp_path)
    case = [{"query": "alpha widgets", "expect": [], "category": "abstention"}]
    def qfn(t):
        return emb.embed([t], is_query=True)[0]
    # high threshold: nothing clears it -> abstains correctly
    hi = ev.evaluate(case, [("proj", db)], k=2, query_vector_fn=qfn,
                     abstain_threshold=1.0)
    assert hi["abstain_rate"] == 1.0 and hi["n_abstain"] == 1
    # zero threshold: any returned hit (score>0) -> fails to abstain
    lo = ev.evaluate(case, [("proj", db)], k=2, query_vector_fn=qfn,
                     abstain_threshold=0.0)
    assert lo["abstain_rate"] == 0.0


def test_aggregate_excludes_abstain(tmp_path):
    db, emb = _index(tmp_path)
    cases = [{"query": "alpha widgets", "expect": ["alpha"]},
             {"query": "nothing relevant here", "expect": []}]
    res = ev.evaluate(cases, [("proj", db)], k=2,
                      query_vector_fn=lambda t: emb.embed([t], is_query=True)[0],
                      abstain_threshold=1.0)
    assert res["n"] == 2 and res["n_positive"] == 1 and res["n_abstain"] == 1
    assert res["recall_at_k"] == 1.0  # averaged over the single positive only
    assert res["abstain_rate"] == 1.0
