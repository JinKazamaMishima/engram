#!/usr/bin/env python3
"""T2 evidence-floor sweep over the durable blind set (numbers, not vibes).

Embeds every blind query ONCE via the warm daemon, then sweeps RECALL_SEM_FLOOR
values in-process through ``search_corpora`` (the hook's exact scopes: this
repo's project corpus + global). Reports, per floor: hit@5 + MRR per kind
(direct / situational / oblique) over the 75 positives, and injection stats
over the 10 negatives. Also prints the evidence distributions at floor=0 —
each target's cos when found, each negative's best cos — so the separability
(or its absence) is visible directly, not inferred.

Selection rule: highest floor with ZERO positive regression vs the floor=0
baseline; abstention on negatives is whatever that floor buys for free.

Usage:  .venv/bin/python scripts/floor_sweep.py [cases.json] [--kw-floor F]
Writes: <cases-dir>/floor_sweep.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recall import config, index  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FLOORS = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
K = 5


def main() -> int:
    argv = sys.argv[1:]
    kw_floor = index.KW_FLOOR
    if "--kw-floor" in argv:
        i = argv.index("--kw-floor")
        kw_floor = float(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = argv
    cases_file = Path(args[0]) if args else (
        Path.home() / ".local/share/recall/eval/cases.json")
    data = json.loads(cases_file.read_text())
    cases, negatives = data["cases"], data["negatives"]

    slug = config.project_slug(REPO)
    scopes = [(slug, config.index_path(slug)),
              (config.GLOBAL_SCOPE, config.index_path(config.GLOBAL_SCOPE))]

    print(f"embedding {len(cases)} + {len(negatives)} queries via the daemon…")
    emb = index.DaemonEmbedder()          # raises if the daemon is down: fail LOUD
    queries = [c["query"] for c in cases] + list(negatives)
    vecs = emb.embed(queries, is_query=True)
    qv = dict(zip(queries, vecs))

    # --- evidence distributions at floor=0 (the separability picture) -------
    pos_cos, neg_cos = [], []
    for c in cases:
        hits = index.search_corpora(scopes, c["query"], query_vector=qv[c["query"]],
                                    k=K, sem_floor=0.0)
        target = next((h for h in hits if h.slug == c["target"]), None)
        pos_cos.append({"target": c["target"], "kind": c["kind"],
                        "cos": round(target.cos, 4) if target else None,
                        "rank": (1 + [h.slug for h in hits].index(c["target"]))
                                if target else None})
    for q in negatives:
        hits = index.search_corpora(scopes, q, query_vector=qv[q],
                                    k=K, sem_floor=0.0)
        neg_cos.append({"query": q[:60],
                        "best_cos": round(max((h.cos for h in hits), default=0.0), 4),
                        "best_bm25": round(min((h.bm25 for h in hits if h.bm25 < 0),
                                               default=0.0), 2),
                        "n": len(hits)})
    found = [p["cos"] for p in pos_cos if p["cos"] is not None]
    print(f"\npositives found@5 at floor=0: {len(found)}/{len(cases)}")
    if found:
        print(f"  target cos: min {min(found):.3f}  p10 "
              f"{sorted(found)[len(found)//10]:.3f}  median "
              f"{sorted(found)[len(found)//2]:.3f}")
    print("  negatives best cos each: "
          + ", ".join(f"{n['best_cos']:.3f}" for n in neg_cos))

    # --- the sweep -----------------------------------------------------------
    rows = []
    for f in FLOORS:
        kinds: dict[str, list] = {}
        for c in cases:
            hits = index.search_corpora(scopes, c["query"],
                                        query_vector=qv[c["query"]],
                                        k=K, sem_floor=f, kw_floor=kw_floor)
            slugs = [h.slug for h in hits]
            rank = slugs.index(c["target"]) + 1 if c["target"] in slugs else None
            kinds.setdefault(c["kind"], []).append(rank)
        neg_inj = 0
        neg_notes = 0
        for q in negatives:
            hits = index.search_corpora(scopes, q, query_vector=qv[q],
                                        k=K, sem_floor=f, kw_floor=kw_floor)
            neg_inj += 1 if hits else 0
            neg_notes += len(hits)
        row = {"floor": f, "kw_floor": kw_floor,
               "neg_injected": neg_inj, "neg_notes": neg_notes}
        for kind, ranks in sorted(kinds.items()):
            hit = [r for r in ranks if r]
            row[f"{kind}_hit"] = f"{len(hit)}/{len(ranks)}"
            row[f"{kind}_mrr"] = round(
                sum(1 / r for r in hit) / len(ranks), 3) if ranks else 0.0
        allr = [r for rs in kinds.values() for r in rs]
        allhit = [r for r in allr if r]
        row["all_hit"] = f"{len(allhit)}/{len(allr)}"
        row["all_pct"] = round(100 * len(allhit) / len(allr), 1)
        rows.append(row)

    hdr = ["floor", "all_hit", "all_pct", "direct_hit", "situational_hit",
           "oblique_hit", "neg_injected", "neg_notes"]
    print("\n" + " | ".join(f"{h:>15s}" for h in hdr))
    print("-" * (18 * len(hdr)))
    for r in rows:
        print(" | ".join(f"{str(r.get(h, '')):>15s}" for h in hdr))

    out = cases_file.with_name("floor_sweep.json")
    out.write_text(json.dumps({"rows": rows, "pos_cos": pos_cos,
                               "neg_cos": neg_cos}, indent=2))
    print(f"\nresults -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
