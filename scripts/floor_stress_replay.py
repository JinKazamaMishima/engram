#!/usr/bin/env python3
"""T2 floor stress-replay over REAL prompts (the operator gate before the floor ships).

Harvests genuine user prompts from the harness LiveBuffer JSONLs (the live
injection path's own traffic — not synthetic eval queries), then runs each
through ``search_corpora`` twice: floor OFF vs the blessed floor. Reports every
note the floor would drop, sorted by cos DESCENDING — the top of that list is
the risk zone (near-floor drops of possibly-relevant notes), the bottom is the
junk the floor exists to kill. A human reads the report; the numbers alone
don't bless this one.

Usage:  .venv/bin/python scripts/floor_stress_replay.py [--floor 0.45] [--max N]
Writes: /tmp/floor_stress_report.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recall import config, index  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BUFFER = Path(os.environ.get("ENGRAM_BUFFER_DIR", str(Path.home() / ".local/share/recall/engram/buffer")))
MIN_PROMPT = 12          # mirrors the hook's gate
OUT = Path("/tmp/floor_stress_report.md")


def harvest(max_n: int) -> list[str]:
    rows: list[tuple[str, str]] = []
    for f in BUFFER.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = (r.get("text") or "").strip()
            if (r.get("role") == "user" and len(t) >= MIN_PROMPT
                    and not t.startswith("/")):
                rows.append((r.get("ts", ""), t))
    rows.sort(reverse=True)                      # newest first
    seen: set[str] = set()
    out: list[str] = []
    for _, t in rows:
        key = t[:200]
        if key not in seen:
            seen.add(key)
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def main() -> int:
    floor = 0.45
    max_n = 150
    if "--floor" in sys.argv:
        floor = float(sys.argv[sys.argv.index("--floor") + 1])
    if "--max" in sys.argv:
        max_n = int(sys.argv[sys.argv.index("--max") + 1])

    prompts = harvest(max_n)
    slug = config.project_slug(REPO)
    scopes = [(slug, config.index_path(slug)),
              (config.GLOBAL_SCOPE, config.index_path(config.GLOBAL_SCOPE))]

    print(f"replaying {len(prompts)} real prompts (floor 0 vs {floor})…")
    emb = index.DaemonEmbedder()                 # daemon down -> fail LOUD
    vecs = emb.embed(prompts, is_query=True)

    drops: list[dict] = []
    affected = 0
    n0 = n1 = 0
    zero_after = []
    for p, qv in zip(prompts, vecs):
        base = index.search_corpora(scopes, p, query_vector=qv, k=5,
                                    sem_floor=0.0)
        floored = index.search_corpora(scopes, p, query_vector=qv, k=5,
                                       sem_floor=floor)
        kept = {h.slug for h in floored}
        gone = [h for h in base if h.slug not in kept]
        n0 += len(base)
        n1 += len(floored)
        if gone:
            affected += 1
        if base and not floored:
            zero_after.append(p)
        for h in gone:
            drops.append({"cos": round(h.cos, 4), "slug": h.slug,
                          "corpus": h.corpus, "prompt": p[:140]})
    drops.sort(key=lambda d: d["cos"], reverse=True)

    lines = [
        "# T2 floor stress-replay — real prompts",
        f"\nfloor={floor} · {len(prompts)} prompts · "
        f"{affected} affected ({100*affected/len(prompts):.0f}%)",
        f"\nnotes injected: {n0} → {n1}  ({n0-n1} dropped, "
        f"{100*(n0-n1)/n0:.0f}% of baseline volume)" if n0 else "",
        f"\nprompts going to ZERO injection: {len(zero_after)}",
        "\n## Drops, closest-to-floor first (top = risk zone)\n",
        "| cos | corpus:slug | prompt |", "|---|---|---|",
    ]
    for d in drops[:80]:
        lines.append(f"| {d['cos']:.3f} | {d['corpus']}:{d['slug']} "
                     f"| {d['prompt'].replace('|', '¦')} |")
    if zero_after:
        lines.append("\n## Prompts fully muted by the floor\n")
        for p in zero_after[:20]:
            lines.append(f"- {p[:160]}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"affected {affected}/{len(prompts)} prompts · "
          f"volume {n0}→{n1} · zero-injection {len(zero_after)}")
    print(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
