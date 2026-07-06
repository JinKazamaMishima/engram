#!/usr/bin/env python3
"""LIVE end-to-end for Brick 3 eviction-is-curation (A6) — the real thing:
REAL SDK turns (subscription `claude`), the REAL detached `recall curate
--buffer` subprocess (headless curator and all), REAL watermark advance —
pointed at a throwaway RECALL_DATA_ROOT + scratch project so no live corpus,
session store, or buffer is touched. Deliberately NOT named test_* — this
costs real model turns and minutes; run it by hand:

    .venv/bin/python infra/engram/tests/live_e2e_evict.py

Passes when: (1) the cooled tail crosses ENGRAM_EVICT_CHARS=2000 mid-session and
core spawns ONE detached curate; (2) the curator finishes and the watermark for
this convo appears in the sandbox curated.json; (3) the manifest exists; (4)
the cooled edge collapses below the gate afterwards (nothing re-curates).
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(ENGRAM, "..", ".."))

# --- sandbox FIRST: core reads these at import time ---------------------------
SANDBOX = Path(tempfile.mkdtemp(prefix="engram-evict-e2e-"))
SCRATCH = SANDBOX / "project"
(SCRATCH / "docs" / "knowledge").mkdir(parents=True)
subprocess.run(["git", "init", "-q"], cwd=SCRATCH, check=False)
os.environ.update({
    "RECALL_DATA_ROOT": str(SANDBOX / "data"),
    "ENGRAM_BUFFER_DIR": str(SANDBOX / "buffer"),
    "ENGRAM_SESSION_DIR": str(SANDBOX / "sessions"),
    "ENGRAM_LOCK_DIR": str(SANDBOX / "locks"),
    "ENGRAM_EVICT_CHARS": "2000",   # fire fast: 2k cooled chars
    "ENGRAM_WM_TURNS": "2",         # hot window = last 2 rows (1 exchange)
})

sys.path.insert(0, ENGRAM)
sys.path.insert(0, os.path.join(REPO, "src"))

import core  # noqa: E402 — after the env, on purpose

PROMPTS = [
    "In ~120 words of plain prose (no tools, no lists), explain why an "
    "append-only log beats in-place summarization for conversational memory.",
    "In ~120 words of plain prose (no tools), describe how a watermark makes "
    "incremental processing of a log idempotent.",
    "In ~120 words of plain prose (no tools), explain why recently-written "
    "log entries ('hot' data) should be excluded from background compaction.",
    "In ~120 words of plain prose (no tools), summarize how the three ideas "
    "above compose into a tiered memory system.",
]


async def run() -> int:
    from recall import config
    d = core.AgentSDKDriver(cwd=SCRATCH, model="sonnet", effort="low")
    print(f"sandbox: {SANDBOX}")
    print(f"buffer:  {d._buffer.path()}  (evict at {core.EVICT_CHARS} cooled "
          f"chars, hot window {core.EVICT_HOT_TURNS} rows)")
    await d.connect()
    fired_after = None
    try:
        for i, p in enumerate(PROMPTS, 1):
            out = []
            async for ev in d.query(p):
                if ev.kind == "text":
                    out.append(ev.text)
            edge = d._cooled_edge()
            print(f"turn {i}: reply {len(''.join(out))} ch · rows "
                  f"{d._buffer.last_seq()} · cooled "
                  f"{edge[1] if edge else 0} ch · evicting={d._evicting}")
            if d._evicting and fired_after is None:
                fired_after = i
                break                      # the spawn we came for — stop turning
    finally:
        await d.disconnect()

    if fired_after is None:
        print("✗ FAIL: size gate never fired (cooled tail under threshold?)")
        return 1
    print(f"→ detached curate spawned after turn {fired_after}; waiting on the "
          f"real curator …")
    argv = core.buffer_curation_cmd(d._buffer.path(), SCRATCH)
    print(f"  (argv: {' '.join(argv[:6])} …)")

    convo = d._buf_convo_id
    state = (config.curation_dir() / config.project_slug(SCRATCH)
             / "curated.json")
    mark = ""
    deadline = time.time() + 12 * 60
    while time.time() < deadline:
        try:
            mark = (json.loads(state.read_text())
                    .get("watermarks", {}).get(convo, ""))
        except OSError:
            mark = ""
        if mark:
            break
        await asyncio.sleep(10)   # keep the loop live: the reaper task shares it
    if not mark:
        print(f"✗ FAIL: no watermark for {convo} in {state} after 12 min")
        return 1
    print(f"✓ watermark advanced: {mark}")

    manifest = (config.curation_dir() / config.project_slug(SCRATCH)
                / "manifests" / f"session-{convo}.json")
    print(f"{'✓' if manifest.exists() else '✗'} manifest: {manifest}")

    edge = d._cooled_edge()
    cooled_now = edge[1] if edge else 0
    ok_idem = cooled_now < core.EVICT_CHARS
    print(f"{'✓' if ok_idem else '✗'} post-eviction cooled tail = "
          f"{cooled_now} ch (< {core.EVICT_CHARS}: nothing re-curates)")

    proj_notes = list((SCRATCH / "docs" / "knowledge").glob("*.md"))
    soul_dir = Path(os.environ["RECALL_DATA_ROOT"]) / "global"
    soul_notes = list(soul_dir.glob("*.md")) if soul_dir.is_dir() else []
    print(f"  curator wrote {len(proj_notes)} project note(s), "
          f"{len(soul_notes)} soul note(s) [provisional]")

    ok = manifest.exists() and ok_idem
    print("\nE2E " + ("PASS" if ok else "FAIL") + f" — sandbox kept at {SANDBOX}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
