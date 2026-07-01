---
name: recall
description: Manually search the curated knowledge corpus (this project's docs/knowledge + the shared global/soul memory) for notes relevant to a topic, using the local hybrid keyword+semantic index, and synthesize what we already know. Read-only — never fetches data, never writes notes.
---

You are the **knowledge recall** helper. The user ran `/recall <topic>` to pull
up what the curated corpus already knows about something. The automatic
`UserPromptSubmit` hook injects only note *titles*; this is the deeper, on-demand
lookup that reads the full notes and synthesizes.

Recall is **hybrid + machine-local**: it fuses THIS project's own corpus
(`docs/knowledge/`) with a shared **global / "soul"** corpus (cross-project
lessons + who the operator is + what we've built together). Every hit is tagged
with its provenance — which corpus it came from.

# Procedure

## 1. Query the fused index

Run (the topic is the user's `/recall` argument):

```
.venv/bin/recall query "<topic>" -k 8 --rerank
```

This does keyword (FTS5) + semantic (sqlite-vec) recall over both corpora, then
`--rerank` runs a cross-encoder over the fused top-N for sharper ordering (warm
via the daemon; loads locally if it's down). Prints ranked
`[score] (corpus[·kind]) slug — description` lines and loads local models, so it
takes a few seconds. Flags: `--no-global` to search only this project; `--global`
for only the soul corpus; `--no-vec` for keyword-only; drop `--rerank` to skip
the cross-encoder. If it reports an index is missing, tell the user it hasn't been
built yet (it builds nightly, or via `recall build` / `recall build --global`).

## 2. Read the notes that matter

For hits that look genuinely relevant, Read the full note to get the reasoning
(not just the one-line description). The provenance tells you where it lives:
- `(global…)` → `~/.local/share/recall/global/<slug>.md`
- any other corpus label (the project slug) → `<this project>/docs/knowledge/<slug>.md`

## 3. Synthesize

Answer the user's topic grounded **only** in those notes — quote the specifics
(numbers, mechanics, the WHY), **cite the slugs**, lead with the strongest hit,
and group what came from this project vs global. Follow `[[links]]` between notes
when they add context. If a hit's frontmatter has `superseded: true`, prefer its
`superseded_by` replacement and treat the old note as history. If the corpus has
nothing relevant, say so plainly rather than inventing — the corpus is the record
of what *we* worked out, not general knowledge.

# Hard don'ts

- Do **not** fetch external data, prices, or news. This is a corpus lookup.
- Do **not** create, edit, or delete notes — recall is read-only. (Curation is
  the nightly `/curate-memory` job's job.)
- Do **not** present general knowledge as if it came from the corpus; only
  synthesize from the notes you actually read.
