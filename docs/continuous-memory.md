# Continuous memory: the three-tier stack

How Engram keeps one conversation alive across days — and why that is also the
token- and KV-efficient shape — in one walk-through. The short version lives in
the README's ["context problem"](../README.md#the-context-problem-tokens-kv-and-why-long-chats-rot)
section; this is the long one.

## The failure mode this is built against

Long-lived agents don't die of amnesia; they die of **recursive consolidation**.
The native fix for a full context window is to summarize the transcript in
place — and, when the summarized transcript fills up again, to summarize the
summary. Each pass is lossy, and each pass treats the previous pass's output as
ground truth, so omissions compound silently. After a few cycles the agent
holds a confident, internally-consistent account of a conversation that never
quite happened.

The design rule that falls out of this: **never consolidate a summary — always
re-derive from an immutable record.** Everything below is that rule, applied.

## The three tiers

```
 turn text ──► tier 1 · LiveBuffer          append-only JSONL on disk
                  │                          {convo_id, seq, ts, role, text}
                  │                          immutable · zero tokens · fail-open
                  │
                  ├──► tier 2 · working set  bounded block, RE-DERIVED from
                  │                          tier 1 every turn; rides the top of
                  │                          the current user message
                  │
                  └──► tier 3 · corpus       eviction-as-curation: cooled buffer
                                             history → `recall curate --buffer`
                                             → durable notes → retrieved per-turn
                                             by the hook, only when relevant
```

**Tier 1 — the LiveBuffer.** Every exchange is appended raw, at the moment it
happens, one JSON row per message. The write path is fail-open (a memory
hiccup can never break a turn) and the file is never edited — resumes and
forks re-key it, they don't rewrite it. This is the immutable record everything
else derives from. It costs nothing in context.

**Tier 2 — the working set.** Each turn, a bounded block (`ENGRAM_WM_TURNS`
recent turns' worth, plus durable per-conversation anchors) is *recomputed from
the buffer* and injected at the top of the outgoing user message. It is never
appended to and never summarizes its own previous output — so it cannot drift.
If it's ever wrong, it's wrong for one turn, and the next derivation corrects
it from source.

**Tier 3 — the corpus.** The durable, git-versioned note store the whole
memory brain shares (curation, FSRS dynamics, dreams, reconsolidation). The
retrieval hook injects the top-scoring notes for *this* prompt into *this*
turn — memory comes back by relevance, not residency.

## Eviction is curation

When enough of the buffer has **cooled** — fallen out of the working-set
window past a size threshold (`ENGRAM_EVICT_CHARS`) — the harness spawns a
detached `recall curate --buffer <file> --incremental --provisional`:

- **Off the hot path.** The curate run is a background process; no turn ever
  waits for memory consolidation.
- **Watermarked.** A per-conversation watermark records how far curation has
  advanced. Only the curate CLI moves it, and only after schema validation
  succeeds — a failed run leaves the tail untouched for the next attempt or
  the nightly sweep.
- **Provisional by design.** Facts distilled mid-conversation are stamped with
  low confidence; later passes raise it when the tail corroborates, or
  supersede it when the conversation reverses a decision. The system is
  allowed to watch itself change its mind.
- **Idempotent + swept.** Anything eviction misses (crash, cutout, shutdown)
  is caught by the nightly `curate-sessions-all` sweep; session-scoped
  idempotency buckets stop double-curation.

The upshot: leaving the context window and being forgotten stop being the same
event. Context is a *cache* over memory, not the memory itself.

## Temporal validity: remembering what *used to be* true

A memory system that only stores facts eventually lies — facts expire. When a
note is superseded, the old note isn't deleted or overwritten: it's stamped
`valid_to: <date>` (the day the fact stopped being true) and injects from then
on with an explicit `⏳ HISTORICAL (was true until …)` label. Retrieval can
still find it — "what did we believe before the redesign?" is a real query —
but it can never masquerade as current truth again.

## Perception rides the same rails

With the Sensorium on, perception events stopped dying in a bounded in-memory
deque and became memory in their own right (`perceive/percept.py`):

- **The gate is the point.** A vision model's free text is a confabulation
  vector, and downstream, *surprise buys permanence* — so filtering runs
  BEFORE anything persists. Eye readings persist only when
  corroboration-stable **and** actually changed (an hour of the same desk is
  one row, not four hundred); presence transitions and wake-word utterances
  ride their upstream face-ID/wake-word gates; ambient noise is dropped,
  fail-closed.
- **Day-keyed buffer, same eviction.** Percepts append to a day-keyed
  LiveBuffer with provenance (event kind + gate evidence) and evict into
  curation through the same watermarked path as conversation. Perception
  grows long-term memory with an audit trail.

## The economics, concretely

**Tokens.** An append-only transcript makes turn *N* cost O(N) input tokens —
O(N²) cumulative. Engram's per-turn context is `system + working set (bounded)
+ retrieved notes (top-k) + recent turns` — effectively **flat**. A week-old
conversation prices like a fresh one, and `/new` is always safe because
continuity re-derives from tiers 1 and 3 rather than living in the transcript.

**Prompt cache.** Caching rewards a byte-stable prefix. Every dynamic block
Engram injects — working set, recalled notes, perception marker — rides inside
the *newest* user message, at the end of the prompt, so all prior turns remain
byte-identical and cache-eligible. Cache reads are billed at a fraction of
fresh input; keeping the mutating parts out of the prefix converts most of
each turn's input into cache reads. In-place compaction, by contrast, rewrites
the entire prefix at once — a full-price re-ingest of the whole context — and
Engram's PreCompact hook makes even that survivable by banking a provisional
curation of the buffer first.

**KV memory (the local-model case).** On self-hosted models the KV-cache is
physical VRAM, scaling linearly with resident tokens — order of ~100 KiB per
token for an 8B-class model in fp16 even with grouped-query attention, so a
100k-token context spends more VRAM on KV than 4-bit weights cost for the
model itself. An agent whose continuity *requires* a huge resident context
simply does not fit on consumer hardware. An agent whose continuity lives on
disk, with a bounded working set in context, does. That is the point of the
architecture: the memory system is what makes an always-on local Engram
feasible, not a convenience layered on the hosted one.

## Cutouts

Every layer has a kill switch, and the stack degrades gracefully to a stock
stateless assistant:

| Env | Effect when `0` |
|-----|-----------------|
| `ENGRAM_BUFFER` | no tier-1 writes at all |
| `ENGRAM_WORKING_MEMORY` | no tier-2 block injection |
| `ENGRAM_EVICT` | buffer kept, but never curated mid-flight (nightly sweep still folds it) |
| `ENGRAM_PERCEPT` | Sensorium runs, but perception leaves no memory |
