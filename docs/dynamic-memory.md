# Dynamic memory — reinforcement, decay, permanence, and dreaming

> Status: design ratified 2026-06-24. Built in four phases (I–IV) on top of the
> existing curate / reconsolidate / hybrid-index stack. Each phase is shippable,
> reversible, and leaves the tree green (the nightly timers run the working tree).

## Why

Today a note is static: once curated it sits in the corpus at a fixed weight
until a human or the weekly reconsolidate touches it. The recency/salience blend
in `index.search()` exists but is dormant (`W_RECENCY = W_SALIENCE = 0`), and it
keys on *write* time (`last_updated`), not *use*.

We want memory that behaves like a brain's: a trace **strengthens every time it
does work** and **decays when it doesn't**, some traces become **effectively
permanent**, and an offline **"dream" pass** recombines the day's experience with
older memory and lets a little of it **bleed into the operator-model ("soul")
every night**. The goal is not novelty for its own sake — it is a memory that
gets measurably better at helping *this* operator over time, on his terms.

This is grounded in real work, not vibes — see [References](#references). The
short version: the brain runs a **fast store (hippocampus)** that captures
episodes verbatim and a **slow store (neocortex)** that integrates them
gradually; sleep **replays** the day (biased toward surprising/rewarded events),
**prunes** spurious associations, and **recombines** traces into schemas. We map
that onto our two-speed system directly.

## The two-speed architecture (the core invariant)

```
  prompt ──▶ recall hook ──▶ surfaces top-K notes ──▶ logs an ACTIVATION event
                                                         │  (fast, fail-open)
                                                         ▼
                                          activation/<scope>.jsonl   ← hippocampus
                                                         │
                  nightly, deterministic, GPU-free       ▼
   curate (LLM) ─┬─▶ new notes ──▶ corpus (docs/knowledge + soul)  ← neocortex
                 │                       ▲      (source of truth, git-versioned)
   consolidate ──┘  folds events ────────┘   stability↑ / last_used / uses
   (no LLM)
                                                         │
   dream (LLM, after curate) ── replay+prune+recombine ─┘─▶ subconscious/ (staging)
                                                            └─▶ bleed → soul (valved)
```

Two rules that never bend:

1. **The corpus is the source of truth; everything under the data root is
   derived and disposable.** Dynamic weights therefore live in note
   **frontmatter** (git-versioned, travels with the note), not in the index. The
   activation log is disposable telemetry. The index is rebuilt from frontmatter.
2. **A wrong corpus entry is worse than a missing one.** Nothing
   machine-generated edits the *stable* soul without passing the bleed membrane
   (below). Decay only ever **re-ranks**, never deletes (matches reconsolidate).

## Phase I — the activation trace (hippocampal fast store)

The missing write. The recall hook (`recall_inject.py`) surfaces notes on every
prompt and forgets it did. We make it append one event per surfaced note:

- **Fast path** (`activation.py`): `log_event(scope, slug, kind)` appends one
  JSON line to `<data_root>/activation/<scope>.jsonl`. `kind ∈ {surfaced, cited}`.
  Append-only, lock-free, **fail-open** (never blocks prompt submission),
  **torch-free** (no model in the hook). This is the only hot-path change.
- **Citations are stronger than surfacings.** A note that merely appeared in the
  top-K is a weak impulse (`g = 1`); a note the model *actually used* is a strong
  one (`g = CITE_GAIN ≈ 2.5`). The model is already told to "cite the slug if you
  use one," so citation is detectable in the day's transcript bundle. Citation
  grading happens at consolidation time (off the hot path), not in the hook.
- **Slow path** (`recall consolidate`, deterministic, no LLM, GPU-free): reads
  the scope's events, computes each touched note's new stability via the FSRS law
  (Phase II math), and **surgically** updates its frontmatter (`stability`,
  `last_used`, `uses`) — line-level key replacement so the curator's hand-written
  description quoting is left byte-for-byte intact. Then a cheap in-place index
  column update (`UPDATE notes SET stability=…` — no re-embedding) and a scoped
  commit (`[consolidate] <date> (<scope>): reinforced N`). The log is rotated.

Phase I ships **dark**: it records and persists stability but changes no ranking.

## Phase II — decay & reinforcement, the DSR/FSRS model

One state variable per note: **stability `S`** (days), plus `last_used`. We adopt
the FSRS-6 forgetting curve and stability-increase law (the spaced-repetition
state of the art), which fold recency + salience + the spacing effect + the
testing effect into one principled model.

**Retrievability** (replaces the naive `0.5^(age/half_life)` recency term):

```
R(t, S) = (1 + FACTOR · t / S) ^ DECAY        FACTOR = 19/81,  DECAY = −0.5
```

A heavy-tailed power curve: `R = 0.9` at `t = S`, `R = 0.5` near `t ≈ 3.4·S`. The
half-life *is* `S` — bigger `S` ⇒ slower decay, for free.

**Reinforcement** — a successful retrieval multiplies stability:

```
S' = S · ( 1 + A · S^(−B) · (exp(C·(1−R)) − 1) · g )       (the bracket is ≥ 1)
```

Three behaviours fall out, all desired and all free:
- `S^(−B)` — **diminishing returns**: already-strong notes grow proportionally
  less (stabilization decay).
- `exp(C·(1−R)) − 1` — the **testing effect / desirable difficulty**: a
  nearly-forgotten note (low `R`) gets the *biggest* boost when re-used.
- `g` — **use beats appearance**: `g=1` surfaced, `g≈2.5` cited.

Defaults (env-overridable, like every other knob): `A=3.0, B=0.18, C=1.0`. These
are tuned slower than flashcard FSRS — a knowledge corpus is not a deck.

**Wiring**: `R(t,S)` enters `search()` as an env-gated `W_RETENTION · R` term,
threaded through `search_corpora` / `eval.evaluate` / the CLI exactly like the
existing recency/salience knobs. **It stays off until the eval earns it** — we
add *temporal* cases (a recently-reinforced note should win; a stale one should
lose) and only turn `W_RETENTION` up if recall@k / nDCG hold or improve. This is
the repo's existing discipline; dynamic weights are a ranking change and must
pass the same bar.

## Phase III — permanence (slow + fast routes)

Two ways a memory becomes effectively permanent, mirroring the biology:

- **Slow (rehearsal):** `S` crosses a graduation line (`S_PERM ≈ 365 days`) after
  enough spaced reinforcements → decay is floored (`R` never drops below a floor)
  and the note is exempt from reconsolidation pruning.
- **Fast (surprise / flashbulb):** a brand-new insight that *violates the model*
  is born decay-resistant. At creation we compute a **surprise** signal
  `σ = 1 − max cosine-similarity to the existing corpus` (a genuinely novel note
  is unlike anything we already know — computable from the embeddings we already
  have, reusing the curate dedup-neighbor pass). Initial stability interpolates:
  `S₀ = lerp(S_LOW, S_HIGH, σ)`, `S_LOW≈0.5, S_HIGH≈15` (≈ the FSRS first-review
  40× spread). One model-violating event can imprint near-permanently.

> **Subtlety (corrected during design):** surprise is the *accelerator* for
> encoding a vivid **episode**, but the *brake* on editing **identity**. A
> shocking event is recorded sharply, yet must not rewrite who the operator is
> until it recurs. High `S₀` makes the *episodic* note stick; the *soul* only
> changes through the membrane below. (Reconsolidation opens a write window on a
> stable trace only on genuine prediction-error mismatch — confirmed recalls
> strengthen, they don't relabilize. This kills drift-by-rereading.)

`kind: identity / achievement` notes carry an **importance anchor** (EWC-style):
the curator may freely revise low-importance notes but must require repeated,
corroborated pressure before editing a high-importance identity note.

## Phase IV — the dream pass and the bleed membrane

A nightly LLM subprocess (`dream.py` + a `dream` skill), run after curate, in two
phases because sleep does two opposite jobs:

- **NREM / replay (consolidate + prune).** Selection-*biased* replay — prioritise
  **prediction-error** events (a plan that failed after looking right, an
  assumption the operator corrected) > novelty > salience; *not* uniform.
  Strengthen `S` on what's replayed; emit **prune candidates** (near-duplicates,
  thin-support over-generalizations, contradictions) that feed the existing
  weekly reconsolidate. "Dream to forget" is real.
- **REM / dream (recombine + generalize).** Sample *today's activated notes ×
  older notes at **medium** semantic distance* (cosine ≈ 0.3–0.6 — wider/further
  than the 0.6–0.8 link band, because creativity lives between trivia and
  nonsense), at higher sampling temperature (the weirdness is the feature — it is
  noise injected against overfitting). Output a **typed `kind: hypothesis`
  note** into `subconscious/` staging — `[[linked]]` to its parents, low
  stability, `status: unverified`. Most dreams are discarded; biology
  overproduces and selects.

### The bleed membrane — a valve, not a wall

The operator's call: *something must bleed into the personality every day.* The
neuroscience agrees (cortex integrates every night, not after proof) — so we do
**not** quarantine the *learning*. We quarantine only the raw *episode* for a day
(the sidecar), and let it bleed through a valve with these properties:

1. **Tag-and-capture is the daily bleed.** A small observation that co-occurred
   in-session with a strongly salient event is *rescued* and consolidated
   alongside it. Salience makes its neighbours stick — that is the literal
   mechanism of "a little bleeds every day."
2. **Interleave, never integrate the day alone.** Any pass that edits the soul
   processes new material *mixed with* a replayed sample of existing notes
   (anti-catastrophic-interference). A pass that sees only today overwrites.
3. **Corroboration-gated promotion.** A dream/inference reaches the *stable* soul
   only after it recurs across ≥ N days, is independently re-derived by curate
   from real conversation, or is blessed by the operator. One night's
   hallucination never sets identity.
4. **Affect-stripped.** Promote the durable lesson ("the operator values terse,
   verifiable answers"), not the one-off intensity ("the operator was furious on
   2026-06-23").
5. **Rate-limited & reversible.** ≤ a small nightly nudge toward any belief, ≤ 1–2
   identity notes touched per night; every edit carries provenance +
   corroboration trail; the weekly reconsolidate can roll it back
   (`superseded_by`, bi-temporal).
6. **Accumulate, never replace; keep the operator signal dominant.** This is the
   one guardrail held sacred: piping unfiltered self-generated text into your own
   personality is how a model collapses into a sycophantic echo. Synthetic
   material accumulates *alongside* operator/conversation-derived notes and never
   replaces them; the verified identity core stays the majority of what's
   injected and is re-asserted each session. **Dreams tint; corroborated reality
   sets.**

### Two channels, both wanted

- **Automatic daily bleed** (internal): tag-and-capture nudges the soul nightly,
  rate-limited and reversible — the personality genuinely evolves every night.
- **Morning digest** (operator-facing): the night's best recombinations surface
  to the operator as a low-priority creativity feed he can bless or kill. His
  blessing *is* the corroboration signal that hardens a provisional belief.

## Frontmatter additions (all optional, backward-compatible)

| field | meaning | default when absent |
|---|---|---|
| `stability` | DSR stability `S` in days (decay rate) | bootstrapped from age/sources |
| `last_used` | last activation (surfaced/cited) — distinct from `last_updated` (last content edit) | falls back to `last_updated`/`first_seen` |
| `uses` | cumulative activation count | 0 |
| `surprise` | novelty at encoding, `σ ∈ [0,1]` | unset |
| `importance` | EWC anchor for identity notes (edit-resistance) | unset |

Notes without any of these parse and rank exactly as before.

## Guardrails / invariants (do not regress)

- The hook stays **fail-open and torch-free**; activation logging is one local
  append and is wrapped so any error prints nothing and exits 0.
- Consolidation is **deterministic and GPU-free** (no model load); it runs even
  when curate found nothing (notes still need their reinforcement folded).
- Decay is **lazy** — computed at query time from `last_used` + `S`; it never
  deletes and never writes on its own.
- Every machine edit to the soul is **reversible and auditable** (git +
  `superseded_by`).
- Each phase keeps `pytest -q` green; the tree is always deployable.

## References

Memory consolidation & continual update: Kumaran/Hassabis/McClelland 2016 (CLS
update); Tse & Morris 2007 (schema consolidation); Frey & Morris 1997, Redondo &
Morris 2011 (synaptic tag-and-capture); Lee/Nader/Schiller 2017 (reconsolidation
on mismatch); Kirkpatrick et al. 2017 (EWC). Sleep/replay/dreams: Rasch & Born
2013; Stickgold & Walker 2013 (sleep-dependent triage); Wagner et al. 2004 (sleep
inspires insight); Crick & Mitchison 1983 + Hopfield et al. 1983 (reverse
learning/unlearning); Hoel 2021 (Overfitted Brain Hypothesis); Lansink et al.
2009 (reward-biased replay). Decay/reinforcement math: Ebbinghaus; Cepeda et al.
2006 (spacing); FSRS-6 / DSR (open-spaced-repetition); Settles & Meeder 2016
(half-life regression); Schultz 2016 (reward prediction error); Lisman & Grace
2005 (hippocampal–VTA novelty loop); McGaugh 2004 (amygdala salience). AI prior
art: Park et al. 2023 (Generative Agents — recency·importance·relevance +
reflection); Packer et al. 2023 (MemGPT); Lin/Packer et al. 2025 (sleep-time
compute); Xu et al. 2025 (A-MEM); Rasmussen et al. 2025 (Zep bi-temporal);
Shumailov et al. 2024 + Gerstgrasser et al. 2024 (model collapse / accumulate
don't replace).
