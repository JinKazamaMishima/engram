# Engram

**A persistent-memory AI assistant.** Engram gives an AI agent a real long-term
memory that survives every session — and, if you want, eyes.

It runs on the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
(your Claude Pro/Max subscription or an API key) and is built on **`recall`**, a
machine-local hybrid memory engine that distills your conversations into a durable
knowledge corpus and auto-injects the relevant pieces into every turn — so the
assistant carries context across sessions instead of starting cold each time.

Everything runs and stays on your machine. The corpus, the models, and any
biometric data never leave your computer. The only network calls are to Anthropic
for model inference.

> An *engram* is the physical trace a memory leaves behind — the mark that
> outlasts the moment. That's the idea: give the model something that persists.

---

## What's in the box

Engram is tiered — install only what you want. The memory brain is the core and
runs anywhere; the rest is optional.

| Tier | What it is | Needs |
|------|-----------|-------|
| **Memory brain** (`recall`) | The core engine. A hybrid keyword + vector corpus, nightly curation that distills your Claude Code conversations into notes, FSRS-based memory dynamics (reinforce/decay), temporal validity (superseded facts stay queryable as *historical*, never silently wrong), a nightly "dream" pass that surfaces connections and imagines what-ifs, and weekly reconsolidation. Installs as skills + a retrieval hook into Claude Code. | Python 3.12+ (CPU is fine) |
| **Engram assistant** | A standalone terminal chat app on the Claude Agent SDK, with the memory wired in — plus **continuous memory**: every exchange lands in an append-only conversation buffer, a bounded working-set block is re-derived from it each turn, and cooled history is evicted *into the corpus* instead of being lost (see [the context problem](#the-context-problem-tokens-kv-and-why-long-chats-rot)). Also: file checkpoints + `/rewind`, `/sessions` + `/fork` + `/export`, a live todo + sub-agent panel, plan ↔ act with an interactive approval card, `/context`, recall as in-process MCP tools, and automatic fallback-model rotation, surfaced loudly in the UI. | + a subscription/API key |
| **Sensorium** *(experimental)* | A perceiving loop: a webcam "eye" (face recognition + a local vision model) that recognizes the operator and lets the assistant *see* — and **remember what it sees**: gate-verified perception events persist to their own buffer and evict into the corpus exactly like conversation does. An experimental **ear** (wake-word-gated local Whisper) ships in the tree but stays off by default. | + an NVIDIA GPU + a webcam |
| **Telegram bridge** *(optional)* | Talk to your Engram from your phone; one bot, one authorized chat. | + a bot token |

## How the memory works

- **Corpus is the source of truth.** Markdown notes (YAML frontmatter + body),
  one insight per file. Each project keeps its own under `docs/knowledge/`; a
  shared **global** corpus holds cross-project notes.
- **Index is disposable.** A per-scope SQLite DB (FTS5 keyword + `sqlite-vec`
  semantic search, fused with Reciprocal Rank Fusion) is rebuilt from the markdown.
- **Embeddings are local.** `Qwen3-Embedding-0.6B` (512-dim Matryoshka) +
  `Qwen3-Reranker-0.6B`, loaded lazily on the GPU. Zero external API; the engine
  falls back to keyword-only if no model/daemon is present.
- **Memory is dynamic.** Notes carry an FSRS stability that is reinforced on use
  and decays otherwise, so what you actually rely on stays sharp and the rest fades.
- **Memory dreams.** A nightly offline pass recombines the day's notes into
  quarantined conjectures and imagines *counterfactuals* — single-change "what-ifs"
  over the day's decisions. Nothing a dream invents is treated as fact: a conjecture
  only graduates into the durable corpus once later days corroborate it (a
  counterfactual on a single confirming match; a wrong prediction retires it).
- **Conversation memory is three-tiered.** The assistant's live conversation
  rides a stack: an immutable append-only **buffer** (tier 1, disk, raw), a
  bounded **working-set block** re-derived from that buffer every turn (tier 2,
  in-context), and the curated **corpus** (tier 3, retrieved on demand). History
  that cools out of the working set is *evicted into curation* — distilled into
  notes off the hot path — so a conversation can span days without its context
  rotting. The full design: [docs/continuous-memory.md](docs/continuous-memory.md).
- **Facts carry validity windows.** When a new note supersedes an old one, the
  old note is stamped with the date the fact *stopped being true* and injects as
  `⏳ HISTORICAL` from then on — the system remembers what used to be true
  without ever presenting it as current.
- **Perception becomes memory.** With the Sensorium on, corroboration-verified
  sightings and wake-word utterances append to a day-keyed perception buffer with
  provenance, and evict into the corpus through the same curation path as
  conversation. An hour of an unchanged desk is one row, not four hundred.

## The context problem: tokens, KV, and why long chats rot

Every chat agent faces the same constraint: the model is stateless, so "memory"
is whatever you re-send in the context window — and the context window is the
most expensive real estate in the system. Three costs compound as a
conversation grows:

1. **Linear re-send, quadratic conversation.** Turn *N* pays input tokens for
   everything before it, so an append-only transcript costs O(N) per turn and
   O(N²) cumulatively. A long-running assistant that keeps continuity by
   dragging its whole history gets more expensive *every single turn*.
2. **Attention degrades.** Models demonstrably lose the middle of very long
   contexts; the marginal token you pay the most for is the one the model is
   most likely to ignore.
3. **Compaction drifts.** The standard fix — summarize the transcript in place,
   then later summarize the summary — is *recursive lossy compression*. Each
   pass bakes the previous pass's omissions in as ground truth. What the first
   summary dropped is unrecoverable, and the agent slowly forgets what actually
   happened while sounding confident about it.

Engram's design goal is **continuity without context residency**: the
conversation's history lives on disk, and only two small, high-value
derivatives of it ride the prompt.

| Tier | Where | Cost in tokens |
|------|-------|----------------|
| 1 — **LiveBuffer** | append-only JSONL on disk; every raw exchange, immutable | zero |
| 2 — **working set** | a bounded block **re-derived from tier 1 every turn** — never appended to, never a summary of a summary | small and *constant* |
| 3 — **corpus** | curated notes; the retrieval hook injects only what's relevant to the current prompt | a handful of notes per turn |

The load-bearing rule is *never consolidate a summary — always re-derive from
the immutable buffer*. Tier 2 is recomputed from raw source each turn, so it
cannot drift the way recursive compaction does. And when buffer history cools
out of the working-set window, **eviction is curation**: a detached background
pass distills it into corpus notes and advances a per-conversation watermark.
Leaving the context window and being forgotten stop being the same event.

### What that does to your token bill

- **Per-turn cost stays roughly flat.** The prompt carries the system prompt, a
  bounded working-set block, a few retrieved notes, and the recent turns — not
  an unbounded transcript. The conversation can be a week old; the turn is
  priced like it's an hour old.
- **Retrieval replaces stuffing.** Notes are injected because they match the
  current prompt, not because they might someday be useful. You pay for what
  the turn actually needs.
- **Nothing is pay-to-keep.** In a stock agent, history you stop re-sending is
  history you lose, so you hoard tokens. Here the buffer and corpus hold it for
  free, and it comes back — verbatim from the buffer, or distilled via recall —
  when it's relevant.
- **Starting fresh is cheap and safe.** A `/new` session drops the accumulated
  transcript, and continuity survives it: the working-set block and the
  retrieval hook re-ground the next turn from tiers 1 and 3.

### What it does to the KV-cache

Prompt caching (and on local models, the literal KV-cache) rewards one thing:
a **byte-stable prefix**. Engram's layout is shaped around that:

- **Dynamic content rides the newest message.** The working-set block, the
  recalled notes, and the perception marker are injected into the *current*
  user message — the end of the prompt — so every prior turn stays byte-stable
  and cache-eligible. Anthropic bills cache reads at a fraction of fresh input
  tokens; a stable prefix turns most of each turn's input into cache reads.
- **Compaction is a cache catastrophe; Engram defuses it.** In-place
  summarization rewrites the entire prefix at once — full-price re-ingestion of
  the whole context. Engram banks history *before* that can hurt: a PreCompact
  hook triggers a provisional curation of the buffer, so if the CLI ever does
  compact, nothing is lost — and the cheaper move (a fresh session, re-grounded
  from memory) is always available.
- **On local models, KV is VRAM.** Even with grouped-query attention, an
  8B-class model's KV-cache runs on the order of ~100 KiB per token in fp16 —
  a 100k-token context is ~12 GiB of KV, more than the 4-bit weights
  themselves. A bounded working set means a bounded KV footprint, which is
  what makes an always-on, long-lived agent *feasible* on consumer hardware at
  all. The memory system is the plan for running Engram on a local model, not
  an accessory to the hosted one.

The full architecture walk-through — tiers, eviction mechanics, temporal
validity, and the perception path — is in
[docs/continuous-memory.md](docs/continuous-memory.md).

## Intended use

Engram is built to run on a machine that **stays on** — a home server, a
workstation you leave running, a NAS. Its memory work runs on a schedule, not on
demand, so the machine needs to be awake when the timers fire (on a headless box,
set `loginctl enable-linger <user>` so the `--user` timers run while you're logged
out). If the machine is off at the scheduled hour the run is simply deferred, not
lost — but the more it's on, the fresher its memory.

### When curation runs

There are two paths, and together they guarantee every conversation is eventually
distilled into memory:

- **Live, per-conversation — Telegram bridge only.** When you `/end` a chat (or
  `/new` to start a fresh one), that just-ended conversation is curated in the
  background right away — fire-and-forget, so you're never blocked — and its memory
  is fresh the same day.
- **Nightly sweep — everything else.** A systemd `--user` timer fires at **23:15
  local** and runs the whole nightly cycle: `curate-sessions-all` walks every one of
  the day's conversations across every registered project and curates each that
  wasn't already curated live; then `consolidate-all` folds what you actually used
  into memory strength; then `dream-all` recombines the day and imagines its
  what-ifs. A second timer runs weekly reconsolidation **Monday 02:00**. Both are
  `Persistent=true`, so a run missed while the machine slept is caught up on next
  boot.

Terminal and Claude Code sessions are **not** curated the instant you close them —
they're picked up by the nightly sweep. Only the Telegram bridge curates live.

### Multiple conversations at once

The unit of curation is a single conversation — one session transcript, tracked by
its own id — so you can run as many in parallel as you like (several terminals, the
bridge, the TUI) and each becomes an independent memory unit:

- **Idempotent per session.** Each session id is recorded once it's curated (a
  `sessions` ledger in `curated.json`). A session curated live is skipped by the
  nightly sweep, and vice-versa — no conversation is curated twice on a later pass.
- **Serialized in normal use.** The nightly sweep curates sessions **one at a time
  in a single process**, and the bridge's one authorized chat ends conversations one
  at a time — so curation runs don't write the corpus concurrently. You don't
  coordinate anything.
- **Collisions self-heal.** In the one narrow window where a live curation is still
  finishing exactly as the nightly sweep begins, the worst case is a single
  conversation curated twice — which the curator's dedup and the weekly
  reconsolidation collapse. Nothing is lost or corrupted.
- **One rule:** let the timers and the bridge drive curation. Don't hand-run
  `recall curate` / `curate-all` / `curate-sessions-all` on top of a run already in
  flight — concurrent writers to the same corpus aren't mutex-guarded. If you script
  your own curation, serialize it.

## Quickstart

```sh
git clone https://github.com/JinKazamaMishima/engram.git
cd engram
./install.sh
```

The guided installer walks you through, step by step:

1. **Preflight** — checks Python, and detects GPU vs CPU.
2. **Log in to Anthropic** — installs the `claude` CLI if needed and runs the
   login (subscription browser sign-in *or* an API key).
3. **Choose your data folder** — where *your* memory corpus, indices, sessions,
   and any enrolled face data live (default `~/.local/share/recall`). This never
   leaves your machine.
4. **Pick your tiers** — memory brain / + assistant / + Sensorium / + Telegram.
5. **Models, skills, hook, and (optional) background services** — each explained,
   each skippable.

### Manual install (memory brain only)

```sh
uv venv
uv pip install -e ".[dev]"            # add ",engram" for the TUI, ",telegram" for the bridge
uv run python scripts/install_skills.py   # /recall etc. → ~/.claude/skills/
uv run python scripts/install_hook.py     # retrieval hook → ~/.claude/settings.json
uv run recall build --global              # build the global index
```

## Usage

```sh
recall build                 # index the current project's docs/knowledge
recall query "why did X?"    # hybrid search
recall paths                 # show resolved corpus/index paths
recall register              # register the current project for nightly curation

engram                       # launch the terminal assistant (needs the ,engram extra)
engram -p                    # …with the Sensorium (camera) on
```

> `recall` is a console command inside the repo's virtualenv. The installer can
> link it onto your `PATH` (`~/.local/bin`); otherwise run it as `uv run recall …`
> from the repo, or call `.venv/bin/recall` directly.

## Uninstall

```sh
./uninstall.sh              # remove Engram — KEEPS your memory corpus
./uninstall.sh --dry-run    # preview exactly what would be removed
./uninstall.sh --purge-data # also delete your corpus (RECALL_DATA_ROOT) — irreversible
```

Reverses the install footprint — the `.venv`, skills, the Claude Code hook,
systemd user units, the `recall` launcher, the shell-rc lines, and config. It
runs on the system Python (no venv needed), shows a plan and confirms first, and
**never deletes your memory corpus** unless you pass `--purge-data`. `uv`, the
`claude` CLI, and Python are left installed; delete the repo folder by hand to
finish.

## Requirements

- **A Claude Pro/Max subscription** (recommended — Engram bills to it, never an
  API key, by default) **or** an `ANTHROPIC_API_KEY`.
- **Python 3.12+.**
- **Linux** is the primary platform; the memory brain and assistant also run on
  **WSL2** and **macOS** (see [Platform support](#platform-support) below).
- **Optional:** an NVIDIA GPU (semantic embeddings + the vision model) and a
  USB webcam (Sensorium).

## Platform support

| | Linux | WSL2 | macOS |
|---|:-:|:-:|:-:|
| Memory brain + assistant | ✓ | ✓ | ✓ |
| Background services (systemd) | ✓ | ✓ *(`systemd=true`)* | ✗ *(use launchd/cron)* |
| Sensorium (webcam) | ✓ | ✗ | ✗ |

**Linux** is first-class. The memory engine and the terminal assistant run
anywhere Python 3.12+ does — macOS included (clipboard uses `pbcopy`; keyword
search works with no GPU). The **systemd** timers are Linux-only; on macOS the
installer detects that and skips them, so schedule `recall curate-all` via
`launchd`/`cron` instead. The **Sensorium**'s camera capture uses V4L2 (Linux) —
macOS and WSL2 aren't supported yet (macOS needs an AVFoundation backend; WSL2
doesn't expose USB webcams without `usbipd`).

## Privacy

Engram is local-first by design. Your knowledge corpus, indices, conversation
history, and any enrolled face data live under your chosen data directory
(default `~/.local/share/recall`) and **never leave your machine**. There is no
telemetry. `.gitignore` is configured so generated data, model weights, secrets,
and biometric files can't be committed by accident.

## Configuration

Common environment variables (all optional):

| Var | Meaning |
|-----|---------|
| `RECALL_DATA_ROOT` | Where your corpus/indices/sessions live. Default `~/.local/share/recall`. |
| `ENGRAM_PERSONA_FILE` | Path to a text file that overrides the assistant's default persona. |
| `ENGRAM_MODEL` / `ENGRAM_EFFORT` | Model id and reasoning effort for the assistant. |
| `ENGRAM_FALLBACK_MODEL` | Fallback for overload rotation (default `claude-opus-4-8`; empty disables). |
| `ENGRAM_USER` | Name the perceiving loop greets / the face-ID gate looks for. |
| `ENGRAM_WORKING_MEMORY` / `ENGRAM_EVICT` / `ENGRAM_BUFFER` | Continuous-memory cutouts — set to `0` to disable the working-set block, eviction, or the buffer entirely. |
| `ENGRAM_WM_TURNS` | Size of the hot working-set window in turns (default 12). |
| `ENGRAM_PERCEPT` | Set to `0` to keep perception events out of memory (Sensorium still runs). |
| `CLAUDE_BIN` | Path to the `claude` CLI (default: found on `PATH`). |

## Layout

```
src/recall/          the memory engine (index, curate, dream, consolidate, cli)
infra/engram/        the terminal assistant (core.py + Textual app.py, buffer.py +
                     working_set.py = continuous memory) + Sensorium (eye/, perceive/;
                     perceive/percept.py = perception memory)
infra/telegram/      the optional Telegram bridge
infra/systemd/       user-level service/timer templates (rendered by the installer)
skills/              /recall, /curate-memory, /dream, /reconsolidate-memory
scripts/             install helpers (skills, hook, systemd) + the CI leak scanner
docs/                architecture notes (continuous-memory.md, dynamic-memory.md)
```

## Development

```sh
uv run pytest                # unit tests use an injected fake embedder — no model download
uv run ruff check .
```

## Contributing

Pull requests welcome. `main` is protected: changes land via PR with review, and
CI runs the test suite **plus a privacy leak-scan** on every PR — so no
contribution can reintroduce personal data or break the tests.

## License

[MIT](LICENSE).
