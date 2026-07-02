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
| **Memory brain** (`recall`) | The core engine. A hybrid keyword + vector corpus, nightly curation that distills your Claude Code conversations into notes, FSRS-based memory dynamics (reinforce/decay), a nightly "dream" pass that surfaces connections and imagines what-ifs, and weekly reconsolidation. Installs as skills + a retrieval hook into Claude Code. | Python 3.12+ (CPU is fine) |
| **Engram assistant** | A standalone terminal chat app on the Claude Agent SDK, with the memory wired in — plus `/context`, sub-agent delegation, and a plan ↔ act mode toggle. | + a subscription/API key |
| **Sensorium** *(experimental)* | A perceiving loop: a webcam "eye" (face recognition + a local vision model) that recognizes the operator and lets the assistant *see*. | + an NVIDIA GPU + a webcam |
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
| `ENGRAM_USER` | Name the perceiving loop greets / the face-ID gate looks for. |
| `CLAUDE_BIN` | Path to the `claude` CLI (default: found on `PATH`). |

## Layout

```
src/recall/          the memory engine (index, curate, dream, consolidate, cli)
infra/engram/        the terminal assistant (core.py + Textual app) + Sensorium (eye/, perceive/)
infra/telegram/      the optional Telegram bridge
infra/systemd/       user-level service/timer templates (rendered by the installer)
skills/              /recall, /curate-memory, /dream, /reconsolidate-memory
scripts/             install helpers (skills, hook, systemd)
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
