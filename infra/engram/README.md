# Engram — a self-hosted terminal harness

A terminal assistant ("Engram") on its own harness, running on your Claude
Pro/Max **subscription** (never an API key) via the Claude Agent SDK. Built so the
model backend is swappable: when a local self-hosted model exists, a new driver
drops in and this front-end is unchanged.

## Run

```bash
./infra/engram/engram                 # the full Textual TUI — the home
./infra/engram/engram --simple        # lightweight rich REPL (tui.py)
./infra/engram/engram --once "…"      # single-shot, print reply, exit (scripts/tests)
```

Optional global command:

```bash
# from the repo root:
ln -sf "$PWD/infra/engram/engram" ~/.local/bin/engram   # then just: engram
```

In-session (type `/` for the dropdown): `/new` fresh thread · `/context`
context-window usage · `/agent <name> <task>` delegate to a sub-agent · `/effort` ·
`/model` · `/paste` image · `/status` · `/exit`. Keys: `ctrl+n` new · `ctrl+v`
paste · `ctrl+y` copy reply · `ctrl+c` quit.

## Look

A bespoke **`engram` theme** (in `app.py`): Engram is a blue-white star in Lyra, so
the home wears a deep night-sky palette — starlight (blue-white) text, one
luminous-cyan accent used sparingly (the star's glint), soft violet for quiet
notes. Restraint over decoration: generous spacing, a single accent stripe on
your turns. Edit `ENGRAM_THEME` / the `CSS` to taste.

## How it works

- **`core.py`** — the `ModelDriver` *seam* + `AgentSDKDriver` (the only driver
  today). It strips `ANTHROPIC_API_KEY` and drives the logged-in `claude` CLI, so
  every turn bills to the subscription. Runs with `setting_sources=["project"]`
  and `cwd`=the repo root, so the **recall hook auto-injects memory** and the
  skills load — exactly like the terminal / Telegram Engram.
- **`app.py`** — the full Textual TUI (the home): scrollable markdown conversation,
  live streaming + tool activity, a status line, the bespoke `engram` theme.
- **`tui.py`** — a lightweight `rich` REPL + `--once` single-shot, for scripting,
  tests, and a no-frills fallback.

## Sub-agents & context

- **`/context`** shows the live context-window breakdown (used/max tokens, %, a
  per-category table) — the same data Claude Code's `/context` shows, via the SDK's
  `get_context_usage()`. Read-only; gated while a reply is streaming.
- **Sub-agents** — Engram ships a small roster (`SUBAGENT_DEFS` in `core.py`: `Explore`,
  `Plan`, `general-purpose`) passed to the SDK as `agents=` (an SDK-driven session does
  **not** expose Claude Code's built-in agent types by default, so we define them).
  - **Auto** — the model delegates via the CLI's `Agent` tool, which runs sub-agents
    **async**: the parent turn's `ResultMessage` arrives *before* the sub-agent finishes.
    So `_stream` keeps reading the message stream past that result while a task is pending
    — surfacing the sub-agent's live progress (the `Task*` messages) and capturing the
    main agent's final synthesis, all in one turn. It stops on a `ResultMessage` with no
    task pending; a sub-agent that goes silent is detached after
    `ENGRAM_SUBAGENT_IDLE_TIMEOUT`s. `parent_tool_use_id` keeps the sub-agent's own
    monologue out of the reply (only markers + progress + the synthesis show).
  - **Explicit** — `/agent <name> <task>` runs one as an isolated **synchronous** sub-query
    (`run_subagent`): a fresh, in-turn delegation whose result streams straight back.

## The seam (why this is the path to local)

Every front-end talks to a `ModelDriver`. Today: `AgentSDKDriver` (subscription).
Tomorrow: a `LocalModelDriver` (OpenAI-compatible) pointing at the self-hosted
1.5–3T model — same interface, no front-end change. The durable, hard layers
(this TUI, the perceiving loop, memory wiring, persona) are model-agnostic.

## Current limits (MVP — follow-ups)

- Session is in-memory (a fresh thread per launch; `/new` resets within a run).
  *Follow-up:* persist + resume the last thread across launches.
- `permission_mode=bypassPermissions` (matches the Telegram bridge; the persona's
  "propose consequential actions first" is the guardrail). *Follow-up:* interactive
  permission prompts now that a human is present.
- Shares the `AgentSDKDriver` logic with `infra/telegram/agent_bridge.py` by
  duplication. *Follow-up:* refactor the bridge onto this core so there's one Engram.
