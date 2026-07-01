# recall Telegram bridge — "Engram"

Talk to Engram (me) about the `recall` repo from your phone. A conversational bridge
that long-polls Telegram and drives a persistent `ClaudeSDKClient` with the full
recall project brain (CLAUDE.md + skills + the recall hook) on your Claude
**subscription** — so every message auto-recalls the relevant corpus/soul notes,
exactly like the terminal.

A standalone bridge: its own bot, its own token, its own state — nothing shared.

## Install

```bash
bash scripts/install_telegram_bridge.sh        # 1st run: writes the env stub
# DM @BotFather -> /newbot -> paste the token into ~/.config/recall/telegram-agent.env
# DM @userinfobot -> paste your numeric chat id into the same file
bash scripts/install_telegram_bridge.sh        # 2nd run: validate + start
```

You supply your bot token and numeric chat id in the env stub. The installer
validates the token, installs `claude-agent-sdk` into the recall venv if missing,
and renders the `systemd --user` unit template into `~/.config/systemd/user/`.

## Use

DM the bot. It remembers the conversation across messages. Compose in markdown —
replies render as rich Telegram HTML (bold, `code`, headers, `||spoilers||`).
Send screenshots and it reads them. Built-in commands (no model call):

| cmd | effect |
|---|---|
| `/new` | end the thread, start fresh |
| `/end` | end the thread (next message starts fresh) |
| `/cancel` | interrupt the in-flight turn |
| `/lock` `/unlock` | **kill-switch** — refuse/resume inbound (lost phone) |
| `/status` | locked? busy? active session? |
| `/ping` | health check |

**Effort / model.** The bridge loads `setting_sources=["project"]`, which excludes
your user `~/.claude/settings.json` — so it pins reasoning effort + model itself
(else it would silently fall back to CLI defaults). Defaults to **`max` effort +
`opus[1m]`** (phone-Engram = terminal-Engram). Override per-instance in the env file:
`RECALL_AGENT_EFFORT` (`low|medium|high|xhigh|max`) and `RECALL_AGENT_MODEL`.

## Security model

- **Auth = bot token** (`~/.config/recall/telegram-agent.env`, mode 600, never in
  the repo). **Allowlist = your numeric chat id** — any other sender is silently
  REJECTed and audit-logged.
- Inbound runs with `bypassPermissions` (async chat — no human to approve prompts
  on the road); the chat-id allowlist is the load-bearing safety. The persona also
  asks me to propose consequential/outward actions (soul writes, commits, shell)
  and wait for your explicit OK.
- Billed to the **subscription** (the unit unsets `ANTHROPIC_API_KEY`).
- To revoke: @BotFather → `/revoke` → new token → re-fill the env → restart.

## Control & files

```
systemctl --user status|restart|stop recall-telegram-engram
journalctl --user -u recall-telegram-engram -f                       # daemon log
tail -F ~/.local/share/recall/telegram-agent/messages.log          # IN/OUT/REJECT audit
```

- `infra/telegram/agent_bridge.py` — the daemon (RECALL_* env-parameterized)
- `infra/telegram/recall-telegram-engram.service` — systemd-user unit
- `scripts/install_telegram_bridge.sh` — idempotent installer
- `src/recall/notify.py` — `_md_to_telegram_html` (the markdown→HTML renderer)
- `~/.config/recall/telegram-agent.env` — bot token + chat id (secrets)
- `~/.local/share/recall/telegram-agent/` — runtime state (offset, lock, session, audit, inbox)
