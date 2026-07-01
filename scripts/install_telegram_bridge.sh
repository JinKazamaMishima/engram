#!/usr/bin/env bash
# Idempotent installer for the recall Telegram conversational bridge ("Engram").
#
#   1st run (no env yet): writes ~/.config/recall/telegram-agent.env (mode 600)
#     and tells you to create a bot via @BotFather and paste its token + your
#     numeric chat id.
#   2nd run (token + chat id filled): validates the token, ensures the SDK is
#     installed, enables linger, RENDERS the systemd-user unit (substituting the
#     placeholders in the template), and starts the service.
#
# Re-runnable any time. Nothing here is destructive.
set -euo pipefail

# repo root = this script's dir's parent, so the repo can live anywhere
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO/.venv/bin/python"
ENV_DIR="$HOME/.config/recall"
ENV_FILE="$ENV_DIR/telegram-agent.env"
UNIT_SRC="$REPO/infra/telegram/recall-telegram-engram.service"
UNIT_DEST="$HOME/.config/systemd/user/recall-telegram-engram.service"

mkdir -p "$ENV_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  umask 077
  cat > "$ENV_FILE" <<EOF
# recall Telegram bridge (Engram) — secrets, mode 600, NOT in the repo.
#   1) Create a bot: DM @BotFather -> /newbot -> copy the token it returns.
#   2) Paste the token below (it IS the auth — treat it as a secret).
#   3) Set your numeric chat id (DM @userinfobot to get it) as the allowlist.
RECALL_TELEGRAM_AGENT_TOKEN=
RECALL_TELEGRAM_AGENT_CHAT_ID=
EOF
  chmod 600 "$ENV_FILE"
  echo "created $ENV_FILE (mode 600)."
  cat <<EOF

NEXT:
  1. DM @BotFather on Telegram -> /newbot -> follow the prompts -> copy the token.
  2. DM @userinfobot to get your numeric chat id.
  3. Paste both into $ENV_FILE
     (RECALL_TELEGRAM_AGENT_TOKEN=...  and  RECALL_TELEGRAM_AGENT_CHAT_ID=...).
  4. Re-run:  bash scripts/install_telegram_bridge.sh
EOF
  exit 0
fi

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
TOKEN="${RECALL_TELEGRAM_AGENT_TOKEN:-}"
CHAT="${RECALL_TELEGRAM_AGENT_CHAT_ID:-}"
if [[ -z "$TOKEN" || -z "$CHAT" ]]; then
  echo "ERROR: set RECALL_TELEGRAM_AGENT_TOKEN and _CHAT_ID in $ENV_FILE first." >&2
  exit 1
fi

echo "validating bot token via getMe..."
RESP="$(curl -s --max-time 15 "https://api.telegram.org/bot${TOKEN}/getMe" || true)"
if ! grep -q '"ok":true' <<<"$RESP"; then
  echo "ERROR: Telegram rejected the token (getMe failed): ${RESP:0:200}" >&2
  exit 1
fi
USERNAME="$(grep -oE '"username":"[^"]+"' <<<"$RESP" | head -1 | cut -d'"' -f4 || true)"
echo "  token OK -> @${USERNAME}"

if ! "$PYTHON" -c 'import claude_agent_sdk' 2>/dev/null; then
  echo "installing claude-agent-sdk into the recall venv..."
  "$REPO/.venv/bin/pip" install -q 'claude-agent-sdk>=0.2,<0.3'
fi

loginctl enable-linger "$(id -un)" >/dev/null 2>&1 || true
mkdir -p "$HOME/.config/systemd/user"
# render the unit template (substitute placeholders) into ~/.config/systemd/user/
sed -e "s|__ENGRAM_HOME__|$REPO|g" \
    -e "s|__PYTHON__|$PYTHON|g" \
    "$UNIT_SRC" > "$UNIT_DEST"
systemctl --user daemon-reload
systemctl --user enable --now recall-telegram-engram.service

cat <<EOF

✅ recall-telegram-engram is running. DM @${USERNAME} and say hi to Engram.
   status: systemctl --user status recall-telegram-engram
   logs:   journalctl --user -u recall-telegram-engram -f
   audit:  tail -F ~/.local/share/recall/telegram-agent/messages.log
   stop:   systemctl --user stop recall-telegram-engram   (/lock from the chat pauses it)
EOF
