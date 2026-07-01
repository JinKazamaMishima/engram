#!/usr/bin/env bash
# Wire recall's systemd --user services. Run AS YOURSELF (not root):
#   bash scripts/install_systemd.sh
#
# These unit files under infra/systemd/ are TEMPLATES containing placeholders
# (__ENGRAM_HOME__, __PYTHON__, __USER__). This script detects the right values
# for THIS machine and RENDERS each template (substitutes the placeholders) into
# ~/.config/systemd/user/ — so the repo can live anywhere under any user.
#
# Does three things, all safe and reversible:
#   1. Brings up recall-embedder on :8973 (Qwen3-Embedding-0.6B + a warm /rerank
#      endpoint). Recall is fail-open, so it can't break anything if it's down.
#   2. Enables the nightly recall-curate.timer (23:15 local): `recall curate-all
#      --commit` over the registered projects.
#   3. Enables the weekly recall-reconsolidate.timer (Mon 02:00): `recall
#      reconsolidate-all --commit` (merge dups, add links) over global + projects.
#
# Undo: systemctl --user disable --now recall-embedder recall-curate.timer \
#         recall-reconsolidate.timer
#       rm -f ~/.config/systemd/user/recall-embedder.service \
#             ~/.config/systemd/user/recall-curate.{service,timer} \
#             ~/.config/systemd/user/recall-reconsolidate.{service,timer}
set -euo pipefail

# --- detect this machine's values -------------------------------------------
ENGRAM_HOME="$(cd "$(dirname "$0")/.." && pwd)"   # repo root = this script's dir's parent
USER_NAME="$(id -un)"
PYTHON="$ENGRAM_HOME/.venv/bin/python"

UD="$HOME/.config/systemd/user"
SRC="$ENGRAM_HOME/infra/systemd"
mkdir -p "$UD"

# render one template: substitute placeholders and write into ~/.config/systemd/user/
render() {
  local name="$1"
  sed -e "s|__ENGRAM_HOME__|$ENGRAM_HOME|g" \
      -e "s|__PYTHON__|$PYTHON|g" \
      -e "s|__USER__|$USER_NAME|g" \
      "$SRC/$name" > "$UD/$name"
}

for unit in \
  recall-embedder.service \
  recall-curate.service \
  recall-curate.timer \
  recall-reconsolidate.service \
  recall-reconsolidate.timer ; do
  render "$unit"
done
systemctl --user daemon-reload

# 1. embedder
systemctl --user enable --now recall-embedder

# 2. nightly curation timer + weekly reconsolidation timer
systemctl --user enable --now recall-curate.timer
systemctl --user enable --now recall-reconsolidate.timer

# survive logout / run headless
loginctl enable-linger "$USER_NAME" 2>/dev/null || true

echo "--- status ---"
echo -n "recall-embedder:    "; systemctl --user is-active recall-embedder || true
echo -n "recall-curate.timer: "; systemctl --user is-active recall-curate.timer || true
echo -n "recall-reconsolidate.timer: "; systemctl --user is-active recall-reconsolidate.timer || true
echo -n "healthz: "; curl -s --max-time 3 http://127.0.0.1:8973/healthz || echo "(not ready yet — give the model a few seconds)"
echo
echo "next nightly curate run:"; systemctl --user list-timers recall-curate.timer --no-pager 2>/dev/null | sed -n 2p || true
