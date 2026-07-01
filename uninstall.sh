#!/usr/bin/env bash
# Engram uninstaller — reverses what install.sh + the setup wizard created.
# Safe by design: it shows a plan and asks before removing anything, and it
# NEVER deletes your memory corpus unless you pass --purge-data.
#
# Runs on the system Python (no virtualenv needed), so it still works even if
# .venv is broken or already gone.
#
#   ./uninstall.sh              # interactive; keeps your data
#   ./uninstall.sh --dry-run    # preview only
#   ./uninstall.sh --purge-data # also delete your memory corpus (irreversible)
#   ./uninstall.sh --help       # all options
set -euo pipefail
cd "$(dirname "$0")"   # repo root

PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
  echo "python3 not found — install Python 3 and re-run, or delete the pieces by hand." >&2
  exit 1
fi

exec "$PY" scripts/uninstall.py "$@"
