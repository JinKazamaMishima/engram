#!/usr/bin/env bash
# Engram installer — bootstrap. Ensures Python 3.12+, uv, a virtualenv, and the
# base package, then hands off to the interactive setup wizard which walks you
# through login, your data folder, and which tiers to install.
set -euo pipefail
cd "$(dirname "$0")"   # portable: no GNU-only `readlink -f` (run from the repo root)

b()  { printf '\033[1m%s\033[0m\n' "$*"; }        # bold
dim(){ printf '\033[2m%s\033[0m\n' "$*"; }        # dim
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
die(){ printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

echo
b "▓▒░ Engram — persistent-memory AI assistant ░▒▓"
dim "This bootstrap sets up a Python environment, then launches a guided setup."
echo

# 1) Python 3.12+
command -v python3 >/dev/null 2>&1 || die "python3 not found — install Python 3.12+ and re-run."
PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)' \
  || die "Python $PYV found, but Engram needs 3.12+. Install a newer Python and re-run."
ok "Python $PYV"

# 2) uv (fast Python package manager) — install if missing
if ! command -v uv >/dev/null 2>&1; then
  b "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed."
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv is installed but not on PATH — open a new shell and re-run."
ok "uv $(uv --version | awk '{print $2}')"

# 3) venv + base install (enough to run the wizard; tier extras come later)
b "Creating virtualenv (.venv) and installing the base…"
uv venv >/dev/null
uv pip install -q -e . --no-deps
uv pip install -q rich pyyaml "sqlite-vec>=0.1.6" numpy
ok "base installed"

# 4) hand off to the interactive wizard
echo
b "Launching setup…"
echo
exec uv run python scripts/setup_wizard.py "$@"
