#!/usr/bin/env bash
# Privacy / secret hygiene scan. Fails (exit 1) if the tree contains absolute
# home paths, email addresses, or obvious secrets — the things that most often
# leak personal data into a public repo. Runs locally and in CI on every PR.
#
# It deliberately scans for GENERIC patterns, not a list of specific names:
# a public repo can't hard-code the very identifiers it's trying to keep out.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

EXCLUDES=(--exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__
          --exclude-dir=.ruff_cache --exclude-dir=.pytest_cache
          --exclude-dir=node_modules --exclude-dir=site
          --exclude=leak_scan.sh --exclude=uv.lock --exclude=*.lock)

# pattern | human description
CHECKS=(
  '/home/[A-Za-z0-9_.-]+/|absolute Linux home path'
  '/Users/[A-Za-z0-9_.-]+/|absolute macOS home path'
  'sk-ant-[A-Za-z0-9-]{16,}|Anthropic API key'
  'gh[pousr]_[A-Za-z0-9]{20,}|GitHub token'
  'github_pat_[A-Za-z0-9_]{20,}|GitHub fine-grained token'
  'AKIA[0-9A-Z]{16}|AWS access key'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----|private key'
  '[A-Za-z0-9._%+-]+@(?!example\.com|anthropic\.com|users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}|email address'
)

hits=0
for entry in "${CHECKS[@]}"; do
  pat="${entry%%|*}"; desc="${entry##*|}"
  if out=$(grep -rInEP "${EXCLUDES[@]}" "$pat" . 2>/dev/null); then
    echo "✗ found ${desc}:"
    echo "$out"
    echo
    hits=1
  fi
done

if [ "$hits" -ne 0 ]; then
  echo "Leak scan FAILED — remove the personal paths / emails / secrets above." >&2
  exit 1
fi
echo "✓ leak scan clean"
