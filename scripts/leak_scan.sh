#!/usr/bin/env bash
# Privacy / secret hygiene scan. Fails (exit 1) if the tree contains absolute
# home paths, email addresses, or obvious secrets — the things that most often
# leak personal data into a public repo. Runs locally and in CI on every PR.
#
# It deliberately scans for GENERIC patterns, not a list of specific names:
# a public repo can't hard-code the very identifiers it's trying to keep out.
# The only allowlist is the FICTIONAL homes the docs and tests use as examples
# (/home/user, /home/you, /Users/you) — never a real one.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

EXCLUDES=(--exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__
          --exclude-dir=.ruff_cache --exclude-dir=.pytest_cache
          --exclude-dir=node_modules --exclude-dir=site
          --exclude=leak_scan.sh --exclude=uv.lock --exclude='*.lock')

# PCRE pattern | human description — parsed at the LAST '|', so patterns may
# contain alternation; descriptions may not.
CHECKS=(
  '/home/(?!user/|you/)[A-Za-z0-9_.-]+/|absolute Linux home path'
  '/Users/(?!you/|user/)[A-Za-z0-9_.-]+/|absolute macOS home path'
  'sk-ant-[A-Za-z0-9-]{16,}|Anthropic API key'
  'gh[pousr]_[A-Za-z0-9]{20,}|GitHub token'
  'github_pat_[A-Za-z0-9_]{20,}|GitHub fine-grained token'
  'AKIA[0-9A-Z]{16}|AWS access key'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----|private key'
  '[A-Za-z0-9._%+-]+@(?!example\.com|anthropic\.com|users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}|email address'
)

hits=0
for entry in "${CHECKS[@]}"; do
  pat="${entry%|*}"; desc="${entry##*|}"
  out=$(command grep -rInP "${EXCLUDES[@]}" -e "$pat" -- . 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "✗ found ${desc}:"
    echo "$out"
    echo
    hits=1
  elif [ "$rc" -ge 2 ]; then
    # grep itself failed (bad pattern / bad flags). A scanner that can't run
    # must FAIL the gate, not silently pass it — that exact failure mode
    # (-E with -P = "conflicting matchers", plus the email pattern truncated
    # at its first '|' by the old parser) muted every check until 2026-07-06.
    echo "leak_scan: grep error (rc=$rc) on pattern [$pat]:" >&2
    echo "$out" >&2
    exit 2
  fi
done

if [ "$hits" -ne 0 ]; then
  echo "Leak scan FAILED — remove the personal paths / emails / secrets above." >&2
  exit 1
fi
echo "✓ leak scan clean"
