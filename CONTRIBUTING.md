# Contributing to Engram

Thanks for your interest in Engram. Contributions are welcome via pull request.

## Setup

```sh
uv venv
uv pip install -e ".[dev,engram]"     # ",telegram" too if you touch the bridge
```

## Before you open a PR

```sh
uv run pytest            # full test suite (engine + harness)
uv run ruff check .      # lint
bash scripts/leak_scan.sh   # privacy / secret hygiene — also enforced in CI
```

## Ground rules

- **`main` is protected.** Changes land via pull request with review and passing
  CI — nobody pushes to `main` directly. Fork, branch, PR.
- **Keep personal data out.** No absolute home paths (`/home/you/...`), email
  addresses, API keys/tokens, or biometric files (`*.npz`, face galleries).
  The leak scan blocks the common cases; use env vars and the data directory
  (`RECALL_DATA_ROOT`) for anything machine- or user-specific.
- **Match the surrounding style.** Keep comments meaningful; run `ruff`.
- **Tests for behavior changes.** The engine tests inject a fake embedder, so
  they run fast with no model download — please keep them that way.

## Project shape

- `src/recall/` — the memory engine (portable, no personal data).
- `infra/engram/` — the terminal assistant + Sensorium (perception).
- `infra/telegram/` — the optional phone bridge.
- `scripts/`, `infra/systemd/` — install + service templates.

By contributing you agree your work is licensed under the [MIT License](LICENSE).
