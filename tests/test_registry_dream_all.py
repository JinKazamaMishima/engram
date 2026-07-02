"""dream-all flag forwarding. Counterfactual (L1) dreaming is scoped to the global
soul; every other flag forwards to all scopes. ``dream.run`` is patched so no model,
no corpus and no git are touched."""
from __future__ import annotations

from pathlib import Path

from recall import registry


class _Out:
    exit_code = 0


def _capture(monkeypatch, projects):
    calls: list[list[str]] = []
    monkeypatch.setattr("recall.dream.run",
                        lambda argv: (calls.append(list(argv)) or _Out()))
    monkeypatch.setattr(registry, "list_projects", lambda: projects)
    return calls


def test_dream_all_scopes_counterfactual_to_global_only(monkeypatch):
    calls = _capture(monkeypatch, [Path("/x/proj-a")])
    assert registry.dream_all(["--counterfactual", "--commit"]) == 0
    g = next(c for c in calls if "global" in c)
    assert "--counterfactual" in g and "--commit" in g          # the soul gets the what-if
    for c in calls:
        if "project" in c:
            assert "--counterfactual" not in c                  # projects do not…
            assert "--commit" in c                              # …but other flags still forward


def test_dream_all_no_counterfactual_by_default(monkeypatch):
    calls = _capture(monkeypatch, [])
    registry.dream_all([])
    assert calls and all("--counterfactual" not in c for c in calls)
