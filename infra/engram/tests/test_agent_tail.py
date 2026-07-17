#!/usr/bin/env python3
"""Hermetic tests for agent_tail (aurora m2): path shapes, the task→file
resolver's precedence chain over fabricated fixture trees, the incremental
TailReader, and the pure panel renderers. No Textual, no network, no real
~/.claude access — bases are injected tmp dirs.

    .venv/bin/python infra/engram/tests/test_agent_tail.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from agent_tail import (  # noqa: E402  (after sys.path shim, like sibling tests)
    TailReader,
    agent_detail_card,
    agent_panel_rows,
    render_row,
    resolve_task_file,
    subagents_dir,
    tmp_task_output,
)

CWD = "/home/user/proj"
SID = "sess-1"


def _row(text):
    return json.dumps({"type": "assistant",
                       "message": {"content": [{"type": "text", "text": text}]},
                       "timestamp": "2026-07-13T00:00:00Z"})


def _fixture(tmp):
    """A fabricated on-disk tree: tmp_base + projects_base with one plain agent
    (meta) and one workflow run (two agents + journal)."""
    tmp_base = Path(tmp) / "tmp"
    projects = Path(tmp) / "projects"
    sub = subagents_dir(CWD, SID, projects)
    sub.mkdir(parents=True)
    # plain agent + meta
    agent = sub / "agent-a111.jsonl"
    agent.write_text(_row("plain agent says hi") + "\n")
    (sub / "agent-a111.meta.json").write_text(json.dumps(
        {"agentType": "Explore", "description": "map the code", "toolUseId": "tu1"}))
    # /tmp task symlink
    tdir = tmp_task_output(CWD, SID, "tA", tmp_base).parent
    tdir.mkdir(parents=True)
    (tdir / "tA.output").symlink_to(agent)
    # workflow run: two agents, first-row `## <label>` headings, journal
    wf = sub / "workflows" / "wf_x1"
    wf.mkdir(parents=True)
    w1 = wf / "agent-w001.jsonl"
    w1.write_text(json.dumps({"type": "user", "message":
                              {"content": "## verify:alpha\ndo the thing"}}) + "\n"
                  + _row("alpha output") + "\n")
    w2 = wf / "agent-w002.jsonl"
    w2.write_text(json.dumps({"type": "user", "message":
                              {"content": [{"type": "text",
                                            "text": "## scan:beta\nprompt"}]}}) + "\n"
                  + _row("beta output") + "\n")
    (wf / "journal.jsonl").write_text(
        json.dumps({"type": "started", "key": "k1", "agentId": "w001"}) + "\n"
        + json.dumps({"type": "started", "key": "k2", "agentId": "w002"}) + "\n")
    os.utime(w2, (2_000_000_000, 2_000_000_000))   # w2 = newest
    return tmp_base, projects


def test_path_shapes():
    p = tmp_task_output(CWD, SID, "t1", "/tmp/base")
    assert str(p).endswith("-home-user-proj/sess-1/tasks/t1.output")
    s = subagents_dir(CWD, SID, "/proj")
    assert str(s) == "/proj/-home-user-proj/sess-1/subagents"


def test_resolver_direct_symlink():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_base, projects = _fixture(tmp)
        row = {"task_id": "tA", "workflow": False}
        p, how = resolve_task_file(row, CWD, SID, tmp_base=tmp_base,
                                   projects_base=projects)
        assert how == "direct" and p.name == "agent-a111.jsonl"


def test_resolver_meta_tooluseid_then_desc():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_base, projects = _fixture(tmp)
        row = {"task_id": "tMISSING", "tool_use_id": "tu1", "workflow": False}
        p, how = resolve_task_file(row, CWD, SID, tmp_base=tmp_base,
                                   projects_base=projects)
        assert how == "meta" and p.name == "agent-a111.jsonl"
        row = {"task_id": "tMISSING", "desc": "map the code", "workflow": False}
        p, how = resolve_task_file(row, CWD, SID, tmp_base=tmp_base,
                                   projects_base=projects)
        assert how == "meta" and p.name == "agent-a111.jsonl"


def test_resolver_workflow_heading_journal_live():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_base, projects = _fixture(tmp)
        base = {"task_id": "tW", "workflow": True}
        p, how = resolve_task_file({**base, "wf_label": "verify:alpha"},
                                   CWD, SID, tmp_base=tmp_base, projects_base=projects)
        assert how == "heading" and p.name == "agent-w001.jsonl"
        # unknown label but valid index → journal order
        p, how = resolve_task_file({**base, "wf_label": "nope", "wf_index": 1},
                                   CWD, SID, tmp_base=tmp_base, projects_base=projects)
        assert how == "journal" and p.name == "agent-w002.jsonl"
        # parent workflow row (no label) → newest-mtime live feed
        p, how = resolve_task_file({**base, "wf_label": None},
                                   CWD, SID, tmp_base=tmp_base, projects_base=projects)
        assert how == "live" and p.name == "agent-w002.jsonl"


def test_resolver_fail_open():
    with tempfile.TemporaryDirectory() as tmp:
        p, how = resolve_task_file({"task_id": "x", "workflow": False}, CWD, SID,
                                   tmp_base=Path(tmp) / "no", projects_base=Path(tmp) / "no2")
        assert p is None and how == "none"
        p, how = resolve_task_file({"task_id": "x", "workflow": False}, CWD, None)
        assert p is None and how == "none"


def test_tail_incremental_and_partial_lines():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "a.jsonl"
        f.write_text(_row("first") + "\n")
        t = TailReader(f)
        out = t.poll()
        assert "first" in out
        assert t.poll() == ""                       # nothing new
        # append a HALF line → buffered, not emitted
        whole = _row("second") + "\n"
        with open(f, "a") as fh:
            fh.write(whole[:10])
        assert t.poll() == ""
        with open(f, "a") as fh:
            fh.write(whole[10:])
        assert "second" in t.poll()                 # completed → emitted


def test_tail_seed_drops_partial_and_truncation_reseeds():
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "a.jsonl"
        pad = _row("x" * 4000) + "\n"
        f.write_text(pad * 8 + _row("tail-visible") + "\n")   # > SEED_BYTES
        t = TailReader(f)
        out = t.poll()
        assert "tail-visible" in out                # seeded near the end, no crash
        f.write_text(_row("fresh-after-truncate") + "\n")     # shrink → re-seed
        assert "fresh-after-truncate" in t.poll()


def test_tail_missing_file_is_silent():
    t = TailReader(Path("/nonexistent/nope.jsonl"))
    assert t.poll() == ""


def test_render_row_shapes():
    assert "hello" in render_row(json.loads(_row("hello")))
    tool = {"type": "assistant", "message": [
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la /tmp"}}]}
    out = render_row(tool)
    assert "· Bash: ls -la /tmp" in out and "[dim]" in out
    assert render_row({"type": "user", "message": {"content": "prompt"}}) == ""
    assert render_row({"not": "a row"}) == ""
    assert "plain" in render_row({"type": "assistant", "message": {"content": "plain"}})


def test_panel_rows_and_card():
    tasks = [
        {"task_id": "t1", "name": "Explore", "status": "running", "tokens": 12400,
         "last_tool": "Grep", "desc": "find callers"},
        {"task_id": "t2", "name": "⚙ deep-research", "status": "running", "workflow": True,
         "wf": {"phases": [{"title": "Scan", "agents": [
             {"label": "scan:a", "state": "done", "model": "sonnet"},
             {"label": "scan:b", "state": "progress", "model": ""}]}],
             "done": 1, "total": 2, "phase": "Scan"}},
        {"task_id": "t3", "name": "old", "status": "completed"},
        {"task_id": "t4", "name": "dead", "status": "failed"},
    ]
    rows = agent_panel_rows(tasks)
    ids = [r["id"] for r in rows]
    # finished tasks now show a ✓/✗ row IN PLACE (no collapsed ~finished counter)
    assert ids == ["t1", "t2", "t2/scan:a", "t2/scan:b", "t3", "t4"]
    assert len(set(ids)) == len(ids)                       # stable + unique
    assert "12k" in rows[0]["label"] and "Grep" in rows[0]["label"]
    assert rows[2]["kind"] == "wfagent" and rows[2]["wf_index"] == 0
    # a done child keeps its ✓; a running child now shows the braille snake, not ⏳
    assert "✓" in rows[2]["label"] and "⠋" in rows[3]["label"] and "⏳" not in rows[3]["label"]
    assert rows[4]["label"] == "✓ old" and rows[5]["label"] == "✗ dead"
    assert not any(r["id"] == "~finished" for r in rows)
    card = agent_detail_card(rows[2], "heading")
    assert "scan:a" in card and "matched by heading" in card
    assert "state only" in agent_detail_card(rows[0], "none")


def test_workflow_disk_agents():
    # aurora m3: reconstruct a fan-out from the journal + agent-file headings,
    # so the panel shows it before any heartbeat carries the tree.
    from agent_tail import workflow_disk_agents
    with tempfile.TemporaryDirectory() as tmp:
        _tmp_base, projects = _fixture(tmp)
        ag = workflow_disk_agents(CWD, SID, projects_base=projects)
        assert [a["label"] for a in ag] == ["verify:alpha", "scan:beta"]
        assert all(a["state"] == "progress" for a in ag), "no result rows yet"
        # a result row flips just that agentId to done; journal order preserved
        wf = subagents_dir(CWD, SID, projects) / "workflows" / "wf_x1"
        with open(wf / "journal.jsonl", "a") as f:
            f.write(json.dumps({"type": "result", "key": "k1",
                                "agentId": "w001", "result": "ok"}) + "\n")
        ag = workflow_disk_agents(CWD, SID, projects_base=projects)
        assert ag[0]["state"] == "done" and ag[1]["state"] == "progress"
        assert workflow_disk_agents(CWD, "no-such", projects_base=projects) == []


def test_enrich_workflow_agents():
    from agent_tail import enrich_workflow_agents
    with tempfile.TemporaryDirectory() as tmp:
        _tmp_base, projects = _fixture(tmp)
        # empty heartbeat snapshot on a LIVE workflow → filled from disk
        thin = [{"task_id": "wf", "workflow": True, "status": "running", "wf": {}}]
        out = enrich_workflow_agents(thin, CWD, SID, projects_base=projects)
        agents = out[0]["wf"]["phases"][0]["agents"]
        assert [a["label"] for a in agents] == ["verify:alpha", "scan:beta"]
        assert out[0]["wf"]["total"] == 2
        # a RICHER heartbeat snapshot wins — disk does not overwrite it
        rich = [{"task_id": "wf", "workflow": True, "status": "running",
                 "wf": {"phases": [{"title": "Scan", "agents": [
                     {"label": "a"}, {"label": "b"}, {"label": "c"}]}]}}]
        assert enrich_workflow_agents(rich, CWD, SID,
                                      projects_base=projects)[0] is rich[0]
        # terminal workflows are never enriched (the run is over)
        done = [{"task_id": "wf", "workflow": True, "status": "completed", "wf": {}}]
        assert enrich_workflow_agents(done, CWD, SID,
                                      projects_base=projects)[0] is done[0]


def test_panel_rows_delegations():
    deleg = {"live": [{"id": 7, "label": "grok·low", "model": "grok-4.5"}],
             "done": 2, "failed": 1, "cost": 0.61}
    rows = agent_panel_rows([], deleg=deleg)
    assert [r["id"] for r in rows] == ["deleg/7", "~deleg"]
    assert rows[0]["kind"] == "deleg" and "📡" in rows[0]["label"]
    assert "grok·low" in rows[0]["label"] and "grok-4.5" in rows[0]["label"]
    assert rows[1]["kind"] == "info"
    assert "✓ 2 done" in rows[1]["label"] and "✗ 1 failed" in rows[1]["label"]
    assert "$0.61" in rows[1]["label"]
    assert agent_panel_rows([]) == [], "no deleg + no tasks → empty"


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception:
                fails += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print("---", "all green" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
