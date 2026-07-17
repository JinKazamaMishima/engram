#!/usr/bin/env python3
"""aurora m2 — pure plumbing for the live agents panel + per-agent output tail.

Everything here is Textual-free and hermetic-testable: path builders for the
Claude Code on-disk transcript layout, the task→file resolver, an incremental
JSONL TailReader, and the pure renderers the TUI panel draws from.

The on-disk contract (probed live 2026-07-13, all fail-open if it drifts):
- plain (Agent-tool) sub-agents write
  ``~/.claude/projects/<flat-cwd>/<sid>/subagents/agent-<agentId>.jsonl``
  (live-growing) + ``agent-<agentId>.meta.json``
  = ``{"agentType", "description", "toolUseId", ...}``;
- ``/tmp/claude-<uid>/<flat-cwd>/<sid>/tasks/<task_id>.output`` is a SYMLINK to
  that jsonl (workflow-run outputs there are final-only regular files);
- workflow agents write under ``<sid>/subagents/workflows/wf_<id>/`` next to a
  ``journal.jsonl`` whose ``{"type":"started","agentId"}`` rows are in start
  order; their meta.json has NO label, but each agent's FIRST user row opens
  with a ``## <label>`` heading.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from rich.markup import escape

SEED_BYTES = 16384     # first poll shows at most this much history
POLL_CAP = 65536       # max bytes consumed per poll; beyond it we skip forward
HEAD_BYTES = 4096      # bounded read when sniffing a wf agent's `## label` head


def _flat(cwd) -> str:
    """Claude Code's flattened-cwd dir name (same convention as
    recall.transcripts.project_transcript_dir)."""
    return str(Path(cwd).resolve()).replace("/", "-")


def tmp_task_output(cwd, sid: str, task_id: str, tmp_base=None) -> Path:
    base = Path(tmp_base) if tmp_base else Path(f"/tmp/claude-{os.getuid()}")
    return base / _flat(cwd) / sid / "tasks" / f"{task_id}.output"


def subagents_dir(cwd, sid: str, projects_base=None) -> Path:
    base = Path(projects_base) if projects_base else Path.home() / ".claude" / "projects"
    return base / _flat(cwd) / sid / "subagents"


def workflow_dirs(cwd, sid: str, projects_base=None) -> list[Path]:
    """Workflow run dirs, newest first."""
    root = subagents_dir(cwd, sid, projects_base) / "workflows"
    try:
        dirs = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return []
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def _first_line_label(path: Path) -> str:
    """The ``## <label>`` heading a workflow agent's first user row opens with
    ('' when absent/unreadable). Bounded read — never slurps a big transcript."""
    try:
        with open(path, "rb") as f:
            head = f.read(HEAD_BYTES)
        row = json.loads(head.split(b"\n", 1)[0])
        content = (row.get("message") or {}).get("content")
        if isinstance(content, list):
            content = next((b.get("text", "") for b in content
                            if isinstance(b, dict) and b.get("type") == "text"), "")
        first = str(content or "").lstrip().splitlines()[0] if content else ""
        return first[2:].strip() if first.startswith("##") else ""
    except Exception:  # noqa: BLE001 — unreadable/partial head → no label
        return ""


def _journal_agent_ids(wf_dir: Path) -> list[str]:
    """agentIds from journal.jsonl 'started' rows, in start order."""
    ids: list[str] = []
    try:
        for raw in (wf_dir / "journal.jsonl").read_text().splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if row.get("type") == "started" and row.get("agentId"):
                ids.append(row["agentId"])
    except OSError:
        pass
    return ids


def workflow_disk_agents(cwd, sid, *, projects_base=None) -> list[dict]:
    """Reconstruct a workflow's agent tree from its newest run dir ON DISK.

    The journal's ``started``/``result`` rows give order + live state (an agentId
    with a matching ``result`` row is done, otherwise still running); each
    agent's first-row ``## <label>`` heading gives its name. This is what lets
    the panel show a fan-out the instant it spawns — fast workflows finish before
    the first progress heartbeat carries the phase/agent tree, so the in-memory
    snapshot is empty exactly when the run is most alive. Fail-open → ``[]``.

    Returns ``[{label, state, model}]`` (model unknown from disk → "")."""
    dirs = workflow_dirs(cwd, sid, projects_base)
    if not dirs:
        return []
    wf = dirs[0]
    started: list[str] = []
    done: set[str] = set()
    try:
        for raw in (wf / "journal.jsonl").read_text().splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            aid = row.get("agentId")
            if not aid:
                continue
            if row.get("type") == "started":
                started.append(aid)
            elif row.get("type") == "result":
                done.add(aid)
    except OSError:
        return []
    agents: list[dict] = []
    for aid in started:
        label = _first_line_label(wf / f"agent-{aid}.jsonl") or f"agent {len(agents)}"
        agents.append({"label": label,
                       "state": "done" if aid in done else "progress", "model": ""})
    return agents


def enrich_workflow_agents(tasks, cwd, sid, *, projects_base=None) -> list:
    """Backfill each LIVE workflow row's agent tree from disk when the in-memory
    heartbeat snapshot is thinner than what the journal already shows. The
    heartbeat wins once it's richer (it carries real per-agent model + state);
    disk fills the gap for fast fan-outs the heartbeat hasn't described yet.
    Returns a new list; unchanged rows are passed through by reference."""
    out = []
    for t in tasks or []:
        if t.get("workflow") and t.get("status") not in (
                "completed", "failed", "stopped", "killed"):
            disk = workflow_disk_agents(cwd, sid, projects_base=projects_base)
            snap = t.get("wf") or {}
            live_n = sum(len(p.get("agents") or [])
                         for p in snap.get("phases") or [])
            if len(disk) > live_n:
                phase = snap.get("phase") or "run"
                t = {**t, "wf": {
                    "phases": [{"title": phase, "agents": disk}],
                    "done": sum(1 for a in disk if a["state"] == "done"),
                    "total": len(disk), "phase": phase}}
        out.append(t)
    return out


def resolve_task_file(row: dict, cwd, sid: Optional[str], *,
                      tmp_base=None, projects_base=None) -> tuple[Optional[Path], str]:
    """Resolve a panel row to its on-disk live transcript. Returns
    ``(path | None, how)`` — ``how`` is the honest provenance label shown on the
    detail card: direct | meta | heading | journal | live | none. Fail-open:
    any miss falls through to the next strategy, never raises."""
    if not sid:
        return None, "none"

    wf_label = row.get("wf_label")
    if not row.get("workflow"):
        # 1 · direct: the /tmp task_id symlink (Agent-tool tasks).
        task_id = row.get("task_id")
        if task_id:
            p = tmp_task_output(cwd, sid, task_id, tmp_base)
            try:
                if p.exists():
                    return p.resolve(), "direct"
            except OSError:
                pass
        # 2 · meta: join on toolUseId (exact), then description (best-effort).
        sub = subagents_dir(cwd, sid, projects_base)
        tuid, desc = row.get("tool_use_id"), (row.get("desc") or "").strip()
        best: tuple[float, Optional[Path]] = (-1.0, None)
        try:
            metas = sorted(sub.glob("agent-*.meta.json"))
        except OSError:
            metas = []
        for m in metas:
            try:
                meta = json.loads(m.read_text())
            except Exception:  # noqa: BLE001 — half-written meta mid-spawn
                continue
            jsonl = m.with_name(m.name.removesuffix(".meta.json") + ".jsonl")
            if not jsonl.exists():
                continue
            if tuid and meta.get("toolUseId") == tuid:
                return jsonl, "meta"
            if desc and meta.get("description") == desc:
                mt = jsonl.stat().st_mtime
                if mt > best[0]:
                    best = (mt, jsonl)
        if best[1] is not None:
            return best[1], "meta"
        return None, "none"

    # Workflow rows — newest run dir.
    dirs = workflow_dirs(cwd, sid, projects_base)
    if not dirs:
        return None, "none"
    wf = dirs[0]
    try:
        jsonls = sorted(wf.glob("agent-*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        jsonls = []
    if not jsonls:
        return None, "none"
    if wf_label:
        # 3a · heading: exact `## <label>` match in each agent's first row.
        for p in jsonls:
            if _first_line_label(p) == wf_label:
                return p, "heading"
        # 3b · journal: k-th started agentId ↔ k-th snapshot agent (best-effort).
        idx = row.get("wf_index")
        if isinstance(idx, int) and idx >= 0:
            ids = _journal_agent_ids(wf)
            if idx < len(ids):
                p = wf / f"agent-{ids[idx]}.jsonl"
                if p.exists():
                    return p, "journal"
    # 3c · live feed: whatever the workflow wrote to most recently.
    return jsonls[0], "live"


def render_row(obj: dict) -> str:
    """One transcript JSONL row → display text ('' when it isn't agent output).
    Assistant rows only: text passes through escaped; tool_use collapses to one
    dim marker line (mirrors fleet.render_member_event's compactness)."""
    if not isinstance(obj, dict) or obj.get("type") != "assistant":
        return ""
    msg = obj.get("message")
    blocks = msg if isinstance(msg, list) else (
        msg.get("content") if isinstance(msg, dict) else None)
    if isinstance(blocks, str):
        return escape(blocks) + "\n"
    out: list[str] = []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and b.get("text"):
            out.append(escape(b["text"]) + "\n")
        elif b.get("type") == "tool_use":
            inp = b.get("input") or {}
            head = next((str(inp[k]) for k in
                         ("description", "command", "file_path", "pattern",
                          "prompt", "query", "url") if inp.get(k)), "")
            head = " ".join(head.split())[:80]
            name = b.get("name", "tool")
            out.append(f"[dim]· {escape(name)}{': ' + escape(head) if head else ''}[/dim]\n")
    return "".join(out)


class TailReader:
    """Incremental reader for one live agent JSONL. ``poll()`` returns freshly
    rendered output ('' when nothing new) and never raises — a vanished or
    rotated file re-seeds on the next poll. Open-per-poll: no held fd."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offset: Optional[int] = None    # None → first poll seeds from tail
        self._buf = b""                       # trailing partial line

    def poll(self) -> str:
        try:
            size = self.path.stat().st_size
        except OSError:
            return ""
        skipped = False
        if self._offset is None or size < self._offset:
            # First poll, or the file shrank (rotation/truncation): seed near the
            # end and drop the (likely partial) first line.
            self._offset = max(0, size - SEED_BYTES)
            self._buf = b""
            seeding = self._offset > 0
        else:
            seeding = False
            if size - self._offset > POLL_CAP:
                self._offset = max(0, size - SEED_BYTES)
                self._buf = b""
                skipped = True
        if size == self._offset:
            return ""
        try:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read(min(size - self._offset, POLL_CAP))
        except OSError:
            return ""
        self._offset += len(chunk)
        data = self._buf + chunk
        lines = data.split(b"\n")
        self._buf = lines.pop()               # trailing partial (b'' when clean)
        if seeding and lines:
            lines.pop(0)                      # seeded mid-file: first line partial
        out: list[str] = []
        if skipped:
            out.append("[dim]⋯ skipped ahead (agent writing faster than the tail) ⋯[/dim]\n")
        for raw in lines:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            piece = render_row(obj)
            if piece:
                out.append(piece)
        return "".join(out)


# --- pure panel renderers -----------------------------------------------------

_STATE_MARK = {"done": "✓", "completed": "✓", "failed": "✗", "stopped": "✗",
               "killed": "✗", "error": "✗"}
_SNAKE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"    # braille 'snake' for a running row — the m1 pulse's glyph set


def agent_panel_rows(tasks: list, *, deleg: dict | None = None,
                     frame: int = 0) -> list[dict]:
    """Task-registry snapshot → ordered panel rows. One row per task: a running row
    leads with a braille 'snake' (advanced by ``frame``) + an indented child row per
    workflow agent; a finished row shows a ✓/✗ IN PLACE (no separate finished-counter
    — the row itself carries the outcome until the turn empties the panel). ``deleg``
    (a delegations.snapshot()) adds a 📡 row per live cross-provider call and a
    collapsed counter for finished ones — grok calls are tool calls, not SDK tasks, so
    this is the only place they surface, and the snapshot carries only a count+cost for
    finished ones (hence a counter there, not per-call rows).
    Row shape: {id, label, kind, task_id, tool_use_id, desc, workflow,
    wf_label|None, wf_index|None, status}."""
    rows: list[dict] = []
    snake = _SNAKE[frame % len(_SNAKE)]
    for t in tasks or []:
        status = t.get("status")
        task_id = t.get("task_id") or ""
        base = {"task_id": task_id, "tool_use_id": t.get("tool_use_id"),
                "desc": t.get("desc") or "", "workflow": bool(t.get("workflow")),
                "status": status or "running"}
        tag = "⚙ " if t.get("workflow") else ""
        name = str(t.get("name", "sub-agent")).removeprefix("⚙ ")
        rid = task_id or name
        if status == "completed":
            rows.append({**base, "id": rid, "kind": "task", "label": f"✓ {tag}{name}",
                         "wf_label": None, "wf_index": None})
            continue
        if status in ("failed", "stopped", "killed"):
            rows.append({**base, "id": rid, "kind": "task", "label": f"✗ {tag}{name}",
                         "wf_label": None, "wf_index": None})
            continue
        bits = [f"{snake} {tag}{name}"]
        if t.get("tokens"):
            bits.append(f"{int(t['tokens']) // 1000}k")
        if t.get("last_tool"):
            bits.append(str(t["last_tool"]))
        rows.append({**base, "id": rid, "kind": "task",
                     "label": "  ".join(bits), "wf_label": None, "wf_index": None})
        wf = t.get("wf") or {}
        idx = 0
        for phase in wf.get("phases") or []:
            for a in phase.get("agents") or []:
                mark = _STATE_MARK.get(a.get("state"), snake)
                label = a.get("label") or f"agent {idx}"
                bits = [f"  {mark} {label}"]
                if a.get("model"):
                    bits.append(str(a["model"]))
                rows.append({**base, "id": f"{task_id}/{label}", "kind": "wfagent",
                             "label": "  ".join(bits), "wf_label": label,
                             "wf_index": idx, "status": a.get("state") or "…"})
                idx += 1
    d = deleg or {}
    for e in d.get("live") or []:
        bits = [f"{snake} 📡 {e.get('label') or 'delegation'}"]
        if e.get("model"):
            bits.append(str(e["model"]))
        rows.append({"id": f"deleg/{e.get('id')}", "kind": "deleg",
                     "label": "  ".join(bits), "task_id": "", "tool_use_id": None,
                     "desc": e.get("model") or "", "workflow": False,
                     "wf_label": None, "wf_index": None, "status": "running",
                     "model": e.get("model") or ""})
    dd, df = int(d.get("done") or 0), int(d.get("failed") or 0)
    if dd or df:
        tail = " · ".join(([f"✓ {dd} done"] if dd else [])
                          + ([f"✗ {df} failed"] if df else []))
        if d.get("cost"):
            tail += f" · ~${float(d['cost']):.2f}"
        rows.append({"id": "~deleg", "kind": "info", "label": f"  📡 {tail}",
                     "task_id": "", "tool_use_id": None, "desc": "",
                     "workflow": False, "wf_label": None, "wf_index": None,
                     "status": "info"})
    return rows


_HOW_LABEL = {
    "direct": "output: direct",
    "meta": "output: matched by meta",
    "heading": "output: matched by heading",
    "journal": "output: journal order (best-effort)",
    "live": "output: live feed (newest in this workflow — best-effort)",
    "none": "no output file — state only",
}


def agent_detail_card(row: dict, how: str) -> str:
    """Rich-markup header line for the detail pane: who + state + honest
    provenance of what the pane shows."""
    who = row.get("wf_label") or row.get("desc") or row.get("id") or "agent"
    state = row.get("status", "")
    src = _HOW_LABEL.get(how, _HOW_LABEL["none"])
    return (f"[b]{escape(str(who))}[/b]"
            + (f"  [dim]·[/dim] {escape(str(state))}" if state else "")
            + f"  [dim]·[/dim] [i]{src}[/i]")
