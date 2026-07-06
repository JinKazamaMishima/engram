#!/usr/bin/env python3
"""Unit tests for the in-process recall MCP tools: search + read against a real
temp corpus with a keyword-only index (no daemon, no torch), plus the driver
wiring and its kill switch.

    .venv/bin/python infra/engram/tests/test_mcp_recall.py
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(ENGRAM, "..", ".."))
sys.path.insert(0, ENGRAM)
sys.path.insert(0, os.path.join(REPO, "src"))

NOTE = """---
name: test-widget-lesson
description: "widgets must be frobnicated before shipping"
tags: [widgets]
---
The full reasoning body: frobnicate FIRST, then ship. Numbers: 42.
"""


def make_world(tmp: Path):
    """A temp RECALL_DATA_ROOT + project with one note each (project + soul),
    keyword-only indices built the way a fresh install would."""
    os.environ["RECALL_DATA_ROOT"] = str(tmp / "data")
    project = tmp / "proj"
    (project / "docs" / "knowledge").mkdir(parents=True)
    (project / "docs" / "knowledge" / "test-widget-lesson.md").write_text(NOTE)
    from recall import config, index
    config.ensure_dirs(config.global_corpus_dir(), config.index_dir())
    (config.global_corpus_dir() / "soul-note.md").write_text(NOTE.replace(
        "test-widget-lesson", "soul-note").replace("frobnicated", "soulified"))
    slug = config.project_slug(project)
    index.build_index(config.project_corpus_dir(project), config.index_path(slug), None)
    index.build_index(config.global_corpus_dir(),
                      config.index_path(config.GLOBAL_SCOPE), None)
    return project


async def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        project = make_world(Path(td))
        from memory_tools import build_recall_server
        import memory_tools
        # Force keyword-only determinism: no daemon reachable in tests anyway,
        # but make it explicit and instant.
        memory_tools._fetch_query_vector = lambda prompt: None

        assert build_recall_server(project) is not None

        # Handler-level: capture the decorated tool objects as the builder makes
        # them (they live inside the server instance otherwise).
        captured = {}
        real_create = memory_tools.create_sdk_mcp_server
        memory_tools.create_sdk_mcp_server = (
            lambda name, version="1.0.0", tools=None:
            captured.update({t.name: t for t in tools}) or
            real_create(name=name, version=version, tools=tools))
        try:
            assert memory_tools.build_recall_server(project) is not None
        finally:
            memory_tools.create_sdk_mcp_server = real_create
        search, read = captured["recall_search"], captured["recall_read_note"]

        out = await search.handler({"query": "how do we ship widgets", "k": 5})
        text = out["content"][0]["text"]
        assert not out.get("is_error"), out
        assert "test-widget-lesson" in text and "frobnicated" in text, text
        assert "soul-note" in text and "(global" in text, f"soul corpus must fuse in: {text}"
        print("✓ recall_search fuses project + soul hits (keyword-only degrade)")

        out = await search.handler({"query": ""})
        assert out.get("is_error"), "empty query must error politely"
        out = await search.handler({"query": "zebra unicorn nothing matches this"})
        assert "No matching notes" in out["content"][0]["text"], out
        print("✓ empty query errors; no-match answers honestly")

        out = await read.handler({"slug": "test-widget-lesson"})
        assert "frobnicate FIRST" in out["content"][0]["text"], out
        out = await read.handler({"slug": "soul-note"})
        assert "soulified" in out["content"][0]["text"], "soul fallback must resolve"
        out = await read.handler({"slug": "no-such-note"})
        assert out.get("is_error") and "no note" in out["content"][0]["text"], out
        out = await read.handler({"slug": "../../etc/passwd"})
        assert out.get("is_error") and "bad slug" in out["content"][0]["text"], out
        print("✓ recall_read_note: project → soul resolution, honest miss, traversal guard")

        # Driver wiring + kill switch.
        from core import AgentSDKDriver
        d = AgentSDKDriver(store=None, cwd=project)
        opts = d._options()
        assert "recall" in (opts.mcp_servers or {}), "server must be wired"
        assert "mcp__recall__recall_search" in (opts.allowed_tools or []), opts.allowed_tools
        os.environ["ENGRAM_RECALL_TOOLS"] = "0"
        try:
            d2 = AgentSDKDriver(store=None, cwd=project)
            o2 = d2._options()
            assert not d2.mcp_servers and not o2.mcp_servers and not o2.allowed_tools
        finally:
            del os.environ["ENGRAM_RECALL_TOOLS"]
        print("✓ driver wires mcp_servers + pre-allowed tool names; ENGRAM_RECALL_TOOLS=0 removes")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
