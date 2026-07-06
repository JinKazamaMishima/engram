#!/usr/bin/env python3
"""Unit tests for file checkpoints + /rewind plumbing: the replayed-UserMessage
capture (with its filters), the option flags, and the rewind_to call path.

    .venv/bin/python infra/engram/tests/test_rewind.py
"""
import asyncio
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    TextBlock,
    UserMessage,
)
from core import AgentSDKDriver  # noqa: E402


def AM(text):
    return AssistantMessage(content=[TextBlock(text=text)], model="m")


def RESULT():
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                         is_error=False, num_turns=1, session_id="s")


class FakeClient:
    def __init__(self, messages):
        self._messages = messages
        self.rewound_to = []

    async def query(self, text, *, prepend=""):
        pass

    async def receive_messages(self):
        for m in self._messages:
            yield m

    async def rewind_files(self, user_message_id):
        self.rewound_to.append(user_message_id)


async def run_stream(d, messages):
    d._client = FakeClient(messages)
    return [ev async for ev in d._stream("hi")]


async def test_checkpoint_capture_and_filters():
    d = AgentSDKDriver(store=None)
    await run_stream(d, [
        UserMessage(content="fix the parser bug", uuid="u1"),          # real prompt ✓
        UserMessage(content="no uuid — ignored"),                       # no anchor ✗
        UserMessage(content="tool result", uuid="u2",
                    tool_use_result={"ok": True}),                      # tool echo ✗
        UserMessage(content="sub-agent echo", uuid="u3",
                    parent_tool_use_id="toolu_x"),                      # sub-internal ✗
        UserMessage(content="fix the parser bug", uuid="u1"),           # dup uuid ✗
        AM("on it"),
        RESULT(),
    ])
    assert [c["uuid"] for c in d.checkpoints] == ["u1"], d.checkpoints
    assert d.checkpoints[0]["preview"] == "fix the parser bug"
    assert d.list_checkpoints() == d.checkpoints and d.list_checkpoints() is not d.checkpoints
    print("✓ capture: real prompts anchor; tool-result / sub-agent / uuid-less / dup echoes don't")


async def test_preview_strips_harness_injections():
    d = AgentSDKDriver(store=None)
    prompt = ("[identity] Camera confirms the operator. Proceed normally.\n"
              "<system-reminder>\nrecalled notes blah\nblah\n</system-reminder>\n"
              "  refactor the   index builder  ")
    await run_stream(d, [
        UserMessage(content=prompt, uuid="u1"),
        UserMessage(content="<task-notification>\nagent done\n</task-notification>",
                    uuid="u2"),                                         # injection-only ✗
        UserMessage(content=[TextBlock(text="blocks work too")], uuid="u3"),
        RESULT(),
    ])
    previews = {c["uuid"]: c["preview"] for c in d.checkpoints}
    assert previews == {"u1": "refactor the index builder",
                        "u3": "blocks work too"}, previews
    print("✓ previews show what was TYPED (identity/reminder/ping spans stripped; "
          "injection-only echoes skipped)")


def test_options_flags():
    d = AgentSDKDriver(store=None)
    opts = d._options()
    assert opts.enable_file_checkpointing is True, opts.enable_file_checkpointing
    assert opts.extra_args == {"replay-user-messages": None}, opts.extra_args
    os.environ["ENGRAM_CHECKPOINTS"] = "0"
    try:
        off = d._options()
        assert not off.enable_file_checkpointing and not off.extra_args, \
            (off.enable_file_checkpointing, off.extra_args)
    finally:
        del os.environ["ENGRAM_CHECKPOINTS"]
    print("✓ checkpointing + replay flags on by default; ENGRAM_CHECKPOINTS=0 kills both")


async def test_rewind_to_and_reset():
    d = AgentSDKDriver(store=None)
    d._client = FakeClient([])
    d.checkpoints.append({"uuid": "u9", "preview": "x", "ts": 0.0})
    await d.rewind_to("u9")
    assert d._client.rewound_to == ["u9"], d._client.rewound_to
    d.reset()
    assert d.checkpoints == [], "reset must drop the old thread's anchors"
    print("✓ rewind_to hits the client's rewind_files; reset clears the anchors")


async def main() -> int:
    await test_checkpoint_capture_and_filters()
    await test_preview_strips_harness_injections()
    test_options_flags()
    await test_rewind_to_and_reset()
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
