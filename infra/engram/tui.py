#!/usr/bin/env python3
"""Engram terminal — the TUI front-end of Engram's own harness (subscription auth).

    .venv/bin/python infra/engram/tui.py                 # interactive REPL
    .venv/bin/python infra/engram/tui.py --once "hi"     # single-shot (scripts/tests)
    .venv/bin/python infra/engram/tui.py --effort max    # crank reasoning

Streams tool activity live, then renders Engram's answer as markdown. /new resets
the thread, /status shows state, /exit quits. Runs on your Claude subscription
(it strips ANTHROPIC_API_KEY — see core.py).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (  # noqa: E402
    _STRIPPED_API_KEY,
    ENGRAM_CWD,
    AgentSDKDriver,
    LaunchLock,
    render_context_md,
)
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402

console = Console()

BANNER = """[bold cyan]
 ╲╱  ENGRAM — your harness, your subscription, your memory
 ╱╲  [/bold cyan][dim]terminal · Claude Agent SDK · recall-backed[/dim]"""


async def _run_turn(driver: AgentSDKDriver, text: str) -> None:
    """Stream one turn: tool activity live in the spinner, then the answer."""
    buf: list[str] = []
    tools: list[str] = []
    status = console.status("[dim]Engram is thinking…[/dim]", spinner="dots")
    status.start()
    try:
        async for ev in driver.query(text):
            if ev.kind == "tool":
                if ev.text not in tools:
                    tools.append(ev.text)
                status.update(f"[dim]⚙ {ev.text}…[/dim]")
            elif ev.kind == "status":          # ephemeral (sub-agent progress)
                status.update(f"[dim]⚙ {ev.text}…[/dim]")
            elif ev.kind == "text":
                buf.append(ev.text)
    finally:
        status.stop()
    if tools:
        console.print(f"[dim]⚙ {', '.join(tools)}[/dim]")
    answer = "".join(buf).strip()
    if answer:
        console.print(Markdown(answer))
    else:
        console.print("[dim](no text in reply)[/dim]")
        if driver.stderr_tail:
            console.print(f"[red dim]{driver.stderr_tail}[/red dim]")


HELP = ("[dim]Type to talk. [/dim][bold]/new[/bold][dim] fresh · "
        "[/dim][bold]/context[/bold][dim] usage · [/dim][bold]/status[/bold][dim] state · "
        "[/dim][bold]/exit[/bold][dim] quit[/dim]")


async def interactive(driver: AgentSDKDriver) -> int:
    console.print(BANNER)
    if _STRIPPED_API_KEY:
        console.print("[dim]· ANTHROPIC_API_KEY stripped → on your Claude subscription[/dim]")
    console.print(HELP + "\n")
    try:
        while True:
            try:
                text = console.input("[bold green]›[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/dim]")
                break
            if not text:
                continue
            if text in ("/exit", "/quit", "/q"):
                break
            if text == "/new":
                await driver.disconnect()
                driver.reset()
                console.print("[dim]🆕 fresh thread[/dim]\n")
                continue
            if text == "/status":
                fb = getattr(driver, "active_fallback", None)
                cfg_fb = getattr(driver, "fallback_model", None)
                fb_str = (f" · [yellow]⚠ ON FALLBACK: {fb}[/yellow]" if fb
                          else (f" · fallback={cfg_fb}" if cfg_fb else ""))
                console.print(f"[dim]session={driver.session_id or 'fresh'} · "
                              f"model={driver.model}{fb_str} · effort={driver.effort} · "
                              f"cwd={driver.cwd}[/dim]\n")
                continue
            if text == "/context":
                try:
                    usage = await driver.get_context_usage()
                except Exception as exc:  # noqa: BLE001 — older CLI / control unsupported
                    console.print(f"[red]context unavailable: {type(exc).__name__}[/red]")
                    if driver.stderr_tail:
                        console.print(f"[red dim]{driver.stderr_tail}[/red dim]")
                    continue
                console.print(Markdown(render_context_md(usage)))
                console.print()
                continue
            try:
                await _run_turn(driver, text)
            except Exception as exc:  # noqa: BLE001 — never crash the REPL on a turn
                console.print(f"[red]error: {type(exc).__name__}: {exc}[/red]")
                if driver.stderr_tail:
                    console.print(f"[red dim]{driver.stderr_tail}[/red dim]")
            fb = getattr(driver, "active_fallback", None)
            if fb:                       # loud per-turn heads-up: the model rotated
                console.print(f"[yellow]⚠ ran on fallback ({fb}) — primary "
                              f"unavailable; rotates back automatically.[/yellow]")
            console.print()
    finally:
        await driver.disconnect()
    return 0


async def once(driver: AgentSDKDriver, text: str) -> int:
    try:
        await _run_turn(driver, text)
    finally:
        await driver.disconnect()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="engram", description="Engram terminal harness")
    ap.add_argument("--once", metavar="TEXT", help="single-shot: send TEXT, print reply, exit")
    ap.add_argument("--model", default=None, help="override model (e.g. opus[1m])")
    ap.add_argument("--effort", default=None, help="low|medium|high|xhigh|max")
    args = ap.parse_args()
    kw: dict = {}
    if args.model:
        kw["model"] = args.model
    if args.effort:
        kw["effort"] = args.effort
    driver = AgentSDKDriver(**kw)
    if args.once:
        return asyncio.run(once(driver, args.once))   # transient tooling — no folder lock
    # Interactive REPL: share app.py's per-folder lock so `engram --simple` and `engram`
    # can't drive the same session from one folder and interleave into one thread.
    lock = LaunchLock(ENGRAM_CWD)
    owner = lock.acquire()
    if owner is not None:
        sys.stderr.write(
            f"\nEngram is already running in this folder (pid {owner}).\n"
            f"Use that terminal, or close it first.\n"
            f"If you're sure it's gone, remove the stale lock:  rm {lock.path}\n\n")
        return 1
    try:
        return asyncio.run(interactive(driver))
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
