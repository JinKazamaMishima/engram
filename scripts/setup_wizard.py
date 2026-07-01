#!/usr/bin/env python3
"""Engram guided setup wizard.

Launched by ./install.sh after the virtualenv exists. Walks through, step by
step: authentication, your data folder, which tiers to install, models, skills +
hook, background services, and an optional first face enrollment. Everything is
explained and skippable; nothing personal ever leaves your machine.
"""
from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO / ".venv" / "bin" / "python"
CONFIG_DIR = Path(os.path.expanduser("~/.config/engram"))
ENV_FILE = CONFIG_DIR / "engram.env"
DEFAULT_DATA_ROOT = os.path.expanduser("~/.local/share/recall")

c = Console()


# ---- helpers ----------------------------------------------------------------

def rule(title: str) -> None:
    c.print()
    c.rule(f"[bold cyan]{title}[/bold cyan]")


def explain(text: str) -> None:
    c.print(Panel(text, border_style="cyan", padding=(0, 1)))


def run(cmd: list[str], *, check: bool = True, quiet: bool = False) -> int:
    if not quiet:
        c.print(f"[dim]$ {' '.join(str(x) for x in cmd)}[/dim]")
    try:
        return subprocess.run(cmd, check=check).returncode
    except subprocess.CalledProcessError as e:
        c.print(f"[red]  command failed (exit {e.returncode})[/red]")
        return e.returncode


def uv_install(pkgs: list[str]) -> None:
    run(["uv", "pip", "install", "-q", *pkgs])


# ---- steps ------------------------------------------------------------------

def welcome() -> None:
    c.print(Panel.fit(
        "[bold]Engram[/bold] — a persistent-memory AI assistant.\n"
        "[dim]This wizard configures Engram on your machine. Your memory corpus,\n"
        "models, and any biometric data stay local and never leave this computer.[/dim]",
        border_style="cyan"))


def preflight() -> dict:
    rule("1 · Preflight")
    facts = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "gpu": shutil.which("nvidia-smi") is not None,
        "webcam": any(Path(f"/dev/video{i}").exists() for i in range(4)),
        "systemd": shutil.which("systemctl") is not None and os.path.isdir("/run/systemd/system"),
        "claude": shutil.which("claude"),
    }
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("Python", f"[green]{facts['python']}[/green]")
    t.add_row("NVIDIA GPU", "[green]yes[/green]" if facts["gpu"] else "[yellow]none (CPU ok)[/yellow]")
    t.add_row("Webcam", "[green]found[/green]" if facts["webcam"] else "[yellow]none[/yellow]")
    t.add_row("systemd", "[green]yes[/green]" if facts["systemd"] else "[yellow]no (services skipped)[/yellow]")
    t.add_row("claude CLI", f"[green]{facts['claude']}[/green]" if facts["claude"] else "[yellow]not found[/yellow]")
    c.print(t)
    return facts


def choose_auth(env: dict, facts: dict) -> None:
    rule("2 · Log in to Anthropic")
    explain(
        "Engram runs on Claude via the Agent SDK. Two ways to authenticate:\n\n"
        "  [bold]1.[/bold] Claude [bold]Pro/Max subscription[/bold] — through the `claude` CLI login "
        "(recommended; no per-token billing).\n"
        "  [bold]2.[/bold] An [bold]Anthropic API key[/bold] — pay-as-you-go.")
    choice = Prompt.ask("Authenticate with", choices=["subscription", "apikey"], default="subscription")
    if choice == "subscription":
        if not facts["claude"]:
            explain(
                "The `claude` CLI isn't installed. Install it, then log in:\n"
                "  [bold]npm install -g @anthropic-ai/claude-code[/bold]   (or see "
                "https://docs.claude.com/claude-code)\n"
                "  [bold]claude[/bold]   — then follow the browser sign-in.\n"
                "You can finish this wizard and do that after; Engram will pick up the login.")
        elif Confirm.ask("Launch `claude` now to sign in (skip if already logged in)?", default=False):
            run(["claude"], check=False)
        env["ENGRAM_FORCE_SUBSCRIPTION"] = "1"
    else:
        key = Prompt.ask("Paste your ANTHROPIC_API_KEY", password=True).strip()
        if key:
            env["ANTHROPIC_API_KEY"] = key
            c.print("  [green]✓[/green] key stored in your config (mode 600)")


def choose_data_dir(env: dict) -> None:
    rule("3 · Your data folder")
    explain(
        "Where your [bold]memory corpus, indices, sessions, and any enrolled face data[/bold] live.\n"
        "[dim]This is the private half of Engram. It stays on your machine — nothing here is ever "
        "uploaded, and it's kept out of git.[/dim]")
    path = Prompt.ask("Data directory", default=DEFAULT_DATA_ROOT).strip()
    path = os.path.abspath(os.path.expanduser(path))
    Path(path).mkdir(parents=True, exist_ok=True)
    env["RECALL_DATA_ROOT"] = path
    c.print(f"  [green]✓[/green] {path}")
    who = Prompt.ask("Name Engram should greet / recognize as you", default=getpass.getuser()).strip()
    if who:
        env["ENGRAM_USER"] = who


def choose_tiers(facts: dict) -> set[str]:
    rule("4 · Choose your tiers")
    explain(
        "[bold]memory[/bold]    the recall engine + skills + hook (works inside Claude Code). Always on.\n"
        "[bold]assistant[/bold] a standalone terminal chat app with the memory wired in.\n"
        "[bold]sensorium[/bold] webcam eye + face-ID + a perceiving loop [dim](experimental; GPU + webcam)[/dim].\n"
        "[bold]telegram[/bold]  talk to your Engram from your phone.")
    tiers = {"memory"}
    if Confirm.ask("Install the [bold]assistant[/bold] (terminal app)?", default=True):
        tiers.add("assistant")
    default_sensorium = facts["gpu"] and facts["webcam"]
    if Confirm.ask("Install the [bold]Sensorium[/bold] (camera perception)?", default=default_sensorium):
        tiers.add("sensorium")
    if Confirm.ask("Install the [bold]Telegram[/bold] bridge?", default=False):
        tiers.add("telegram")
    return tiers


def install_deps(tiers: set[str]) -> None:
    rule("5 · Install")
    # memory core (pyyaml/sqlite-vec/numpy) is already in the bootstrap venv.
    if Confirm.ask("Add [bold]local semantic search[/bold] models (torch + Qwen3 embeddings, ~2GB; "
                   "keyword search works without it)?", default=False):
        c.print("[dim]  installing the pinned ML stack — this can take a few minutes…[/dim]")
        uv_install(["torch==2.12.1", "sentence-transformers==5.6.0", "transformers==5.9.0"])
    if tiers & {"assistant", "sensorium"}:
        uv_install(["textual>=8,<9", "rich>=13", "claude-agent-sdk>=0.2,<0.3"])
    if "sensorium" in tiers:
        c.print("[dim]  installing perception deps (opencv + onnx face-ID)…[/dim]")
        uv_install(["opencv-python>=4.13,<5", "onnxruntime>=1.18,<2"])
    if "telegram" in tiers:
        uv_install(["claude-agent-sdk>=0.2,<0.3"])
    c.print("  [green]✓[/green] dependencies installed")


def install_skills_and_hook() -> None:
    rule("6 · Skills + memory hook")
    explain(
        "Installs the [bold]/recall, /curate-memory, /dream, /reconsolidate-memory[/bold] skills into\n"
        "~/.claude/skills/, and a [bold]UserPromptSubmit hook[/bold] that auto-injects relevant memory\n"
        "into every Claude Code turn (edits ~/.claude/settings.json).")
    if Confirm.ask("Install skills?", default=True):
        run([str(VENV_PY), str(REPO / "scripts" / "install_skills.py")], check=False)
    if Confirm.ask("Install the memory hook (edits ~/.claude/settings.json)?", default=True):
        run([str(VENV_PY), str(REPO / "scripts" / "install_hook.py")], check=False)


def maybe_services(tiers: set[str], facts: dict) -> None:
    if not facts["systemd"]:
        return
    rule("7 · Background services (optional)")
    explain(
        "systemd [dim](user)[/dim] units: a warm [bold]embedder daemon[/bold] and the [bold]nightly "
        "curation[/bold] timer that distills your conversations into memory while you sleep.")
    if Confirm.ask("Install and enable the background services?", default=False):
        run(["bash", str(REPO / "scripts" / "install_systemd.sh")], check=False)


def maybe_enroll_face(tiers: set[str], facts: dict) -> None:
    if "sensorium" not in tiers or not facts["webcam"]:
        return
    rule("8 · Enroll your face (optional)")
    explain(
        "The Sensorium recognizes you before the assistant engages. Enrollment captures a few\n"
        "frames from your webcam and stores an embedding in your data folder [dim](never uploaded)[/dim].")
    who = os.environ.get("ENGRAM_USER") or getpass.getuser()
    if Confirm.ask(f"Enroll now as '{who}'?", default=False):
        run([str(VENV_PY), str(REPO / "infra" / "engram" / "eye" / "face.py"), "enroll", who], check=False)
    else:
        c.print(f"[dim]  later: .venv/bin/python infra/engram/eye/face.py enroll {who}[/dim]")


def write_config(env: dict) -> None:
    rule("9 · Save configuration")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Engram configuration — written by the setup wizard.\n"]
    for k, v in env.items():
        lines.append(f"{k}={v}\n")
    ENV_FILE.write_text("".join(lines))
    os.chmod(ENV_FILE, 0o600)  # may hold an API key
    c.print(f"  [green]✓[/green] {ENV_FILE}")
    snippet = f"[ -f {ENV_FILE} ] && set -a && . {ENV_FILE} && set +a"
    shell = os.environ.get("SHELL", "")
    rc = None
    if shell.endswith("zsh"):            # macOS default + many Linux setups
        rc = Path(os.path.expanduser("~/.zshrc"))
    elif shell.endswith("bash"):
        rc = Path(os.path.expanduser("~/.bashrc"))
    if rc is not None and Confirm.ask(f"Add a line to {rc.name} so these load in new shells?", default=True):
        try:
            existing = rc.read_text() if rc.exists() else ""
            if str(ENV_FILE) not in existing:
                with rc.open("a") as f:
                    f.write(f"\n# Engram\n{snippet}\n")
            c.print(f"  [green]✓[/green] added to {rc.name} — open a new shell or `source` it")
        except OSError as e:
            c.print(f"[yellow]  couldn't edit {rc} ({e}); add this line yourself:[/yellow]\n  {snippet}")
    else:
        c.print(f"[dim]  add this to your shell profile ({shell or 'your shell'}):  {snippet}[/dim]")


def smoke_test(env: dict) -> None:
    rule("10 · Smoke test")
    envp = {**os.environ, **env}
    recall = REPO / ".venv" / "bin" / "recall"
    c.print("[dim]$ recall paths[/dim]")
    subprocess.run([str(recall), "paths"], env=envp, check=False)


def summary(tiers: set[str], env: dict) -> None:
    rule("Done")
    lines = ["[bold green]Engram is set up.[/bold green]\n"]
    lines.append(f"[dim]data:[/dim] {env.get('RECALL_DATA_ROOT', DEFAULT_DATA_ROOT)}    "
                 f"[dim]tiers:[/dim] {', '.join(sorted(tiers))}\n")
    lines.append("Next:")
    lines.append("  • [bold]recall build --global[/bold]    build the memory index")
    lines.append("  • open [bold]Claude Code[/bold]         memory now auto-injects each turn")
    if "assistant" in tiers:
        lines.append("  • [bold].venv/bin/python infra/engram/app.py[/bold]   launch the assistant")
    if "sensorium" in tiers:
        lines.append("  • add [bold]-p[/bold] to the assistant for camera perception")
    if "telegram" in tiers:
        lines.append("  • configure the bridge: [bold]bash scripts/install_telegram_bridge.sh[/bold]")
    lines.append("\n[dim]Re-run ./install.sh anytime to change your setup.[/dim]")
    c.print(Panel("\n".join(lines), border_style="green"))


def main() -> None:
    welcome()
    env: dict = {}
    facts = preflight()
    try:
        choose_auth(env, facts)
        choose_data_dir(env)
        tiers = choose_tiers(facts)
        install_deps(tiers)
        install_skills_and_hook()
        maybe_services(tiers, facts)
        maybe_enroll_face(tiers, facts)
        write_config(env)
        smoke_test(env)
        summary(tiers, env)
    except (KeyboardInterrupt, EOFError):
        c.print("\n[yellow]Setup interrupted — re-run ./install.sh to continue.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
