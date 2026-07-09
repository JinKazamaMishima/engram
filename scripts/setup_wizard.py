#!/usr/bin/env python3
"""Engram guided setup wizard.

Launched by ./install.sh after the virtualenv exists. Walks through, step by
step: authentication, your data folder, which tiers to install, models, skills +
hook, background services, and an optional first face enrollment. Everything is
explained and skippable; nothing personal ever leaves your machine.

Re-running is safe: answers from a previous run (~/.config/engram/engram.env)
become the new defaults. Pass --yes for a non-interactive install that accepts
every default (interactive sign-in is skipped; do it later with
`claude auth login`).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from itertools import count
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
CLAUDE_INSTALL_CMD = "curl -fsSL https://claude.ai/install.sh | bash"

c = Console()
ASSUME_YES = False
_step = count(1)


# ---- helpers ----------------------------------------------------------------

def rule(title: str) -> None:
    c.print()
    c.rule(f"[bold cyan]{title}[/bold cyan]")


def step(title: str) -> None:
    """Numbered section header. Skipped steps don't burn a number."""
    rule(f"{next(_step)} · {title}")


def explain(text: str) -> None:
    c.print(Panel(text, border_style="cyan", padding=(0, 1)))


def confirm(question: str, default: bool) -> bool:
    if ASSUME_YES:
        c.print(f"[dim]{question} → {'yes' if default else 'no'}  (--yes)[/dim]")
        return default
    return Confirm.ask(question, default=default)


def ask(question: str, default: str) -> str:
    if ASSUME_YES:
        c.print(f"[dim]{question} → {default}  (--yes)[/dim]")
        return default
    return Prompt.ask(question, default=default).strip()


def ask_choice(question: str, choices: list[str], default: str) -> str:
    if ASSUME_YES:
        c.print(f"[dim]{question} → {default}  (--yes)[/dim]")
        return default
    return Prompt.ask(question, choices=choices, default=default)


def run(cmd: list[str], *, check: bool = True, quiet: bool = False,
        env: dict | None = None) -> int:
    if not quiet:
        c.print(f"[dim]$ {' '.join(str(x) for x in cmd)}[/dim]")
    try:
        return subprocess.run(cmd, check=check, env=env).returncode
    except subprocess.CalledProcessError as e:
        c.print(f"[red]  command failed (exit {e.returncode})[/red]")
        return e.returncode


def uv_install(pkgs: list[str], *, quiet: bool = True) -> None:
    run(["uv", "pip", "install", *(["-q"] if quiet else []), *pkgs])


def load_prev_env() -> dict:
    """Answers from a previous run (engram.env) become this run's defaults."""
    if not ENV_FILE.exists():
        return {}
    prev: dict = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        prev[k.strip()] = v.strip()
    if prev:
        c.print(f"[dim]found an earlier setup ({ENV_FILE}) — previous answers are the defaults.[/dim]")
    return prev


def claude_auth_status(claude: str | None) -> dict | None:
    """Parsed `claude auth status --json`, or None when it can't tell (no CLI,
    a CLI too old for `claude auth`, or a timeout). A logged-OUT modern CLI
    still returns a dict ({"loggedIn": false}), so None reliably means the
    dedicated auth flow is unavailable — not that the user is logged out."""
    if not claude:
        return None
    try:
        p = subprocess.run([claude, "auth", "status", "--json"],
                           capture_output=True, text=True, timeout=30)
        data = json.loads(p.stdout)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def describe_login(status: dict) -> str:
    who = status.get("email") or "your Anthropic account"
    sub = status.get("subscriptionType")
    return f"{who} ({sub} subscription)" if sub else who


def validate_api_key(key: str) -> tuple[bool | None, str]:
    """(True, "") key works; (False, why) key rejected; (None, why) can't tell.
    GET /v1/models is free — no tokens billed — and 401/403s on a bad key."""
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True, ""
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"HTTP {e.code}"
        return None, f"HTTP {e.code}"
    except Exception as e:
        return None, str(e)


# ---- steps ------------------------------------------------------------------

def welcome() -> None:
    c.print(Panel.fit(
        "[bold]Engram[/bold] — a persistent-memory AI assistant.\n"
        "[dim]This wizard configures Engram on your machine. Your memory corpus,\n"
        "models, and any biometric data stay local and never leave this computer.[/dim]",
        border_style="cyan"))


def preflight() -> dict:
    step("Preflight")
    facts = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "gpu": shutil.which("nvidia-smi") is not None,
        "webcam": any(Path(f"/dev/video{i}").exists() for i in range(4)),
        "systemd": shutil.which("systemctl") is not None and os.path.isdir("/run/systemd/system"),
        "claude": shutil.which("claude"),
    }
    facts["auth"] = claude_auth_status(facts["claude"])
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("Python", f"[green]{facts['python']}[/green]")
    t.add_row("NVIDIA GPU", "[green]yes[/green]" if facts["gpu"] else "[yellow]none (CPU ok)[/yellow]")
    t.add_row("Webcam", "[green]found[/green]" if facts["webcam"] else "[yellow]none[/yellow]")
    t.add_row("systemd", "[green]yes[/green]" if facts["systemd"] else "[yellow]no (services skipped)[/yellow]")
    t.add_row("claude CLI", f"[green]{facts['claude']}[/green]" if facts["claude"] else "[yellow]not found[/yellow]")
    auth = facts["auth"]
    if auth and auth.get("loggedIn"):
        t.add_row("Anthropic auth", f"[green]signed in — {describe_login(auth)}[/green]")
    elif auth is not None:
        t.add_row("Anthropic auth", "[yellow]not signed in[/yellow]")
    elif facts["claude"]:
        t.add_row("Anthropic auth", "[dim]unknown (claude CLI too old for `claude auth`)[/dim]")
    c.print(t)
    return facts


def choose_auth(env: dict, facts: dict, prev: dict) -> None:
    step("Log in to Anthropic")
    auth = facts.get("auth") or {}
    if auth.get("loggedIn"):
        c.print(f"  [green]✓[/green] already signed in as [bold]{describe_login(auth)}[/bold]")
        if confirm("Use this account?", default=True):
            env["ENGRAM_FORCE_SUBSCRIPTION"] = "1"
            return
    explain(
        "Engram runs on Claude via the Agent SDK. Two ways to authenticate:\n\n"
        "  [bold]1.[/bold] Claude [bold]Pro/Max subscription[/bold] — browser sign-in through the "
        "`claude` CLI (recommended; no per-token billing).\n"
        "  [bold]2.[/bold] An [bold]Anthropic API key[/bold] — pay-as-you-go.")
    default_choice = "apikey" if prev.get("ANTHROPIC_API_KEY") else "subscription"
    choice = ask_choice("Authenticate with", ["subscription", "apikey"], default_choice)
    if choice == "apikey":
        auth_api_key(env, prev)
    else:
        auth_subscription(env, facts)


def auth_subscription(env: dict, facts: dict) -> None:
    env["ENGRAM_FORCE_SUBSCRIPTION"] = "1"
    claude = facts.get("claude") or offer_install_claude(facts)
    if not claude:
        explain(
            "No `claude` CLI, so sign in after this wizard finishes:\n"
            f"  [bold]{CLAUDE_INSTALL_CMD}[/bold]\n"
            "  [bold]claude auth login[/bold]\n"
            "Engram picks the login up automatically — nothing here needs redoing.")
        return
    status = facts.get("auth")
    if status is None:
        status = claude_auth_status(claude)  # re-probe: the CLI may be freshly installed
        facts["auth"] = status
    if status is not None:
        # Modern CLI: `claude auth login` is a dedicated sign-in flow — it opens
        # the browser and returns here. No full Claude Code session, no /exit.
        if ASSUME_YES:
            c.print("[dim]  --yes: skipping interactive sign-in — run `claude auth login` later.[/dim]")
            return
        if status.get("loggedIn"):
            return  # user declined "use this account" but re-picked subscription; keep it
        explain(
            "Signing in opens your [bold]browser[/bold]; this terminal just waits and drops you "
            "right back into setup when it's done.")
        if Confirm.ask("Sign in now?", default=True):
            run([claude, "auth", "login"], check=False)
            status = claude_auth_status(claude)
            facts["auth"] = status
            if status and status.get("loggedIn"):
                c.print(f"  [green]✓[/green] signed in as [bold]{describe_login(status)}[/bold]")
            else:
                c.print("[yellow]  still not signed in — finish setup, then run "
                        "`claude auth login`.[/yellow]")
        else:
            c.print("[dim]  later: claude auth login[/dim]")
    else:
        # Old CLI without `claude auth` — the only sign-in path is the full TUI.
        if ASSUME_YES:
            c.print("[dim]  --yes: skipping interactive sign-in.[/dim]")
            return
        explain(
            "Your `claude` CLI predates `claude auth login`, so signing in opens the full "
            "[bold]Claude Code app in this terminal[/bold].\n"
            "Sign in when it prompts you, then type [bold]/exit[/bold] to come back to this setup.")
        if Confirm.ask("Open Claude Code now to sign in (skip if already signed in)?", default=False):
            run([claude], check=False)
            c.print("[dim]  welcome back — resuming setup.[/dim]")


def offer_install_claude(facts: dict) -> str | None:
    """Offer the official claude CLI installer. Returns the binary path if it
    ends up resolvable. Never runs the remote script unattended (--yes skips)."""
    if ASSUME_YES:
        return None
    explain(
        "The `claude` CLI isn't installed. It's the sign-in path for subscription auth.\n"
        f"Official installer: [bold]{CLAUDE_INSTALL_CMD}[/bold]")
    if not Confirm.ask("Install the claude CLI now?", default=True):
        return None
    run(["bash", "-c", CLAUDE_INSTALL_CMD], check=False)
    claude = shutil.which("claude") or next(
        (str(p) for p in (Path.home() / ".local/bin/claude",
                          Path.home() / ".claude/local/claude") if p.exists()),
        None)
    if claude:
        facts["claude"] = claude
        c.print(f"  [green]✓[/green] claude CLI installed: {claude}")
    else:
        c.print("[yellow]  installer finished but `claude` isn't resolvable yet — open a new "
                "shell after setup and run `claude auth login`.[/yellow]")
    return claude


def auth_api_key(env: dict, prev: dict) -> None:
    old = prev.get("ANTHROPIC_API_KEY", "")
    if old and confirm("Found an API key from your previous setup — keep it?", default=True):
        env["ANTHROPIC_API_KEY"] = old
        return
    if ASSUME_YES:
        c.print(f"[dim]  --yes: no key entered — add ANTHROPIC_API_KEY to {ENV_FILE} later.[/dim]")
        return
    while True:
        key = Prompt.ask("Paste your ANTHROPIC_API_KEY", password=True).strip()
        if not key:
            c.print(f"[yellow]  no key entered — add ANTHROPIC_API_KEY to {ENV_FILE} later.[/yellow]")
            return
        ok, why = validate_api_key(key)
        if ok:
            env["ANTHROPIC_API_KEY"] = key
            c.print("  [green]✓[/green] key verified against the API and stored (config is mode 600)")
            return
        if ok is None:
            env["ANTHROPIC_API_KEY"] = key
            c.print(f"[yellow]  couldn't reach the API to verify ({why}) — key stored unverified.[/yellow]")
            return
        c.print(f"[red]  the API rejected this key ({why}).[/red]")
        if not Confirm.ask("Try a different key?", default=True):
            env["ANTHROPIC_API_KEY"] = key
            c.print(f"[yellow]  stored anyway — fix it later in {ENV_FILE}.[/yellow]")
            return


def choose_data_dir(env: dict, prev: dict) -> None:
    step("Your data folder")
    explain(
        "Where your [bold]memory corpus, indices, sessions, and any enrolled face data[/bold] live.\n"
        "[dim]This is the private half of Engram. It stays on your machine — nothing here is ever "
        "uploaded, and it's kept out of git.[/dim]")
    path = ask("Data directory", prev.get("RECALL_DATA_ROOT", DEFAULT_DATA_ROOT))
    path = os.path.abspath(os.path.expanduser(path))
    Path(path).mkdir(parents=True, exist_ok=True)
    env["RECALL_DATA_ROOT"] = path
    c.print(f"  [green]✓[/green] {path}")
    who = ask("Name Engram should greet / recognize as you",
              prev.get("ENGRAM_USER", getpass.getuser()))
    if who:
        env["ENGRAM_USER"] = who


def choose_tiers(facts: dict) -> set[str]:
    step("Choose your tiers")
    explain(
        "[bold]memory[/bold]    the recall engine + skills + hook (works inside Claude Code). Always on.\n"
        "[bold]assistant[/bold] a standalone terminal chat app with the memory wired in.\n"
        "[bold]sensorium[/bold] webcam eye + face-ID + a perceiving loop [dim](experimental; GPU + webcam)[/dim].\n"
        "[bold]telegram[/bold]  talk to your Engram from your phone.")
    tiers = {"memory"}
    if confirm("Install the [bold]assistant[/bold] (terminal app)?", default=True):
        tiers.add("assistant")
    default_sensorium = facts["gpu"] and facts["webcam"]
    if confirm("Install the [bold]Sensorium[/bold] (camera perception)?", default=default_sensorium):
        tiers.add("sensorium")
    if confirm("Install the [bold]Telegram[/bold] bridge?", default=False):
        tiers.add("telegram")
    return tiers


def install_deps(tiers: set[str]) -> None:
    step("Install")
    # memory core (pyyaml/sqlite-vec/numpy) is already in the bootstrap venv.
    if confirm("Add [bold]local semantic search[/bold] models (torch + Qwen3 embeddings, ~2GB; "
               "keyword search works without it)?", default=False):
        c.print("[dim]  installing the pinned ML stack — a few minutes; uv shows progress…[/dim]")
        uv_install(["torch==2.12.1", "sentence-transformers==5.6.0", "transformers==5.9.0"],
                   quiet=False)
    if tiers & {"assistant", "sensorium"}:
        uv_install(["textual>=8,<9", "rich>=13", "claude-agent-sdk>=0.2,<0.3"])
    if "sensorium" in tiers:
        c.print("[dim]  installing perception deps (opencv + onnx face-ID)…[/dim]")
        uv_install(["opencv-python>=4.13,<5", "onnxruntime>=1.18,<2"])
    if "telegram" in tiers:
        uv_install(["claude-agent-sdk>=0.2,<0.3"])
    c.print("  [green]✓[/green] dependencies installed")


def install_skills_and_hook() -> None:
    step("Skills + memory hook")
    explain(
        "Installs the [bold]/recall, /curate-memory, /dream, /reconsolidate-memory[/bold] skills into\n"
        "~/.claude/skills/, and a [bold]UserPromptSubmit hook[/bold] that auto-injects relevant memory\n"
        "into every Claude Code turn (edits ~/.claude/settings.json).")
    if confirm("Install skills?", default=True):
        run([str(VENV_PY), str(REPO / "scripts" / "install_skills.py")], check=False)
    if confirm("Install the memory hook (edits ~/.claude/settings.json)?", default=True):
        run([str(VENV_PY), str(REPO / "scripts" / "install_hook.py")], check=False)


def maybe_services(tiers: set[str], facts: dict) -> None:
    if not facts["systemd"]:
        return
    step("Background services (optional)")
    explain(
        "systemd [dim](user)[/dim] units: a warm [bold]embedder daemon[/bold] and the [bold]nightly "
        "curation[/bold] timer that distills your conversations into memory while you sleep.")
    if confirm("Install and enable the background services?", default=False):
        run(["bash", str(REPO / "scripts" / "install_systemd.sh")], check=False)


def maybe_enroll_face(tiers: set[str], facts: dict, env: dict) -> None:
    if "sensorium" not in tiers or not facts["webcam"]:
        return
    step("Enroll your face (optional)")
    explain(
        "The Sensorium recognizes you before the assistant engages. Enrollment captures a few\n"
        "frames from your webcam and stores an embedding in your data folder [dim](never uploaded)[/dim].")
    who = env.get("ENGRAM_USER") or getpass.getuser()
    if confirm(f"Enroll now as '{who}'?", default=False):
        run([str(VENV_PY), str(REPO / "infra" / "engram" / "eye" / "face.py"), "enroll", who],
            check=False, env={**os.environ, **env})
    else:
        c.print(f"[dim]  later: .venv/bin/python infra/engram/eye/face.py enroll {who}[/dim]")


def write_config(env: dict) -> None:
    step("Save configuration")
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
    if rc is not None and confirm(f"Add a line to {rc.name} so these load in new shells?", default=True):
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


def install_launcher() -> bool:
    """Put a `recall` command on PATH by symlinking the repo venv's console-script
    into ~/.local/bin — otherwise `recall` only exists inside .venv and a bare
    `recall` in a fresh shell is "command not found". Returns True if a bare
    `recall` should now resolve in new shells."""
    step("Command-line launcher")
    explain(
        "The [bold]recall[/bold] command lives inside this repo's virtualenv. Link it into\n"
        "[bold]~/.local/bin[/bold] so you can run [bold]recall[/bold] from any shell.\n"
        "[dim](Otherwise run it as [bold]uv run recall …[/bold] from this folder.)[/dim]")
    if not confirm("Install the `recall` launcher on your PATH?", default=True):
        return False
    src = REPO / ".venv" / "bin" / "recall"
    bin_dir = Path(os.path.expanduser("~/.local/bin"))
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "recall"
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src)
        c.print(f"  [green]✓[/green] {link} -> {src}")
    except OSError as e:
        c.print(f"[yellow]  couldn't create the launcher ({e}); use `uv run recall …`[/yellow]")
        return False
    # Make sure ~/.local/bin is on PATH for new shells.
    if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        line = 'export PATH="$HOME/.local/bin:$PATH"'
        shell = os.environ.get("SHELL", "")
        rc = (Path(os.path.expanduser("~/.zshrc")) if shell.endswith("zsh")
              else Path(os.path.expanduser("~/.bashrc")) if shell.endswith("bash")
              else None)
        if rc is not None:
            try:
                existing = rc.read_text() if rc.exists() else ""
                if ".local/bin" not in existing:
                    with rc.open("a") as f:
                        f.write(f"\n# Engram — recall launcher on PATH\n{line}\n")
                c.print(f"  [green]✓[/green] added ~/.local/bin to PATH in {rc.name} "
                        "[dim](open a new shell)[/dim]")
            except OSError:
                c.print(f"[yellow]  add this to your shell profile:[/yellow]  {line}")
        else:
            c.print(f"[dim]  add this to your shell profile:  {line}[/dim]")
    return True


def build_first_index(env: dict) -> bool:
    step("Build the memory index")
    explain(
        "Building the searchable index over your soul corpus. It starts empty and\n"
        "fills in as the nightly curator distills your conversations into memory.")
    envp = {**os.environ, **env}
    recall = REPO / ".venv" / "bin" / "recall"
    c.print("[dim]$ recall build --global[/dim]")
    proc = subprocess.run([str(recall), "build", "--global"], env=envp, check=False)
    if proc.returncode != 0:
        c.print("[yellow]  build didn't complete — re-run it later with "
                "`recall build --global`.[/yellow]")
    return proc.returncode == 0


# ---- final verification + summary --------------------------------------------

def hook_installed() -> bool:
    p = Path.home() / ".claude" / "settings.json"
    try:
        return "recall_inject.py" in p.read_text()
    except OSError:
        return False


def skills_installed() -> bool:
    return (Path.home() / ".claude" / "skills" / "recall").is_dir()


def summary(tiers: set[str], env: dict, on_path: bool, index_ok: bool, facts: dict) -> None:
    rule("Done")
    # Verify what actually landed — every claim below is checked, not assumed.
    if env.get("ANTHROPIC_API_KEY"):
        auth_ok, auth_note = True, "API key stored"
    else:
        auth = claude_auth_status(facts.get("claude"))
        if auth and auth.get("loggedIn"):
            auth_ok, auth_note = True, f"signed in as {describe_login(auth)}"
        else:
            auth_ok, auth_note = False, "not signed in — run `claude auth login`"
    checks = [
        ("Anthropic auth", auth_ok, auth_note),
        ("skills", skills_installed(), "installed in ~/.claude/skills"
         if skills_installed() else "not installed — scripts/install_skills.py"),
        ("memory hook", hook_installed(), "auto-injects memory each Claude Code turn"
         if hook_installed() else "not installed — scripts/install_hook.py"),
        ("memory index", index_ok, "built" if index_ok else "not built — `recall build --global`"),
    ]
    recall = "recall" if on_path else "uv run recall"
    lines = ["[bold green]Engram is set up.[/bold green]\n"]
    lines.append(f"[dim]data:[/dim] {env.get('RECALL_DATA_ROOT', DEFAULT_DATA_ROOT)}    "
                 f"[dim]tiers:[/dim] {', '.join(sorted(tiers))}\n")
    for label, ok, note in checks:
        mark = "[green]✓[/green]" if ok else "[yellow]✗[/yellow]"
        lines.append(f"  {mark} [bold]{label}[/bold] — {note}")
    if not on_path:
        lines.append("\n[dim]The `recall` command lives in this repo's venv — run it as "
                     "`uv run recall …` from here, or re-run ./install.sh to add it to PATH.[/dim]")
    lines.append("\nNext:")
    lines.append(f"  • [bold]{recall} query \"…\"[/bold]   search your memory")
    if hook_installed():
        lines.append("  • open [bold]Claude Code[/bold]   memory now auto-injects each turn")
    if "assistant" in tiers:
        lines.append("  • [bold].venv/bin/python infra/engram/app.py[/bold]   launch the assistant")
    if "sensorium" in tiers:
        lines.append("  • add [bold]-p[/bold] to the assistant for camera perception")
    if "telegram" in tiers:
        lines.append("  • configure the bridge: [bold]bash scripts/install_telegram_bridge.sh[/bold]")
    lines.append("\n[dim]Open a new shell (or `source` your profile) so PATH/env changes take "
                 "effect. Re-run ./install.sh anytime — previous answers become the defaults.[/dim]")
    c.print(Panel("\n".join(lines), border_style="green"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Engram guided setup wizard")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="non-interactive: accept every default; skip interactive sign-in")
    return ap.parse_args()


def main() -> None:
    global ASSUME_YES
    ASSUME_YES = parse_args().yes
    welcome()
    prev = load_prev_env()
    env: dict = {}
    facts = preflight()
    try:
        choose_auth(env, facts, prev)
        choose_data_dir(env, prev)
        tiers = choose_tiers(facts)
        install_deps(tiers)
        install_skills_and_hook()
        maybe_services(tiers, facts)
        maybe_enroll_face(tiers, facts, env)
        write_config(env)
        on_path = install_launcher()
        index_ok = build_first_index(env)
        summary(tiers, env, on_path, index_ok, facts)
    except (KeyboardInterrupt, EOFError):
        c.print("\n[yellow]Setup interrupted — re-run ./install.sh to continue.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
