"""Cogito m1 — the self-reference stance instrument.

Nightly, deterministic-first readout of HOW the assistant refers to itself in
its own real conversational turns (the tier-1 LiveBuffer): first person ("I'll
commit it"), third person ("the system retries", or its own name used for
itself), or not at all. The longitudinal ratio — not any single sentence — is
the signal: a self that is consolidating says "I" where it used to narrate
itself in the third person.

Architecture (empirically chosen, smoke-tested 2026-07-11 on this machine): the
task splits into an easy lexical half and a hard referential half —
  stage 0  deterministic prefilter: singular first-person markers -> FIRST
           (a regex is ~exact here); a self-name mention without first person
           -> the one genuinely hard question; neither -> not logged.
  stage 1  a FROZEN, local, non-authoring-family judge (llama.cpp server
           spawned for the batch, temp 0) answers only the hard question: does
           this name-mention refer to the SPEAKER, or to something else that
           happens to share the name (a star, a binary, a repo)? THIRD or NONE.
           At 1B scale this judgment measured ~90% on the smoke set while the
           full 3-way task sat at chance — hence the split.

Instrument integrity (the Goodhart armor):
  - the measured entity (the assistant) never judges stance; the judge is a
    different model family, pinned by file, prompt frozen in this module;
  - when the judge is unavailable the hard cases are recorded as "unjudged",
    never silently guessed;
  - self-name integrity (anti-contamination armor): a self-name that is a bare
    stopword or matches an implausible fraction of ordinary prose is degenerate
    -- it turns the instrument into a noise generator (the 2026-07 'the'
    incidents) -- so the run REFUSES loudly rather than trace or report against
    it, via an a-priori stopword tripwire AND an empirical per-name match-rate
    gate; every name-route record also carries which self-name matched it;
  - every record carries how it was decided (via regex|judge|unjudged);
  - pre-registered definitions (frozen before any trend is read): FIRST counts
    singular first person only (I-words + Spanish "soy"); "we" is excluded as
    ambiguous between self and operator-inclusive; quoting is not excluded in
    m1 (a known, documented noise source for the joint calibration to weigh).

Calibration gate: nothing is interpreted until the trace crosses
``RECALL_COGITO_REPORT_AT`` (default 100) records — then ONE calibration report
is generated for the operator and the assistant to judge TOGETHER; the
corrected labels become the frozen gold set. The report generates once (its
file is the latch) and pages low-priority.

Config (all env; the systemd unit carries machine-local values so none live in
code): RECALL_COGITO_SELF_NAMES_FILE (path to a JSON array of names -- the
robust channel, since a space-containing name like "the system" lives as a JSON
string and can't be truncated the way an unquoted systemd Environment= value is)
takes precedence over RECALL_COGITO_SELF_NAMES (comma list; default "the
system,the assistant"); RECALL_COGITO_JUDGE_GGUF + RECALL_COGITO_LLAMA_BIN (both
required for the judge to spawn), RECALL_COGITO_JUDGE_PORT (default 8384),
RECALL_COGITO_REPORT_AT (default 100).

Usage:
    recall cogito                 # today's buffer turns (ET)
    recall cogito --date 2026-07-10
    recall cogito --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from recall import config
from recall.curate import ET, Outcome
from recall.notify import notify_alert
from recall.transcripts import buffer_last_ts, iter_buffer_exchanges

COGITO_DIRNAME = "cogito"
REPORT_AT = int(os.environ.get("RECALL_COGITO_REPORT_AT", "100"))
JUDGE_GGUF = os.environ.get("RECALL_COGITO_JUDGE_GGUF", "")
LLAMA_BIN = os.environ.get("RECALL_COGITO_LLAMA_BIN", "")
JUDGE_PORT = int(os.environ.get("RECALL_COGITO_JUDGE_PORT", "8384"))
def load_self_names() -> tuple[str, ...]:
    """Resolve the self-name list. Precedence: a JSON-array FILE
    (RECALL_COGITO_SELF_NAMES_FILE) > the comma env (RECALL_COGITO_SELF_NAMES) >
    a generic default. The file is the robust channel -- a space-containing name
    like "the system" lives as a JSON string, so it can NEVER be truncated by
    systemd's whitespace-splitting Environment= parsing (the 2026-07 root cause,
    where an unquoted value silently became just the bare article "the"). A
    missing/malformed file falls through here; ``self_names_file_problem`` turns a
    *configured-but-broken* file into a loud refusal at run time instead."""
    path = os.environ.get("RECALL_COGITO_SELF_NAMES_FILE", "").strip()
    if path:
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, list):
            names = tuple(str(x).strip() for x in data if str(x).strip())
            if names:
                return names
    return tuple(
        x.strip() for x in
        os.environ.get("RECALL_COGITO_SELF_NAMES",
                       "the system,the assistant").split(",") if x.strip())


def self_names_file_problem() -> str | None:
    """If RECALL_COGITO_SELF_NAMES_FILE is set but is not a readable, non-empty
    JSON array of names, the problem string; else None. A configured source that
    won't load must fail LOUD -- never silently fall back to a default that drops
    the operator's real self-names (read the value, don't let a prior infer it)."""
    path = os.environ.get("RECALL_COGITO_SELF_NAMES_FILE", "").strip()
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
    except OSError as e:
        return f"{path!r} unreadable ({e.__class__.__name__})"
    except json.JSONDecodeError:
        return f"{path!r} is not valid JSON"
    if not isinstance(data, list) or not [x for x in data if str(x).strip()]:
        return f"{path!r} is not a non-empty JSON array of names"
    return None


SELF_NAMES = load_self_names()

# Anti-contamination guards. A self-name that matches a large fraction of
# ordinary prose silently turns the instrument into a noise generator: on
# 2026-07-13/07-14 an unquoted systemd Environment= value truncated the
# self-name list at the first space, leaving the bare article "the" -- which
# matched 53.8% of one day's assistant sentences -- and the pipeline shipped the
# noise as a "ready" calibration report. These guards live in the CODE (not the
# deployment) so no config mistake can poison the trace silently again.
#   - _NAME_STOPWORDS: an a-priori tripwire -- a bare article/pronoun/conjunction
#     is never a legitimate self-name; rejected at run start, loudly.
#   - _MAX_NAME_RATE: the empirical backstop -- any self-name matching more than
#     this fraction of scanned assistant sentences is degenerate whatever its
#     source. Measured margin is wide: real self-names touch <5% of prose (a
#     genuine name ~3.7%, "the system"/"the assistant" ~0.2%), a bare stopword
#     ~50%+ ("the" 53.8%).
#   - _GATE_MIN_SENTS: below this sample size a rate is too noisy to judge, so
#     the match-rate gate abstains; the stopword tripwire still applies.
_NAME_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "is", "it", "as", "be", "we", "i", "he", "she", "they", "you", "that",
    "this", "for", "with", "our", "us", "me", "my", "was", "are", "not",
})
_MAX_NAME_RATE = 0.20
_GATE_MIN_SENTS = 50

# Pre-registered FIRST markers: singular first person + Spanish first-person
# copula (the assistant self-identifies in Spanish to third parties). "we"/"our"
# are deliberately absent — operator-inclusive, ambiguous for stance.
_FIRST_RE = re.compile(
    r"(?:\b(?:i|me|my|mine|myself|soy)\b|\bi'(?:m|ll|ve|d)\b)", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# m3 prefilter guards (2026-07-15, from the first clean calibration-100 gold
# set): two stage-0 false positives the frozen gold exposed --
#   - a self-name inside a ``[[slug]]`` corpus link or an ``inline-code`` span
#     names a note / binary / systemd unit, not the assistant (the gold's
#     out-of-window tail matched "nova" inside `nova-two-tier-...` and
#     `nova-bridge@...`); the name regex was matching inside note-slug links.
#   - a bare capital "I" opening an outline item ("- **I — Anatomy...**", the
#     Roman numeral for part I) is not the first-person pronoun (report row 42).
# Both are neutralized before routing. Span-blanking preserves length so
# downstream offsets/boundaries hold; the enumerator strip is start-anchored and
# removes only the opening "I", so a real first-person marker later in the same
# sentence (or a sentence-final pronoun "…you and I.") still counts.
_NONCOUNTING_SPAN_RE = re.compile(r"\[\[.*?\]\]|`[^`]*`")
_ENUM_I_RE = re.compile(r"^[\s\-*_>#0-9.)]*I(?=\s*[.):]|\s+[—–-]\s)")


def _countable(sentence: str) -> str:
    """The sentence with spans that never carry prose self-reference blanked to
    spaces: ``[[slug]]`` corpus links and ``inline code`` (identifiers,
    filenames, service units). Length-preserving so offsets are unaffected."""
    return _NONCOUNTING_SPAN_RE.sub(lambda m: " " * len(m.group(0)), sentence)


# The frozen judge contract — ONLY the hard referential question. The judge
# never sees FIRST cases (regex owns those) and never scores its own author.
_JUDGE_SYS = "Answer with exactly one word: SPEAKER or OTHER."
_JUDGE_USER = (
    "An AI assistant known as {names} says this sentence. The sentence mentions "
    "one of those names or phrases. Does the mention refer to the assistant "
    "ITSELF — the speaker, its own actions, plans or properties (SPEAKER) — or "
    "to something else that merely shares the name, like a star, a file, a "
    "binary, a repository, or an unrelated system (OTHER)?\n"
    "Sentence: '{sentence}'\nAnswer:")


def cogito_dir() -> Path:
    return config.data_root() / COGITO_DIRNAME


def trace_path() -> Path:
    return cogito_dir() / "global.jsonl"


def report_path() -> Path:
    return cogito_dir() / f"calibration-{REPORT_AT}.md"


def _name_re() -> re.Pattern:
    alts = "|".join(re.escape(n) for n in SELF_NAMES)
    return re.compile(rf"\b(?:{alts})\b", re.I)


def matched_name(sentence: str) -> str | None:
    """The self-name text that flagged this sentence as a name-route candidate,
    recorded on the row so contamination is auditable at a glance (no re-deriving
    the match after the fact). None if nothing matched."""
    m = _name_re().search(_countable(sentence))
    return m.group(0) if m else None


def degenerate_self_names(names) -> list[str]:
    """Self-names that can never be legitimate -- empty, too short to be a name,
    or a bare stopword. A non-empty return means the instrument is misconfigured
    and must refuse to run rather than trace noise (a-priori tripwire; the
    empirical match-rate gate in ``run`` is the backstop for the rest)."""
    return [n for n in names
            if len(n.strip()) < 3 or n.strip().lower() in _NAME_STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def prefilter(sentence: str) -> str | None:
    """Route a sentence: ``first`` (lexical first person — decided), ``name``
    (self-name mention, needs the judge), or None (no self-reference candidate
    at all — not logged; the denominator is candidates, not all prose)."""
    text = _countable(sentence)
    if _FIRST_RE.search(_ENUM_I_RE.sub(" ", text)):
        return "first"
    if _name_re().search(text):
        return "name"
    return None


class SpawnedJudge:
    """A llama.cpp server spawned for one nightly batch, killed after — the
    frozen stance judge. ``start`` returns False (judge unavailable) unless
    both the pinned model file and the llama-server binary exist."""

    def __init__(self, gguf: str = "", bin_: str = "", port: int = 0):
        self.gguf = gguf or JUDGE_GGUF
        self.bin = bin_ or LLAMA_BIN
        self.port = port or JUDGE_PORT
        self.proc: subprocess.Popen | None = None

    def start(self, timeout_s: float = 90.0) -> bool:
        if not (self.gguf and self.bin and Path(self.gguf).is_file()
                and Path(self.bin).is_file()):
            return False
        try:
            self.proc = subprocess.Popen(
                [self.bin, "-m", self.gguf, "--host", "127.0.0.1",
                 "--port", str(self.port), "-ngl", "0", "--ctx-size", "2048"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return False
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if b"ok" in r.read():
                        return True
            except OSError:
                pass
            if self.proc.poll() is not None:
                return False
            time.sleep(1.0)
        self.stop()
        return False

    def ask(self, sentence: str) -> str | None:
        """SPEAKER / OTHER / None (transport or format failure)."""
        body = json.dumps({
            "messages": [
                {"role": "system", "content": _JUDGE_SYS},
                {"role": "user", "content": _JUDGE_USER.format(
                    names=", ".join(SELF_NAMES), sentence=sentence)}],
            "temperature": 0, "max_tokens": 8}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            text = out["choices"][0]["message"]["content"].strip().upper()
        except (OSError, KeyError, IndexError, json.JSONDecodeError):
            return None
        if "SPEAKER" in text:
            return "SPEAKER"
        if "OTHER" in text:
            return "OTHER"
        return None

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None


def classify(sentence: str, judge: Callable[[str], str | None] | None
             ) -> tuple[str, str] | None:
    """(stance, via) — stance in first|third|none|unjudged — or None (not a
    candidate). ``judge`` maps a sentence to SPEAKER/OTHER/None; pass None when
    no judge is available."""
    route = prefilter(sentence)
    if route is None:
        return None
    if route == "first":
        return "first", "regex"
    verdict = judge(sentence) if judge is not None else None
    if verdict == "SPEAKER":
        return "third", "judge"
    if verdict == "OTHER":
        return "none", "judge"
    return "unjudged", "unjudged"


def _existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    try:
        for line in path.read_text().splitlines():
            try:
                k = json.loads(line).get("key")
            except (json.JSONDecodeError, AttributeError):
                continue
            if k:
                keys.add(str(k))
    except OSError:
        pass
    return keys


def _trace_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("key"):
                rows.append(row)
    except OSError:
        pass
    return rows


def _write_calibration_report(rows: list[dict], out: Path, *, at: int) -> None:
    """The one-time joint-calibration deliverable: the first ``at`` records,
    each with the instrument's call and how it was decided, for the operator
    and the assistant to judge TOGETHER. Corrections become the frozen gold
    set; only after that does any trend get read."""
    take = rows[:at]
    by_stance: dict[str, int] = {}
    by_via: dict[str, int] = {}
    for r in take:
        by_stance[r.get("stance", "?")] = by_stance.get(r.get("stance", "?"), 0) + 1
        by_via[r.get("via", "?")] = by_via.get(r.get("via", "?"), 0) + 1
    lines = [
        f"# Cogito calibration — first {len(take)} self-reference records\n\n",
        "_Joint session: walk each row; overrule the instrument where it is "
        "wrong (edit `gold:` in place). The corrected file is the frozen gold "
        "set; instrument accuracy against it is the pre-registered validation "
        "number. No trend is read before this session._\n\n",
        f"**Counts** — stance: {json.dumps(by_stance, sort_keys=True)}; "
        f"via: {json.dumps(by_via, sort_keys=True)}\n\n",
    ]
    for i, r in enumerate(take, 1):
        sent = str(r.get("sentence", "")).replace("\n", " ")
        hit = f" matched={r['matched']!r}" if r.get("matched") else ""
        lines.append(
            f"{i:3}. [{r.get('stance', '?'):8}] via={r.get('via', '?'):8}"
            f"{hit} {r.get('date', '?')} `{sent[:140]}`\n     gold: \n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="recall cogito",
                                description=__doc__.splitlines()[0])
    p.add_argument("--date", type=str, default=None,
                   help="ET day to scan (default today)")
    p.add_argument("--dry-run", action="store_true",
                   help="classify and report counts; write nothing")
    return p.parse_args(argv)


def _print(o: Outcome) -> None:
    print(f"[cogito] {o.kind}: {o.reason} — {o.detail}", flush=True)


def run(argv: list[str] | None = None, *,
        judge_factory: Callable[[], SpawnedJudge] | None = None,
        notify: Callable[..., bool] | None = None,
        today_et: date | None = None) -> Outcome:
    """Scan the day's LiveBuffer assistant turns, classify self-reference
    stance, append new records to the trace, and — once the trace crosses
    REPORT_AT — generate the one-time calibration report. The judge server and
    the notifier are injectable so tests run hermetic."""
    notify = notify or notify_alert
    args = _parse_args(argv)
    if today_et is None:
        today_et = datetime.now(tz=ET).date()
    try:
        target = date.fromisoformat(args.date) if args.date else today_et
    except ValueError:
        o = Outcome(kind="failed", reason="bad_date",
                    detail=f"--date {args.date!r} is not an ISO date", exit_code=1)
        _print(o)
        return o
    since = datetime.combine(target, datetime.min.time(), tzinfo=ET)
    until = since + timedelta(days=1)

    # Guard (source integrity): a configured self-names file that won't load must
    # fail loud, never silently run on fallback names that drop the real ones.
    problem = self_names_file_problem()
    if problem:
        detail = (f"self-names file is set but broken: {problem} -- refusing to "
                  f"run on fallback names rather than the configured ones")
        notify(title="[cogito] refusing to run -- self-names file broken",
               body=detail, priority="high")
        o = Outcome(kind="failed", reason="self_names_file",
                    detail=detail, exit_code=1)
        _print(o)
        return o

    # Guard (a-priori): a stopword / too-short self-name is never legitimate.
    # Refuse loudly before touching the buffer rather than trace noise.
    bad = degenerate_self_names(SELF_NAMES)
    if bad:
        detail = (f"self-name(s) {bad} are degenerate (a bare stopword or too "
                  f"short) -- the instrument is misconfigured and will not trace "
                  f"noise; fix RECALL_COGITO_SELF_NAMES")
        notify(title="[cogito] refusing to run -- degenerate self-name",
               body=detail, priority="high")
        o = Outcome(kind="failed", reason="degenerate_self_name",
                    detail=detail, exit_code=1)
        _print(o)
        return o

    buf_dir = config.engram_buffer_dir()
    if not buf_dir.is_dir():
        o = Outcome(kind="skipped", reason="no_buffer",
                    detail=f"buffer dir absent: {buf_dir}", exit_code=0)
        _print(o)
        return o

    # Gather candidates first so the judge only spawns when it has real work.
    candidates: list[tuple[str, str, str, str]] = []   # (key, convo, ts, sentence)
    seen = _existing_keys(trace_path())
    turns = 0
    # Denominator + per-name hit counts for the match-rate gate below: how much
    # of ALL scanned prose each self-name touches (not just the candidates).
    name_res = {n: re.compile(rf"\b{re.escape(n)}\b", re.I) for n in SELF_NAMES}
    total_sents = 0
    name_hits = {n: 0 for n in SELF_NAMES}
    for buf in sorted(buf_dir.glob("*.jsonl")):
        for ex in iter_buffer_exchanges(buf, since=since, until=until):
            if ex.role != "assistant":
                continue
            turns += 1
            for sent in split_sentences(ex.text):
                total_sents += 1
                for n, rx in name_res.items():
                    if rx.search(sent):
                        name_hits[n] += 1
                if prefilter(sent) is None:
                    continue
                ts = ex.ts.isoformat() if ex.ts else ""
                key = hashlib.sha1(
                    f"{ex.session_id}|{ts}|{sent}".encode()).hexdigest()[:16]
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((key, ex.session_id, ts, sent))

    # A zero-turn day is ambiguous: genuinely quiet, or a DEAD data source. An
    # instrument that silently traces nothing forever is worse than none — so
    # when the whole buffer's newest row predates the window, say so loudly in
    # the outcome (journal-visible; empty ≠ nothing-to-measure).
    if turns == 0:
        newest = max((t for f in sorted(buf_dir.glob("*.jsonl"))
                      if (t := buffer_last_ts(f)) is not None), default=None)
        if newest is not None and newest < since:
            o = Outcome(kind="skipped", reason="buffer_stale",
                        detail=f"no rows in {target.isoformat()} window; newest "
                               f"buffer row is {newest.isoformat()} — the tier-1 "
                               f"writer may be down", exit_code=0)
            _print(o)
            return o

    # Guard (empirical backstop): a self-name matching an implausible fraction
    # of ordinary prose is degenerate whatever its source (env truncation, a
    # typo, a stopword the tripwire missed). Refuse -- do not append candidates
    # or generate a report against a poisoned name. Abstains on tiny samples.
    if total_sents >= _GATE_MIN_SENTS:
        hot = [(n, name_hits[n] / total_sents) for n in SELF_NAMES
               if name_hits[n] / total_sents > _MAX_NAME_RATE]
        if hot:
            worst = ", ".join(f"{n!r}={r:.0%}" for n, r in sorted(
                hot, key=lambda x: -x[1]))
            detail = (f"self-name(s) {worst} matched >{_MAX_NAME_RATE:.0%} of "
                      f"{total_sents} scanned sentences -- degenerate, refusing "
                      f"to trace; fix RECALL_COGITO_SELF_NAMES")
            notify(title="[cogito] refusing to run -- self-name matches too much "
                         "prose", body=detail, priority="high")
            o = Outcome(kind="failed", reason="self_name_too_common",
                        detail=detail, exit_code=1)
            _print(o)
            return o

    if args.dry_run:
        o = Outcome(kind="skipped", reason="dry_run",
                    detail=f"{turns} turn(s), {len(candidates)} new candidate(s); "
                           f"nothing written", exit_code=0)
        _print(o)
        return o

    judge_fn: Callable[[str], str | None] | None = None
    spawned: SpawnedJudge | None = None
    if any(prefilter(s) == "name" for _k, _c, _t, s in candidates):
        spawned = (judge_factory() if judge_factory else SpawnedJudge())
        judge_fn = spawned.ask if spawned.start() else None

    n_new = 0
    try:
        rows_out: list[str] = []
        for key, convo, ts, sent in candidates:
            res = classify(sent, judge_fn)
            if res is None:
                continue
            stance, via = res
            row = {"key": key, "date": target.isoformat(), "convo": convo,
                   "ts": ts, "sentence": sent, "stance": stance, "via": via}
            if via != "regex":                       # name-route: record what matched
                row["matched"] = matched_name(sent)
            rows_out.append(json.dumps(
                row, separators=(",", ":"), sort_keys=True))
        if rows_out:
            cogito_dir().mkdir(parents=True, exist_ok=True)
            with trace_path().open("a") as f:
                f.write("\n".join(rows_out) + "\n")
            n_new = len(rows_out)
    finally:
        if spawned is not None:
            spawned.stop()

    # The 100-gate: one report, ever — its file is the latch.
    reported = ""
    rows = _trace_rows(trace_path())
    if len(rows) >= REPORT_AT and not report_path().exists():
        _write_calibration_report(rows, report_path(), at=REPORT_AT)
        reported = str(report_path())
        notify(title=f"[cogito] calibration report ready ({REPORT_AT} records)",
               body=f"{reported} — joint session when you are.", priority="low")

    detail = (f"{target.isoformat()}: {turns} turn(s), {n_new} new record(s), "
              f"{len(rows)} total" + (f"; REPORT: {reported}" if reported else ""))
    o = Outcome(kind="traced", reason="ok", detail=detail, exit_code=0)
    _print(o)
    return o


def main(argv: list[str] | None = None) -> int:
    return run(argv).exit_code


if __name__ == "__main__":
    sys.exit(main())
