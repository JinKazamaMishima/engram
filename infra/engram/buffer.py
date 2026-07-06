"""LiveBuffer — tier 1 of the continuous-STM stack (Brick 3).

An append-only JSONL of the RAW conversation, one file per conversation id,
one row per message: ``{"convo_id","seq","ts","role","text"}``. The buffer is
the immutable source everything else re-derives from — the working-set block
is rebuilt from its tail every turn, and eviction curates its cooled tail into
the LTM corpus (``recall curate --buffer``). It is never summarized in place.

Every write is immediate (a crash mid-turn must not lose the exchange) and the
whole API is FAIL-OPEN: a lost row, an unwritable directory, a torn line must
never crash a turn — memory is a passenger, never the driver. The reader side
(``recall.transcripts.iter_buffer_exchanges``) tolerates partial lines the
same way.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


class LiveBuffer:
    """Appender for one driver's conversation rows. ``dir_=None`` disables it
    entirely (perception minds, store-less test drivers) — every method then
    no-ops. ``convo_id`` is read through a getter so the driver can rekey the
    conversation (turn-1 launch→sid rename, fork, resume) without rebuilding
    the buffer object."""

    def __init__(self, dir_: Optional[Path],
                 convo_id: Callable[[], str]) -> None:
        self._dir = Path(dir_) if dir_ is not None else None
        self._convo_id = convo_id
        self._seq = 0

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    def path(self) -> Optional[Path]:
        if self._dir is None:
            return None
        return self._dir / f"{self._convo_id()}.jsonl"

    # ---- write side --------------------------------------------------------

    def append(self, role: str, text: str) -> None:
        """One raw message, written immediately. The caller passes the text the
        HUMAN exchanged — never an injected/derived block (the log-raw/
        inject-derived invariant lives in the driver)."""
        if self._dir is None or not text:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._seq += 1
            row = {"convo_id": self._convo_id(), "seq": self._seq,
                   "ts": datetime.now(timezone.utc).isoformat(),
                   "role": role, "text": text}
            with self.path().open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False,
                                   separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001 — fail-open, never break a turn
            pass

    def reseed(self) -> None:
        """Continue ``seq`` above whatever the current file already holds —
        called after any rekey/resume so a conversation's rows stay strictly
        ordered across process restarts and session renames."""
        self._seq = self.last_seq()

    # ---- read side (tolerant) ----------------------------------------------

    def _rows(self) -> list[dict]:
        p = self.path()
        if p is None:
            return []
        try:
            raw = p.read_text(errors="replace")
        except OSError:
            return []
        out: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail write — the next row is intact
            if isinstance(row, dict) and row.get("role") and row.get("ts"):
                out.append(row)
        out.sort(key=lambda r: (r.get("seq", 0), str(r.get("ts", ""))))
        return out

    def tail(self, n: int) -> list[dict]:
        """The newest ``n`` rows, oldest first — the working-set window."""
        if n <= 0:
            return []
        return self._rows()[-n:]

    def tail_after(self, iso_ts: str) -> list[dict]:
        """Rows strictly after ``iso_ts`` — the un-evicted tail the size gate
        measures. A garbled watermark string returns everything (at-least-once
        beats silent loss)."""
        rows = self._rows()
        if not iso_ts:
            return rows
        try:
            mark = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        except ValueError:
            return rows
        out = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if ts > mark:
                out.append(r)
        return out

    def last_seq(self) -> int:
        rows = self._rows()
        return int(rows[-1].get("seq", 0)) if rows else 0

    # ---- rekey --------------------------------------------------------------

    def migrate(self, old_id: str, new_id: str, *, copy: bool = False) -> None:
        """Follow a conversation-identity change on disk. ``copy=True`` (fork)
        duplicates the parent file under the new id — the branched context
        genuinely contains those turns — leaving the parent intact. Otherwise
        the file is MOVED: plain rename when the target is free, merge-append
        (then unlink the source) when the target already exists (rejoining a
        resumed conversation). All best-effort."""
        if self._dir is None or old_id == new_id:
            return
        src = self._dir / f"{old_id}.jsonl"
        dst = self._dir / f"{new_id}.jsonl"
        try:
            if not src.exists():
                return
            if copy:
                shutil.copyfile(src, dst)
            elif dst.exists():
                with dst.open("a") as f:
                    f.write(src.read_text(errors="replace"))
                src.unlink()
            else:
                os.replace(src, dst)
        except Exception:  # noqa: BLE001 — fail-open; worst case rows stay
            pass           # under the old id, swept by the nightly net
