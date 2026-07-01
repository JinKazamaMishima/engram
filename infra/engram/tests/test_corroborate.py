#!/usr/bin/env python3
"""Headless tests for the clean-perception dampeners (loop.py, Track B).

Covers the two cheap anti-confabulation defenses:
  (1) prompt discipline (B.4) — ENGAGE_PROMPT tells the eye to admit uncertainty;
  (2) temporal corroboration (B.2) — a scene term is only trusted once it RECURS
      across ≥ CORROBORATE_MIN of the last CORROBORATE_WINDOW eye reads.

The headline case is the user's own example: resting a head on a hand for ONE read
gets misread as "holding a coffee cup". The filter must never let "coffee"/"cup"
reach the corroborated (eviction-eligible) set, while a genuinely persistent object
("laptop", seen across reads) must graduate to it.

    .venv/bin/python infra/engram/tests/test_corroborate.py
"""
import os
import sys
from collections import deque

PERCEIVE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "perceive"))
sys.path.insert(0, PERCEIVE)
sys.path.insert(0, os.path.join(os.path.dirname(PERCEIVE), "eye"))   # sibling senses

from loop import (  # noqa: E402
    CORROBORATE_MIN,
    CORROBORATE_WINDOW,
    ENGAGE_PROMPT,
    PerceiveLoop,
    _content_terms,
)


class _Stub:
    """Just enough state for the unbound _corroborate method — no camera/eye/faceid."""
    def __init__(self):
        self._reading_terms: deque = deque(maxlen=CORROBORATE_WINDOW)


def _run_sequence(readings):
    stub = _Stub()
    corroborate = PerceiveLoop._corroborate.__get__(stub)   # bind method to the stub
    return [corroborate(text) for text in readings]         # [(corroborated, stable), ...]


def test_prompt_discipline():
    p = ENGAGE_PROMPT.lower()
    assert "unsure" in p and "guess" in p, "ENGAGE_PROMPT must tell the eye to admit doubt"
    assert "clearly see" in p, "ENGAGE_PROMPT must ask for only what's clearly visible"
    print("✓ prompt discipline: ENGAGE_PROMPT constrains the eye (unsure/guess/clearly see)")


def test_content_terms_strip_prompt_echo():
    # a reading that is ALL generic/prompt words yields no scene terms
    assert _content_terms("Look at this webcam frame — who is here?") == set()
    # real objects survive
    assert _content_terms("A laptop and a coffee cup") == {"laptop", "coffee", "cup"}
    print("✓ content terms: prompt-echo stripped, scene nouns kept")


def test_coffee_cup_misread_is_rejected():
    readings = [
        "A person is sitting at his desk working on a laptop.",
        "A person is typing on the laptop, looking at the screen.",
        "A person is resting his head on his hand, maybe holding a coffee cup.",  # the misread
        "A person is working on the laptop at his desk.",
    ]
    results = _run_sequence(readings)
    all_corroborated = set().union(*(set(c) for c, _ in results))

    # the one-off confabulation never becomes eviction-eligible
    assert "coffee" not in all_corroborated, "coffee (single misread) must NOT corroborate"
    assert "cup" not in all_corroborated, "cup (single misread) must NOT corroborate"

    # a genuinely persistent object graduates to the trusted set
    corr_last, stable_last = results[-1]
    assert "laptop" in corr_last, f"persistent 'laptop' should corroborate, got {corr_last}"
    assert stable_last is True, "the final, corroborated reading should be marked stable"

    # the misread reading (#3) carries NO trusted specific — coffee/cup excluded
    corr_misread, _ = results[2]
    assert "coffee" not in corr_misread and "cup" not in corr_misread
    print(f"✓ misread rejected: coffee/cup never trusted; 'laptop' graduated {corr_last}")


def test_needs_min_recurrence():
    # the SAME object seen only twice (< CORROBORATE_MIN=3) must not corroborate yet
    readings = ["a red mug on the table", "the red mug is still there"]
    results = _run_sequence(readings)
    assert CORROBORATE_MIN >= 3
    assert all("mug" not in c for c, _ in results), "2 reads < MIN should not corroborate"
    print(f"✓ threshold honored: an object needs ≥{CORROBORATE_MIN} of "
          f"{CORROBORATE_WINDOW} reads before it's trusted")


def main() -> int:
    tests = [
        test_prompt_discipline,
        test_content_terms_strip_prompt_echo,
        test_coffee_cup_misread_is_rejected,
        test_needs_min_recurrence,
    ]
    for t in tests:
        t()
    print(f"\nall {len(tests)} corroboration tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
