"""DSR/FSRS memory dynamics — how a note's stability grows when it is used and how
its retrievability decays with time. Pure functions, no torch, no I/O: this is the
math the consolidate fold and (Phase II) the search ranking both call.

We adopt the FSRS-6 forgetting curve and stability-increase law (the
spaced-repetition state of the art), which fold recency + salience + the spacing
effect + the testing effect into a single state variable ``S`` (stability, in
days). See docs/dynamic-memory.md for the derivation and references. All knobs are
env-overridable, matching the rest of recall.
"""
from __future__ import annotations

import math
import os

# FSRS-6 forgetting-curve constants (fixed; R = 0.9 at age == S).
FACTOR = 19.0 / 81.0
DECAY = -0.5


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# Stability-increase law. Tuned slower than flashcard FSRS — a knowledge corpus is
# rehearsed far less deliberately than a study deck, so a single recall should not
# vault a note to permanence.
STAB_A = _env_float("RECALL_STAB_A", 3.0)     # overall reinforcement gain
STAB_B = _env_float("RECALL_STAB_B", 0.18)    # stabilization-decay exponent (diminishing returns)
STAB_C = _env_float("RECALL_STAB_C", 1.0)     # testing-effect strength (low R -> bigger jump)
CITE_GAIN = _env_float("RECALL_CITE_GAIN", 2.5)   # g for a cited note (a surfaced note is g=1)

S_MIN = _env_float("RECALL_S_MIN", 0.5)       # floor for stability (days)
S_DEFAULT = _env_float("RECALL_S_DEFAULT", 2.0)   # stability for a note first used with none set
S0_LOW = _env_float("RECALL_S0_LOW", 0.5)     # initial stability for an unsurprising note
S0_HIGH = _env_float("RECALL_S0_HIGH", 15.0)  # initial stability for a maximally surprising note
S_PERM = _env_float("RECALL_S_PERM", 365.0)   # graduation line to effective permanence
PERM_FLOOR = _env_float("RECALL_PERM_FLOOR", 0.6)   # R floor for permanent notes (never fully fade)


def retrievability(age_days: float, stability: float) -> float:
    """Probability a note is still 'recallable' at ``age_days`` since last use,
    given stability ``S`` (days). FSRS-6 power curve — heavy-tailed, R=0.9 at
    age==S, R≈0.5 near age≈3.4·S. Guards: age<0 -> 0, S<=0 -> S_MIN."""
    s = max(float(stability), S_MIN)
    t = max(0.0, float(age_days))
    return (1.0 + FACTOR * t / s) ** DECAY


def reinforce(stability: float, r: float, *, gain: float = 1.0) -> float:
    """New stability after ONE successful retrieval at current retrievability
    ``r``. The bracket is ≥ 1, so stability is monotonically non-decreasing under
    use. Lower ``r`` (a nearly-forgotten note) -> bigger jump (the testing
    effect); bigger ``S`` -> smaller relative jump (diminishing returns); ``gain``
    weights a citation above a mere surfacing."""
    s = max(float(stability), S_MIN)
    r = min(max(float(r), 0.0), 1.0)
    g = max(0.0, float(gain))
    sinc = 1.0 + STAB_A * s ** (-STAB_B) * (math.exp(STAB_C * (1.0 - r)) - 1.0) * g
    return s * max(1.0, sinc)


def initial_stability(surprise: float) -> float:
    """Initial stability ``S₀`` for a new note from its surprise σ∈[0,1] (the
    flashbulb route): a model-violating insight is born decay-resistant. Linear
    interpolation across the FSRS first-review spread."""
    sig = min(max(float(surprise), 0.0), 1.0)
    return S0_LOW + sig * (S0_HIGH - S0_LOW)


def is_permanent(stability: float) -> bool:
    """A note that has graduated to effective permanence (slow rehearsal route or
    fast flashbulb route) — exempt from decay-driven demotion and pruning."""
    return float(stability) >= S_PERM


def effective_retrievability(age_days: float, stability: float) -> float:
    """Retrievability with the permanence floor applied: a graduated note never
    falls out of recall no matter how long since it was last used."""
    r = retrievability(age_days, stability)
    return max(r, PERM_FLOOR) if is_permanent(stability) else r


def surprise_from_similarity(max_similarity: float) -> float:
    """σ = 1 − (max cosine-similarity to the existing corpus), clamped to [0,1].
    A genuinely novel insight is unlike anything we already know."""
    return min(1.0, max(0.0, 1.0 - float(max_similarity)))
