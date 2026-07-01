"""Tests for recall.dynamics — the DSR/FSRS math (pure functions, no I/O)."""
from __future__ import annotations

from recall import dynamics as D


def test_retrievability_is_0_9_at_one_stability():
    # FSRS-6 curve is pinned so R == 0.9 exactly at age == S.
    assert abs(D.retrievability(10.0, 10.0) - 0.9) < 1e-9
    assert abs(D.retrievability(100.0, 100.0) - 0.9) < 1e-9


def test_retrievability_monotonic_decreasing_in_age():
    s = 20.0
    rs = [D.retrievability(age, s) for age in (0, 5, 20, 100, 1000)]
    assert rs == sorted(rs, reverse=True)
    assert rs[0] == 1.0                       # fresh: age 0 -> R 1
    assert all(0.0 < r <= 1.0 for r in rs)


def test_retrievability_higher_stability_decays_slower():
    age = 30.0
    assert D.retrievability(age, 5.0) < D.retrievability(age, 50.0)


def test_reinforce_increases_stability_and_is_monotonic():
    s0 = 5.0
    s1 = D.reinforce(s0, D.retrievability(5.0, s0))
    assert s1 > s0                             # use strengthens
    # bracket >= 1: even at R≈1 (just-used) stability never drops
    assert D.reinforce(100.0, 1.0) >= 100.0


def test_reinforce_testing_effect_bigger_jump_when_nearly_forgotten():
    s = 10.0
    fresh = D.reinforce(s, r=0.95)            # just used -> small gain
    faded = D.reinforce(s, r=0.30)            # nearly forgotten -> big gain
    assert (faded - s) > (fresh - s) > 0


def test_reinforce_diminishing_returns_in_stability():
    # same retrievability, larger S -> smaller *relative* growth
    rel_small = D.reinforce(3.0, 0.6) / 3.0
    rel_big = D.reinforce(100.0, 0.6) / 100.0
    assert rel_small > rel_big


def test_reinforce_citation_gain_beats_surfacing():
    s, r = 8.0, 0.7
    surfaced = D.reinforce(s, r, gain=1.0)
    cited = D.reinforce(s, r, gain=D.CITE_GAIN)
    assert cited > surfaced > s


def test_initial_stability_lerps_with_surprise():
    assert abs(D.initial_stability(0.0) - D.S0_LOW) < 1e-9
    assert abs(D.initial_stability(1.0) - D.S0_HIGH) < 1e-9
    mid = D.initial_stability(0.5)
    assert D.S0_LOW < mid < D.S0_HIGH
    # clamps out-of-range
    assert D.initial_stability(-3.0) == D.S0_LOW
    assert D.initial_stability(9.0) == D.S0_HIGH


def test_permanence_threshold_and_floor():
    assert not D.is_permanent(D.S_PERM - 1)
    assert D.is_permanent(D.S_PERM + 1)
    # a graduated note never falls below the floor no matter how old
    assert D.effective_retrievability(10_000.0, D.S_PERM + 50) >= D.PERM_FLOOR
    # a non-permanent note is NOT floored
    assert D.effective_retrievability(10_000.0, 5.0) < D.PERM_FLOOR


def test_surprise_from_similarity():
    assert D.surprise_from_similarity(1.0) == 0.0      # identical to corpus -> no surprise
    assert D.surprise_from_similarity(0.0) == 1.0      # unlike everything -> max surprise
    assert abs(D.surprise_from_similarity(0.7) - 0.3) < 1e-9
    assert D.surprise_from_similarity(1.5) == 0.0      # clamps
