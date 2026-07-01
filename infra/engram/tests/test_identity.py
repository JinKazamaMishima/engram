#!/usr/bin/env python3
"""Headless tests for the v1 per-prompt identity gate (app.py::_identity_note).

Every TYPED turn is prefixed with a marker telling the model WHO the camera sees at
the keyboard, so Engram knows whether it's really Ada. Pure function → no camera/TUI:
we feed synthetic PerceptionBridge.snapshot() dicts and assert the marker.

Doctrine: informational, NOT a hard lock — a false-negative (Ada looked away) must
never lock her out; the marker just flips Engram from 'assume Ada' to 'verify first'.

    .venv/bin/python infra/engram/tests/test_identity.py
"""
import os
import sys

ENGRAM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ENGRAM)

from app import EngramApp  # noqa: E402

note = EngramApp._identity_note   # pure staticmethod (target, snap) -> str


def test_target_present_confirms():
    snap = {"ok": True, "present": ["Ada"], "faces": [("Ada", 0.71)], "state": "engaged"}
    m = note("Ada", snap)
    assert m.startswith("[identity]") and m.endswith("\n\n")
    assert "operator is Ada" in m and "0.71" in m
    assert "NOT" not in m
    print("✓ target present → confirms Ada, shows cosine")


def test_stranger_warns_and_withholds():
    snap = {"ok": True, "present": [], "faces": [("unknown", 0.22)], "state": "passive"}
    m = note("Ada", snap)
    assert "NOT Ada" in m and "unknown 0.22" in m
    assert "withhold" in m.lower(), "must tell the model to withhold Ada's private context"
    print("✓ unknown face → warns it's NOT Ada, withhold private context")


def test_known_other_is_still_not_target():
    # a different ENROLLED person (present, but not the target) must still warn
    snap = {"ok": True, "present": ["Bob"], "faces": [("Bob", 0.58)], "state": "passive"}
    m = note("Ada", snap)
    assert "NOT Ada" in m and "Bob 0.58" in m
    print("✓ known non-target (Bob) → still flagged NOT Ada")


def test_nobody_is_unverified_not_locked_out():
    snap = {"ok": True, "present": [], "faces": [], "state": "idle"}
    m = note("Ada", snap)
    assert "unverified" in m.lower()
    assert "off-camera" in m.lower(), "must acknowledge the benign looked-away case"
    assert "NOT Ada" not in m, "absence is not an accusation"
    print("✓ nobody in frame → unverified + benign (off-camera), not locked out")


def test_perception_off_is_silent():
    assert note("Ada", None) == "", "no bridge → no marker (normal launch untouched)"
    assert note("Ada", {"ok": False, "error": "no camera"}) == "", "bridge error → silent"
    print("✓ perception off / errored → empty marker (Telegram + normal launch untouched)")


def test_target_present_without_cosine():
    # debounced presence can hold even when the target isn't in the current faces frame
    snap = {"ok": True, "present": ["Ada"], "faces": [], "state": "engaged"}
    m = note("Ada", snap)
    assert "operator is Ada" in m and "face match" not in m  # no stale cosine invented
    print("✓ debounced presence w/o current face → confirms without a fake cosine")


def main() -> int:
    tests = [
        test_target_present_confirms,
        test_stranger_warns_and_withholds,
        test_known_other_is_still_not_target,
        test_nobody_is_unverified_not_locked_out,
        test_perception_off_is_silent,
        test_target_present_without_cosine,
    ]
    for t in tests:
        t()
    print(f"\nall {len(tests)} identity-gate tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
