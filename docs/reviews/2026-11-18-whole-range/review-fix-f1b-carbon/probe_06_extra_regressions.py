"""Probe 06: Regression tests for the two cli.py paths surviving the existing F1 test suite."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from tests.test_wr_f1_wrong_game import DAY, FIRST, LEG, MATCH, PITCH, RIGHT, WRONG, World


def test_all_clear_after_wrong_game_then_present_unverified_bridge():
    """Pins cli.py:2113 (`or "wrong_game" in prior_escalations`) when prior status is present_unverified."""
    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td))
        w.commit(pitch=PITCH)
        w.capture([WRONG])
        r1 = w.check("18:30", MATCH)
        assert r1["dm"] == [FIRST.format(legs=LEG)]
        assert r1["marker"]["escalations"] == ["wrong_game"]

        # Bridge through present_unverified so was_alerted is False and "initial" is absent
        r2 = w.check("18:40", MATCH, crosswalk={})
        assert r2["marker"]["status"] == "present_unverified"
        assert r2["marker"]["escalations"] == ["wrong_game"]

        # Full game proof must still send the all-clear DM!
        w.units.joinpath(f"{DAY}T210000Z.json.gz").unlink()
        w.capture([RIGHT])
        r3 = w.check("18:50", MATCH)
        assert r3["exit"] == 0
        assert r3["marker"]["status"] == "confirmed"
        assert r3["dm"] == [f"✅ BTS pick entry confirmed for {DAY} (match)."]


def test_dm_failed_after_wrong_game_and_present_unverified_keeps_consumed_wrong_game_tier():
    """Pins cli.py:2263 (`if prior_wrong_game or reason == "wrong_game":`) when removal initial DM fails."""
    with tempfile.TemporaryDirectory() as td:
        w = World(Path(td))
        w.commit(pitch=PITCH)
        w.capture([WRONG])
        assert w.check("18:30", MATCH)["marker"]["escalations"] == ["wrong_game"]

        # Bridge through present_unverified so was_alerted becomes False
        assert w.check("18:40", MATCH, crosswalk={})["marker"]["status"] == "present_unverified"

        # Operator removes entry and initial DM fails -> writes dm_failed, MUST preserve escalations=["wrong_game"]!
        r3 = w.check("18:45", [], dm="failure")
        assert r3["exit"] == 1
        assert r3["marker"]["status"] == "dm_failed"
        assert r3["marker"]["reason"] == "no_pick"
        assert r3["marker"]["game_unresolved"] is True
        assert r3["marker"]["escalations"] == ["wrong_game"]

        # If entry returns in wrong game at 18:50, wrong_game tier must NOT re-fire!
        r4 = w.check("18:50", MATCH)
        assert r4["exit"] == 1
        assert r4["dm"] == []


def run_all() -> dict:
    test_all_clear_after_wrong_game_then_present_unverified_bridge()
    test_dm_failed_after_wrong_game_and_present_unverified_keeps_consumed_wrong_game_tier()
    summary = {
        "all_clear_after_present_unverified_bridge": True,
        "dm_failed_preserves_wrong_game_escalation_after_present_unverified": True,
    }
    (OUT_DIR / "log_06_extra_regressions.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
