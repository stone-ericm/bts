"""Known-answer tests for the 2027 season walk (closing test part d).

Each case drives one date through the real decision path (orchestrator.run_and_pick
with the shipped policy artifacts) via the harness in scripts/audit/season_walk/.
"""
import datetime as dt
import gzip
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from scripts.audit.season_walk import walk


def test_oracle_counts_today_through_the_finale():
    assert walk.LAST_GAME_DAY == date(2027, 9, 26)
    assert walk.expected_days_left(date(2027, 3, 24)) == 187
    assert walk.expected_days_left(date(2027, 9, 26)) == 1
    assert walk.expected_days_left(date(2027, 9, 27)) == 0
    assert walk.expected_days_left(date(2027, 10, 5)) == 0
    assert walk.expected_days_left(date(2027, 3, 1)) == 210


def test_stored_boundary_is_the_day_after_the_finale():
    assert walk.table_boundary_problems() == []


def test_the_datetime_patch_is_undone():
    walk.walk_one(date(2027, 7, 15), walk.State("dated", 12, 12, "used", 0.96))
    assert dt.datetime.now().year == dt.datetime.now(dt.timezone.utc).year
    assert dt.datetime.__name__ == "datetime"


@pytest.mark.parametrize("saver", ["used", "active"])
def test_last_game_day_at_56_plays(saver):
    rec = walk.walk_one(date(2027, 9, 26), walk.State("dated", 56, 56, saver, 0.96))
    assert rec["error"] is None
    assert rec["days_left"] == 1 and rec["days_effective"] == 1
    assert rec["action"] in {"single", "double"}
    assert rec["batter"] == "Batter 0"
    assert rec["chosen_on_kind"] == "dated"


def test_last_game_day_at_56_plays_on_the_automatic_path():
    rec = walk.walk_one(date(2027, 9, 26), walk.State("automatic", 56, 56, "used", 0.96))
    assert rec["error"] is None
    assert rec["action"] in {"single", "double"}
    assert rec["chosen_on_kind"] == "automatic"


def test_late_boundary_day_with_a_high_trusted_best_plays():
    # 2027-09-20 at 43 with best 56: 7 days left, 43 + 2*7 = 57 is reachable.
    rec = walk.walk_one(date(2027, 9, 20), walk.State("dated", 43, 56, "used", 0.96))
    assert rec["error"] is None
    assert rec["days_left"] == 7
    assert rec["objective"] == "reach57"
    assert rec["action"] in {"single", "double"}


def test_after_the_finale_is_a_named_skip():
    rec = walk.walk_one(date(2027, 10, 1), walk.State("dated", 20, 20, "used", 0.96))
    assert rec["error"] is None
    assert rec["days_left"] == 0
    assert rec["action"] == "skip" and rec["named"] is True
    assert rec["source"] == "mdp"


def test_opening_night_dated_zero_plays():
    rec = walk.walk_one(date(2027, 3, 24), walk.State("dated", 0, 0, "not_earned", 0.96))
    assert rec["error"] is None
    assert rec["days_left"] == 187 and rec["days_effective"] == 180
    assert rec["action"] in {"single", "double"}
    assert rec["saver_available"] is True


def test_automatic_midseason_reads_the_flag():
    rec = walk.walk_one(date(2027, 7, 15), walk.State("automatic", 12, 12, "active", 0.96))
    assert rec["error"] is None
    assert rec["chosen_on_kind"] == "automatic"
    assert rec["saver_available"] is True
    assert rec["action"] in {"single", "double"}


def test_check_flags_errors_unnamed_empties_and_wrong_day_counts():
    good = dict(day="2027-07-15", state="dated/12/12/used/0.96", error=None, action="single",
                named=False, days_left=74, days_effective=74, batter="Batter 0",
                no_pick_reason=None, in_season=True, objective="reach57", source="mdp")
    assert walk.check([good]) == []
    problems = walk.check([
        good | {"error": "ContestStateError: x"},
        good | {"action": None, "batter": None, "no_pick_reason": "no_eligible"},
        good | {"action": "skip", "batter": None, "named": False},
        good | {"days_left": 73},
    ])
    assert len(problems) == 4
    assert any("error" in p for p in problems)
    assert any("unnamed" in p for p in problems)
    assert any("days left" in p for p in problems)


def test_check_requires_the_last_day_to_play_at_56():
    rec = dict(day="2027-09-26", state="dated/56/56/used/0.96", error=None, action="skip",
               named=True, days_left=1, days_effective=1, batter=None, no_pick_reason=None,
               in_season=True, objective="reach57", source="mdp")
    assert any("last game day" in p for p in walk.check([rec]))


def test_year_boundary_and_fallback_warn_as_documented():
    result = walk.boundary_checks()
    assert result["problems"] == []
    assert result["season_end_2028"] == "2028-09-28"
    assert result["warn_2028"] and "not configured" in result["warn_2028"][0]
    assert result["warn_2027"] == []
    assert result["days_2026_12_31"] == 0
    assert result["days_2027_01_01"] == 269
    assert result["switch_2027_at_2027_12_31_2359"] == "accepted"
    assert result["switch_2027_at_2028_01_01_0001"].startswith("refused")


def test_the_walk_catches_the_old_last_game_day_boundary():
    """Failure direction: store the finale itself (the 2026-11-13 table) and the walk must report it."""
    from unittest.mock import patch
    import bts.strategy as strategy
    with patch.dict(strategy.SEASON_END_DATES, {2027: "2027-09-26"}):
        records = [walk.walk_one(date(2027, 9, 26), walk.State("dated", 56, 56, "used", 0.96)),
                   walk.walk_one(date(2027, 9, 20), walk.State("dated", 43, 56, "used", 0.96))]
        boundary = walk.table_boundary_problems()
    problems = walk.check(records)
    assert any("last game day at streak 56 skipped" in p for p in problems)
    assert any("2027-09-20" in p and "days left 6, expected 7" in p for p in problems)
    assert boundary and "2027-09-26" in boundary[0]


def test_skip_records_carry_the_real_dm_text():
    rec = walk.walk_one(date(2027, 10, 1), walk.State("dated", 20, 20, "used", 0.96))
    assert rec["skip_dm"].startswith("BTS 2027-10-01: No pick")


def test_check_flags_a_skip_message_that_calls_a_clearing_best_below_its_bar():
    base = dict(day="2027-07-15", state="dated/56/56/used/0.96", error=None, action="skip",
                named=True, days_left=74, days_effective=74, batter=None, no_pick_reason=None,
                in_season=True, objective="reach57", source="mdp", best_p=0.96)
    wrong = base | {"skip_dm": "BTS 2027-07-15: No pick yet — model's best is Batter 0 (T0) 96%, "
                               "below the ~80% bar (streak holds at 56). Will pick if a confirmed lineup clears it.",
                    "skip_bar": None}
    assert any("skip message" in p for p in walk.check([wrong]))
    right = base | {"best_p": 0.70, "skip_bar": None, "skip_dm": wrong["skip_dm"].replace("96%", "70%")}
    assert not any("skip message" in p for p in walk.check([right]))


def test_real_k2_dm_uses_the_below_bar_comparison():
    rec = walk.walk_one(date(2027, 7, 15), walk.State("dated", 12, 12, "used", 0.72))
    assert rec["action"] == "skip" and rec["best_p"] < rec["skip_bar"]
    assert "below the" in rec["skip_dm"] and "using unrounded values." in rec["skip_dm"]
    assert walk.check([rec]) == []


def test_saved_e_walk_agrees_with_baseline():
    results = Path(walk.__file__).with_name("results")
    with gzip.open(results / "walk-bc143131.json.gz", "rt") as file:
        baseline = json.load(file)
    with gzip.open(results / "walk-e.json.gz", "rt") as file:
        updated = json.load(file)
    assert baseline["runs"] == updated["runs"] == 4599
    assert baseline["by_day"] == updated["by_day"]
    assert baseline["boundary"] == updated["boundary"]
    assert updated["problems"] == [] and walk.check(updated["records"]) == []
    flags = {
        entry.split(": skip message", 1)[0] for entry in baseline["problems"]
        if ": skip message" in entry
    }
    assert len(flags) == 841
    counts = Counter()
    fields = ("day", "state", "action", "pick_bar", "expected_days_left", "days_left", "days_effective")
    for before, after in zip(baseline["records"], updated["records"], strict=True):
        assert [before[key] for key in fields] == [after[key] for key in fields]
        if f"{before['day']} {before['state']}" in flags:
            case = "K4" if before["skip_bar"] is None else "K3"
            phrase = {"K4": "No probability comparison is available for this check.",
                      "K3": "at or above the"}[case]
            assert phrase in after["skip_dm"] and "below the" not in after["skip_dm"]
            counts[case] += 1
    assert counts == {"K3": 229, "K4": 612}


@pytest.mark.parametrize("action", ["skip", "single"])
def test_longest_walk_flags_wrong_action_even_with_legacy_objective(action):
    from bts.simulate import tail_policy

    boundary = getattr(tail_policy, "LONGEST_STREAK_FROM", 57)
    record = dict(day="2027-09-26", state=f"dated/{boundary}/{boundary}/used/0.96",
                  action=action, named=action == "skip", in_season=True,
                  batter="Batter 0" if action == "single" else None,
                  days_left=1, days_effective=1, objective="reach57", source="mdp")
    assert any(f"action {action!r}, expected 'double'" in problem
               for problem in walk.check([record]))


def test_longest_walk_records_the_actual_choice():
    from bts.simulate import tail_policy

    boundary = getattr(tail_policy, "LONGEST_STREAK_FROM", 57)
    record = walk.walk_one(date(2027, 9, 26), walk.State("dated", boundary, boundary, "used", .96))
    assert (record["objective"], record["action"], record["double"]) == (
        "longest_streak", "double", "Batter 1")
    assert record["chosen_on_valid"] is True
    assert walk.check([record]) == []


@pytest.mark.parametrize("field,value,message", [
    ("days_effective", None, "effective days"),
    ("source", None, "decision source"),
    ("source", "heuristic", "decision source"),
    ("artifacts", None, "policy artifacts"),
])
def test_r4_longest_walk_flags_incomplete_policy_fields(field, value, message):
    from bts.simulate.tail_policy import LONGEST_STREAK_FROM

    boundary = LONGEST_STREAK_FROM
    record = walk.walk_one(date(2027, 9, 26), walk.State("dated", boundary, boundary, "used", .96))
    assert walk.check([record]) == []
    assert any(message in problem for problem in walk.check([record | {field: value}]))
