"""Bundled policy replays at the exclusive season boundary."""

import json
from datetime import date, timedelta

import pytest

from bts import scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import tail_policy
from tests.test_wr_a_pick import ROOT, review_world


@pytest.mark.parametrize("day,streak,best,top,count,action", [
    ("2027-09-26", 56, 56, .96, 1, "single"),
    ("2027-09-20", 43, 56, .96, 7, "double"),
    ("2027-09-25", 55, 55, .60, 2, "skip"),
])
def test_2027_season_boundary_and_bundled_policy_replays(
    review_world, day, streak, best, top, count, action,
):
    with review_world.offline(review_world.instant(day, 18, 25)) as (world, _):
        for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
            assert (world.models / name).resolve() == ROOT / "data/models" / name
            assert (world.models / name).is_file()
        world.automatic(streak)
        path = world.account / "contest_streak.json"
        path.write_text(json.dumps(dict(
            json.loads(path.read_text()), best_streak=best, source="mlb_bts_profile",
        )))
        world.write_account("saver_state.json", {"season": 2027, "state": "used"})
        world.predictions = lambda: world.frame((top, top))
        scheduler.run_day(day, world.config)
        decision = load_decision(day, world.picks)
        assert decision["action"] == action
        if action != "skip":
            assert decision["delivery_status"] == "delivered"
            assert len(world.sends("dm")) == 1
        assert strategy._days_remaining(day) == count
        assert strategy._days_remaining("2027-03-24") == 187
        assert strategy._days_remaining("2027-09-27") == 0


def test_deployed_formula_and_health_parity():
    for year in (2026, 2027):
        for offset in range(365):
            day = date(year, 1, 1) + timedelta(days=offset)
            days = strategy._days_remaining(day.isoformat())
            if year == 2026:
                assert days == max(0, (date(2026, 9, 28) - day).days)
            assert tail_policy._days_remaining(day) == days


@pytest.mark.parametrize("defect", ["missing", "corrupt", "directory", "unpaired"])
def test_artifact_warning_uses_supplied_date(tmp_path, defect):
    base = ROOT / "data/models/mdp_policy.npz"
    tail = tmp_path / "tail.npz"
    if defect == "directory":
        tail.mkdir()
    elif defect == "corrupt":
        tail.write_bytes(b"invalid artifact")
    elif defect == "unpaired":
        tail = ROOT / "data/models/mdp_tail_policy.npz"
        base = ROOT / "data/models/mdp_policy_old_aug_sept.npz"
    for day, expected in [
        (date(2027, 8, 29), "WARN"), (date(2027, 8, 30), "CRITICAL"),
        (date(2027, 9, 26), "CRITICAL"), (date(2027, 9, 27), "WARN"),
        (date(2027, 12, 31), "WARN"),
    ]:
        assert {a.level for a in tail_policy.check(base_path=base, tail_path=tail, today=day)} == {
            expected
        }
    good = tail_policy.check(
        base_path=ROOT / "data/models/mdp_policy.npz",
        tail_path=ROOT / "data/models/mdp_tail_policy.npz", today=date(2028, 1, 1),
    )
    assert len(good) == 1 and good[0].level == "WARN" and "not configured" in good[0].message
