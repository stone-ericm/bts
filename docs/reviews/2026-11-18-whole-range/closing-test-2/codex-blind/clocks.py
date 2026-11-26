"""Exercise season reads and overlapping score/reconcile entry points."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import World, contest_state, picks, wall
from opening import D1, D2
from season import outcome


def result_boundary(root, target, morning):
    from bts import daily_decision, strategy
    w = World(root, morning, "01:00", health=False)
    seed_day = "2027-09-26"
    with w.switched_on():
        w.slate(target)
        daily = picks.DailyPick(
            date=target, run_time=wall(target, "17:00").isoformat(),
            pick=picks.Pick(
                batter_name="Fixture Batter", batter_id=100, team="T0", lineup_position=1,
                pitcher_name="Fixture Pitcher", pitcher_id=900, p_game_hit=.97, game_pk=700,
                game_time=wall(target, "19:05").isoformat(), projected_lineup=False, flags=[],
            ),
            double_down=None, result=None, runner_up=None,
            notification_sent=True, notification_id="fixture-notification",
            delivered_at=wall(target, "17:00").isoformat(),
        )
        picks.save_pick(daily, w.picks)
        if target == "2027-12-31":
            picks.save_streak(9, w.picks, saver_available=True, season=2027)
        else:
            picks.save_streak(16, w.picks, saver_available=False, season=2026)
        daily_decision.write_decision(target, w.picks, action="single", source="mdp",
                                      delivery_status="delivered", scoreable=True)
        w.invoke("check-results", "--date", target, "--picks-dir", w.picks)
        w.event("after_grading", value=json.loads((w.picks / "streak.json").read_text()))
        w.set_day(morning, "02:00")
        w.invoke("reconcile", "--picks-dir", w.picks)
        w.event("after_reconcile", value=json.loads((w.picks / "streak.json").read_text()))
        w.set_day(morning, "07:40")
        w.invoke("reconcile", "--picks-dir", w.picks)
        w.event("after_second_reconcile", value=json.loads((w.picks / "streak.json").read_text()))
        w.set_day(morning, "08:30")
        outcome(w, target)
        for day in (D1, D2, seed_day, "2027-09-27", "2027-12-31", "2028-01-01",
                    "2028-03-24", "2028-09-27", "2028-09-28"):
            w.event("day_count", slate=day, count=strategy._days_remaining(day))
        return {"target": target, "events": w.events, "snapshot": w.snapshot()}


def midnight(root):
    from bts.health import contest_state as health
    w = World(root, D1, "23:59", health=False)
    with w.switched_on():
        w.dated(58, 60, "used")
        for day, clock in ((D1, "23:59"), (D2, "00:00"), (D2, "00:05"), ("2028-01-01", "00:05")):
            w.set_day(day, clock)
            try:
                live = contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
                live = repr(live)
            except contest_state.ContestStateError as error:
                live = str(error)
            historical = None
            if day == D2:
                historical = health.check(w.picks, expected=True, now=wall(D1, "23:59"),
                                          historical_day=wall(D1, "23:59").date())
            w.event("clock_read", live=live, health=[a.message for a in health.check(w.picks, expected=True)],
                    historical=[a.message for a in historical] if historical is not None else None)
        w.invoke("season-contract", "enable", "--season", 2027, "--picks-dir", w.picks)
        w.set_day(D1, "23:59")
        assert w.invoke("season-contract", "enable", "--season", 2027,
                        "--picks-dir", w.picks).exit_code == 0
        assert w.fetch(58, 60, D1).exit_code == 0
        w.set_day("2028-03-24", "10:00")
        try:
            contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
        except contest_state.ContestStateError as error:
            assert "other_season" in str(error) or "2027" in str(error)
            w.event("existing_2027_switch_on_2028_slate", refusal=str(error))
        else:
            raise AssertionError("last season's activation unexpectedly became live")
        return {"target": "midnight", "events": w.events, "snapshot": w.snapshot()}


def main():
    output = Path(sys.argv[1]).resolve()
    rows = []
    for target, morning in ((D1, D2), ("2027-12-31", "2028-01-01")):
        with TemporaryDirectory(prefix="wrblind-scoring-") as temp:
            rows.append(result_boundary(Path(temp), target, morning))
    with TemporaryDirectory(prefix="wrblind-midnight-") as temp:
        rows.append(midnight(Path(temp)))
    output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
