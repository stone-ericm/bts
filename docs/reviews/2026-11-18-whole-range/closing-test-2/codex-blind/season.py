"""Compare full automatic days, policy boundaries, scoring, and entry identity."""
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import ACCOUNT, IS_BASE, RECIPIENT, World, contest_state, picks, wall


def outcome(world, day):
    return world.invoke("day-outcome", "--date", day, "--config", world.config_path,
                        "--picks-dir", world.picks, "--expected-username", ACCOUNT,
                        "--dm-recipient", RECIPIENT)


def previous(day):
    return str(date.fromisoformat(day) - timedelta(days=1))


def season_case(root, day, streak, state, scenario):
    from bts import daily_decision, strategy
    year = date.fromisoformat(day).year
    world = World(root, day, "01:00", health=False, required=not IS_BASE)
    with world.switched_on():
        probabilities = (.65, .64, .63) if scenario == "skip" else (.98, .97, .90)
        world.slate(day, ("13:05", "16:05", "19:05"), probabilities)
        if not IS_BASE:
            assert world.invoke("season-contract", "enable", "--season", year,
                                "--picks-dir", world.picks).exit_code == 0
        fetched = world.fetch(streak, streak, previous(day))
        assert fetched.exit_code == 0, fetched.output
        assert world.invoke("saver-state", "--init", state, "--force", "--season", year,
                            "--picks-dir", world.picks).exit_code == 0
        seed = {} if IS_BASE else {"season": year}
        picks.save_streak(streak, world.picks, saver_available=state == "active", **seed)
        if scenario == "postponed":
            world.slots[day, 700] = "void"
        if scenario == "void":
            def postpone():
                world.slots[day, 700] = "void"
            world.at("12:35", postpone)
        current = contest_state.load_decision_streak_state(world.picks, require_contest_state=not IS_BASE)
        world.run()
        daily = picks.load_pick(day, world.picks)
        record = daily_decision.load_decision(day, world.picks)
        # The next accepted fixture account reading represents one additional hit.
        if not IS_BASE and streak == 57 and daily is not None:
            next_day = str(date.fromisoformat(day) + timedelta(days=1))
            world.set_day(next_day, "01:10")
            assert world.fetch(58, 58, day).exit_code == 0
            nxt = contest_state.load_decision_streak_state(world.picks, require_contest_state=True)
            world.event("next_account_read", streak=nxt.streak, status=nxt.best_status)
            outcome(world, day)
            world.dated(58, 58, state)
            next_dated = contest_state.load_decision_streak_state(world.picks, require_contest_state=True)
            world.event("next_dated_read", streak=next_dated.streak, status=next_dated.best_status)
            checker = __import__("bts.health.contest_state", fromlist=["check"])
            world.event("health_high_state", alerts=[a.message for a in checker.check(world.picks, expected=True)])
            world.set_day(day, "21:00")
            world.event("saved_pick_check", refusal=picks.saved_pick_refusal(daily, current))
        return {
            "day": day, "streak": streak, "saver": state, "scenario": scenario,
            "calendar_days": strategy._days_remaining(day), "state_read": repr(current),
            "decision": record, "events": world.events, "snapshot": world.snapshot(),
        }


def doubleheader(root, entered_game):
    world = World(root, "2027-07-15", "09:55", health=False)
    day = world.day
    with world.switched_on():
        world.slate(day, ("13:05", "20:05"), (.98,), doubleheader=True)
        if not IS_BASE:
            world.dated(20, 20, "active")
        else:
            assert world.fetch(20, 20, previous(day)).exit_code == 0
        world.crosswalk = {1: 100}
        from bts.entry_receipt import units_snapshot_dir
        units = units_snapshot_dir(world.picks)
        units.mkdir(parents=True)
        (units / f"{day}T130000Z.json.gz").write_bytes(gzip.compress(json.dumps(
            {"units": [{"id": 77, "feedId": 700, "roundId": 1},
                       {"id": 88, "feedId": 701, "roundId": 1}]}).encode()))
        world.pending = [{"playerId": 1, "roundId": 1, "unitId": 77 if entered_game == 700 else 88}]

        def check():
            args = [] if IS_BASE else ["--cron"]
            world.invoke("check-pick-entered", *args, "--picks-dir", world.picks,
                         "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)

        world.at("12:05", check)
        world.run()
        world.set_day("2027-07-16", "08:30")
        if not IS_BASE:
            outcome(world, day)
        return {"entered": entered_game, "events": world.events, "snapshot": world.snapshot()}


def main():
    target = Path(sys.argv[1]).resolve()
    rows = []
    for day in ("2027-07-15", "2027-09-20", "2027-09-26", "2027-09-27", "2028-03-24"):
        for streak in (55, 56, 57, 58, 110):
            with TemporaryDirectory(prefix="wrblind-season-") as temp:
                rows.append(season_case(Path(temp), day, streak, "active", "main"))
    for scenario, streak, day in (("skip", 56, "2027-07-15"), ("postponed", 12, "2027-09-20"),
                                  ("void", 58, "2027-09-26")):
        with TemporaryDirectory(prefix="wrblind-season-") as temp:
            rows.append(season_case(Path(temp), day, streak, "used", scenario))
    entry = []
    for game in (700, 701):
        with TemporaryDirectory(prefix="wrblind-doubleheader-") as temp:
            entry.append(doubleheader(Path(temp), game))
    target.write_text(json.dumps({"season": rows, "doubleheader": entry}, indent=2))


if __name__ == "__main__":
    main()
