"""Check the overnight scorer and actual next-day post-57 continuation."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import World, contest_state, picks
from interactions import auto
from opening import D1, D2


def continuation(root):
    day = "2027-07-15"
    tomorrow = "2027-07-16"
    world = World(root, day, "10:00", health=False)
    with world.switched_on():
        world.slate(day)
        world.slate(tomorrow, ("13:05", "19:05"))
        auto(world, 57, 57)
        picks.save_streak(57, world.picks, saver_available=True, season=2027)
        world.run()
        assert picks.load_streak(world.picks, season=2027) == 58
        world.set_day(tomorrow, "01:10")
        assert world.fetch(58, 58, day).exit_code == 0
        assert contest_state.load_decision_streak_state(world.picks, require_contest_state=True).streak == 58
        world.set_day(tomorrow, "10:00")
        world.run()
        daily = picks.load_pick(tomorrow, world.picks)
        assert daily.chosen_on["streak"] == 58 and daily.result == "hit"
        assert picks.load_streak(world.picks, season=2027) == 59
    return {"case": "post_57_continuation", "events": world.events, "snapshot": world.snapshot()}


def early_scoring(root):
    world = World(root, D1, "05:10:30", health=False)
    with world.switched_on():
        world.slate(D1, ("06:10",))
        world.dated()
        world.run()
        assert picks.load_pick(D1, world.picks).result == "unresolved"
        world.set_day(D2, "01:00")
        result = world.invoke("check-results", "--date", D1, "--picks-dir", world.picks)
        assert result.exit_code == 0, result.output
        assert picks.load_pick(D1, world.picks).result == "hit"
        assert picks.load_streak(world.picks, season=2027) == 1
    return {"case": "early_overnight_grade", "events": world.events, "snapshot": world.snapshot()}


if __name__ == "__main__":
    rows = []
    for probe in (continuation, early_scoring):
        with TemporaryDirectory(prefix="wrblind-chain-") as temp:
            rows.append(probe(Path(temp)))
    Path(sys.argv[1]).resolve().write_text(json.dumps(rows, indent=2))
