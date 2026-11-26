"""Exercise the real command's ET clock across the gap and both folds."""
import datetime as calendar
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts import entry_receipt
from tests.test_wr_f1_wrong_game import MATCH, RIGHT, World

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
RealDatetime = calendar.datetime

def trip(name, start, pitch):
    now = RealDatetime.fromisoformat(start).astimezone(ET)
    game = RealDatetime.fromisoformat(pitch).astimezone(ET)
    world = World(HERE / "clock-worlds" / name, now.date().isoformat())
    world.commit(pitch=pitch)
    world.capture([RIGHT], f"{world.day}T040000Z")
    invoke = world.runner.invoke

    class PinnedDatetime(RealDatetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

    def production_clock(command, args):
        options = list(args)
        idx = options.index("--now-et")
        del options[idx:idx + 2]
        with patch("datetime.datetime", PinnedDatetime):
            return invoke(command, options)

    world.runner.invoke = production_clock
    output = world.check(now.strftime("%H:%M"), pending=MATCH)
    receipt = entry_receipt.discover(world.picks, world.day)[0]
    duration = (game.timestamp() - now.timestamp()) / 60
    print(name, "start", now.isoformat(), "pitch", game.isoformat(), "actual minutes", duration,
          "reported minutes", receipt["minutes_to_pitch"], "outcome", receipt["outcome"],
          "fetches", output["fetches"])
    return receipt["outcome"]

assert trip("spring-gap", "2027-03-14T01:45:00-05:00", "2027-03-14T03:05:00-04:00") == "outside_window"
assert trip("fall-too-early", "2027-11-07T01:00:00-04:00", "2027-11-07T01:55:00-05:00") == "observed"
assert trip("fall-first-fold", "2027-11-07T01:45:00-04:00", "2027-11-07T01:55:00-05:00") == "observed"
assert trip("fall-second-fold", "2027-11-07T01:45:00-05:00", "2027-11-07T01:55:00-05:00") == "observed"
