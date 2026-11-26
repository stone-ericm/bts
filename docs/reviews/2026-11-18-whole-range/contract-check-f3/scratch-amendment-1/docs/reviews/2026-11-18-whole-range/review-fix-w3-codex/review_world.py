"""Load the archived offline fixture with all writes inside the review folder."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = (ROOT / "docs" / "reviews" / "2026-11-18-whole-range" /
           "slice-3-codex" / "offline.py")
spec = spec_from_file_location("review_fixture", FIXTURE)
fixture = module_from_spec(spec)
spec.loader.exec_module(fixture)
fixture.HERE = HERE
fixture.ROOT = ROOT

DAY = "2027-03-24"
offline = fixture.offline
game = fixture.game


def at(hour, minute=0, second=0, *, day=DAY):
    return fixture.instant(day, hour, minute, second)


def events(world, kind):
    return [(when.isoformat(), body.splitlines()[0])
            for event, when, body in world.events if event == kind]


def health(world):
    return [line for text in world.sends("health-dm") for line in text.splitlines()
            if line.startswith("- [")]


def pending_window(world, monkeypatch, scheduler):
    world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
    monkeypatch.setattr(scheduler, "plan_run_times", lambda *_args, **_kwargs: [
        {"time_et": at(18, 25), "game_pks": [101]},
        {"time_et": at(18, 50), "game_pks": [102]},
    ])
