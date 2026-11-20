"""Scoped, offline fixtures shared by the C2 review probes."""

from contextlib import contextmanager
from datetime import datetime
from importlib import import_module
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from pytest import MonkeyPatch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OPENING = ROOT / "docs/reviews/2026-11-18-whole-range/slice-1-codex"
FAILURE_PREFIX = "Unknown-outcome alert failed: "
ET = ZoneInfo("America/New_York")


class ProbeTimeout(BaseException):
    pass


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise ProbeTimeout(f"offline probe exceeded {seconds} seconds")

    handler = signal.signal(signal.SIGALRM, expired)
    prior = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        if prior[0]:
            signal.setitimer(signal.ITIMER_REAL, *prior)


@contextmanager
def fixtures(label):
    with TemporaryDirectory(prefix=f"{label}-", dir=HERE) as temp, MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT))
        patch.syspath_prepend(str(OPENING))
        yield Path(temp), patch, import_module("opening_rehearsal")


@contextmanager
def world_at(root, monkeypatch, opening, moment):
    from bts import scheduler, strategy
    from tests.test_saved_pick_paths_cs5a import prepare_run_day, world

    root.mkdir(parents=True, exist_ok=True)
    clock = opening.Clock(moment)
    with opening.sandbox(clock, root), monkeypatch.context() as scoped:
        scoped.setattr(strategy, "_mdp_cache", None)
        config, directory, now, statuses = world.__wrapped__(root, scoped)
        now[0] = moment
        prepare_run_day(scoped)
        scoped.setattr(scheduler, "notify_ready", opening.no_op)
        scoped.setattr(scheduler, "notify_watchdog", opening.no_op)
        yield config, directory, now, statuses, scoped


def receipt(directory, day):
    return directory.parent / "health_state" / "delivery_outcome_unknown" / f"{day}.json"


def shared(directory):
    return directory.parent / "health_state" / "health_dm_delivery_status.json"


def raw(path):
    return path.read_bytes() if path.is_file() else None


def status_body(day):
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


def day_pick(day, pitch="19:00"):
    from tests.test_pick_chosen_on_cs5a import daily_pick

    daily = daily_pick()
    daily.date = day
    daily.pick.batter_name = "Example Batter"
    daily.run_time = datetime.fromisoformat(f"{day}T09:30:00").replace(tzinfo=ET).isoformat()
    daily.pick.game_time = datetime.fromisoformat(f"{day}T{pitch}:00").replace(tzinfo=ET).isoformat()
    return daily


def prepare_schedules(*daily_picks):
    from bts import scheduler

    games = {
        daily.date: [{
            "gamePk": daily.pick.game_pk, "gameDate": daily.pick.game_time,
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"home": {"team": {"name": "Home"}}, "away": {"team": {"name": "Away"}}},
        }]
        for daily in daily_picks
    }
    scheduler.fetch_schedule.side_effect = lambda day: games.get(day, [])


def attempt_direct(config, directory, daily):
    from bts import picks, scheduler

    state = scheduler.SchedulerState(daily.date, daily.run_time, [], [], [], False, None, None, None)
    picks.save_pick(daily, directory)
    return scheduler._deliver_and_lock_pick(daily, config, directory, state, daily.date, "review")
