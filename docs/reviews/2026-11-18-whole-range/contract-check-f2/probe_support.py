import datetime as dt
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import cli
from bts.health import alert, contest_state as health, runner
from tests.test_season_contract_cs4b1 import contract, put
from tests.test_season_contract_cs4b2 import automatic


ET = ZoneInfo("America/New_York")
REAL_DATETIME = dt.datetime
DAY = REAL_DATETIME(2027, 7, 15, 22, tzinfo=ET)
HERE = Path(__file__).resolve().parent


@contextmanager
def clock(now):
    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    with (patch.object(dt, "datetime", Clock), patch.object(cli, "datetime", Clock),
          patch.object(health, "datetime", Clock), patch.object(runner, "datetime", Clock)):
        yield


def command(picks, now, *args):
    with clock(now):
        result = CliRunner().invoke(cli.cli, [*args, "--picks-dir", str(picks)])
    assert result.exit_code == 0, (result.output, result.exception)
    return result.output


def manual(picks, now=DAY, state="used", streak=14):
    command(picks, now, "set-contest-streak", "--date", now.date().isoformat(),
            "--streak", str(streak), "--best-streak", str(max(streak, 15)), "--saver", state)


def auto(picks, now=DAY, streak=14, best=15):
    put(picks, "season_contract.json", contract(season=now.year))
    return put(picks, "contest_streak.json", automatic(
        season=now.year, active_streak=streak, best_streak=best,
        source_date=(now.date() - dt.timedelta(days=1)).isoformat(), recorded_at=now.isoformat(),
    ))


def flag(picks, state, season=DAY.year):
    return put(picks, "saver_state.json", {"season": season, "state": state})


def end_of_day(picks, day, now, transport):
    run = runner._safe_run
    with (clock(now), patch.object(alert, "send_dm", transport),
          patch.object(runner, "_safe_run", lambda name, fn: run(name, fn) if name == "contest_state" else [])):
        alerts = runner.run_all_checks(picks, picks.parent / "models", "offline.test",
                                       today=day, contest_state_expected=True)
    return [(item.level, item.source, item.message) for item in alerts]
