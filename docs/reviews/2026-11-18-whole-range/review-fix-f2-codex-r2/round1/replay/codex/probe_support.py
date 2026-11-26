import datetime as dt
import json
import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas  # Initialize before pinning datetime.
from click.testing import CliRunner

import bts
from bts import cli, contest_state, saver_state, season_contract
from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DATETIME = dt.datetime
DAY = REAL_DATETIME(2027, 7, 15, 22, tzinfo=ET)
DECEMBER = REAL_DATETIME(2027, 12, 31, 23, 52, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"
STATUS = "health_dm_delivery_status.json"
SECOND = STATUS + ".dated_used_mismatch.json"
RECIPIENT = "offline.test"
ACTIVATION = "f" * 32

logging.disable(logging.CRITICAL)


@contextmanager
def scratch():
    with TemporaryDirectory(prefix="wr-f2-", dir=Path(__file__).parent) as name:
        yield Path(name)


@contextmanager
def clock(now):
    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    with ExitStack() as stack:
        for module in (dt, cli, contest_state, saver_state, health, runner, alert):
            stack.enter_context(patch.object(module, "datetime", Clock))
        yield


def command(now, *args, picks=None):
    directory = ("--picks-dir", str(picks)) if picks is not None else ()
    with clock(now):
        result = CliRunner().invoke(cli.cli, [*args, *directory], catch_exceptions=False)
    assert result.exit_code == 0, (result.output, result.exception)
    return result.output


def observed(picks, when=DAY, state="used", streak=12):
    command(when, "set-contest-streak", "--date", str(when.date()), "--streak", str(streak),
            "--best-streak", str(max(streak, 16)), "--saver", state, picks=picks)
    return picks / "account_state/contest_streak.manual.json"


def put(directory, name, payload):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def flag(picks, state, year=DAY.year):
    return put(picks / "account_state", "saver_state.json", {"state": state, "season": year})


def automatic(picks, when, streak=9, best=16, coverage=None):
    account = picks / "account_state"
    put(account, "season_contract.json", {
        "schema_version": "bts_season_contract_v1", "season": when.year,
        "status": "automatic", "activation_id": ACTIVATION,
        "switched_at": (when - dt.timedelta(hours=1)).isoformat(),
        "revoked_at": None, "note": None,
    })
    return put(account, "contest_streak.json", {
        "schema_version": season_contract.AUTO_SCHEMA, "season": when.year,
        "activation_id": ACTIVATION, "active_streak": streak, "best_streak": best,
        "source": "mlb_bts_profile", "source_date": str(coverage or when.date()),
        "recorded_at": when.isoformat(),
    })


def eod(picks, day=DAY, *, wall=None, recipient=RECIPIENT, existing=(), transport=None,
        expected=True, overrides=None):
    original = runner._safe_run
    sends = []

    def run_check(name, check):
        if name == "contest_state":
            return original(name, check)
        return list(existing) if name == "calibration" else []

    def capture(recipient, body):
        sends.append((recipient, body))
        return transport(recipient, body) if transport else "confirmed"

    with (clock(wall or day), patch.object(runner, "_safe_run", run_check),
          patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
          patch.object(alert, "send_dm", capture)):
        alerts = runner.run_all_checks(
            picks, picks.parent / "models", recipient, today=day.date(),
            thresholds_overrides=overrides, contest_state_expected=expected,
        )
    return alerts, sends


def receipt(picks, file=STATUS):
    return picks.parent / "health_state" / file


def read_receipt(picks, file=STATUS):
    return json.loads(receipt(picks, file).read_text())


def health_files(picks):
    directory = picks.parent / "health_state"
    return {str(p.relative_to(directory)): p.read_bytes().decode("utf-8")
            for p in sorted(directory.rglob("*")) if p.is_file()}
