"""Processed-slate contest health clock regressions."""

import json
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from bts.health import contest_state, runner
from bts.contest_state import DATED_MANUAL_SCHEMA


ET = ZoneInfo("America/New_York")


def test_oom_path_pages_only_in_same_day_mode(tmp_path, monkeypatch):
    day = date(2027, 3, 24)
    wall = datetime.combine(day, time(23, 50), ET)
    picks = tmp_path / "oom" / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True)
    manual = account / "contest_streak.manual.json"
    manual.write_text('{"streak": 0}')

    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            return wall.astimezone(tz)

    original_safe_run = runner._safe_run
    monkeypatch.setattr(runner, "datetime", Clock, raising=False)
    monkeypatch.setattr(contest_state, "datetime", Clock)
    monkeypatch.setattr(
        runner, "_safe_run",
        lambda name, fn: original_safe_run(name, fn) if name == "contest_state" else [],
    )

    def run():
        return runner.run_all_checks(picks, tmp_path / "models", None, today=day)

    same_day = run()
    assert same_day[0].message == (
        f"legacy/expired manual override present at {manual}; "
        "archive it (auto contest_streak.json is the live source)"
    )
    assert same_day[1].source == "analytics_job_oom"
    assert same_day[1].message == f"OOM evidence detected: contest_state: {same_day[0].message}"
    wall += timedelta(hours=1)
    assert [(alert.level, alert.source) for alert in run()] == [("WARN", "contest_state")]


@pytest.mark.parametrize("day", [date(2027, 6, 30), date(2027, 12, 31)])
def test_mode_sample_and_last_et_microsecond(tmp_path, day):
    following = day + timedelta(days=1)
    wall = datetime.combine(following, time(0, 15), ET)
    samples = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            samples.append(tz)
            return wall.astimezone(tz)

    with (
        patch.object(runner, "datetime", Clock, create=True),
        patch.object(contest_state, "check", return_value=[]) as check,
    ):
        runner.run_all_checks(
            tmp_path, tmp_path, None, today=day, contest_state_expected=True,
        )

    check.assert_called_once_with(
        tmp_path,
        expected=True,
        now=datetime.combine(day, time.max, ET).astimezone(timezone.utc),
        historical_day=day,
    )
    assert samples == [timezone.utc]


@pytest.mark.parametrize("hour,minute", [(0, 15), (2, 0)])
def test_eod_uses_processed_observation_and_safe_recovery_copy(tmp_path, hour, minute):
    day = date(2027, 3, 24)
    following = day + timedelta(days=1)
    wall = datetime.combine(following, time(hour, minute), ET)
    picks = tmp_path / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True)
    (account / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": DATED_MANUAL_SCHEMA,
        "season": day.year,
        "effective_date": day.isoformat(),
        "active_streak": 0,
        "best_streak": 0,
        "saver_state": "not_earned",
        "recorded_at": datetime.combine(day, time(9), ET).isoformat(),
        "expires_at": datetime.combine(day, time.max, ET).isoformat(),
        "source": "offline",
    }))

    class Clock(datetime):
        @classmethod
        def now(cls, tz):
            return wall.astimezone(tz)

    with (
        patch.object(runner, "datetime", Clock, create=True),
        patch.object(contest_state, "datetime", Clock),
    ):
        alerts = runner.run_all_checks(
            picks, tmp_path / "models", None, today=day, contest_state_expected=True,
        )
    assert [alert for alert in alerts if alert.source in {"contest_state", "saver_state"}] == []
