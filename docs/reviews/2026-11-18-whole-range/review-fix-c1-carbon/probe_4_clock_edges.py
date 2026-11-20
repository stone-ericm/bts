"""Item 4 probe: mode clock sample count & placement, summer/winter/DST UTC endpoints, microsecond expiry, UTC grace, New Year seasons, normal-mode rollover."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
import logging
from pathlib import Path
import signal
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
MATCH_ID = "a" * 32


def deny_network(*_args, **_kwargs):
    raise AssertionError("Network access is forbidden")


def at_et(day: date, hr: int = 0, mn: int = 0, sec: int = 0, us: int = 0) -> datetime:
    return datetime.combine(day, time(hr, mn, sec, us), ET)


def test_single_wall_sample_and_calibration_boundary(root: Path) -> None:
    from bts.health import alert, contest_state, runner

    day = date(2027, 6, 30)
    current = [at_et(day, 23, 59, 59)]
    events: list[str] = []

    class RunnerClock(datetime):
        @classmethod
        def now(cls, tz=None):
            events.append(f"runner.now({tz})")
            return current[0].astimezone(tz) if tz else current[0].replace(tzinfo=None)

    class ContestClock(datetime):
        @classmethod
        def now(cls, tz=None):
            events.append(f"contest.now({tz})")
            return current[0].astimezone(tz) if tz else current[0].replace(tzinfo=None)

    # Case A: today=None -> 0 runner.now samples, contest_state.check called with no now/historical_day
    with (
        patch.object(runner, "datetime", RunnerClock),
        patch.object(contest_state, "datetime", ContestClock),
        patch.object(contest_state, "check", wraps=contest_state.check) as spy,
        patch.object(alert, "send_dm", return_value="ack"),
    ):
        runner.run_all_checks(root / "p0", root / "m0", None, today=None, contest_state_expected=False)
    assert [e for e in events if e.startswith("runner.now")] == []
    spy.assert_called_once_with(root / "p0", expected=False)

    # Case B: calibration crosses midnight (23:59:59 -> 00:00:01) BEFORE runner samples mode -> 1 sample after calibration, historical mode!
    events.clear()

    def slow_calibration(*_a, **_k):
        events.append("calibration")
        current[0] = at_et(day + timedelta(days=1), 0, 0, 1)
        return []

    def check_advances_wall(*args, **kwargs):
        events.append(f"contest.check(historical_day={kwargs.get('historical_day')})")
        # Advance wall clock by 2 days mid-check; historical evaluation must stay pinned to D's endpoint!
        current[0] = at_et(day + timedelta(days=3), 12, 0)
        return orig_check(*args, **kwargs)

    orig_check = contest_state.check
    orig_safe = runner._safe_run
    with (
        patch.object(runner, "datetime", RunnerClock),
        patch.object(contest_state, "datetime", ContestClock),
        patch.object(runner.calibration, "check", slow_calibration),
        patch.object(contest_state, "check", side_effect=check_advances_wall) as spy2,
        patch.object(runner, "_safe_run", lambda n, fn: orig_safe(n, fn) if n in {"calibration", "contest_state"} else []),
    ):
        runner.run_all_checks(root / "p1", root / "m1", None, today=day, contest_state_expected=True)
    assert events == [
        "calibration",
        "runner.now(UTC)",
        f"contest.check(historical_day={day})",
    ], events
    spy2.assert_called_once_with(
        root / "p1",
        expected=True,
        now=datetime(2027, 7, 1, 3, 59, 59, 999999, tzinfo=UTC),
        historical_day=day,
    )


def test_endpoints_and_legacy_expiry(root: Path) -> None:
    from bts.health import contest_state, runner

    # Verify summer EDT, winter EST, DST spring-forward (2027-03-14), DST fall-back (2027-11-07), and Dec 31
    expected_utc = [
        (date(2027, 6, 30), datetime(2027, 7, 1, 3, 59, 59, 999999, tzinfo=UTC)),
        (date(2027, 1, 15), datetime(2027, 1, 16, 4, 59, 59, 999999, tzinfo=UTC)),
        (date(2027, 3, 14), datetime(2027, 3, 15, 3, 59, 59, 999999, tzinfo=UTC)),
        (date(2027, 11, 7), datetime(2027, 11, 8, 4, 59, 59, 999999, tzinfo=UTC)),
        (date(2027, 12, 31), datetime(2028, 1, 1, 4, 59, 59, 999999, tzinfo=UTC)),
    ]
    for day, exp_utc in expected_utc:
        computed = datetime.combine(day, time.max, ET).astimezone(UTC)
        assert computed == exp_utc, (day, computed, exp_utc)

        # Test legacy override expiry at: 23:59:59.500000 ET, endpoint - 1us, exact endpoint, endpoint + 1us
        for label, exp_dt, should_win in [
            ("half_sec", at_et(day, 23, 59, 59, 500000), False),
            ("minus_1us", exp_utc - timedelta(microseconds=1), False),
            ("exact", exp_utc, False),
            ("plus_1us", exp_utc + timedelta(microseconds=1), True),
        ]:
            picks = root / f"exp_{day}_{label}" / "picks"
            acct = picks / "account_state"
            acct.mkdir(parents=True)
            (acct / "contest_streak.manual.json").write_text(json.dumps({
                "streak": 5, "source_date": day.isoformat(), "override_expires_at": exp_dt.isoformat(),
            }))
            # Put a corrupt automatic file behind it: if legacy wins (> now), corrupt auto is masked;
            # if legacy expired (<= now), corrupt auto is read and raises loader CRITICAL!
            (acct / "contest_streak.json").write_text("{corrupt_auto")
            wall = at_et(day + timedelta(days=1), 0, 15)

            class Clock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return wall.astimezone(tz)

            orig_safe = runner._safe_run
            with (
                patch.object(runner, "datetime", Clock),
                patch.object(runner, "_safe_run", lambda n, fn: orig_safe(n, fn) if n == "contest_state" else []),
            ):
                alerts = runner.run_all_checks(picks, root / "m", None, today=day, contest_state_expected=True)
            assert len(alerts) == 1
            if should_win:
                assert "The selected legacy manual override is ineligible for a live decision." in alerts[0].message
            else:
                assert "The automatic contest streak file is malformed or unreadable." in alerts[0].message


def test_utc_grace_and_new_year_and_rollover(root: Path) -> None:
    from bts.health import alert, contest_state, runner

    # Future grace in UTC across summer, winter, and Dec 31, even when wall clock is 10 days later
    for day in (date(2027, 6, 30), date(2027, 1, 15), date(2027, 12, 31)):
        for delta_src, expect_future_crit in [(1, False), (2, False), (3, True), (10, True)]:
            for wall_days in (1, 10):
                picks = root / f"grace_{day}_{delta_src}_{wall_days}" / "picks"
                acct = picks / "account_state"
                acct.mkdir(parents=True)
                (acct / "contest_streak.json").write_text(json.dumps({
                    "schema_version": "bts_contest_streak_auto_v2",
                    "season": day.year, "activation_id": MATCH_ID,
                    "active_streak": 6, "best_streak": 6,
                    "source_date": (day + timedelta(days=delta_src)).isoformat(),
                    "recorded_at": at_et(day, 20).isoformat(),
                }))
                (acct / "season_contract.json").write_text(json.dumps({
                    "schema_version": "bts_season_contract_v1", "season": day.year, "activation_id": MATCH_ID,
                    "status": "automatic", "switched_at": at_et(day, 8).isoformat(), "revoked_at": None,
                }))
                (acct / "saver_state.json").write_text(json.dumps({"season": day.year, "state": "active"}))
                wall = at_et(day + timedelta(days=wall_days), 15, 0)

                class Clock(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return wall.astimezone(tz)

                orig_safe = runner._safe_run
                with (
                    patch.object(runner, "datetime", Clock),
                    patch.object(runner, "_safe_run", lambda n, fn: orig_safe(n, fn) if n == "contest_state" else []),
                ):
                    alerts = runner.run_all_checks(picks, root / "m", None, today=day, contest_state_expected=True)
                if expect_future_crit:
                    assert len(alerts) == 1 and "beyond the allowed future grace." in alerts[0].message
                else:
                    assert alerts == [], (day, delta_src, wall_days, alerts)

    # Normal-mode rollover on New Year's Eve (mode sample at 2027-12-31 23:59:59 -> check reads 2028-01-01 00:00:01)
    ny = date(2027, 12, 31)
    picks = root / "ny_rollover" / "picks"
    acct = picks / "account_state"
    acct.mkdir(parents=True)
    (acct / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_manual_v3", "season": 2027,
        "effective_date": "2027-12-31", "active_streak": 4, "best_streak": 4,
        "saver_state": "not_earned", "recorded_at": at_et(ny, 9).isoformat(),
        "expires_at": datetime.combine(ny, time.max, ET).isoformat(), "source": "cli",
    }))

    class SampleSameDay(datetime):
        @classmethod
        def now(cls, tz=None):
            return at_et(ny, 23, 59, 59).astimezone(tz)

    class CrossedNewYear(datetime):
        @classmethod
        def now(cls, tz=None):
            return at_et(date(2028, 1, 1), 0, 0, 1).astimezone(tz)

    orig_safe = runner._safe_run
    with (
        patch.object(runner, "datetime", SampleSameDay),
        patch.object(contest_state, "datetime", CrossedNewYear),
        patch.object(contest_state, "check", wraps=contest_state.check) as spy,
        patch.object(alert, "send_dm", return_value="ack"),
        patch.object(runner, "_safe_run", lambda n, fn: orig_safe(n, fn) if n == "contest_state" else []),
    ):
        alerts = runner.run_all_checks(picks, root / "m", None, today=ny, contest_state_expected=True)
    spy.assert_called_once_with(picks, expected=True)
    # Live classification and live text win on normal-mode rollover, even at New Year (dated 2028-01-01):
    assert any("--date 2028-01-01" in a.message for a in alerts), alerts
    assert not any(a.message.startswith("Prior slate ") for a in alerts), alerts


def main() -> None:
    signal.alarm(45)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=HERE) as tmp:
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            test_single_wall_sample_and_calibration_boundary(Path(tmp))
            test_endpoints_and_legacy_expiry(Path(tmp))
            test_utc_grace_and_new_year_and_rollover(Path(tmp))
            print("[PASS] Item 4: single mode sample, summer/winter/DST endpoints, microsecond expiry, UTC grace, New Year seasons, and New Year normal-mode rollover verified")
    signal.alarm(0)


if __name__ == "__main__":
    main()
