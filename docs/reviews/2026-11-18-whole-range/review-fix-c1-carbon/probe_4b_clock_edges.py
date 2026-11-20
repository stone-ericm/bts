"""Item 4 probe (v2): mode clock sample count & placement, summer/winter/DST UTC endpoints, microsecond expiry, UTC grace, New Year seasons, normal-mode rollover."""

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

from probe_4_clock_edges import (
    ET,
    HERE,
    UTC,
    at_et,
    deny_network,
    test_single_wall_sample_and_calibration_boundary,
    test_utc_grace_and_new_year_and_rollover,
)


def test_endpoints_and_legacy_expiry_v2(root: Path) -> None:
    from bts.health import runner

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
            (acct / "saver_state.json").write_text(json.dumps({"season": day.year, "state": "active"}))
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
            assert len(alerts) == 1, alerts
            if should_win:
                assert "The selected legacy manual override is ineligible for a live decision." in alerts[0].message
            else:
                assert "The automatic contest streak file is malformed or unreadable." in alerts[0].message


def main() -> None:
    signal.alarm(45)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=HERE) as tmp:
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            test_single_wall_sample_and_calibration_boundary(Path(tmp))
            test_endpoints_and_legacy_expiry_v2(Path(tmp))
            test_utc_grace_and_new_year_and_rollover(Path(tmp))
            print("[PASS] Item 4: single mode sample, summer/winter/DST endpoints, microsecond expiry, UTC grace, New Year seasons, and New Year normal-mode rollover verified")
    signal.alarm(0)


if __name__ == "__main__":
    main()
