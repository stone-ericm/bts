"""Drop-in regression test for B1: stat()-failing saver_state.json symlinks on an in-effect dated used observation."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from bts.health import alert, contest_state as health, runner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 30, tzinfo=ET)


def _write_dated_used(picks: Path) -> None:
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    exp = REAL_DT.combine(DAY.date(), dt.time.max, ET)
    (acct / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": "2027-07-15",
        "season": 2027,
        "expires_at": exp.isoformat(),
        "active_streak": 14,
        "best_streak": 15,
        "saver_state": "used",
        "recorded_at": "2027-07-15T20:00:00Z",
        "source": "cli_dated",
    }), encoding="utf-8")


@pytest.mark.parametrize("symlink_kind", ["unsearchable_dir", "enametoolong"])
@pytest.mark.parametrize("historical", [False, True])
def test_dated_used_flag_symlink_stat_oserror_preserves_alerts_and_sends_no_crash_dm(
    tmp_path, monkeypatch, symlink_kind, historical,
):
    picks = tmp_path / "data" / "picks"
    _write_dated_used(picks)
    resolved = [dt.date(2027, 7, 16), dt.date(2027, 7, 17)]
    monkeypatch.setattr(health, "latest_resolved_pick_date", lambda _: max(resolved))
    monkeypatch.setattr(health, "_scan_resolved_pick_dates", lambda *a, **kw: resolved)
    monkeypatch.setattr(health, "resolved_pick_settlement_gap", lambda *a: 2)

    baseline = health.check(
        picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None,
    )
    assert len(baseline) == 1 and baseline[0].source == "contest_state" and baseline[0].level == "WARN"

    locked_dir = None
    target = picks / "account_state" / "saver_state.json"
    if symlink_kind == "unsearchable_dir":
        locked_dir = picks / "account_state" / "locked_sub"
        locked_dir.mkdir()
        (locked_dir / "s.json").write_text('{"season": 2027, "state": "active"}', encoding="utf-8")
        locked_dir.chmod(0o000)
        os.symlink(locked_dir / "s.json", target)
    else:
        os.symlink("x" * 300, target)

    wall = DAY + dt.timedelta(hours=3) if historical else DAY

    class WallClock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz) if tz else wall.replace(tzinfo=None)

    try:
        dms = []
        orig_safe_run = runner._safe_run
        with patch.object(runner, "datetime", WallClock), \
             patch.object(alert, "datetime", WallClock), \
             patch.object(health, "datetime", WallClock), \
             patch.object(alert, "send_dm", side_effect=lambda r, b: dms.append((r, b)) or "ok"), \
             patch.object(runner, "_safe_run", side_effect=lambda n, fn: orig_safe_run(n, fn) if n == "contest_state" else []):
            eod_alerts = runner.run_all_checks(
                picks_dir=picks,
                models_dir=tmp_path / "data" / "models",
                dm_recipient="ops.bsky.social",
                today=DAY.date(),
                contest_state_expected=True,
            )
        actual = health.check(
            picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None,
        )
        assert actual == baseline
        assert eod_alerts == baseline
        assert dms == []
    finally:
        if locked_dir is not None:
            locked_dir.chmod(0o700)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rA"]))
