"""Tests for Tier-1 blend training cron miss check."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from bts.health.blend_training import check, SOURCE

ET = ZoneInfo("America/New_York")


class TestBlendTraining:
    def test_no_alert_when_tomorrow_pkl_exists(self, tmp_path):
        # Today is 2026-04-27; tomorrow's pkl present → no alert
        (tmp_path / "blend_2026-04-28.pkl").write_text("")
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert alerts == []

    def test_critical_when_tomorrow_pkl_missing(self, tmp_path):
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert len(alerts) == 1
        a = alerts[0]
        assert a.level == "CRITICAL"
        assert a.source == SOURCE
        assert "blend_2026-04-28.pkl" in a.message

    def test_other_pkls_dont_satisfy(self, tmp_path):
        # An older pkl exists but not tomorrow's → still CRITICAL
        (tmp_path / "blend_2026-04-25.pkl").write_text("")
        (tmp_path / "blend_2026-04-26.pkl").write_text("")
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"


class TestBlendTrainingTimeGuard:
    """Time-of-day guard: don't alert before the 3 AM cron has had a chance to fire.

    Production pattern: cron writes blend_<N+1>.pkl at 3:06 AM ET on day N. Alerts
    fired between midnight and ~4 AM ET will fire on legitimately-missing files
    that the cron is about to create. Cutoff: 4 AM ET (1h grace after cron).
    """

    def test_no_alert_before_4am_et_even_if_pkl_missing(self, tmp_path):
        # 1:30 AM ET on 2026-04-30 — cron hasn't fired yet for blend_2026-05-01.pkl
        now = datetime(2026, 4, 30, 1, 30, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert alerts == []

    def test_no_alert_at_359am_et_even_if_pkl_missing(self, tmp_path):
        # 3:59 AM ET — cron may still be running (typical 3:06 AM start)
        now = datetime(2026, 4, 30, 3, 59, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert alerts == []

    def test_critical_at_4am_et_when_pkl_missing(self, tmp_path):
        # 4:00 AM ET — cron should have completed; missing file is real failure
        now = datetime(2026, 4, 30, 4, 0, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"

    def test_critical_at_3pm_et_when_pkl_missing(self, tmp_path):
        # Mid-afternoon — clear miss
        now = datetime(2026, 4, 30, 15, 0, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert len(alerts) == 1

    def test_no_alert_before_4am_when_pkl_present(self, tmp_path):
        # Even pre-4am, presence of pkl means definite no-alert
        (tmp_path / "blend_2026-05-01.pkl").write_text("")
        now = datetime(2026, 4, 30, 2, 0, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert alerts == []

    def test_now_defaults_to_actual_now(self, tmp_path):
        # When now is None, falls through to default datetime.now(ET).
        # This test just confirms the function accepts now=None without crashing.
        # Actual behavior depends on real wall clock; we don't assert specific alerts.
        result = check(tmp_path, today=date(2026, 4, 30), now=None)
        assert isinstance(result, list)
