"""Tests for Tier-1 blend training cron miss check."""

from datetime import date

from bts.health.blend_training import check, SOURCE


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
