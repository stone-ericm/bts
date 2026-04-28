"""Tests for Tier-3 streak.json validation check."""

import json

from bts.health.streak_validation import check, SOURCE


class TestStreakValidation:
    def test_no_alert_valid(self, tmp_path):
        (tmp_path / "streak.json").write_text(json.dumps({
            "streak": 7, "saver_available": True,
        }))
        assert check(tmp_path) == []

    def test_no_alert_minimal_valid(self, tmp_path):
        # Without saver_available is OK
        (tmp_path / "streak.json").write_text(json.dumps({"streak": 0}))
        assert check(tmp_path) == []

    def test_critical_when_missing(self, tmp_path):
        alerts = check(tmp_path)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert alerts[0].source == SOURCE

    def test_critical_when_malformed(self, tmp_path):
        (tmp_path / "streak.json").write_text("{not json")
        alerts = check(tmp_path)
        assert alerts[0].level == "CRITICAL"

    def test_critical_when_streak_not_int(self, tmp_path):
        (tmp_path / "streak.json").write_text(json.dumps({"streak": "seven"}))
        alerts = check(tmp_path)
        assert alerts[0].level == "CRITICAL"
        assert "streak field" in alerts[0].message

    def test_critical_when_streak_negative(self, tmp_path):
        (tmp_path / "streak.json").write_text(json.dumps({"streak": -3}))
        alerts = check(tmp_path)
        assert alerts[0].level == "CRITICAL"
        assert "negative" in alerts[0].message

    def test_critical_when_saver_not_bool(self, tmp_path):
        (tmp_path / "streak.json").write_text(json.dumps({
            "streak": 5, "saver_available": "yes",
        }))
        alerts = check(tmp_path)
        assert alerts[0].level == "CRITICAL"
        assert "saver_available" in alerts[0].message
