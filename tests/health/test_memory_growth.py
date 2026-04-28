"""Tests for Tier-3 scheduler memory growth check."""

from unittest.mock import patch

from bts.health.memory_growth import check, SOURCE


class TestMemoryGrowth:
    def test_no_alert_below_threshold(self, tmp_path):
        # 90 MB (typical baseline) → no alert
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=90 * 1024):
            assert check(pid=12345) == []

    def test_info_at_200mb(self, tmp_path):
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=210 * 1024):
            alerts = check(pid=12345)
            assert len(alerts) == 1
            assert alerts[0].level == "INFO"
            assert alerts[0].source == SOURCE

    def test_warn_at_500mb(self, tmp_path):
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=600 * 1024):
            alerts = check(pid=12345)
            assert alerts[0].level == "WARN"

    def test_critical_at_1gb(self, tmp_path):
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=1100 * 1024):
            alerts = check(pid=12345)
            assert alerts[0].level == "CRITICAL"

    def test_no_alert_when_proc_unavailable(self):
        # On Mac, /proc doesn't exist → return None → no alert
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=None):
            assert check(pid=12345) == []
