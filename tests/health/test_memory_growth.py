"""Tests for Tier-3 scheduler memory growth check.

Thresholds tuned 2026-04-28 after first real CRITICAL at 2.9 GB post-prediction.
Pre-tuning thresholds (200/500/1024) fired spurious CRITICAL on normal
post-pick-prediction RSS. Post-tuning: 1024/3072/6144.
"""

from unittest.mock import patch

from bts.health.memory_growth import check, SOURCE


class TestMemoryGrowth:
    def test_no_alert_at_sleeping_baseline(self, tmp_path):
        # 90 MB (sleeping-state baseline) → no alert
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=90 * 1024):
            assert check(pid=12345) == []

    def test_no_alert_at_post_prediction_baseline(self, tmp_path):
        # 800 MB (after a pick-prediction cycle, normal) → no alert
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=800 * 1024):
            assert check(pid=12345) == []

    def test_info_at_1gb(self, tmp_path):
        # 1.1 GB → INFO (notable growth, worth observing)
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=1100 * 1024):
            alerts = check(pid=12345)
            assert len(alerts) == 1
            assert alerts[0].level == "INFO"
            assert alerts[0].source == SOURCE

    def test_warn_at_3gb(self, tmp_path):
        # 3.5 GB → WARN (beyond expected post-prediction)
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=3500 * 1024):
            alerts = check(pid=12345)
            assert alerts[0].level == "WARN"

    def test_no_critical_at_2_9gb_post_prediction(self, tmp_path):
        # 2.9 GB (the value that triggered the spurious CRITICAL misfire on
        # 2026-04-28) is INFO under the tuned thresholds — that's normal
        # post-prediction RSS, no action needed. Captures the regression.
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=int(2.9 * 1024 * 1024)):
            alerts = check(pid=12345)
            assert alerts[0].level == "INFO"

    def test_critical_at_6gb(self, tmp_path):
        # 6.5 GB → CRITICAL (~40% of bts-mlb's 16 GB)
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=int(6.5 * 1024 * 1024)):
            alerts = check(pid=12345)
            assert alerts[0].level == "CRITICAL"

    def test_no_alert_when_proc_unavailable(self):
        # On Mac, /proc doesn't exist → return None → no alert
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=None):
            assert check(pid=12345) == []
