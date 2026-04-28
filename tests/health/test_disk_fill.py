"""Tests for Tier-3 disk fill check."""

from collections import namedtuple
from unittest.mock import patch

from bts.health.disk_fill import check, SOURCE


_Usage = namedtuple("_Usage", ["total", "used", "free"])


class TestDiskFill:
    def test_no_alert_below_info(self, tmp_path):
        # 50% used → no alert
        usage = _Usage(total=100 * 1024 ** 3, used=50 * 1024 ** 3, free=50 * 1024 ** 3)
        with patch("bts.health.disk_fill.shutil.disk_usage", return_value=usage):
            assert check(tmp_path) == []

    def test_info_at_80(self, tmp_path):
        usage = _Usage(total=100 * 1024 ** 3, used=82 * 1024 ** 3, free=18 * 1024 ** 3)
        with patch("bts.health.disk_fill.shutil.disk_usage", return_value=usage):
            alerts = check(tmp_path)
            assert len(alerts) == 1
            assert alerts[0].level == "INFO"
            assert alerts[0].source == SOURCE

    def test_warn_at_90(self, tmp_path):
        usage = _Usage(total=100 * 1024 ** 3, used=92 * 1024 ** 3, free=8 * 1024 ** 3)
        with patch("bts.health.disk_fill.shutil.disk_usage", return_value=usage):
            alerts = check(tmp_path)
            assert alerts[0].level == "WARN"

    def test_critical_at_95(self, tmp_path):
        usage = _Usage(total=100 * 1024 ** 3, used=96 * 1024 ** 3, free=4 * 1024 ** 3)
        with patch("bts.health.disk_fill.shutil.disk_usage", return_value=usage):
            alerts = check(tmp_path)
            assert alerts[0].level == "CRITICAL"

    def test_handles_oserror(self, tmp_path):
        with patch("bts.health.disk_fill.shutil.disk_usage", side_effect=OSError("not found")):
            assert check(tmp_path) == []

    def test_message_includes_gb(self, tmp_path):
        usage = _Usage(total=100 * 1024 ** 3, used=92 * 1024 ** 3, free=8 * 1024 ** 3)
        with patch("bts.health.disk_fill.shutil.disk_usage", return_value=usage):
            alerts = check(tmp_path)
            assert "GB" in alerts[0].message
