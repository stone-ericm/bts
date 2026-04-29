"""Tests for Tier-3 scheduler memory growth check.

Thresholds tuned 2026-04-28 after first real CRITICAL at 2.9 GB post-prediction.
Pre-tuning thresholds (200/500/1024) fired spurious CRITICAL on normal
post-pick-prediction RSS. Post-tuning: 1024/3072/6144.

Item #5 from 2026-04-28 retro: Tuesday EOD weekly digest INFO alert
collecting trend stats from a daily-appended history file. Tuesday picked
on action-window grounds: weekday > weekend (alerts age before being read);
mid-week > Monday (no week-start alert pile-up); not Friday (issues land
just before weekend gap of low attention).
"""

import json
from datetime import date
from pathlib import Path
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


class TestMemoryGrowthHistory:
    """Item #5: daily history append + Tuesday EOD digest.

    history_path = data/health_state/memory_growth_history.jsonl in prod.
    Each line: {"date": "YYYY-MM-DD", "rss_mb": float}. Append-only; the
    digest reads the last 14 days for the weekly stats summary.
    """

    def test_writes_history_on_first_call(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=800 * 1024):
            check(pid=12345, history_path=history, today=date(2026, 4, 27))  # Monday
        assert history.exists()
        rows = [json.loads(l) for l in history.read_text().strip().splitlines()]
        assert len(rows) == 1
        assert rows[0]["date"] == "2026-04-27"
        assert rows[0]["rss_mb"] == 800.0

    def test_appends_on_subsequent_calls(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        for d, mb in [(date(2026, 4, 27), 800), (date(2026, 4, 28), 850)]:
            with patch("bts.health.memory_growth._read_vmrss_kb", return_value=mb * 1024):
                check(pid=12345, history_path=history, today=d)
        rows = [json.loads(l) for l in history.read_text().strip().splitlines()]
        assert len(rows) == 2
        assert rows[1]["date"] == "2026-04-28"
        assert rows[1]["rss_mb"] == 850.0

    def test_no_digest_on_non_tuesday(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        # Build up some history first
        for d, mb in [
            (date(2026, 4, 21), 800), (date(2026, 4, 22), 820), (date(2026, 4, 23), 810),
            (date(2026, 4, 24), 830), (date(2026, 4, 25), 815), (date(2026, 4, 26), 825),
            (date(2026, 4, 27), 805),  # Monday
        ]:
            with patch("bts.health.memory_growth._read_vmrss_kb", return_value=mb * 1024):
                check(pid=12345, history_path=history, today=d)
        # Monday — should be no digest alert (and no threshold alert at 805 MB < 1024)
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=805 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 4, 27))
        digest_alerts = [a for a in alerts if "weekly memory digest" in a.message.lower()]
        assert digest_alerts == []

    def test_emits_digest_on_tuesday(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        # 14 days of history ending Tue 2026-04-28
        from datetime import timedelta
        start = date(2026, 4, 15)  # Wed
        for i in range(14):
            d = start + timedelta(days=i)
            mb = 800 + i * 5  # gradual creep: 800, 805, ..., 865
            with patch("bts.health.memory_growth._read_vmrss_kb", return_value=mb * 1024):
                check(pid=12345, history_path=history, today=d)
        # Tuesday 2026-04-28 — should emit digest in addition to (no) threshold alert
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=865 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 4, 28))
        digest = [a for a in alerts if "weekly memory digest" in a.message.lower()]
        assert len(digest) == 1
        assert digest[0].level == "INFO"
        assert digest[0].source == SOURCE
        # The message should expose median + latest + trend
        msg = digest[0].message
        assert "median" in msg.lower()
        assert "latest" in msg.lower()
        # At ~5MB/day creep, 7d trend should be ~+35MB or similar — assert positive
        assert "+" in msg or "trend" in msg.lower()

    def test_history_write_failure_doesnt_break_check(self, tmp_path):
        # Read-only history dir: write fails, but the threshold check still runs
        history = tmp_path / "ro" / "memory_growth_history.jsonl"
        # Don't create the parent dir → write will fail at parent lookup
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=1100 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 4, 27))
        # Threshold alert still fires (history is best-effort)
        threshold_alerts = [a for a in alerts if "RSS" in a.message and "weekly" not in a.message.lower()]
        assert len(threshold_alerts) >= 1

    def test_threshold_alert_still_fires_with_history(self, tmp_path):
        # The history feature is additive; threshold alerts must still fire.
        history = tmp_path / "memory_growth_history.jsonl"
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=1100 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 4, 27))
        threshold = [a for a in alerts if "RSS" in a.message and "weekly" not in a.message.lower()]
        assert len(threshold) == 1
        assert threshold[0].level == "INFO"  # 1.1 GB is in INFO range

    def test_no_history_path_means_no_history_writes(self, tmp_path):
        # Backward compat: existing callers that pass no history_path see no
        # behavior change — no file written, no digest emitted.
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=800 * 1024):
            alerts = check(pid=12345)  # no history_path, no today
        assert alerts == []
        # No file in tmp_path either — but we never told the check about it
        assert not (tmp_path / "memory_growth_history.jsonl").exists()
