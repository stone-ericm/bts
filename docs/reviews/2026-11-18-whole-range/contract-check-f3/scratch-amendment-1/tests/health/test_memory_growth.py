"""Tests for Tier-3 scheduler memory growth check.

Thresholds recalibrated 2026-05-23 after prod history showed normal
post-pick-prediction RSS around 2.8-3.6 GB and cold sleeping RSS around
140 MB. Daily threshold alerts now use a higher absolute floor plus recent
post-prediction baseline deltas.

Item #5 from 2026-04-28 retro: Tuesday EOD weekly digest INFO alert
collecting trend stats from a daily-appended history file. Tuesday picked
on action-window grounds: weekday > weekend (alerts age before being read);
mid-week > Monday (no week-start alert pile-up); not Friday (issues land
just before weekend gap of low attention).
"""

import json
from datetime import date, timedelta
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

    def test_no_alert_at_1_1gb(self, tmp_path):
        # 1.1 GB used to emit INFO, but is now below the daily alert floor.
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=1100 * 1024):
            assert check(pid=12345) == []

    def test_no_alert_at_normal_3_5gb_post_prediction_rss(self, tmp_path):
        # 3.5 GB appears in normal prod post-prediction history.
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=3500 * 1024):
            assert check(pid=12345) == []

    def test_info_at_4_5gb_absolute_floor(self, tmp_path):
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=4700 * 1024):
            alerts = check(pid=12345)
            assert len(alerts) == 1
            assert alerts[0].level == "INFO"
            assert alerts[0].source == SOURCE

    def test_warn_at_5gb_absolute_floor(self, tmp_path):
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=5200 * 1024):
            alerts = check(pid=12345)
            assert len(alerts) == 1
            assert alerts[0].level == "WARN"
            assert alerts[0].source == SOURCE

    def test_no_critical_at_2_9gb_post_prediction(self, tmp_path):
        # 2.9 GB is normal post-prediction RSS, no action needed.
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=int(2.9 * 1024 * 1024)):
            assert check(pid=12345) == []

    def test_critical_at_6gb(self, tmp_path):
        # 6.5 GB → CRITICAL (~40% of bts-mlb's 16 GB)
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=int(6.5 * 1024 * 1024)):
            alerts = check(pid=12345)
            assert alerts[0].level == "CRITICAL"

    def test_warn_on_growth_over_recent_post_prediction_baseline(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        history.write_text("\n".join([
            json.dumps({"date": "2026-05-18", "rss_mb": 2750.0}),
            json.dumps({"date": "2026-05-19", "rss_mb": 2800.0}),
            json.dumps({"date": "2026-05-20", "rss_mb": 2850.0}),
        ]) + "\n")
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=3900 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 5, 21))
        threshold = [a for a in alerts if "RSS" in a.message]
        assert len(threshold) == 1
        assert threshold[0].level == "WARN"
        assert "delta" in threshold[0].message

    def test_critical_on_large_growth_over_recent_post_prediction_baseline(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        history.write_text("\n".join([
            json.dumps({"date": "2026-05-18", "rss_mb": 2750.0}),
            json.dumps({"date": "2026-05-19", "rss_mb": 2800.0}),
            json.dumps({"date": "2026-05-20", "rss_mb": 2850.0}),
        ]) + "\n")
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=5900 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 5, 21))
        threshold = [a for a in alerts if "RSS" in a.message]
        assert threshold[0].level == "CRITICAL"

    def test_ignores_cold_history_for_growth_baseline(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        history.write_text("\n".join([
            json.dumps({"date": "2026-05-18", "rss_mb": 138.0}),
            json.dumps({"date": "2026-05-19", "rss_mb": 140.0}),
            json.dumps({"date": "2026-05-20", "rss_mb": 139.0}),
        ]) + "\n")
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=3200 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 5, 21))
        threshold = [a for a in alerts if "RSS" in a.message]
        assert threshold == []

    def test_oom_level_history_does_not_mask_recent_baseline(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        history.write_text("\n".join([
            json.dumps({"date": "2026-05-15", "rss_mb": 2800.0}),
            json.dumps({"date": "2026-05-16", "rss_mb": 2850.0}),
            json.dumps({"date": "2026-05-17", "rss_mb": 7000.0}),
            json.dumps({"date": "2026-05-18", "rss_mb": 7100.0}),
            json.dumps({"date": "2026-05-19", "rss_mb": 7200.0}),
        ]) + "\n")
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=3900 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 5, 20))
        threshold = [a for a in alerts if "RSS" in a.message]
        assert len(threshold) == 1
        assert threshold[0].level == "WARN"

    def test_prod_post_prediction_history_band_does_not_warn(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        prod_loaded_mb = [
            3206.2, 3164.7, 3197.8, 3177.6, 3550.6, 3586.2,
            2895.5, 3023.7, 3350.0, 2775.7,
        ]
        start = date(2026, 5, 10)
        history.write_text("\n".join(
            json.dumps({"date": (start + timedelta(days=i)).isoformat(), "rss_mb": mb})
            for i, mb in enumerate(prod_loaded_mb)
        ) + "\n")
        for mb in prod_loaded_mb:
            with patch("bts.health.memory_growth._read_vmrss_kb", return_value=int(mb * 1024)):
                alerts = check(pid=12345, history_path=history, today=date(2026, 5, 21))
            threshold = [a for a in alerts if "RSS" in a.message]
            assert threshold == []

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

    def test_digest_uses_latest_row_per_date(self, tmp_path):
        history = tmp_path / "memory_growth_history.jsonl"
        history.write_text("\n".join([
            json.dumps({"date": "2026-04-15", "rss_mb": 800.0}),
            json.dumps({"date": "2026-04-15", "rss_mb": 1800.0}),
            json.dumps({"date": "2026-04-16", "rss_mb": 820.0}),
        ]) + "\n")
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=2500 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 4, 21))
        digest = [a for a in alerts if "weekly memory digest" in a.message.lower()]
        assert len(digest) == 1
        assert "3 data points" in digest[0].message
        assert "1800.0" in digest[0].message

    def test_new_nested_history_path_doesnt_break_check(self, tmp_path):
        # New nested history paths are created best-effort, but threshold
        # alerts should not depend on the history append side path.
        history = tmp_path / "ro" / "memory_growth_history.jsonl"
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=5200 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 4, 27))
        threshold_alerts = [a for a in alerts if "RSS" in a.message and "weekly" not in a.message.lower()]
        assert len(threshold_alerts) >= 1

    def test_threshold_alert_still_fires_with_history(self, tmp_path):
        # The history feature is additive; threshold alerts must still fire.
        history = tmp_path / "memory_growth_history.jsonl"
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=5200 * 1024):
            alerts = check(pid=12345, history_path=history, today=date(2026, 4, 27))
        threshold = [a for a in alerts if "RSS" in a.message and "weekly" not in a.message.lower()]
        assert len(threshold) == 1
        assert threshold[0].level == "WARN"

    def test_no_history_path_means_no_history_writes(self, tmp_path):
        # Backward compat: existing callers that pass no history_path see no
        # behavior change — no file written, no digest emitted.
        with patch("bts.health.memory_growth._read_vmrss_kb", return_value=800 * 1024):
            alerts = check(pid=12345)  # no history_path, no today
        assert alerts == []
        # No file in tmp_path either — but we never told the check about it
        assert not (tmp_path / "memory_growth_history.jsonl").exists()
