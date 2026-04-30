"""Tests for Tier-2 pitcher data sparsity check."""

import json
from datetime import date

from bts.health.pitcher_sparsity import check, SOURCE, FLAG_NEEDLE


def _write_pick(picks_dir, date_iso, *, pick_flags=None, dd_flags=None, has_pick=True):
    data = {"date": date_iso}
    if has_pick:
        data["pick"] = {"batter_name": "X", "p_game_hit": 0.7, "flags": pick_flags or []}
    if dd_flags is not None:
        data["double_down"] = {"batter_name": "Y", "flags": dd_flags}
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(data))


class TestPitcherSparsity:
    def test_no_alert_when_under_warn_count(self, tmp_path):
        # 4 of 14 days flagged — below warn_count=5
        for i in range(14):
            d = f"2026-04-{i+1:02d}"
            flags = [FLAG_NEEDLE] if i < 4 else []
            _write_pick(tmp_path, d, pick_flags=flags)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert alerts == []

    def test_warn_at_warn_count(self, tmp_path):
        for i in range(14):
            d = f"2026-04-{i+1:02d}"
            flags = [FLAG_NEEDLE] if i < 5 else []
            _write_pick(tmp_path, d, pick_flags=flags)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert len(alerts) == 1
        assert alerts[0].level == "WARN"
        assert alerts[0].source == SOURCE
        assert "5/14" in alerts[0].message

    def test_critical_at_critical_count(self, tmp_path):
        for i in range(14):
            d = f"2026-04-{i+1:02d}"
            flags = [FLAG_NEEDLE] if i < 8 else []
            _write_pick(tmp_path, d, pick_flags=flags)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert "8/14" in alerts[0].message

    def test_double_down_flag_counts_too(self, tmp_path):
        # 5 days flagged — but only via double-down flag, not primary pick flag
        for i in range(14):
            d = f"2026-04-{i+1:02d}"
            dd_flags = [FLAG_NEEDLE] if i < 5 else []
            _write_pick(tmp_path, d, pick_flags=[], dd_flags=dd_flags)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert len(alerts) == 1
        assert alerts[0].level == "WARN"

    def test_no_alert_when_insufficient_history(self, tmp_path):
        # Only 3 days of pick data — below min_examined=5
        for i in range(3):
            d = f"2026-04-{i+1:02d}"
            _write_pick(tmp_path, d, pick_flags=[FLAG_NEEDLE])
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert alerts == []

    def test_skip_days_dont_count(self, tmp_path):
        # 8 days have no pick (skip days) — should not be examined
        for i in range(14):
            d = f"2026-04-{i+1:02d}"
            if i < 8:
                _write_pick(tmp_path, d, has_pick=False)
            else:
                _write_pick(tmp_path, d, pick_flags=[FLAG_NEEDLE])
        # 6 examined, all flagged → 6/6 = 100% but < critical_count=8
        # Though the threshold uses absolute counts, not percentages, so this would
        # actually fall below the WARN threshold (5 < 6 yes — wait 6 >= 5, WARN).
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert len(alerts) == 1
        assert alerts[0].level == "WARN"
        assert "6/6" in alerts[0].message

    def test_handles_corrupt_pick_file(self, tmp_path):
        # 4 valid + 1 corrupt — should not crash, only count the 4
        for i in range(4):
            d = f"2026-04-{i+1:02d}"
            _write_pick(tmp_path, d, pick_flags=[FLAG_NEEDLE])
        (tmp_path / "2026-04-05.json").write_text("not json{{{")
        alerts = check(tmp_path, today=date(2026, 4, 14))
        # 4 examined < min_examined=5 → no alert
        assert alerts == []

    def test_no_alert_when_picks_dir_missing(self, tmp_path):
        # picks_dir doesn't exist
        missing = tmp_path / "nope"
        alerts = check(missing, today=date(2026, 4, 14))
        assert alerts == []

    def test_window_excludes_old_days(self, tmp_path):
        # Days outside lookback window shouldn't count
        for i in range(30):
            d = f"2026-04-{i+1:02d}" if i < 30 else None
            if d:
                _write_pick(tmp_path, d, pick_flags=[FLAG_NEEDLE])
        # Today is 2026-04-30; lookback is 14d → window is 2026-04-16 to 2026-04-30
        # That's 15 days of flagged → CRITICAL (>= 8).
        alerts = check(tmp_path, today=date(2026, 4, 30))
        assert alerts[0].level == "CRITICAL"
        # Should mention examined ≤ 15 (one of these days is today inclusive)
        msg = alerts[0].message
        assert "/14" in msg or "/15" in msg
