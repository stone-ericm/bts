"""Tests for Tier-2 projected_lineup frequency check."""

import json
from datetime import date

from bts.health.projected_lineup import check, SOURCE


def _write_day(picks_dir, date_iso, *, pick_proj=False, dd_proj=False, has_dd=True):
    data = {
        "date": date_iso,
        "pick": {"batter_name": "X", "p_game_hit": 0.75, "projected_lineup": pick_proj},
    }
    if has_dd:
        data["double_down"] = {"batter_name": "Y", "p_game_hit": 0.72, "projected_lineup": dd_proj}
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(data))


class TestProjectedLineup:
    def test_no_alert_below_info(self, tmp_path):
        # 14 days, 1 projected (7%) — well below 30% info threshold
        for i in range(1, 14):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=False)
        _write_day(tmp_path, "2026-04-14", pick_proj=True)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert alerts == []

    def test_info_at_30pct(self, tmp_path):
        # 5 of 14 = 35.7% → INFO
        for i in range(1, 6):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=True)
        for i in range(6, 15):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=False)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert len(alerts) == 1
        assert alerts[0].level == "INFO"
        assert alerts[0].source == SOURCE

    def test_warn_at_50pct(self, tmp_path):
        # 7 of 14 = 50% → WARN
        for i in range(1, 8):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=True)
        for i in range(8, 15):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=False)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert len(alerts) == 1
        assert alerts[0].level == "WARN"

    def test_dd_proj_counts(self, tmp_path):
        # If pick is fine but DD is projected, it still counts
        for i in range(1, 6):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=False, dd_proj=True)
        for i in range(6, 15):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=False, dd_proj=False)
        alerts = check(tmp_path, today=date(2026, 4, 14))
        assert len(alerts) == 1
        assert alerts[0].level == "INFO"

    def test_no_alert_with_too_few_days(self, tmp_path):
        # Only 5 days < min_days=7 → no alert even at 100%
        for i in range(1, 6):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=True)
        alerts = check(tmp_path, today=date(2026, 4, 5))
        assert alerts == []

    def test_skips_days_without_pick(self, tmp_path):
        # Days with no pick aren't counted in denominator
        for i in range(1, 8):
            _write_day(tmp_path, f"2026-04-{i:02d}", pick_proj=True)
        # 7 with picks, 5 without (rest days)
        for i in range(8, 13):
            (tmp_path / f"2026-04-{i:02d}.json").write_text(json.dumps({
                "date": f"2026-04-{i:02d}",
                "pick": None,
            }))
        alerts = check(tmp_path, today=date(2026, 4, 12))
        # 7/7 = 100% → WARN (since rest days were excluded)
        assert len(alerts) == 1
        assert alerts[0].level == "WARN"
