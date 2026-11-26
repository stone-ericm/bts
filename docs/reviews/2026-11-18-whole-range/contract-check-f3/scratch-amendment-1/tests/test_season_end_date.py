"""The season end must follow the date being decided, not one hardcoded year.

Found 2026-11-12: ``SEASON_END_DATE = "2026-09-28"`` made every 2027 date read
as zero days left, and at zero days the reach-57 lookup returns skip. The whole
2027 season would have been skipped without an error.
"""
from __future__ import annotations

from datetime import date

from bts.health import tail_policy as tail_health
from bts.simulate.tail_policy import OBJECTIVE_REACH57
from bts.strategy import _days_remaining, _regime, season_end_date


def test_2026_dates_are_unchanged():
    assert season_end_date("2026-09-04") == "2026-09-28"
    assert _days_remaining("2026-09-04") == 24
    assert _days_remaining("2026-06-17") == 103


def test_2027_opening_night_has_a_full_season_ahead():
    # Whole-range fix set A contract A1 uses the exclusive season boundary.
    assert season_end_date("2027-03-24") == "2027-09-27"
    assert _days_remaining("2027-03-24") == 187


def test_2027_last_day_and_after():
    # Whole-range fix set A contract A1 counts the playing finale.
    assert _days_remaining("2027-09-26") == 1
    assert _days_remaining("2027-09-27") == 0
    assert _days_remaining("2027-10-05") == 0


def test_2027_opener_is_a_live_reach_57_state_not_a_dead_one():
    objective, days = _regime(None, 0, "2027-03-24")
    assert objective == OBJECTIVE_REACH57
    assert days == 180   # clamped to the table's season length, and not zero


def test_unconfigured_year_falls_back_to_late_september_and_keeps_picking():
    assert season_end_date("2031-06-01") == "2031-09-28"
    assert _days_remaining("2031-06-01") == 119


def test_health_warns_when_the_year_has_no_configured_season_end(tmp_path):
    alerts = tail_health.season_end_alerts(today=date(2031, 6, 1))
    assert [a.level for a in alerts] == ["WARN"]
    assert "2031" in alerts[0].message


def test_health_is_silent_for_a_configured_year():
    assert tail_health.season_end_alerts(today=date(2027, 6, 1)) == []


def test_health_check_carries_the_season_end_warning(tmp_path):
    alerts = tail_health.check(base_path=tmp_path / "missing.npz",
                               tail_path=tmp_path / "missing_tail.npz",
                               today=date(2031, 6, 1))
    assert any("season end" in a.message and "2031" in a.message for a in alerts)
