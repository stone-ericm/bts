"""Regression test: London Series (first pitch 2:10 PM BST) TZ round-trip.

Ensures our scheduler converts the MLB StatsAPI UTC gameDate to ET correctly
during British Summer Time and handles the morning-game fallback_deadline_min
dispatch properly. Also covers the DST-boundary window where the UK and US
switch DST on different dates.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from bts.scheduler import resolve_fallback_deadline_min

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def test_london_series_first_pitch_converts_correctly():
    """A 2:10 PM London (BST, UTC+1) first pitch = 13:10 UTC = 09:10 ET (during EDT)."""
    # Simulate MLB StatsAPI output: gameDate is UTC
    utc_first_pitch = datetime(2026, 6, 20, 13, 10, tzinfo=UTC)
    et_first_pitch = utc_first_pitch.astimezone(ET)
    assert et_first_pitch.hour == 9
    assert et_first_pitch.minute == 10


def test_london_series_uses_morning_fallback_buffer():
    """London 9:10 AM ET first pitch should trigger the morning fallback_deadline_min."""
    game_et = datetime(2026, 6, 20, 9, 10, tzinfo=ET)
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35, morning_min=25, morning_cutoff_hour=11,
    )
    assert m == 25


def test_dst_boundary_march_game():
    """DST transition: US spring-forward is March 8 2026, UK is March 29 2026.

    On March 15, 2026, the UK is still on GMT (UTC+0) but the US is on EDT (UTC-4).
    So a London 2:10 PM local first pitch = 14:10 UTC = 10:10 AM ET (not 9:10).
    Morning buffer should still apply at 10:10 AM.
    """
    utc_first_pitch = datetime(2026, 3, 15, 14, 10, tzinfo=UTC)
    et_first_pitch = utc_first_pitch.astimezone(ET)
    assert et_first_pitch.hour == 10
    assert et_first_pitch.minute == 10
    m = resolve_fallback_deadline_min(
        earliest_game_et=et_first_pitch,
        standard_min=35, morning_min=25, morning_cutoff_hour=11,
    )
    assert m == 25


def test_london_series_expected_lock_math():
    """Integration: London 9:10 AM ET + morning buffer (25 min) = 8:45 AM ET lock deadline."""
    game_et = datetime(2026, 6, 20, 9, 10, tzinfo=ET)
    m = resolve_fallback_deadline_min(game_et)  # use defaults
    expected_lock = game_et - timedelta(minutes=m)
    assert expected_lock == datetime(2026, 6, 20, 8, 45, tzinfo=ET)
