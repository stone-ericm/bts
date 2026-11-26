"""Unit tests for scheduler's dynamic fallback_deadline_min logic."""
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def test_normal_evening_game_uses_standard_buffer():
    from bts.scheduler import resolve_fallback_deadline_min
    game_et = datetime(2026, 4, 20, 19, 10, tzinfo=ET)  # 7:10 PM
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    assert m == 35


def test_patriots_day_morning_game_uses_standard_buffer():
    """Patriots' Day 11:10 AM ET is NOT before 11:00 — cutoff is strict <."""
    from bts.scheduler import resolve_fallback_deadline_min
    game_et = datetime(2026, 4, 20, 11, 10, tzinfo=ET)
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    assert m == 35


def test_london_morning_game_uses_morning_buffer():
    from bts.scheduler import resolve_fallback_deadline_min
    game_et = datetime(2026, 6, 20, 9, 10, tzinfo=ET)  # London 9:10 AM ET
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    assert m == 25


def test_exactly_cutoff_hour_uses_standard():
    """Cutoff is strict "<", so exactly 11:00 AM ET uses standard."""
    from bts.scheduler import resolve_fallback_deadline_min
    game_et = datetime(2026, 4, 20, 11, 0, tzinfo=ET)
    m = resolve_fallback_deadline_min(
        earliest_game_et=game_et,
        standard_min=35,
        morning_min=25,
        morning_cutoff_hour=11,
    )
    assert m == 35


def test_morning_game_lock_time_is_25_min_before_pitch():
    """Integration check: morning game → lock deadline = first pitch - 25 min."""
    from datetime import timedelta
    from bts.scheduler import resolve_fallback_deadline_min
    game_et = datetime(2026, 6, 22, 9, 10, tzinfo=ET)
    m = resolve_fallback_deadline_min(game_et)  # use defaults
    lock = game_et - timedelta(minutes=m)
    assert lock == datetime(2026, 6, 22, 8, 45, tzinfo=ET)


def test_rejects_morning_min_greater_than_standard_min():
    from bts.scheduler import resolve_fallback_deadline_min
    game_et = datetime(2026, 4, 20, 9, 10, tzinfo=ET)
    with pytest.raises(ValueError, match="morning_min"):
        resolve_fallback_deadline_min(
            game_et,
            standard_min=20,
            morning_min=30,   # invalid — morning should be <= standard
            morning_cutoff_hour=11,
        )


def test_midnight_game_uses_morning_buffer():
    """A hypothetical midnight-ET start (hour=0) should use morning buffer."""
    from bts.scheduler import resolve_fallback_deadline_min
    game_et = datetime(2026, 6, 20, 0, 0, tzinfo=ET)
    m = resolve_fallback_deadline_min(
        game_et, standard_min=35, morning_min=25, morning_cutoff_hour=11,
    )
    assert m == 25
