"""Tests for lineup-time distribution analysis."""
import json
from pathlib import Path

import pytest

from bts.data.lineup_analyze import (
    compute_minutes_before_first_pitch,
    compute_distribution,
    Distribution,
)


def test_compute_minutes_before_first_pitch_positive_when_lineup_before_game():
    # Game at 19:05 ET (23:05 UTC), lineup confirmed at 18:00 ET (22:00 UTC)
    # → 65 minutes before first pitch
    result = compute_minutes_before_first_pitch(
        lineup_time_utc="2026-04-10T22:00:00+00:00",
        game_time_et="2026-04-10T19:05:00-04:00",
    )
    assert result == 65


def test_compute_minutes_before_first_pitch_negative_when_after():
    # Lineup 'confirmed' 10 min after first pitch (anomaly, should still handle)
    result = compute_minutes_before_first_pitch(
        lineup_time_utc="2026-04-10T23:15:00+00:00",
        game_time_et="2026-04-10T19:05:00-04:00",
    )
    assert result == -10


def test_compute_distribution_percentiles():
    # 10 games, minutes-before-first-pitch values from 30 to 120
    samples = [30, 45, 60, 70, 80, 90, 100, 110, 115, 120]
    dist = compute_distribution(samples)
    assert dist.n == 10
    assert dist.p10 == pytest.approx(44, abs=1)
    assert dist.p50 == pytest.approx(85, abs=1)
    assert dist.p90 == pytest.approx(116, abs=1)
    assert dist.p95 == pytest.approx(118, abs=1)


def test_compute_distribution_empty():
    dist = compute_distribution([])
    assert dist.n == 0
    assert dist.p50 is None
