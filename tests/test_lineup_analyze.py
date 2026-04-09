"""Tests for lineup-time distribution analysis."""
import json
from datetime import datetime, timezone, timedelta
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


def test_load_samples_reads_both_sides(tmp_path: Path):
    jsonl = tmp_path / "2026-04-10.jsonl"
    jsonl.write_text(
        json.dumps({
            "game_pk": 1,
            "game_time_et": "2026-04-10T19:05:00-04:00",
            "first_away_confirmed_utc": "2026-04-10T22:00:00+00:00",
            "first_home_confirmed_utc": "2026-04-10T21:45:00+00:00",
            "poll_count": 5,
        }) + "\n"
    )

    from bts.data.lineup_analyze import load_samples_from_jsonl
    samples = load_samples_from_jsonl(tmp_path, from_date="2026-04-10", to_date="2026-04-10")
    # Two samples per game (away + home)
    assert len(samples) == 2
    # 23:05 UTC is first pitch; 22:00 UTC is 65 min before; 21:45 is 80 min before
    assert sorted(samples) == [65, 80]


def test_load_samples_ignores_null_confirmations(tmp_path: Path):
    jsonl = tmp_path / "2026-04-10.jsonl"
    jsonl.write_text(
        json.dumps({
            "game_pk": 1,
            "game_time_et": "2026-04-10T19:05:00-04:00",
            "first_away_confirmed_utc": None,
            "first_home_confirmed_utc": "2026-04-10T21:45:00+00:00",
            "poll_count": 5,
        }) + "\n"
    )
    from bts.data.lineup_analyze import load_samples_from_jsonl
    samples = load_samples_from_jsonl(tmp_path, from_date="2026-04-10", to_date="2026-04-10")
    assert samples == [80]


def test_load_samples_respects_date_range(tmp_path: Path):
    for date, minutes_before in [("2026-04-08", 60), ("2026-04-10", 75)]:
        jsonl = tmp_path / f"{date}.jsonl"
        game_time_et = f"{date}T19:05:00-04:00"
        lineup_utc = (datetime.fromisoformat(game_time_et)
                      - timedelta(minutes=minutes_before)).astimezone(timezone.utc).isoformat()
        jsonl.write_text(
            json.dumps({
                "game_pk": 1,
                "game_time_et": game_time_et,
                "first_away_confirmed_utc": lineup_utc,
                "first_home_confirmed_utc": lineup_utc,
                "poll_count": 1,
            }) + "\n"
        )

    from bts.data.lineup_analyze import load_samples_from_jsonl
    samples = load_samples_from_jsonl(tmp_path, from_date="2026-04-10", to_date="2026-04-10")
    assert samples == [75, 75]  # Only 2026-04-10's two sides


def test_backfill_from_scheduler_state(tmp_path: Path):
    # Fake Pi5 scheduler state: two runs on one day
    picks_dir = tmp_path / "picks"
    date_dir = picks_dir / "2026-04-05"
    date_dir.mkdir(parents=True)
    scheduler_state = {
        "date": "2026-04-05",
        "games": [
            {"game_pk": 1, "game_time_et": "2026-04-05T19:05:00-04:00",
             "lineup_confirmed": True, "is_doubleheader_game2": False},
            {"game_pk": 2, "game_time_et": "2026-04-05T19:10:00-04:00",
             "lineup_confirmed": True, "is_doubleheader_game2": False},
        ],
        "confirmed_game_pks": [1, 2],
        "runs_completed": [
            {"time": "2026-04-05T18:20:00-04:00", "new_lineups": 1, "skipped": False,
             "pick_name": None, "pick_p": None},  # 45 min before 19:05 ET
            {"time": "2026-04-05T18:25:00-04:00", "new_lineups": 1, "skipped": False,
             "pick_name": None, "pick_p": None},  # 40 min before 19:05 ET
        ],
        "pick_locked": True,
        "pick_locked_at": "2026-04-05T18:25:00-04:00",
        "result_status": "final",
        "next_wakeup": None,
        "schedule_fetched_at": "2026-04-05T10:00:00-04:00",
    }
    (date_dir / "scheduler_state.json").write_text(json.dumps(scheduler_state))

    from bts.data.lineup_analyze import backfill_from_scheduler_state
    samples = backfill_from_scheduler_state(picks_dir)

    # Two runs, each with 1 new_lineup → 2 samples
    # First run 45 min before first pitch (of game 1 at 19:05)
    # Second run 40 min before
    assert len(samples) == 2
    assert sorted(samples) == [40, 45]


def test_backfill_returns_empty_when_no_state(tmp_path: Path):
    picks_dir = tmp_path / "empty"
    picks_dir.mkdir()
    from bts.data.lineup_analyze import backfill_from_scheduler_state
    assert backfill_from_scheduler_state(picks_dir) == []
