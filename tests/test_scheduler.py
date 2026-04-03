"""Tests for the dynamic lineup scheduler."""

import json
import pytest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _game(game_pk: int, time_et: str, team_away: str = "NYM", team_home: str = "ATL"):
    """Build a mock MLB schedule game entry."""
    et_dt = datetime.strptime(f"2026-04-03 {time_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    utc_iso = et_dt.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    return {
        "gamePk": game_pk,
        "gameDate": utc_iso,
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"name": team_away}},
            "home": {"team": {"name": team_home}},
        },
    }


class TestComputeRunTimes:
    def test_single_game(self):
        from bts.scheduler import compute_run_times
        games = [_game(100, "19:05")]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)
        assert len(runs) == 1
        assert runs[0]["time_et"].hour == 18
        assert runs[0]["time_et"].minute == 20
        assert runs[0]["game_pks"] == [100]

    def test_clusters_nearby_games(self):
        from bts.scheduler import compute_run_times
        games = [
            _game(100, "19:05"),
            _game(200, "19:10"),
            _game(300, "19:15"),
        ]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)
        assert len(runs) == 1
        assert sorted(runs[0]["game_pks"]) == [100, 200, 300]

    def test_separates_distant_games(self):
        from bts.scheduler import compute_run_times
        games = [
            _game(100, "13:10"),
            _game(200, "19:05"),
        ]
        runs = compute_run_times(games, offset_min=45, cluster_min=10)
        assert len(runs) == 2
        assert runs[0]["game_pks"] == [100]
        assert runs[1]["game_pks"] == [200]


class TestDetectDoubleheaderGame2:
    def test_finds_doubleheader(self):
        from bts.scheduler import detect_doubleheader_game2s
        games = [
            _game(100, "13:10", "NYM", "ATL"),
            _game(200, "19:05", "NYM", "ATL"),
            _game(300, "19:10", "LAD", "SF"),
        ]
        dh2s = detect_doubleheader_game2s(games)
        assert dh2s == {200}

    def test_no_doubleheader(self):
        from bts.scheduler import detect_doubleheader_game2s
        games = [
            _game(100, "19:05", "NYM", "ATL"),
            _game(200, "19:10", "LAD", "SF"),
        ]
        dh2s = detect_doubleheader_game2s(games)
        assert dh2s == set()


class TestComputeWakeUpTime:
    def test_default_when_no_early_games(self):
        from bts.scheduler import compute_wakeup_time
        games = [_game(100, "19:05")]
        wakeup = compute_wakeup_time(games, default_hour_et=10, early_buffer_min=60)
        assert wakeup.hour == 10
        assert wakeup.minute == 0

    def test_early_wakeup_for_international_game(self):
        from bts.scheduler import compute_wakeup_time
        games = [_game(100, "06:10"), _game(200, "19:05")]
        wakeup = compute_wakeup_time(games, default_hour_et=10, early_buffer_min=60)
        assert wakeup.hour == 5
        assert wakeup.minute == 10
