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


class TestCheckConfirmedLineups:
    @patch("bts.scheduler.retry_urlopen")
    def test_detects_confirmed_lineup(self, mock_urlopen):
        from bts.scheduler import check_confirmed_lineups

        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {
                    "ID123": {"battingOrder": "100", "person": {"fullName": "A"}},
                    "ID456": {"battingOrder": "200", "person": {"fullName": "B"}},
                }},
                "home": {"players": {
                    "ID789": {"battingOrder": "100", "person": {"fullName": "C"}},
                }},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        result = check_confirmed_lineups([111])
        assert result == {111: True}

    @patch("bts.scheduler.retry_urlopen")
    def test_detects_no_lineup(self, mock_urlopen):
        from bts.scheduler import check_confirmed_lineups

        feed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {}},
                "home": {"players": {}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed).encode()

        result = check_confirmed_lineups([111])
        assert result == {111: False}

    @patch("bts.scheduler.retry_urlopen")
    def test_counts_new_confirmations(self, mock_urlopen):
        from bts.scheduler import count_new_confirmations

        feed_confirmed = {
            "liveData": {"boxscore": {"teams": {
                "away": {"players": {"ID1": {"battingOrder": "100", "person": {"fullName": "A"}}}},
                "home": {"players": {"ID2": {"battingOrder": "100", "person": {"fullName": "B"}}}},
            }}},
        }
        mock_urlopen.return_value.read.return_value = json.dumps(feed_confirmed).encode()

        previously_confirmed = set()
        new_count = count_new_confirmations([111], previously_confirmed)
        assert new_count == 1
        assert 111 in previously_confirmed


class TestSchedulerState:
    def test_save_and_load_roundtrip(self, tmp_path):
        from bts.scheduler import SchedulerState, save_state, load_state

        state = SchedulerState(
            date="2026-04-03",
            schedule_fetched_at="2026-04-03T10:00:00-04:00",
            games=[{"game_pk": 100, "game_time_et": "2026-04-03T19:05:00-04:00",
                     "lineup_confirmed": False, "is_doubleheader_game2": False}],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=False,
            pick_locked_at=None,
            result_status=None,
            next_wakeup=None,
        )
        save_state(state, tmp_path)

        loaded = load_state("2026-04-03", tmp_path)
        assert loaded is not None
        assert loaded.date == "2026-04-03"
        assert len(loaded.games) == 1
        assert loaded.pick_locked is False

    def test_load_returns_none_when_missing(self, tmp_path):
        from bts.scheduler import load_state

        assert load_state("2026-04-03", tmp_path) is None


class TestSchedulerRun:
    @patch("bts.scheduler.check_confirmed_lineups")
    def test_skips_run_when_no_new_lineups(self, mock_lineups, tmp_path):
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: False}

        result = run_single_check(
            date="2026-04-03",
            all_game_pks=[100],
            confirmed_game_pks=set(),
            config={"orchestrator": {"picks_dir": str(tmp_path)}, "tiers": []},
            early_lock_gap=0.03,
        )
        assert result["skipped"] is True
        assert result["new_lineups"] == 0

    @patch("bts.scheduler.check_confirmed_lineups")
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.strategy.get_game_statuses", return_value={100: "P"})
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_triggers_prediction_on_new_lineup(
        self, _mdp, _statuses, mock_cascade, mock_lineups, tmp_path
    ):
        import pandas as pd
        from bts.scheduler import run_single_check

        mock_lineups.return_value = {100: True}
        mock_cascade.return_value = (
            pd.DataFrame([{
                "batter_name": "Test", "batter_id": 1, "team": "NYM",
                "lineup": 1, "pitcher_name": "P", "pitcher_id": 2,
                "game_pk": 100, "game_time": "2026-04-03T23:05:00Z",
                "p_hit_pa": 0.30, "p_game_hit": 0.82, "flags": "",
            }]),
            "mac",
        )

        result = run_single_check(
            date="2026-04-03",
            all_game_pks=[100],
            confirmed_game_pks=set(),
            config={
                "orchestrator": {"picks_dir": str(tmp_path)},
                "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
            },
            early_lock_gap=0.03,
        )
        assert result["skipped"] is False
        assert result["new_lineups"] == 1
