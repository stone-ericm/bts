"""Tests for lineup collection polling logic."""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bts.data.lineup_collect import (
    CollectionState,
    GameCollectionEntry,
    LineupPollResult,
    poll_game_lineup,
    run_collection_tick,
)


def test_poll_returns_no_lineup_when_battingorder_empty():
    fake_response = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {"ID123": {"battingOrder": ""}}},
                    "home": {"players": {"ID456": {"battingOrder": ""}}},
                }
            }
        }
    }
    with patch("bts.data.lineup_collect.retry_urlopen") as mock_fetch:
        mock_fetch.return_value.read.return_value = json.dumps(fake_response).encode()
        result = poll_game_lineup(game_pk=12345)
    assert result == LineupPollResult(game_pk=12345, away_confirmed=False, home_confirmed=False)


def test_poll_returns_away_confirmed_when_away_has_battingorder():
    fake_response = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {"ID123": {"battingOrder": "100"}}},
                    "home": {"players": {"ID456": {}}},
                }
            }
        }
    }
    with patch("bts.data.lineup_collect.retry_urlopen") as mock_fetch:
        mock_fetch.return_value.read.return_value = json.dumps(fake_response).encode()
        result = poll_game_lineup(game_pk=12345)
    assert result.away_confirmed is True
    assert result.home_confirmed is False


def test_poll_returns_both_confirmed():
    fake_response = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {"ID1": {"battingOrder": "100"}}},
                    "home": {"players": {"ID2": {"battingOrder": "200"}}},
                }
            }
        }
    }
    with patch("bts.data.lineup_collect.retry_urlopen") as mock_fetch:
        mock_fetch.return_value.read.return_value = json.dumps(fake_response).encode()
        result = poll_game_lineup(game_pk=12345)
    assert result.away_confirmed is True
    assert result.home_confirmed is True


def test_poll_returns_both_false_on_api_error():
    with patch("bts.data.lineup_collect.retry_urlopen", side_effect=Exception("network down")):
        result = poll_game_lineup(game_pk=12345)
    assert result.game_pk == 12345
    assert result.away_confirmed is False
    assert result.home_confirmed is False


def test_collection_state_records_first_confirmation():
    state = CollectionState(date="2026-04-10")
    now = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)

    state.record_poll(
        game_pk=12345,
        game_time_et="2026-04-10T19:05:00-04:00",
        poll_time_utc=now,
        away_confirmed=True,
        home_confirmed=False,
    )

    entry = state.games[12345]
    assert entry.first_away_confirmed_utc == now.isoformat()
    assert entry.first_home_confirmed_utc is None
    assert entry.poll_count == 1


def test_collection_state_does_not_overwrite_first_confirmation():
    state = CollectionState(date="2026-04-10")
    first = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)
    second = datetime(2026, 4, 10, 17, 35, tzinfo=timezone.utc)

    state.record_poll(12345, "2026-04-10T19:05:00-04:00", first, True, False)
    state.record_poll(12345, "2026-04-10T19:05:00-04:00", second, True, True)

    entry = state.games[12345]
    assert entry.first_away_confirmed_utc == first.isoformat()
    assert entry.first_home_confirmed_utc == second.isoformat()
    assert entry.poll_count == 2


def test_collection_state_serializes_to_jsonl(tmp_path: Path):
    state = CollectionState(date="2026-04-10")
    now = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)
    state.record_poll(12345, "2026-04-10T19:05:00-04:00", now, True, True)

    state.write_jsonl(tmp_path)

    out_file = tmp_path / "2026-04-10.jsonl"
    assert out_file.exists()
    line = json.loads(out_file.read_text().strip())
    assert line["game_pk"] == 12345
    assert line["first_away_confirmed_utc"] == now.isoformat()
    assert line["first_home_confirmed_utc"] == now.isoformat()
    assert line["poll_count"] == 1


def test_run_collection_tick_polls_only_games_needing_confirmation():
    state = CollectionState(date="2026-04-10")
    now_confirmed = datetime(2026, 4, 10, 17, 30, tzinfo=timezone.utc)
    # Game 1 already has both sides confirmed from a previous tick
    state.record_poll(1, "2026-04-10T19:05:00-04:00", now_confirmed, True, True)
    # Game 2 needs confirmation
    state.games[2] = GameCollectionEntry(
        game_pk=2,
        game_time_et="2026-04-10T19:10:00-04:00",
    )

    mock_poll = MagicMock()
    mock_poll.return_value = LineupPollResult(game_pk=2, away_confirmed=True, home_confirmed=False)

    now = datetime(2026, 4, 10, 17, 45, tzinfo=timezone.utc)
    with patch("bts.data.lineup_collect.poll_game_lineup", mock_poll):
        run_collection_tick(state, now_utc=now)

    # Game 1 should NOT be polled (both confirmed)
    # Game 2 should be polled once
    assert mock_poll.call_count == 1
    mock_poll.assert_called_with(2)
    # Game 2 now has away confirmed
    assert state.games[2].first_away_confirmed_utc == now.isoformat()
