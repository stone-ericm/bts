"""Tests for lineup collection polling logic."""
import json
from unittest.mock import patch, MagicMock

import pytest

from bts.data.lineup_collect import poll_game_lineup, LineupPollResult


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
