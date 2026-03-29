import json
from unittest.mock import patch, MagicMock
from bts.data.pull import discover_games


MOCK_SCHEDULE_RESPONSE = {
    "dates": [{
        "games": [
            {
                "gamePk": 823651,
                "officialDate": "2025-06-01",
                "status": {"detailedState": "Final"},
                "teams": {
                    "away": {"team": {"name": "Pittsburgh Pirates"}},
                    "home": {"team": {"name": "New York Mets"}},
                },
            },
            {
                "gamePk": 823652,
                "officialDate": "2025-06-01",
                "status": {"detailedState": "Scheduled"},
                "teams": {
                    "away": {"team": {"name": "Boston Red Sox"}},
                    "home": {"team": {"name": "Cincinnati Reds"}},
                },
            },
        ]
    }]
}


def _mock_urlopen(url, **kwargs):
    resp = MagicMock()
    resp.read.return_value = json.dumps(MOCK_SCHEDULE_RESPONSE).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@patch("bts.data.pull.urlopen", side_effect=_mock_urlopen)
def test_discover_games_returns_final_games_only(mock_open):
    games = discover_games("2025-06-01", "2025-06-01")
    assert len(games) == 1
    assert games[0]["gamePk"] == 823651


@patch("bts.data.pull.urlopen", side_effect=_mock_urlopen)
def test_discover_games_includes_date(mock_open):
    games = discover_games("2025-06-01", "2025-06-01")
    assert games[0]["date"] == "2025-06-01"
