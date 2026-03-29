import json
from unittest.mock import patch, MagicMock
from bts.data.pull import discover_games, download_game_feed, pull_feeds, enrich_weather


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


def test_download_game_feed_writes_json(tmp_path):
    sample_feed = {"gameData": {"game": {"pk": 123456}}, "liveData": {}}

    def _mock_urlopen_feed(url, **kwargs):
        resp = MagicMock()
        resp.read.return_value = json.dumps(sample_feed).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("bts.data.pull.urlopen", side_effect=_mock_urlopen_feed):
        path = download_game_feed(123456, tmp_path / "2025")

    assert path.exists()
    assert path.name == "123456.json"
    data = json.loads(path.read_text())
    assert data["gameData"]["game"]["pk"] == 123456


def test_download_game_feed_skips_existing(tmp_path):
    out_dir = tmp_path / "2025"
    out_dir.mkdir()
    existing = out_dir / "123456.json"
    existing.write_text('{"already": "here"}')

    with patch("bts.data.pull.urlopen") as mock_open:
        path = download_game_feed(123456, out_dir)

    mock_open.assert_not_called()
    assert json.loads(path.read_text()) == {"already": "here"}


MOCK_OPEN_METEO_RESPONSE = {
    "hourly": {
        "time": ["2025-06-15T19:00"],
        "surface_pressure": [1013.2],
        "relative_humidity_2m": [65],
        "dewpoint_2m": [18.5],
    }
}


def test_enrich_weather_writes_sidecar(tmp_path):
    game_feed = {
        "gameData": {
            "datetime": {"officialDate": "2025-06-15", "dateTime": "2025-06-15T23:10:00Z"},
            "venue": {
                "location": {"defaultCoordinates": {"latitude": 40.757, "longitude": -73.845}},
            },
        },
    }
    raw_dir = tmp_path / "2025"
    raw_dir.mkdir()
    (raw_dir / "999999.json").write_text(json.dumps(game_feed))

    def _mock_meteo(url, **kwargs):
        resp = MagicMock()
        resp.read.return_value = json.dumps(MOCK_OPEN_METEO_RESPONSE).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("bts.data.pull.urlopen", side_effect=_mock_meteo):
        enrich_weather(raw_dir)

    weather_path = raw_dir / "999999_weather.json"
    assert weather_path.exists()
    data = json.loads(weather_path.read_text())
    assert data["surface_pressure"] == 1013.2
    assert data["relative_humidity"] == 65


def test_enrich_weather_skips_existing(tmp_path):
    raw_dir = tmp_path / "2025"
    raw_dir.mkdir()
    (raw_dir / "999999.json").write_text("{}")
    (raw_dir / "999999_weather.json").write_text('{"already": "enriched"}')

    with patch("bts.data.pull.urlopen") as mock_open:
        enrich_weather(raw_dir)

    mock_open.assert_not_called()


def test_pull_feeds_orchestrates(tmp_path):
    games = [{"gamePk": 111, "date": "2025-06-01"}, {"gamePk": 222, "date": "2025-06-01"}]
    sample_feed = {"gameData": {}, "liveData": {}}

    def _mock_urlopen_any(url, **kwargs):
        resp = MagicMock()
        resp.read.return_value = json.dumps(sample_feed).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("bts.data.pull.discover_games", return_value=games), \
         patch("bts.data.pull.urlopen", side_effect=_mock_urlopen_any):
        paths = pull_feeds("2025-06-01", "2025-06-01", tmp_path, delay=0)

    assert len(paths) == 2
    assert (tmp_path / "2025" / "111.json").exists()
    assert (tmp_path / "2025" / "222.json").exists()
