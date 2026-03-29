import json
import pandas as pd
from bts.data.build import parse_game_feed, build_season
from bts.data.schema import PA_COLUMNS


def test_parse_game_feed_returns_correct_count(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert len(rows) == 2


def test_parse_game_feed_hit_fields(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["is_hit"] == 1
    assert single["event_type"] == "single"
    assert single["batter_id"] == 100001
    assert single["pitcher_id"] == 300001


def test_parse_game_feed_non_hit_fields(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    strikeout = rows[1]
    assert strikeout["is_hit"] == 0
    assert strikeout["event_type"] == "strikeout"
    assert strikeout["batter_id"] == 200001


def test_parse_game_feed_pitch_sequences(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["pitch_count"] == 4
    assert single["pitch_types"] == ["FF", "SL", "CH", "FF"]
    assert single["pitch_calls"] == ["B", "C", "S", "X"]
    assert len(single["pitch_px"]) == 4
    assert len(single["pitch_pz"]) == 4


def test_parse_game_feed_context(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["game_pk"] == 999999
    assert single["date"] == "2025-06-15"
    assert single["season"] == 2025
    assert single["venue_id"] == 3289
    assert single["hp_umpire_id"] == 427215
    assert single["weather_temp"] == 78
    assert single["weather_wind_speed"] == 9
    assert single["weather_wind_dir"] == "Out To CF"
    assert single["roof_type"] == "Open"


def test_parse_game_feed_lineup_position(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert rows[0]["lineup_position"] == 3
    assert rows[1]["lineup_position"] == 1


def test_parse_game_feed_is_home(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert rows[0]["is_home"] is False
    assert rows[1]["is_home"] is True


def test_parse_game_feed_strike_zone(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert rows[0]["sz_top"] == 3.4
    assert rows[0]["sz_bottom"] == 1.7


def test_parse_game_feed_launch_data(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["launch_speed"] == 98.3
    assert single["launch_angle"] == 12.0
    strikeout = rows[1]
    assert strikeout["launch_speed"] is None
    assert strikeout["launch_angle"] is None


def test_parse_game_feed_count(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    assert rows[0]["final_count_balls"] == 1
    assert rows[0]["final_count_strikes"] == 2
    assert rows[1]["final_count_balls"] == 0
    assert rows[1]["final_count_strikes"] == 3


def test_parse_game_feed_has_all_columns(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    for col in PA_COLUMNS:
        assert col in rows[0], f"Missing column: {col}"


def test_build_season_creates_parquet(sample_feed_path, tmp_path):
    raw_dir = sample_feed_path.parent.parent  # tmp_path/raw
    output_path = tmp_path / "processed" / "pa_2025.parquet"

    build_season(raw_dir, output_path, season=2025)

    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert len(df) == 2
    assert df["is_hit"].sum() == 1
    assert df["game_pk"].iloc[0] == 999999


def test_build_season_preserves_pitch_lists(sample_feed_path, tmp_path):
    raw_dir = sample_feed_path.parent.parent
    output_path = tmp_path / "processed" / "pa_2025.parquet"

    build_season(raw_dir, output_path, season=2025)

    df = pd.read_parquet(output_path)
    row = df.iloc[0]
    # Parquet round-trips list columns as array-like; verify as list
    pitch_types = list(row["pitch_types"])
    assert isinstance(pitch_types, list)
    assert pitch_types == ["FF", "SL", "CH", "FF"]


def test_build_season_merges_weather(sample_feed_path, tmp_path):
    weather_path = sample_feed_path.parent / "999999_weather.json"
    weather_path.write_text(json.dumps({
        "surface_pressure": 1010.5,
        "relative_humidity": 72.0,
    }))

    raw_dir = sample_feed_path.parent.parent
    output_path = tmp_path / "processed" / "pa_2025.parquet"

    build_season(raw_dir, output_path, season=2025)

    df = pd.read_parquet(output_path)
    assert df["atm_pressure"].iloc[0] == 1010.5
    assert df["humidity"].iloc[0] == 72.0


def test_build_season_filters_game_type(sample_game_feed, tmp_path):
    raw_dir = tmp_path / "raw"
    season_dir = raw_dir / "2025"
    season_dir.mkdir(parents=True)

    # Regular season game
    reg = sample_game_feed.copy()
    reg["gameData"] = {**sample_game_feed["gameData"], "game": {"pk": 111, "season": "2025", "type": "R"}}
    (season_dir / "111.json").write_text(json.dumps(reg))

    # Spring training game
    spring = sample_game_feed.copy()
    spring["gameData"] = {**sample_game_feed["gameData"], "game": {"pk": 222, "season": "2025", "type": "S"}}
    (season_dir / "222.json").write_text(json.dumps(spring))

    output_path = tmp_path / "processed" / "pa_2025.parquet"
    df = build_season(raw_dir, output_path, season=2025)

    # Default should only include regular season
    assert df["game_pk"].unique().tolist() == [111]
