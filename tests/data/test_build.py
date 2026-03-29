import pandas as pd
from bts.data.build import parse_game_feed
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
