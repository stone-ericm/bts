import copy
import json
import pandas as pd
from bts.data.build import (
    parse_game_feed,
    build_season,
    filter_out_resumed_portion,
    read_pa_for_bts_scoring,
)
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


def test_parse_game_feed_statcast_hit_data(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["trajectory"] == "line_drive"
    assert single["hardness"] == "hard"
    assert single["total_distance"] == 310.0
    strikeout = rows[1]
    assert strikeout["trajectory"] is None
    assert strikeout["hardness"] is None
    assert strikeout["total_distance"] is None


def test_parse_game_feed_statcast_pitch_data(sample_game_feed):
    rows = parse_game_feed(sample_game_feed)
    single = rows[0]
    assert single["pitch_speeds"] == [93.5, 85.2, 84.1, 94.0]
    assert single["pitch_spin_rates"] == [2400, 2700, 1800, 2350]
    assert single["pitch_extensions"] == [6.3, 6.1, 6.2, 6.4]
    assert single["pitch_break_vertical"] == [-15.0, -32.0, -28.0, -14.0]
    assert single["pitch_break_horizontal"] == [8.0, -2.0, -12.0, 9.0]
    assert single["pitch_end_speeds"] == [85.0, 78.0, 76.0, 86.0]


def test_parse_game_feed_normal_game_never_resumed(sample_game_feed):
    """No resumeDateTime -> is_resumed_portion is always False, even when a normal
    late game's UTC play times cross a calendar date. This is the timezone trap a
    raw date-prefix comparison would fall into; gating on resumeDateTime avoids it.
    """
    feed = copy.deepcopy(sample_game_feed)
    for i, play in enumerate(feed["liveData"]["plays"]["allPlays"]):
        # UTC times straddling midnight, but NOT a suspension (no resumeDateTime)
        play["about"]["startTime"] = "2025-06-15T23:55:00Z" if i == 0 else "2025-06-16T00:30:00Z"
    rows = parse_game_feed(feed)
    assert all(row["is_resumed_portion"] is False for row in rows)


def test_parse_game_feed_flags_resumed_portion(sample_game_feed):
    """A suspended game flags only plays at/after resumeDateTime as the resumed portion."""
    feed = copy.deepcopy(sample_game_feed)
    feed["gameData"]["datetime"]["resumeDateTime"] = "2025-06-16T18:00:00Z"
    plays = feed["liveData"]["plays"]["allPlays"]
    plays[0]["about"]["startTime"] = "2025-06-15T23:30:00Z"  # original day -> pre-suspension
    plays[1]["about"]["startTime"] = "2025-06-16T18:30:00Z"  # resume day -> resumed
    rows = parse_game_feed(feed)
    assert rows[0]["is_resumed_portion"] is False
    assert rows[1]["is_resumed_portion"] is True


def test_parse_game_feed_resume_boundary_inclusive(sample_game_feed):
    """The boundary is `startTime >= resumeDateTime` (resumption instant is resumed)."""
    feed = copy.deepcopy(sample_game_feed)
    feed["gameData"]["datetime"]["resumeDateTime"] = "2025-06-16T18:00:00Z"
    plays = feed["liveData"]["plays"]["allPlays"]
    plays[0]["about"]["startTime"] = "2025-06-16T17:59:59Z"  # one second before -> pre
    plays[1]["about"]["startTime"] = "2025-06-16T18:00:00Z"  # exactly at -> resumed
    rows = parse_game_feed(feed)
    assert rows[0]["is_resumed_portion"] is False
    assert rows[1]["is_resumed_portion"] is True


def test_filter_out_resumed_portion_drops_resumed_rows():
    df = pd.DataFrame({
        "batter_id": [1, 1, 2],
        "is_hit": [0, 1, 1],
        "is_resumed_portion": [False, True, False],
    })
    out = filter_out_resumed_portion(df)
    assert len(out) == 2
    assert not out["is_resumed_portion"].any()
    # batter 1's resumed-portion hit is dropped; only the pre-suspension out remains
    assert out[out["batter_id"] == 1]["is_hit"].tolist() == [0]


def test_filter_out_resumed_portion_backward_compatible():
    df = pd.DataFrame({"batter_id": [1], "is_hit": [1]})  # pre-enrichment PA, no column
    out = filter_out_resumed_portion(df)
    assert len(out) == 1  # returned unchanged


def test_read_pa_for_bts_scoring_excludes_resumed(tmp_path):
    path = tmp_path / "pa.parquet"
    pd.DataFrame({
        "date": ["2026-06-16", "2026-06-16"],
        "batter_id": [1, 1],
        "game_pk": [100, 100],
        "is_hit": [0, 1],
        "season": [2026, 2026],  # extra column not requested
        "is_resumed_portion": [False, True],
    }).to_parquet(path, index=False)
    out = read_pa_for_bts_scoring(path, ["date", "batter_id", "game_pk", "is_hit"])
    assert list(out.columns) == ["date", "batter_id", "game_pk", "is_hit"]  # flag dropped
    assert len(out) == 1  # resumed-portion row excluded
    assert out["is_hit"].tolist() == [0]


def test_read_pa_for_bts_scoring_backward_compatible(tmp_path):
    path = tmp_path / "pa_old.parquet"
    pd.DataFrame({
        "date": ["2026-06-16"], "batter_id": [1], "game_pk": [100], "is_hit": [1],
    }).to_parquet(path, index=False)  # no is_resumed_portion column
    out = read_pa_for_bts_scoring(path, ["date", "batter_id", "game_pk", "is_hit"])
    assert len(out) == 1  # unchanged
    assert list(out.columns) == ["date", "batter_id", "game_pk", "is_hit"]


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
