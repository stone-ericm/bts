from bts.data.schema import PA_COLUMNS, HIT_EVENTS, PA_ENDING_EVENTS


def test_pa_columns_has_required_fields():
    required = [
        "game_pk", "date", "season", "batter_id", "pitcher_id",
        "bat_side", "pitch_hand",
        "lineup_position", "is_home", "hp_umpire_id", "venue_id",
        "pitch_count", "pitch_types", "pitch_calls", "pitch_px", "pitch_pz",
        "sz_top", "sz_bottom", "final_count_balls", "final_count_strikes",
        "launch_speed", "launch_angle", "event_type", "is_hit",
        "weather_temp", "weather_wind_speed", "weather_wind_dir", "roof_type",
        "atm_pressure", "humidity",
    ]
    for col in required:
        assert col in PA_COLUMNS, f"Missing column: {col}"


def test_hit_events_are_subset_of_pa_ending():
    for event in HIT_EVENTS:
        assert event in PA_ENDING_EVENTS, f"{event} not in PA_ENDING_EVENTS"


def test_hit_events_contains_all_hit_types():
    assert "single" in HIT_EVENTS
    assert "double" in HIT_EVENTS
    assert "triple" in HIT_EVENTS
    assert "home_run" in HIT_EVENTS
    assert len(HIT_EVENTS) == 4


def test_pa_columns_has_statcast_fields():
    statcast_fields = [
        "trajectory", "hardness", "total_distance",
        "pitch_speeds", "pitch_spin_rates", "pitch_extensions",
        "pitch_break_vertical", "pitch_break_horizontal",
        "pitch_end_speeds",
        "fielding_catcher_id",
        "challenge_player_id", "challenge_role", "challenge_overturned", "challenge_team_batting",
    ]
    for field in statcast_fields:
        assert field in PA_COLUMNS, f"Missing Statcast field: {field}"
