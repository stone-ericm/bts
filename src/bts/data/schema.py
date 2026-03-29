"""PA table schema definitions and event constants."""

HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})

PA_ENDING_EVENTS = frozenset({
    "single", "double", "triple", "home_run",
    "field_out", "strikeout", "walk", "hit_by_pitch",
    "force_out", "grounded_into_double_play", "double_play",
    "field_error", "sac_fly", "sac_bunt",
    "fielders_choice", "fielders_choice_out",
    "strikeout_double_play", "catcher_interf",
    "sac_fly_double_play", "triple_play", "sac_bunt_double_play",
})

# Ordered list of columns in the PA Parquet table.
# Pitch sequence columns (pitch_types, pitch_calls, pitch_px, pitch_pz)
# are stored as nested lists in Parquet.
PA_COLUMNS = [
    "game_pk",
    "date",
    "season",
    "batter_id",
    "pitcher_id",
    "bat_side",
    "pitch_hand",
    "lineup_position",
    "is_home",
    "hp_umpire_id",
    "venue_id",
    "pitch_count",
    "pitch_types",
    "pitch_calls",
    "pitch_px",
    "pitch_pz",
    "sz_top",
    "sz_bottom",
    "final_count_balls",
    "final_count_strikes",
    "launch_speed",
    "launch_angle",
    "event_type",
    "is_hit",
    "weather_temp",
    "weather_wind_speed",
    "weather_wind_dir",
    "roof_type",
    "atm_pressure",
    "humidity",
]
