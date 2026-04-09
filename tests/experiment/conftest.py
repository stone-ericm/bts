import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mini_pa_df():
    """Minimal PA DataFrame with enough structure to test experiment hooks.

    50 PAs across 5 dates, 5 batters, 2 pitchers. Includes all 15 baseline
    feature columns (filled with plausible random values) plus is_hit, season,
    game_pk, batter_id, pitcher_id, date.
    """
    rng = np.random.default_rng(42)
    n = 50
    dates = pd.date_range("2025-06-01", periods=5, freq="D")
    batter_ids = [100001, 100002, 100003, 100004, 100005]
    pitcher_ids = [200001, 200002]

    rows = []
    for i in range(n):
        rows.append({
            "date": dates[i % 5],
            "season": 2025,
            "game_pk": 900000 + (i % 5),
            "batter_id": batter_ids[i % 5],
            "pitcher_id": pitcher_ids[i % 2],
            "pitch_hand": "R" if i % 2 == 0 else "L",
            "bat_side": "L" if i % 3 == 0 else "R",
            "is_hit": int(rng.random() > 0.7),
            "batter_hr_7g": rng.uniform(0.2, 0.4),
            "batter_hr_30g": rng.uniform(0.2, 0.35),
            "batter_hr_60g": rng.uniform(0.22, 0.33),
            "batter_hr_120g": rng.uniform(0.23, 0.32),
            "batter_whiff_60g": rng.uniform(0.15, 0.35),
            "batter_count_tendency_30g": rng.uniform(-0.5, 0.5),
            "batter_gb_hit_rate": rng.uniform(0.15, 0.25),
            "platoon_hr": rng.uniform(0.2, 0.35),
            "pitcher_hr_30g": rng.uniform(0.2, 0.3),
            "pitcher_entropy_30g": rng.uniform(0.5, 2.0),
            "pitcher_catcher_framing": rng.uniform(0.25, 0.35),
            "opp_bullpen_hr_30g": rng.uniform(0.22, 0.3),
            "weather_temp": rng.uniform(60, 90),
            "park_factor": rng.uniform(0.9, 1.1),
            "days_rest": rng.choice([0, 1, 2, 3]),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_profiles_df():
    """Mock daily profiles DataFrame (output of blend_walk_forward)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-06-01", periods=10, freq="D")
    rows = []
    for d in dates:
        for rank in range(1, 11):
            rows.append({
                "date": d.date(),
                "rank": rank,
                "batter_id": 100000 + rank,
                "p_game_hit": max(0.5, 0.95 - rank * 0.04 + rng.normal(0, 0.02)),
                "actual_hit": int(rng.random() > (0.1 + rank * 0.03)),
                "n_pas": rng.choice([3, 4, 5]),
                "season": 2025,
            })
    return pd.DataFrame(rows)
