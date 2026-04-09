"""Blend walk-forward backtest that saves daily prediction profiles.

Adapts the existing walk_forward_evaluate to use the 12-model blend
(same BLEND_CONFIGS from predict.py) and save top-10 ranked predictions
per day to parquet for strategy simulation.
"""

import sys
from pathlib import Path

PROFILE_COLUMNS = ["date", "rank", "batter_id", "p_game_hit", "actual_hit", "n_pas"]


def blend_walk_forward(
    df: "pd.DataFrame",
    test_season: int,
    retrain_every: int = 7,
    top_n: int = 10,
    blend_configs: list | None = None,
    lgb_params: dict | None = None,
) -> "pd.DataFrame":
    """Run blend walk-forward evaluation and return daily profiles.

    For each game day in the test season:
    1. Train all N blend models on data before that day (retrained periodically)
    2. Predict P(hit|PA) with each model, average for blend ranking
    3. Aggregate to game-level P(>=1 hit) per batter
    4. Save top-N batters with blend p_game_hit and actual_hit

    Args:
        df: Feature-enriched PA DataFrame.
        test_season: Season to evaluate on.
        retrain_every: Retrain models every N days.
        top_n: Number of top-ranked batters to save per day.
        blend_configs: List of (name, cols) or (name, cols, extra_params) tuples.
            Defaults to BLEND_CONFIGS. 3-tuple allows per-model objective overrides.
        lgb_params: LightGBM training parameters. Defaults to LGB_PARAMS.

    Returns DataFrame with PROFILE_COLUMNS.
    """
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from bts.features.compute import FEATURE_COLS, STATCAST_COLS, TRAIN_START_YEAR
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS

    if blend_configs is None:
        blend_configs = BLEND_CONFIGS
    if lgb_params is None:
        lgb_params = LGB_PARAMS

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    season_data = df[df["season"] == test_season]
    test_start = season_data["date"].min()
    train_pool = df[(df["date"] < test_start) & (df["season"] >= TRAIN_START_YEAR)].copy()
    test_data = season_data.copy()
    test_dates = sorted(test_data["date"].unique())

    print(f"Blend walk-forward: {len(test_dates)} test days, "
          f"train pool: {len(train_pool):,} PAs, "
          f"{len(blend_configs)} models", file=sys.stderr)

    all_profiles = []
    blend = None

    for i, day in enumerate(test_dates):
        day_data = test_data[test_data["date"] == day].copy()

        # Retrain periodically
        if blend is None or (i % retrain_every == 0):
            available = pd.concat([train_pool, test_data[test_data["date"] < day]])
            train_y = available["is_hit"]

            blend = {}
            for config in blend_configs:
                if len(config) == 2:
                    name, cols = config
                    extra_params = {}
                else:
                    name, cols, extra_params = config

                merged_params = {**lgb_params, **extra_params}
                train_X = available[cols]
                mask = train_X.notna().any(axis=1)
                model = lgb.LGBMClassifier(**merged_params, random_state=42)
                model.fit(train_X[mask], train_y[mask])
                blend[name] = (model, cols)

            if (i + 1) % 30 == 0 or i == 0:
                print(f"  Day {i+1}/{len(test_dates)} ({pd.Timestamp(day).date()}) "
                      f"— retrained on {len(available):,} PAs", file=sys.stderr)

        # Predict with all blend models
        blend_pa_scores = {}
        for name, (model, cols) in blend.items():
            pred_X = day_data[cols]
            valid = pred_X.notna().any(axis=1)
            probs = pd.Series(np.nan, index=day_data.index)
            if valid.any():
                probs[valid] = model.predict_proba(pred_X[valid])[:, 1]
            blend_pa_scores[name] = probs

        # Average PA-level predictions across models
        pa_blend = pd.DataFrame(blend_pa_scores).mean(axis=1)
        day_data["p_hit_blend"] = pa_blend

        # Aggregate to game level: P(>=1 hit) = 1 - prod(1 - P(hit|PA))
        game_preds = day_data.groupby(["batter_id", "game_pk"]).agg(
            p_game_hit=("p_hit_blend", lambda x: 1 - np.prod(1 - x.values)),
            actual_hit=("is_hit", "max"),
            n_pas=("is_hit", "count"),
        ).reset_index()

        # Rank and take top N
        game_preds = game_preds.nlargest(top_n, "p_game_hit").reset_index(drop=True)
        game_preds["rank"] = range(1, len(game_preds) + 1)
        game_preds["date"] = pd.Timestamp(day).date()

        all_profiles.append(game_preds[PROFILE_COLUMNS])

    result = pd.concat(all_profiles, ignore_index=True)
    print(f"  Done: {len(result)} profile rows ({len(test_dates)} days × top-{top_n})", file=sys.stderr)
    return result


def save_profiles(df: "pd.DataFrame", season: int, output_dir: "Path") -> "Path":
    """Save daily profiles to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"backtest_{season}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)", file=sys.stderr)
    return path


def run_backtest(
    data_dir: str = "data/processed",
    output_dir: str = "data/simulation",
    seasons: list[int] | None = None,
    retrain_every: int = 7,
) -> None:
    """Run blend backtest for specified seasons and save profiles.

    Loads all PA parquets, computes features once, then runs blend
    walk-forward for each test season.
    """
    import pandas as pd
    from bts.features.compute import compute_all_features

    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]

    proc = Path(data_dir)
    out = Path(output_dir)

    # Load all data and compute features once
    print("Loading PA data...", file=sys.stderr)
    dfs = []
    for parquet in sorted(proc.glob("pa_*.parquet")):
        dfs.append(pd.read_parquet(parquet))
    if not dfs:
        raise RuntimeError("No parquet files found. Run 'bts data build' first.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Computing features on {len(df):,} PAs...", file=sys.stderr)
    df = compute_all_features(df)

    for season in seasons:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Season {season}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        profiles_df = blend_walk_forward(df, season, retrain_every=retrain_every)
        save_profiles(profiles_df, season, out)
