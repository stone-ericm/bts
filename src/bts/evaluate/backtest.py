"""Walk-forward backtesting for PA-level hit prediction."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from bts.features.compute import FEATURE_COLS, TRAIN_START_YEAR
from bts.model.predict import LGB_PARAMS


def walk_forward_evaluate(
    df: pd.DataFrame,
    test_season: int = 2025,
    retrain_every: int = 7,
) -> dict:
    """Run walk-forward evaluation on a feature-enriched PA DataFrame.

    For each game day in the test season:
    1. Train on all data before that day
    2. Predict P(hit) for each PA on that day
    3. Aggregate to game-level P(>=1 hit) per batter
    4. Rank batters and compute Precision@K

    Args:
        df: PA DataFrame with all features computed (from compute_all_features).
        test_season: Year to use as test set.
        retrain_every: Retrain model every N days (speeds up walk-forward).

    Returns:
        Dict with P@K metrics and daily predictions.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Split — train on TRAIN_START_YEAR onward (older data hurts)
    test_start = df[df["season"] == test_season]["date"].min()
    train_pool = df[(df["date"] < test_start) & (df["season"] >= TRAIN_START_YEAR)].copy()
    test_data = df[df["date"] >= test_start].copy()

    test_dates = sorted(test_data["date"].unique())
    print(f"Walk-forward: {len(test_dates)} test days, train pool: {len(train_pool):,} PAs")

    # Collect daily predictions
    all_preds = []
    model = None
    last_train_date = None

    for i, day in enumerate(test_dates):
        day_data = test_data[test_data["date"] == day].copy()

        # Retrain periodically
        if model is None or (i % retrain_every == 0):
            available = pd.concat([train_pool, test_data[test_data["date"] < day]])
            train_X = available[FEATURE_COLS].copy()
            train_y = available["is_hit"]

            # Drop rows with all-NaN features
            valid_mask = train_X.notna().any(axis=1)
            train_X = train_X[valid_mask]
            train_y = train_y[valid_mask]

            model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42)
            model.fit(train_X, train_y)
            last_train_date = day

        # Predict
        pred_X = day_data[FEATURE_COLS].copy()
        day_data["p_hit"] = model.predict_proba(pred_X)[:, 1]

        # Aggregate to game-level: P(>=1 hit) = 1 - prod(1 - P(hit|PA))
        game_preds = day_data.groupby(["batter_id", "game_pk"]).agg(
            p_game_hit=("p_hit", lambda x: 1 - np.prod(1 - x.values)),
            actual_hit=("is_hit", "max"),
            n_pas=("is_hit", "count"),
        ).reset_index()
        game_preds["date"] = day

        all_preds.append(game_preds)

        if (i + 1) % 30 == 0:
            print(f"  Day {i+1}/{len(test_dates)} ({pd.Timestamp(day).date()})")

    # Combine all predictions
    preds = pd.concat(all_preds, ignore_index=True)

    # Compute Precision@K
    metrics = {}
    for k in [1, 5, 10, 50, 100, 500]:
        if k > len(preds):
            continue

        # For each day, take top-K predictions
        daily_precision = []
        for day, day_preds in preds.groupby("date"):
            top_k = day_preds.nlargest(min(k, len(day_preds)), "p_game_hit")
            precision = top_k["actual_hit"].mean()
            daily_precision.append(precision)

        metrics[f"P@{k}"] = np.mean(daily_precision)

    # Overall stats
    metrics["total_days"] = len(test_dates)
    metrics["total_batter_games"] = len(preds)
    metrics["mean_p_hit"] = preds["p_game_hit"].mean()
    metrics["actual_hit_rate"] = preds["actual_hit"].mean()

    # Feature importance
    if model is not None:
        importance = dict(zip(FEATURE_COLS, model.feature_importances_))
        metrics["feature_importance"] = dict(
            sorted(importance.items(), key=lambda x: -x[1])
        )

    return metrics, preds
