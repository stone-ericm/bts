"""Shared evaluation harness for architecture alignment experiments.

All experiment phases import from this module for consistent walk-forward
backtesting, metric computation, and result comparison.

NOTE: game_time is not available in the PA parquet files (no column exists).
      It is set to None throughout and carried as a null column for Phase 3
      compatibility. Phase 3 (densest bucket) will need an alternative source
      for game_time, or can filter by bucket rank rather than actual game time.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(data_dir: str = "data/processed") -> pd.DataFrame:
    """Load all PA parquets and compute features.

    Loads pa_*.parquet from data_dir, concatenates them, then calls
    compute_all_features. Returns the feature-enriched DataFrame.
    """
    from bts.features.compute import compute_all_features

    proc = Path(data_dir)
    parquets = sorted(proc.glob("pa_*.parquet"))
    if not parquets:
        raise RuntimeError(f"No pa_*.parquet files found in {data_dir}. Run 'bts data build' first.")

    dfs = [pd.read_parquet(p) for p in parquets]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} PAs from {len(parquets)} parquets", file=sys.stderr)

    df = compute_all_features(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def walk_forward_backtest(
    df: pd.DataFrame,
    test_seasons: list[int],
    blend_configs: list[tuple],
    lgb_params: dict,
    game_level: bool = False,
    retrain_every: int = 7,
    top_n: int = 10,
) -> pd.DataFrame:
    """Walk-forward blend evaluation across test seasons.

    Returns a profile DataFrame with columns:
        [date, rank, batter_id, p_game_hit, actual_hit, game_time]

    Args:
        df: Feature-enriched PA DataFrame (all seasons).
        test_seasons: List of seasons to test (e.g. [2024, 2025]).
        blend_configs: List of (name, feature_cols) tuples (BLEND_CONFIGS).
        lgb_params: LightGBM hyperparameters dict (LGB_PARAMS).
        game_level: If True, aggregate to game level before training.
            If False (PA-level, live-aligned), train on PAs and aggregate
            per model THEN average — matching the live pipeline order.
        retrain_every: Retrain blend every N days in the test season.
        top_n: Number of top batters to include per day in profiles.

    Returns:
        DataFrame of daily top-N profiles across all test seasons.
    """
    import lightgbm as lgb
    from bts.features.compute import TRAIN_START_YEAR

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # game_time is not in the parquet — set to None for Phase 3 compatibility
    has_game_time = "game_time" in df.columns

    if game_level:
        df = _aggregate_to_game_level(df)

    all_profiles = []

    for test_season in test_seasons:
        season_profiles = _run_season(
            df=df,
            test_season=test_season,
            blend_configs=blend_configs,
            lgb_params=lgb_params,
            game_level=game_level,
            retrain_every=retrain_every,
            top_n=top_n,
            train_start_year=TRAIN_START_YEAR,
            has_game_time=has_game_time,
        )
        all_profiles.append(season_profiles)

    result = pd.concat(all_profiles, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


def _aggregate_to_game_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PA-level DataFrame to one row per (batter_id, game_pk, date).

    Target = max(is_hit). Features: first value per game. Used for game_level=True.
    """
    from bts.features.compute import FEATURE_COLS, STATCAST_COLS

    feature_cols = FEATURE_COLS + STATCAST_COLS
    available_feature_cols = [c for c in feature_cols if c in df.columns]

    agg_dict = {"is_hit": "max", "season": "first", "lineup_position": "first"}
    for col in available_feature_cols:
        agg_dict[col] = "first"
    if "game_time" in df.columns:
        agg_dict["game_time"] = "first"

    game_df = (
        df.groupby(["batter_id", "game_pk", "date"])
        .agg(agg_dict)
        .reset_index()
    )
    return game_df


def _run_season(
    df: pd.DataFrame,
    test_season: int,
    blend_configs: list[tuple],
    lgb_params: dict,
    game_level: bool,
    retrain_every: int,
    top_n: int,
    train_start_year: int,
    has_game_time: bool,
) -> pd.DataFrame:
    """Walk-forward evaluation for a single test season."""
    import lightgbm as lgb

    season_data = df[df["season"] == test_season]
    if len(season_data) == 0:
        raise ValueError(f"No data found for season {test_season}")

    test_start = season_data["date"].min()
    train_pool = df[
        (df["date"] < test_start) & (df["season"] >= train_start_year)
    ].copy()
    test_data = season_data.copy()
    test_dates = sorted(test_data["date"].unique())

    print(
        f"Season {test_season}: {len(test_dates)} test days, "
        f"train pool: {len(train_pool):,} rows, "
        f"{len(blend_configs)} models, "
        f"mode={'game' if game_level else 'PA-live-aligned'}",
        file=sys.stderr,
    )

    all_profiles = []
    blend = None

    for i, day in enumerate(test_dates):
        day_data = test_data[test_data["date"] == day].copy()

        # Retrain periodically
        if blend is None or (i % retrain_every == 0):
            available = pd.concat(
                [train_pool, test_data[test_data["date"] < day]]
            )
            train_y = available["is_hit"]

            blend = {}
            for name, cols in blend_configs:
                avail_cols = [c for c in cols if c in available.columns]
                train_X = available[avail_cols]
                mask = train_X.notna().any(axis=1)
                model = lgb.LGBMClassifier(**lgb_params, random_state=42)
                model.fit(train_X[mask], train_y[mask])
                blend[name] = (model, avail_cols)

            if (i + 1) % 30 == 0 or i == 0:
                print(
                    f"  Day {i+1}/{len(test_dates)} ({pd.Timestamp(day).date()}) "
                    f"— retrained on {len(available):,} rows",
                    file=sys.stderr,
                )

        if game_level:
            profiles_day = _predict_game_level(
                day_data=day_data,
                blend=blend,
                top_n=top_n,
                day=day,
                has_game_time=has_game_time,
            )
        else:
            profiles_day = _predict_pa_live_aligned(
                day_data=day_data,
                blend=blend,
                top_n=top_n,
                day=day,
                has_game_time=has_game_time,
            )

        all_profiles.append(profiles_day)

    result = pd.concat(all_profiles, ignore_index=True)
    print(
        f"  Done season {test_season}: {len(result)} profile rows "
        f"({len(test_dates)} days × top-{top_n})",
        file=sys.stderr,
    )
    return result


def _predict_game_level(
    day_data: pd.DataFrame,
    blend: dict,
    top_n: int,
    day,
    has_game_time: bool,
) -> pd.DataFrame:
    """Game-level prediction: each model predicts P(game_hit) directly.

    Average game-level predictions across models.
    """
    # Each model gets one row per (batter_id, game_pk) — already aggregated
    model_game_scores: dict[int, list[float]] = {}

    for name, (model, cols) in blend.items():
        avail_cols = [c for c in cols if c in day_data.columns]
        pred_X = day_data[avail_cols]
        valid = pred_X.notna().any(axis=1)
        p_game = pd.Series(np.nan, index=day_data.index)
        if valid.any():
            p_game[valid] = model.predict_proba(pred_X[valid])[:, 1]

        for idx, val in p_game.items():
            if pd.notna(val):
                bid = day_data.at[idx, "batter_id"]
                if bid not in model_game_scores:
                    model_game_scores[bid] = []
                model_game_scores[bid].append(val)

    # Average across models per batter
    blend_avg = {bid: np.mean(scores) for bid, scores in model_game_scores.items()}

    game_preds = day_data.copy()
    game_preds["p_game_hit"] = game_preds["batter_id"].map(blend_avg)
    game_preds["actual_hit"] = game_preds["is_hit"].astype(int)

    if has_game_time and "game_time" in game_preds.columns:
        game_preds = game_preds.groupby(["batter_id", "game_pk"]).agg(
            p_game_hit=("p_game_hit", "first"),
            actual_hit=("actual_hit", "max"),
            game_time=("game_time", "first"),
        ).reset_index()
    else:
        game_preds = game_preds.groupby(["batter_id", "game_pk"]).agg(
            p_game_hit=("p_game_hit", "first"),
            actual_hit=("actual_hit", "max"),
        ).reset_index()
        game_preds["game_time"] = None

    game_preds = game_preds.dropna(subset=["p_game_hit"])
    game_preds = game_preds.nlargest(top_n, "p_game_hit").reset_index(drop=True)
    game_preds["rank"] = range(1, len(game_preds) + 1)
    game_preds["date"] = pd.Timestamp(day).date()

    return game_preds[["date", "rank", "batter_id", "p_game_hit", "actual_hit", "game_time"]]


def _predict_pa_live_aligned(
    day_data: pd.DataFrame,
    blend: dict,
    top_n: int,
    day,
    has_game_time: bool,
) -> pd.DataFrame:
    """PA-level prediction, live-aligned aggregation order.

    KEY: Each model aggregates to game level via 1 - prod(1-p) independently,
    THEN average game-level predictions across models. This matches the live
    pipeline's averaging order (vs. old backtest which averaged PAs first).
    """
    model_game_scores: dict[tuple, list[float]] = {}  # (batter_id, game_pk) → [per-model p_game]

    for name, (model, cols) in blend.items():
        avail_cols = [c for c in cols if c in day_data.columns]
        pred_X = day_data[avail_cols]
        valid = pred_X.notna().any(axis=1)
        probs = pd.Series(np.nan, index=day_data.index)
        if valid.any():
            probs[valid] = model.predict_proba(pred_X[valid])[:, 1]

        day_data_copy = day_data.copy()
        day_data_copy["_p_hit"] = probs

        # Aggregate to game level: 1 - prod(1 - p) per model
        game_grouped = day_data_copy.groupby(["batter_id", "game_pk"])
        for (bid, gpk), grp in game_grouped:
            p_vals = grp["_p_hit"].dropna().values
            if len(p_vals) == 0:
                continue
            p_game = 1.0 - np.prod(1.0 - p_vals)
            key = (bid, gpk)
            if key not in model_game_scores:
                model_game_scores[key] = []
            model_game_scores[key].append(p_game)

    # Average across models per (batter_id, game_pk)
    blend_avg = {key: np.mean(scores) for key, scores in model_game_scores.items()}

    # Build result DataFrame
    actuals = (
        day_data.groupby(["batter_id", "game_pk"])["is_hit"]
        .max()
        .reset_index()
        .rename(columns={"is_hit": "actual_hit"})
    )

    if has_game_time and "game_time" in day_data.columns:
        gt = day_data.groupby(["batter_id", "game_pk"])["game_time"].first().reset_index()
        actuals = actuals.merge(gt, on=["batter_id", "game_pk"], how="left")
    else:
        actuals["game_time"] = None

    actuals["p_game_hit"] = actuals.apply(
        lambda r: blend_avg.get((r["batter_id"], r["game_pk"]), np.nan), axis=1
    )
    actuals = actuals.dropna(subset=["p_game_hit"])
    actuals = actuals.nlargest(top_n, "p_game_hit").reset_index(drop=True)
    actuals["rank"] = range(1, len(actuals) + 1)
    actuals["date"] = pd.Timestamp(day).date()

    return actuals[["date", "rank", "batter_id", "p_game_hit", "actual_hit", "game_time"]]


def compute_metrics(profiles: pd.DataFrame, season_length: int = 180) -> dict:
    """Compute evaluation metrics from backtest profiles.

    Args:
        profiles: DataFrame with [date, rank, batter_id, p_game_hit, actual_hit, game_time].
        season_length: Season length for MDP solver (default 180).

    Returns:
        Dict with keys:
            p_at_1: dict of {season: P@1, "avg": overall_avg}
            mdp_p57: float P(reaching streak 57) from MDP solver
            early_bins: QualityBins for non-September
            late_bins: QualityBins for September only (if >= 50 rows)
            longest_replay: int longest streak from rank-1 actual hits
            mean_p_hit: float mean predicted P(game_hit) at rank 1
    """
    from bts.simulate.quality_bins import compute_bins
    from bts.simulate.mdp import solve_mdp

    profiles = profiles.copy()
    profiles["date"] = pd.to_datetime(profiles["date"])

    # --- P@1 by season and overall ---
    r1 = profiles[profiles["rank"] == 1].copy()
    r1["season"] = r1["date"].dt.year

    p_at_1_by_season = r1.groupby("season")["actual_hit"].mean().to_dict()
    p_at_1_avg = r1["actual_hit"].mean()
    p_at_1 = {**{int(k): float(v) for k, v in p_at_1_by_season.items()}, "avg": float(p_at_1_avg)}

    # --- Quality bins (early: ALL data, matching deployed MDP) ---
    early_bins = compute_bins(profiles)

    # --- Late bins (September only, min 50 rows) ---
    late_mask = profiles["date"].dt.month == 9
    late_profiles = profiles[late_mask]
    if len(late_profiles) >= 50:
        late_bins = compute_bins(late_profiles)
    else:
        late_bins = None

    # --- MDP solve ---
    mdp_solution = solve_mdp(
        bins=early_bins,
        season_length=season_length,
        late_bins=late_bins,
        late_phase_days=30,
    )
    mdp_p57 = mdp_solution.optimal_p57

    # --- Longest replay streak from rank-1 actual hits ---
    r1_sorted = r1.sort_values("date")
    longest_replay = _longest_streak(r1_sorted["actual_hit"].astype(int).tolist())

    # --- Mean predicted P(hit) at rank 1 ---
    mean_p_hit = float(r1["p_game_hit"].mean())

    return {
        "p_at_1": p_at_1,
        "mdp_p57": mdp_p57,
        "early_bins": early_bins,
        "late_bins": late_bins,
        "longest_replay": longest_replay,
        "mean_p_hit": mean_p_hit,
    }


def _longest_streak(hits: list[int]) -> int:
    """Compute the longest consecutive run of 1s in a list."""
    best = 0
    current = 0
    for h in hits:
        if h == 1:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def print_comparison(results: dict, label: str = "") -> None:
    """Print formatted comparison table of variants.

    Args:
        results: Dict of {variant_name: metrics_dict} where metrics_dict
            comes from compute_metrics().
        label: Optional label to print above the table.
    """
    if not results:
        print("No results to compare.")
        return

    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

    names = list(results.keys())
    first_name = names[0]
    first = results[first_name]

    # Collect all seasons across all variants
    all_seasons = sorted(set(
        k for m in results.values()
        for k in m["p_at_1"].keys()
        if isinstance(k, int)
    ))

    # Header
    col_w = 14
    header_parts = ["Variant".ljust(20)]
    for s in all_seasons:
        header_parts.append(f"P@1 {s}".center(col_w))
    header_parts.append("Avg P@1".center(col_w))
    header_parts.append("MDP P(57)".center(col_w))
    header_parts.append("Replay".center(col_w))
    header_parts.append("Mean P".center(col_w))
    print("\n" + "  ".join(header_parts))
    print("-" * (20 + (len(all_seasons) + 4) * (col_w + 2)))

    for i, (name, metrics) in enumerate(results.items()):
        p_at_1 = metrics["p_at_1"]
        is_baseline = i == 0

        row = [name[:20].ljust(20)]

        for s in all_seasons:
            val = p_at_1.get(s, float("nan"))
            base_val = first["p_at_1"].get(s, float("nan"))
            if is_baseline or np.isnan(val) or np.isnan(base_val):
                row.append(f"{val:.3f}".center(col_w))
            else:
                delta = val - base_val
                sign = "+" if delta >= 0 else ""
                row.append(f"{val:.3f} ({sign}{delta:.3f})".center(col_w))

        # Avg P@1
        avg_val = p_at_1.get("avg", float("nan"))
        base_avg = first["p_at_1"].get("avg", float("nan"))
        if is_baseline or np.isnan(avg_val) or np.isnan(base_avg):
            row.append(f"{avg_val:.4f}".center(col_w))
        else:
            delta = avg_val - base_avg
            sign = "+" if delta >= 0 else ""
            row.append(f"{avg_val:.4f}({sign}{delta:.4f})".center(col_w))

        # MDP P(57)
        mdp_val = metrics["mdp_p57"]
        base_mdp = first["mdp_p57"]
        if is_baseline:
            row.append(f"{mdp_val:.4f}".center(col_w))
        else:
            delta = mdp_val - base_mdp
            sign = "+" if delta >= 0 else ""
            row.append(f"{mdp_val:.4f}({sign}{delta:.4f})".center(col_w))

        # Replay streak
        row.append(str(metrics["longest_replay"]).center(col_w))

        # Mean P
        row.append(f"{metrics['mean_p_hit']:.4f}".center(col_w))

        print("  ".join(row))

    print()
