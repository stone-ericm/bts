"""Scorecard computation engine for BTS model validation.

Reads backtest profiles and computes precision@K, miss analysis,
calibration, and streak simulation metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_precision_at_k(
    profiles_df: pd.DataFrame,
    k_values: list[int] | None = None,
    by_season: bool = False,
) -> dict:
    """Compute Precision@K metrics from backtest profiles.

    For each K value, takes the top-K ranked batters per day, computes
    their hit rate, and averages across days.

    Args:
        profiles_df: DataFrame with columns [date, rank, batter_id,
            p_game_hit, actual_hit, n_pas].
        k_values: K cutoffs to evaluate. Defaults to
            [1, 5, 10, 25, 50, 100, 250, 500].
        by_season: If True and a 'season' column exists, returns
            {season: {k: precision}}. Otherwise returns {k: precision}.

    Returns:
        Dict mapping K → precision, or season → {K → precision} if
        by_season=True and 'season' column is present.
    """
    if k_values is None:
        k_values = [1, 5, 10, 25, 50, 100, 250, 500]

    if by_season and "season" in profiles_df.columns:
        return {
            season: _precision_at_k_flat(group, k_values)
            for season, group in profiles_df.groupby("season")
        }
    return _precision_at_k_flat(profiles_df, k_values)


def _precision_at_k_flat(df: pd.DataFrame, k_values: list[int]) -> dict[int, float]:
    """Compute flat {k: precision} from a single-season (or full) DataFrame."""
    result: dict[int, float] = {}
    for k in k_values:
        top_k = df[df["rank"] <= k]
        if top_k.empty:
            continue
        # Skip if all days have fewer than k rows (except K=1, always allowed)
        if k > 1 and (top_k.groupby("date").size() < k).all():
            continue
        daily_precision = top_k.groupby("date")["actual_hit"].mean()
        result[k] = float(daily_precision.mean())
    return result


def compute_miss_analysis(profiles_df: pd.DataFrame) -> dict:
    """Analyse miss patterns for rank-1 picks.

    Args:
        profiles_df: Backtest profiles DataFrame.

    Returns:
        Dict with keys:
        - n_miss_days: int — number of days rank-1 missed
        - rank_2_hit_rate_on_miss: float | None — fraction of miss days
          where rank-2 got a hit (None if n_miss_days == 0)
        - mean_p_hit_on_miss: float — mean predicted P(hit) on miss days
        - mean_p_hit_on_hit: float — mean predicted P(hit) on hit days
    """
    rank1 = profiles_df[profiles_df["rank"] == 1].copy()
    miss_dates = rank1[rank1["actual_hit"] == 0]["date"]
    hit_dates = rank1[rank1["actual_hit"] == 1]["date"]

    n_miss_days = len(miss_dates)

    if n_miss_days == 0:
        rank_2_hit_rate = None
    else:
        rank2 = profiles_df[profiles_df["rank"] == 2]
        rank2_on_miss = rank2[rank2["date"].isin(miss_dates)]
        if rank2_on_miss.empty:
            rank_2_hit_rate = None
        else:
            rank_2_hit_rate = float(rank2_on_miss["actual_hit"].mean())

    mean_p_hit_on_miss = float(
        rank1[rank1["date"].isin(miss_dates)]["p_game_hit"].mean()
    ) if n_miss_days > 0 else None

    mean_p_hit_on_hit = float(
        rank1[rank1["date"].isin(hit_dates)]["p_game_hit"].mean()
    ) if len(hit_dates) > 0 else float("nan")

    return {
        "n_miss_days": n_miss_days,
        "rank_2_hit_rate_on_miss": rank_2_hit_rate,
        "mean_p_hit_on_miss": mean_p_hit_on_miss,
        "mean_p_hit_on_hit": mean_p_hit_on_hit,
    }


def compute_calibration(
    profiles_df: pd.DataFrame,
    n_deciles: int = 10,
) -> list[tuple[float, float, int]]:
    """Compute calibration curve for rank 1-10 predictions.

    Buckets predicted probabilities into quantile bins and compares
    predicted vs actual hit rates per bucket.

    Args:
        profiles_df: Backtest profiles DataFrame.
        n_deciles: Number of quantile bins. Uses pd.qcut with
            duplicates='drop'.

    Returns:
        List of (predicted_mean, actual_mean, count) tuples, one per
        non-empty bin, sorted by predicted_mean ascending.
    """
    top10 = profiles_df[profiles_df["rank"] <= 10].copy()

    bins = pd.qcut(top10["p_game_hit"], q=n_deciles, duplicates="drop")
    grouped = top10.groupby(bins, observed=True)

    result = []
    for _, group in grouped:
        if group.empty:
            continue
        pred_mean = float(group["p_game_hit"].mean())
        actual_mean = float(group["actual_hit"].mean())
        count = len(group)
        result.append((pred_mean, actual_mean, count))

    return sorted(result, key=lambda x: x[0])


def compute_streak_metrics(
    profiles_df: pd.DataFrame,
    n_trials: int = 10_000,
    season_length: int = 180,
) -> dict:
    """Compute streak simulation metrics using the combined strategy.

    Runs Monte Carlo simulation and a single deterministic replay of the
    profiles, returning summary streak statistics and P(57).

    Args:
        profiles_df: Backtest profiles DataFrame.
        n_trials: Number of Monte Carlo trials.
        season_length: Days per simulated season (bootstrap sampling).

    Returns:
        Dict with keys:
        - mean_max_streak: float
        - median_max_streak: int
        - p90_max_streak: int
        - p99_max_streak: int
        - p_57_monte_carlo: float
        - longest_replay_streak: int
    """
    from bts.simulate.monte_carlo import load_profiles, run_monte_carlo, simulate_season
    from bts.simulate.strategies import ALL_STRATEGIES

    strategy = ALL_STRATEGIES["combined"]
    profiles = load_profiles(profiles_df)

    mc_result = run_monte_carlo(
        profiles=profiles,
        strategy=strategy,
        n_trials=n_trials,
        season_length=season_length,
    )

    # Single deterministic replay through all profiles in order
    replay_result = simulate_season(profiles, strategy)

    streaks = np.array(mc_result.max_streaks)

    return {
        "mean_max_streak": float(np.mean(streaks)),
        "median_max_streak": mc_result.median_streak,
        "p90_max_streak": int(np.percentile(streaks, 90)),
        "p99_max_streak": int(np.percentile(streaks, 99)),
        "p_57_monte_carlo": mc_result.p_57,
        "longest_replay_streak": replay_result.max_streak,
    }
