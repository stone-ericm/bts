"""Scorecard computation engine for BTS model validation.

Reads backtest profiles and computes precision@K, miss analysis,
calibration, and streak simulation metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        # Skip K values where no day has enough candidates (profiles have top-10 only)
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
    ) if len(hit_dates) > 0 else None

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


# ---------------------------------------------------------------------------
# Full scorecard aggregation
# ---------------------------------------------------------------------------

def compute_full_scorecard(
    profiles_df: pd.DataFrame,
    mc_trials: int = 10_000,
    season_length: int = 180,
) -> dict:
    """Compute the complete BTS validation scorecard.

    Calls all 4 scoring functions and adds metadata, exact P(57) via
    absorbing Markov chain, and MDP-optimal P(57).

    Args:
        profiles_df: DataFrame with columns [date, rank, batter_id,
            p_game_hit, actual_hit, n_pas]. May have a 'season' column;
            if absent, season is inferred from the year of the date column.
        mc_trials: Monte Carlo trials for streak simulation.
        season_length: Days per simulated season.

    Returns:
        Dict suitable for JSON serialization (see save_scorecard for
        numpy type handling).
    """
    from bts.simulate.quality_bins import compute_bins
    from bts.simulate.exact import exact_p57
    from bts.simulate.mdp import solve_mdp
    from bts.simulate.strategies import ALL_STRATEGIES

    df = profiles_df.copy()

    # Ensure season column exists (infer from date year if absent)
    if "season" not in df.columns:
        df["season"] = pd.to_datetime(df["date"]).dt.year

    n_days = int(df["date"].nunique())
    n_rows = len(df)

    # Core metrics
    precision = compute_precision_at_k(df)
    precision_by_season = compute_precision_at_k(df, by_season=True)
    miss_analysis = compute_miss_analysis(df)
    calibration = compute_calibration(df)
    streak_metrics = compute_streak_metrics(
        df, n_trials=mc_trials, season_length=season_length
    )

    from bts.validate.proper_scoring import compute_proper_scoring
    proper_scoring = compute_proper_scoring(df)

    # Summarize P@1 per season for quick comparison
    p_at_1_by_season: dict[int, float] = {}
    for season_key, season_precision in precision_by_season.items():
        if 1 in season_precision:
            p_at_1_by_season[int(season_key)] = season_precision[1]

    # Exact P(57) via absorbing Markov chain
    p_57_exact: float | None = None
    p_57_mdp: float | None = None

    bins = None
    try:
        bins = compute_bins(df)
    except Exception:
        pass

    if bins is not None:
        try:
            strategy = ALL_STRATEGIES["combined"]
            p_57_exact = exact_p57(strategy, bins, season_length=season_length)
        except Exception:
            pass
        try:
            mdp_sol = solve_mdp(bins, season_length=season_length)
            p_57_mdp = mdp_sol.optimal_p57
        except Exception:
            pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_days": n_days,
        "n_rows": n_rows,
        "mc_trials": mc_trials,
        "season_length": season_length,
        "precision": precision,
        "precision_by_season": precision_by_season,
        "p_at_1_by_season": p_at_1_by_season,
        "miss_analysis": miss_analysis,
        "calibration": calibration,
        "streak_metrics": streak_metrics,
        "p_57_exact": p_57_exact,
        "p_57_mdp": p_57_mdp,
        "proper_scoring": proper_scoring,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_scorecard(scorecard: dict, path: str | Path) -> Path:
    """Serialize scorecard to JSON and write to disk.

    Handles numpy scalar types (np.integer, np.floating) and arrays.
    Creates parent directories as needed.

    Args:
        scorecard: Dict returned by compute_full_scorecard (or any dict
            containing numpy types).
        path: Output file path (string or Path).

    Returns:
        Path object pointing to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard, indent=2, default=_json_default))
    return out


# ---------------------------------------------------------------------------
# Diff / comparison
# ---------------------------------------------------------------------------

def _diff_numeric(baseline_val: Any, variant_val: Any) -> dict | None:
    """Return {baseline, variant, delta} if both values are numeric, else None."""
    if baseline_val is None or variant_val is None:
        return None
    try:
        b = float(baseline_val)
        v = float(variant_val)
        return {"baseline": b, "variant": v, "delta": v - b}
    except (TypeError, ValueError):
        return None


def diff_scorecards(baseline: dict, variant: dict) -> dict:
    """Compute numeric deltas between two scorecards.

    For each comparable numeric field, produces a dict with keys
    {baseline, variant, delta}.

    Sections diffed:
    - precision (K → float)
    - p_at_1_by_season (season → float)
    - miss_analysis (field → scalar)
    - streak_metrics (field → scalar)
    - p_57_exact (scalar)
    - p_57_mdp (scalar)
    - proper_scoring (flat dotted keys: {bucket}.{field} for log_loss,
      brier, decomposition.{reliability,resolution,uncertainty}, and
      top_bin.{mean_p,mean_y,gap}). Tabular reliability_table, top_bin
      counts/intervals, and metadata are omitted.

    Args:
        baseline: Reference scorecard dict.
        variant: New scorecard dict to compare against.

    Returns:
        Dict of diffs (fields with no numeric counterpart are omitted).
    """
    result: dict = {}

    # precision: {k: float}
    b_prec = baseline.get("precision", {})
    v_prec = variant.get("precision", {})
    if b_prec and v_prec:
        prec_diff: dict = {}
        for k in b_prec:
            if k in v_prec:
                d = _diff_numeric(b_prec[k], v_prec[k])
                if d is not None:
                    prec_diff[k] = d
        if prec_diff:
            result["precision"] = prec_diff

    # p_at_1_by_season: {season: float}
    # JSON serialization converts int → str, so look up with both types and
    # preserve the original baseline key type in the output.
    b_p1s = baseline.get("p_at_1_by_season", {})
    v_p1s = variant.get("p_at_1_by_season", {})
    if b_p1s and v_p1s:
        # Build str→value lookup for variant to handle int/str mismatches
        v_p1s_by_str = {str(k): v for k, v in v_p1s.items()}
        p1s_diff: dict = {}
        for season, b_val in b_p1s.items():
            v_val = v_p1s.get(season)
            if v_val is None:
                v_val = v_p1s_by_str.get(str(season))
            if v_val is not None:
                d = _diff_numeric(b_val, v_val)
                if d is not None:
                    p1s_diff[season] = d
        if p1s_diff:
            result["p_at_1_by_season"] = p1s_diff

    # miss_analysis: flat dict of scalars
    b_ma = baseline.get("miss_analysis", {})
    v_ma = variant.get("miss_analysis", {})
    if b_ma and v_ma:
        ma_diff: dict = {}
        for field in b_ma:
            if field in v_ma:
                d = _diff_numeric(b_ma[field], v_ma[field])
                if d is not None:
                    ma_diff[field] = d
        if ma_diff:
            result["miss_analysis"] = ma_diff

    # streak_metrics: flat dict of scalars
    b_sm = baseline.get("streak_metrics", {})
    v_sm = variant.get("streak_metrics", {})
    if b_sm and v_sm:
        sm_diff: dict = {}
        for field in b_sm:
            if field in v_sm:
                d = _diff_numeric(b_sm[field], v_sm[field])
                if d is not None:
                    sm_diff[field] = d
        if sm_diff:
            result["streak_metrics"] = sm_diff

    # Top-level scalar fields
    for field in ("p_57_exact", "p_57_mdp"):
        d = _diff_numeric(baseline.get(field), variant.get(field))
        if d is not None:
            result[field] = d

    # proper_scoring: flatten nested scalars under {bucket}.{field} keys.
    # Skipped: reliability_table (tabular), top_bin.n / top_bin.ci_lo / top_bin.ci_hi
    # (counts and intervals, not performance scalars). metadata is not numeric.
    b_ps = baseline.get("proper_scoring", {})
    v_ps = variant.get("proper_scoring", {})
    if b_ps and v_ps:
        ps_diff: dict = {}
        for bucket in ("all_top10", "rank1"):
            b_bucket = b_ps.get(bucket, {})
            v_bucket = v_ps.get(bucket, {})
            if not b_bucket or not v_bucket:
                continue
            # bucket-level scalars
            for field in ("log_loss", "brier"):
                d = _diff_numeric(b_bucket.get(field), v_bucket.get(field))
                if d is not None:
                    ps_diff[f"{bucket}.{field}"] = d
            # decomposition scalars
            b_dec = b_bucket.get("decomposition", {})
            v_dec = v_bucket.get("decomposition", {})
            for field in ("reliability", "resolution", "uncertainty"):
                d = _diff_numeric(b_dec.get(field), v_dec.get(field))
                if d is not None:
                    ps_diff[f"{bucket}.decomposition.{field}"] = d
            # top_bin scalars
            b_tb = b_bucket.get("top_bin", {})
            v_tb = v_bucket.get("top_bin", {})
            for field in ("mean_p", "mean_y", "gap"):
                d = _diff_numeric(b_tb.get(field), v_tb.get(field))
                if d is not None:
                    ps_diff[f"{bucket}.top_bin.{field}"] = d
        if ps_diff:
            result["proper_scoring"] = ps_diff

    return result
