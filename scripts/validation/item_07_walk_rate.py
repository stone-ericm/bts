"""
Item 07: Walk-Rate / BB% Feature Investigation

Tests whether explicit BB% as a feature or hard filter helps beyond the existing
batter_count_tendency_30g proxy.

r/beatthestreak community universally avoids high-walk players (Soto is the poster child).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SIM_DIR = DATA_DIR / "simulation"
SEASONS = [2021, 2022, 2023, 2024, 2025]

# Walk event types in the PA data
WALK_EVENTS = {"walk", "intent_walk"}


# ---------------------------------------------------------------------------
# Step 1: Load PA data and compute rolling 30-game-date BB%
# ---------------------------------------------------------------------------

def load_pa_data() -> pd.DataFrame:
    """Load all PA data (2021-2025) needed for BB% computation.
    Include earlier seasons to warm up rolling windows for 2021 season start."""
    dfs = []
    for season in [2019, 2020] + SEASONS:
        path = PROCESSED_DIR / f"pa_{season}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        df = pd.read_parquet(path, columns=["game_pk", "date", "season", "batter_id",
                                             "event_type", "final_count_balls",
                                             "final_count_strikes"])
        dfs.append(df)
    pa = pd.concat(dfs, ignore_index=True)
    pa["date"] = pd.to_datetime(pa["date"])
    return pa


def compute_rolling_bb_rate(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 30-game-date BB% per batter with shift(1) temporal guard.

    Returns a DataFrame with columns: [batter_id, date, bb_rate_30g]
    One row per (batter, game_date) — the value is available *before* that date.
    """
    pa["is_walk"] = pa["event_type"].isin(WALK_EVENTS).astype(int)

    # Aggregate to date level: walks and PAs per batter per game-date
    date_agg = (
        pa.groupby(["batter_id", "date"])
        .agg(walks=("is_walk", "sum"), n_pa=("is_walk", "count"))
        .reset_index()
        .sort_values(["batter_id", "date"])
    )

    # Rolling 30 game-dates with shift(1) — only prior dates contribute
    def rolling_bb(grp):
        shifted_walks = grp["walks"].shift(1)
        shifted_pa = grp["n_pa"].shift(1)
        roll_walks = shifted_walks.rolling(30, min_periods=10).sum()
        roll_pa = shifted_pa.rolling(30, min_periods=10).sum()
        return roll_walks / roll_pa

    date_agg["bb_rate_30g"] = (
        date_agg.groupby("batter_id", group_keys=False).apply(rolling_bb)
    )

    return date_agg[["batter_id", "date", "bb_rate_30g"]]


# ---------------------------------------------------------------------------
# Step 2: Load backtest profiles
# ---------------------------------------------------------------------------

def load_backtest() -> pd.DataFrame:
    dfs = []
    for season in SEASONS:
        path = SIM_DIR / f"backtest_{season}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        df["season"] = season
        dfs.append(df)
    bt = pd.concat(dfs, ignore_index=True)
    bt["date"] = pd.to_datetime(bt["date"])
    return bt


# ---------------------------------------------------------------------------
# Step 3: Merge BB% into backtest profiles
# ---------------------------------------------------------------------------

def merge_bb_into_backtest(bt: pd.DataFrame, bb_rates: pd.DataFrame) -> pd.DataFrame:
    """Join rolling BB% onto backtest picks."""
    merged = bt.merge(bb_rates, on=["batter_id", "date"], how="left")
    n_missing = merged["bb_rate_30g"].isna().sum()
    n_total = len(merged)
    print(f"  BB% merge: {n_missing}/{n_total} picks missing BB rate "
          f"({n_missing/n_total*100:.1f}%)")
    return merged


# ---------------------------------------------------------------------------
# Analysis A: Correlation BB% vs batter_count_tendency_30g
# ---------------------------------------------------------------------------

def compute_count_tendency(pa: pd.DataFrame) -> pd.DataFrame:
    """Recompute batter_count_tendency_30g for correlation analysis."""
    pa["count_diff"] = pa["final_count_balls"] - pa["final_count_strikes"]
    date_count = (
        pa.groupby(["batter_id", "date"])["count_diff"]
        .mean()
        .reset_index()
        .rename(columns={"count_diff": "avg_count_diff"})
        .sort_values(["batter_id", "date"])
    )

    def rolling_tendency(grp):
        return grp["avg_count_diff"].shift(1).rolling(30, min_periods=10).mean()

    date_count["batter_count_tendency_30g"] = (
        date_count.groupby("batter_id", group_keys=False).apply(rolling_tendency)
    )
    return date_count[["batter_id", "date", "batter_count_tendency_30g"]]


def analyze_correlation(bb_rates: pd.DataFrame, count_tendency: pd.DataFrame) -> dict:
    """Correlate BB% with count_tendency on (batter, date) pairs."""
    merged = bb_rates.merge(count_tendency, on=["batter_id", "date"], how="inner")
    merged = merged.dropna(subset=["bb_rate_30g", "batter_count_tendency_30g"])

    r, p = stats.pearsonr(merged["bb_rate_30g"], merged["batter_count_tendency_30g"])
    r_sp, p_sp = stats.spearmanr(merged["bb_rate_30g"], merged["batter_count_tendency_30g"])
    return {
        "n": len(merged),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_r": r_sp,
        "spearman_p": p_sp,
    }


# ---------------------------------------------------------------------------
# Analysis B: P@1 by BB% quartile of the rank-1 pick
# ---------------------------------------------------------------------------

def analyze_p1_by_bb_quartile(bt_bb: pd.DataFrame) -> pd.DataFrame:
    """For rank-1 picks, bin by BB% quartile and compute P@1."""
    rank1 = bt_bb[bt_bb["rank"] == 1].copy()
    rank1 = rank1.dropna(subset=["bb_rate_30g"])

    rank1["bb_quartile"] = pd.qcut(
        rank1["bb_rate_30g"], q=4, labels=["Q1 (low BB)", "Q2", "Q3", "Q4 (high BB)"]
    )

    result = (
        rank1.groupby("bb_quartile", observed=True)
        .agg(
            days=("actual_hit", "count"),
            hits=("actual_hit", "sum"),
            mean_bb_pct=("bb_rate_30g", "mean"),
        )
        .assign(p1=lambda d: d["hits"] / d["days"])
    )
    return result


# ---------------------------------------------------------------------------
# Analysis C: Hard filter test — exclude BB% > 15% from rank-1 candidates
# ---------------------------------------------------------------------------

def analyze_hard_filter(bt_bb: pd.DataFrame, threshold: float = 0.15) -> dict:
    """
    Simulate: if rank-1 pick has BB% > threshold, fall back to next eligible rank.

    For each game-date, we have ranks 1-10. If rank-1 is filtered, use rank-2, etc.
    """
    rank1 = bt_bb[bt_bb["rank"] == 1].copy()
    baseline_p1 = rank1["actual_hit"].mean()
    baseline_n = len(rank1)

    # Days where rank-1 is a high-walk batter
    rank1_with_bb = rank1.dropna(subset=["bb_rate_30g"])
    filtered_days = rank1_with_bb[rank1_with_bb["bb_rate_30g"] > threshold]["date"]
    n_filtered = len(filtered_days)

    # For filtered days, find the best non-filtered rank
    filtered_outcomes = []
    for date in filtered_days:
        day_picks = bt_bb[bt_bb["date"] == date].sort_values("rank")
        # Find first pick that is not above the threshold
        eligible = day_picks[
            day_picks["bb_rate_30g"].isna() | (day_picks["bb_rate_30g"] <= threshold)
        ]
        if len(eligible) > 0:
            # Take the highest-ranked eligible pick
            best = eligible.iloc[0]
            filtered_outcomes.append(best["actual_hit"])
        else:
            # No eligible pick — treat as miss (conservative)
            filtered_outcomes.append(0)

    # Build new P@1: unfiltered days keep rank-1, filtered days use fallback
    unfiltered_rank1 = rank1_with_bb[rank1_with_bb["bb_rate_30g"] <= threshold]["actual_hit"]
    # Also include rank1 rows where bb_rate is NaN (can't filter, keep as is)
    rank1_na_bb = rank1[rank1["bb_rate_30g"].isna()]["actual_hit"]

    new_hits = unfiltered_rank1.sum() + sum(filtered_outcomes) + rank1_na_bb.sum()
    new_total = baseline_n
    new_p1 = new_hits / new_total

    # P@1 for just the filtered-day picks vs their original rank-1
    filtered_rank1 = rank1_with_bb[rank1_with_bb["bb_rate_30g"] > threshold]

    return {
        "threshold": threshold,
        "baseline_p1": baseline_p1,
        "baseline_n": baseline_n,
        "n_filtered_days": n_filtered,
        "pct_days_affected": n_filtered / len(rank1_with_bb) if len(rank1_with_bb) > 0 else 0,
        "filtered_rank1_p1": filtered_rank1["actual_hit"].mean() if len(filtered_rank1) > 0 else None,
        "fallback_p1": np.mean(filtered_outcomes) if filtered_outcomes else None,
        "new_overall_p1": new_p1,
        "delta_p1": new_p1 - baseline_p1,
    }


# ---------------------------------------------------------------------------
# Analysis D: High-walk batters appearing as rank-1
# ---------------------------------------------------------------------------

def analyze_high_walk_picks(bt_bb: pd.DataFrame, top_n_batters: int = 10) -> pd.DataFrame:
    """
    Find batters with highest overall BB% and their hit rate when picked rank-1.
    """
    rank1 = bt_bb[bt_bb["rank"] == 1].dropna(subset=["bb_rate_30g"])

    # Per-batter stats for rank-1 appearances
    batter_stats = (
        rank1.groupby("batter_id")
        .agg(
            n_rank1_days=("actual_hit", "count"),
            hit_rate=("actual_hit", "mean"),
            mean_bb_pct=("bb_rate_30g", "mean"),
            mean_p_game=("p_game_hit", "mean"),
        )
        .sort_values("mean_bb_pct", ascending=False)
        .head(top_n_batters)
    )
    return batter_stats


def find_soto(pa: pd.DataFrame) -> int | None:
    """Try to identify Juan Soto's batter_id by known walk rate profile."""
    # Soto is typically among the league leaders in walks each season
    # In 2024 with Yankees he had ~16% BB
    # Try batter IDs from known high-walk patterns in our data
    walk_df = pa[pa["event_type"].isin(WALK_EVENTS)]
    total_pas = pa.groupby("batter_id").size().rename("total_pa")
    walk_counts = walk_df.groupby("batter_id").size().rename("walks")
    bb_pct = (walk_counts / total_pas).rename("bb_pct")
    bb_df = pd.concat([total_pas, walk_counts, bb_pct], axis=1).dropna()
    # Filter to reasonable sample
    candidates = bb_df[(bb_df["total_pa"] >= 500) & (bb_df["bb_pct"] >= 0.14)]
    if len(candidates) > 0:
        return candidates["bb_pct"].idxmax()
    return None


# ---------------------------------------------------------------------------
# Multi-threshold filter sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(bt_bb: pd.DataFrame) -> pd.DataFrame:
    """Test multiple BB% filter thresholds."""
    thresholds = [0.10, 0.12, 0.15, 0.18, 0.20]
    results = []
    for t in thresholds:
        r = analyze_hard_filter(bt_bb, threshold=t)
        results.append(r)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Item 07: Walk-Rate / BB% Feature Investigation")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/5] Loading PA data (2019-2025 for warm-up)...")
    pa = load_pa_data()
    print(f"  Loaded {len(pa):,} PAs across {pa['season'].nunique()} seasons")

    print("\n[2/5] Computing rolling 30-game-date BB rate...")
    bb_rates = compute_rolling_bb_rate(pa)
    # Only keep analysis seasons
    analysis_dates = bt_all = None  # placeholder

    print("\n[3/5] Loading backtest profiles...")
    bt = load_backtest()
    print(f"  Loaded {len(bt):,} picks across {bt['season'].nunique()} seasons")

    print("\n[4/5] Merging BB% into backtest picks...")
    bt_bb = merge_bb_into_backtest(bt, bb_rates)

    print("\n[5/5] Running analyses...")

    # --- Analysis A: Correlation ---
    print("\n--- A. Correlation: BB% vs count_tendency_30g ---")
    count_tendency = compute_count_tendency(pa)
    corr = analyze_correlation(bb_rates, count_tendency)
    print(f"  N pairs: {corr['n']:,}")
    print(f"  Pearson r = {corr['pearson_r']:.4f}, p = {corr['pearson_p']:.2e}")
    print(f"  Spearman r = {corr['spearman_r']:.4f}, p = {corr['spearman_p']:.2e}")

    # --- Analysis B: P@1 by BB quartile ---
    print("\n--- B. P@1 by BB% quartile (rank-1 picks) ---")
    quartile_table = analyze_p1_by_bb_quartile(bt_bb)
    print(quartile_table.to_string())

    # --- Analysis C: Hard filter sweep ---
    print("\n--- C. Hard filter threshold sweep ---")
    sweep = sweep_thresholds(bt_bb)
    print(sweep[["threshold", "n_filtered_days", "pct_days_affected",
                  "filtered_rank1_p1", "fallback_p1", "new_overall_p1", "delta_p1"]].to_string(index=False))

    # Primary filter test at 15%
    print("\n--- C2. Primary filter at BB% > 15% ---")
    filter_result = analyze_hard_filter(bt_bb, threshold=0.15)
    for k, v in filter_result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # --- Analysis D: High-walk rank-1 batters ---
    print("\n--- D. Highest-walk batters appearing as rank-1 ---")
    high_walk_picks = analyze_high_walk_picks(bt_bb, top_n_batters=10)
    print(high_walk_picks.to_string())

    # --- Soto check ---
    print("\n--- D2. Juan Soto check ---")
    # Soto's MLB batter_id is 665742
    SOTO_ID = 665742
    soto_rows = bt_bb[(bt_bb["batter_id"] == SOTO_ID) & (bt_bb["rank"] == 1)]
    if len(soto_rows) > 0:
        print(f"  Soto (id={SOTO_ID}) rank-1 appearances: {len(soto_rows)}")
        print(f"  Hit rate: {soto_rows['actual_hit'].mean():.4f}")
        print(f"  Mean BB%: {soto_rows['bb_rate_30g'].mean():.4f}")
        print(f"  Mean p_game_hit: {soto_rows['p_game_hit'].mean():.4f}")
    else:
        print(f"  Soto (id={SOTO_ID}) never appeared as rank-1 in backtest seasons")
        # Check if he appears at all in the PA data
        soto_pa = pa[pa["batter_id"] == SOTO_ID]
        if len(soto_pa) > 0:
            soto_walks = soto_pa[soto_pa["event_type"].isin(WALK_EVENTS)]
            print(f"  Soto in PA data: {len(soto_pa)} PAs, {len(soto_walks)} walks "
                  f"= {len(soto_walks)/len(soto_pa)*100:.1f}% BB")
        else:
            print(f"  Soto not found in PA data (id={SOTO_ID} may be wrong)")
            # Find top walker
        top_walker_id = find_soto(pa)
        if top_walker_id:
            print(f"  Highest-BB batter in data: id={top_walker_id}")
            tw_rows = bt_bb[(bt_bb["batter_id"] == top_walker_id) & (bt_bb["rank"] == 1)]
            print(f"  Rank-1 appearances: {len(tw_rows)}")
            if len(tw_rows) > 0:
                print(f"  Hit rate: {tw_rows['actual_hit'].mean():.4f}")
                print(f"  Mean BB%: {tw_rows['bb_rate_30g'].mean():.4f}")

    # --- Baseline P@1 summary ---
    print("\n--- Summary ---")
    rank1 = bt_bb[bt_bb["rank"] == 1]
    print(f"  Overall P@1 (all seasons): {rank1['actual_hit'].mean():.4f}")
    print(f"  Total rank-1 pick-days: {len(rank1)}")

    return {
        "correlation": corr,
        "quartile_table": quartile_table,
        "filter_sweep": sweep,
        "high_walk_picks": high_walk_picks,
        "filter_15pct": filter_result,
    }


if __name__ == "__main__":
    results = main()
