#!/usr/bin/env python3
"""MDP policy parameter sweep on 2024+2025 backtest profiles.

Sweeps late_phase_days, n_bins, and season_length, computes optimal P(57)
for each combination, and prints the top-N configurations. No retraining
required — operates entirely on existing backtest_*.parquet profiles.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/mdp_policy_sweep.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.simulate.quality_bins import compute_bins


def split_by_phase(profiles_df: pd.DataFrame, late_phase_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split profiles into early-phase and late-phase subsets.

    For each season, the LAST `late_phase_days` distinct calendar dates
    are considered late-phase; everything else is early.
    """
    if late_phase_days <= 0:
        return profiles_df, profiles_df.iloc[0:0]

    early_rows = []
    late_rows = []
    for season, group in profiles_df.groupby("season"):
        dates = sorted(group["date"].unique())
        if len(dates) <= late_phase_days:
            late_rows.append(group)
            continue
        cutoff = dates[-late_phase_days]
        late_rows.append(group[group["date"] >= cutoff])
        early_rows.append(group[group["date"] < cutoff])
    early = pd.concat(early_rows, ignore_index=True) if early_rows else profiles_df.iloc[0:0]
    late = pd.concat(late_rows, ignore_index=True) if late_rows else profiles_df.iloc[0:0]
    return early, late


def run_sweep(profiles_df: pd.DataFrame) -> list[dict]:
    # Conservative bins: with N days/bin = total_days / (n_bins * n_phases), we
    # want >=30 days/bin for stable empirical estimates.
    late_days_grid = [0, 20, 28, 30, 32, 34, 35, 36, 37, 38, 40, 42, 45, 50]
    n_bins_grid = [3, 4, 5]
    season_length_grid = [180, 182, 185, 186]

    results = []
    combos = list(itertools.product(late_days_grid, n_bins_grid, season_length_grid))
    for i, (late_days, n_bins, season_length) in enumerate(combos):
        early_df, late_df = split_by_phase(profiles_df, late_days)

        if late_days == 0 or len(late_df) < 30:
            bins = compute_bins(early_df, n_bins=n_bins)
            min_bin_size = min(b.frequency for b in bins.bins) * early_df["date"].nunique()
            sol = solve_mdp(bins, season_length=season_length)
            mode = "single-phase"
        else:
            early_bins = compute_bins(early_df, n_bins=n_bins)
            late_bins = compute_bins(late_df, n_bins=n_bins)
            early_min = min(b.frequency for b in early_bins.bins) * early_df["date"].nunique()
            late_min = min(b.frequency for b in late_bins.bins) * late_df["date"].nunique()
            min_bin_size = min(early_min, late_min)
            sol = solve_mdp(
                early_bins,
                season_length=season_length,
                late_bins=late_bins,
                late_phase_days=late_days,
            )
            mode = "phase-aware"

        results.append({
            "late_days": late_days,
            "n_bins": n_bins,
            "season_length": season_length,
            "mode": mode,
            "p57_mdp": sol.optimal_p57,
            "n_early_days": early_df["date"].nunique() if len(early_df) else 0,
            "n_late_days": late_df["date"].nunique() if len(late_df) else 0,
            "min_bin_size": round(min_bin_size, 1),
        })

    return results


def main():
    import sys
    profiles_path = Path("data/simulation")
    parquets = sorted(profiles_path.glob("backtest_*.parquet"))
    if not parquets:
        raise SystemExit("No backtest_*.parquet found")

    # Allow --seasons 2024,2025 override, default to all 5 seasons
    seasons_arg = None
    if "--seasons" in sys.argv:
        seasons_arg = sys.argv[sys.argv.index("--seasons") + 1].split(",")
    if seasons_arg:
        parquets = [p for p in parquets if any(s in p.stem for s in seasons_arg)]

    print(f"Loading {len(parquets)} parquet(s): {[p.name for p in parquets]}")
    dfs = []
    for p in parquets:
        df = pd.read_parquet(p)
        if "season" not in df.columns:
            season = int(p.stem.replace("backtest_", ""))
            df["season"] = season
        dfs.append(df)
    profiles_df = pd.concat(dfs, ignore_index=True)
    profiles_df["date"] = pd.to_datetime(profiles_df["date"])
    n_seasons = profiles_df["season"].nunique()
    print(f"Total rows: {len(profiles_df):,}, "
          f"days: {profiles_df['date'].nunique()}, "
          f"seasons ({n_seasons}): {sorted(profiles_df['season'].unique())}")

    print("\nRunning MDP parameter sweep...")
    results = run_sweep(profiles_df)
    print(f"Done — {len(results)} configurations evaluated.\n")

    # Flag configs with too-small bins (<30 samples/bin)
    robust = [r for r in results if r["min_bin_size"] >= 30]
    fragile = [r for r in results if r["min_bin_size"] < 30]

    robust.sort(key=lambda r: -r["p57_mdp"])
    print(f"Top 12 (robust: min_bin_size >= 30 samples):")
    print(f"{'late_days':>10} {'n_bins':>7} {'season_len':>10} {'mode':<13} "
          f"{'min_bin':>8} {'P(57)':>10}")
    print("-" * 64)
    for r in robust[:12]:
        print(f"{r['late_days']:>10} {r['n_bins']:>7} {r['season_length']:>10} "
              f"{r['mode']:<13} {r['min_bin_size']:>8.0f} {r['p57_mdp']*100:>9.3f}%")

    print(f"\nFragile configs omitted ({len(fragile)} with min_bin_size < 30):")
    fragile.sort(key=lambda r: -r["p57_mdp"])
    for r in fragile[:5]:
        print(f"  SKIP  {r['late_days']:>4} late_days  n_bins={r['n_bins']}  "
              f"season={r['season_length']}  min_bin={r['min_bin_size']:.0f}  "
              f"P(57)={r['p57_mdp']*100:.3f}%  (too few samples)")

    print("\nBaseline reference configs:")
    refs = [
        (0, 5, 180, "single-phase n=5 (scorecard default)"),
        (30, 5, 180, "phase-aware n=5 (late=30, shipped config)"),
    ]
    for ld, nb, sl, label in refs:
        match = [r for r in results if r["late_days"] == ld and r["n_bins"] == nb and r["season_length"] == sl]
        if match:
            r = match[0]
            print(f"  {label}: P(57) = {r['p57_mdp']*100:.3f}% "
                  f"(min_bin={r['min_bin_size']:.0f})")


if __name__ == "__main__":
    main()
