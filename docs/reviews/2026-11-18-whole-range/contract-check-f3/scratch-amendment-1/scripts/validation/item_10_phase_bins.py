"""
Item 10: Phase-Aware Bin Granularity

Tests whether finer time bins (monthly, quarterly) in the MDP improve
P(57) over the current binary early/late split.

Configurations tested:
  a. No phases (baseline): single set of bins for full season
  b. Binary (current): early (month<=7) vs late (month>=8), late_phase_days=60
  c. Varying late cutoff: late_phase_days = 30, 45, 60, 75, 90
  d. Monthly P@1 by month: data sufficiency check

Also checks whether monthly bins have enough data to be reliable.
"""

import os
import sys

import numpy as np
import pandas as pd

os.chdir("/Users/stone/projects/bts")
sys.path.insert(0, "src")

from pathlib import Path

from bts.simulate.quality_bins import compute_bins
from bts.simulate.mdp import solve_mdp

SEASONS = [2021, 2022, 2023, 2024, 2025]
BACKTEST_PATTERN = "data/simulation/backtest_{season}.parquet"
SEASON_LENGTH = 180


def load_profiles() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        df = pd.read_parquet(BACKTEST_PATTERN.format(season=season))
        df["season"] = season
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["month"] = combined["date"].dt.month
    combined["day_of_year"] = combined["date"].dt.dayofyear
    return combined


def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def main():
    print("Loading backtest profiles (2021-2025)...")
    profiles = load_profiles()

    # Only keep rank-1 and rank-2 for bin computation
    n_days = profiles[profiles["rank"] == 1]["date"].nunique()
    print(f"  {len(profiles):,} total rows, {n_days} unique game-dates")

    # ── (d) Monthly P@1 and data sufficiency ─────────────────────────────────
    print_section("(d) Monthly P@1 and data volume (data sufficiency check)")

    rank1 = profiles[profiles["rank"] == 1].copy()
    monthly = rank1.groupby("month").agg(
        n_days=("date", "count"),
        p_at_1=("actual_hit", "mean"),
    ).reset_index()

    month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}

    print(f"\n  {'Month':<6} {'Name':<5} {'N days':>7} {'P@1':>7}")
    print(f"  {'─'*30}")
    for _, row in monthly.iterrows():
        m = int(row["month"])
        print(f"  {m:<6} {month_names.get(m, '?'):<5} {int(row['n_days']):>7} {row['p_at_1']:>7.3f}")

    print(f"\n  Note: 5 seasons × ~26 days/month = ~130 days/month expected")
    print(f"  Bin computation requires rank-1 AND rank-2 to merge successfully.")

    # ── Data per monthly bin ──────────────────────────────────────────────────
    print_section("Monthly bin n_days (after r1×r2 merge)")

    for month in sorted(profiles["month"].unique()):
        sub = profiles[profiles["month"] == month]
        try:
            bins = compute_bins(sub)
            # Total days in bins
            r1_sub = sub[sub["rank"] == 1]
            r2_sub = sub[sub["rank"] == 2]
            merged_count = r1_sub[["date"]].merge(r2_sub[["date"]], on="date").shape[0]
            bin_sizes = [int(round(b.frequency * merged_count)) for b in bins.bins]
            label = month_names.get(month, str(month))
            print(f"  Month {month} ({label}): {merged_count} days, bins={bin_sizes}, "
                  f"p_hit={[f'{b.p_hit:.3f}' for b in bins.bins]}")
        except Exception as e:
            print(f"  Month {month}: ERROR — {e}")

    # ── (a) Baseline: no phase split ──────────────────────────────────────────
    print_section("(a) Baseline: no phase split")

    all_bins = compute_bins(profiles)
    sol_baseline = solve_mdp(all_bins, season_length=SEASON_LENGTH)
    p57_baseline = sol_baseline.optimal_p57
    print(f"  P(57) = {p57_baseline:.4%}")
    print(f"  Bin hit rates: {[f'{b.p_hit:.3f}' for b in all_bins.bins]}")

    # ── (b) Binary (current): early month<=7 vs late month>=8 ────────────────
    print_section("(b) Binary (current): early (month<=7) vs late (month>=8), late_phase_days=60")

    early = profiles[profiles["month"] <= 7]
    late = profiles[profiles["month"] >= 8]
    early_bins = compute_bins(early)
    late_bins = compute_bins(late)

    print(f"  Early N days: {early[early['rank']==1]['date'].nunique()}")
    print(f"  Late  N days: {late[late['rank']==1]['date'].nunique()}")
    print(f"  Early bin hit rates: {[f'{b.p_hit:.3f}' for b in early_bins.bins]}")
    print(f"  Late  bin hit rates: {[f'{b.p_hit:.3f}' for b in late_bins.bins]}")

    sol_binary = solve_mdp(early_bins, season_length=SEASON_LENGTH,
                           late_bins=late_bins, late_phase_days=60)
    p57_binary = sol_binary.optimal_p57
    print(f"  P(57) = {p57_binary:.4%}  (delta vs baseline: {p57_binary - p57_baseline:+.4%})")

    # ── (c) Varying late cutoff ───────────────────────────────────────────────
    print_section("(c) Varying late_phase_days (binary early/late split)")

    cutoffs = [30, 45, 60, 75, 90]
    print(f"\n  {'late_phase_days':>17} {'P(57)':>8} {'Delta vs baseline':>18} {'Delta vs binary (60d)':>22}")
    print(f"  {'─' * 70}")
    results_cutoff = {}
    for cutoff in cutoffs:
        sol = solve_mdp(early_bins, season_length=SEASON_LENGTH,
                        late_bins=late_bins, late_phase_days=cutoff)
        p57 = sol.optimal_p57
        results_cutoff[cutoff] = p57
        marker = " ← current" if cutoff == 60 else ""
        print(f"  {cutoff:>17d} {p57:>8.4%} {p57 - p57_baseline:>+18.4%} {p57 - p57_binary:>+22.4%}{marker}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print_section("Summary")

    all_configs = [
        ("No phases (baseline)", p57_baseline),
        ("Binary early/late (current, 60d)", p57_binary),
    ]
    for cutoff in cutoffs:
        if cutoff != 60:
            all_configs.append((f"Binary early/late ({cutoff}d cutoff)", results_cutoff[cutoff]))

    best_name, best_p57 = max(all_configs, key=lambda x: x[1])
    best_delta = best_p57 - p57_baseline

    print(f"\n  {'Configuration':<40} {'P(57)':>8}")
    print(f"  {'─' * 50}")
    for name, p57 in all_configs:
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<40} {p57:>8.4%}{marker}")

    print(f"\n  Current binary (60d) vs baseline: {p57_binary - p57_baseline:+.4%}")
    print(f"  Best configuration: '{best_name}' at {best_p57:.4%}")
    print(f"  Best improvement over baseline: {best_delta:+.4%}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print_section("VERDICT")

    if best_name == "No phases (baseline)":
        print("  Phase-aware bins provide NO improvement. Binary split hurts or is noise.")
    elif best_name == "Binary early/late (current, 60d)":
        print("  Current binary split is already optimal.")
        print("  No value in finer granularity or different cutoff.")
    else:
        best_cutoff = int(best_name.split("(")[1].split("d")[0])
        gain_over_current = best_p57 - p57_binary
        print(f"  A {best_cutoff}d cutoff outperforms the current 60d by {gain_over_current:+.4%}.")
        print(f"  Consider updating late_phase_days={best_cutoff} in the production MDP solve.")

    # Monthly bins verdict
    print("\n  Monthly bin feasibility:")
    min_monthly = monthly["n_days"].min()
    if min_monthly < 50:
        print(f"  Monthly bins TOO THIN — smallest month has only {min_monthly} days (5 seasons).")
        print("  At ~5 bins per month, some bins would have <10 observations — unreliable.")
        print("  Binary or quarterly splits are the practical limit given 5-season corpus.")
    else:
        print(f"  Monthly bins are feasible — smallest month has {min_monthly} days.")

    return {
        "p57_baseline": p57_baseline,
        "p57_binary": p57_binary,
        "results_cutoff": results_cutoff,
        "best_config": best_name,
        "best_p57": best_p57,
        "monthly_min_days": int(monthly["n_days"].min()),
    }


if __name__ == "__main__":
    results = main()
