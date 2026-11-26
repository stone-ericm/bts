"""Out-of-sample validation of MDP policy and bin-count sensitivity."""

import pandas as pd
import numpy as np
from pathlib import Path
from bts.simulate.quality_bins import compute_bins
from bts.simulate.mdp import solve_mdp, lookup_action
from bts.simulate.monte_carlo import load_profiles, simulate_season
from bts.simulate.strategies import ALL_STRATEGIES

# Load profiles per season
dfs = {int(p.stem.split("_")[1]): pd.read_parquet(p)
       for p in sorted(Path("data/simulation").glob("backtest_*.parquet"))}

print("=== Out-of-sample replay: train on 4 seasons, MDP-play on 1 ===")
for test_year in sorted(dfs.keys()):
    train = pd.concat([v for k, v in dfs.items() if k != test_year], ignore_index=True)
    test = dfs[test_year]
    bins = compute_bins(train, n_bins=5)
    sol = solve_mdp(bins, season_length=180)

    test_profiles = load_profiles(test)
    streak = 0
    max_streak = 0
    saver_used = False
    play_days = 0

    for day_idx, day in enumerate(test_profiles):
        days_remaining = len(test_profiles) - day_idx
        saver = not saver_used
        action = lookup_action(
            sol.policy_table, bins.boundaries,
            streak, days_remaining, saver, day.top1_p, sol.season_length,
        )
        if action == "skip":
            continue
        play_days += 1
        if action == "double":
            if day.top1_hit and day.top2_hit:
                streak += 2
            else:
                if saver and 10 <= streak <= 15:
                    saver_used = True
                else:
                    streak = 0
        else:
            if day.top1_hit:
                streak += 1
            else:
                if saver and 10 <= streak <= 15:
                    saver_used = True
                else:
                    streak = 0
        max_streak = max(max_streak, streak)

    h_result = simulate_season(test_profiles, ALL_STRATEGIES["combined"])
    print(f"  {test_year}: MDP streak={max_streak} ({play_days}/{len(test_profiles)} days)  "
          f"heuristic streak={h_result.max_streak}")

print()
print("=== In-sample vs out-of-sample MDP P(57) ===")
all_data = pd.concat(dfs.values(), ignore_index=True)
bins_all = compute_bins(all_data, n_bins=5)
sol_all = solve_mdp(bins_all, season_length=180)
print(f"  In-sample (all 5 seasons): MDP={sol_all.optimal_p57:.4%}")

oos_p57s = []
for test_year in sorted(dfs.keys()):
    train = pd.concat([v for k, v in dfs.items() if k != test_year], ignore_index=True)
    bins = compute_bins(train, n_bins=5)
    sol = solve_mdp(bins, season_length=180)
    oos_p57s.append(sol.optimal_p57)
    print(f"  Leave-{test_year}-out: MDP={sol.optimal_p57:.4%}")
print(f"  Average OOS: MDP={np.mean(oos_p57s):.4%}")

print()
print("=== Season-phase-aware bins (early vs late) ===")
all_data["month"] = pd.to_datetime(all_data["date"]).dt.month

# Split into early (Mar-Jul) and late (Aug-Sep)
early = all_data[all_data["month"] <= 7]
late = all_data[all_data["month"] >= 8]

print(f"  Early (Mar-Jul): {early['date'].nunique()} days")
bins_early = compute_bins(early, n_bins=5)
for b in bins_early.bins:
    print(f"    Q{b.index+1}: P(hit)={b.p_hit:.1%}  P(both)={b.p_both:.1%}")

print(f"  Late (Aug-Sep): {late['date'].nunique()} days")
bins_late = compute_bins(late, n_bins=5)
for b in bins_late.bins:
    print(f"    Q{b.index+1}: P(hit)={b.p_hit:.1%}  P(both)={b.p_both:.1%}")

# Solve MDP for each phase and report
sol_early = solve_mdp(bins_early, season_length=120)  # ~4 months
sol_late = solve_mdp(bins_late, season_length=60)    # ~2 months
print(f"\n  Early-only MDP (120 days): {sol_early.optimal_p57:.4%}")
print(f"  Late-only MDP (60 days): {sol_late.optimal_p57:.4%}")
