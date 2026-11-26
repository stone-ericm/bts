"""
Item 11: Miss-Day Pattern Analysis
Analyze rank-1 miss days from backtest profiles (2021-2025) to find actionable patterns.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "simulation"
SEASONS = [2021, 2022, 2023, 2024, 2025]


def load_all_profiles() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        df = pd.read_parquet(DATA_DIR / f"backtest_{season}.parquet")
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    df = load_all_profiles()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    r1 = df[df["rank"] == 1].copy()
    r2 = df[df["rank"] == 2].copy()

    # -------------------------------------------------------------------------
    # A. Basic stats
    # -------------------------------------------------------------------------
    section("A. Basic Stats by Season")
    print(f"{'Season':>7}  {'Days':>5}  {'Hits':>5}  {'Misses':>7}  {'P@1':>6}")
    print("-" * 40)
    total_days = total_hits = total_misses = 0
    for season in SEASONS:
        s = r1[r1["season"] == season]
        days = len(s)
        hits = s["actual_hit"].sum()
        misses = days - hits
        p1 = hits / days
        total_days += days
        total_hits += hits
        total_misses += misses
        print(f"{season:>7}  {days:>5}  {hits:>5}  {misses:>7}  {p1:>6.3f}")
    print("-" * 40)
    print(f"{'ALL':>7}  {total_days:>5}  {total_hits:>5}  {total_misses:>7}  {total_hits/total_days:>6.3f}")

    # -------------------------------------------------------------------------
    # B. Confidence signal: p_game_hit on hit vs miss days
    # -------------------------------------------------------------------------
    section("B. Confidence Signal: p_game_hit on Hit vs Miss Days")
    hit_days = r1[r1["actual_hit"] == 1]
    miss_days = r1[r1["actual_hit"] == 0]

    print(f"  Hit days   — mean p_game_hit: {hit_days['p_game_hit'].mean():.4f}  (n={len(hit_days)})")
    print(f"  Miss days  — mean p_game_hit: {miss_days['p_game_hit'].mean():.4f}  (n={len(miss_days)})")
    print(f"  Gap: {hit_days['p_game_hit'].mean() - miss_days['p_game_hit'].mean():.4f}")

    # t-test
    t, p = stats.ttest_ind(hit_days["p_game_hit"], miss_days["p_game_hit"])
    print(f"  t-test: t={t:.3f}, p={p:.4f}")

    # Distribution of p_game_hit on miss days
    print(f"\n  p_game_hit distribution on miss days:")
    bins = [0.0, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 1.0]
    counts, edges = np.histogram(miss_days["p_game_hit"], bins=bins)
    for i, c in enumerate(counts):
        print(f"    [{edges[i]:.2f}-{edges[i+1]:.2f}): {c} misses ({100*c/len(miss_days):.1f}%)")

    # -------------------------------------------------------------------------
    # C. Confidence gap: rank-1 minus rank-2
    # -------------------------------------------------------------------------
    section("C. Confidence Gap (rank-1 minus rank-2) on Hit vs Miss Days")

    # Merge rank-1 and rank-2 on same day+season
    merged = pd.merge(
        r1[["date", "season", "p_game_hit", "actual_hit"]].rename(columns={"p_game_hit": "p1", "actual_hit": "hit1"}),
        r2[["date", "season", "p_game_hit", "actual_hit"]].rename(columns={"p_game_hit": "p2", "actual_hit": "hit2"}),
        on=["date", "season"],
    )
    merged["gap"] = merged["p1"] - merged["p2"]

    gap_hit = merged[merged["hit1"] == 1]["gap"]
    gap_miss = merged[merged["hit1"] == 0]["gap"]

    print(f"  Hit days   — mean gap: {gap_hit.mean():.4f}  median: {gap_hit.median():.4f}")
    print(f"  Miss days  — mean gap: {gap_miss.mean():.4f}  median: {gap_miss.median():.4f}")
    t2, p2 = stats.ttest_ind(gap_hit, gap_miss)
    print(f"  t-test on gap: t={t2:.3f}, p={p2:.4f}")

    # Skip threshold analysis
    print(f"\n  Skip-if-gap-small threshold analysis:")
    print(f"  {'Threshold':>10}  {'Skipped':>8}  {'Skip%':>6}  {'Remain P@1':>11}  {'Delta P@1':>10}")
    print(f"  {'-'*55}")
    base_p1 = merged["hit1"].mean()
    for thresh in [0.002, 0.005, 0.01, 0.02]:
        played = merged[merged["gap"] >= thresh]
        skipped = len(merged) - len(played)
        skip_pct = 100 * skipped / len(merged)
        if len(played) > 0:
            remain_p1 = played["hit1"].mean()
            delta = remain_p1 - base_p1
        else:
            remain_p1 = float("nan")
            delta = float("nan")
        print(f"  {thresh:>10.3f}  {skipped:>8}  {skip_pct:>5.1f}%  {remain_p1:>11.4f}  {delta:>+10.4f}")

    # Show where misses fall in gap distribution
    print(f"\n  Gap distribution on miss days:")
    gap_bins = [-0.01, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0]
    counts_m, edges_m = np.histogram(gap_miss, bins=gap_bins)
    counts_h, _ = np.histogram(gap_hit, bins=gap_bins)
    for i, (cm, ch) in enumerate(zip(counts_m, counts_h)):
        total_bin = cm + ch
        miss_rate_in_bin = cm / total_bin if total_bin > 0 else float("nan")
        print(f"    [{edges_m[i]:.3f}-{edges_m[i+1]:.3f}): {cm} misses / {total_bin} total  ({100*miss_rate_in_bin:.1f}% miss rate)")

    # -------------------------------------------------------------------------
    # D. Rank-2 hit rate on rank-1 miss days
    # -------------------------------------------------------------------------
    section("D. Rank-2 Hit Rate on Rank-1 Miss Days")
    miss_merged = merged[merged["hit1"] == 0]
    hit_merged = merged[merged["hit1"] == 1]

    r2_on_miss = miss_merged["hit2"].mean()
    r2_on_hit = hit_merged["hit2"].mean()
    overall_r2 = merged["hit2"].mean()

    print(f"  Rank-2 hit rate when rank-1 misses: {r2_on_miss:.4f} ({100*r2_on_miss:.1f}%)")
    print(f"  Rank-2 hit rate when rank-1 hits:   {r2_on_hit:.4f} ({100*r2_on_hit:.1f}%)")
    print(f"  Overall rank-2 hit rate:             {overall_r2:.4f} ({100*overall_r2:.1f}%)")
    t3, p3 = stats.ttest_ind(miss_merged["hit2"], hit_merged["hit2"])
    print(f"  t-test: t={t3:.3f}, p={p3:.4f}")
    print(f"  Correlation(hit1, hit2): {merged['hit1'].corr(merged['hit2']):.4f}")

    # -------------------------------------------------------------------------
    # E. P@1 by month
    # -------------------------------------------------------------------------
    section("E. P@1 by Month (all seasons combined)")
    month_names = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
    print(f"  {'Month':>5}  {'Days':>5}  {'Hits':>5}  {'P@1':>6}")
    print(f"  {'-'*28}")
    for month in sorted(r1["month"].unique()):
        m = r1[r1["month"] == month]
        days = len(m)
        hits = m["actual_hit"].sum()
        p1_m = hits / days
        name = month_names.get(month, str(month))
        print(f"  {name:>5}  {days:>5}  {hits:>5}  {p1_m:>6.3f}")

    # Per-season by month
    print(f"\n  P@1 by Month and Season:")
    header = f"  {'Month':>5}" + "".join(f"  {s:>6}" for s in SEASONS)
    print(header)
    print(f"  {'-'*(len(header)-2)}")
    for month in sorted(r1["month"].unique()):
        name = month_names.get(month, str(month))
        row = f"  {name:>5}"
        for season in SEASONS:
            m = r1[(r1["month"] == month) & (r1["season"] == season)]
            if len(m) > 0:
                row += f"  {m['actual_hit'].mean():>6.3f}"
            else:
                row += f"  {'---':>6}"
        print(row)

    # -------------------------------------------------------------------------
    # F. Miss clustering: are misses random or clustered?
    # -------------------------------------------------------------------------
    section("F. Miss Clustering Analysis")

    # Collect miss dates per season and compute inter-miss gaps
    all_gaps = []
    for season in SEASONS:
        s = r1[r1["season"] == season].sort_values("date")
        miss_dates = s[s["actual_hit"] == 0]["date"].sort_values()
        n_days = len(s)
        n_misses = len(miss_dates)
        if n_misses < 2:
            continue
        gaps = miss_dates.diff().dropna().dt.days
        all_gaps.extend(gaps.tolist())
        print(f"  {season}: {n_misses} misses in {n_days} days — avg inter-miss gap: {gaps.mean():.1f}d, median: {gaps.median():.0f}d")

    # Under Poisson process, inter-arrival times are exponential
    # Expected mean gap = n_days / n_misses (same as observed mean)
    # Test: fit exponential and check goodness of fit
    all_gaps_arr = np.array(all_gaps)
    n_total_days = sum(len(r1[r1["season"] == s]) for s in SEASONS)
    n_total_misses = int((r1["actual_hit"] == 0).sum())
    expected_mean_gap = n_total_days / n_total_misses
    observed_mean_gap = all_gaps_arr.mean()

    print(f"\n  Combined: {n_total_misses} misses in {n_total_days} days")
    print(f"  Expected mean gap (Poisson): {expected_mean_gap:.1f}d")
    print(f"  Observed mean gap:           {observed_mean_gap:.1f}d")

    # Variance ratio test: Poisson has var(gap) = mean^2; overdispersion indicates clustering
    expected_var = expected_mean_gap ** 2
    observed_var = all_gaps_arr.var()
    print(f"  Expected variance (Poisson): {expected_var:.1f}")
    print(f"  Observed variance:           {observed_var:.1f}")
    print(f"  Variance ratio (obs/exp):    {observed_var/expected_var:.3f}")
    if observed_var / expected_var > 1.2:
        print(f"  -> Overdispersed: misses cluster MORE than random")
    elif observed_var / expected_var < 0.8:
        print(f"  -> Underdispersed: misses cluster LESS than random (regular spacing)")
    else:
        print(f"  -> Consistent with random (Poisson) distribution")

    # KS test against exponential
    exp_scale = observed_mean_gap
    ks_stat, ks_p = stats.kstest(all_gaps_arr, "expon", args=(0, exp_scale))
    print(f"  KS test vs Exponential: stat={ks_stat:.4f}, p={ks_p:.4f}")
    if ks_p < 0.05:
        print(f"  -> Gaps NOT consistent with exponential (clustering likely)")
    else:
        print(f"  -> Gaps consistent with exponential (random miss timing)")

    # Gap distribution
    print(f"\n  Inter-miss gap distribution:")
    gap_bins2 = [0, 1, 2, 3, 5, 7, 10, 14, 100]
    counts_g, edges_g = np.histogram(all_gaps_arr, bins=gap_bins2)
    for i, c in enumerate(counts_g):
        pct = 100 * c / len(all_gaps_arr)
        print(f"    {int(edges_g[i]):>3d}-{int(edges_g[i+1]):>3d}d: {c:>4}  ({pct:.1f}%)")

    # Streak-of-hits between misses
    print(f"\n  Longest hit streaks between misses (per season):")
    for season in SEASONS:
        s = r1[r1["season"] == season].sort_values("date")
        max_streak = current = 0
        for h in s["actual_hit"]:
            if h == 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        print(f"  {season}: longest hit streak = {max_streak}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    section("SUMMARY")
    print(f"  Overall P@1: {total_hits/total_days:.4f} ({total_hits}/{total_days})")
    print(f"  Hit days mean confidence:  {hit_days['p_game_hit'].mean():.4f}")
    print(f"  Miss days mean confidence: {miss_days['p_game_hit'].mean():.4f}")
    print(f"  Confidence gap separability: {'YES' if p < 0.05 else 'NO'} (p={p:.4f})")
    best_thresh = None
    best_delta = 0.0
    for thresh in [0.002, 0.005, 0.01, 0.02]:
        played = merged[merged["gap"] >= thresh]
        if len(played) > 0:
            delta = played["hit1"].mean() - base_p1
            if delta > best_delta:
                best_delta = delta
                best_thresh = thresh
    if best_thresh:
        played_best = merged[merged["gap"] >= best_thresh]
        skip_n = len(merged) - len(played_best)
        print(f"  Best gap threshold: {best_thresh} — skips {skip_n} days, P@1 delta = {best_delta:+.4f}")
    print(f"  Rank-2 hit rate on rank-1 miss days: {r2_on_miss:.4f}")
    print(f"  Miss clustering: variance ratio = {observed_var/expected_var:.3f}")


if __name__ == "__main__":
    main()
