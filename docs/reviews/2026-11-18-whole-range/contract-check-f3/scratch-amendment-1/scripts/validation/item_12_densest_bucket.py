"""
Item 12: Densest-Bucket Strategy Validation

Tests the cost of rank displacement caused by the densest-bucket filter.
Backtest profiles use confirmed lineups, so lineup uncertainty can't be tested
directly — but we CAN measure: "if densest-bucket forces rank-2 instead of
rank-1, how much does P@1 drop?"

This is the COST function for rank displacement. The BENEFIT (lineup certainty)
is measured separately by item 8b.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data" / "simulation"
SEASONS = [2021, 2022, 2023, 2024, 2025]


def load_profiles() -> pd.DataFrame:
    dfs = []
    for season in SEASONS:
        path = DATA_DIR / f"backtest_{season}.parquet"
        df = pd.read_parquet(path)
        df["season"] = season
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def compute_p_at_1(df: pd.DataFrame, rank: int = 1) -> float:
    """Precision@1: fraction of days where the rank-N player got a hit."""
    subset = df[df["rank"] == rank]
    return subset["actual_hit"].mean()


def blended_p_at_1(p_rank1: float, p_rank2: float, displacement_rate: float) -> float:
    """P@1 when we're forced to rank-2 on `displacement_rate` fraction of days."""
    return (1 - displacement_rate) * p_rank1 + displacement_rate * p_rank2


def p57(p_at_1: float) -> float:
    """Approximate P(57-game streak) = p^57. Rough but directionally correct."""
    return p_at_1**57


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    print("Loading backtest profiles...")
    df = load_profiles()
    total_dates = df[df["rank"] == 1]["date"].nunique()
    total_player_days = len(df)
    print(f"  Seasons: {SEASONS}")
    print(f"  Unique game dates: {total_dates}")
    print(f"  Total player-days: {total_player_days}")
    print(f"  Ranks available: {sorted(df['rank'].unique())}")

    # -------------------------------------------------------------------------
    # A. Rank-1 and Rank-2 P@1 overall
    # -------------------------------------------------------------------------
    print_section("A. Overall Rank-1 vs Rank-2 P@1")

    p1 = compute_p_at_1(df, rank=1)
    p2 = compute_p_at_1(df, rank=2)
    gap = p1 - p2

    n_rank1 = (df["rank"] == 1).sum()
    n_rank2 = (df["rank"] == 2).sum()

    print(f"  Rank-1 P@1: {p1:.4f}  (n={n_rank1})")
    print(f"  Rank-2 P@1: {p2:.4f}  (n={n_rank2})")
    print(f"  Gap:        {gap:.4f} ({gap * 100:.2f} percentage points)")

    # -------------------------------------------------------------------------
    # B. By-season breakdown
    # -------------------------------------------------------------------------
    print_section("B. By-Season Rank-1 vs Rank-2 P@1")

    season_rows = []
    for season in SEASONS:
        s = df[df["season"] == season]
        s_p1 = compute_p_at_1(s, rank=1)
        s_p2 = compute_p_at_1(s, rank=2)
        n = (s["rank"] == 1).sum()
        season_rows.append(
            {
                "season": season,
                "rank1_p1": s_p1,
                "rank2_p1": s_p2,
                "gap": s_p1 - s_p2,
                "n_days": n,
            }
        )

    season_df = pd.DataFrame(season_rows)
    print(f"\n  {'Season':<8} {'Rank-1 P@1':<14} {'Rank-2 P@1':<14} {'Gap':<10} {'N days'}")
    print(f"  {'-' * 60}")
    for _, row in season_df.iterrows():
        print(
            f"  {int(row['season']):<8} {row['rank1_p1']:<14.4f} {row['rank2_p1']:<14.4f} "
            f"{row['gap']:<10.4f} {int(row['n_days'])}"
        )

    # -------------------------------------------------------------------------
    # C. Blended P@1 at various displacement rates
    # -------------------------------------------------------------------------
    print_section("C. Blended P@1 at Various Displacement Rates")

    displacement_rates = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]

    print(f"\n  {'Displacement':<14} {'Blended P@1':<14} {'P(57)':<12} {'P(57) vs baseline'}")
    print(f"  {'-' * 60}")

    baseline_p57 = p57(p1)
    for rate in displacement_rates:
        blended = blended_p_at_1(p1, p2, rate)
        p57_val = p57(blended)
        p57_delta = p57_val - baseline_p57
        p57_pct_change = (p57_delta / baseline_p57) * 100 if baseline_p57 > 0 else float("nan")
        label = f"{rate * 100:.0f}%"
        print(
            f"  {label:<14} {blended:<14.4f} {p57_val:<12.6f} "
            f"{p57_pct_change:+.1f}%"
        )

    # -------------------------------------------------------------------------
    # D. P(57) sensitivity summary
    # -------------------------------------------------------------------------
    print_section("D. P(57) Sensitivity Summary")

    print(f"\n  Baseline P(57) at rank-1 P@1={p1:.4f}: {baseline_p57 * 100:.4f}%")
    print(f"  If always forced to rank-2 (P@1={p2:.4f}): {p57(p2) * 100:.4f}%")

    # At what displacement rate does P(57) drop by 10%?
    # Solve: blended(rate)^57 = 0.9 * baseline_p57
    target = 0.9 * baseline_p57
    target_p1 = target ** (1 / 57)
    # blended = p1 - rate * gap
    # target_p1 = p1 - rate * gap  => rate = (p1 - target_p1) / gap
    if gap > 0:
        rate_for_10pct_drop = (p1 - target_p1) / gap
        print(
            f"\n  Displacement rate needed for -10% P(57): "
            f"{rate_for_10pct_drop * 100:.1f}%"
        )
    else:
        print("\n  Gap is zero — no displacement risk.")

    # -------------------------------------------------------------------------
    # E. Distribution of p_game_hit for rank-1 vs rank-2
    # -------------------------------------------------------------------------
    print_section("E. p_game_hit Distribution: Rank-1 vs Rank-2")

    r1_probs = df[df["rank"] == 1]["p_game_hit"]
    r2_probs = df[df["rank"] == 2]["p_game_hit"]
    prob_gap = r1_probs.values - r2_probs.values[: len(r1_probs)]  # paired

    # Align by date for proper pairing
    r1 = df[df["rank"] == 1][["date", "season", "p_game_hit"]].rename(
        columns={"p_game_hit": "p1"}
    )
    r2 = df[df["rank"] == 2][["date", "season", "p_game_hit"]].rename(
        columns={"p_game_hit": "p2"}
    )
    paired = r1.merge(r2, on=["date", "season"])
    paired["prob_gap"] = paired["p1"] - paired["p2"]

    print(f"\n  Rank-1 p_game_hit: mean={r1_probs.mean():.4f}, median={r1_probs.median():.4f}")
    print(f"  Rank-2 p_game_hit: mean={r2_probs.mean():.4f}, median={r2_probs.median():.4f}")
    print(f"  Probability gap (rank1 - rank2): mean={paired['prob_gap'].mean():.4f}, "
          f"median={paired['prob_gap'].median():.4f}")
    print(f"  Days where gap < 0.01: {(paired['prob_gap'] < 0.01).sum()} "
          f"({(paired['prob_gap'] < 0.01).mean() * 100:.1f}%)")
    print(f"  Days where gap < 0.02: {(paired['prob_gap'] < 0.02).sum()} "
          f"({(paired['prob_gap'] < 0.02).mean() * 100:.1f}%)")

    # -------------------------------------------------------------------------
    # F. Key finding summary
    # -------------------------------------------------------------------------
    print_section("F. Key Findings")

    print(f"""
  Rank-1 P@1:   {p1:.4f}
  Rank-2 P@1:   {p2:.4f}
  Gap:          {gap:.4f} ({gap * 100:.2f} pp)

  Baseline P(57): {baseline_p57 * 100:.4f}%

  Cost of displacement at 10% rate:
    Blended P@1 = {blended_p_at_1(p1, p2, 0.10):.4f}
    P(57) = {p57(blended_p_at_1(p1, p2, 0.10)) * 100:.4f}%
    Delta = {(p57(blended_p_at_1(p1, p2, 0.10)) - baseline_p57) * 100:.4f}% absolute

  Cost of displacement at 20% rate:
    Blended P@1 = {blended_p_at_1(p1, p2, 0.20):.4f}
    P(57) = {p57(blended_p_at_1(p1, p2, 0.20)) * 100:.4f}%
    Delta = {(p57(blended_p_at_1(p1, p2, 0.20)) - baseline_p57) * 100:.4f}% absolute
""")


if __name__ == "__main__":
    main()
