"""
Item 3: Batting Order Signal Investigation

Tests whether explicit lineup_position adds predictive value beyond n_pas
(PA-count aggregation). lokikg (top r/beatthestreak model) lists lineup position
as his #1 feature; we dropped it because it double-counts with PA aggregation.

Analyses:
  a. P@1 by lineup slot — when rank-1 pick bats 1st vs 3rd vs 8th, what's P@1?
  b. Rank-1 lineup slot distribution — does the model naturally prefer high-PA slots?
  c. n_pas by lineup slot — do backtest profiles show more PAs for higher lineup slots?
  d. Times-through-order proxy — P@1 by lineup slot AFTER controlling for n_pas.
     The subtle argument: leadoff hitters see the starter on their first PA (weakest
     point in the order for the pitcher). If P@1 varies with slot AFTER controlling
     for n_pas, there's residual signal PA-count doesn't capture.
"""

import numpy as np
import pandas as pd
from scipy import stats


SEASONS = [2021, 2022, 2023, 2024, 2025]
BACKTEST_PATTERN = "data/simulation/backtest_{season}.parquet"
PA_PATTERN = "data/processed/pa_{season}.parquet"


def load_backtest() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        df = pd.read_parquet(BACKTEST_PATTERN.format(season=season))
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_pa_data() -> pd.DataFrame:
    """Load PA data, keeping only columns needed for batting order join."""
    frames = []
    for season in SEASONS:
        df = pd.read_parquet(
            PA_PATTERN.format(season=season),
            columns=["date", "batter_id", "lineup_position"],
        )
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def derive_batter_lineup_slot(pa: pd.DataFrame) -> pd.DataFrame:
    """
    For each (batter_id, date), determine the modal lineup slot.
    A batter should have the same lineup_position across all PAs in a game,
    but take the mode in case of rare substitution edge cases.
    """
    return (
        pa.groupby(["batter_id", "date"])["lineup_position"]
        .agg(lambda x: int(x.dropna().mode().iloc[0]) if x.notna().any() else np.nan)
        .reset_index()
        .rename(columns={"lineup_position": "batting_slot"})
    )


def two_proportion_z_test(n1_hits, n1, n2_hits, n2):
    """Two-proportion z-test. Returns (z-stat, two-sided p-value)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = n1_hits / n1
    p2 = n2_hits / n2
    p_pool = (n1_hits + n2_hits) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val


def main():
    print("Loading backtest profiles...")
    backtest = load_backtest()
    rank1 = backtest[backtest["rank"] == 1].copy()
    print(f"  {len(rank1)} rank-1 picks across {SEASONS}")

    print("Loading PA data (lineup_position column)...")
    pa = load_pa_data()
    print(f"  {len(pa):,} PA rows across {SEASONS}")

    # Normalize join keys
    rank1["date"] = pd.to_datetime(rank1["date"]).dt.date
    pa["date"] = pd.to_datetime(pa["date"]).dt.date
    rank1["batter_id"] = rank1["batter_id"].astype(int)
    pa["batter_id"] = pa["batter_id"].astype(int)

    print("Deriving modal lineup slot per batter×date...")
    batter_slot = derive_batter_lineup_slot(pa)

    # Join lineup slot onto rank-1 picks
    rank1 = rank1.merge(batter_slot, on=["batter_id", "date"], how="left")
    n_missing = rank1["batting_slot"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} rank-1 picks had no lineup_position — dropping")
    rank1 = rank1.dropna(subset=["batting_slot"])
    rank1["batting_slot"] = rank1["batting_slot"].astype(int)
    total = len(rank1)
    print(f"  {total} rank-1 picks with lineup slot resolved")

    # ── (a) P@1 by lineup slot ────────────────────────────────────────────────
    print("\n── (a) P@1 by lineup slot (rank-1 picks) ───────────────────")
    print(f"  {'Slot':<6} {'P@1':>7} {'n':>6} {'hits':>6} {'avg_n_pas':>10}")
    slot_stats = []
    for slot in range(1, 10):
        sub = rank1[rank1["batting_slot"] == slot]
        n = len(sub)
        if n == 0:
            continue
        hits = sub["actual_hit"].sum()
        p1 = hits / n
        avg_npas = sub["n_pas"].mean()
        slot_stats.append({"slot": slot, "n": n, "hits": hits, "p1": p1, "avg_npas": avg_npas})
        print(f"  {slot:<6} {p1:>7.3f} {n:>6} {hits:>6} {avg_npas:>10.2f}")

    slot_df = pd.DataFrame(slot_stats)

    # ── (b) Rank-1 lineup slot distribution ──────────────────────────────────
    print("\n── (b) Rank-1 pick distribution by lineup slot ─────────────")
    slot_counts = rank1["batting_slot"].value_counts().sort_index()
    print(f"  {'Slot':<6} {'Count':>6} {'%':>7}")
    for slot, count in slot_counts.items():
        pct = 100 * count / total
        print(f"  {slot:<6} {count:>6} {pct:>7.1f}%")

    # ── (c) n_pas by lineup slot in backtest profiles ─────────────────────────
    print("\n── (c) Avg n_pas by lineup slot in backtest profiles ────────")
    npas_by_slot = rank1.groupby("batting_slot")["n_pas"].agg(["mean", "count"])
    print(f"  {'Slot':<6} {'avg n_pas':>10} {'n':>6}")
    for slot, row in npas_by_slot.iterrows():
        print(f"  {slot:<6} {row['mean']:>10.3f} {row['count']:>6.0f}")

    # Correlation between lineup slot and n_pas
    corr_slot_npas, p_corr = stats.pearsonr(rank1["batting_slot"], rank1["n_pas"])
    print(f"\n  Pearson r (slot vs n_pas): {corr_slot_npas:.3f}  p={p_corr:.4f}")
    print("  (Negative r = top of order gets more PAs, as expected)")

    # ── (d) Residual signal: P@1 by slot AFTER controlling for n_pas ─────────
    print("\n── (d) Times-through-order: P@1 by slot within n_pas buckets ─")
    print("    Tests whether leadoff hitters have a starter-first-PA advantage")
    print("    beyond what n_pas already captures.\n")

    # Bin n_pas into 3 quantile groups (low / mid / high)
    rank1["npas_bin"] = pd.qcut(rank1["n_pas"], q=3, labels=["low_npas", "mid_npas", "high_npas"])

    # Top of order (1-3) vs middle (4-6) vs bottom (7-9)
    rank1["order_group"] = pd.cut(
        rank1["batting_slot"],
        bins=[0, 3, 6, 9],
        labels=["top (1-3)", "middle (4-6)", "bottom (7-9)"],
    )

    pivot = (
        rank1.groupby(["npas_bin", "order_group"])["actual_hit"]
        .agg(["sum", "count"])
        .reset_index()
    )
    pivot["p1"] = pivot["sum"] / pivot["count"]
    pivot.columns = ["npas_bin", "order_group", "hits", "n", "p1"]

    print(f"  {'n_pas bucket':<12} {'order group':<14} {'P@1':>7} {'n':>6}")
    for _, row in pivot.iterrows():
        print(f"  {row['npas_bin']:<12} {str(row['order_group']):<14} {row['p1']:>7.3f} {row['n']:>6.0f}")

    # Statistical test: top vs bottom of order, overall (n_pas not controlled)
    print()
    top = rank1[rank1["batting_slot"].isin([1, 2, 3])]
    bot = rank1[rank1["batting_slot"].isin([7, 8, 9])]
    z_tb, p_tb = two_proportion_z_test(
        top["actual_hit"].sum(), len(top),
        bot["actual_hit"].sum(), len(bot),
    )
    print(f"  Top (1-3) P@1: {top['actual_hit'].mean():.3f}  (n={len(top)})")
    print(f"  Bot (7-9) P@1: {bot['actual_hit'].mean():.3f}  (n={len(bot)})")
    print(f"  z={z_tb:.3f}  p={p_tb:.4f}  ({'SIGNIFICANT p<0.05' if p_tb < 0.05 else 'not significant'})")

    # Within equal n_pas: test top vs bottom
    print()
    print("  Within-bucket top vs bottom z-tests:")
    print(f"  {'bucket':<12} {'top P@1':>8} {'(n)':>5} {'bot P@1':>8} {'(n)':>5} {'z':>7} {'p':>8}")
    for bucket in ["low_npas", "mid_npas", "high_npas"]:
        sub = rank1[rank1["npas_bin"] == bucket]
        t = sub[sub["batting_slot"].isin([1, 2, 3])]
        b = sub[sub["batting_slot"].isin([7, 8, 9])]
        if len(t) == 0 or len(b) == 0:
            continue
        z, pv = two_proportion_z_test(
            t["actual_hit"].sum(), len(t),
            b["actual_hit"].sum(), len(b),
        )
        tp1 = t["actual_hit"].mean()
        bp1 = b["actual_hit"].mean()
        sig = "SIGNIFICANT" if pv < 0.05 else ""
        print(f"  {bucket:<12} {tp1:>8.3f} {len(t):>5} {bp1:>8.3f} {len(b):>5} {z:>7.3f} {pv:>8.4f}  {sig}")

    # ── P@1 by slot, model vs actual ─────────────────────────────────────────
    print("\n── Model predicted p_game_hit by lineup slot (rank-1 only) ─")
    pred_by_slot = rank1.groupby("batting_slot")["p_game_hit"].mean()
    actual_by_slot = rank1.groupby("batting_slot")["actual_hit"].mean()
    print(f"  {'Slot':<6} {'Pred p_hit':>10} {'Actual P@1':>11} {'Diff':>8}")
    for slot in range(1, 10):
        if slot not in pred_by_slot.index:
            continue
        pred = pred_by_slot[slot]
        actual = actual_by_slot[slot]
        print(f"  {slot:<6} {pred:>10.3f} {actual:>11.3f} {actual - pred:>+8.3f}")

    # ── Per-season slot distribution check ───────────────────────────────────
    print("\n── Slot 1-3 picks as % of rank-1 total, by season ──────────")
    print(f"  {'Season':<8} {'top-3 %':>8} {'n top-3':>8} {'total':>7}")
    for season in SEASONS:
        s = rank1[rank1["season"] == season]
        s_top = s[s["batting_slot"].isin([1, 2, 3])]
        pct = 100 * len(s_top) / len(s) if len(s) > 0 else float("nan")
        print(f"  {season:<8} {pct:>8.1f}% {len(s_top):>8} {len(s):>7}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n── VERDICT ─────────────────────────────────────────────────")

    overall_corr_p1_slot, pv_p1_slot = stats.pearsonr(
        slot_df["slot"], slot_df["p1"]
    ) if len(slot_df) >= 3 else (float("nan"), float("nan"))
    print(f"  Pearson r (slot vs P@1):  {overall_corr_p1_slot:.3f}  p={pv_p1_slot:.4f}")

    if p_tb >= 0.05:
        print("  Top-vs-bottom P@1 gap is not statistically significant.")
        print("  PA-count aggregation appears to fully subsume the batting order signal.")
    else:
        print(f"  SIGNIFICANT top-vs-bottom P@1 gap: z={z_tb:.3f}, p={p_tb:.4f}")
        print("  Residual batting order signal may exist beyond n_pas.")
        print("  Consider whether n_pas sufficiently captures this or if slot should be re-added.")

    return {
        "slot_stats": slot_df.to_dict(orient="records"),
        "corr_slot_npas": corr_slot_npas,
        "p_corr_slot_npas": p_corr,
        "z_top_vs_bot": z_tb,
        "p_top_vs_bot": p_tb,
        "corr_slot_p1": overall_corr_p1_slot,
        "p_corr_slot_p1": pv_p1_slot,
    }


if __name__ == "__main__":
    import os
    os.chdir("/Users/stone/projects/bts")
    results = main()
