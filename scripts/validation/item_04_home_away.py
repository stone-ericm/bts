"""
Item 4: Home/Away Analysis

Checks whether the visiting team PA advantage (guaranteed bottom of 9th)
is already captured by the model, or if there's residual signal in rank-1 picks.

Steps:
1. Load backtest profiles (2021-2025) — rank-1 picks with p_game_hit, actual_hit
2. Load PA data (2021-2025) — batter_id, date, is_home, game_pk
3. Join: determine if rank-1 batter was home or away
4. Compute P@1 by home/away split
5. Compute average PAs per game for home vs away batters (all batters, not just rank-1)
6. Statistical significance test (two-proportion z-test)
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
    """Load PA data, keeping only columns needed for home/away join."""
    frames = []
    for season in SEASONS:
        df = pd.read_parquet(
            PA_PATTERN.format(season=season),
            columns=["game_pk", "date", "batter_id", "is_home"],
        )
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def derive_batter_home_flag(pa: pd.DataFrame) -> pd.DataFrame:
    """
    For each (batter_id, date), determine if they played as home or away.
    A batter may appear multiple PAs in a game — take the mode (all PAs should
    have the same is_home value for a given game).
    """
    return (
        pa.groupby(["batter_id", "date"])["is_home"]
        .agg(lambda x: bool(x.mode()[0]))
        .reset_index()
        .rename(columns={"is_home": "batter_is_home"})
    )


def two_proportion_z_test(n_home_hits, n_home, n_away_hits, n_away):
    """Two-proportion z-test. Returns z-stat and two-sided p-value."""
    p_home = n_home_hits / n_home
    p_away = n_away_hits / n_away
    p_pool = (n_home_hits + n_away_hits) / (n_home + n_away)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_home + 1 / n_away))
    if se == 0:
        return 0.0, 1.0
    z = (p_home - p_away) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val


def main():
    print("Loading backtest profiles...")
    backtest = load_backtest()
    rank1 = backtest[backtest["rank"] == 1].copy()
    print(f"  {len(rank1)} rank-1 picks across {SEASONS}")

    print("Loading PA data...")
    pa = load_pa_data()
    print(f"  {len(pa):,} PA rows across {SEASONS}")

    # Normalize types for join (backtest dates are datetime.date objects, PA dates are str)
    rank1["date"] = pd.to_datetime(rank1["date"]).dt.date
    pa["date"] = pd.to_datetime(pa["date"]).dt.date
    rank1["batter_id"] = rank1["batter_id"].astype(int)
    pa["batter_id"] = pa["batter_id"].astype(int)

    print("Deriving home/away flag per batter×date...")
    batter_home = derive_batter_home_flag(pa)

    # Join home flag onto rank-1 picks
    rank1 = rank1.merge(batter_home, on=["batter_id", "date"], how="left")
    n_missing = rank1["batter_is_home"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} rank-1 picks could not be matched to PA data — dropping")
    rank1 = rank1.dropna(subset=["batter_is_home"])

    # ── P@1 by home/away ──────────────────────────────────────────────────────
    home = rank1[rank1["batter_is_home"] == True]
    away = rank1[rank1["batter_is_home"] == False]

    n_home = len(home)
    n_away = len(away)
    n_home_hits = home["actual_hit"].sum()
    n_away_hits = away["actual_hit"].sum()

    p1_home = n_home_hits / n_home if n_home > 0 else float("nan")
    p1_away = n_away_hits / n_away if n_away > 0 else float("nan")

    z, p_val = two_proportion_z_test(n_home_hits, n_home, n_away_hits, n_away)

    print("\n── Rank-1 pick composition ─────────────────────────────────")
    print(f"  Home picks : {n_home:4d}  ({100*n_home/(n_home+n_away):.1f}%)")
    print(f"  Away picks : {n_away:4d}  ({100*n_away/(n_home+n_away):.1f}%)")

    print("\n── P@1 by home/away ─────────────────────────────────────────")
    print(f"  Home P@1 : {p1_home:.3f}  ({n_home_hits}/{n_home})")
    print(f"  Away P@1 : {p1_away:.3f}  ({n_away_hits}/{n_away})")
    print(f"  Diff     : {p1_home - p1_away:+.3f}  (home − away)")
    print(f"  z-stat   : {z:.3f}")
    print(f"  p-value  : {p_val:.3f}  ({'SIGNIFICANT p<0.05' if p_val < 0.05 else 'not significant'})")

    # ── Average PAs per game by home/away (all batters, not just rank-1) ─────
    print("\n── Average PAs per game by home/away (all batters) ─────────")
    pa_per_game = (
        pa.groupby(["game_pk", "batter_id", "is_home"])
        .size()
        .reset_index(name="pa_count")
    )
    avg_pa = pa_per_game.groupby("is_home")["pa_count"].mean()
    for is_home, avg in avg_pa.items():
        label = "Home" if is_home else "Away"
        print(f"  {label} avg PAs/game: {avg:.3f}")

    pa_diff = avg_pa.get(False, float("nan")) - avg_pa.get(True, float("nan"))
    print(f"  Away − Home diff : {pa_diff:+.3f} PAs/game")

    # ── Per-season breakdown ──────────────────────────────────────────────────
    print("\n── Per-season P@1 split ────────────────────────────────────")
    print(f"  {'Season':<8} {'Home P@1':>9} {'(n)':>6} {'Away P@1':>9} {'(n)':>6} {'Diff':>7}")
    for season in SEASONS:
        s = rank1[rank1["season"] == season]
        sh = s[s["batter_is_home"] == True]
        sa = s[s["batter_is_home"] == False]
        ph = sh["actual_hit"].mean() if len(sh) > 0 else float("nan")
        pa_ = sa["actual_hit"].mean() if len(sa) > 0 else float("nan")
        diff = ph - pa_
        print(f"  {season:<8} {ph:>9.3f} {len(sh):>6} {pa_:>9.3f} {len(sa):>6} {diff:>+7.3f}")

    # ── Model's average predicted probability by home/away ────────────────────
    print("\n── Model predicted p_game_hit by home/away (rank-1 only) ───")
    pred_home = rank1[rank1["batter_is_home"] == True]["p_game_hit"].mean()
    pred_away = rank1[rank1["batter_is_home"] == False]["p_game_hit"].mean()
    print(f"  Home avg p_game_hit: {pred_home:.3f}")
    print(f"  Away avg p_game_hit: {pred_away:.3f}")
    print(f"  Diff (home − away) : {pred_home - pred_away:+.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── VERDICT ─────────────────────────────────────────────────")
    if p_val >= 0.05:
        print("  No statistically significant home/away P@1 difference.")
        print("  Signal is either already captured by the model or negligible.")
    else:
        direction = "home" if p1_home > p1_away else "away"
        print(f"  SIGNIFICANT difference: {direction} batters perform better at rank-1.")
        print(f"  Residual signal may exist — consider whether model already encodes this.")

    return {
        "n_home": n_home,
        "n_away": n_away,
        "p1_home": p1_home,
        "p1_away": p1_away,
        "z": z,
        "p_val": p_val,
        "avg_pa_home": float(avg_pa.get(True, float("nan"))),
        "avg_pa_away": float(avg_pa.get(False, float("nan"))),
        "pred_home": pred_home,
        "pred_away": pred_away,
    }


if __name__ == "__main__":
    import os
    os.chdir("/Users/stone/projects/bts")
    results = main()
