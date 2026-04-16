#!/usr/bin/env python3
"""Replay + Monte Carlo A/B for pooled vs production MDP policies.

This is a second-signal cross-check of the forward-evaluator A/B in
scripts/pooled_policy_ab.py. Where the forward evaluator computes the
EXACT transition-matrix P(57) on bin aggregates, this script uses the
concrete per-day profile data directly:

    1. REPLAY: for each (seed, season) pair, iterate days chronologically
       through the 180-day season applying the policy to each day's
       actual rank-1/rank-2 predictions. Records max_streak and whether
       57 was reached. 16 seeds × 5 seasons = 80 trajectories per policy.

    2. MONTE CARLO: for each (seed, season), bootstrap 2000 synthetic
       180-day seasons by sampling days with replacement from the
       season's profile pool. Compute P(57) per seed-season, then
       aggregate.

If both replay and MC show the same direction (pooled > production)
with consistent magnitude, the finding is robust to the abstraction
chosen (exact transition matrix vs concrete day replay).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/mc_replay_ab.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import load_policy


POLICY_PROD = Path("data/models/mdp_policy.npz")
POLICY_POOLED = Path("data/models/mdp_policy_pooled_v1.npz")
PROFILES_ROOT = Path("data/hetzner_results/pooled_bins_run")
SEASON_LENGTH = 180
MC_TRIALS = 2000


@dataclass
class PolicyBundle:
    """Wraps a loaded .npz policy with lookup helpers."""
    table: np.ndarray
    boundaries: list[float]
    season_length: int
    name: str = "?"

    @classmethod
    def load(cls, path: Path, name: str) -> "PolicyBundle":
        table, boundaries, season_length = load_policy(path)
        return cls(table=table, boundaries=boundaries, season_length=season_length, name=name)

    def classify(self, confidence: float) -> int:
        q = 0
        for b in self.boundaries:
            if confidence >= b:
                q += 1
        n_bins = self.table.shape[3]
        return min(q, n_bins - 1)

    def action(self, streak: int, days_remaining: int, saver: int, confidence: float) -> int:
        if streak >= 57 or days_remaining <= 0:
            return 0
        d = min(days_remaining, self.table.shape[1] - 1)
        s = min(streak, 56)
        q = self.classify(confidence)
        return int(self.table[s, d, saver, q])


def _pair_ranks(season_df: pd.DataFrame) -> pd.DataFrame:
    """Pair rank-1 and rank-2 per date, return chronologically-sorted rows
    with columns [date, p_game_hit, top1_hit, top2_hit]."""
    r1 = season_df[season_df["rank"] == 1][["date", "p_game_hit", "actual_hit"]]
    r2 = season_df[season_df["rank"] == 2][["date", "actual_hit"]]
    merged = r1.merge(r2.rename(columns={"actual_hit": "top2_hit"}), on="date")
    merged = merged.rename(columns={"actual_hit": "top1_hit"})
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def replay_season(season_days: pd.DataFrame, policy: PolicyBundle) -> dict:
    """Chronological replay of one season under a fixed MDP policy.

    Returns max_streak (int), reached_57 (bool), play_days (int).
    """
    streak = 0
    max_streak = 0
    saver = 1
    reached_57 = False
    play_days = 0

    n_days = len(season_days)
    for i, row in enumerate(season_days.itertuples(index=False)):
        days_remaining = SEASON_LENGTH - i
        if days_remaining <= 0:
            break
        action = policy.action(streak, days_remaining, saver, row.p_game_hit)
        if action == 0:
            continue
        play_days += 1

        if action == 1:  # single
            if row.top1_hit:
                streak += 1
            else:
                if saver and 10 <= streak <= 15:
                    saver = 0
                else:
                    streak = 0
        else:  # double
            if row.top1_hit and row.top2_hit:
                streak += 2
            else:
                if saver and 10 <= streak <= 15:
                    saver = 0
                else:
                    streak = 0

        if streak > max_streak:
            max_streak = streak
        if streak >= 57:
            reached_57 = True
            break

    return {"max_streak": max_streak, "reached_57": reached_57, "play_days": play_days}


def mc_season_p57(season_days: pd.DataFrame, policy: PolicyBundle,
                   n_trials: int = MC_TRIALS, rng_seed: int = 12345) -> float:
    """Bootstrap P(57) from a season's profile pool.

    Samples n_trials synthetic seasons, each 180 days drawn with
    replacement from the season's profile pool. Returns fraction that
    reached streak 57.
    """
    rng = np.random.default_rng(rng_seed)
    n_days = len(season_days)
    reached = 0

    p_col = season_days["p_game_hit"].values
    h1_col = season_days["top1_hit"].values
    h2_col = season_days["top2_hit"].values

    for _ in range(n_trials):
        idxs = rng.integers(0, n_days, size=SEASON_LENGTH)
        streak = 0
        saver = 1
        for i, idx in enumerate(idxs):
            days_remaining = SEASON_LENGTH - i
            confidence = float(p_col[idx])
            action = policy.action(streak, days_remaining, saver, confidence)
            if action == 0:
                continue
            if action == 1:
                if h1_col[idx]:
                    streak += 1
                else:
                    if saver and 10 <= streak <= 15:
                        saver = 0
                    else:
                        streak = 0
            else:
                if h1_col[idx] and h2_col[idx]:
                    streak += 2
                else:
                    if saver and 10 <= streak <= 15:
                        saver = 0
                    else:
                        streak = 0
            if streak >= 57:
                reached += 1
                break
    return reached / n_trials


def main() -> None:
    prod = PolicyBundle.load(POLICY_PROD, "prod")
    pool = PolicyBundle.load(POLICY_POOLED, "pooled")

    seed_dirs = sorted(PROFILES_ROOT.glob("*/simulation_seed*"))
    print(f"Loaded {len(seed_dirs)} seed directories from {PROFILES_ROOT}")
    print(f"Policy prod:   shape={prod.table.shape}  boundaries={[round(b, 4) for b in prod.boundaries]}")
    print(f"Policy pooled: shape={pool.table.shape}  boundaries={[round(b, 4) for b in pool.boundaries]}")

    replay_rows = []
    mc_rows = []

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name.replace("simulation_seed", "")
        seed = int(seed_name)

        for parq in sorted(seed_dir.glob("backtest_*.parquet")):
            season = int(parq.stem.split("_")[1])
            df = pd.read_parquet(parq)
            season_days = _pair_ranks(df)
            if len(season_days) == 0:
                continue

            r_prod = replay_season(season_days, prod)
            r_pool = replay_season(season_days, pool)
            replay_rows.append({
                "seed": seed, "season": season,
                "prod_max_streak": r_prod["max_streak"],
                "pool_max_streak": r_pool["max_streak"],
                "prod_reached_57": int(r_prod["reached_57"]),
                "pool_reached_57": int(r_pool["reached_57"]),
                "prod_play_days": r_prod["play_days"],
                "pool_play_days": r_pool["play_days"],
            })

            mc_prod = mc_season_p57(season_days, prod)
            mc_pool = mc_season_p57(season_days, pool)
            mc_rows.append({
                "seed": seed, "season": season,
                "mc_prod_p57": mc_prod, "mc_pool_p57": mc_pool,
            })
            print(f"  seed={seed:>8}  season={season}  "
                  f"replay: prod {r_prod['max_streak']:>2} / pool {r_pool['max_streak']:>2}   "
                  f"MC: prod {mc_prod:.2%} / pool {mc_pool:.2%}")

    rdf = pd.DataFrame(replay_rows)
    mdf = pd.DataFrame(mc_rows)

    print("\n" + "=" * 60)
    print("REPLAY (chronological, per (seed, season))")
    print("=" * 60)
    print(f"  trajectories: {len(rdf)}")
    print(f"  prod mean max_streak = {rdf['prod_max_streak'].mean():.2f}  ±{rdf['prod_max_streak'].std(ddof=1):.2f}")
    print(f"  pool mean max_streak = {rdf['pool_max_streak'].mean():.2f}  ±{rdf['pool_max_streak'].std(ddof=1):.2f}")
    print(f"  gap  mean max_streak = {(rdf['pool_max_streak'] - rdf['prod_max_streak']).mean():+.2f}")
    print(f"  prod reached 57: {rdf['prod_reached_57'].sum()} / {len(rdf)}")
    print(f"  pool reached 57: {rdf['pool_reached_57'].sum()} / {len(rdf)}")
    pool_wins = (rdf["pool_max_streak"] > rdf["prod_max_streak"]).sum()
    ties = (rdf["pool_max_streak"] == rdf["prod_max_streak"]).sum()
    print(f"  per-trajectory wins (pool > prod): {pool_wins} / {len(rdf)}  "
          f"(ties: {ties}, losses: {len(rdf) - pool_wins - ties})")

    print("\n" + "=" * 60)
    print("MONTE CARLO (bootstrap, per (seed, season), 2000 trials)")
    print("=" * 60)
    print(f"  mean prod P(57) = {mdf['mc_prod_p57'].mean():.3%}  ±{mdf['mc_prod_p57'].std(ddof=1):.3%}")
    print(f"  mean pool P(57) = {mdf['mc_pool_p57'].mean():.3%}  ±{mdf['mc_pool_p57'].std(ddof=1):.3%}")
    gap = mdf["mc_pool_p57"] - mdf["mc_prod_p57"]
    print(f"  mean gap        = {gap.mean() * 100:+.3f}pp  ±{gap.std(ddof=1) * 100:.3f}pp")
    pool_wins_mc = (gap > 0).sum()
    print(f"  pool wins in {pool_wins_mc} / {len(mdf)} seed-seasons")

    out = Path("data/validation/pooled_policy_mc_replay_ab.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "replay_trajectories": replay_rows,
        "mc_results": mc_rows,
        "replay_summary": {
            "n_trajectories": int(len(rdf)),
            "prod_mean_max_streak": float(rdf["prod_max_streak"].mean()),
            "pool_mean_max_streak": float(rdf["pool_max_streak"].mean()),
            "mean_max_streak_gap": float((rdf["pool_max_streak"] - rdf["prod_max_streak"]).mean()),
            "prod_reached_57": int(rdf["prod_reached_57"].sum()),
            "pool_reached_57": int(rdf["pool_reached_57"].sum()),
            "pool_wins": int(pool_wins),
            "ties": int(ties),
        },
        "mc_summary": {
            "mean_prod_p57": float(mdf["mc_prod_p57"].mean()),
            "mean_pool_p57": float(mdf["mc_pool_p57"].mean()),
            "mean_gap": float(gap.mean()),
            "std_gap": float(gap.std(ddof=1)),
            "pool_wins": int(pool_wins_mc),
            "n_seed_seasons": int(len(mdf)),
        },
    }, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
