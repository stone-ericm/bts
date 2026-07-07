#!/usr/bin/env python3
"""Confirm the MDP-vs-simple question on the REALISTIC estimated_pa profiles
(data/hetzner_results/mdp_estpa_run, 24 seeds x 5 seasons, rank-1 hit ~0.75),
which reproduce the 2026-06-10 audit regime. Corrects the earlier run that used
the hindsight actual_pa profiles (rank-1 0.865).

Metrics: max_streak (Eric's longest-streak objective) + reach20 (audit milestone)
+ reached_57 + resets. Tests BOTH naive rank1+rank2 pairing AND the different-game
DD constraint (game_pk available here). replay_season copied verbatim from
scripts/mc_replay_ab.py.
"""
from __future__ import annotations
import glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from bts.simulate.mdp import load_policy

SEASON_LENGTH = 180
MDP_POLICY = Path("data/models/mdp_policy.npz")
ROOT = "data/hetzner_results/mdp_estpa_run"


@dataclass
class PolicyBundle:
    table: np.ndarray; boundaries: list; season_length: int; name: str = "?"
    @classmethod
    def load(cls, path, name):
        t, b, sl = load_policy(path); return cls(t, b, sl, name)
    def classify(self, c):
        q = 0
        for b in self.boundaries:
            if c >= b: q += 1
        return min(q, self.table.shape[3] - 1)
    def action(self, streak, days_remaining, saver, confidence):
        if streak >= 57 or days_remaining <= 0: return 0
        d = min(days_remaining, self.table.shape[1] - 1); s = min(streak, 56)
        return int(self.table[s, d, saver, self.classify(confidence)])


class FuncPolicy:
    def __init__(self, name, fn): self.name = name; self.fn = fn
    def action(self, streak, days_remaining, saver, confidence):
        if streak >= 57 or days_remaining <= 0: return 0
        return self.fn(streak, days_remaining, saver, confidence)


def replay_season(season_days, policy):  # verbatim logic from mc_replay_ab.py
    streak = max_streak = play_days = resets = 0; saver = 1; reached_57 = False
    for i, row in enumerate(season_days.itertuples(index=False)):
        days_remaining = SEASON_LENGTH - i
        if days_remaining <= 0: break
        action = policy.action(streak, days_remaining, saver, row.p_game_hit)
        if action == 0: continue
        play_days += 1
        hit = row.top1_hit if action == 1 else (row.top1_hit and row.top2_hit)
        if hit:
            streak += (1 if action == 1 else 2)
        else:
            if saver and 10 <= streak <= 15: saver = 0
            else: streak = 0; resets += 1
        if streak > max_streak: max_streak = streak
        if streak >= 57: reached_57 = True; break
    return {"max_streak": max_streak, "reached_57": reached_57, "resets": resets}


def pair_naive(df):
    r1 = df[df["rank"] == 1][["date", "p_game_hit", "actual_hit"]]
    r2 = df[df["rank"] == 2][["date", "actual_hit"]]
    m = r1.merge(r2.rename(columns={"actual_hit": "top2_hit"}), on="date")
    return m.rename(columns={"actual_hit": "top1_hit"}).sort_values("date").reset_index(drop=True)


def pair_diffgame(df):
    """rank-1 + best rank>=2 in a DIFFERENT game_pk (production's DD rule)."""
    rows = []
    for date, g in df.sort_values("rank").groupby("date"):
        r1 = g[g["rank"] == 1]
        if r1.empty: continue
        r1 = r1.iloc[0]
        others = g[(g["rank"] >= 2) & (g["game_pk"] != r1["game_pk"])]
        top2_hit = int(others.iloc[0]["actual_hit"]) if not others.empty else int(r1["actual_hit"])
        rows.append({"date": date, "p_game_hit": r1["p_game_hit"],
                     "top1_hit": int(r1["actual_hit"]), "top2_hit": top2_hit})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def run(pairing_fn, label):
    mdp = PolicyBundle.load(MDP_POLICY, "mdp_deployed")
    policies = [mdp,
                FuncPolicy("always_single", lambda s, d, v, c: 1),
                FuncPolicy("always_double", lambda s, d, v, c: 2),
                FuncPolicy("double_until_50", lambda s, d, v, c: 2 if s < 50 else 1)]
    files = sorted(glob.glob(f"{ROOT}/**/backtest_*.parquet", recursive=True))
    recs = []
    for f in files:
        season = int(re.search(r"backtest_(\d+)\.parquet", f).group(1))
        seed = re.search(r"simulation_seed(\d+)", f).group(1)
        sd = pairing_fn(pd.read_parquet(f))
        for pol in policies:
            r = replay_season(sd, pol)
            recs.append({"seed": seed, "season": season, "policy": pol.name, **r})
    df = pd.DataFrame(recs)
    print(f"\n{'='*72}\n{label}  ({df.groupby('policy').size().iloc[0]} trajectories/policy)\n{'='*72}")
    print(f"  {'policy':16s}  {'mean_max':>8s}  {'reach20':>8s}  {'reach30':>8s}  {'reached57':>9s}  {'resets':>7s}")
    for pol in policies:
        s = df[df.policy == pol.name]
        print(f"  {pol.name:16s}  {s.max_streak.mean():8.2f}  {(s.max_streak>=20).mean():8.1%}  "
              f"{(s.max_streak>=30).mean():8.1%}  {int(s.reached_57.sum()):>4d}/{len(s):<4d}  {s.resets.mean():7.1f}")
    ad = df[df.policy == "always_double"].reset_index(drop=True).max_streak
    md = df[df.policy == "mdp_deployed"].reset_index(drop=True).max_streak
    print(f"  --> always_double vs mdp on max_streak: AD_wins={(ad.values>md.values).sum()} "
          f"ties={(ad.values==md.values).sum()} MDP_wins={(md.values>ad.values).sum()}  "
          f"(mean gap AD-MDP = {(ad.mean()-md.mean()):+.2f})")


if __name__ == "__main__":
    run(pair_naive, "REALISTIC estimated_pa | NAIVE rank1+rank2 pairing")
    run(pair_diffgame, "REALISTIC estimated_pa | DIFFERENT-GAME DD constraint (production rule)")
