#!/usr/bin/env python3
"""Robustness pass for the estimated_pa MDP re-solve (pre-ship gauntlet).

Three analyses Codex flagged as required before any live swap:

  ABLATION   — separate the boundary fix from the aggressive new table by
               evaluating all 4 (table × boundaries) combos:
                 prod        = prod table  + prod boundaries  (deployed baseline)
                 cand        = cand table  + cand boundaries  (full re-solve)
                 bnd_only    = prod table  + cand boundaries  (SAFE fix: just re-bin)
                 table_only  = cand table  + prod boundaries
               If bnd_only ≈ cand, the conservative re-bin captures the benefit
               and the aggressive table is unnecessary.

  SHRINKAGE  — re-evaluate each combo with per-bin hit/both rates shrunk by
               δ ∈ {0, 2, 5} pp (the model may be overconfident live). If the
               candidate's edge or its skip-rate reduction reverses under mild
               pessimism, that's fragile.

  MILESTONES — realized-outcome forward replay per (season, seed): P(reach
               10/20/30/40), mean & max streak, resets, play rate. P(57) is ~0
               and useless; these achievable milestones show whether the extra
               aggression actually helps or just resets the streak more.

Usage:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/mdp_estpa_robustness.py \
    --profiles-root data/hetzner_results/mdp_estpa_run \
    --prod-policy /tmp/deployed_mdp_policy.npz \
    --cand-policy data/models/mdp_policy_pooled_estpa_v1.npz \
    --holdout-seasons 2024,2025
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import load_policy
from bts.simulate.quality_bins import compute_bins_with_boundaries, QualityBins, QualityBin
from bts.simulate.pooled_policy import load_pooled_profiles, evaluate_mdp_policy

SEASON_LENGTH = 180
MILESTONES = [10, 20, 30, 40, 57]


@dataclass
class Variant:
    name: str
    table: np.ndarray
    boundaries: list


def _shrink_bins(bins: QualityBins, delta: float) -> QualityBins:
    """Return bins with p_hit and p_both reduced by `delta` (pp as fraction),
    clamped to [0,1]. Models the live hit rate being worse than the backtest."""
    out = []
    for b in bins.bins:
        out.append(QualityBin(
            index=b.index, p_range=b.p_range,
            p_hit=float(max(0.0, b.p_hit - delta)),
            p_both=float(max(0.0, b.p_both - delta)),
            frequency=b.frequency,
        ))
    return QualityBins(bins=out, boundaries=bins.boundaries)


def _analytic_per_seed(profiles, variant, holdout_seasons, delta=0.0):
    """Per-seed analytic E[P(57)] for a variant, optional bin shrinkage."""
    vals = []
    for seed in sorted(profiles["seed"].unique()):
        sdf = profiles[(profiles["seed"] == seed) & (profiles["season"].isin(holdout_seasons))]
        bins = compute_bins_with_boundaries(sdf, variant.boundaries)
        if delta:
            bins = _shrink_bins(bins, delta)
        vals.append(float(evaluate_mdp_policy(
            variant.table, bins, season_length=SEASON_LENGTH, late_bins=None)))
    return np.array(vals)


def _milestone_replay(holdout_df, variant):
    """Forward-walk each (season, seed) on REALIZED outcomes; track milestones.
    Mirrors ope_eval._terminal_mc_replay's streak/saver/double dynamics."""
    table, bnd = variant.table, np.asarray(variant.boundaries, float)
    df = holdout_df.sort_values(["season", "seed", "date", "rank"])
    rows = []
    for (season, seed), g in df.groupby(["season", "seed"]):
        streak = 0; saver = 1; max_streak = 0; resets = 0
        played = skipped = doubled = 0
        days_remaining = SEASON_LENGTH
        for _date, day in g.groupby("date"):
            if days_remaining <= 0:
                break
            r1 = day[day["rank"] == 1]; r2 = day[day["rank"] == 2]
            if len(r1) == 0:
                action = 0
            else:
                p1 = float(r1.iloc[0]["p_game_hit"])
                qb = int(np.digitize(p1, bnd)); qb = max(0, min(qb, table.shape[3] - 1))
                d = min(days_remaining, table.shape[1] - 1)
                action = int(table[min(streak, 57), d, saver, qb])
            saver_active = bool(saver) and (10 <= streak <= 15)
            if action == 0:
                skipped += 1
            elif action == 1:
                played += 1
                hit = int(r1.iloc[0]["actual_hit"]) if len(r1) else 0
                if hit:
                    streak += 1
                elif saver_active:
                    saver = 0
                else:
                    streak = 0; resets += 1
            else:  # double
                played += 1; doubled += 1
                if len(r2) == 0:
                    hit = int(r1.iloc[0]["actual_hit"]) if len(r1) else 0
                    if hit: streak += 1
                    elif saver_active: saver = 0
                    else: streak = 0; resets += 1
                else:
                    both = int(r1.iloc[0]["actual_hit"]) & int(r2.iloc[0]["actual_hit"])
                    if both: streak += 2
                    elif saver_active: saver = 0
                    else: streak = 0; resets += 1
            max_streak = max(max_streak, streak)
            days_remaining -= 1
        rows.append({"season": int(season), "seed": int(seed), "max_streak": max_streak,
                     "resets": resets, "played": played, "skipped": skipped,
                     "doubled": doubled,
                     **{f"reach{m}": int(max_streak >= m) for m in MILESTONES}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles-root", type=Path, required=True)
    ap.add_argument("--prod-policy", type=Path, required=True)
    ap.add_argument("--cand-policy", type=Path, required=True)
    ap.add_argument("--holdout-seasons", default="2024,2025")
    ap.add_argument("--out", type=Path, default=Path("data/validation/mdp_estpa_robustness.json"))
    args = ap.parse_args()

    holdout = [int(s) for s in args.holdout_seasons.split(",")]
    seed_dirs = sorted(args.profiles_root.glob("*/simulation_seed*"))
    profiles = load_pooled_profiles(seed_dirs)
    profiles["season"] = profiles["season"].astype(int)
    hold = profiles[profiles["season"].isin(holdout)].copy()
    print(f"holdout {holdout}: {hold['seed'].nunique()} seeds, {len(hold):,} rows")

    pt, pb, _ = load_policy(args.prod_policy)
    ct, cb, _ = load_policy(args.cand_policy)
    variants = [
        Variant("prod (P table+P bnd)", pt, pb),
        Variant("cand (C table+C bnd)", ct, cb),
        Variant("bnd_only (P table+C bnd)", pt, cb),
        Variant("table_only (C table+P bnd)", ct, pb),
    ]

    # ---------- ABLATION ----------
    print("\n" + "=" * 72 + "\nABLATION — analytic E[P(57)] (mean over 24 seeds), isolate boundary vs table")
    print("=" * 72)
    abl = {}
    base = _analytic_per_seed(hold, variants[0], holdout)
    for v in variants:
        vals = _analytic_per_seed(hold, v, holdout)
        abl[v.name] = vals
        gap = vals - base
        print(f"  {v.name:28s} P57={vals.mean():.4%}   vs prod {gap.mean():+.4%}  "
              f"(wins {int((gap>0).sum())}/{len(gap)})")
    # how much of cand's gain does the safe bnd_only fix capture?
    cand_gain = abl[variants[1].name].mean() - base.mean()
    bnd_gain = abl[variants[2].name].mean() - base.mean()
    if cand_gain != 0:
        print(f"\n  → boundaries-only captures {bnd_gain/cand_gain:.0%} of the full re-solve's gain")

    # ---------- SHRINKAGE ----------
    print("\n" + "=" * 72 + "\nSHRINKAGE — analytic E[P(57)] with per-bin hit/both rates shrunk (live pessimism)")
    print("=" * 72)
    print(f"  {'variant':28s}  δ=0pp      δ=2pp      δ=5pp")
    shr = {}
    for v in variants:
        cells = []
        for d in (0.0, 0.02, 0.05):
            cells.append(float(_analytic_per_seed(hold, v, holdout, delta=d).mean()))
        shr[v.name] = cells
        print(f"  {v.name:28s}  {cells[0]:.4%}   {cells[1]:.4%}   {cells[2]:.4%}")
    print("  (if cand falls to/below prod as δ grows, the edge is fragile to overconfidence)")

    # ---------- MILESTONES ----------
    print("\n" + "=" * 72 + "\nMILESTONES — realized-outcome replay, 48 (season,seed) trajectories")
    print("=" * 72)
    print(f"  {'variant':28s} {'reach10':>7} {'reach20':>7} {'reach30':>7} {'reach40':>7} "
          f"{'maxStrk':>7} {'resets':>6} {'play%':>6}")
    mile = {}
    for v in variants:
        m = _milestone_replay(hold, v)
        mile[v.name] = m
        pr = {k: float(m[f"reach{k}"].mean()) for k in MILESTONES}
        play_rate = m["played"].sum() / (m["played"].sum() + m["skipped"].sum())
        print(f"  {v.name:28s} {pr[10]:>6.0%} {pr[20]:>6.1%} {pr[30]:>6.1%} {pr[40]:>6.1%} "
              f"{m['max_streak'].mean():>7.1f} {m['resets'].mean():>6.1f} {play_rate:>5.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "holdout_seasons": holdout,
        "ablation": {k: {"mean_p57": float(v.mean()), "vs_prod": float(v.mean()-base.mean())}
                     for k, v in abl.items()},
        "boundaries_only_captures_frac": float(bnd_gain/cand_gain) if cand_gain else None,
        "shrinkage": shr,
        "milestones": {k: {"reach": {str(m): float(df[f"reach{m}"].mean()) for m in MILESTONES},
                           "mean_max_streak": float(df["max_streak"].mean()),
                           "mean_resets": float(df["resets"].mean()),
                           "play_rate": float(df["played"].sum()/(df["played"].sum()+df["skipped"].sum()))}
                       for k, df in mile.items()},
    }, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
