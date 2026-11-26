#!/usr/bin/env python3
"""As-deployed A/B for the estimated_pa MDP re-solve.

Compares a PROD policy vs a CANDIDATE policy on the estimated_pa profile
holdout, replaying EACH policy through ITS OWN saved boundaries — the way it
would actually run in production. This is the boundary-faithful comparison the
quality-bin-collapse fix requires; the stock scripts/pooled_policy_ab.py is
WRONG here because it re-bins both policies on shared quantiles (hiding the
boundary-driven benefit). See docs/audit/2026-06-10-mdp-estpa-ab-methodology.md.

Primary metric: analytic E[P(57)] via evaluate_mdp_policy on as-deployed bins
(compute_bins_with_boundaries → each policy's own boundaries, zero-freq bins
retained). Exact backward induction, no rare-event MC noise.
Secondary metric: empirical _terminal_mc_replay (noisy, directional check).

Both computed PER SEED (compute_bins_with_boundaries merges rank-1/rank-2 on
date only, so it must see one seed at a time), then bootstrapped over seeds.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/mdp_estpa_ab.py \
        --profiles-root data/hetzner_results/mdp_estpa_run \
        --prod-policy /tmp/deployed_mdp_policy.npz \
        --cand-policy data/models/mdp_policy_pooled_estpa_v1.npz \
        --holdout-seasons 2024,2025 \
        --out data/validation/mdp_estpa_ab_insample.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import load_policy
from bts.simulate.quality_bins import compute_bins_with_boundaries
from bts.simulate.pooled_policy import load_pooled_profiles, evaluate_mdp_policy
from bts.validate.ope_eval import _terminal_mc_replay

SEASON_LENGTH = 180


def _per_seed_eval(seed_df, table, boundaries):
    """Analytic + empirical as-deployed value for one policy on one seed's
    holdout, binning through the policy's OWN boundaries."""
    bins = compute_bins_with_boundaries(seed_df, boundaries)
    v_analytic = float(evaluate_mdp_policy(
        table, bins, season_length=SEASON_LENGTH, late_bins=None,
    ))
    v_replay, n_traj, n_term = _terminal_mc_replay(
        seed_df, table, bins, season_length=SEASON_LENGTH,
        late_bins=None, late_dates=None,
    )
    return v_analytic, float(v_replay), int(n_traj), int(n_term)


def _decision_shift(holdout_df, prod_t, prod_b, cand_t, cand_b):
    """How the two policies' live decisions differ. Captures the collapse:
    bin occupancy under each policy's own boundaries, the fraction of decisions
    that change across a representative state grid, and the skip/single/double
    action-mix shift. A policy-pair property, computed on the pooled rank-1
    holdout picks (the live decision stream)."""
    p = holdout_df[holdout_df["rank"] == 1]["p_game_hit"].to_numpy()
    pb = np.digitize(p, np.asarray(prod_b, float))
    cb = np.digitize(p, np.asarray(cand_b, float))
    n_bins = prod_t.shape[3]
    occ_prod = (np.bincount(pb, minlength=n_bins) / len(p)).tolist()
    occ_cand = (np.bincount(cb, minlength=n_bins) / len(p)).tolist()
    J = np.zeros((n_bins, n_bins))
    for a, b in zip(pb, cb):
        J[a, b] += 1
    J /= J.sum()
    streaks, days, savers = [0, 5, 10, 15, 20, 30, 40, 50], [30, 60, 90, 120, 150], [0, 1]
    dis = agg = more_aggr = 0.0
    pmix = np.zeros(3); cmix = np.zeros(3)
    for s in streaks:
        for d in days:
            dd = min(d, prod_t.shape[1] - 1)
            for sv in savers:
                for a in range(n_bins):
                    for b in range(n_bins):
                        f = J[a, b]
                        if f == 0:
                            continue
                        pa, ca = int(prod_t[s, dd, sv, a]), int(cand_t[s, dd, sv, b])
                        pmix[pa] += f; cmix[ca] += f; agg += f
                        if pa != ca:
                            dis += f
                        if ca > pa:
                            more_aggr += f
    return {
        "n_picks": int(len(p)), "pick_range": [float(p.min()), float(p.max())],
        "bin_occupancy_prod": occ_prod, "bin_occupancy_cand": occ_cand,
        "decision_disagreement": dis / agg, "cand_more_aggressive": more_aggr / agg,
        "action_mix_prod": (pmix / agg).tolist(), "action_mix_cand": (cmix / agg).tolist(),
    }


def _bootstrap_ci(gaps: np.ndarray, n_boot: int = 10000, seed: int = 12345):
    """Percentile CI for the mean of paired per-seed gaps (resample seeds)."""
    rng = np.random.default_rng(seed)
    n = len(gaps)
    means = np.array([rng.choice(gaps, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles-root", type=Path, required=True,
                    help="Dir containing per-box subdirs with simulation_seed* children")
    ap.add_argument("--prod-policy", type=Path, required=True)
    ap.add_argument("--cand-policy", type=Path, required=True)
    ap.add_argument("--holdout-seasons", default="2024,2025",
                    help="Comma seasons to evaluate on (the deployment holdout)")
    ap.add_argument("--out", type=Path, default=Path("data/validation/mdp_estpa_ab.json"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    holdout_seasons = {int(s) for s in args.holdout_seasons.split(",")}
    seed_dirs = sorted(args.profiles_root.glob("*/simulation_seed*"))
    print(f"Loading {len(seed_dirs)} seed dirs from {args.profiles_root}")
    profiles = load_pooled_profiles(seed_dirs)
    profiles["season"] = profiles["season"].astype(int)
    profiles = profiles[profiles["season"].isin(holdout_seasons)].copy()
    seeds = sorted(int(s) for s in profiles["seed"].unique())
    print(f"Holdout seasons {sorted(holdout_seasons)} | seeds={len(seeds)} | rows={len(profiles):,}")

    prod_table, prod_b, prod_sl = load_policy(args.prod_policy)
    cand_table, cand_b, cand_sl = load_policy(args.cand_policy)
    print(f"\nPROD  {args.prod_policy.name}: bins={prod_table.shape[3]} "
          f"boundaries={[round(x,4) for x in prod_b]} season_length={prod_sl}")
    print(f"CAND  {args.cand_policy.name}: bins={cand_table.shape[3]} "
          f"boundaries={[round(x,4) for x in cand_b]} season_length={cand_sl}")

    rows = []
    print(f"\n{'seed':>10} {'prodP57':>9} {'candP57':>9} {'gap':>8} | "
          f"{'prodRep':>8} {'candRep':>8}")
    for seed in seeds:
        sdf = profiles[profiles["seed"] == seed].copy()
        vp, rp, ntp, ntermp = _per_seed_eval(sdf, prod_table, prod_b)
        vc, rc, ntc, ntermc = _per_seed_eval(sdf, cand_table, cand_b)
        rows.append({"seed": seed, "v_prod": vp, "v_cand": vc,
                     "replay_prod": rp, "replay_cand": rc,
                     "n_traj": ntp, "term_prod": ntermp, "term_cand": ntermc})
        print(f"{seed:>10} {vp:>8.3%} {vc:>8.3%} {vc-vp:>+7.3%} | {rp:>7.2%} {rc:>7.2%}")

    vprod = np.array([r["v_prod"] for r in rows])
    vcand = np.array([r["v_cand"] for r in rows])
    gaps = vcand - vprod
    lo, hi = _bootstrap_ci(gaps, n_boot=args.n_boot)
    rep_prod = np.array([r["replay_prod"] for r in rows])
    rep_cand = np.array([r["replay_cand"] for r in rows])

    print("\n" + "=" * 64)
    print("ANALYTIC as-deployed E[P(57)] (primary):")
    print(f"  mean prod = {vprod.mean():.4%}  ± {vprod.std(ddof=1):.4%}")
    print(f"  mean cand = {vcand.mean():.4%}  ± {vcand.std(ddof=1):.4%}")
    print(f"  mean gap  = {gaps.mean():+.4%}   95% CI [{lo:+.4%}, {hi:+.4%}]")
    print(f"  cand wins in {int((gaps>0).sum())}/{len(gaps)} seeds")
    print(f"\nEMPIRICAL replay E[P(57)] (secondary, rare-event noisy):")
    print(f"  mean prod = {rep_prod.mean():.3%} | mean cand = {rep_cand.mean():.3%} "
          f"| gap {rep_cand.mean()-rep_prod.mean():+.3%}")
    shift = _decision_shift(profiles, prod_table, prod_b, cand_table, cand_b)
    ACT = ["skip", "single", "double"]
    print(f"\nDECISION SHIFT (collapse + behavioral change on {shift['n_picks']} live picks):")
    print(f"  bin occupancy PROD: {[round(x,3) for x in shift['bin_occupancy_prod']]}  "
          f"(collapse → bin0={shift['bin_occupancy_prod'][0]:.0%})")
    print(f"  bin occupancy CAND: {[round(x,3) for x in shift['bin_occupancy_cand']]}")
    print(f"  decisions changed: {shift['decision_disagreement']:.1%}  "
          f"(cand more aggressive in {shift['cand_more_aggressive']:.1%})")
    print(f"  action mix PROD: " + ", ".join(f"{ACT[i]} {shift['action_mix_prod'][i]:.0%}" for i in range(3)))
    print(f"  action mix CAND: " + ", ".join(f"{ACT[i]} {shift['action_mix_cand'][i]:.0%}" for i in range(3)))
    ship = "SHIP" if (gaps.mean() > 0 and lo > 0) else (
        "TIE/HOLD" if lo <= 0 <= hi else "REGRESSION")
    print(f"\n  VERDICT: {ship}  (CI {'excludes' if lo>0 or hi<0 else 'includes'} 0)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "holdout_seasons": sorted(holdout_seasons),
        "prod_policy": str(args.prod_policy), "cand_policy": str(args.cand_policy),
        "prod_boundaries": prod_b, "cand_boundaries": cand_b,
        "per_seed": rows,
        "analytic": {"mean_prod": float(vprod.mean()), "mean_cand": float(vcand.mean()),
                     "mean_gap": float(gaps.mean()), "ci95": [lo, hi],
                     "cand_wins": int((gaps > 0).sum()), "n_seeds": len(seeds)},
        "empirical_replay": {"mean_prod": float(rep_prod.mean()),
                             "mean_cand": float(rep_cand.mean())},
        "decision_shift": shift,
        "verdict": ship,
    }, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
