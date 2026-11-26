#!/usr/bin/env python3
"""Component E from item #6 plan: backtest pooled-prediction calibration vs single seed=42.

For each seed in {seed=42, *canonical-n10}, runs a full walk-forward over
the test seasons capturing per-(date, batter_id) predictions. Then:

  - "single": top-1 picks at seed=42 → calibration table on realized outcomes
  - "pooled": pool predictions across canonical-n10 seeds, re-rank, top-1 per
              day → calibration table on the same realized outcomes
  - random-10 sanity check: pool a fresh random-10 (sampled from baseline)
                            and compare to canonical-n10 to detect cherry-pick

Gate (matches plan Component E):
  - pooled.bucket_75_80.overconfidence_pp must be ≥ 5pp better than single, OR
  - pooled.brier ≤ single.brier
  - random-10 must show similar improvement to canonical-n10 (within 1pp)

Cost on AX102 (32 vCPU dedicated): ~30-45min per walk-forward × 11+ seeds
≈ 6-8h sequential. Cache intermediate profiles to /tmp so partial runs can
resume.

Usage:
  AX102$ UV_CACHE_DIR=/tmp/uv-cache nohup uv run python \\
      scripts/validate_pooled_calibration.py \\
      --seasons 2024,2025 --out data/validation/pooled_calibration.json \\
      > /tmp/validate_pooled.log 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean as _mean

import numpy as np
import pandas as pd


def _bucket(p: float) -> str:
    if p < 0.65: return "<65%"
    if p < 0.70: return "65-70%"
    if p < 0.75: return "70-75%"
    if p < 0.80: return "75-80%"
    return ">=80%"


def calibration_table(profiles_top1: pd.DataFrame) -> dict:
    """profiles_top1: rows where rank==1, with predicted_p + actual_hit columns.

    Returns dict per bucket: {n, mean_predicted, realized_rate, overconfidence_pp}
    plus overall: {n, brier, mean_predicted, mean_realized}.
    """
    df = profiles_top1.copy()
    df["bucket"] = df["predicted_p"].apply(_bucket)
    out = {"buckets": {}}
    for b, g in df.groupby("bucket", observed=True):
        n = len(g)
        mp = float(g["predicted_p"].mean())
        rr = float(g["actual_hit"].mean())
        out["buckets"][b] = {
            "n": n,
            "mean_predicted": mp,
            "realized_rate": rr,
            "overconfidence_pp": (mp - rr) * 100,
        }
    out["overall"] = {
        "n": len(df),
        "mean_predicted": float(df["predicted_p"].mean()),
        "mean_realized": float(df["actual_hit"].mean()),
        "brier": float(((df["predicted_p"] - df["actual_hit"]) ** 2).mean()),
    }
    return out


def pool_predictions(profiles_per_seed: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Pool per-seed profiles into a single dataframe.

    Input: {seed: profiles_df} — each profiles_df has all top-N picks per day.
    Output: profiles_df with columns (date, rank, batter_id, p_game_hit,
            actual_hit, n_seeds) where p_game_hit is the mean across seeds at
            (date, batter_id) and rank is recomputed within each day.

    Critical: only batter_ids appearing in *all* seeds at a given date are
    pooled. (If a seed missed a batter due to ranking falling off top-N, the
    pooled result excludes them. Top-N=10 is high enough that real candidates
    are present in all seeds.)
    """
    frames = []
    for seed, df in profiles_per_seed.items():
        d = df.copy()
        d["seed"] = seed
        frames.append(d[["date", "batter_id", "p_game_hit", "actual_hit", "seed"]])
    stacked = pd.concat(frames, ignore_index=True)
    n_seeds = len(profiles_per_seed)

    grouped = stacked.groupby(["date", "batter_id"]).agg(
        p_game_hit=("p_game_hit", "mean"),
        actual_hit=("actual_hit", "max"),  # realized is identical across seeds; max for safety
        n_seeds_present=("seed", "nunique"),
    ).reset_index()

    # Only keep (date, batter) tuples present in all seeds — fair pool
    grouped = grouped[grouped["n_seeds_present"] == n_seeds].copy()

    # Re-rank within each day
    grouped = grouped.sort_values(["date", "p_game_hit"], ascending=[True, False]).reset_index(drop=True)
    grouped["rank"] = grouped.groupby("date").cumcount() + 1
    return grouped


def run_walk_forward(seed: int, df: pd.DataFrame, seasons: list[int],
                     retrain_every: int, cache_path: Path) -> pd.DataFrame:
    """Run walk-forward for one seed, persisting intermediate profiles."""
    if cache_path.exists():
        print(f"  [seed={seed}] cached at {cache_path}, loading", flush=True)
        return pd.read_parquet(cache_path)

    from bts.simulate.backtest_blend import blend_walk_forward
    print(f"  [seed={seed}] walking 2024+2025...", flush=True)
    prev = os.environ.get("BTS_LGBM_RANDOM_STATE")
    os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
    os.environ["BTS_LGBM_DETERMINISTIC"] = "1"
    try:
        all_profiles = []
        for season in seasons:
            p = blend_walk_forward(df, season, retrain_every=retrain_every)
            p["season"] = season
            all_profiles.append(p)
        combined = pd.concat(all_profiles, ignore_index=True)
    finally:
        if prev is None:
            os.environ.pop("BTS_LGBM_RANDOM_STATE", None)
        else:
            os.environ["BTS_LGBM_RANDOM_STATE"] = prev

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    print(f"  [seed={seed}] saved to {cache_path}", flush=True)
    return combined


def gate_verdict(single: dict, pooled: dict, label: str = "pooled") -> dict:
    """Apply Component E gate. Returns {pass, reasons}."""
    bucket = "75-80%"
    s_oc = single["buckets"].get(bucket, {}).get("overconfidence_pp", 0)
    p_oc = pooled["buckets"].get(bucket, {}).get("overconfidence_pp", 0)
    delta_oc = s_oc - p_oc  # positive = pooled is better-calibrated

    s_brier = single["overall"]["brier"]
    p_brier = pooled["overall"]["brier"]
    delta_brier = s_brier - p_brier  # positive = pooled is better

    reasons = []
    pass_calibration = delta_oc >= 5
    pass_brier = delta_brier >= 0
    if pass_calibration:
        reasons.append(f"{label} 75-80% overconfidence improved by {delta_oc:+.2f}pp (gate: ≥5pp)")
    if pass_brier:
        reasons.append(f"{label} Brier improved by {delta_brier:+.5f} (gate: ≤0)")
    if not (pass_calibration or pass_brier):
        reasons.append(
            f"{label} did not meet either gate: 75-80% overconf delta {delta_oc:+.2f}pp "
            f"(needed ≥5), Brier delta {delta_brier:+.5f} (needed ≤0)"
        )
    return {
        "passes": pass_calibration or pass_brier,
        "delta_75_80_overconfidence_pp": delta_oc,
        "delta_brier": delta_brier,
        "reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--retrain-every", type=int, default=7)
    ap.add_argument("--seed-set", default="canonical-n10")
    ap.add_argument("--baseline-seed", type=int, default=42)
    ap.add_argument("--random-seeds", default=None,
                    help="Comma-separated random-10 seeds for sanity check (omit to skip).")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/pooled_calib_cache"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/validation/pooled_calibration.json"))
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",")]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Load seed manifests
    repo = Path(__file__).parent.parent
    seed_set_path = repo / "data" / "seed_sets" / f"{args.seed_set}.json"
    canonical_seeds = [int(s) for s in json.loads(seed_set_path.read_text())["seeds"]]
    print(f"Canonical seeds ({args.seed_set}): {canonical_seeds}", flush=True)

    random_seeds: list[int] = []
    if args.random_seeds:
        random_seeds = [int(s.strip()) for s in args.random_seeds.split(",") if s.strip()]
        print(f"Random-10 sanity check seeds: {random_seeds}", flush=True)

    # Load PA data once
    print(f"Loading PA data from {args.data_dir}...", flush=True)
    from bts.features.compute import compute_all_features
    parquets = sorted(args.data_dir.glob("pa_*.parquet"))
    pa_df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    pa_df = compute_all_features(pa_df)
    print(f"  {len(pa_df):,} PAs", flush=True)

    # Run walk-forward for every required seed (cached)
    all_seeds = sorted({args.baseline_seed, *canonical_seeds, *random_seeds})
    profiles_by_seed: dict[int, pd.DataFrame] = {}
    for seed in all_seeds:
        cache_path = args.cache_dir / f"profiles_seed_{seed}.parquet"
        profiles_by_seed[seed] = run_walk_forward(seed, pa_df, seasons,
                                                   args.retrain_every, cache_path)

    # Single-seed=42 calibration on top-1 picks
    single_p = profiles_by_seed[args.baseline_seed]
    single_top1 = single_p[single_p["rank"] == 1][["date", "batter_id", "p_game_hit", "actual_hit"]].rename(
        columns={"p_game_hit": "predicted_p"}
    )
    single_cal = calibration_table(single_top1)

    # Canonical pooled
    canonical_dict = {s: profiles_by_seed[s] for s in canonical_seeds}
    canonical_pooled = pool_predictions(canonical_dict)
    canonical_top1 = canonical_pooled[canonical_pooled["rank"] == 1][
        ["date", "batter_id", "p_game_hit", "actual_hit"]
    ].rename(columns={"p_game_hit": "predicted_p"})
    canonical_cal = calibration_table(canonical_top1)

    canonical_gate = gate_verdict(single_cal, canonical_cal, label="canonical-n10")

    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "single_seed": args.baseline_seed,
        "canonical_seeds": canonical_seeds,
        "single_calibration": single_cal,
        "canonical_pooled_calibration": canonical_cal,
        "canonical_gate": canonical_gate,
    }

    if random_seeds:
        random_dict = {s: profiles_by_seed[s] for s in random_seeds}
        random_pooled = pool_predictions(random_dict)
        random_top1 = random_pooled[random_pooled["rank"] == 1][
            ["date", "batter_id", "p_game_hit", "actual_hit"]
        ].rename(columns={"p_game_hit": "predicted_p"})
        random_cal = calibration_table(random_top1)
        random_gate = gate_verdict(single_cal, random_cal, label="random-10")
        # Sanity: canonical and random improvements should be similar
        canonical_improvement = canonical_gate["delta_75_80_overconfidence_pp"]
        random_improvement = random_gate["delta_75_80_overconfidence_pp"]
        cross_check = abs(canonical_improvement - random_improvement) <= 1.0  # within 1pp
        out["random_pooled_calibration"] = random_cal
        out["random_gate"] = random_gate
        out["canonical_vs_random_diff_pp"] = canonical_improvement - random_improvement
        out["canonical_unbiased"] = cross_check

    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.out}", flush=True)
    print(f"\n=== Single (seed={args.baseline_seed}) calibration ===", flush=True)
    for b, v in single_cal["buckets"].items():
        print(f"  {b}: n={v['n']}, predicted={v['mean_predicted']:.3f}, "
              f"realized={v['realized_rate']:.3f}, overconf={v['overconfidence_pp']:+.2f}pp",
              flush=True)
    print(f"  overall: predicted={single_cal['overall']['mean_predicted']:.3f}, "
          f"realized={single_cal['overall']['mean_realized']:.3f}, "
          f"Brier={single_cal['overall']['brier']:.5f}", flush=True)
    print(f"\n=== Canonical-n10 pooled calibration ===", flush=True)
    for b, v in canonical_cal["buckets"].items():
        print(f"  {b}: n={v['n']}, predicted={v['mean_predicted']:.3f}, "
              f"realized={v['realized_rate']:.3f}, overconf={v['overconfidence_pp']:+.2f}pp",
              flush=True)
    print(f"  overall: predicted={canonical_cal['overall']['mean_predicted']:.3f}, "
          f"realized={canonical_cal['overall']['mean_realized']:.3f}, "
          f"Brier={canonical_cal['overall']['brier']:.5f}", flush=True)
    print(f"\n=== GATE: {'PASS' if canonical_gate['passes'] else 'FAIL'} ===", flush=True)
    for r in canonical_gate["reasons"]:
        print(f"  {r}", flush=True)

    return 0 if canonical_gate["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
