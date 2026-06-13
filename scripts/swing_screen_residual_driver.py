#!/usr/bin/env python3
"""Stage-1 residual-stacking screen driver (amendment #3, Codex round-4).

Full-history production prior (production features only) cross-fit OOF onto
covered training rows + predicted onto eval rows; every arm then trains on
covered/warmed rows with [prod_prior, prior_daily_rank] + its swing columns.
The paired delta isolates swing value ON TOP of the full-strength production
model — no covered-only-baseline confound.

Windows (Codex): full prior train 2019-01..2024-06-30; covered swing-layer
train 2023-07-01..2024-06-30 with rolling_60g_swing_coverage>=0.90; eval
2024-07-01..season-end, same coverage gate. 2025/2026 untouched. 30 seeds.

Usage:
  UV_CACHE_DIR=/tmp/uv-cache nice -n 15 .venv/bin/python \
    scripts/swing_screen_residual_driver.py --out data/validation/swing_screen_residual
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bts.experiment.swing_screen import (  # noqa: E402
    ARMS, FAMILY_OF, build_arm_frame, build_prod_prior_oof, run_residual_arm,
)
from bts.features.compute import FEATURE_COLS, compute_all_features  # noqa: E402
from bts.features.swing import (  # noqa: E402
    attach_swing_features, daily_swing_aggregates,
    rolling_swing_features,
)

SEEDS = list(range(30))
COV_START, COV_END = "2023-07-01", "2024-06-30"
EVAL_START, EVAL_END = "2024-07-01", "2024-11-01"
MIN_COV60 = 0.90
# Soft-oracle reveal fraction — tuned via scripts/calibrate_soft_oracle.py to
# give ~+0.005 daily rank-AUC (the candidate effect size). Set 2026-06-13.
SOFT_REVEAL = float(os.environ.get("BTS_SOFT_REVEAL", "0.005"))  # ~+0.005 daily rank-AUC (calibrated 2026-06-13)


def _swing_coverage_60g(daily_b: pd.DataFrame) -> pd.DataFrame:
    """Per (batter,date): fraction of the prior 60 game-dates with tracked
    swings (shift(1) — leak-free)."""
    d = daily_b[["batter", "date", "n_swings_tracked"]].copy()
    d["_tracked"] = (d["n_swings_tracked"] > 0).astype(float)
    d = d.sort_values(["batter", "date"], kind="mergesort")
    d["swing_coverage_60g"] = (
        d.groupby("batter")["_tracked"].transform(lambda s: s.shift(1).rolling(60, min_periods=1).mean())
    )
    return d[["batter", "date", "swing_coverage_60g"]].rename(columns={"batter": "batter_id"})


def build_pa_frame() -> pd.DataFrame:
    proc = Path("data/processed")
    pa = pd.concat([pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))], ignore_index=True)
    pa = compute_all_features(pa)
    pa["date"] = pd.to_datetime(pa["date"])

    bronze = pd.concat([pd.read_parquet(p) for p in sorted(proc.glob("swing_*.parquet"))], ignore_index=True)
    daily_b = daily_swing_aggregates(bronze, entity="batter")
    daily_p = daily_swing_aggregates(bronze, entity="pitcher")
    pa = attach_swing_features(
        pa,
        batter_feats=rolling_swing_features(daily_b, entity="batter"),
        pitcher_feats=rolling_swing_features(daily_p, entity="pitcher"),
    )
    pa = pa.merge(_swing_coverage_60g(daily_b), on=["batter_id", "date"], how="left")
    pa["swing_coverage_60g"] = pa["swing_coverage_60g"].fillna(0.0)
    # gross canary = same-day game outcome (proven to explode 2026-06-13)
    pa["ORACLE_game_hit"] = pa.groupby(["game_pk", "batter_id"])["is_hit"].transform("max").astype(float)
    # soft-oracle canary = reveal the game outcome for a SOFT_REVEAL fraction of
    # rows, pure noise otherwise — a graded leak tuned to ~+0.005 (Codex r5).
    # Deterministic per row (hash of game_pk*1000+batter_id), no global RNG.
    h = ((pa["game_pk"].astype("int64") * 1009 + pa["batter_id"].astype("int64")) % 997) / 997.0
    revealed = h < SOFT_REVEAL
    noise = ((pa["game_pk"].astype("int64") * 7919 + pa["batter_id"].astype("int64")) % 1013) / 1013.0
    pa["SOFT_ORACLE"] = np.where(revealed, pa["ORACLE_game_hit"], noise)
    m3 = rolling_swing_features(daily_b, entity="batter", windows=[30], shift_days=0)
    m3 = m3[["batter", "date", "batter_miss_dist_30g"]].rename(
        columns={"batter": "batter_id", "batter_miss_dist_30g": "M3LEAK_batter_miss_dist_30g"})
    pa = pa.merge(m3, on=["batter_id", "date"], how="left")
    return pa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    print("building PA + swing frame...", flush=True)
    t0 = time.time()
    pa = build_pa_frame()
    pa["date"] = pd.to_datetime(pa["date"])
    full = (pa["date"] >= "2019-01-01") & (pa["date"] <= COV_END)
    covered_warm = (pa["swing_coverage_60g"] >= MIN_COV60)
    cov = (pa["date"] >= COV_START) & (pa["date"] <= COV_END) & covered_warm
    ev = (pa["date"] >= EVAL_START) & (pa["date"] <= EVAL_END) & covered_warm
    print(f"  frame ready in {time.time()-t0:.0f}s | full={full.sum()} cov={cov.sum()} eval={ev.sum()} "
          f"({ev.sum() and pa.loc[ev,'date'].dt.date.nunique()} eval days)", flush=True)

    arms = args.arms or ARMS
    seeds = args.seeds or SEEDS
    args.out.mkdir(parents=True, exist_ok=True)
    total = len(seeds) * len(arms)
    done = 0
    for seed in seeds:
        # prod_prior depends only on seed — build once, reuse across arms
        prior = build_prod_prior_oof(pa, FEATURE_COLS, full, cov, ev, seed=seed)
        pa_s = pa.assign(prod_prior=prior)
        for arm in arms:
            done += 1
            target = args.out / f"{arm}_seed{seed}.json"
            if target.exists():
                print(f"[{done}/{total}] skip {target.name}", flush=True)
                continue
            t1 = time.time()
            frame, swing_cols = build_arm_frame(arm, pa_s)
            swing_cols = swing_cols + ["swing_coverage_60g"]
            res = run_residual_arm(frame, [] if arm == "baseline" else swing_cols,
                                   cov, ev, seed=seed, arm_name=arm)
            res["family"] = FAMILY_OF[arm]
            target.write_text(json.dumps(res))
            print(f"[{done}/{total}] {arm} seed={seed} auc={res['auc_mean']:.4f} "
                  f"ndcg={res['ndcg_mean']:.4f} ({time.time()-t1:.0f}s)", flush=True)
    print("RESIDUAL DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
