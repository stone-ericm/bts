#!/usr/bin/env python3
"""Stage-1 screen driver: all arms x seeds, resumable. Run on bts-hetzner
overnight (nice'd; scheduler contention acceptable — runs are independent).

Usage:
  UV_CACHE_DIR=/tmp/uv-cache nice -n 15 .venv/bin/python \
      scripts/swing_screen_driver.py --out data/validation/swing_screen_2024
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bts.experiment.swing_screen import ARMS, run_screen_arm  # noqa: E402
from bts.features.compute import compute_all_features  # noqa: E402
from bts.features.swing import (  # noqa: E402
    attach_swing_features, build_gross_sentinel, daily_swing_aggregates,
    rolling_swing_features,
)

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]  # amendment #2: 10 common seeds
TRAIN_SEASONS = (2019, 2020, 2021, 2022, 2023)
SCREEN_SEASON = 2024
# Spec amendment 2026-06-12: 2024H1 joins training so swing features have
# learnable coverage (8.8% otherwise — the controls gate caught this);
# screen = 2024H2 only. 2025/2026 remain untouched for confirmation.
TRAIN_EXTRA_THROUGH = "2024-06-30"
SCREEN_START = "2024-07-01"


def build_pa_frame() -> pd.DataFrame:
    proc = Path("data/processed")
    pa = pd.concat(
        [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))],
        ignore_index=True,
    )
    pa = compute_all_features(pa)
    pa["date"] = pd.to_datetime(pa["date"])

    bronze = pd.concat(
        [pd.read_parquet(p) for p in sorted(proc.glob("swing_*.parquet"))],
        ignore_index=True,
    )
    daily_b = daily_swing_aggregates(bronze, entity="batter")
    daily_p = daily_swing_aggregates(bronze, entity="pitcher")
    pa = attach_swing_features(
        pa,
        batter_feats=rolling_swing_features(daily_b, entity="batter"),
        pitcher_feats=rolling_swing_features(daily_p, entity="pitcher"),
    )
    # sentinel columns (registry guards their use; amendment #2: two sentinels)
    pa = build_gross_sentinel(pa, daily_b, entity="batter")
    m3 = rolling_swing_features(daily_b, entity="batter", windows=[30], shift_days=0)
    m3 = m3[["batter", "date", "batter_miss_dist_30g"]].rename(
        columns={"batter": "batter_id", "batter_miss_dist_30g": "M3LEAK_batter_miss_dist_30g"}
    )
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
    print(f"  frame ready: {len(pa)} rows in {time.time()-t0:.0f}s", flush=True)

    arms = args.arms or ARMS
    seeds = args.seeds or SEEDS
    total = len(arms) * len(seeds)
    done = 0
    for arm in arms:
        for seed in seeds:
            done += 1
            target = args.out / f"{arm}_seed{seed}.json"
            if target.exists():
                print(f"[{done}/{total}] skip {target.name}", flush=True)
                continue
            t1 = time.time()
            res = run_screen_arm(
                arm=arm, pa=pa, train_seasons=TRAIN_SEASONS,
                screen_season=SCREEN_SEASON, seed=seed, out_dir=args.out,
                train_extra_through=TRAIN_EXTRA_THROUGH, screen_start=SCREEN_START,
            )
            print(f"[{done}/{total}] {arm} seed={seed} ndcg={res['ndcg_mean']:.4f} "
                  f"auc={res['auc']:.4f} ({time.time()-t1:.0f}s)", flush=True)
    print("DRIVER DONE", flush=True)


if __name__ == "__main__":
    main()
