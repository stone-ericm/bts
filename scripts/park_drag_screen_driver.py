#!/usr/bin/env python3
"""park_drag 2026 backtest driver: all arms x seeds, resumable.

Usage:
  UV_CACHE_DIR=/tmp/uv-cache nice -n 10 uv run python \
      scripts/park_drag_screen_driver.py --out data/validation/park_drag_screen_2026

Train = 2019-2025 + 2026 through TRAIN_EXTRA_THROUGH (early-season coverage
amendment, mirrors the swing screen); screen slate = SCREEN_START..data end.
The report breaks out the post-May-24 window (league drag change-point).
POST-SELECTION CAVEAT: 2026 motivated the feature — supporting evidence only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# the export must resolve identically for compute_all_features and the
# expanding-variant attach below
os.environ.setdefault("BTS_PARK_DRAG_TABLE", "data/external/park_drag/park_drag_export.csv")

from bts.experiment.park_drag_screen import (  # noqa: E402
    ARMS, rolling_outcome_pf, run_screen_arm,
)
from bts.features import park_drag  # noqa: E402
from bts.features.compute import FEATURE_COLS, compute_all_features  # noqa: E402

SEEDS = [42, 101, 202, 303, 404]
TRAIN_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
SCREEN_SEASON = 2026
TRAIN_EXTRA_THROUGH = os.environ.get("BTS_PD_TRAIN_EXTRA", "2026-04-30")
SCREEN_START = os.environ.get("BTS_PD_SCREEN_START", "2026-05-01")
SOFT_REVEAL = float(os.environ.get("BTS_SOFT_REVEAL", "0.005"))

SLATE_COLS = ["date", "season", "game_pk", "batter_id", "lineup_position",
              "is_hit", "venue_id"]


def build_pa_frame() -> pd.DataFrame:
    proc = Path("data/processed")
    seasons = list(TRAIN_SEASONS) + [SCREEN_SEASON]
    pa = pd.concat(
        [pd.read_parquet(proc / f"pa_{s}.parquet") for s in seasons],
        ignore_index=True,
    )
    park_drag._reset_cache()
    pa = compute_all_features(pa)
    pa["date"] = pd.to_datetime(pa["date"])
    assert pa["park_drag_delta"].notna().any(), (
        "park_drag_delta all-NaN — check BTS_PARK_DRAG_TABLE resolves the export")

    export = park_drag.load_table()
    pa = pa.merge(
        export[["venue_id", "date", "park_drag_delta_expanding"]],
        on=["venue_id", "date"], how="left",
    )
    # M3-style same-day-leak canary: export row at date+1 includes date's games
    leak = export[["venue_id", "date", "park_drag_delta"]].copy()
    leak["date"] = leak["date"] - pd.Timedelta(days=1)
    leak = leak.rename(columns={"park_drag_delta": "LEAK_pd_next_date"})
    pa = pa.merge(leak, on=["venue_id", "date"], how="left")

    pa = pa.merge(rolling_outcome_pf(pa), on=["venue_id", "date"], how="left")

    # sentinels (deterministic hash reveal — mirrors swing residual driver)
    pa["ORACLE_game_hit"] = (
        pa.groupby(["game_pk", "batter_id"])["is_hit"].transform("max").astype(float))
    h = ((pa["game_pk"].astype("int64") * 1009
          + pa["batter_id"].astype("int64")) % 997) / 997.0
    noise = ((pa["game_pk"].astype("int64") * 7919
              + pa["batter_id"].astype("int64")) % 1013) / 1013.0
    pa["SOFT_ORACLE"] = np.where(h < SOFT_REVEAL, pa["ORACLE_game_hit"], noise)

    keep = list(dict.fromkeys(
        SLATE_COLS + FEATURE_COLS
        + ["park_drag_delta", "park_drag_delta_expanding", "rolling_outcome_pf",
           "ORACLE_game_hit", "SOFT_ORACLE", "LEAK_pd_next_date"]))
    return pa[[c for c in keep if c in pa.columns]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    arms = args.arms or ARMS
    seeds = args.seeds or SEEDS

    t0 = time.time()
    print("building PA frame (features + variants + sentinels)...", flush=True)
    pa = build_pa_frame()
    print(f"frame ready: {len(pa)} rows, {pa.date.max().date()} max date, "
          f"{time.time() - t0:.0f}s", flush=True)
    print(f"park_drag_delta coverage 2026: "
          f"{pa.loc[pa.season == 2026, 'park_drag_delta'].notna().mean():.2%}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for arm in arms:
            out_file = args.out / f"{arm}_seed{seed}.json"
            if out_file.exists():
                print(f"skip {arm} seed {seed} (exists)", flush=True)
                continue
            t1 = time.time()
            res = run_screen_arm(
                arm, pa, TRAIN_SEASONS, SCREEN_SEASON, seed, out_dir=args.out,
                train_extra_through=TRAIN_EXTRA_THROUGH, screen_start=SCREEN_START,
            )
            print(f"{arm} seed {seed}: day_auc-mean via report; "
                  f"auc={res['auc']:.4f} top1={res['top1_hit']:.3f} "
                  f"({time.time() - t1:.0f}s)", flush=True)
    print(f"ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
