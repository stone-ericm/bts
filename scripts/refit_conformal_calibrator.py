#!/usr/bin/env python3
"""scripts/refit_conformal_calibrator.py — daily refresh of conformal calibrators.

Loads 2025+2026 backtest profiles (from `bts simulate backtest` outputs),
fits LR classifier (predicting P(year==2026 | features)), fits both calibrators
(WeightedMondrianConformalCalibrator + BucketWilsonCalibrator), serializes
dated artifacts to `data/conformal/`, and appends a row to validation_log.jsonl.

Run by the daily systemd timer (deploy/systemd/bts-conformal-refit.timer).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/refit_conformal_calibrator.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    from bts.model.conformal import (
        fit_bucket_wilson_calibrator,
        fit_lr_classifier,
        compute_lr_weights,
        fit_weighted_mondrian_conformal_calibrator,
    )

    today = date.today().isoformat()
    out_dir = REPO / "data" / "conformal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 2025 + 2026 backtest profiles
    profiles = []
    for season in (2025, 2026):
        path = REPO / "data" / "simulation" / f"backtest_{season}.parquet"
        if not path.exists():
            print(f"WARNING: {path} not found; skipping season {season}", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        df["season"] = season
        profiles.append(df)
    if not profiles:
        print("ERROR: no backtest profiles found", file=sys.stderr)
        sys.exit(1)
    cal = pd.concat(profiles, ignore_index=True)
    print(f"loaded calibration: {len(cal)} rows across seasons {sorted(cal['season'].unique().tolist())}", file=sys.stderr)

    # Build features for LR classifier
    # Backtest profile only has (date, rank, batter_id, p_game_hit, actual_hit, n_pas).
    # Join to PA frame to get the 16 FEATURE_COLS for each row.
    # Simpler approach for v1: use just (p_game_hit, n_pas, rank) as features.
    # The LR classifier is purely for covariate-shift correction — feature set
    # need not be the full FEATURE_COLS, just enough to capture distributional
    # differences between 2025 and 2026 predictions.
    cal_features = cal[["p_game_hit", "rank", "n_pas"]].copy()
    years = cal["season"].values

    print(f"fitting LR classifier (P(year==2026 | features))...", file=sys.stderr)
    lr_clf = fit_lr_classifier(cal_features, years, target_year=2026)
    weights = compute_lr_weights(lr_clf, cal_features)
    print(f"  LR weights: mean={weights.mean():.3f} median={np.median(weights):.3f} "
          f"p5={np.quantile(weights, 0.05):.3f} p95={np.quantile(weights, 0.95):.3f}",
          file=sys.stderr)

    # Fit weighted Mondrian conformal calibrator
    print("fitting WeightedMondrianConformalCalibrator...", file=sys.stderr)
    conformal_cal = fit_weighted_mondrian_conformal_calibrator(
        predicted_p=cal["p_game_hit"].values,
        actual_hit=cal["actual_hit"].values,
        weights=weights,
        alphas=[0.05, 0.10, 0.20],
    )

    # Fit bucket Wilson calibrator (no weighting; pure frequentist)
    print("fitting BucketWilsonCalibrator...", file=sys.stderr)
    pairs = list(zip(cal["p_game_hit"].tolist(), cal["actual_hit"].tolist()))
    wilson_cal = fit_bucket_wilson_calibrator(pairs, alphas=[0.05, 0.10, 0.20])

    # Serialize all three artifacts
    joblib.dump(conformal_cal, out_dir / f"calibrator_{today}.pkl")
    joblib.dump(wilson_cal, out_dir / f"wilson_calibrator_{today}.pkl")
    joblib.dump(lr_clf, out_dir / f"lr_classifier_{today}.pkl")
    print(f"wrote 3 artifacts to {out_dir}", file=sys.stderr)

    # Append validation log row
    log_path = out_dir / "validation_log.jsonl"
    log_entry = {
        "fit_date": today,
        "n_calibration": len(cal),
        "n_effective_after_lr_weighting": float(weights.sum()),
        "lr_weight_summary": {
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
            "p95": float(np.quantile(weights, 0.95)),
            "p5": float(np.quantile(weights, 0.05)),
        },
        "alphas": [0.05, 0.10, 0.20],
        # Note: marginal_coverage / bucket_coverage / median_interval_width
        # are computed by validate_conformal.py separately. The refit script
        # only logs FIT-time diagnostics; coverage diagnostics come from CV.
        "n_buckets_populated_conformal": len(conformal_cal.bucket_quantiles),
        "n_buckets_populated_wilson": len(wilson_cal.bucket_lower),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"appended row to {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
