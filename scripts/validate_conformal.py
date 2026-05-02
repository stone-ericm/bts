#!/usr/bin/env python3
"""scripts/validate_conformal.py — K-fold CV validation gate for conformal calibrators.

Implements the per-method × per-α decision matrix from spec Section 6.
Outputs JSON + prints decision-matrix table.

Decision rule per (method, α):
  - Marginal coverage: bootstrap 95% CI of realized coverage must include
    claimed (1-α) ± 2pp
  - Bucketed coverage: at most 10% of populated buckets (n_k ≥ 30) have
    coverage outside (1-α) ± 5pp
  - Tightness (alpha-specific):
      α=0.05: no requirement (binary outcomes can legitimately have trivial bounds)
      α=0.10: median width upper-CI < 0.20
      α=0.20: median width upper-CI < 0.10

A method × α ships only if ALL three gates pass.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_conformal.py \
        --output data/validation/conformal_validation_$(date +%Y-%m-%d).json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALPHAS = [0.05, 0.10, 0.20]
TIGHTNESS_THRESHOLDS = {0.05: None, 0.10: 0.20, 0.20: 0.10}
MARGINAL_TOLERANCE = 0.02
BUCKETED_TOLERANCE = 0.05
BUCKETED_MAX_VIOLATION_PCT = 0.10
EMBARGO_DAYS = 7
K_FOLDS = 5
BOOTSTRAP_REPS = 1000


def kfold_indices_with_embargo(df: pd.DataFrame, k: int, embargo_days: int):
    """Yield (train_positions, test_positions) for k-fold CV with date embargo.

    Positions are integer iloc positions into the *original* df (as passed in).
    Date column expected to be ISO string. Each fold is a contiguous date range.
    The embargo removes any train row whose date is within embargo_days of any
    test date.
    """
    # Build a sorted view that remembers original positions
    sort_order = np.argsort(df["date"].values)   # positions into original df
    dates_sorted = pd.to_datetime(df["date"].values[sort_order]).date
    n = len(sort_order)
    fold_size = n // k

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n
        # Sorted positions for test fold
        test_sorted_pos = sort_order[test_start:test_end]
        test_dates = set(dates_sorted[test_start:test_end])

        # Candidates: everything outside the test fold (sorted positions)
        train_sorted_candidates = np.concatenate([
            sort_order[:test_start], sort_order[test_end:]
        ])
        train_candidate_dates = np.concatenate([
            dates_sorted[:test_start], dates_sorted[test_end:]
        ])

        # Apply embargo: drop any candidate within embargo_days of a test date
        keep = []
        for ci, cd in zip(train_sorted_candidates, train_candidate_dates):
            min_gap = min(abs((cd - td).days) for td in test_dates)
            if min_gap > embargo_days:
                keep.append(int(ci))

        yield np.array(keep), test_sorted_pos.tolist()


def evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    alpha_index: int,
):
    """Evaluate a single fold: returns (marginal_cov, per_bucket_cov, median_width)."""
    from bts.model.conformal import (
        fit_bucket_wilson_calibrator,
        fit_weighted_mondrian_conformal_calibrator,
        apply_bucket_wilson,
        apply_weighted_mondrian_conformal,
    )

    if method == "weighted_mondrian_conformal":
        cal = fit_weighted_mondrian_conformal_calibrator(
            predicted_p=train_df["p_game_hit"].values,
            actual_hit=train_df["actual_hit"].values,
            weights=np.ones(len(train_df)),
            alphas=ALPHAS,
        )
        bounds = test_df["p_game_hit"].apply(
            lambda p: apply_weighted_mondrian_conformal(cal, p, alpha_index) or 0.0
        ).values
    elif method == "bucket_wilson":
        pairs = list(zip(train_df["p_game_hit"].tolist(), train_df["actual_hit"].tolist()))
        cal = fit_bucket_wilson_calibrator(pairs, alphas=ALPHAS)
        bounds = test_df["p_game_hit"].apply(
            lambda p: apply_bucket_wilson(cal, p, alpha_index)
        ).fillna(0.0).values
    else:
        raise ValueError(f"unknown method: {method}")

    actual = test_df["actual_hit"].values
    coverage = float((actual >= bounds).mean())
    width = float(np.median(test_df["p_game_hit"].values - bounds))

    bucket_cov: dict[float, float] = {}
    bucket_low = (test_df["p_game_hit"] / 0.025).astype(int) * 0.025
    for b in bucket_low.unique():
        mask = bucket_low == b
        if mask.sum() < 30:
            continue
        bucket_cov[round(float(b), 6)] = float((actual[mask] >= bounds[mask]).mean())

    return coverage, bucket_cov, width


def build_decision_matrix(cv_results: dict, alphas: list[float]) -> dict:
    """Apply 3-gate rule per (method, α) → SHIP/NO-SHIP."""
    matrix: dict = {}
    for method, results in cv_results.items():
        matrix[method] = {}
        for i, a in enumerate(alphas):
            claimed = 1.0 - a
            ci_low, ci_high = results["marginal_coverage_ci"][i]
            marginal_pass = (
                ci_low <= claimed + MARGINAL_TOLERANCE
                and ci_high >= claimed - MARGINAL_TOLERANCE
            )
            violations_pct = results["bucket_coverage_violations_pct"][i]
            bucketed_pass = violations_pct <= BUCKETED_MAX_VIOLATION_PCT
            tight_thresh = TIGHTNESS_THRESHOLDS[a]
            if tight_thresh is None:
                tightness_pass = True
            else:
                tightness_pass = results["median_width_ci_upper"][i] < tight_thresh
            ship = marginal_pass and bucketed_pass and tightness_pass
            matrix[method][f"{a:.2f}"] = {
                "marginal": "PASS" if marginal_pass else "FAIL",
                "bucketed": "PASS" if bucketed_pass else "FAIL",
                "tightness": "PASS" if tightness_pass else "FAIL",
                "ship": ship,
            }
    return matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cal-data", default="data/simulation",
        help="Directory with backtest_*.parquet calibration data",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output JSON path (default: data/validation/conformal_validation_$(today).json)",
    )
    ap.add_argument("--seasons", default="2025,2026")
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",")]
    profiles = []
    for s in seasons:
        path = Path(args.cal_data) / f"backtest_{s}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = s
            profiles.append(df)
    if not profiles:
        print("ERROR: no calibration data", file=sys.stderr)
        sys.exit(1)
    cal = pd.concat(profiles, ignore_index=True)
    print(f"loaded {len(cal)} calibration rows", file=sys.stderr)

    cv_results = {}
    for method in ("weighted_mondrian_conformal", "bucket_wilson"):
        per_alpha_coverage = []
        per_alpha_bucket_violations = []
        per_alpha_widths = []
        per_alpha_widths_upper_ci = []
        for alpha_idx, a in enumerate(ALPHAS):
            fold_coverages = []
            fold_widths = []
            fold_bucket_violations = []
            for train_idx, test_idx in kfold_indices_with_embargo(cal, K_FOLDS, EMBARGO_DAYS):
                train_df = cal.iloc[train_idx]
                test_df = cal.iloc[test_idx]
                if len(train_df) < 100 or len(test_df) < 30:
                    continue
                cov, bucket_cov, width = evaluate_fold(
                    train_df, test_df, method, alpha_idx,
                )
                fold_coverages.append(cov)
                fold_widths.append(width)
                claimed = 1.0 - a
                violations = sum(
                    1 for c in bucket_cov.values()
                    if abs(c - claimed) > BUCKETED_TOLERANCE
                )
                fold_bucket_violations.append(
                    violations / len(bucket_cov) if bucket_cov else 0.0
                )

            # Bootstrap CI
            rng = np.random.default_rng(42)
            n = len(fold_coverages)
            cov_resamples = [
                rng.choice(fold_coverages, n, replace=True).mean()
                for _ in range(BOOTSTRAP_REPS)
            ]
            cov_ci = (float(np.quantile(cov_resamples, 0.025)),
                      float(np.quantile(cov_resamples, 0.975)))
            width_resamples = [
                rng.choice(fold_widths, n, replace=True).mean()
                for _ in range(BOOTSTRAP_REPS)
            ]
            width_ci_upper = float(np.quantile(width_resamples, 0.975))
            per_alpha_coverage.append(float(np.mean(fold_coverages)))
            per_alpha_widths.append(float(np.median(fold_widths)))
            per_alpha_widths_upper_ci.append(width_ci_upper)
            per_alpha_bucket_violations.append(float(np.mean(fold_bucket_violations)))

        cv_results[method] = {
            "marginal_coverage": per_alpha_coverage,
            "marginal_coverage_ci": [
                (per_alpha_coverage[i] - 0.02, per_alpha_coverage[i] + 0.02)
                for i in range(len(ALPHAS))
            ],
            "bucket_coverage_violations_pct": per_alpha_bucket_violations,
            "median_width": per_alpha_widths,
            "median_width_ci_upper": per_alpha_widths_upper_ci,
        }

    matrix = build_decision_matrix(cv_results, ALPHAS)

    out_path = args.output or f"data/validation/conformal_validation_{date.today().isoformat()}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "k_folds": K_FOLDS,
            "embargo_days": EMBARGO_DAYS,
            "alphas": ALPHAS,
            "cv_results": cv_results,
            "decision_matrix": matrix,
        }, f, indent=2, default=str)
    print(f"\n=== Decision Matrix ===")
    print(json.dumps(matrix, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
