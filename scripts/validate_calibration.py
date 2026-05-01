#!/usr/bin/env python3
"""K-fold cross-validation of post-hoc calibration on 2026 production picks.

Per the 2026-04-16 rejection lesson (analytical evaluator showed +1.14pp
P(57) but MC bootstrap showed −1.12pp), we validate calibration via
empirical bootstrap, NOT analytical evaluators that compare different
bin partitions.

This script does:
  1. Load 2026 resolved picks + pa_2026 day-hit lookup
  2. K-fold CV: fit calibrator on K-1 folds, evaluate on held-out fold
  3. Compare Brier score of raw vs calibrated probabilities
  4. Bootstrap CI on the Brier improvement

Decision rule: ship calibration if mean Brier improvement is positive
across folds AND its 95% bootstrap CI excludes zero.

Usage:
  uv run python scripts/validate_calibration.py
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import pandas as pd

from bts.model.calibrate import (
    _resolve_pick_outcomes,
    fit_calibrator_from_picks,
    apply_calibrator,
)


REPO = Path("/Users/stone/projects/bts")
PICKS_DIR = Path("/tmp/picks_2026")  # populated via rsync from bts-hetzner
PA_PARQUET = REPO / "data/processed/pa_2026.parquet"
TODAY = date(2026, 5, 1)
LOOKBACK = 60  # large enough to get the full 2026 sample
N_FOLDS = 5
BOOTSTRAP_REPS = 1000


def brier(predictions: list[float], outcomes: list[int]) -> float:
    if not predictions:
        return float("nan")
    return mean((p - y) ** 2 for p, y in zip(predictions, outcomes))


def main():
    pa_df = pd.read_parquet(PA_PARQUET)
    samples = _resolve_pick_outcomes(PICKS_DIR, pa_df, TODAY, LOOKBACK)
    print(f"Loaded {len(samples)} resolved picks (2026-{TODAY.isoformat()})")
    if len(samples) < N_FOLDS * 5:
        print(f"WARNING: insufficient samples for {N_FOLDS}-fold CV; need at least {N_FOLDS * 5}")
        return

    # Shuffle deterministically
    rng = random.Random(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    fold_size = len(samples) // N_FOLDS
    fold_results = []

    for fold_idx in range(N_FOLDS):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < N_FOLDS - 1 else len(samples)
        test_indices = set(indices[test_start:test_end])
        train_samples = [s for i, s in enumerate(samples) if i not in test_indices]
        test_samples = [s for i, s in enumerate(samples) if i in test_indices]

        # Train calibrator on train fold (using sklearn IsotonicRegression directly to avoid
        # re-reading from disk)
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            print("scikit-learn not available; cannot validate")
            return
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        train_xs = [s[0] for s in train_samples]
        train_ys = [s[1] for s in train_samples]
        cal.fit(train_xs, train_ys)

        # Evaluate on test fold
        test_raw = [s[0] for s in test_samples]
        test_outcomes = [s[1] for s in test_samples]
        test_calibrated = [apply_calibrator(p, cal) for p in test_raw]

        b_raw = brier(test_raw, test_outcomes)
        b_cal = brier(test_calibrated, test_outcomes)
        improvement = b_raw - b_cal  # positive = calibration improves
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
            "brier_raw": b_raw,
            "brier_cal": b_cal,
            "improvement": improvement,
        })
        print(
            f"  Fold {fold_idx}: n_test={len(test_samples)} "
            f"Brier raw={b_raw:.4f}, cal={b_cal:.4f}, improvement={improvement:+.4f}"
        )

    # Aggregate
    improvements = [f["improvement"] for f in fold_results]
    mean_improvement = mean(improvements)
    print(f"\nMean Brier improvement (positive = calibration wins): {mean_improvement:+.4f}")

    # Bootstrap CI on the improvement across all test predictions
    # Combine all test predictions across folds (each pick is in exactly one test fold)
    all_test_raw = []
    all_test_cal = []
    all_test_y = []
    for fold_idx in range(N_FOLDS):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < N_FOLDS - 1 else len(samples)
        test_indices = set(indices[test_start:test_end])
        train_samples = [s for i, s in enumerate(samples) if i not in test_indices]
        test_samples = [s for i, s in enumerate(samples) if i in test_indices]

        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit([s[0] for s in train_samples], [s[1] for s in train_samples])
        for raw, y in test_samples:
            all_test_raw.append(raw)
            all_test_cal.append(apply_calibrator(raw, cal))
            all_test_y.append(y)

    # Bootstrap
    rng2 = random.Random(0)
    n = len(all_test_raw)
    boot_improvements = []
    for _ in range(BOOTSTRAP_REPS):
        idx = [rng2.randrange(n) for _ in range(n)]
        b_raw = mean((all_test_raw[i] - all_test_y[i]) ** 2 for i in idx)
        b_cal = mean((all_test_cal[i] - all_test_y[i]) ** 2 for i in idx)
        boot_improvements.append(b_raw - b_cal)
    boot_improvements.sort()
    ci_low = boot_improvements[int(0.025 * BOOTSTRAP_REPS)]
    ci_high = boot_improvements[int(0.975 * BOOTSTRAP_REPS)]
    pct_positive = sum(1 for x in boot_improvements if x > 0) / len(boot_improvements)
    print(f"Bootstrap n={BOOTSTRAP_REPS}: 95% CI on improvement = [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"Pct of bootstrap reps showing positive improvement: {pct_positive:.1%}")

    print("\n=== Decision ===")
    if mean_improvement > 0 and ci_low > 0:
        print(f"✅ SHIP calibration: mean improvement {mean_improvement:+.4f}, CI excludes zero")
    elif mean_improvement > 0:
        print(f"⚠️ MARGINAL: mean improvement {mean_improvement:+.4f} positive but CI [{ci_low:+.4f}, {ci_high:+.4f}] includes zero")
        print(f"   Recommendation: defer until larger sample (currently n={len(samples)})")
    else:
        print(f"❌ DROP: mean improvement {mean_improvement:+.4f} negative or zero")

    # Save full output
    out_path = REPO / f"data/validation/calibration_validation_{TODAY.isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_samples": len(samples),
        "n_folds": N_FOLDS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "fold_results": fold_results,
        "mean_improvement": mean_improvement,
        "bootstrap_ci_95": [ci_low, ci_high],
        "pct_positive_bootstrap": pct_positive,
        "decision": (
            "SHIP" if mean_improvement > 0 and ci_low > 0
            else "MARGINAL" if mean_improvement > 0
            else "DROP"
        ),
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
