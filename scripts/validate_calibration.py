#!/usr/bin/env python3
"""Cross-validated validation for post-hoc calibration on resolved BTS picks.

This is an offline gate, not production wiring. It answers one question:
does an isotonic map fitted on resolved pick outcomes improve a proper score
out of fold? If not, calibration remains off.

Example:
  PYTHONPATH=src uv run python scripts/validate_calibration.py \
    --picks-dir data/picks \
    --pa-parquet data/processed/pa_2026.parquet \
    --date 2026-05-23 \
    --lookback-days 90
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Sequence

import pandas as pd

from bts.model.calibrate import _resolve_pick_outcomes, apply_calibrator


DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_FOLDS = 5
DEFAULT_BOOTSTRAP_REPS = 1000
DEFAULT_MIN_N = 200
DEFAULT_SEED = 42
DEFAULT_BUCKETS = ((0.0, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.0))


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    brier_raw: float
    brier_calibrated: float
    improvement: float


def brier(predictions: Sequence[float], outcomes: Sequence[int]) -> float:
    if not predictions:
        return float("nan")
    return mean((p - y) ** 2 for p, y in zip(predictions, outcomes))


def sample_summary(samples: Sequence[tuple[float, int]]) -> dict:
    if not samples:
        return {
            "n": 0,
            "raw_mean_p": None,
            "hit_rate": None,
            "buckets": [],
        }

    buckets = []
    for low, high in DEFAULT_BUCKETS:
        cell = [(p, y) for p, y in samples if low <= p < high]
        if not cell:
            continue
        buckets.append({
            "low": low,
            "high": high,
            "n": len(cell),
            "mean_p": mean(p for p, _ in cell),
            "hit_rate": mean(y for _, y in cell),
        })

    return {
        "n": len(samples),
        "raw_mean_p": mean(p for p, _ in samples),
        "hit_rate": mean(y for _, y in samples),
        "buckets": buckets,
    }


def load_samples(
    picks_dir: Path,
    pa_parquet: Path,
    today: date,
    lookback_days: int,
) -> list[tuple[float, int]]:
    pa_df = pd.read_parquet(pa_parquet)
    return _resolve_pick_outcomes(picks_dir, pa_df, today, lookback_days)


def cross_validate_isotonic(
    samples: Sequence[tuple[float, int]],
    *,
    n_folds: int = DEFAULT_FOLDS,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
    min_n: int = DEFAULT_MIN_N,
) -> dict:
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError as e:
        return {
            "decision": "UNAVAILABLE",
            "reason": f"scikit-learn unavailable: {e}",
            "n_samples": len(samples),
        }

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be >= 1")
    if len(samples) < n_folds * 5:
        return {
            "decision": "INSUFFICIENT_FOLDS",
            "reason": f"need at least {n_folds * 5} samples for {n_folds}-fold CV",
            "n_samples": len(samples),
            "min_n": min_n,
        }

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    fold_size = len(indices) // n_folds
    fold_results: list[FoldResult] = []
    all_test_raw: list[float] = []
    all_test_calibrated: list[float] = []
    all_test_y: list[int] = []

    for fold_idx in range(n_folds):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < n_folds - 1 else len(indices)
        test_indices = set(indices[test_start:test_end])
        train = [s for i, s in enumerate(samples) if i not in test_indices]
        test = [s for i, s in enumerate(samples) if i in test_indices]

        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit([p for p, _ in train], [y for _, y in train])

        test_raw = [p for p, _ in test]
        test_y = [y for _, y in test]
        test_calibrated = [apply_calibrator(p, calibrator) for p in test_raw]

        b_raw = brier(test_raw, test_y)
        b_cal = brier(test_calibrated, test_y)
        fold_results.append(FoldResult(
            fold=fold_idx,
            n_train=len(train),
            n_test=len(test),
            brier_raw=b_raw,
            brier_calibrated=b_cal,
            improvement=b_raw - b_cal,
        ))
        all_test_raw.extend(test_raw)
        all_test_calibrated.extend(test_calibrated)
        all_test_y.extend(test_y)

    b_raw_all = brier(all_test_raw, all_test_y)
    b_cal_all = brier(all_test_calibrated, all_test_y)
    improvement = b_raw_all - b_cal_all

    boot_improvements = []
    rng_boot = random.Random(seed + 1)
    n = len(all_test_raw)
    for _ in range(bootstrap_reps):
        idx = [rng_boot.randrange(n) for _ in range(n)]
        b_raw = mean((all_test_raw[i] - all_test_y[i]) ** 2 for i in idx)
        b_cal = mean((all_test_calibrated[i] - all_test_y[i]) ** 2 for i in idx)
        boot_improvements.append(b_raw - b_cal)
    boot_improvements.sort()
    ci_low = boot_improvements[int(0.025 * bootstrap_reps)]
    ci_high = boot_improvements[int(0.975 * bootstrap_reps)]
    pct_positive = sum(1 for x in boot_improvements if x > 0) / len(boot_improvements)

    if len(samples) < min_n:
        decision = "WAIT_FOR_N"
        reason = f"n={len(samples)} below min_n={min_n}"
    elif improvement > 0 and ci_low > 0:
        decision = "SHIP"
        reason = "Brier improvement is positive and 95% bootstrap CI excludes zero"
    elif improvement > 0:
        decision = "MARGINAL"
        reason = "Brier improvement is positive but 95% bootstrap CI includes zero"
    else:
        decision = "DROP"
        reason = "cross-fitted isotonic does not improve Brier score"

    return {
        "decision": decision,
        "reason": reason,
        "n_samples": len(samples),
        "min_n": min_n,
        "n_folds": n_folds,
        "bootstrap_reps": bootstrap_reps,
        "seed": seed,
        "brier_raw": b_raw_all,
        "brier_calibrated": b_cal_all,
        "improvement": improvement,
        "bootstrap_ci_95": [ci_low, ci_high],
        "pct_positive_bootstrap": pct_positive,
        "fold_results": [asdict(r) for r in fold_results],
    }


def run_validation(
    *,
    picks_dir: Path,
    pa_parquet: Path,
    today: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    n_folds: int = DEFAULT_FOLDS,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_SEED,
    min_n: int = DEFAULT_MIN_N,
) -> dict:
    samples = load_samples(picks_dir, pa_parquet, today, lookback_days)
    cv = cross_validate_isotonic(
        samples,
        n_folds=n_folds,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
        min_n=min_n,
    )
    return {
        "date": today.isoformat(),
        "picks_dir": str(picks_dir),
        "pa_parquet": str(pa_parquet),
        "lookback_days": lookback_days,
        "sample_summary": sample_summary(samples),
        "cross_validation": cv,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks-dir", type=Path, default=Path("data/picks"))
    parser.add_argument("--pa-parquet", type=Path, default=None)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    today = date.fromisoformat(args.date)
    pa_parquet = args.pa_parquet or Path(f"data/processed/pa_{today.year}.parquet")
    output = args.output or Path(f"data/validation/calibration_validation_{today.isoformat()}.json")

    result = run_validation(
        picks_dir=args.picks_dir,
        pa_parquet=pa_parquet,
        today=today,
        lookback_days=args.lookback_days,
        n_folds=args.folds,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        min_n=args.min_n,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    summary = result["sample_summary"]
    cv = result["cross_validation"]
    print(f"Loaded {summary['n']} resolved pick samples")
    print(
        f"Raw mean={summary['raw_mean_p']:.4f} hit_rate={summary['hit_rate']:.4f}"
        if summary["n"] else "No resolved samples"
    )
    print(
        f"Decision={cv['decision']} improvement={cv.get('improvement')} "
        f"CI={cv.get('bootstrap_ci_95')}"
    )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
