"""Conformal lower-bound gate v2 — SOTA tracker item #11 phase 0/1.

Replaces the broken-for-binary-y per-row coverage metric in the parked
`scripts/validate_conformal.py` with **lower-bound calibration validity**:
for each bucket of predictions, test whether the observed hit-rate's
Wilson lower CI is at least the calibrator's claimed mean lower bound
for that bucket (minus a small tolerance).

Composes #5 manifest-with-lockbox (calibrator fitted on fold train,
evaluated on fold holdout, lockbox held out) and #12 reliability/Brier
machinery (used as DIAGNOSTICS, not pass/fail gates) per Codex bus #86.

Tightness gate from the parked work is preserved: median bound width
must be <= a configurable threshold (default 0.30).

Output schema: `conformal_validation_v2` with manifest_metadata, full
method×alpha matrix, per-fold results, ship_set, explicit verdict.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bts.model.conformal import (
    BucketWilsonCalibrator,
    WeightedMondrianConformalCalibrator,
    _bucket_low,
    apply_bucket_wilson,
    apply_weighted_mondrian_conformal,
    fit_bucket_wilson_calibrator,
    fit_weighted_mondrian_conformal_calibrator,
    wilson_lower_one_sided,
    DEFAULT_BUCKET_WIDTH,
)


GATE_SCHEMA_VERSION = "conformal_validation_v2"
DEFAULT_ALPHAS = (0.05, 0.10, 0.20)
DEFAULT_METHODS = ("bucket_wilson", "weighted_mondrian_conformal")

# Gate thresholds (configurable via run_gate_matrix kwargs).
DEFAULT_VALIDITY_TOLERANCE = 0.01
DEFAULT_TIGHTNESS_THRESHOLD = 0.30
DEFAULT_MIN_BUCKET_N = 30
DEFAULT_WILSON_ALPHA = 0.05  # for the OBSERVED-rate Wilson CI in the gate test


def evaluate_per_bucket_validity(
    p_pred: np.ndarray,
    bounds: np.ndarray,
    actual: np.ndarray,
    *,
    bucket_width: float = DEFAULT_BUCKET_WIDTH,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
    validity_tolerance: float = DEFAULT_VALIDITY_TOLERANCE,
    wilson_alpha: float = DEFAULT_WILSON_ALPHA,
) -> dict:
    """Per-bucket lower-bound calibration validity test.

    For each bucket (defined by `_bucket_low(p_pred, bucket_width)`), compute:
    - observed hit-rate Wilson lower CI (one-sided at `wilson_alpha`)
    - mean / p90 / max of the calibrator's claimed lower bounds in that bucket

    Test passes the bucket if `wilson_lower >= mean_bound - validity_tolerance`.
    The p90/max bounds are reported as DIAGNOSTICS only (not in the gate),
    per Codex #86 — requiring `wilson_lower >= max_bound` is too brittle for
    P0/P1 since a single outlier prediction would fail the gate.

    Returns a dict with per-bucket results and aggregate counts.
    """
    p_arr = np.asarray(p_pred, dtype=float)
    b_arr = np.asarray(bounds, dtype=float)
    y_arr = np.asarray(actual, dtype=int)
    if not (len(p_arr) == len(b_arr) == len(y_arr)):
        raise ValueError("p_pred, bounds, actual must have matching length")

    # Filter rows where bound is None/NaN (sparse-bucket fallback returned None
    # in the calibrator); these can't be gated.
    finite_mask = np.isfinite(b_arr)
    p_arr = p_arr[finite_mask]
    b_arr = b_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    n_excluded_no_bound = int((~finite_mask).sum())

    bucket_low_vec = np.array([_bucket_low(p, bucket_width) for p in p_arr])

    bin_results: list[dict] = []
    populated = 0
    sparse = 0
    passing = 0

    for b_low in sorted(set(bucket_low_vec.tolist())):
        mask = bucket_low_vec == b_low
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        is_sparse = n_k < min_bucket_n
        if is_sparse:
            sparse += 1
        else:
            populated += 1

        bin_p = p_arr[mask]
        bin_b = b_arr[mask]
        bin_y = y_arr[mask]
        successes = int(bin_y.sum())

        mean_bound = float(bin_b.mean())
        p90_bound = float(np.quantile(bin_b, 0.9))
        max_bound = float(bin_b.max())

        observed_rate = successes / n_k
        wilson_lower = wilson_lower_one_sided(successes, n_k, wilson_alpha)

        passes = (not is_sparse) and (wilson_lower >= mean_bound - validity_tolerance)
        if passes:
            passing += 1

        bin_results.append({
            "bucket_low": b_low,
            "n": n_k,
            "sparse": is_sparse,
            "mean_p": float(bin_p.mean()),
            "mean_bound": mean_bound,
            "p90_bound": p90_bound,
            "max_bound": max_bound,
            "observed_rate": observed_rate,
            "wilson_lower": wilson_lower,
            "validity_tolerance": validity_tolerance,
            "passes_validity": passes,
        })

    all_pass = (populated > 0) and (passing == populated)
    return {
        "bin_results": bin_results,
        "n_populated_bins": populated,
        "n_sparse_bins": sparse,
        "n_passing": passing,
        "n_excluded_no_bound": n_excluded_no_bound,
        "all_populated_pass": all_pass,
        "min_bucket_n": min_bucket_n,
        "wilson_alpha": wilson_alpha,
        "bucket_width": bucket_width,
    }


def evaluate_tightness(
    bounds: np.ndarray,
    p_pred: np.ndarray,
    *,
    threshold: float = DEFAULT_TIGHTNESS_THRESHOLD,
) -> dict:
    """Median bound width = median(p_pred - bounds). Pass if <= threshold."""
    b_arr = np.asarray(bounds, dtype=float)
    p_arr = np.asarray(p_pred, dtype=float)
    finite = np.isfinite(b_arr)
    if not finite.any():
        return {
            "median_width": float("nan"),
            "p90_width": float("nan"),
            "threshold": threshold,
            "passes_tightness": False,
            "n_finite": 0,
        }
    widths = p_arr[finite] - b_arr[finite]
    return {
        "median_width": float(np.median(widths)),
        "p90_width": float(np.quantile(widths, 0.9)),
        "threshold": threshold,
        "passes_tightness": bool(np.median(widths) <= threshold),
        "n_finite": int(finite.sum()),
    }


def _fit_calibrator(
    method: str,
    alpha: float,
    train_df: pd.DataFrame,
    *,
    p_col: str = "p_game_hit",
    y_col: str = "actual_hit",
    bucket_width: float = DEFAULT_BUCKET_WIDTH,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
):
    """Fit the chosen calibrator on a fold's train slice.

    Calibrator config (bucket_width, min_bucket_n) is threaded through from
    the gate config so that the calibrator that generated the bounds and the
    evaluator that gates them use matching parameters.

    bucket_wilson uses raw (p, y) pairs.
    weighted_mondrian_conformal uses uniform weights in P0/P1 (covariate-shift
    LR weighting deferred to P2 per Codex #86).
    """
    p = train_df[p_col].to_numpy()
    y = train_df[y_col].to_numpy()
    alphas = [alpha]

    if method == "bucket_wilson":
        return fit_bucket_wilson_calibrator(
            list(zip(p.tolist(), y.tolist())),
            alphas=alphas,
            bucket_width=bucket_width,
            min_bucket_n=min_bucket_n,
        )
    if method == "weighted_mondrian_conformal":
        weights = np.ones(len(p))
        return fit_weighted_mondrian_conformal_calibrator(
            predicted_p=p,
            actual_hit=y.astype(float),
            weights=weights,
            alphas=alphas,
            bucket_width=bucket_width,
            min_bucket_eff_n=float(min_bucket_n),
        )
    raise ValueError(f"Unknown method: {method!r}")


def _apply_calibrator(method: str, cal, p_pred: np.ndarray) -> np.ndarray:
    """Apply the calibrator to a vector of p_pred; alpha_index=0 since we
    fit each (method, alpha) cell independently."""
    out = np.empty(len(p_pred), dtype=float)
    if method == "bucket_wilson":
        for i, p in enumerate(p_pred):
            v = apply_bucket_wilson(cal, float(p), alpha_index=0)
            out[i] = float("nan") if v is None else v
    elif method == "weighted_mondrian_conformal":
        for i, p in enumerate(p_pred):
            v = apply_weighted_mondrian_conformal(cal, float(p), alpha_index=0)
            out[i] = float("nan") if v is None else v
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return out


def run_gate_for_method_alpha(
    method: str,
    alpha: float,
    profiles_df: pd.DataFrame,
    manifest_path: str | Path,
    *,
    p_col: str = "p_game_hit",
    y_col: str = "actual_hit",
    bucket_width: float = DEFAULT_BUCKET_WIDTH,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
    validity_tolerance: float = DEFAULT_VALIDITY_TOLERANCE,
    tightness_threshold: float = DEFAULT_TIGHTNESS_THRESHOLD,
    wilson_alpha: float = DEFAULT_WILSON_ALPHA,
) -> dict:
    """Fit calibrator on each fold's train; evaluate on holdout; aggregate.

    Returns:
        dict with method, alpha, fold_results (list of per-fold validity +
        tightness diagnostics), aggregate verdict (PASS / FAIL / INSUFFICIENT_DATA),
        and fail_reasons.
    """
    from bts.validate.splits import (
        load_manifest,
        apply_fold,
        assert_no_lockbox_leakage,
    )

    folds, lockbox = load_manifest(manifest_path)
    assert_no_lockbox_leakage(folds, lockbox)

    fold_results = []
    fail_reasons: list[str] = []
    cell_passes = True
    insufficient = False

    for fold in folds:
        train_df, holdout_df = apply_fold(profiles_df, fold)
        if len(train_df) == 0 or len(holdout_df) == 0:
            insufficient = True
            fail_reasons.append(f"fold {fold.fold_idx}: empty train or holdout")
            fold_results.append({
                "fold_idx": fold.fold_idx,
                "skipped": True,
                "reason": "empty train or holdout",
            })
            continue

        cal = _fit_calibrator(
            method, alpha, train_df,
            p_col=p_col, y_col=y_col,
            bucket_width=bucket_width, min_bucket_n=min_bucket_n,
        )
        bounds = _apply_calibrator(method, cal, holdout_df[p_col].to_numpy())

        validity = evaluate_per_bucket_validity(
            holdout_df[p_col].to_numpy(),
            bounds,
            holdout_df[y_col].to_numpy(),
            bucket_width=bucket_width,
            min_bucket_n=min_bucket_n,
            validity_tolerance=validity_tolerance,
            wilson_alpha=wilson_alpha,
        )
        tightness = evaluate_tightness(
            bounds, holdout_df[p_col].to_numpy(),
            threshold=tightness_threshold,
        )

        if validity["n_populated_bins"] == 0:
            insufficient = True
            fail_reasons.append(
                f"fold {fold.fold_idx}: 0 populated bins "
                f"(min_bucket_n={min_bucket_n})"
            )
        elif not validity["all_populated_pass"]:
            cell_passes = False
            failing = [
                b["bucket_low"] for b in validity["bin_results"]
                if not b["passes_validity"] and not b["sparse"]
            ]
            fail_reasons.append(
                f"fold {fold.fold_idx}: validity fail buckets {failing}"
            )

        if not tightness["passes_tightness"]:
            cell_passes = False
            fail_reasons.append(
                f"fold {fold.fold_idx}: tightness median={tightness['median_width']:.4f} "
                f"> threshold={tightness['threshold']:.4f}"
            )

        # #12 reliability/Brier as DIAGNOSTICS only (per Codex #86, #89);
        # NOT a shipping gate. Lower-bound predictions are intentionally
        # conservative so proper scores aren't comparable to calibrated
        # forecasts as pass/fail — useful for sharpness/conservatism context.
        from bts.validate.proper_scoring import brier_score, murphy_decomposition
        finite_mask = np.isfinite(bounds)
        if finite_mask.any():
            diag_b = bounds[finite_mask]
            diag_y = holdout_df[y_col].to_numpy()[finite_mask]
            lower_bound_diagnostics = {
                "n_finite": int(finite_mask.sum()),
                "brier": brier_score(diag_b, diag_y),
                "murphy_decomposition": murphy_decomposition(
                    diag_b, diag_y, n_bins=10, binning="quantile"
                ),
                "note": "diagnostic only — does NOT affect verdict",
            }
        else:
            lower_bound_diagnostics = {
                "n_finite": 0,
                "note": "no finite bounds; diagnostics skipped",
            }

        fold_results.append({
            "fold_idx": fold.fold_idx,
            "n_train": int(len(train_df)),
            "n_holdout": int(len(holdout_df)),
            "validity": validity,
            "tightness": tightness,
            "lower_bound_diagnostics": lower_bound_diagnostics,
        })

    if insufficient:
        verdict = "INSUFFICIENT_DATA"
    elif cell_passes:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "method": method,
        "alpha": alpha,
        "verdict": verdict,
        "fail_reasons": fail_reasons,
        "fold_results": fold_results,
        "thresholds": {
            "validity_tolerance": validity_tolerance,
            "tightness_threshold": tightness_threshold,
            "min_bucket_n": min_bucket_n,
            "wilson_alpha": wilson_alpha,
            "bucket_width": bucket_width,
        },
    }


def run_gate_matrix(
    profiles_df: pd.DataFrame,
    manifest_path: str | Path,
    *,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    p_col: str = "p_game_hit",
    y_col: str = "actual_hit",
    bucket_width: float = DEFAULT_BUCKET_WIDTH,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
    validity_tolerance: float = DEFAULT_VALIDITY_TOLERANCE,
    tightness_threshold: float = DEFAULT_TIGHTNESS_THRESHOLD,
    wilson_alpha: float = DEFAULT_WILSON_ALPHA,
) -> dict:
    """Run the gate over the (method × alpha) matrix; return v2 schema dict.

    The output captures full manifest metadata, the cell matrix with per-fold
    results, the SHIP set (cells with verdict=PASS), and an explicit
    PRODUCTION_DEPLOY_READY / NO_PRODUCTION_DEPLOY top-level verdict.
    """
    manifest_raw = json.loads(Path(manifest_path).read_text())

    matrix: dict = {}
    ship_set: list[dict] = []
    for method in methods:
        for alpha in alphas:
            cell = run_gate_for_method_alpha(
                method, alpha, profiles_df, manifest_path,
                p_col=p_col, y_col=y_col,
                bucket_width=bucket_width,
                min_bucket_n=min_bucket_n,
                validity_tolerance=validity_tolerance,
                tightness_threshold=tightness_threshold,
                wilson_alpha=wilson_alpha,
            )
            matrix[f"{method}__alpha={alpha}"] = cell
            if cell["verdict"] == "PASS":
                ship_set.append({"method": method, "alpha": alpha})

    deploy_verdict = "PRODUCTION_DEPLOY_READY" if ship_set else "NO_PRODUCTION_DEPLOY"

    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_metadata": {
            "manifest_path": str(manifest_path),
            "schema_version": manifest_raw.get("schema_version"),
            "created_at": manifest_raw.get("created_at"),
            "split_params": manifest_raw.get("split_params"),
            "universe": manifest_raw.get("universe"),
        },
        "lockbox_held_out": True,
        "lockbox": {
            "start_date": manifest_raw["lockbox"]["start_date"],
            "end_date": manifest_raw["lockbox"]["end_date"],
            "description": manifest_raw["lockbox"]["description"],
        },
        "methods": list(methods),
        "alphas": list(alphas),
        "method_alpha_matrix": matrix,
        "ship_set": ship_set,
        "verdict": deploy_verdict,
        "thresholds": {
            "validity_tolerance": validity_tolerance,
            "tightness_threshold": tightness_threshold,
            "min_bucket_n": min_bucket_n,
            "wilson_alpha": wilson_alpha,
            "bucket_width": bucket_width,
        },
    }
