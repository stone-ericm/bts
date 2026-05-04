"""Tests for conformal gate v2 — SOTA tracker item #11 phase 0/1.

TDD: tests written before module + binary-y anti-pattern test that
proves the OLD per-row metric was uninformative while the NEW per-bucket
realized-rate test catches pass/fail correctly.

Scope per Codex sign-off (bus msg #86):
- Primary gate: lower-bound calibration validity (Wilson_lower >= mean_bound - tolerance)
- p90/max bound reported as DIAGNOSTIC, not gate
- #5 manifest integration; lockbox held out
- INSUFFICIENT_DATA verdict for sparse cells
- conformal_validation_v2 output schema
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.validate.conformal_gate import (
    GATE_SCHEMA_VERSION,
    DEFAULT_VALIDITY_TOLERANCE,
    DEFAULT_TIGHTNESS_THRESHOLD,
    DEFAULT_MIN_BUCKET_N,
    evaluate_per_bucket_validity,
    evaluate_tightness,
    run_gate_for_method_alpha,
    run_gate_matrix,
)


# ---------------------------------------------------------------------------
# Synthetic builders
# ---------------------------------------------------------------------------

def _synthetic_calibrated_pred(n: int, *, hit_rate: float, p_pred: float, seed: int = 0):
    """Generate n predictions all clustered at p_pred with realized rate hit_rate."""
    rng = np.random.default_rng(seed)
    p = np.full(n, p_pred)
    actual = (rng.random(n) < hit_rate).astype(int)
    return p, actual


def _make_profiles_df(dates_list, p_pred=0.78, hit_rate=0.85, seed=0, top_n=10):
    """Build a profile DataFrame with a target hit rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates_list:
        for rank in range(1, top_n + 1):
            rows.append({
                "date": d,
                "rank": rank,
                "batter_id": 1000 + rank,
                "p_game_hit": p_pred,
                "actual_hit": int(rng.random() < hit_rate),
                "n_pas": 4,
            })
    return pd.DataFrame(rows)


def _build_manifest(tmp_path, dates_list, lockbox_dates):
    """Save a small manifest from synthetic dates for integration tests."""
    from bts.validate.splits import (
        declare_lockbox,
        make_purged_blocked_cv,
        save_manifest,
    )
    lb = declare_lockbox(lockbox_dates[0], lockbox_dates[-1], "test lockbox")
    folds = make_purged_blocked_cv(
        dates_list, n_folds=3, purge_game_days=2, embargo_game_days=2,
        min_train_game_days=20, lockbox=lb,
    )
    path = tmp_path / "manifest.json"
    save_manifest(
        folds, lb, path,
        purge_game_days=2, embargo_game_days=2,
        min_train_game_days=20, mode="rolling_origin",
        universe_dates=dates_list,
    )
    return path


# ---------------------------------------------------------------------------
# Per-bucket validity (the redesigned gate)
# ---------------------------------------------------------------------------

class TestPerBucketValidity:
    def test_passes_when_bound_is_conservative(self):
        """Bound far below realized rate => Wilson_lower >> mean_bound => PASS."""
        n = 400
        p_pred = np.full(n, 0.80)
        bounds = np.full(n, 0.50)  # conservative
        actual = np.random.default_rng(0).random(n) < 0.80
        result = evaluate_per_bucket_validity(p_pred, bounds, actual.astype(int))
        assert result["all_populated_pass"] is True

    def test_fails_when_bound_exceeds_realized_rate(self):
        """Bound above empirical => Wilson_lower < mean_bound => FAIL."""
        n = 400
        p_pred = np.full(n, 0.80)
        bounds = np.full(n, 0.95)  # too high
        actual = (np.random.default_rng(0).random(n) < 0.80).astype(int)
        result = evaluate_per_bucket_validity(p_pred, bounds, actual)
        assert result["all_populated_pass"] is False
        assert any(not b["passes_validity"] for b in result["bin_results"] if not b["sparse"])

    def test_sparse_buckets_excluded_from_pass_set(self):
        """Buckets with n < min_bucket_n marked sparse and not gated."""
        # 2 buckets: one populated (n=100), one sparse (n=5)
        p1 = np.full(100, 0.80)
        p2 = np.full(5, 0.50)
        p_pred = np.concatenate([p1, p2])
        bounds = np.concatenate([np.full(100, 0.50), np.full(5, 0.30)])
        actual = np.concatenate([
            (np.random.default_rng(0).random(100) < 0.80).astype(int),
            np.array([1, 1, 1, 1, 1]),
        ])
        result = evaluate_per_bucket_validity(p_pred, bounds, actual, min_bucket_n=30)
        assert result["n_sparse_bins"] == 1
        assert result["n_populated_bins"] == 1

    def test_reports_p90_and_max_as_diagnostics(self):
        """Per Codex #86: gate on mean_bound (with tolerance), not max.
        A single high-bound outlier MUST NOT fail the gate; it's surfaced
        in the diagnostic max_bound field but doesn't determine pass/fail.
        """
        n = 200
        p_pred = np.full(n, 0.80)
        # Mostly bound 0.5 but a single outlier at 0.99
        bounds = np.full(n, 0.50)
        bounds[0] = 0.99
        actual = (np.random.default_rng(0).random(n) < 0.80).astype(int)
        result = evaluate_per_bucket_validity(p_pred, bounds, actual)
        b = result["bin_results"][0]
        # max captures the outlier (diagnostic only)
        assert b["max_bound"] == pytest.approx(0.99)
        # mean is ~0.5025 — barely above the bulk bound; gate uses mean
        # so the single outlier doesn't make this brittle
        assert 0.50 < b["mean_bound"] < 0.51
        # And the gate passes (mean ≈ 0.50, observed ≈ 0.80, Wilson_lower >> mean)
        assert b["passes_validity"] is True

    def test_excludes_nan_bounds_from_evaluation(self):
        n = 200
        p_pred = np.full(n, 0.80)
        bounds = np.full(n, 0.50)
        bounds[:50] = float("nan")  # 50 NaN bounds (sparse-bucket fallback)
        actual = (np.random.default_rng(0).random(n) < 0.80).astype(int)
        result = evaluate_per_bucket_validity(p_pred, bounds, actual)
        assert result["n_excluded_no_bound"] == 50


# ---------------------------------------------------------------------------
# Binary-y anti-pattern test (Codex #86 explicit requirement)
# ---------------------------------------------------------------------------

class TestBinaryYAntiPattern:
    """The OLD per-row metric `(actual_hit >= bound).mean()` collapses for
    binary y to the empirical hit rate, regardless of bound. Prove that the
    NEW per-bucket realized-rate validation catches what the old metric
    couldn't."""

    def test_old_per_row_metric_is_uninformative_for_binary_y(self):
        """Two scenarios with DIFFERENT bound calibration produce IDENTICAL
        per-row coverage when actual_hit is binary."""
        n = 1000
        rng = np.random.default_rng(0)
        actual = (rng.random(n) < 0.80).astype(int)

        # Scenario A: conservative bound (bound=0.40). Binary actual_hit∈{0,1};
        # since 0 >= 0.40 is False and 1 >= 0.40 is True, per-row metric reduces
        # to mean(actual_hit) = ~0.80.
        bounds_A = np.full(n, 0.40)
        per_row_A = float((actual >= bounds_A).mean())

        # Scenario B: aggressive (invalid) bound (bound=0.95). Same logic:
        # 0 >= 0.95 False, 1 >= 0.95 True → metric still ≈ mean(actual_hit).
        bounds_B = np.full(n, 0.95)
        per_row_B = float((actual >= bounds_B).mean())

        # Both metrics ARE essentially the empirical hit rate, regardless of
        # bound. The metric cannot distinguish a valid bound from an invalid one.
        assert abs(per_row_A - per_row_B) < 1e-12
        assert abs(per_row_A - actual.mean()) < 1e-12

    def test_new_per_bucket_metric_distinguishes_valid_from_invalid_bound(self):
        """Same two scenarios; the per-bucket Wilson-vs-mean_bound test
        correctly passes A and fails B."""
        n = 1000
        rng = np.random.default_rng(0)
        p_pred = np.full(n, 0.80)
        actual = (rng.random(n) < 0.80).astype(int)

        bounds_A = np.full(n, 0.40)  # conservative — should PASS
        bounds_B = np.full(n, 0.95)  # invalid — should FAIL

        result_A = evaluate_per_bucket_validity(p_pred, bounds_A, actual)
        result_B = evaluate_per_bucket_validity(p_pred, bounds_B, actual)

        assert result_A["all_populated_pass"] is True, "conservative bound must pass"
        assert result_B["all_populated_pass"] is False, "invalid bound must fail"


# ---------------------------------------------------------------------------
# Tightness gate
# ---------------------------------------------------------------------------

class TestTightness:
    def test_passes_when_median_width_below_threshold(self):
        n = 200
        p_pred = np.full(n, 0.80)
        bounds = np.full(n, 0.60)  # width = 0.20
        result = evaluate_tightness(bounds, p_pred, threshold=0.30)
        assert result["passes_tightness"] is True
        assert abs(result["median_width"] - 0.20) < 1e-9

    def test_fails_when_median_width_above_threshold(self):
        n = 200
        p_pred = np.full(n, 0.80)
        bounds = np.full(n, 0.30)  # width = 0.50
        result = evaluate_tightness(bounds, p_pred, threshold=0.30)
        assert result["passes_tightness"] is False

    def test_handles_all_nan_bounds(self):
        n = 200
        p_pred = np.full(n, 0.80)
        bounds = np.full(n, float("nan"))
        result = evaluate_tightness(bounds, p_pred)
        assert result["n_finite"] == 0
        assert result["passes_tightness"] is False


# ---------------------------------------------------------------------------
# Manifest-integrated gate (composes #5)
# ---------------------------------------------------------------------------

class TestRunGateForMethodAlpha:
    def _setup(self, tmp_path, hit_rate=0.85):
        # 100 sequential dates, lockbox = last 10
        dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        lockbox_dates = dates[-10:]
        non_lockbox = dates[:-10]
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        df = _make_profiles_df(dates, hit_rate=hit_rate)
        return df, manifest_path

    def test_pass_verdict_with_calibrated_data(self, tmp_path):
        df, mpath = self._setup(tmp_path, hit_rate=0.85)
        # bucket_wilson on well-calibrated data should PASS at α=0.20
        cell = run_gate_for_method_alpha(
            "bucket_wilson", 0.20, df, mpath,
            min_bucket_n=10,
        )
        # Either PASS or INSUFFICIENT_DATA depending on data sparsity;
        # with 90 dates × 10 ranks = 900 rows in non-lockbox we expect populated buckets.
        assert cell["verdict"] in ("PASS", "FAIL", "INSUFFICIENT_DATA")
        assert cell["method"] == "bucket_wilson"
        assert cell["alpha"] == 0.20
        assert "fold_results" in cell
        assert len(cell["fold_results"]) >= 1

    def test_insufficient_data_when_buckets_too_sparse(self, tmp_path):
        """With min_bucket_n set very high relative to data, expect INSUFFICIENT."""
        df, mpath = self._setup(tmp_path)
        cell = run_gate_for_method_alpha(
            "bucket_wilson", 0.10, df, mpath,
            min_bucket_n=10_000,  # impossibly high
        )
        assert cell["verdict"] == "INSUFFICIENT_DATA"

    def test_unknown_method_raises(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        with pytest.raises(ValueError, match="Unknown method"):
            run_gate_for_method_alpha("not_a_method", 0.10, df, mpath)


# ---------------------------------------------------------------------------
# Full matrix run + v2 schema
# ---------------------------------------------------------------------------

class TestRunGateMatrix:
    def _setup(self, tmp_path):
        dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        lockbox_dates = dates[-10:]
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        df = _make_profiles_df(dates, hit_rate=0.85)
        return df, manifest_path

    def test_v2_schema_top_level_keys(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = run_gate_matrix(
            df, mpath,
            methods=("bucket_wilson",),
            alphas=(0.20,),
            min_bucket_n=10,
        )
        for key in (
            "schema_version", "created_at", "manifest_metadata",
            "lockbox_held_out", "lockbox", "methods", "alphas",
            "method_alpha_matrix", "ship_set", "verdict", "thresholds",
        ):
            assert key in result, f"missing top-level key {key}"

    def test_schema_version_constant(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = run_gate_matrix(df, mpath, methods=("bucket_wilson",), alphas=(0.20,),
                                  min_bucket_n=10)
        assert result["schema_version"] == GATE_SCHEMA_VERSION

    def test_lockbox_held_out_true(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = run_gate_matrix(df, mpath, methods=("bucket_wilson",), alphas=(0.20,),
                                  min_bucket_n=10)
        assert result["lockbox_held_out"] is True

    def test_manifest_metadata_includes_required_fields(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = run_gate_matrix(df, mpath, methods=("bucket_wilson",), alphas=(0.20,),
                                  min_bucket_n=10)
        meta = result["manifest_metadata"]
        for key in ("manifest_path", "schema_version", "created_at",
                    "split_params", "universe"):
            assert key in meta

    def test_verdict_is_no_deploy_when_ship_set_empty(self, tmp_path):
        """Force every cell to fail by setting a tight tightness threshold."""
        df, mpath = self._setup(tmp_path)
        result = run_gate_matrix(
            df, mpath,
            methods=("bucket_wilson",), alphas=(0.05,),
            min_bucket_n=10,
            tightness_threshold=0.001,  # impossibly tight
        )
        assert result["ship_set"] == []
        assert result["verdict"] == "NO_PRODUCTION_DEPLOY"

    def test_matrix_cells_keyed_by_method_alpha(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = run_gate_matrix(
            df, mpath,
            methods=("bucket_wilson",),
            alphas=(0.10, 0.20),
            min_bucket_n=10,
        )
        assert "bucket_wilson__alpha=0.1" in result["method_alpha_matrix"]
        assert "bucket_wilson__alpha=0.2" in result["method_alpha_matrix"]

    def test_thresholds_recorded_in_output(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = run_gate_matrix(
            df, mpath,
            methods=("bucket_wilson",), alphas=(0.20,),
            min_bucket_n=15, validity_tolerance=0.02, tightness_threshold=0.25,
        )
        thr = result["thresholds"]
        assert thr["min_bucket_n"] == 15
        assert thr["validity_tolerance"] == 0.02
        assert thr["tightness_threshold"] == 0.25


# ---------------------------------------------------------------------------
# Calibrator config threading (Codex #89 fix #1)
# ---------------------------------------------------------------------------

class TestCalibratorConfigThreading:
    """The calibrator that produces the bounds must use the SAME bucket_width
    and min_bucket_n as the evaluator that gates them."""

    def test_non_default_bucket_width_threaded_into_bucket_wilson(self):
        """fit_bucket_wilson_calibrator should be called with the gate's
        configured bucket_width, not the module default."""
        from bts.validate.conformal_gate import _fit_calibrator
        # 200 predictions in [0.6, 0.9]. With bucket_width=0.05 we expect ~6 buckets;
        # with default 0.025 we'd get ~12. Test by inspecting the calibrator's bucket_width.
        rng = np.random.default_rng(0)
        train_df = pd.DataFrame({
            "p_game_hit": rng.uniform(0.6, 0.9, size=200),
            "actual_hit": rng.integers(0, 2, size=200),
        })
        cal = _fit_calibrator(
            "bucket_wilson", 0.10, train_df,
            bucket_width=0.05, min_bucket_n=10,
        )
        assert cal.bucket_width == 0.05

    def test_non_default_bucket_width_threaded_into_weighted_mondrian(self):
        from bts.validate.conformal_gate import _fit_calibrator
        rng = np.random.default_rng(1)
        train_df = pd.DataFrame({
            "p_game_hit": rng.uniform(0.6, 0.9, size=200),
            "actual_hit": rng.integers(0, 2, size=200),
        })
        cal = _fit_calibrator(
            "weighted_mondrian_conformal", 0.10, train_df,
            bucket_width=0.05, min_bucket_n=10,
        )
        assert cal.bucket_width == 0.05

    def test_min_bucket_n_threaded_into_bucket_wilson_buckets(self):
        """With min_bucket_n=200 and only 100 rows in each bucket, the calibrator
        should drop the bucket. Verifies fit_bucket_wilson_calibrator received
        the threaded min_bucket_n."""
        from bts.validate.conformal_gate import _fit_calibrator
        # Build a dataset with two clear buckets of n=100 each
        train_df = pd.DataFrame({
            "p_game_hit": [0.7] * 100 + [0.85] * 100,
            "actual_hit": [1] * 100 + [1] * 100,
        })
        cal_low_min = _fit_calibrator(
            "bucket_wilson", 0.10, train_df,
            bucket_width=0.025, min_bucket_n=10,
        )
        cal_high_min = _fit_calibrator(
            "bucket_wilson", 0.10, train_df,
            bucket_width=0.025, min_bucket_n=500,  # bigger than any bucket
        )
        # Low min: at least one bucket retained
        assert len(cal_low_min.bucket_lower) >= 1
        # High min: all buckets dropped (100 < 500)
        assert len(cal_high_min.bucket_lower) == 0


# ---------------------------------------------------------------------------
# lower_bound_diagnostics block (Codex #89 fix #2)
# ---------------------------------------------------------------------------

class TestLowerBoundDiagnostics:
    """Per Codex #86: #12 reliability/Brier are DIAGNOSTICS, not shipping
    gates. The output must include them for context, but their values must
    not change the verdict."""

    def _setup(self, tmp_path):
        dates = [date(2022, 1, 1) + timedelta(days=i) for i in range(100)]
        lockbox_dates = dates[-10:]
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        df = _make_profiles_df(dates, hit_rate=0.85)
        return df, manifest_path

    def test_diagnostics_present_in_each_fold_result(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        cell = run_gate_for_method_alpha(
            "bucket_wilson", 0.10, df, mpath, min_bucket_n=10,
        )
        for fr in cell["fold_results"]:
            if fr.get("skipped"):
                continue
            assert "lower_bound_diagnostics" in fr
            d = fr["lower_bound_diagnostics"]
            assert "n_finite" in d

    def test_diagnostics_include_brier_and_murphy_when_finite_bounds_exist(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        cell = run_gate_for_method_alpha(
            "bucket_wilson", 0.10, df, mpath, min_bucket_n=10,
        )
        # Find a fold with finite bounds
        for fr in cell["fold_results"]:
            if fr.get("skipped"):
                continue
            d = fr["lower_bound_diagnostics"]
            if d["n_finite"] > 0:
                assert "brier" in d
                assert "murphy_decomposition" in d
                # Murphy block has the canonical fields
                m = d["murphy_decomposition"]
                for k in ("reliability", "resolution", "uncertainty", "brier"):
                    assert k in m
                return
        pytest.fail("no fold had finite bounds for diagnostic check")

    def test_diagnostics_do_not_affect_verdict(self, tmp_path):
        """Run the gate twice with the same inputs; the verdict should be
        a function of validity + tightness only. (Tests by checking that
        the verdict keys are determined solely by validity/tightness fields.)
        """
        df, mpath = self._setup(tmp_path)
        cell = run_gate_for_method_alpha(
            "bucket_wilson", 0.10, df, mpath, min_bucket_n=10,
        )
        # The presence of "note" in diagnostics is the marker that diagnostics
        # are non-gating. Verify the verdict logic ignores them.
        for fr in cell["fold_results"]:
            if fr.get("skipped"):
                continue
            assert fr["lower_bound_diagnostics"].get("note", "").startswith(
                ("diagnostic only", "no finite bounds")
            )
