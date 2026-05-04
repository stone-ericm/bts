"""Tests for SOTA #14 P0/P1 — CE-IS rare-event MC over manifest.

Per #14 P0 design memo (PR #14, merged at 908f764) + Codex implementation
sign-off in #134:
- Schema shape, lockbox held out, aggregate_deferred, n_folds, diagnostics
- Strict JSON allow_nan=False
- Deterministic seed reproducibility
- Exact oracle test: p=0.95 / season_length=70 / streak_threshold=57 vs exact_p57
- Black-box CE tuning test: theta moves + finite diagnostics
  (NO claim about proposal event-rate improvement; that requires
   proposal_event_rate exposure, deferred to P1.5+)
- CLI tests via CliRunner
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.validate.rare_event_mc_eval import (
    SCHEMA_VERSION,
    DEFAULT_MIN_ESS,
    DEFAULT_MAX_WEIGHT_SHARE,
    _build_holdout_profiles,
    _ensure_seed_column,
    _verdict_flag,
    evaluate_ceis_on_manifest,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _game_dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _make_profiles_df(
    dates_list,
    *,
    seed_label: int | None = 0,
    season: int = 2022,
    top_n: int = 10,
    p_pred_low: float = 0.65,
    p_pred_high: float = 0.92,
    rng_seed: int = 0,
):
    """Synthetic rank-row profiles with varying p_game_hit."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for d_idx, d in enumerate(dates_list):
        for rank in range(1, top_n + 1):
            day_offset = (d_idx * 0.003) % 0.05
            p = p_pred_low + (p_pred_high - p_pred_low) * (rank - 1) / max(top_n - 1, 1)
            p = min(0.999, max(0.001, p + day_offset))
            row = {
                "date": d,
                "rank": rank,
                "batter_id": 1000 + rank,
                "p_game_hit": p,
                "actual_hit": int(rng.random() < p),
                "n_pas": 4,
                "season": season,
            }
            if seed_label is not None:
                row["seed"] = seed_label
            rows.append(row)
    return pd.DataFrame(rows)


def _build_manifest(tmp_path, dates_list, lockbox_dates):
    from bts.validate.splits import (
        declare_lockbox,
        make_purged_blocked_cv,
        save_manifest,
    )
    lb = declare_lockbox(lockbox_dates[0], lockbox_dates[-1], "test lockbox")
    folds = make_purged_blocked_cv(
        dates_list,
        n_folds=3,
        purge_game_days=2,
        embargo_game_days=2,
        min_train_game_days=20,
        lockbox=lb,
    )
    path = tmp_path / "manifest.json"
    save_manifest(
        folds, lb, path,
        purge_game_days=2,
        embargo_game_days=2,
        min_train_game_days=20,
        mode="rolling_origin",
        universe_dates=dates_list,
    )
    return path


# ---------------------------------------------------------------------------
# Helpers: seed fallback + verdict flag + holdout profile builder
# ---------------------------------------------------------------------------

class TestSeedFallback:
    def test_inserts_zero_when_missing(self):
        df = _make_profiles_df(_game_dates(date(2022, 4, 1), 3), seed_label=None)
        assert "seed" not in df.columns
        out = _ensure_seed_column(df)
        assert "seed" in out.columns
        assert (out["seed"] == 0).all()

    def test_preserves_existing_seed(self):
        df = _make_profiles_df(_game_dates(date(2022, 4, 1), 3), seed_label=42)
        out = _ensure_seed_column(df)
        assert (out["seed"] == 42).all()


class TestBuildHoldoutProfiles:
    def test_takes_only_rank_1(self):
        df = _make_profiles_df(_game_dates(date(2022, 4, 1), 5), top_n=10)
        profiles = _build_holdout_profiles(df)
        # 5 dates × 1 rank-1 each = 5 entries
        assert len(profiles) == 5
        assert all("p_game" in p for p in profiles)

    def test_orders_by_date(self):
        """Per Codex #137 nonblocking: protect the date-order contract by
        asserting the p_game sequence reflects ascending date order."""
        # Tiny intentionally-unsorted DataFrame: distinct p per date, shuffled
        d0 = date(2022, 4, 1)
        rows = [
            {"date": d0 + timedelta(days=2), "rank": 1, "p_game_hit": 0.30},
            {"date": d0 + timedelta(days=0), "rank": 1, "p_game_hit": 0.10},
            {"date": d0 + timedelta(days=4), "rank": 1, "p_game_hit": 0.50},
            {"date": d0 + timedelta(days=1), "rank": 1, "p_game_hit": 0.20},
            {"date": d0 + timedelta(days=3), "rank": 1, "p_game_hit": 0.40},
        ]
        df = pd.DataFrame(rows)
        profiles = _build_holdout_profiles(df)
        # Date-ordered output: 0.10, 0.20, 0.30, 0.40, 0.50
        assert [p["p_game"] for p in profiles] == [0.10, 0.20, 0.30, 0.40, 0.50]

    def test_raises_on_duplicate_rank_1_dates(self):
        """Per Codex #137 #1: fail closed when multi-seed input would
        silently extend the horizon."""
        d0 = date(2022, 4, 1)
        rows = [
            {"date": d0, "rank": 1, "p_game_hit": 0.30},
            {"date": d0, "rank": 1, "p_game_hit": 0.40},  # duplicate date (seed 1)
            {"date": d0 + timedelta(days=1), "rank": 1, "p_game_hit": 0.50},
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="duplicate-date|multi-seed"):
            _build_holdout_profiles(df)


class TestVerdictFlag:
    def test_ok_when_both_within_thresholds(self):
        flag = _verdict_flag(
            ess=5000, max_weight_share=0.05,
            min_ess=1000, max_weight_share_threshold=0.1,
        )
        assert flag == "OK"

    def test_warning_when_ess_below_threshold(self):
        flag = _verdict_flag(
            ess=500, max_weight_share=0.05,
            min_ess=1000, max_weight_share_threshold=0.1,
        )
        assert flag == "IS_DIAGNOSTIC_WARNING"

    def test_warning_when_max_weight_above_threshold(self):
        flag = _verdict_flag(
            ess=5000, max_weight_share=0.5,
            min_ess=1000, max_weight_share_threshold=0.1,
        )
        assert flag == "IS_DIAGNOSTIC_WARNING"


# ---------------------------------------------------------------------------
# Exact oracle test (Codex #125 specified parameters)
# ---------------------------------------------------------------------------

class TestExactP57Oracle:
    """`exact_p57` is hard-coded to absorbing state 57 — must use threshold=57.
    Codex-verified setup: p=0.95, season_length=70, exact ≈ 0.08866."""

    def test_theta_zero_matches_exact_p57_constant_p_one_bin(self):
        from bts.simulate.exact import exact_p57
        from bts.simulate.quality_bins import QualityBins, QualityBin
        from bts.simulate.strategies import Strategy
        from bts.simulate.rare_event_mc import estimate_p57_with_ceis

        # One-bin QualityBins at p=0.95
        bins = QualityBins(
            bins=[QualityBin(
                index=0,
                p_range=(0.0, 1.0),
                p_hit=0.95,
                p_both=0.9025,
                frequency=1.0,
            )],
            boundaries=[],
        )
        strategy_always_play = Strategy(
            name="always_play",
            skip_threshold=None,
            double_threshold=None,
            streak_saver=False,
            streak_config=None,
        )
        exact = exact_p57(strategy_always_play, bins, season_length=70)

        # CE-IS at theta=0 on constant-p=0.95 profiles, season=70, threshold=57
        profiles = [{"p_game": 0.95}] * 70
        ceis = estimate_p57_with_ceis(
            profiles,
            strategy=None,
            n_rounds=0,
            theta=np.zeros(4),
            n_final=20000,
            seed=42,
            streak_threshold=57,
        )

        # Loose tolerance — within 3/4 of CI half-width
        ci_half = (ceis.ci_upper - ceis.ci_lower) / 2
        assert abs(ceis.point_estimate - exact) < 3 * ci_half / 4 + 1e-3, (
            f"CE-IS theta=0 estimate {ceis.point_estimate:.4f} vs "
            f"exact_p57 {exact:.4f} (CI half {ci_half:.4f})"
        )


# ---------------------------------------------------------------------------
# Black-box CE tuning test (Codex #125 replacement)
# ---------------------------------------------------------------------------

class TestBlackBoxCETuning:
    """Per Codex #125: CE tuning should move theta away from zero and
    return finite strict-JSON-safe diagnostics. Does NOT claim CE
    improves event-rate (requires proposal_event_rate exposure, deferred
    to P1.5+)."""

    def test_ce_tuning_changes_theta_and_returns_finite_diagnostics(self):
        from bts.simulate.rare_event_mc import estimate_p57_with_ceis

        # Synthetic with low base p and short horizon — CE should move theta
        profiles = [{"p_game": 0.5}] * 70
        result = estimate_p57_with_ceis(
            profiles,
            strategy=None,
            n_rounds=4,
            n_per_round=2000,
            n_final=5000,
            seed=42,
            streak_threshold=10,  # low threshold so the rare event is achievable
        )

        # theta moved away from zero
        assert result.theta_final[0] != 0.0

        # All public diagnostics finite (strict JSON would accept)
        diags = {
            "ess": result.ess,
            "max_weight_share": result.max_weight_share,
            "log_weight_variance": result.log_weight_variance,
            "point_estimate": result.point_estimate,
        }
        for k, v in diags.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

        # Per Codex #137 #2: max_weight_share below the warning threshold
        assert result.max_weight_share < DEFAULT_MAX_WEIGHT_SHARE, (
            f"max_weight_share={result.max_weight_share:.4f} >= "
            f"DEFAULT_MAX_WEIGHT_SHARE={DEFAULT_MAX_WEIGHT_SHARE} "
            "— CE tuning produced degenerate weights"
        )


# ---------------------------------------------------------------------------
# Manifest integration
# ---------------------------------------------------------------------------

class TestEvaluateCeisOnManifest:
    def _setup(self, tmp_path):
        # 100 dates across two seasons; lockbox = last 10
        dates = _game_dates(date(2022, 4, 1), 100)
        lockbox_dates = dates[-10:]
        df_2022 = _make_profiles_df(dates[:50], season=2022, seed_label=0)
        df_2023 = _make_profiles_df(dates[50:], season=2023, seed_label=0)
        df = pd.concat([df_2022, df_2023], ignore_index=True)
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        return df, manifest_path

    def test_returns_v1_schema_top_level_keys(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,  # achievable in synthetic test
        )
        for key in (
            "schema_version", "created_at", "estimand", "estimator",
            "manifest_metadata", "lockbox_held_out", "lockbox",
            "n_folds", "fold_results", "aggregate_deferred", "thresholds",
        ):
            assert key in result, f"missing top-level key {key}"

    def test_schema_version_constant(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        assert result["schema_version"] == SCHEMA_VERSION

    def test_lockbox_held_out_true(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        assert result["lockbox_held_out"] is True

    def test_aggregate_deferred_true(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        assert result["aggregate_deferred"] is True

    def test_n_folds_matches_manifest(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        assert result["n_folds"] == 3
        assert len(result["fold_results"]) == 3

    def test_each_fold_has_required_fields(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        for fr in result["fold_results"]:
            for key in (
                "fold_idx", "n_train_dates", "n_holdout_dates",
                "fixed_window_estimate", "ci_lower", "ci_upper",
                "theta_train", "diagnostics",
            ):
                assert key in fr, f"missing fold field {key}"
            for key in ("ess", "max_weight_share", "log_weight_variance",
                        "n_final", "verdict_flag"):
                assert key in fr["diagnostics"]

    def test_strict_json_allow_nan_false_round_trips(self, tmp_path):
        """Per Codex #134: artifact contract requires strict JSON
        (allow_nan=False) — fails closed if a metric becomes non-finite."""
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        # Should not raise
        s = json.dumps(result, default=str, allow_nan=False)
        loaded = json.loads(s)
        assert loaded["schema_version"] == SCHEMA_VERSION

    def test_deterministic_seed_reproducibility(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        kw = dict(
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10, seed=42,
        )
        a = evaluate_ceis_on_manifest(df, mpath, **kw)
        b = evaluate_ceis_on_manifest(df, mpath, **kw)
        for fra, frb in zip(a["fold_results"], b["fold_results"]):
            assert fra["fixed_window_estimate"] == frb["fixed_window_estimate"]
            assert fra["theta_train"] == frb["theta_train"]

    def test_manifest_metadata_includes_required_fields(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_ceis_on_manifest(
            df, mpath,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        for key in ("manifest_path", "schema_version", "created_at",
                    "split_params", "universe"):
            assert key in result["manifest_metadata"]

    def test_works_with_missing_seed_column(self, tmp_path):
        """Per #134/#106 pattern: must handle rank-row backtest profiles
        without a seed column."""
        dates = _game_dates(date(2022, 4, 1), 100)
        lockbox_dates = dates[-10:]
        df_a = _make_profiles_df(dates[:50], season=2022, seed_label=None)
        df_b = _make_profiles_df(dates[50:], season=2023, seed_label=None)
        df = pd.concat([df_a, df_b], ignore_index=True)
        assert "seed" not in df.columns
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        result = evaluate_ceis_on_manifest(
            df, manifest_path,
            n_rounds_train=2, n_per_round_train=200,
            n_final_train=200, n_final_holdout=500,
            streak_threshold=10,
        )
        assert result["n_folds"] == 3


# ---------------------------------------------------------------------------
# CLI tests via CliRunner
# ---------------------------------------------------------------------------

class TestRareEventCeIsCLI:
    def _setup_synthetic_data_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dates = _game_dates(date(2022, 4, 1), 100)
        lockbox_dates = dates[-10:]
        df_a = _make_profiles_df(dates[:50], season=2022, seed_label=0)
        df_b = _make_profiles_df(dates[50:], season=2023, seed_label=0)
        for season, sub_df in [(2022, df_a), (2023, df_b)]:
            sub_df.drop(columns="season").to_parquet(
                data_dir / f"backtest_{season}.parquet"
            )
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        return data_dir, manifest_path

    def test_cli_runs_basic_returns_v1_schema(self, tmp_path):
        from click.testing import CliRunner
        from bts.cli import cli
        data_dir, manifest_path = self._setup_synthetic_data_dir(tmp_path)
        out_path = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "rare-event-ce-is",
            "--profiles-dir", str(data_dir),
            "--manifest", str(manifest_path),
            "--n-rounds-train", "2",
            "--n-per-round-train", "200",
            "--n-final-train", "200",
            "--n-final-holdout", "500",
            "--streak-threshold", "10",
            "--output", str(out_path),
        ])
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["lockbox_held_out"] is True
        assert data["aggregate_deferred"] is True
        assert "fold_results" in data

    def test_cli_unknown_arg_raises_usage_error(self, tmp_path):
        from click.testing import CliRunner
        from bts.cli import cli
        data_dir, manifest_path = self._setup_synthetic_data_dir(tmp_path)
        out_path = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "rare-event-ce-is",
            "--profiles-dir", str(data_dir),
            "--manifest", str(manifest_path),
            "--bogus-flag", "value",
            "--output", str(out_path),
        ])
        assert result.exit_code != 0
        assert "no such option" in result.output.lower() or "--bogus-flag" in result.output.lower()
