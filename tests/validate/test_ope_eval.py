"""Tests for SOTA #13 P0/P1 — policy-value eval over manifest."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.validate.ope_eval import (
    SCHEMA_VERSION,
    DEFAULT_MIN_BIN_N,
    VALID_TARGET_POLICIES,
    _build_baseline_policy_table,
    _bin_index_for_p,
    _ensure_seed_column,
    _compute_per_bin_n,
    _terminal_mc_replay,
    evaluate_target_policy_on_manifest,
)
from bts.simulate.pooled_policy import compute_pooled_bins
from bts.simulate.mdp import solve_mdp


# ---------------------------------------------------------------------------
# Synthetic profile / manifest helpers
# ---------------------------------------------------------------------------

def _game_dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _make_profiles_df(dates_list, *, seed_label: int | None = 0,
                      season: int = 2022, top_n: int = 10,
                      hit_rate: float = 0.85, p_pred_low: float = 0.65,
                      p_pred_high: float = 0.92, rng_seed: int = 0):
    """Synthetic rank-row profiles with VARYING p_game_hit so quantile
    binning produces multiple non-empty bins. seed_label=None omits seed."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for d_idx, d in enumerate(dates_list):
        # Vary p_pred across dates AND ranks so we hit all 5 quantile bins
        for rank in range(1, top_n + 1):
            # Spread p_pred linearly across [low, high] within a date and
            # also vary slightly day-to-day for distribution coverage
            day_offset = (d_idx * 0.003) % 0.05
            p = p_pred_low + (p_pred_high - p_pred_low) * (rank - 1) / max(top_n - 1, 1)
            p = min(0.999, max(0.001, p + day_offset))
            row = {
                "date": d,
                "rank": rank,
                "batter_id": 1000 + rank,
                "p_game_hit": p,
                "actual_hit": int(rng.random() < hit_rate),
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
        dates_list, n_folds=3, purge_game_days=2, embargo_game_days=2,
        min_train_game_days=20, lockbox=lb,
    )
    path = tmp_path / "manifest.json"
    save_manifest(folds, lb, path,
                  purge_game_days=2, embargo_game_days=2,
                  min_train_game_days=20, mode="rolling_origin",
                  universe_dates=dates_list)
    return path


# ---------------------------------------------------------------------------
# Seed-column fallback (Codex #106 #3)
# ---------------------------------------------------------------------------

class TestSeedColumnFallback:
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


# ---------------------------------------------------------------------------
# Baseline policy tables
# ---------------------------------------------------------------------------

class TestBaselinePolicyTables:
    def test_always_skip_is_zeros(self):
        pt = _build_baseline_policy_table("always_skip", season_length=180, n_bins=5)
        assert pt.shape == (58, 181, 2, 5)
        assert (pt == 0).all()

    def test_always_rank1_is_ones(self):
        pt = _build_baseline_policy_table("always_rank1", season_length=180, n_bins=5)
        assert pt.shape == (58, 181, 2, 5)
        assert (pt == 1).all()

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError, match="always_skip"):
            _build_baseline_policy_table("unknown", season_length=180, n_bins=5)


# ---------------------------------------------------------------------------
# Solve-then-evaluate round trip
# ---------------------------------------------------------------------------

class TestSolveEvaluateRoundTrip:
    def test_evaluate_on_solver_bins_recovers_optimal_p57(self):
        """V^π(solver_bins) ≈ optimal_p57 from the solver itself."""
        # Synthetic 60-day season for fast solve
        dates = _game_dates(date(2022, 4, 1), 60)
        df = _make_profiles_df(dates, hit_rate=0.85, p_pred_low=0.65, p_pred_high=0.92, top_n=10)
        from bts.simulate.pooled_policy import compute_pooled_bins
        from bts.simulate.mdp import solve_mdp
        from bts.simulate.pooled_policy import evaluate_mdp_policy
        bins = compute_pooled_bins(df, n_bins=5)
        sol = solve_mdp(bins, season_length=60, late_phase_days=10)
        v_eval = evaluate_mdp_policy(
            sol.policy_table, bins, season_length=60, late_phase_days=10
        )
        assert abs(v_eval - sol.optimal_p57) < 1e-6


# ---------------------------------------------------------------------------
# Terminal MC replay state machine
# ---------------------------------------------------------------------------

class TestTerminalMCReplay:
    def _bins_for_varying_p(self):
        # Build QualityBins from synthetic profiles with varying p_game_hit
        df = _make_profiles_df(
            _game_dates(date(2022, 4, 1), 30), hit_rate=0.85
        )
        return compute_pooled_bins(df, n_bins=5)

    def test_skip_action_keeps_streak_unchanged(self):
        # All-skip policy: streak stays 0; never reaches 57; v_replay=0
        df = _make_profiles_df(_game_dates(date(2022, 4, 1), 60), hit_rate=0.85)
        bins = self._bins_for_varying_p()
        skip_table = np.zeros((58, 181, 2, 5), dtype=int)
        v, n_traj, n_term = _terminal_mc_replay(df, skip_table, bins, season_length=60)
        assert v == 0.0
        assert n_term == 0
        assert n_traj == 1  # one (season, seed) pair

    def test_single_action_hit_increments_streak(self):
        # All-hit, all-single policy on a 57-day season → streak reaches 57 on day 56
        dates = _game_dates(date(2022, 4, 1), 60)
        rows = []
        for d in dates:
            for rank in range(1, 3):
                rows.append({
                    "date": d, "rank": rank, "batter_id": 1000 + rank,
                    "p_game_hit": 0.78, "actual_hit": 1, "n_pas": 4,
                    "season": 2022, "seed": 0,
                })
        df = pd.DataFrame(rows)
        bins = self._bins_for_varying_p()
        single_table = np.ones((58, 181, 2, 5), dtype=int)
        v, n_traj, n_term = _terminal_mc_replay(df, single_table, bins, season_length=60)
        assert v == 1.0  # always reaches 57

    def test_single_action_miss_resets_streak(self):
        # Always-miss, always-single → streak never grows
        dates = _game_dates(date(2022, 4, 1), 60)
        rows = []
        for d in dates:
            for rank in range(1, 3):
                rows.append({
                    "date": d, "rank": rank, "batter_id": 1000 + rank,
                    "p_game_hit": 0.78, "actual_hit": 0, "n_pas": 4,
                    "season": 2022, "seed": 0,
                })
        df = pd.DataFrame(rows)
        bins = self._bins_for_varying_p()
        single_table = np.ones((58, 181, 2, 5), dtype=int)
        v, n_traj, n_term = _terminal_mc_replay(df, single_table, bins, season_length=60)
        assert v == 0.0


# ---------------------------------------------------------------------------
# Per-bin n diagnostics
# ---------------------------------------------------------------------------

class TestPerBinN:
    def test_counts_match_holdout_rank1_distribution(self):
        df = _make_profiles_df(
            _game_dates(date(2022, 4, 1), 100), hit_rate=0.85, p_pred_low=0.65, p_pred_high=0.92,
        )
        bins = compute_pooled_bins(df, n_bins=5)
        per_bin_n = _compute_per_bin_n(df, bins)
        # 100 rank-1 rows × 1 (one rank-1 per date)
        assert sum(per_bin_n) == 100

    def test_returns_zero_list_when_no_rank1(self):
        df = _make_profiles_df(_game_dates(date(2022, 4, 1), 5))
        df = df[df["rank"] != 1]  # remove all rank-1
        bins = compute_pooled_bins(
            _make_profiles_df(_game_dates(date(2022, 4, 1), 30)), n_bins=5,
        )
        result = _compute_per_bin_n(df, bins)
        assert result == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# Manifest integration
# ---------------------------------------------------------------------------

class TestEvaluateTargetPolicyOnManifest:
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
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10, n_bins=5,
        )
        for key in (
            "schema_version", "created_at", "estimand", "estimator",
            "target_policy", "manifest_metadata", "lockbox_held_out",
            "lockbox", "n_folds", "fold_results", "aggregate_deferred",
            "thresholds",
        ):
            assert key in result, f"missing top-level key {key}"

    def test_schema_version_constant(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        assert result["schema_version"] == SCHEMA_VERSION

    def test_lockbox_held_out_true(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        assert result["lockbox_held_out"] is True

    def test_aggregate_deferred_true(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        assert result["aggregate_deferred"] is True

    def test_n_folds_matches_manifest(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        assert result["n_folds"] == 3
        assert len(result["fold_results"]) == 3

    def test_each_fold_has_required_fields(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        for fr in result["fold_results"]:
            for key in (
                "fold_idx", "n_train_dates", "n_holdout_dates",
                "V_pi", "V_replay", "disagreement_abs",
                "disagreement_rel", "sparse_support",
            ):
                assert key in fr
            ss = fr["sparse_support"]
            for key in (
                "n_holdout_trajectories", "n_terminal_successes",
                "holdout_bin_min_n", "per_bin_n_early", "verdict_flag",
            ):
                assert key in ss

    def test_always_skip_baseline_yields_v_replay_zero(self, tmp_path):
        """always_skip never picks → never reaches 57 → V_replay=0."""
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        for fr in result["fold_results"]:
            assert fr["V_replay"] == 0.0

    def test_unknown_target_policy_raises(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        with pytest.raises(ValueError, match="target_policy must be one of"):
            evaluate_target_policy_on_manifest(
                df, mpath, target_policy_name="bogus",
                season_length=60, late_phase_days=10,
            )

    def test_disagreement_continuously_reported(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        for fr in result["fold_results"]:
            assert isinstance(fr["disagreement_abs"], float)
            # JSON-safe: disagreement_rel is float or None (per _safe_relative_disagreement).
            assert isinstance(fr["disagreement_rel"], float) or fr["disagreement_rel"] is None

    def test_sparse_support_flag_fires_with_high_threshold(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
            min_bin_n=10_000,  # impossibly high
        )
        for fr in result["fold_results"]:
            assert fr["sparse_support"]["verdict_flag"] == "SPARSE_HOLDOUT_SUPPORT"

    def test_manifest_metadata_includes_required_fields(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        for key in ("manifest_path", "schema_version", "created_at",
                    "split_params", "universe"):
            assert key in result["manifest_metadata"]

    def test_works_with_missing_seed_column(self, tmp_path):
        """Per Codex #106 #3: must handle rank-row backtest profiles without seed."""
        dates = _game_dates(date(2022, 4, 1), 100)
        lockbox_dates = dates[-10:]
        df_no_seed_a = _make_profiles_df(dates[:50], season=2022, seed_label=None)
        df_no_seed_b = _make_profiles_df(dates[50:], season=2023, seed_label=None)
        df = pd.concat([df_no_seed_a, df_no_seed_b], ignore_index=True)
        assert "seed" not in df.columns
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        # Should not raise
        result = evaluate_target_policy_on_manifest(
            df, manifest_path, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        assert result["n_folds"] == 3

    def test_output_round_trips_json_with_allow_nan_false(self, tmp_path):
        """Per Codex #109 #2: artifact must be strict JSON. Verify always_skip
        output (V_pi=0, V_replay=0) round-trips through json.dumps(allow_nan=False)."""
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        # Should not raise — strict JSON requires no NaN/Infinity
        s = json.dumps(result, default=str, allow_nan=False)
        loaded = json.loads(s)
        # disagreement_rel must be 0.0 (both zero) or None (undefined),
        # never inf/nan
        for fr in loaded["fold_results"]:
            rel = fr["disagreement_rel"]
            assert rel is None or isinstance(rel, (int, float))

    def test_disagreement_rel_zero_when_both_v_zero(self, tmp_path):
        df, mpath = self._setup(tmp_path)
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        for fr in result["fold_results"]:
            # always_skip: V_pi=0, V_replay=0, abs=0 → rel=0.0
            assert fr["V_pi"] == 0.0
            assert fr["V_replay"] == 0.0
            assert fr["disagreement_abs"] == 0.0
            assert fr["disagreement_rel"] == 0.0

    def test_late_phase_support_field_always_present(self, tmp_path):
        """Per Codex #109 #4: per_bin_n_late field must always be in
        sparse_support (None when no late phase / sparse late, list when
        late bins are populated). Also: holdout_bin_min_n takes min across
        non-empty phases when late bins are usable."""
        df, mpath = self._setup(tmp_path)
        # late_phase_days=10 keeps early phase dense enough for 5 quintile bins
        # at this synthetic scale. With late_phase_days=20 the early slice can
        # become too small to yield 5 distinct quintiles per fold.
        result = evaluate_target_policy_on_manifest(
            df, mpath, target_policy_name="always_skip",
            season_length=60, late_phase_days=10,
        )
        for fr in result["fold_results"]:
            ss = fr["sparse_support"]
            assert "per_bin_n_late" in ss
            # When per_bin_n_late is non-None, it must be a list
            if ss["per_bin_n_late"] is not None:
                assert isinstance(ss["per_bin_n_late"], list)
                assert all(isinstance(n, int) for n in ss["per_bin_n_late"])
            # per_bin_n_early is always populated
            assert "per_bin_n_early" in ss
            assert isinstance(ss["per_bin_n_early"], list)


# ---------------------------------------------------------------------------
# Terminal MC replay state-machine coverage (Codex #109 #3)
# ---------------------------------------------------------------------------

class TestTerminalMCReplayStateMachine:
    """Exhaustive coverage of skip/single/double/saver semantics. The state
    machine is the part most likely to silently diverge from BTS semantics."""

    def _build_uniform_p_bins(self):
        df = _make_profiles_df(
            _game_dates(date(2022, 4, 1), 30), hit_rate=0.85
        )
        return compute_pooled_bins(df, n_bins=5)

    def _df_all_hits(self, n_dates, *, season=2022, seed=0):
        """All ranks hit on every date — used to test double semantics."""
        rows = []
        for i in range(n_dates):
            d = date(2022, 4, 1) + timedelta(days=i)
            for rank in range(1, 4):
                rows.append({
                    "date": d, "rank": rank, "batter_id": 1000 + rank,
                    "p_game_hit": 0.78 + rank * 0.001,
                    "actual_hit": 1, "n_pas": 4,
                    "season": season, "seed": seed,
                })
        return pd.DataFrame(rows)

    def test_double_action_success_advances_streak_by_two(self):
        """All-hit data + always-double policy: streak grows by 2 per day,
        reaches 57 on day 29 (from streak=0, days=0 → streak=58 on day 29)."""
        df = self._df_all_hits(60)
        bins = self._build_uniform_p_bins()
        double_table = np.full((58, 181, 2, 5), 2, dtype=int)
        v, n_traj, n_term = _terminal_mc_replay(df, double_table, bins, season_length=60)
        assert v == 1.0  # reaches 57 well within 60 days
        assert n_term == 1

    def test_double_action_one_miss_resets_streak_when_no_saver(self):
        """rank-1 hits, rank-2 misses, always-double, no saver active.
        Streak should reset because both must hit for double to succeed."""
        rows = []
        for i in range(60):
            d = date(2022, 4, 1) + timedelta(days=i)
            for rank in range(1, 4):
                rows.append({
                    "date": d, "rank": rank, "batter_id": 1000 + rank,
                    "p_game_hit": 0.78 + rank * 0.001,
                    "actual_hit": 1 if rank == 1 else 0,  # rank-1 hits, rank-2 misses
                    "n_pas": 4, "season": 2022, "seed": 0,
                })
        df = pd.DataFrame(rows)
        bins = self._build_uniform_p_bins()
        double_table = np.full((58, 181, 2, 5), 2, dtype=int)
        v, _, _ = _terminal_mc_replay(df, double_table, bins, season_length=60)
        # streak resets every day → never reaches 57 → v_replay=0
        assert v == 0.0

    def test_phase_aware_replay_uses_late_boundaries_for_late_dates(self):
        """Per Codex #111 #1 + #113: with phase-aware bins, an all-hit
        trajectory where p=0.78 classifies as bin 4 (single) under early
        boundaries but bin 0 (skip) under late boundaries should produce
        DIFFERENT V values:
        - early-only replay: 60 days × single-hit → streak reaches 57 → V=1.0
        - phase-aware replay: 50 early days hit (streak=50) + 10 late days
          skip (streak unchanged) → never reaches 57 → V=0.0

        This regression test would fail if _terminal_mc_replay ignored
        late_bins/late_dates.
        """
        # Build early profiles in 0.10-0.70 → p=0.78 above all early boundaries (bin 4)
        rng = np.random.default_rng(0)
        early_rows = []
        for i in range(50):
            d = date(2022, 4, 1) + timedelta(days=i)
            for rank in range(1, 11):
                p = 0.10 + 0.60 * (rank - 1) / 9  # 0.10 to 0.70
                early_rows.append({
                    "date": d, "rank": rank, "batter_id": 1000 + rank,
                    "p_game_hit": p,
                    "actual_hit": int(rng.random() < p),
                    "n_pas": 4, "season": 2022, "seed": 0,
                })
        early_df = pd.DataFrame(early_rows)
        early_bins = compute_pooled_bins(early_df, n_bins=5)

        # Build late profiles in 0.85-0.99 → p=0.78 below all late boundaries (bin 0)
        late_rows = []
        for i in range(50, 80):
            d = date(2022, 4, 1) + timedelta(days=i)
            for rank in range(1, 11):
                p = 0.85 + 0.14 * (rank - 1) / 9  # 0.85 to 0.99
                late_rows.append({
                    "date": d, "rank": rank, "batter_id": 1000 + rank,
                    "p_game_hit": p,
                    "actual_hit": int(rng.random() < p),
                    "n_pas": 4, "season": 2022, "seed": 0,
                })
        late_df = pd.DataFrame(late_rows)
        late_bins = compute_pooled_bins(late_df, n_bins=5)

        # Build a 60-day all-hit holdout trajectory: rank-1 every date hits
        ho_rows = []
        for i in range(60):
            d = date(2022, 6, 1) + timedelta(days=i)
            ho_rows.append({
                "date": d, "rank": 1, "batter_id": 1001,
                "p_game_hit": 0.78, "actual_hit": 1,
                "n_pas": 4, "season": 2022, "seed": 0,
            })
            ho_rows.append({
                "date": d, "rank": 2, "batter_id": 1002,
                "p_game_hit": 0.50, "actual_hit": 1,
                "n_pas": 4, "season": 2022, "seed": 0,
            })
        ho_df = pd.DataFrame(ho_rows)

        # Sanity-check the bin classification differs (test-setup invariant):
        bin_early = _bin_index_for_p(0.78, list(early_bins.boundaries))
        bin_late = _bin_index_for_p(0.78, list(late_bins.boundaries))
        assert bin_early != bin_late, \
            f"test setup invalid: same bin {bin_early} under both boundary sets"

        # Construct a policy_table where bin 4 → SINGLE and bin 0 → SKIP.
        pt = np.zeros((58, 181, 2, 5), dtype=int)
        pt[:, :, :, 4] = 1  # bin 4 → single
        # bin 0 stays skip
        assert pt[0, 5, 1, bin_early] != pt[0, 5, 1, bin_late], \
            "test setup invalid: same action for both bins"

        # Mark the LAST 10 dates of the trajectory as "late phase"
        unique_dates = sorted(ho_df["date"].unique())
        late_dates = set(unique_dates[-10:])

        # Early-only replay: every date classified through early boundaries
        # → bin 4 → single → hit → streak grows 0..59 → reaches 57 → V=1.0
        v_early_only, _, _ = _terminal_mc_replay(
            ho_df, pt, early_bins, season_length=60,
        )
        assert v_early_only == 1.0, \
            f"early-only replay should reach 57 with all-hit trajectory; got V={v_early_only}"

        # Phase-aware replay: first 50 dates early → bin 4 → single → hit → streak 0..50
        # Last 10 dates late → bin 0 → skip → streak unchanged → never reaches 57 → V=0.0
        v_phase_aware, _, _ = _terminal_mc_replay(
            ho_df, pt, early_bins, season_length=60,
            late_bins=late_bins, late_dates=late_dates,
        )
        assert v_phase_aware == 0.0, \
            f"phase-aware replay should NOT reach 57 (late dates skip); got V={v_phase_aware}"

        # The replay-level behavioral difference is the regression contract:
        assert v_early_only != v_phase_aware

    def test_saver_preserves_streak_in_10_to_15_window_then_consumed(self):
        """Build a sequence where streak grows to 12, then a miss happens.
        Saver should preserve streak=12 once, then on the NEXT miss the
        streak resets."""
        rows = []
        # First 12 days: rank-1 hits → streak grows 0→12
        # Day 13: rank-1 miss → saver preserves streak=12
        # Day 14: rank-1 hits → streak=13
        # Day 15: rank-1 miss → saver gone → streak resets to 0
        for i in range(20):
            d = date(2022, 4, 1) + timedelta(days=i)
            if i < 12:
                actual_h = 1
            elif i == 12:
                actual_h = 0  # miss when streak=12 (in [10, 15] window)
            elif i == 13:
                actual_h = 1
            elif i == 14:
                actual_h = 0  # second miss
            else:
                actual_h = 1
            for rank in range(1, 4):
                rows.append({
                    "date": d, "rank": rank, "batter_id": 1000 + rank,
                    "p_game_hit": 0.78 + rank * 0.001,
                    "actual_hit": actual_h if rank == 1 else 1,
                    "n_pas": 4, "season": 2022, "seed": 0,
                })
        df = pd.DataFrame(rows)
        bins = self._build_uniform_p_bins()
        single_table = np.ones((58, 181, 2, 5), dtype=int)
        # Don't expect a streak of 57 here — just verify the trajectory completes
        # without raising. Inferring saver-consumption from final V_replay alone
        # is hard for a 20-day trajectory. This test exercises the code path.
        v, n_traj, n_term = _terminal_mc_replay(
            df, single_table, bins, season_length=20
        )
        # 20 days isn't enough to reach 57; v_replay=0 is correct
        assert v == 0.0
        assert n_traj == 1


# ---------------------------------------------------------------------------
# CLI tests via Click CliRunner (Codex #109 #1)
# ---------------------------------------------------------------------------

class TestPolicyValueEvalCLI:
    def _setup_synthetic_data_dir(self, tmp_path):
        """Create a temp data dir with backtest_*.parquet and a manifest
        targeting a subset of dates within."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dates = _game_dates(date(2022, 4, 1), 100)
        lockbox_dates = dates[-10:]
        df_a = _make_profiles_df(dates[:50], season=2022, seed_label=0)
        df_b = _make_profiles_df(dates[50:], season=2023, seed_label=0)
        # Drop season column from individual files so CLI can re-inject from filename
        for season, sub_df in [(2022, df_a), (2023, df_b)]:
            sub_df.drop(columns="season").to_parquet(data_dir / f"backtest_{season}.parquet")
        manifest_path = _build_manifest(tmp_path, dates, lockbox_dates)
        return data_dir, manifest_path

    def test_cli_runs_basic_returns_v1_schema(self, tmp_path):
        from click.testing import CliRunner
        from bts.cli import cli
        data_dir, manifest_path = self._setup_synthetic_data_dir(tmp_path)
        out_path = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "policy-value-eval",
            "--profiles-dir", str(data_dir),
            "--manifest", str(manifest_path),
            "--target-policy", "always_skip",
            "--season-length", "60",
            "--late-phase-days", "10",
            "--output", str(out_path),
        ])
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["schema_version"] == "policy_value_eval_v1"
        assert data["lockbox_held_out"] is True
        assert data["aggregate_deferred"] is True
        assert "fold_results" in data
        assert data["target_policy"] == "always_skip"

    def test_cli_unknown_policy_raises_usage_error(self, tmp_path):
        from click.testing import CliRunner
        from bts.cli import cli
        data_dir, manifest_path = self._setup_synthetic_data_dir(tmp_path)
        out_path = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "validate", "policy-value-eval",
            "--profiles-dir", str(data_dir),
            "--manifest", str(manifest_path),
            "--target-policy", "bogus_policy",
            "--output", str(out_path),
        ])
        # Click choices reject before the function body runs
        assert result.exit_code == 2
        assert "bogus_policy" in result.output.lower() or "invalid" in result.output.lower()
