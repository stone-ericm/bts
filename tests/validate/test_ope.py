"""Tests for cross-fitted DR-OPE on the BTS MDP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.validate.ope import fitted_q_evaluation, corrected_audit_pipeline


# ---------------------------------------------------------------------------
# Module-level helpers for T4 tests (NOT @pytest.fixture — called as plain
# functions per Codex round 1 guidance; fixtures called as () are inconsistent
# with their decorator).
# ---------------------------------------------------------------------------

def _synthetic_profiles_5_seasons(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        for day_idx in range(150):
            date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day_idx)
            for s in [42, 43]:  # 2 seeds → 300 rows per season, 1500 total
                p1 = rng.uniform(0.6, 0.9)
                p2 = rng.uniform(0.4, p1)
                rows.append({
                    "season": season, "date": date, "seed": s,
                    "top1_p": p1, "top1_hit": int(rng.random() < p1),
                    "top2_p": p2, "top2_hit": int(rng.random() < p2),
                })
    return pd.DataFrame(rows)


def _synthetic_pa_5_seasons(seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in [2021, 2022, 2023, 2024, 2025]:
        for day_idx in range(150):
            date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day_idx)
            for batter_id in range(20):
                bg_id = f"{season}-{day_idx}-{batter_id}"
                p_pa = rng.uniform(0.15, 0.35)
                for pa_num in range(4):
                    rows.append({
                        "season": season, "date": date,
                        "batter_game_id": bg_id, "pa_num": pa_num,
                        "p_pa": p_pa, "actual_hit": int(rng.random() < p_pa),
                    })
    return pd.DataFrame(rows)


def _all_skip_policy(bins) -> np.ndarray:
    """Fake solver returning the shape _trajectory_dataframe_from_profiles expects.

    Codex round 1 caught: a dict-of-tuples fake returned wrong shape and would
    have made tests pass-by-error. The real solver returns np.ndarray with shape
    (n_streak_states, n_days_states, n_saver_states, n_bins). ACTION_SKIP=0.
    """
    n_bins = len(bins.bins)
    season_length = 200  # enough for any realistic test season
    return np.zeros((57, season_length + 1, 2, n_bins), dtype=int)


class TestFittedQEvaluation:
    def test_fqe_recovers_true_value_on_toy_mdp(self, toy_mdp_2state_2action):
        """FQE on synthetic data from the toy MDP should match analytical truth."""
        rng = np.random.default_rng(42)
        mdp = toy_mdp_2state_2action

        n_trajectories = 5000
        rows = []
        for _ in range(n_trajectories):
            s = 0
            for t in range(mdp["horizon"]):
                a = 1  # always advance
                next_states, probs = zip(*mdp["transitions"][(s, a)].items())
                sn = rng.choice(next_states, p=probs)
                r = mdp["rewards"](s, a, sn)
                rows.append({"t": t, "s": s, "a": a, "sn": sn, "r": r})
                s = sn

        df = pd.DataFrame(rows)
        target_policy = lambda s, t: 1  # always advance

        v_hat = fitted_q_evaluation(
            df, target_policy, n_states=2, n_actions=2, horizon=mdp["horizon"]
        )

        true_v = mdp["true_value"]["always_advance"]
        assert abs(v_hat - true_v) < 0.03, f"FQE recovered {v_hat:.4f} vs true {true_v:.4f}"


class TestDROPE:
    def test_dr_recovers_true_value_on_toy_mdp(self, toy_mdp_2state_2action):
        """DR estimator under full-information replay matches FQE asymptotically."""
        rng = np.random.default_rng(0)
        mdp = toy_mdp_2state_2action

        n_trajectories = 5000
        rows = []
        for traj_id in range(n_trajectories):
            s = 0
            for t in range(mdp["horizon"]):
                a = 1
                next_states, probs = zip(*mdp["transitions"][(s, a)].items())
                sn = rng.choice(next_states, p=probs)
                r = mdp["rewards"](s, a, sn)
                rows.append({"trajectory_id": traj_id, "t": t, "s": s, "a": a, "sn": sn, "r": r})
                s = sn
        df = pd.DataFrame(rows)
        target_policy = lambda s, t: 1

        from bts.validate.ope import dr_ope_full_information
        v_dr = dr_ope_full_information(
            df, target_policy, n_states=2, n_actions=2, horizon=mdp["horizon"]
        )
        true_v = mdp["true_value"]["always_advance"]
        assert abs(v_dr - true_v) < 0.03


class TestPairedHierarchicalBootstrap:
    def test_stationary_bootstrap_resamples_blocks(self):
        """Stationary bootstrap of Politis-Romano resamples contiguous day blocks."""
        from bts.validate.ope import stationary_bootstrap_indices
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(n_days=100, expected_block_length=7, rng=rng)
        assert len(idx) == 100
        assert idx.min() >= 0
        assert idx.max() < 100
        contiguous = sum(1 for i in range(len(idx) - 1) if idx[i + 1] == idx[i] + 1)
        assert contiguous > 50, f"only {contiguous} contiguous transitions; bootstrap may be IID-only"

    def test_paired_hierarchical_bootstrap_preserves_seed_bundles(self):
        """Resampled days should keep all seed rows together."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "season": [2024] * (50 * 24),
            "date": np.repeat(pd.date_range("2024-04-01", periods=50, freq="D"), 24),
            "seed": np.tile(range(24), 50),
            "value": rng.normal(size=50 * 24),
        })
        from bts.validate.ope import paired_hierarchical_bootstrap_sample
        bs = paired_hierarchical_bootstrap_sample(df, expected_block_length=7, rng=rng)
        per_date_counts = bs.groupby("date").size()
        assert (per_date_counts == 24).all(), f"some dates have != 24 seeds: {per_date_counts.value_counts()}"


class TestAuditModes:
    def test_fixed_policy_and_pipeline_modes_produce_finite_estimates(self, tmp_path):
        """Fixed-policy reuses a frozen policy; pipeline rebuilds per fold."""
        rng = np.random.default_rng(42)
        seasons = [2022, 2023, 2024]
        n_days = 100
        n_seeds = 24
        rows = []
        for season in seasons:
            for d in range(n_days):
                date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=d)
                for seed in range(n_seeds):
                    rows.append({
                        "season": season, "date": date, "seed": seed,
                        "top1_p": rng.uniform(0.65, 0.90),
                        "top1_hit": int(rng.random() < 0.78),
                        "top2_p": rng.uniform(0.65, 0.85),
                        "top2_hit": int(rng.random() < 0.75),
                    })
        profiles = pd.DataFrame(rows)

        from bts.validate.ope import audit_fixed_policy, audit_pipeline
        # Fixed-policy: a frozen policy table = always-skip baseline.
        n_streak, n_days_dim, n_saver, n_bins = 58, 200, 2, 5
        frozen_action_table = np.zeros((n_streak, n_days_dim, n_saver, n_bins), dtype=int)
        fixed_result = audit_fixed_policy(
            profiles,
            frozen_policy={"action_table": frozen_action_table},
            test_seasons=[2024],
            n_bootstrap=200,
        )
        assert isinstance(fixed_result.point_estimate, float)
        assert 0.0 <= fixed_result.point_estimate <= 1.0

        # Pipeline mode: LOSO across all 3 seasons.
        pipeline_result = audit_pipeline(
            profiles,
            fold_seasons=seasons,
            n_bootstrap=200,
        )
        assert isinstance(pipeline_result.point_estimate, float)
        assert 0.0 <= pipeline_result.point_estimate <= 1.0


def test_corrected_audit_pipeline_refits_parameters_per_fold(monkeypatch):
    """corrected_audit_pipeline calls dependence estimators once per fold AND
    each call's input excludes the held-out season (no leakage).
    """
    from bts.validate import ope as ope_mod

    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    leakage_violations = []  # collect held_out values that appeared in passed-in df

    def make_no_leak_wrapper(real_fn, fn_name):
        # The body of corrected_audit_pipeline tracks which season is held_out
        # in a `current_held_out` local. We can't directly observe that from
        # the wrapper, but we CAN observe that no single call's input df should
        # contain ALL 5 seasons — if it does, fold-local slicing failed.
        # Stronger check: every call's input should have len(unique seasons) == 4.
        def wrapper(df, *args, **kwargs):
            if "season" in df.columns:
                seasons_in_df = set(df["season"].unique())
                if len(seasons_in_df) != 4:
                    leakage_violations.append(
                        (fn_name, sorted(seasons_in_df))
                    )
            return real_fn(df, *args, **kwargs)
        return wrapper

    from bts.validate import dependence as dep_mod
    monkeypatch.setattr(
        dep_mod, "pa_residual_correlation",
        make_no_leak_wrapper(dep_mod.pa_residual_correlation, "pa_residual_correlation"),
    )
    monkeypatch.setattr(
        dep_mod, "fit_logistic_normal_random_intercept",
        make_no_leak_wrapper(dep_mod.fit_logistic_normal_random_intercept, "fit_lnri"),
    )
    # pair_residual_correlation takes pair_df (different column shape) — wrap
    # using a call-count-tracking no-leak helper:
    real_pair = dep_mod.pair_residual_correlation
    def no_leak_pair(pair_df, **kwargs):
        # pair_df doesn't have a 'season' column directly; count calls to verify
        # per-fold-ness.
        no_leak_pair.call_count += 1
        return real_pair(pair_df, **kwargs)
    no_leak_pair.call_count = 0
    monkeypatch.setattr(dep_mod, "pair_residual_correlation", no_leak_pair)

    result = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20,
        rho_pair_n_permutations=20,
    )

    # Per-fold count: pair_residual_correlation called 5x (once per fold).
    assert no_leak_pair.call_count == 5, (
        f"expected pair_residual_correlation called 5 times, got {no_leak_pair.call_count}"
    )
    # Critical: NO leakage violations should have been recorded.
    assert leakage_violations == [], f"Held-out leaked into estimator input: {leakage_violations}"


def test_corrected_audit_pipeline_returns_fold_metadata_with_per_bin_rho():
    """fold_metadata has per-fold rho_PA, tau, rho_pair_per_bin (shape 5),
    rho_pair_per_bin_ci, n_per_bin, stability dict.
    """
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    result = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20,
        rho_pair_n_permutations=20,
    )
    assert hasattr(result, "fold_metadata")
    assert len(result.fold_metadata) == 5
    for fold_meta in result.fold_metadata:
        assert "held_out_season" in fold_meta
        assert "rho_PA" in fold_meta
        assert "tau" in fold_meta
        assert "rho_pair_per_bin" in fold_meta
        assert fold_meta["rho_pair_per_bin"].shape == (5,)
        assert "rho_pair_per_bin_ci_lo" in fold_meta
        assert "rho_pair_per_bin_ci_hi" in fold_meta
        assert "rho_pair_n_per_bin" in fold_meta
        assert fold_meta["rho_pair_n_per_bin"].shape == (5,)
        assert "stability" in fold_meta
        assert "small_sample_warning" in fold_meta["stability"]
