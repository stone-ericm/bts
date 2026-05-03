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

    expected_held_in_per_call = []  # list of expected held-in season sets (one per fold)
    actual_held_in_per_call = []   # list of actual season sets seen by estimator

    def make_strict_no_leak_wrapper(real_fn, fn_name):
        def wrapper(df, *args, **kwargs):
            if "season" in df.columns:
                seasons_in_df = set(df["season"].unique())
                actual_held_in_per_call.append((fn_name, seasons_in_df))
            return real_fn(df, *args, **kwargs)
        return wrapper

    from bts.validate import dependence as dep_mod
    monkeypatch.setattr(
        dep_mod, "pa_residual_correlation",
        make_strict_no_leak_wrapper(dep_mod.pa_residual_correlation, "pa_residual_correlation"),
    )
    monkeypatch.setattr(
        dep_mod, "fit_logistic_normal_random_intercept",
        make_strict_no_leak_wrapper(dep_mod.fit_logistic_normal_random_intercept, "fit_lnri"),
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
        pa_n_bootstrap=20,  # keep tests fast
    )

    # Per-fold count: pair_residual_correlation called 5x (once per fold).
    assert no_leak_pair.call_count == 5, (
        f"expected pair_residual_correlation called 5 times, got {no_leak_pair.call_count}"
    )

    # Exact-set leakage check: each estimator must have been called 5 times,
    # once per fold, and the seasons in each call must be exactly
    # all_seasons - {held_out} for some held_out in fold_seasons.
    # This catches both cardinality bugs (|seasons| != 4) AND identity bugs
    # (wrong complement, e.g., train=[2021,2023,2024,2025] when held_out=2022
    # would have cardinality 4 but would also have 2021 when it should not).
    all_seasons = {2021, 2022, 2023, 2024, 2025}
    pa_calls = [s for fn, s in actual_held_in_per_call if fn == "pa_residual_correlation"]
    lnri_calls = [s for fn, s in actual_held_in_per_call if fn == "fit_lnri"]
    assert len(pa_calls) == 5, f"pa_residual_correlation called {len(pa_calls)} times, expected 5"
    assert len(lnri_calls) == 5, f"fit_lnri called {len(lnri_calls)} times, expected 5"
    expected_sets = {frozenset(all_seasons - {h}) for h in all_seasons}
    assert {frozenset(s) for s in pa_calls} == expected_sets, (
        f"pa_residual_correlation leaked or used wrong complement: {pa_calls}"
    )
    assert {frozenset(s) for s in lnri_calls} == expected_sets, (
        f"fit_lnri leaked or used wrong complement: {lnri_calls}"
    )


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
        pa_n_bootstrap=20,  # keep tests fast
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
        # Catches the silent-degenerate-output bug: if per-bin rho computation
        # returned all-zero arrays for some reason, the structural assertions above
        # would pass. The n_per_bin must show actual observations populated the bins.
        assert fold_meta["rho_pair_n_per_bin"].sum() > 0, (
            f"Fold for season {fold_meta['held_out_season']}: "
            f"rho_pair_n_per_bin is all zero — bin classification failed silently"
        )


# ---------------------------------------------------------------------------
# v2.5 ablation mode flag tests
# ---------------------------------------------------------------------------

def test_corrected_audit_pipeline_default_modes_match_v2():
    """When no mode flags are specified, fold_metadata still has all expected keys
    and the 3 mode fields record the v2 defaults.
    """
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    result = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20,
        rho_pair_n_permutations=20,
        pa_n_bootstrap=20,
    )

    assert len(result.fold_metadata) == 5
    for fm in result.fold_metadata:
        assert fm["params_mode"] == "fold-local"
        assert fm["rho_pair_mode"] == "per-bin"
        assert fm["policy_mode"] == "per-fold"
        # rho_pair_scope is derived from policy_mode: "fold-local" when per-fold.
        assert fm["rho_pair_scope"] == "fold-local"
        # Legacy keys still present.
        assert "rho_PA" in fm
        assert "tau" in fm
        assert "stability" in fm
        assert fm["rho_pair_per_bin"].shape == (5,)


def test_corrected_audit_pipeline_rejects_undefined_combo():
    """fold-local + global is the operationally undefined combination; must raise ValueError."""
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    with pytest.raises(ValueError, match="operationally undefined"):
        corrected_audit_pipeline(
            profiles, pa_df,
            fold_seasons=[2021, 2022, 2023, 2024, 2025],
            mdp_solve_fn=_all_skip_policy,
            n_bootstrap=20,
            rho_pair_n_permutations=20,
            pa_n_bootstrap=20,
            params_mode="fold-local",
            policy_mode="global",
        )


def test_corrected_audit_pipeline_pooled_params_uses_full_data(monkeypatch):
    """params_mode='pooled' calls pa_residual_correlation once with all 5 seasons;
    params_mode='fold-local' calls it 5 times each with exactly 4 seasons.
    """
    from bts.validate import dependence as dep_mod

    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()
    all_seasons = {2021, 2022, 2023, 2024, 2025}

    # --- pooled mode: exactly 1 call with all 5 seasons ---
    pooled_call_seasons = []
    real_pa_corr = dep_mod.pa_residual_correlation

    def pooled_spy(df, *args, **kwargs):
        if "season" in df.columns:
            pooled_call_seasons.append(set(df["season"].unique()))
        return real_pa_corr(df, *args, **kwargs)

    monkeypatch.setattr(dep_mod, "pa_residual_correlation", pooled_spy)
    corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=10,
        rho_pair_n_permutations=10,
        pa_n_bootstrap=10,
        params_mode="pooled",
        rho_pair_mode="per-bin",
        policy_mode="per-fold",
    )
    assert len(pooled_call_seasons) == 1, (
        f"pooled mode called pa_residual_correlation {len(pooled_call_seasons)} times, expected 1"
    )
    assert pooled_call_seasons[0] == all_seasons, (
        f"pooled mode passed seasons {pooled_call_seasons[0]}, expected all 5"
    )

    # --- fold-local mode: 5 calls each with 4 seasons ---
    local_call_seasons = []

    def local_spy(df, *args, **kwargs):
        if "season" in df.columns:
            local_call_seasons.append(set(df["season"].unique()))
        return real_pa_corr(df, *args, **kwargs)

    monkeypatch.setattr(dep_mod, "pa_residual_correlation", local_spy)
    corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=10,
        rho_pair_n_permutations=10,
        pa_n_bootstrap=10,
        params_mode="fold-local",
        rho_pair_mode="per-bin",
        policy_mode="per-fold",
    )
    assert len(local_call_seasons) == 5, (
        f"fold-local mode called pa_residual_correlation {len(local_call_seasons)} times, expected 5"
    )
    for s in local_call_seasons:
        assert len(s) == 4, f"fold-local call had {len(s)} seasons, expected 4"
    expected_complements = {frozenset(all_seasons - {h}) for h in all_seasons}
    assert {frozenset(s) for s in local_call_seasons} == expected_complements


def test_corrected_audit_pipeline_global_policy_solves_once(monkeypatch):
    """policy_mode='global' calls mdp_solve_fn exactly once; 'per-fold' calls it 5 times."""
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    call_counts = {"n": 0}

    def counting_solver(bins):
        call_counts["n"] += 1
        return _all_skip_policy(bins)

    # global mode: MDP solved once before the fold loop.
    call_counts["n"] = 0
    corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=counting_solver,
        n_bootstrap=10,
        rho_pair_n_permutations=10,
        pa_n_bootstrap=10,
        params_mode="pooled",
        rho_pair_mode="per-bin",
        policy_mode="global",
    )
    assert call_counts["n"] == 1, (
        f"global mode called mdp_solve_fn {call_counts['n']} times, expected 1"
    )

    # per-fold mode: MDP solved once per fold = 5 times.
    call_counts["n"] = 0
    corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=counting_solver,
        n_bootstrap=10,
        rho_pair_n_permutations=10,
        pa_n_bootstrap=10,
        params_mode="fold-local",
        rho_pair_mode="per-bin",
        policy_mode="per-fold",
    )
    assert call_counts["n"] == 5, (
        f"per-fold mode called mdp_solve_fn {call_counts['n']} times, expected 5"
    )


def test_corrected_audit_pipeline_scalar_rho_pair_uses_global_scalar(monkeypatch):
    """rho_pair_mode='scalar' feeds a scalar (not a length-5 array) to
    build_corrected_transition_table at every call site.
    """
    from bts.validate import dependence as dep_mod

    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    rho_args_seen = []
    real_build = dep_mod.build_corrected_transition_table

    def spy_build(bins, *, rho_PA_within_game, tau_squared, rho_pair_cross_game, n_pa_per_game):
        rho_args_seen.append(np.asarray(rho_pair_cross_game))
        return real_build(
            bins,
            rho_PA_within_game=rho_PA_within_game,
            tau_squared=tau_squared,
            rho_pair_cross_game=rho_pair_cross_game,
            n_pa_per_game=n_pa_per_game,
        )

    monkeypatch.setattr(dep_mod, "build_corrected_transition_table", spy_build)

    corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=10,
        rho_pair_n_permutations=10,
        pa_n_bootstrap=10,
        params_mode="pooled",
        rho_pair_mode="scalar",
        policy_mode="per-fold",
    )

    assert len(rho_args_seen) == 5, f"expected 5 build calls, got {len(rho_args_seen)}"
    for arr in rho_args_seen:
        # A scalar or shape-() array has ndim=0; shape-() and shape-(1,) are both
        # acceptable as "scalar" — the key thing is it's NOT a length-5 vector.
        assert arr.size == 1, (
            f"scalar mode passed a rho_pair of size {arr.size} to "
            f"build_corrected_transition_table; expected size 1 (scalar)"
        )


def test_corrected_audit_pipeline_cell_101_fold_local_scalar_per_fold():
    """Cell 101 (fold-local params + scalar rho_pair + per-fold policy) is
    the only valid cell not exercised by the dimension-focused tests.
    Verify it produces a coherent verdict + fold metadata.
    """
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    result = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20,
        rho_pair_n_permutations=20,
        pa_n_bootstrap=20,
        params_mode="fold-local",
        rho_pair_mode="scalar",
        policy_mode="per-fold",
    )
    assert isinstance(result.point_estimate, float)
    assert len(result.fold_metadata) == 5
    for fm in result.fold_metadata:
        assert fm["params_mode"] == "fold-local"
        assert fm["rho_pair_mode"] == "scalar"
        assert fm["policy_mode"] == "per-fold"
        # rho_pair_per_bin in scalar mode is a synthesized broadcast (np.full)
        assert fm["rho_pair_per_bin"].shape == (5,)
        assert np.all(fm["rho_pair_per_bin"] == fm["rho_pair_per_bin"][0])  # all equal (broadcast scalar)


# ---------------------------------------------------------------------------
# v2.6 Route A: profile-level block-bootstrap CI tests
# ---------------------------------------------------------------------------

def test_corrected_audit_pipeline_block_bootstrap_default_off():
    """n_block_bootstrap=0 (default) preserves existing 5-fold percentile CI behavior."""
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    # Run with n_block_bootstrap=0 (default) and n_block_bootstrap=0 explicit.
    result_default = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20, rho_pair_n_permutations=20, pa_n_bootstrap=20,
    )
    result_explicit_zero = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20, rho_pair_n_permutations=20, pa_n_bootstrap=20,
        n_block_bootstrap=0,
    )
    # Both should return identical results (same code path).
    assert result_default.point_estimate == result_explicit_zero.point_estimate
    assert result_default.ci_lower == result_explicit_zero.ci_lower
    assert result_default.ci_upper == result_explicit_zero.ci_upper


def test_corrected_audit_pipeline_block_bootstrap_point_unchanged():
    """Block-bootstrap doesn't change the point estimate — only the CI."""
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    # Baseline (no block-bootstrap)
    result_no_bb = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20, rho_pair_n_permutations=20, pa_n_bootstrap=20,
        n_block_bootstrap=0,
    )
    # With block-bootstrap (low rep count for test speed)
    result_bb = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20, rho_pair_n_permutations=20, pa_n_bootstrap=20,
        n_block_bootstrap=10, expected_block_length=7,
    )
    # Point estimate must be identical (computed on unresampled held-out data).
    assert result_no_bb.point_estimate == result_bb.point_estimate
    # CI shape: both have lo and hi (or both None).
    assert (result_no_bb.ci_lower is None) == (result_bb.ci_lower is None)


def test_corrected_audit_pipeline_block_bootstrap_ci_finite():
    """Block-bootstrap CI is finite (not NaN, not None) when n_block_bootstrap > 0 and folds >= 1."""
    profiles = _synthetic_profiles_5_seasons()
    pa_df = _synthetic_pa_5_seasons()

    result = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=[2021, 2022, 2023, 2024, 2025],
        mdp_solve_fn=_all_skip_policy,
        n_bootstrap=20, rho_pair_n_permutations=20, pa_n_bootstrap=20,
        n_block_bootstrap=10,
    )
    assert result.ci_lower is not None
    assert result.ci_upper is not None
    assert np.isfinite(result.ci_lower)
    assert np.isfinite(result.ci_upper)
    # CI must at least touch the point estimate (lo <= point or point <= hi within float tolerance).
    assert (
        result.ci_lower <= result.point_estimate + 1e-9
        or result.ci_upper >= result.point_estimate - 1e-9
    )
