"""Tests for cross-fitted DR-OPE on the BTS MDP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.validate.ope import fitted_q_evaluation


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
