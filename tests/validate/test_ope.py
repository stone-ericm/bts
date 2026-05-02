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
