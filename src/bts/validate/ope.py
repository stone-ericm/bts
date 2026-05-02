"""Doubly Robust Off-Policy Evaluation for the BTS MDP.

References:
- Jiang & Li 2016. Doubly Robust Off-policy Value Evaluation for Reinforcement
  Learning. ICML.
- Le, Voloshin & Yue 2019. Batch Policy Learning under Constraints. ICML.
- Precup, Sutton & Singh 2000. Eligibility Traces for Off-Policy Policy
  Evaluation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def fitted_q_evaluation(
    df: pd.DataFrame,
    target_policy: Callable[[int, int], int],
    *,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> float:
    """Tabular FQE: estimate V^pi(s_0=0) via backward induction on observed transitions.

    Args:
        df: dataframe with columns t, s, a, sn, r.
        target_policy: callable (state, t) -> action.
        n_states, n_actions, horizon: MDP dimensions.

    Returns:
        Estimated V^pi(s=0) at t=0.
    """
    Q = np.zeros((horizon + 1, n_states, n_actions))
    counts = np.zeros((horizon, n_states, n_actions, n_states))
    rew_sum = np.zeros((horizon, n_states, n_actions, n_states))
    for row in df.itertuples():
        counts[row.t, row.s, row.a, row.sn] += 1
        rew_sum[row.t, row.s, row.a, row.sn] += row.r
    P = np.zeros_like(counts)
    R = np.zeros_like(counts)
    for t in range(horizon):
        for s in range(n_states):
            for a in range(n_actions):
                tot = counts[t, s, a].sum()
                if tot > 0:
                    P[t, s, a] = counts[t, s, a] / tot
                    R[t, s, a] = rew_sum[t, s, a] / np.maximum(counts[t, s, a], 1)
    for t in reversed(range(horizon)):
        for s in range(n_states):
            for a in range(n_actions):
                v_next = sum(
                    P[t, s, a, sn] * (R[t, s, a, sn] + Q[t + 1, sn, target_policy(sn, t + 1)])
                    for sn in range(n_states)
                )
                Q[t, s, a] = v_next
    return float(Q[0, 0, target_policy(0, 0)])


@dataclass
class DROPEResult:
    """Result of one DR-OPE evaluation."""

    point_estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_trajectories: int = 0
    nuisance_v_hat: float | None = None
    bootstrap_distribution: np.ndarray | None = None


def dr_ope_full_information(
    df: pd.DataFrame,
    target_policy: Callable[[int, int], int],
    *,
    n_states: int,
    n_actions: int,
    horizon: int,
) -> float:
    """DR-OPE estimator under full-information action replay (rho=1).

    For each trajectory i:
        V_DR_i = V_hat(s_0)
                 + sum_t [r_t + V_hat(s_{t+1}) − Q_hat(s_t, a_t)]

    Returns the mean V_DR_i across trajectories.

    Assumes full-information replay where the target policy's action outcome is
    observed for each (s, t) — appropriate for BTS where rank-1, rank-2, and
    skip outcomes are all logged daily.
    """
    counts = np.zeros((horizon, n_states, n_actions, n_states))
    rew_sum = np.zeros((horizon, n_states, n_actions, n_states))
    for row in df.itertuples():
        counts[row.t, row.s, row.a, row.sn] += 1
        rew_sum[row.t, row.s, row.a, row.sn] += row.r
    P = np.zeros_like(counts)
    R = np.zeros_like(counts)
    for t in range(horizon):
        for s in range(n_states):
            for a in range(n_actions):
                tot = counts[t, s, a].sum()
                if tot > 0:
                    P[t, s, a] = counts[t, s, a] / tot
                    R[t, s, a] = rew_sum[t, s, a] / np.maximum(counts[t, s, a], 1)
    Q = np.zeros((horizon + 1, n_states, n_actions))
    for t in reversed(range(horizon)):
        for s in range(n_states):
            for a in range(n_actions):
                Q[t, s, a] = sum(
                    P[t, s, a, sn] * (R[t, s, a, sn] + Q[t + 1, sn, target_policy(sn, t + 1)])
                    for sn in range(n_states)
                )
    V = np.array([
        [Q[t, s, target_policy(s, t)] for s in range(n_states)]
        for t in range(horizon + 1)
    ])

    v_dr_values = []
    for traj_id, traj in df.groupby("trajectory_id"):
        traj = traj.sort_values("t")
        v_correction = 0.0
        for row in traj.itertuples():
            target_a = target_policy(row.s, row.t)
            if row.a == target_a:
                v_next = V[row.t + 1, row.sn]
                q_t = Q[row.t, row.s, row.a]
                v_correction += (row.r + v_next - q_t)
        v_dr_i = V[0, 0] + v_correction
        v_dr_values.append(v_dr_i)
    return float(np.mean(v_dr_values))
