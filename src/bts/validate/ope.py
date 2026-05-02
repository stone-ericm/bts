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
