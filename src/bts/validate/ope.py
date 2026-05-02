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
from dataclasses import dataclass

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


def stationary_bootstrap_indices(
    n_days: int,
    *,
    expected_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis & Romano 1994 stationary bootstrap.

    Resamples a length-n_days index array using geometric block lengths with
    expected length `expected_block_length`. Wraps around the day axis.

    Reference: Politis & Romano 1994, "The Stationary Bootstrap." JASA.
    """
    p = 1.0 / expected_block_length
    out = np.empty(n_days, dtype=np.int64)
    out[0] = rng.integers(n_days)
    for i in range(1, n_days):
        if rng.random() < p:
            out[i] = rng.integers(n_days)
        else:
            out[i] = (out[i - 1] + 1) % n_days
    return out


def paired_hierarchical_bootstrap_sample(
    df: pd.DataFrame,
    *,
    expected_block_length: int = 7,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Resample the day axis with stationary bootstrap; keep all seeds per day together.

    The dataframe must have at least: 'season', 'date', 'seed', plus payload columns.
    Within each season, dates are resampled via stationary bootstrap. For each
    resampled date, ALL rows (across seeds and other within-day groupings) are
    included — the day is the unit of dependence in BTS, so all 24 seeds share
    the realized baseball outcomes for that day.

    The output date column is reassigned to the *slot* date (i.e., the original date
    at position i in the sorted unique-dates array), not the source date that was
    drawn. This ensures each output date slot appears exactly once, preserving the
    block-contiguous temporal structure while allowing repeated draws to be identified
    by their assigned slot rather than their source date.
    """
    out_chunks = []
    for season, season_df in df.groupby("season"):
        unique_dates = season_df["date"].drop_duplicates().sort_values().to_numpy()
        n_days = len(unique_dates)
        idx = stationary_bootstrap_indices(
            n_days, expected_block_length=expected_block_length, rng=rng
        )
        resampled_dates = unique_dates[idx]
        for slot_date, source_date in zip(unique_dates, resampled_dates):
            chunk = season_df[season_df["date"] == source_date].copy()
            chunk["date"] = slot_date
            out_chunks.append(chunk)
    return pd.concat(out_chunks, ignore_index=True)


def dr_ope_with_bootstrap(
    df: pd.DataFrame,
    target_policy: "Callable[[int, int], int]",
    *,
    n_states: int,
    n_actions: int,
    horizon: int,
    n_bootstrap: int = 2000,
    expected_block_length: int = 7,
    seed: int = 42,
    alpha: float = 0.05,
) -> "DROPEResult":
    """DR-OPE with paired hierarchical block bootstrap CI.

    Computes the DR-OPE point estimate via dr_ope_full_information, then
    n_bootstrap paired-hierarchical resamples to estimate (1 - alpha) percentile CI.
    """
    point = dr_ope_full_information(
        df, target_policy, n_states=n_states, n_actions=n_actions, horizon=horizon
    )
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        bs_df = paired_hierarchical_bootstrap_sample(
            df, expected_block_length=expected_block_length, rng=rng
        )
        bootstrap_values[b] = dr_ope_full_information(
            bs_df, target_policy, n_states=n_states, n_actions=n_actions, horizon=horizon
        )
    lo = float(np.quantile(bootstrap_values, alpha / 2))
    hi = float(np.quantile(bootstrap_values, 1 - alpha / 2))
    return DROPEResult(
        point_estimate=point,
        ci_lower=lo,
        ci_upper=hi,
        n_trajectories=df["trajectory_id"].nunique() if "trajectory_id" in df.columns else 0,
        bootstrap_distribution=bootstrap_values,
    )
