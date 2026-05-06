"""Exact P(57) computation via absorbing Markov chain.

For a fixed strategy (mapping from quality bin → action at each streak),
builds the transition matrix and computes exact P(reaching state 57)
within a finite number of plays. No Monte Carlo noise.
"""

import numpy as np

from bts.simulate.quality_bins import QualityBins
from bts.simulate.strategies import Strategy, get_thresholds

ACTION_SKIP = 0
ACTION_SINGLE = 1
ACTION_DOUBLE = 2


def _resolve_action(strategy: Strategy, streak: int, qbin_p_range: tuple[float, float],
                     qbin_p_both: float) -> str:
    """Determine action (skip/single/double) for a strategy at a given streak and quality bin."""
    skip_thresh, double_thresh = get_thresholds(strategy, streak)

    # Use the midpoint of the bin's p_range as representative confidence
    mid_p = (qbin_p_range[0] + qbin_p_range[1]) / 2

    if skip_thresh is not None and mid_p < skip_thresh:
        return "skip"

    if double_thresh is not None and qbin_p_both >= double_thresh:
        return "double"

    return "single"


def build_transition_matrix(strategy: Strategy, bins: QualityBins) -> np.ndarray:
    """Build the 58x58 transition matrix for a given strategy.

    States 0-56 are transient (streak values), state 57 is absorbing (win).
    Each row sums to 1. Skip days contribute to self-loops (stay at current streak).

    Note: the saver is modeled as a fixed property of states 10-15 (always
    saves on first miss). This is an approximation — the full saver dynamics
    (consumed on use) require the MDP's richer state space.
    """
    n_states = 58  # 0-57
    T = np.zeros((n_states, n_states))

    # State 57 is absorbing
    T[57, 57] = 1.0

    for s in range(57):
        for qbin in bins.bins:
            action = _resolve_action(strategy, s, qbin.p_range, qbin.p_both)

            if action == "skip":
                T[s, s] += qbin.frequency
                continue

            p_hit = qbin.p_hit
            p_both = qbin.p_both

            # Saver logic: at states 10-15, a miss preserves the streak
            saver_active = strategy.streak_saver and 10 <= s <= 15

            if action == "single":
                next_s = min(s + 1, 57)
                T[s, next_s] += qbin.frequency * p_hit
                if saver_active:
                    T[s, s] += qbin.frequency * (1 - p_hit)
                else:
                    T[s, 0] += qbin.frequency * (1 - p_hit)

            elif action == "double":
                next_s = min(s + 2, 57)
                T[s, next_s] += qbin.frequency * p_both
                if saver_active:
                    T[s, s] += qbin.frequency * (1 - p_both)
                else:
                    T[s, 0] += qbin.frequency * (1 - p_both)

    return T


def exact_p57(strategy: Strategy, bins: QualityBins, season_length: int = 180) -> float:
    """Compute exact P(reaching streak 57) within a season.

    Builds the transition matrix for the strategy, then computes
    T^season_length[0, 57] via matrix exponentiation.
    """
    T = build_transition_matrix(strategy, bins)
    result = np.linalg.matrix_power(T, season_length)
    return float(result[0, 57])


def exact_p57_policy_table(
    policy_table: np.ndarray,
    bins: QualityBins,
    season_length: int | None = None,
) -> float:
    """Compute exact P(57) for a saved MDP policy table.

    Unlike :func:`exact_p57`, this uses the full MDP state carried by saved
    policies: streak, days remaining, saver availability, and quality bin.
    The policy is evaluated under the supplied evaluation-time bin manifold,
    which is uniform across days; if the policy was trained with phase-aware
    bins, training and evaluation transition models differ.
    """
    _validate_policy_table(policy_table, bins)
    if season_length is None:
        season_length = policy_table.shape[1] - 1
    season_length = min(int(season_length), policy_table.shape[1] - 1)

    n_streaks = 58
    n_days = season_length + 1
    n_saver = 2
    n_bins = len(bins.bins)
    freq = np.array([b.frequency for b in bins.bins])
    p_hit = np.array([b.p_hit for b in bins.bins])
    p_both = np.array([b.p_both for b in bins.bins])

    V = np.zeros((n_streaks, n_days, n_saver, n_bins))
    V[57, :, :, :] = 1.0

    for d in range(1, n_days):
        next_freq = freq

        def ev(next_s: int, next_saver: int) -> float:
            return float(np.dot(next_freq, V[next_s, d - 1, next_saver, :]))

        for s in range(57):
            for saver in range(n_saver):
                for q in range(n_bins):
                    action = int(policy_table[s, d, saver, q])
                    if action == ACTION_SKIP:
                        V[s, d, saver, q] = ev(s, saver)
                    elif action == ACTION_SINGLE:
                        next_hit = min(s + 1, 57)
                        ph = p_hit[q]
                        if saver and 10 <= s <= 15:
                            V[s, d, saver, q] = ph * ev(next_hit, saver) + (1 - ph) * ev(s, 0)
                        else:
                            V[s, d, saver, q] = ph * ev(next_hit, saver) + (1 - ph) * ev(0, saver)
                    elif action == ACTION_DOUBLE:
                        next_dbl = min(s + 2, 57)
                        pb = p_both[q]
                        if saver and 10 <= s <= 15:
                            V[s, d, saver, q] = pb * ev(next_dbl, saver) + (1 - pb) * ev(s, 0)
                        else:
                            V[s, d, saver, q] = pb * ev(next_dbl, saver) + (1 - pb) * ev(0, saver)
                    else:
                        raise ValueError(f"policy_table contains invalid action {action}; expected 0, 1, or 2")

    return float(np.dot(freq, V[0, season_length, 1, :]))


def _validate_policy_table(policy_table: np.ndarray, bins: QualityBins) -> None:
    if policy_table.ndim != 4:
        raise ValueError(f"policy_table must be 4D, got shape {policy_table.shape}")
    if policy_table.shape[0] < 57:
        raise ValueError(f"policy_table first dimension must cover streaks 0..56, got {policy_table.shape[0]}")
    if policy_table.shape[1] < 1:
        raise ValueError("policy_table must have at least one days_remaining column")
    if policy_table.shape[2] != 2:
        raise ValueError(f"policy_table saver dimension must be 2, got {policy_table.shape[2]}")
    if policy_table.shape[3] != len(bins.bins):
        raise ValueError(
            f"policy_table quality-bin dimension {policy_table.shape[3]} does not match "
            f"computed bins {len(bins.bins)}"
        )
