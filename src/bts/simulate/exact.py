"""Exact P(57) computation via absorbing Markov chain.

For a fixed strategy (mapping from quality bin → action at each streak),
builds the transition matrix and computes exact P(reaching state 57)
within a finite number of plays. No Monte Carlo noise.
"""

import numpy as np

from bts.simulate.quality_bins import QualityBins
from bts.simulate.strategies import Strategy, get_thresholds


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
