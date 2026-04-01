"""Reachability MDP solver for optimal BTS strategy.

Finds the provably optimal action (skip/single/double) for every
possible state (streak, days_remaining, saver_available, quality_bin)
via backward induction. State space: 57 × 181 × 2 × N_bins ≈ 103K.
"""

from dataclasses import dataclass

import numpy as np

from bts.simulate.quality_bins import QualityBins

ACTIONS = ("skip", "single", "double")


@dataclass
class MDPSolution:
    """Result of MDP solve: optimal value function and policy."""
    optimal_p57: float
    value_table: np.ndarray    # shape: (58, season_length+1, 2, n_bins)
    policy_table: np.ndarray   # shape: (57, season_length+1, 2, n_bins), dtype=int
    quality_bins: QualityBins
    season_length: int

    def policy(self, streak: int, days_remaining: int, saver: bool, quality_bin: int) -> str:
        """Return optimal action for a given state."""
        if streak >= 57 or days_remaining <= 0:
            return "skip"
        d = min(days_remaining, self.season_length)
        return ACTIONS[self.policy_table[streak, d, int(saver), quality_bin]]

    def extract_thresholds(self) -> str:
        """Summarize the policy as human-readable threshold patterns."""
        lines = []
        n_bins = len(self.quality_bins.bins)
        bin_labels = [f"Q{b.index + 1}" for b in self.quality_bins.bins]

        sample_days = [20, 50, 80, 120, 160]
        sample_streaks = [0, 5, 10, 15, 20, 30, 40, 50, 55]

        for d in sample_days:
            if d > self.season_length:
                continue
            lines.append(f"\n  Days remaining = {d}:")
            for s in sample_streaks:
                if s >= 57:
                    continue
                actions_saver = [self.policy(s, d, True, q) for q in range(n_bins)]
                actions_no_saver = [self.policy(s, d, False, q) for q in range(n_bins)]

                parts = [f"{bl}={a}" for bl, a in zip(bin_labels, actions_saver)]
                saver_diff = any(a != b for a, b in zip(actions_saver, actions_no_saver))
                saver_str = " (saver differs)" if saver_diff else ""
                lines.append(f"    streak={s:2d}: {' '.join(parts)}{saver_str}")

        return "\n".join(lines)


def solve_mdp(bins: QualityBins, season_length: int = 180) -> MDPSolution:
    """Solve the reachability MDP via backward induction.

    State: (streak, days_remaining, saver_available, quality_bin)
    Actions: skip, single, double
    Objective: maximize P(reaching streak 57)
    """
    n_streaks = 58   # 0-57
    n_days = season_length + 1  # 0 to season_length
    n_saver = 2      # 0=used/off, 1=available
    n_bins = len(bins.bins)

    # Precompute bin frequencies and transition probs
    freq = np.array([b.frequency for b in bins.bins])
    p_hit = np.array([b.p_hit for b in bins.bins])
    p_both = np.array([b.p_both for b in bins.bins])

    # Value function and policy
    V = np.zeros((n_streaks, n_days, n_saver, n_bins))
    policy = np.zeros((n_streaks, n_days, n_saver, n_bins), dtype=np.int8)

    # Terminal condition: V(57, *, *, *) = 1
    V[57, :, :, :] = 1.0

    # Backward induction: d = 1..season_length
    for d in range(1, n_days):
        for s in range(57):
            for saver in range(n_saver):
                for q in range(n_bins):
                    # Expected value over next day's quality for a given next state
                    def ev(next_s, next_saver):
                        return float(np.dot(freq, V[next_s, d - 1, next_saver, :]))

                    # Skip: streak holds, lose a day
                    v_skip = ev(s, saver)

                    # Single: hit → s+1, miss → reset or saver
                    ph = p_hit[q]
                    next_hit = min(s + 1, 57)
                    if saver and 10 <= s <= 15:
                        # Saver catches the miss: streak holds, saver consumed
                        v_single = ph * ev(next_hit, saver) + (1 - ph) * ev(s, 0)
                    else:
                        v_single = ph * ev(next_hit, saver) + (1 - ph) * ev(0, saver)

                    # Double: both hit → s+2, any miss → reset or saver
                    pb = p_both[q]
                    next_dbl = min(s + 2, 57)
                    if saver and 10 <= s <= 15:
                        v_double = pb * ev(next_dbl, saver) + (1 - pb) * ev(s, 0)
                    else:
                        v_double = pb * ev(next_dbl, saver) + (1 - pb) * ev(0, saver)

                    # Pick best action
                    values = [v_skip, v_single, v_double]
                    best = int(np.argmax(values))
                    V[s, d, saver, q] = values[best]
                    policy[s, d, saver, q] = best

    # Optimal P(57) = E_q[V(0, season_length, saver=1, q)]
    optimal_p57 = float(np.dot(freq, V[0, season_length, 1, :]))

    return MDPSolution(
        optimal_p57=optimal_p57,
        value_table=V,
        policy_table=policy,
        quality_bins=bins,
        season_length=season_length,
    )
