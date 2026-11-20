"""Reachability MDP solver for optimal BTS strategy.

Finds the provably optimal action (skip/single/double) for every
possible state (streak, days_remaining, saver_available, quality_bin)
via backward induction. State space: 57 × 181 × 2 × N_bins ≈ 103K.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bts.simulate.quality_bins import QualityBins

ACTIONS = ("skip", "single", "double")

DEFAULT_POLICY_PATH = Path("data/models/mdp_policy.npz")


@dataclass(frozen=True)
class TransitionOutcome:
    """One probabilistic transition branch for a BTS MDP action."""

    next_streak: int
    saver_available: bool
    probability: float


def transition_outcomes(
    action: str | int,
    streak: int,
    saver_available: bool,
    *,
    p_hit: float,
    p_both: float,
    target: int = 57,
) -> tuple[TransitionOutcome, ...]:
    """Return next-state branches for one BTS decision.

    ``target`` caps successful transitions so first-passage evaluators count a
    double that crosses the threshold without landing exactly on it.
    """
    if isinstance(action, int):
        if action < 0:
            raise ValueError(f"invalid action index: {action}")
        try:
            action_name = ACTIONS[action]
        except IndexError as exc:
            raise ValueError(f"invalid action index: {action}") from exc
    else:
        action_name = action
        if action_name not in ACTIONS:
            raise ValueError(f"invalid action: {action_name!r}")

    capped_streak = min(streak, target)
    if action_name == "skip":
        return (TransitionOutcome(capped_streak, bool(saver_available), 1.0),)

    saver_catches_miss = bool(saver_available) and 10 <= streak <= 15
    miss_streak = streak if saver_catches_miss else 0
    miss_saver = False if saver_catches_miss else bool(saver_available)

    if action_name == "single":
        success_probability = float(p_hit)
        success_streak = min(streak + 1, target)
    else:
        success_probability = float(p_both)
        success_streak = min(streak + 2, target)

    return (
        TransitionOutcome(success_streak, bool(saver_available), success_probability),
        TransitionOutcome(min(miss_streak, target), miss_saver, 1.0 - success_probability),
    )


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

    def save(self, path: Path | str = DEFAULT_POLICY_PATH) -> Path:
        """Save policy table + bin boundaries to disk, atomically.

        Writes to a temp file in the destination directory, then ``os.replace``s it
        into place. A reader (``load_policy`` / ``strategy._load_mdp``) therefore
        never sees a half-written ``.npz`` even if the write is interrupted (killed
        process, disk full) mid-``savez``: the target path always points at a
        complete file, old or new. ``os.replace`` is atomic only within a single
        filesystem, so the temp lives in the destination's own directory. Passing an
        open file object to ``savez_compressed`` avoids its ``.npz`` auto-append, so
        the temp path is exactly what we replace.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(
                    f,
                    policy_table=self.policy_table,
                    boundaries=np.array(self.quality_bins.boundaries),
                    season_length=np.array(self.season_length),
                    optimal_p57=np.array(self.optimal_p57),
                )
            # mkstemp creates the temp 0600; restore the mode the previous direct write
            # produced so an atomic save never silently tightens perms (a reader running
            # as a different user would otherwise hit PermissionError, which _load_mdp
            # does not catch). Preserve an existing target's mode; else a normal
            # umask-respecting create.
            try:
                mode = path.stat().st_mode & 0o777
            except FileNotFoundError:
                umask = os.umask(0)
                os.umask(umask)
                mode = 0o666 & ~umask
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path


def load_policy(path: Path | str = DEFAULT_POLICY_PATH) -> tuple[np.ndarray, list[float], int]:
    """Load a saved MDP policy. Returns (policy_table, bin_boundaries, season_length)."""
    data = np.load(path)
    return (
        data["policy_table"],
        data["boundaries"].tolist(),
        int(data["season_length"]),
    )


def lookup_action(
    policy_table: np.ndarray,
    boundaries: list[float],
    streak: int,
    days_remaining: int,
    saver: bool,
    top_pick_confidence: float,
    season_length: int = 180,
) -> str:
    """Look up the optimal action from a saved MDP policy.

    This is the production entry point — called from strategy.py.
    """
    if streak >= 57 or days_remaining <= 0:
        return "skip"

    # Classify quality bin
    q = 0
    for i, boundary in enumerate(boundaries):
        if top_pick_confidence >= boundary:
            q = i + 1
    n_bins = len(boundaries) + 1
    q = min(q, n_bins - 1)

    d = min(days_remaining, season_length)
    s = min(streak, 56)
    return ACTIONS[policy_table[s, d, int(saver), q]]


def solve_mdp(
    bins: QualityBins,
    season_length: int = 180,
    late_bins: QualityBins | None = None,
    late_phase_days: int = 60,
) -> MDPSolution:
    """Solve the reachability MDP via backward induction.

    State: (streak, days_remaining, saver_available, quality_bin)
    Actions: skip, single, double
    Objective: maximize P(reaching streak 57)

    Args:
        bins: Quality bins for early phase (or all season if late_bins is None).
        late_bins: If provided, use these bins when days_remaining <= late_phase_days.
            Models the empirical observation that hit rates degrade in Aug-Sep.
        late_phase_days: Days remaining threshold for switching to late_bins.
    """
    n_streaks = 58   # 0-57
    n_days = season_length + 1  # 0 to season_length
    n_saver = 2      # 0=used/off, 1=available
    n_bins = len(bins.bins)

    # Precompute bin frequencies and transition probs for each phase
    freq_early = np.array([b.frequency for b in bins.bins])
    p_hit_early = np.array([b.p_hit for b in bins.bins])
    p_both_early = np.array([b.p_both for b in bins.bins])

    if late_bins is not None:
        freq_late = np.array([b.frequency for b in late_bins.bins])
        p_hit_late = np.array([b.p_hit for b in late_bins.bins])
        p_both_late = np.array([b.p_both for b in late_bins.bins])
    else:
        freq_late = freq_early
        p_hit_late = p_hit_early
        p_both_late = p_both_early

    # Value function and policy
    V = np.zeros((n_streaks, n_days, n_saver, n_bins))
    policy = np.zeros((n_streaks, n_days, n_saver, n_bins), dtype=np.int8)

    # Terminal condition: V(57, *, *, *) = 1
    V[57, :, :, :] = 1.0

    # Backward induction: d = 1..season_length
    for d in range(1, n_days):
        # Select bins for this day's phase
        is_late = (d <= late_phase_days) if late_bins is not None else False
        freq = freq_late if is_late else freq_early
        p_hit = p_hit_late if is_late else p_hit_early
        p_both = p_both_late if is_late else p_both_early

        for s in range(57):
            for saver in range(n_saver):
                for q in range(n_bins):
                    # Expected value over next day's quality for a given next state
                    # Next day is d-1 remaining — use ITS phase for the expectation
                    next_is_late = ((d - 1) <= late_phase_days) if late_bins is not None else False
                    next_freq = freq_late if next_is_late else freq_early

                    def ev(next_s, next_saver):
                        return float(np.dot(next_freq, V[next_s, d - 1, next_saver, :]))

                    ph = p_hit[q]
                    pb = p_both[q]

                    def action_value(action: int) -> float:
                        return sum(
                            branch.probability * ev(
                                branch.next_streak,
                                int(branch.saver_available),
                            )
                            for branch in transition_outcomes(
                                action,
                                s,
                                bool(saver),
                                p_hit=ph,
                                p_both=pb,
                            )
                        )

                    # Pick best action
                    values = [action_value(action) for action in range(len(ACTIONS))]
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
