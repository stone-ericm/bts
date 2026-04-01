# Exact Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Monte Carlo noise with exact P(57) computation and find the provably optimal BTS strategy via MDP.

**Architecture:** Compute empirical quality bins from backtest profiles, build absorbing Markov chain for exact strategy evaluation, solve reachability MDP via backward induction for optimal policy. 103K state space, solvable in seconds.

**Tech Stack:** Python 3.12, numpy (matrix ops), pandas (profile loading), click/rich (CLI). All existing deps.

**Spec:** `docs/superpowers/specs/2026-04-01-exact-solver-design.md`

---

## File Structure

```
src/bts/simulate/
    quality_bins.py     — QualityBins dataclass, compute_bins() from profiles
    exact.py            — exact_p57() via absorbing Markov chain
    mdp.py              — solve_mdp() → MDPSolution with policy + extraction
    cli.py              — add solve + exact commands (modify existing)
    strategies.py       — add strategy_to_bin_actions() helper (modify existing)

tests/simulate/
    test_quality_bins.py
    test_exact.py
    test_mdp.py
```

---

### Task 1: Quality Bins

**Files:**
- Create: `src/bts/simulate/quality_bins.py`
- Create: `tests/simulate/test_quality_bins.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/simulate/test_quality_bins.py
"""Tests for empirical quality bin computation."""

import numpy as np
import pandas as pd
import pytest
from bts.simulate.quality_bins import QualityBin, QualityBins, compute_bins


def _make_profiles(n_days=100):
    """Create synthetic profile DataFrame with known distribution."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_days):
        date = f"2024-{(i // 28) + 4:02d}-{(i % 28) + 1:02d}"
        # Rank 1: confidence varies, hit rate correlates with confidence
        p1 = 0.75 + rng.random() * 0.15  # 0.75-0.90
        hit1 = 1 if rng.random() < (0.5 + p1 * 0.5) else 0
        # Rank 2: slightly lower
        p2 = p1 - 0.03 + rng.normal(0, 0.01)
        hit2 = 1 if rng.random() < (0.5 + p2 * 0.5) else 0
        for rank, p, hit in [(1, p1, hit1), (2, p2, hit2)]:
            rows.append({"date": date, "rank": rank, "batter_id": rank * 1000,
                          "p_game_hit": p, "actual_hit": hit, "n_pas": 4})
    return pd.DataFrame(rows)


class TestComputeBins:
    def test_returns_5_bins(self):
        df = _make_profiles()
        bins = compute_bins(df)
        assert isinstance(bins, QualityBins)
        assert len(bins.bins) == 5

    def test_frequencies_sum_to_1(self):
        df = _make_profiles()
        bins = compute_bins(df)
        total = sum(b.frequency for b in bins.bins)
        assert abs(total - 1.0) < 0.01

    def test_bins_ordered_by_confidence(self):
        df = _make_profiles()
        bins = compute_bins(df)
        for i in range(len(bins.bins) - 1):
            assert bins.bins[i].p_range[1] <= bins.bins[i + 1].p_range[1]

    def test_p_hit_between_0_and_1(self):
        df = _make_profiles()
        bins = compute_bins(df)
        for b in bins.bins:
            assert 0 <= b.p_hit <= 1
            assert 0 <= b.p_both <= 1

    def test_classify_returns_bin_index(self):
        df = _make_profiles()
        bins = compute_bins(df)
        idx = bins.classify(0.82)
        assert 0 <= idx <= 4

    def test_classify_extreme_values(self):
        df = _make_profiles()
        bins = compute_bins(df)
        assert bins.classify(0.50) == 0  # below all boundaries → lowest bin
        assert bins.classify(0.99) == 4  # above all boundaries → highest bin
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_quality_bins.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement quality_bins.py**

```python
# src/bts/simulate/quality_bins.py
"""Empirical prediction quality bins from backtest profiles.

Bins daily profiles into equal-frequency quintiles by top-pick confidence.
Each bin stores the empirical P(hit) and P(both hit) for use in the
absorbing chain and MDP solvers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityBin:
    """One quality tier with empirical transition probabilities."""
    index: int
    p_range: tuple[float, float]  # (min, max) of top-1 p_game_hit
    p_hit: float                  # empirical P(rank-1 gets a hit)
    p_both: float                 # empirical P(rank-1 AND rank-2 both hit)
    frequency: float              # fraction of days in this bin


@dataclass
class QualityBins:
    """Collection of quality bins with classification helper."""
    bins: list[QualityBin]
    boundaries: list[float]  # quintile cutpoints (4 values for 5 bins)

    def classify(self, p_game_hit: float) -> int:
        """Return bin index (0-4) for a given confidence value."""
        for i, boundary in enumerate(self.boundaries):
            if p_game_hit < boundary:
                return i
        return len(self.boundaries)  # highest bin


def compute_bins(profiles_df: pd.DataFrame, n_bins: int = 5) -> QualityBins:
    """Compute quality bins from backtest profile DataFrame.

    Args:
        profiles_df: DataFrame with columns [date, rank, p_game_hit, actual_hit].
        n_bins: Number of equal-frequency bins (default 5 = quintiles).

    Returns:
        QualityBins with empirical hit rates per bin.
    """
    r1 = profiles_df[profiles_df["rank"] == 1].copy()
    r2 = profiles_df[profiles_df["rank"] == 2].copy()

    # Merge rank-1 and rank-2 by date
    merged = r1[["date", "p_game_hit", "actual_hit"]].merge(
        r2[["date", "actual_hit"]].rename(columns={"actual_hit": "top2_hit"}),
        on="date",
    )

    # Compute quintile boundaries
    quantiles = [i / n_bins for i in range(1, n_bins)]
    boundaries = [float(merged["p_game_hit"].quantile(q)) for q in quantiles]

    # Assign bins
    merged["bin"] = np.digitize(merged["p_game_hit"], boundaries)

    bins = []
    for i in range(n_bins):
        group = merged[merged["bin"] == i]
        if len(group) == 0:
            continue
        bins.append(QualityBin(
            index=i,
            p_range=(float(group["p_game_hit"].min()), float(group["p_game_hit"].max())),
            p_hit=float(group["actual_hit"].mean()),
            p_both=float((group["actual_hit"] & group["top2_hit"]).mean()),
            frequency=len(group) / len(merged),
        ))

    return QualityBins(bins=bins, boundaries=boundaries)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_quality_bins.py -v`
Expected: All 6 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/quality_bins.py tests/simulate/test_quality_bins.py
git commit -m "feat: empirical quality bins from backtest profiles"
```

---

### Task 2: Absorbing Markov Chain (Exact P(57))

**Files:**
- Create: `src/bts/simulate/exact.py`
- Create: `tests/simulate/test_exact.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/simulate/test_exact.py
"""Tests for exact P(57) computation via absorbing Markov chain."""

import numpy as np
import pytest
from bts.simulate.quality_bins import QualityBin, QualityBins
from bts.simulate.exact import exact_p57, build_transition_matrix
from bts.simulate.strategies import Strategy


def _simple_bins():
    """One-bin QualityBins where every day is identical (p_hit=0.9, p_both=0.8)."""
    return QualityBins(
        bins=[QualityBin(index=0, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=1.0)],
        boundaries=[],
    )


def _two_bins():
    """Two bins: bad (50% hit, freq=0.3) and good (90% hit, freq=0.7)."""
    return QualityBins(
        bins=[
            QualityBin(index=0, p_range=(0.7, 0.8), p_hit=0.5, p_both=0.3, frequency=0.3),
            QualityBin(index=1, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=0.7),
        ],
        boundaries=[0.8],
    )


class TestBuildTransitionMatrix:
    def test_rows_sum_to_1(self):
        bins = _simple_bins()
        strategy = Strategy(name="always-single")
        T = build_transition_matrix(strategy, bins)
        # All rows except absorbing state (57) should sum to 1
        for s in range(57):
            assert abs(T[s].sum() - 1.0) < 1e-10, f"Row {s} sums to {T[s].sum()}"
        # Absorbing state stays at 57
        assert T[57, 57] == 1.0

    def test_shape(self):
        bins = _simple_bins()
        strategy = Strategy(name="test")
        T = build_transition_matrix(strategy, bins)
        assert T.shape == (58, 58)


class TestExactP57:
    def test_perfect_hit_rate_high_p57(self):
        """With 90% hit rate and 200 plays, P(57) should be substantial."""
        bins = _simple_bins()  # p_hit = 0.9
        strategy = Strategy(name="always-single")
        p = exact_p57(strategy, bins, season_length=200)
        assert p > 0.01  # should be meaningfully positive

    def test_zero_hit_rate_zero_p57(self):
        """With 0% hit rate, P(57) = 0."""
        bins = QualityBins(
            bins=[QualityBin(index=0, p_range=(0.5, 0.6), p_hit=0.0, p_both=0.0, frequency=1.0)],
            boundaries=[],
        )
        strategy = Strategy(name="test")
        p = exact_p57(strategy, bins, season_length=200)
        assert p == 0.0

    def test_more_plays_increases_p57(self):
        """More plays (longer season) should increase P(57)."""
        bins = _simple_bins()
        strategy = Strategy(name="test")
        p_short = exact_p57(strategy, bins, season_length=100)
        p_long = exact_p57(strategy, bins, season_length=300)
        assert p_long >= p_short

    def test_skip_strategy_reduces_plays(self):
        """A strategy that skips bad days should differ from always-play."""
        bins = _two_bins()  # bad=0.3freq, good=0.7freq
        always_play = Strategy(name="always", skip_threshold=None)
        skip_bad = Strategy(name="skip", skip_threshold=0.8)  # skips bin 0
        p_always = exact_p57(always_play, bins, season_length=180)
        p_skip = exact_p57(skip_bad, bins, season_length=180)
        # With 50% hit on bad days dragging down the average, skipping should help
        assert p_skip > p_always

    def test_doubling_changes_p57(self):
        """Doubling should produce different P(57) than singles-only."""
        bins = _simple_bins()
        singles = Strategy(name="singles")
        doubles = Strategy(name="doubles", double_threshold=0.50)
        p_singles = exact_p57(singles, bins, season_length=180)
        p_doubles = exact_p57(doubles, bins, season_length=180)
        assert p_singles != p_doubles

    def test_saver_increases_p57(self):
        """Streak saver should increase P(57)."""
        bins = _simple_bins()
        no_saver = Strategy(name="no-saver", streak_saver=False)
        with_saver = Strategy(name="saver", streak_saver=True)
        p_no = exact_p57(no_saver, bins, season_length=180)
        p_yes = exact_p57(with_saver, bins, season_length=180)
        assert p_yes >= p_no
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_exact.py -v`
Expected: ImportError

- [ ] **Step 3: Implement exact.py**

```python
# src/bts/simulate/exact.py
"""Exact P(57) computation via absorbing Markov chain.

For a fixed strategy (mapping from quality bin → action at each streak),
builds the transition matrix and computes exact P(reaching state 57)
within a finite number of plays. No Monte Carlo noise.
"""

import numpy as np

from bts.simulate.quality_bins import QualityBins
from bts.simulate.strategies import Strategy, get_thresholds


def _resolve_action(strategy: Strategy, streak: int, quality_bin_p_range: tuple[float, float],
                     quality_bin_p_both: float, double_possible: bool) -> str:
    """Determine action (skip/single/double) for a strategy at a given streak and quality bin."""
    skip_thresh, double_thresh = get_thresholds(strategy, streak)

    # Use the midpoint of the bin's p_range as representative confidence
    mid_p = (quality_bin_p_range[0] + quality_bin_p_range[1]) / 2

    if skip_thresh is not None and mid_p < skip_thresh:
        return "skip"

    if double_thresh is not None and double_possible:
        # P(both) for the bin — use the bin's p_both as representative
        # The double threshold compares against P(both hit), which we approximate
        # as needing the bin's midpoint to produce P(both) >= threshold
        # Simpler: if the bin's empirical p_both >= double_thresh, double
        if quality_bin_p_both >= double_thresh:
            return "double"

    return "single"


def build_transition_matrix(strategy: Strategy, bins: QualityBins) -> np.ndarray:
    """Build the 58×58 transition matrix for a given strategy.

    States 0-56 are transient (streak values), state 57 is absorbing (win).
    The matrix encodes one "play day" — skip days are handled by adjusting
    the effective number of plays.

    Returns (T, skip_rate) where skip_rate is the fraction of days skipped.
    """
    n_states = 58  # 0-57
    T = np.zeros((n_states, n_states))

    # State 57 is absorbing
    T[57, 57] = 1.0

    for s in range(57):
        for qbin in bins.bins:
            action = _resolve_action(strategy, s, qbin.p_range, qbin.p_both, True)

            if action == "skip":
                # Skip: streak holds, but this is a non-play day.
                # We handle skips by adjusting effective plays, not in the matrix.
                # In the matrix, skip days contribute to "stay at s" weighted by freq.
                T[s, s] += qbin.frequency
                continue

            p_hit = qbin.p_hit
            p_both = qbin.p_both

            # Saver logic
            saver_active = strategy.streak_saver and 10 <= s <= 15

            if action == "single":
                next_s = min(s + 1, 57)
                T[s, next_s] += qbin.frequency * p_hit
                if saver_active:
                    T[s, s] += qbin.frequency * (1 - p_hit)  # saver saves, stay at s
                else:
                    T[s, 0] += qbin.frequency * (1 - p_hit)  # reset

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

    # Matrix exponentiation: T^n gives n-step transition probabilities
    # Use repeated squaring for efficiency
    result = np.linalg.matrix_power(T, season_length)

    return float(result[0, 57])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_exact.py -v`
Expected: All 8 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/exact.py tests/simulate/test_exact.py
git commit -m "feat: exact P(57) via absorbing Markov chain"
```

---

### Task 3: MDP Solver

**Files:**
- Create: `src/bts/simulate/mdp.py`
- Create: `tests/simulate/test_mdp.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/simulate/test_mdp.py
"""Tests for reachability MDP solver."""

import numpy as np
import pytest
from bts.simulate.quality_bins import QualityBin, QualityBins
from bts.simulate.mdp import solve_mdp, MDPSolution


def _simple_bins():
    """One bin: p_hit=0.9, p_both=0.8, frequency=1.0."""
    return QualityBins(
        bins=[QualityBin(index=0, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=1.0)],
        boundaries=[],
    )


def _two_bins():
    """Two bins: bad (50% hit, freq=0.3) and good (90% hit, freq=0.7)."""
    return QualityBins(
        bins=[
            QualityBin(index=0, p_range=(0.7, 0.8), p_hit=0.5, p_both=0.3, frequency=0.3),
            QualityBin(index=1, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=0.7),
        ],
        boundaries=[0.8],
    )


class TestSolveMDP:
    def test_returns_mdp_solution(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        assert isinstance(sol, MDPSolution)
        assert 0 <= sol.optimal_p57 <= 1

    def test_terminal_state_value_is_1(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        # V(57, any_d, any_saver, any_q) should be 1.0
        for d in range(50):
            for saver in [0, 1]:
                for q in range(len(bins.bins)):
                    assert sol.value_table[57, d, saver, q] == 1.0

    def test_zero_days_value_is_0(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        # V(s<57, 0, any, any) should be 0.0
        for s in range(57):
            for saver in [0, 1]:
                for q in range(len(bins.bins)):
                    assert sol.value_table[s, 0, saver, q] == 0.0

    def test_optimal_p57_positive_with_good_bins(self):
        bins = _simple_bins()  # p_hit = 0.9
        sol = solve_mdp(bins, season_length=200)
        assert sol.optimal_p57 > 0.01

    def test_optimal_beats_or_matches_always_single(self):
        """MDP optimal should be >= any fixed strategy."""
        from bts.simulate.exact import exact_p57
        from bts.simulate.strategies import Strategy

        bins = _two_bins()
        sol = solve_mdp(bins, season_length=100)
        p_single = exact_p57(Strategy(name="single"), bins, season_length=100)
        assert sol.optimal_p57 >= p_single - 1e-10  # allow tiny float error

    def test_policy_returns_valid_action(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        action = sol.policy(streak=10, days_remaining=30, saver=True, quality_bin=0)
        assert action in ("skip", "single", "double")

    def test_more_days_increases_value(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=200)
        # Value at streak 0 should increase with more days remaining
        v_10 = sol.value_table[0, 10, 1, 0]
        v_100 = sol.value_table[0, 100, 1, 0]
        assert v_100 >= v_10

    def test_skip_optimal_for_bad_bin(self):
        """With a terrible bin (20% hit rate), the MDP should prefer skip."""
        bins = QualityBins(
            bins=[
                QualityBin(index=0, p_range=(0.5, 0.6), p_hit=0.2, p_both=0.05, frequency=0.3),
                QualityBin(index=1, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=0.7),
            ],
            boundaries=[0.7],
        )
        sol = solve_mdp(bins, season_length=100)
        # At any mid-streak with plenty of days, bad bin should be skipped
        action = sol.policy(streak=20, days_remaining=80, saver=False, quality_bin=0)
        assert action == "skip"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_mdp.py -v`
Expected: ImportError

- [ ] **Step 3: Implement mdp.py**

```python
# src/bts/simulate/mdp.py
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

        # Sample key (streak, days) combinations
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
                bin_labels = [f"Q{q+1}" for q in range(n_bins)]

                # Format as: streak=X: Q1=skip Q2=single Q3=double ...
                parts = [f"{bl}={a}" for bl, a in zip(bin_labels, actions_saver)]
                saver_str = " (saver)" if any(a != b for a, b in zip(actions_saver, actions_no_saver)) else ""
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
                    # Expected value over next day's quality for each next state
                    def ev(next_s, next_saver):
                        """E_q'[V(next_s, d-1, next_saver, q')]"""
                        return float(np.dot(freq, V[next_s, d - 1, next_saver, :]))

                    # Skip: streak holds, lose a day
                    v_skip = ev(s, saver)

                    # Single: hit → s+1, miss → reset or saver
                    ph = p_hit[q]
                    next_hit = min(s + 1, 57)
                    if saver and 10 <= s <= 15:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_mdp.py -v`
Expected: All 8 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/mdp.py tests/simulate/test_mdp.py
git commit -m "feat: reachability MDP solver for optimal BTS policy"
```

---

### Task 4: CLI Commands

**Files:**
- Modify: `src/bts/simulate/cli.py`
- Modify: `tests/simulate/test_monte_carlo.py` (add CLI tests)

- [ ] **Step 1: Write failing tests**

Add to `tests/simulate/test_monte_carlo.py`:

```python
class TestSolveCLI:
    def test_simulate_solve(self, tmp_path):
        """CLI runs MDP solver on saved profiles."""
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "solve", "--profiles-dir", str(tmp_path), "--season-length", "50",
        ])
        assert result.exit_code == 0
        assert "Optimal P(57)" in result.output
        assert "Heuristic P(57)" in result.output

    def test_simulate_exact(self, tmp_path):
        """CLI computes exact P(57) for a named strategy."""
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "exact", "--profiles-dir", str(tmp_path),
            "--strategy", "baseline", "--season-length", "50",
        ])
        assert result.exit_code == 0
        assert "baseline" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py::TestSolveCLI -v`
Expected: AttributeError (no `solve` or `exact` commands)

- [ ] **Step 3: Add solve and exact commands to cli.py**

Add to `src/bts/simulate/cli.py`, after the existing `run_sim` command:

```python
@simulate.command()
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest profile parquets")
@click.option("--season-length", default=180, type=int, help="Days per season")
def solve(profiles_dir: str, season_length: int):
    """Solve MDP for optimal strategy — exact P(57) and policy extraction."""
    from rich.console import Console
    from bts.simulate.quality_bins import compute_bins
    from bts.simulate.exact import exact_p57
    from bts.simulate.mdp import solve_mdp
    from bts.simulate.monte_carlo import load_all_profiles
    from bts.simulate.strategies import ALL_STRATEGIES

    import pandas as pd
    from pathlib import Path

    console = Console()
    profiles_path = Path(profiles_dir)

    # Load profiles and compute bins
    dfs = [pd.read_parquet(p) for p in sorted(profiles_path.glob("backtest_*.parquet"))]
    if not dfs:
        click.echo("No profile parquets found.", err=True)
        raise SystemExit(1)
    profiles_df = pd.concat(dfs, ignore_index=True)
    bins = compute_bins(profiles_df)

    console.print(f"[bold]Quality Bins ({len(bins.bins)} bins, "
                  f"{profiles_df['date'].nunique()} days)[/bold]")
    for b in bins.bins:
        console.print(f"  Q{b.index+1} [{b.p_range[0]:.3f}-{b.p_range[1]:.3f}]: "
                      f"P(hit)={b.p_hit:.1%}  P(both)={b.p_both:.1%}  freq={b.frequency:.1%}")

    # Solve MDP
    console.print(f"\n[bold]Solving MDP ({season_length} days)...[/bold]")
    sol = solve_mdp(bins, season_length=season_length)
    console.print(f"  [green]Optimal P(57) = {sol.optimal_p57:.4%}[/green]")

    # Compare with heuristic
    heuristic = ALL_STRATEGIES.get("combined", ALL_STRATEGIES["streak-aware"])
    p_heuristic = exact_p57(heuristic, bins, season_length=season_length)
    console.print(f"  Heuristic P(57) = {p_heuristic:.4%}")
    gap = sol.optimal_p57 - p_heuristic
    if gap > 0.0001:
        console.print(f"  [yellow]Gap: +{gap:.4%} — room for improvement[/yellow]")
    else:
        console.print(f"  [green]Gap: {gap:.4%} — heuristic is near-optimal[/green]")

    # Policy summary
    console.print(f"\n[bold]Policy Summary:[/bold]")
    console.print(sol.extract_thresholds())


@simulate.command()
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest profile parquets")
@click.option("--strategy", "strategy_name", default="combined",
              help="Strategy to evaluate (default: combined)")
@click.option("--season-length", default=180, type=int, help="Days per season")
def exact(profiles_dir: str, strategy_name: str, season_length: int):
    """Compute exact P(57) for a named strategy via absorbing chain."""
    from rich.console import Console
    from bts.simulate.quality_bins import compute_bins
    from bts.simulate.exact import exact_p57
    from bts.simulate.strategies import ALL_STRATEGIES

    import pandas as pd
    from pathlib import Path

    console = Console()
    profiles_path = Path(profiles_dir)

    if strategy_name not in ALL_STRATEGIES:
        click.echo(f"Unknown strategy: {strategy_name}. "
                   f"Options: {', '.join(ALL_STRATEGIES.keys())}", err=True)
        raise SystemExit(1)

    dfs = [pd.read_parquet(p) for p in sorted(profiles_path.glob("backtest_*.parquet"))]
    if not dfs:
        click.echo("No profile parquets found.", err=True)
        raise SystemExit(1)
    profiles_df = pd.concat(dfs, ignore_index=True)
    bins = compute_bins(profiles_df)

    strategy = ALL_STRATEGIES[strategy_name]
    p = exact_p57(strategy, bins, season_length=season_length)
    console.print(f"{strategy_name}: P(57) = {p:.4%}  (exact, {season_length}-day season)")
```

- [ ] **Step 4: Run CLI tests**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py::TestSolveCLI -v`
Expected: Both PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/cli.py tests/simulate/test_monte_carlo.py
git commit -m "feat: CLI commands for MDP solver and exact P(57)"
```

---

### Task 5: Run Solver and Validate

**Files:** No new files. Uses existing CLI.

- [ ] **Step 1: Compute exact P(57) for all heuristic strategies**

```bash
cd /Users/stone/projects/bts
for s in baseline current skip-conservative skip-aggressive sprint streak-aware combined; do
    UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate exact --strategy $s
done
```

Compare with Monte Carlo results from earlier. The exact values should fall within the Monte Carlo 95% CIs.

- [ ] **Step 2: Run MDP solver**

```bash
cd /Users/stone/projects/bts
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate solve
```

Examine:
- Optimal P(57) — how much better is it than our heuristic?
- Policy summary — where does the optimal policy differ from our thresholds?
- Quality of bins — do the empirical hit rates match expectations?

- [ ] **Step 3: Cross-validate MDP with Monte Carlo**

Run the MDP's optimal policy through Monte Carlo to verify they agree. This is manual — read the MDP policy, create a matching Strategy, run Monte Carlo:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
from bts.simulate.quality_bins import compute_bins
from bts.simulate.mdp import solve_mdp
from bts.simulate.monte_carlo import load_all_profiles, run_monte_carlo, DailyProfile
from bts.simulate.strategies import Strategy
from pathlib import Path
import pandas as pd

# Load and solve
dfs = [pd.read_parquet(p) for p in sorted(Path('data/simulation').glob('backtest_*.parquet'))]
profiles_df = pd.concat(dfs, ignore_index=True)
bins = compute_bins(profiles_df)
sol = solve_mdp(bins, season_length=180)

print(f'MDP Optimal P(57): {sol.optimal_p57:.4%}')
print(f'Policy summary: {sol.extract_thresholds()[:500]}')
"
```

- [ ] **Step 4: Document findings**

Update `ARCHITECTURE.md` with the MDP findings and `project_bts.md` memory with exact P(57) values.

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add -A
git commit -m "results: MDP solver validation and exact P(57) values"
```
