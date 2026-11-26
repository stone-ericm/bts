# Strategy Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Monte Carlo strategy simulator that compares streak-maximizing strategies against blend backtest predictions from 5 seasons (2021-2025).

**Architecture:** Extract daily prediction profiles from blend walk-forward backtest, bootstrap synthetic seasons by sampling profiles with replacement, simulate strategy profiles (skip days, dynamic doubling, streak-aware thresholds, streak saver) against each season, report P(57) and streak distributions.

**Tech Stack:** Python 3.12, pandas, numpy, lightgbm, click, rich (all existing deps)

**Spec:** `docs/superpowers/specs/2026-04-01-strategy-simulator-design.md`

---

## File Structure

```
src/bts/simulate/
    __init__.py              — empty
    strategies.py            — Strategy dataclass + profile definitions + threshold resolution
    monte_carlo.py           — simulate_season(), run_monte_carlo(), run_replay()
    backtest_blend.py        — blend_walk_forward() + save daily profiles to parquet
    cli.py                   — Click commands: bts simulate backtest, bts simulate run

tests/simulate/
    __init__.py              — empty
    test_strategies.py       — threshold resolution, streak-aware config
    test_monte_carlo.py      — simulation loop with deterministic profiles, Monte Carlo output shape
    test_backtest_blend.py   — output schema validation
```

---

### Task 1: Strategy Profiles

**Files:**
- Create: `src/bts/simulate/__init__.py`
- Create: `src/bts/simulate/strategies.py`
- Create: `tests/simulate/__init__.py`
- Create: `tests/simulate/test_strategies.py`

- [ ] **Step 1: Write failing tests for strategy definitions and threshold resolution**

```python
# tests/simulate/test_strategies.py
"""Tests for strategy profile definitions and threshold resolution."""

import pytest
from bts.simulate.strategies import Strategy, get_thresholds, ALL_STRATEGIES


class TestStrategy:
    def test_baseline_has_no_skip_no_double(self):
        s = ALL_STRATEGIES["baseline"]
        assert s.skip_threshold is None
        assert s.double_threshold is None
        assert s.streak_saver is True

    def test_current_has_double_at_065(self):
        s = ALL_STRATEGIES["current"]
        assert s.skip_threshold is None
        assert s.double_threshold == 0.65

    def test_sprint_has_aggressive_double(self):
        s = ALL_STRATEGIES["sprint"]
        assert s.double_threshold == 0.50


class TestGetThresholds:
    def test_flat_strategy_ignores_streak(self):
        s = ALL_STRATEGIES["current"]
        skip, double = get_thresholds(s, streak=0)
        assert skip is None
        assert double == 0.65
        skip2, double2 = get_thresholds(s, streak=40)
        assert skip2 is None
        assert double2 == 0.65

    def test_streak_aware_early_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=5)
        assert skip is None
        assert double == 0.55

    def test_streak_aware_saver_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=12)
        assert skip is None
        assert double == 0.60

    def test_streak_aware_protect_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=25)
        assert skip == 0.78
        assert double == 0.65

    def test_streak_aware_lockdown_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=35)
        assert skip == 0.80
        assert double is None

    def test_streak_aware_sprint_finish(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=50)
        assert skip == 0.78
        assert double == 0.60

    def test_all_strategies_registered(self):
        expected = {
            "baseline", "current", "skip-conservative", "skip-aggressive",
            "sprint", "streak-aware", "combined",
        }
        assert set(ALL_STRATEGIES.keys()) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_strategies.py -v`
Expected: ModuleNotFoundError for `bts.simulate.strategies`

- [ ] **Step 3: Implement strategies.py**

```python
# src/bts/simulate/strategies.py
"""Strategy profiles for BTS streak simulation.

Each strategy defines skip/double thresholds, optionally varying by
current streak length. The simulator resolves thresholds via get_thresholds().
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Strategy:
    """A BTS pick strategy profile.

    Attributes:
        name: Profile identifier.
        skip_threshold: Min p_game_hit of top pick to play. None = never skip.
        double_threshold: Min P(both hit) to double down. None = never double.
        streak_saver: Whether to model the streak saver mechanic.
        streak_config: Streak-aware overrides. List of (max_streak, skip, double)
            tuples sorted by max_streak ascending. The first entry where
            current_streak <= max_streak is used. Overrides flat thresholds.
    """
    name: str
    skip_threshold: float | None = None
    double_threshold: float | None = None
    streak_saver: bool = True
    streak_config: tuple[tuple[int, float | None, float | None], ...] | None = None


def get_thresholds(strategy: Strategy, streak: int) -> tuple[float | None, float | None]:
    """Resolve skip and double thresholds for a given streak length.

    Returns (skip_threshold, double_threshold).
    """
    if strategy.streak_config is not None:
        for max_streak, skip, double in strategy.streak_config:
            if streak <= max_streak:
                return skip, double
        # Past last config entry — use the last one
        _, skip, double = strategy.streak_config[-1]
        return skip, double
    return strategy.skip_threshold, strategy.double_threshold


# --- Streak-aware config ---
# (max_streak, skip_threshold, double_threshold)
_STREAK_AWARE_CONFIG = (
    (9, None, 0.55),     # aggressive start, less to lose
    (15, None, 0.60),    # saver protects, moderate doubling
    (30, 0.78, 0.65),    # tighten up
    (45, 0.80, None),    # singles only, skip bad days
    (56, 0.78, 0.60),    # sprint to finish
)

ALL_STRATEGIES: dict[str, Strategy] = {
    "baseline": Strategy(name="baseline"),
    "current": Strategy(name="current", double_threshold=0.65),
    "skip-conservative": Strategy(name="skip-conservative", skip_threshold=0.78, double_threshold=0.65),
    "skip-aggressive": Strategy(name="skip-aggressive", skip_threshold=0.82, double_threshold=0.65),
    "sprint": Strategy(name="sprint", double_threshold=0.50),
    "streak-aware": Strategy(name="streak-aware", streak_config=_STREAK_AWARE_CONFIG),
    "combined": Strategy(name="combined", streak_config=_STREAK_AWARE_CONFIG),
}
```

Also create empty `__init__.py` files:
```python
# src/bts/simulate/__init__.py
# (empty)
```
```python
# tests/simulate/__init__.py
# (empty)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_strategies.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/__init__.py src/bts/simulate/strategies.py tests/simulate/__init__.py tests/simulate/test_strategies.py
git commit -m "feat: strategy profiles for streak simulation"
```

---

### Task 2: Simulation Engine — Core Loop

**Files:**
- Create: `src/bts/simulate/monte_carlo.py`
- Create: `tests/simulate/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests for simulate_season**

```python
# tests/simulate/test_monte_carlo.py
"""Tests for Monte Carlo streak simulation."""

import numpy as np
import pytest
from bts.simulate.strategies import Strategy, ALL_STRATEGIES
from bts.simulate.monte_carlo import DailyProfile, simulate_season, SeasonResult


def _profile(top1_p: float, top1_hit: int, top2_p: float = 0.70, top2_hit: int = 1) -> DailyProfile:
    """Create a daily profile for testing."""
    return DailyProfile(top1_p=top1_p, top1_hit=top1_hit, top2_p=top2_p, top2_hit=top2_hit)


class TestSimulateSeason:
    def test_all_hits_produces_full_streak(self):
        """10 days, all hits, no skipping → streak of 10."""
        profiles = [_profile(0.85, 1)] * 10
        strategy = ALL_STRATEGIES["baseline"]
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 10
        assert result.play_days == 10

    def test_miss_resets_streak(self):
        """Hit, hit, miss, hit → max streak 2."""
        profiles = [
            _profile(0.85, 1),
            _profile(0.85, 1),
            _profile(0.85, 0),
            _profile(0.85, 1),
        ]
        strategy = ALL_STRATEGIES["baseline"]
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 2

    def test_skip_preserves_streak(self):
        """With skip threshold 0.80: high-conf hit, low-conf skip, high-conf hit → streak 2."""
        profiles = [
            _profile(0.85, 1),
            _profile(0.75, 0),  # below threshold AND would miss — but we skip
            _profile(0.85, 1),
        ]
        strategy = Strategy(name="test-skip", skip_threshold=0.80)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 2
        assert result.play_days == 2

    def test_double_down_advances_by_two(self):
        """Both hit on a double → streak advances by 2."""
        profiles = [_profile(0.85, 1, 0.82, 1)] * 5
        strategy = Strategy(name="test-double", double_threshold=0.50)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 10  # 5 days × 2
        assert result.play_days == 5

    def test_double_down_miss_resets(self):
        """One miss in a double → reset."""
        profiles = [
            _profile(0.85, 1, 0.82, 1),
            _profile(0.85, 1, 0.82, 0),  # second pick misses
            _profile(0.85, 1, 0.82, 1),
        ]
        strategy = Strategy(name="test-double", double_threshold=0.50)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 2  # first day

    def test_double_threshold_prevents_double(self):
        """P(both) below threshold → single pick only."""
        profiles = [_profile(0.75, 1, 0.70, 1)] * 3
        # P(both) = 0.75 * 0.70 = 0.525, below 0.65 threshold
        strategy = Strategy(name="test-high-thresh", double_threshold=0.65)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 3  # singles only

    def test_streak_saver_saves_at_10(self):
        """10 hits then a miss → saver preserves streak at 10."""
        profiles = [_profile(0.85, 1)] * 10 + [_profile(0.85, 0)] + [_profile(0.85, 1)] * 3
        strategy = Strategy(name="test-saver", streak_saver=True)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 13  # 10 + saved + 3 more
        assert result.streak_saver_used is True

    def test_streak_saver_does_not_save_above_15(self):
        """16 hits then a miss → no save, reset."""
        profiles = [_profile(0.85, 1)] * 16 + [_profile(0.85, 0)]
        strategy = Strategy(name="test-saver", streak_saver=True)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 16
        assert result.streak_saver_used is False  # wasn't eligible

    def test_streak_saver_only_fires_once(self):
        """Save at 10, rebuild to 12, miss again → reset."""
        profiles = (
            [_profile(0.85, 1)] * 10  # streak = 10
            + [_profile(0.85, 0)]      # saved at 10
            + [_profile(0.85, 1)] * 2  # streak = 12
            + [_profile(0.85, 0)]      # no save, reset
            + [_profile(0.85, 1)] * 5  # new streak = 5
        )
        strategy = Strategy(name="test-saver", streak_saver=True)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 12

    def test_empty_profiles(self):
        result = simulate_season([], ALL_STRATEGIES["baseline"])
        assert result.max_streak == 0
        assert result.play_days == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py -v`
Expected: ImportError

- [ ] **Step 3: Implement simulation engine**

```python
# src/bts/simulate/monte_carlo.py
"""Monte Carlo streak simulation engine.

Simulates BTS seasons under different strategy profiles by replaying
or bootstrapping from historical daily prediction profiles.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from bts.simulate.strategies import Strategy, get_thresholds


class DailyProfile(NamedTuple):
    """One day's prediction data for simulation."""
    top1_p: float   # blend P(game hit) for rank-1 pick
    top1_hit: int   # 1 if rank-1 actually got a hit, 0 otherwise
    top2_p: float   # blend P(game hit) for rank-2 pick
    top2_hit: int   # 1 if rank-2 actually got a hit, 0 otherwise


@dataclass
class SeasonResult:
    """Result of simulating one season."""
    max_streak: int
    play_days: int
    streak_saver_used: bool


def simulate_season(profiles: list[DailyProfile], strategy: Strategy) -> SeasonResult:
    """Simulate one BTS season under a strategy.

    Args:
        profiles: Ordered daily profiles (one per game day).
        strategy: Strategy to apply.

    Returns:
        SeasonResult with max streak achieved and stats.
    """
    streak = 0
    max_streak = 0
    play_days = 0
    saver_available = strategy.streak_saver
    saver_used = False

    for day in profiles:
        skip_thresh, double_thresh = get_thresholds(strategy, streak)

        # Skip check
        if skip_thresh is not None and day.top1_p < skip_thresh:
            continue  # streak holds

        play_days += 1

        # Double check
        doubling = False
        if double_thresh is not None:
            p_both = day.top1_p * day.top2_p
            if p_both >= double_thresh:
                doubling = True

        # Resolve outcome
        if doubling:
            if day.top1_hit and day.top2_hit:
                streak += 2
            else:
                # Miss — check streak saver
                if saver_available and not saver_used and 10 <= streak <= 15:
                    saver_used = True
                    # streak holds
                else:
                    streak = 0
        else:
            if day.top1_hit:
                streak += 1
            else:
                if saver_available and not saver_used and 10 <= streak <= 15:
                    saver_used = True
                else:
                    streak = 0

        max_streak = max(max_streak, streak)

    return SeasonResult(
        max_streak=max_streak,
        play_days=play_days,
        streak_saver_used=saver_used,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/monte_carlo.py tests/simulate/test_monte_carlo.py
git commit -m "feat: simulation engine for BTS streak strategies"
```

---

### Task 3: Monte Carlo Wrapper + Replay

**Files:**
- Modify: `src/bts/simulate/monte_carlo.py`
- Modify: `tests/simulate/test_monte_carlo.py`

- [ ] **Step 1: Write failing tests for run_monte_carlo and run_replay**

Add to `tests/simulate/test_monte_carlo.py`:

```python
from bts.simulate.monte_carlo import (
    DailyProfile, simulate_season, SeasonResult,
    run_monte_carlo, MonteCarloResult, load_profiles, run_replay, ReplayResult,
)

import pandas as pd


def _make_profile_df(n_days: int = 30, hit_rate: float = 0.85) -> pd.DataFrame:
    """Create a synthetic backtest profile DataFrame."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_days):
        date = f"2024-{(i // 28) + 4:02d}-{(i % 28) + 1:02d}"
        for rank in range(1, 11):
            p = max(0.5, 0.90 - rank * 0.02 + rng.normal(0, 0.02))
            hit = 1 if rng.random() < hit_rate else 0
            rows.append({"date": date, "rank": rank, "batter_id": rank * 1000,
                          "p_game_hit": p, "actual_hit": hit, "n_pas": 4})
    return pd.DataFrame(rows)


class TestLoadProfiles:
    def test_extracts_top2_per_day(self):
        df = _make_profile_df(n_days=5)
        profiles = load_profiles(df)
        assert len(profiles) == 5
        assert all(isinstance(p, DailyProfile) for p in profiles)

    def test_profiles_use_rank_1_and_2(self):
        df = _make_profile_df(n_days=1)
        profiles = load_profiles(df)
        r1 = df[df["rank"] == 1].iloc[0]
        r2 = df[df["rank"] == 2].iloc[0]
        assert profiles[0].top1_p == r1["p_game_hit"]
        assert profiles[0].top1_hit == r1["actual_hit"]
        assert profiles[0].top2_p == r2["p_game_hit"]
        assert profiles[0].top2_hit == r2["actual_hit"]


class TestRunMonteCarlo:
    def test_returns_correct_shape(self):
        df = _make_profile_df(n_days=60)
        profiles = load_profiles(df)
        result = run_monte_carlo(profiles, ALL_STRATEGIES["baseline"], n_trials=100, season_length=30)
        assert isinstance(result, MonteCarloResult)
        assert result.n_trials == 100
        assert len(result.max_streaks) == 100
        assert 0 <= result.p_57 <= 1
        assert result.median_streak >= 0
        assert result.p95_streak >= result.median_streak

    def test_perfect_hit_rate_reaches_57(self):
        """If every profile is a hit, P(57) should be 1.0 with enough days."""
        profiles = [_profile(0.90, 1)] * 60
        result = run_monte_carlo(profiles, ALL_STRATEGIES["baseline"], n_trials=50, season_length=60)
        assert result.p_57 == 1.0

    def test_zero_hit_rate_never_reaches_57(self):
        profiles = [_profile(0.50, 0)] * 60
        result = run_monte_carlo(profiles, ALL_STRATEGIES["baseline"], n_trials=50, season_length=60)
        assert result.p_57 == 0.0


class TestRunReplay:
    def test_replays_each_season(self):
        season_profiles = {
            2024: [_profile(0.85, 1)] * 10 + [_profile(0.85, 0)] + [_profile(0.85, 1)] * 5,
            2025: [_profile(0.85, 1)] * 20,
        }
        results = run_replay(season_profiles, ALL_STRATEGIES["baseline"])
        assert len(results) == 2
        assert 2024 in results
        assert 2025 in results
        assert results[2025].max_streak == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py::TestLoadProfiles tests/simulate/test_monte_carlo.py::TestRunMonteCarlo tests/simulate/test_monte_carlo.py::TestRunReplay -v`
Expected: ImportError for `load_profiles`, `run_monte_carlo`, etc.

- [ ] **Step 3: Implement Monte Carlo wrapper and replay**

Add to `src/bts/simulate/monte_carlo.py`:

```python
@dataclass
class MonteCarloResult:
    """Aggregated results from N simulated seasons."""
    n_trials: int
    max_streaks: list[int]
    p_57: float
    p_30: float
    p_20: float
    median_streak: int
    p95_streak: int
    mean_play_days: float
    ci_95_lower: float
    ci_95_upper: float


def load_profiles(df: "pd.DataFrame") -> list[DailyProfile]:
    """Convert a backtest DataFrame to a list of DailyProfiles.

    Expects columns: date, rank, p_game_hit, actual_hit.
    Extracts rank 1 and 2 for each date.
    """
    profiles = []
    for date, group in df.sort_values(["date", "rank"]).groupby("date"):
        r1 = group[group["rank"] == 1]
        r2 = group[group["rank"] == 2]
        if r1.empty:
            continue
        top1 = r1.iloc[0]
        top2 = r2.iloc[0] if not r2.empty else top1  # fallback to rank-1
        profiles.append(DailyProfile(
            top1_p=float(top1["p_game_hit"]),
            top1_hit=int(top1["actual_hit"]),
            top2_p=float(top2["p_game_hit"]),
            top2_hit=int(top2["actual_hit"]),
        ))
    return profiles


def run_monte_carlo(
    profiles: list[DailyProfile],
    strategy: Strategy,
    n_trials: int = 10_000,
    season_length: int = 180,
    seed: int = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by bootstrapping seasons.

    Samples `season_length` daily profiles with replacement from the pool,
    simulates each under the given strategy, collects streak statistics.
    """
    rng = np.random.default_rng(seed)
    pool = np.array(range(len(profiles)))
    max_streaks = []
    play_days_list = []

    for _ in range(n_trials):
        indices = rng.choice(pool, size=season_length, replace=True)
        season = [profiles[i] for i in indices]
        result = simulate_season(season, strategy)
        max_streaks.append(result.max_streak)
        play_days_list.append(result.play_days)

    streaks = np.array(max_streaks)
    p_57 = float(np.mean(streaks >= 57))

    # 95% CI for P(57) using normal approximation
    n = len(streaks)
    se = np.sqrt(p_57 * (1 - p_57) / n) if n > 0 else 0
    ci_lower = max(0.0, p_57 - 1.96 * se)
    ci_upper = min(1.0, p_57 + 1.96 * se)

    return MonteCarloResult(
        n_trials=n_trials,
        max_streaks=max_streaks,
        p_57=p_57,
        p_30=float(np.mean(streaks >= 30)),
        p_20=float(np.mean(streaks >= 20)),
        median_streak=int(np.median(streaks)),
        p95_streak=int(np.percentile(streaks, 95)),
        mean_play_days=float(np.mean(play_days_list)),
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
    )


def run_replay(
    season_profiles: dict[int, list[DailyProfile]],
    strategy: Strategy,
) -> dict[int, SeasonResult]:
    """Replay a strategy against actual historical seasons (no bootstrapping)."""
    return {
        season: simulate_season(profiles, strategy)
        for season, profiles in season_profiles.items()
    }


def load_all_profiles(profiles_dir: "Path") -> list[DailyProfile]:
    """Load all backtest profile parquets and convert to DailyProfiles.

    Separate from backtest_blend.py to avoid lightgbm import on Pi5.
    """
    import pandas as pd
    dfs = []
    for p in sorted(profiles_dir.glob("backtest_*.parquet")):
        dfs.append(pd.read_parquet(p))
    if not dfs:
        return []
    combined = pd.concat(dfs, ignore_index=True)
    return load_profiles(combined)


def load_season_profiles(profiles_dir: "Path") -> dict[int, list[DailyProfile]]:
    """Load backtest profiles grouped by season (for replay mode)."""
    import pandas as pd
    seasons = {}
    for p in sorted(profiles_dir.glob("backtest_*.parquet")):
        season = int(p.stem.split("_")[1])
        df = pd.read_parquet(p)
        seasons[season] = load_profiles(df)
    return seasons
```

- [ ] **Step 4: Run all monte_carlo tests**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/monte_carlo.py tests/simulate/test_monte_carlo.py
git commit -m "feat: Monte Carlo wrapper with bootstrap and replay modes"
```

---

### Task 4: Blend Walk-Forward Backtest

**Files:**
- Create: `src/bts/simulate/backtest_blend.py`
- Create: `tests/simulate/test_backtest_blend.py`

- [ ] **Step 1: Write failing test for output schema**

```python
# tests/simulate/test_backtest_blend.py
"""Tests for blend walk-forward backtest output."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestBlendBacktestOutput:
    def test_output_schema(self, tmp_path):
        """Verify the output parquet has the expected columns and types."""
        # Create a minimal synthetic output to validate schema
        from bts.simulate.backtest_blend import PROFILE_COLUMNS

        assert PROFILE_COLUMNS == ["date", "rank", "batter_id", "p_game_hit", "actual_hit", "n_pas"]

    def test_load_saved_profiles(self, tmp_path):
        """Round-trip: save profiles, load them back."""
        from bts.simulate.backtest_blend import save_profiles
        from bts.simulate.monte_carlo import load_all_profiles

        df = pd.DataFrame({
            "date": ["2024-04-01"] * 10,
            "rank": list(range(1, 11)),
            "batter_id": [i * 1000 for i in range(1, 11)],
            "p_game_hit": [0.90 - i * 0.02 for i in range(10)],
            "actual_hit": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
            "n_pas": [4] * 10,
        })
        save_profiles(df, 2024, tmp_path)
        loaded = load_all_profiles(tmp_path)
        assert len(loaded) == 1  # 1 day
        assert loaded[0].top1_p == df.iloc[0]["p_game_hit"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_backtest_blend.py -v`
Expected: ImportError

- [ ] **Step 3: Implement backtest_blend.py**

```python
# src/bts/simulate/backtest_blend.py
"""Blend walk-forward backtest that saves daily prediction profiles.

Adapts the existing walk_forward_evaluate to use the 12-model blend
(same BLEND_CONFIGS from predict.py) and save top-10 ranked predictions
per day to parquet for strategy simulation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from bts.features.compute import compute_all_features, FEATURE_COLS, STATCAST_COLS, TRAIN_START_YEAR
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.simulate.monte_carlo import DailyProfile, load_profiles

PROFILE_COLUMNS = ["date", "rank", "batter_id", "p_game_hit", "actual_hit", "n_pas"]


def blend_walk_forward(
    df: pd.DataFrame,
    test_season: int,
    retrain_every: int = 7,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run blend walk-forward evaluation and return daily profiles.

    For each game day in the test season:
    1. Train all 12 blend models on data before that day (retrained periodically)
    2. Predict P(hit|PA) with each model, average for blend ranking
    3. Aggregate to game-level P(>=1 hit) per batter
    4. Save top-N batters with blend p_game_hit and actual_hit

    Returns DataFrame with PROFILE_COLUMNS.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    test_start = df[df["season"] == test_season]["date"].min()
    train_pool = df[(df["date"] < test_start) & (df["season"] >= TRAIN_START_YEAR)].copy()
    test_data = df[df["date"] >= test_start].copy()
    test_dates = sorted(test_data["date"].unique())

    print(f"Blend walk-forward: {len(test_dates)} test days, "
          f"train pool: {len(train_pool):,} PAs, "
          f"{len(BLEND_CONFIGS)} models", file=sys.stderr)

    all_profiles = []
    blend = None

    for i, day in enumerate(test_dates):
        day_data = test_data[test_data["date"] == day].copy()

        # Retrain periodically
        if blend is None or (i % retrain_every == 0):
            available = pd.concat([train_pool, test_data[test_data["date"] < day]])
            train_y = available["is_hit"]

            blend = {}
            for name, cols in BLEND_CONFIGS:
                train_X = available[cols]
                mask = train_X.notna().any(axis=1)
                model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42, verbose=-1)
                model.fit(train_X[mask], train_y[mask])
                blend[name] = (model, cols)

            if (i + 1) % 30 == 0 or i == 0:
                print(f"  Day {i+1}/{len(test_dates)} ({pd.Timestamp(day).date()}) "
                      f"— retrained on {len(available):,} PAs", file=sys.stderr)

        # Predict with all blend models
        blend_pa_scores = {}
        for name, (model, cols) in blend.items():
            pred_X = day_data[cols]
            valid = pred_X.notna().any(axis=1)
            probs = pd.Series(np.nan, index=day_data.index)
            if valid.any():
                probs[valid] = model.predict_proba(pred_X[valid])[:, 1]
            blend_pa_scores[name] = probs

        # Average PA-level predictions across models
        pa_blend = pd.DataFrame(blend_pa_scores).mean(axis=1)
        day_data["p_hit_blend"] = pa_blend

        # Aggregate to game level: P(>=1 hit) = 1 - prod(1 - P(hit|PA))
        game_preds = day_data.groupby(["batter_id", "game_pk"]).agg(
            p_game_hit=("p_hit_blend", lambda x: 1 - np.prod(1 - x.values)),
            actual_hit=("is_hit", "max"),
            n_pas=("is_hit", "count"),
        ).reset_index()

        # Rank and take top N
        game_preds = game_preds.nlargest(top_n, "p_game_hit").reset_index(drop=True)
        game_preds["rank"] = range(1, len(game_preds) + 1)
        game_preds["date"] = pd.Timestamp(day).date()

        all_profiles.append(game_preds[PROFILE_COLUMNS])

    result = pd.concat(all_profiles, ignore_index=True)
    print(f"  Done: {len(result)} profile rows ({len(test_dates)} days × top-{top_n})", file=sys.stderr)
    return result


def save_profiles(df: pd.DataFrame, season: int, output_dir: Path) -> Path:
    """Save daily profiles to parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"backtest_{season}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)", file=sys.stderr)
    return path


def run_backtest(
    data_dir: str = "data/processed",
    output_dir: str = "data/simulation",
    seasons: list[int] | None = None,
    retrain_every: int = 7,
) -> None:
    """Run blend backtest for specified seasons and save profiles.

    Loads all PA parquets, computes features once, then runs blend
    walk-forward for each test season.
    """
    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]

    proc = Path(data_dir)
    out = Path(output_dir)

    # Load all data and compute features once
    print("Loading PA data...", file=sys.stderr)
    dfs = []
    for parquet in sorted(proc.glob("pa_*.parquet")):
        dfs.append(pd.read_parquet(parquet))
    if not dfs:
        raise RuntimeError("No parquet files found. Run 'bts data build' first.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Computing features on {len(df):,} PAs...", file=sys.stderr)
    df = compute_all_features(df)

    for season in seasons:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Season {season}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        profiles_df = blend_walk_forward(df, season, retrain_every=retrain_every)
        save_profiles(profiles_df, season, out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_backtest_blend.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/backtest_blend.py tests/simulate/test_backtest_blend.py
git commit -m "feat: blend walk-forward backtest with profile export"
```

---

### Task 5: CLI Integration

**Files:**
- Create: `src/bts/simulate/cli.py`
- Modify: `src/bts/cli.py` — add `cli.add_command(simulate)` import

- [ ] **Step 1: Write failing test for CLI commands**

Add to `tests/simulate/test_monte_carlo.py`:

```python
from click.testing import CliRunner
from bts.simulate.cli import simulate


class TestCLI:
    def test_simulate_run_with_synthetic_profiles(self, tmp_path):
        """CLI runs Monte Carlo on saved profile parquets."""
        # Create synthetic profiles
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "run", "--profiles-dir", str(tmp_path), "--trials", "100",
        ])
        assert result.exit_code == 0
        assert "baseline" in result.output
        assert "P(57)" in result.output

    def test_simulate_run_replay_only(self, tmp_path):
        """CLI replay mode."""
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "run", "--profiles-dir", str(tmp_path), "--replay-only",
        ])
        assert result.exit_code == 0
        assert "Replay" in result.output

    def test_simulate_run_single_strategy(self, tmp_path):
        """CLI with --strategy flag runs only that strategy."""
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "run", "--profiles-dir", str(tmp_path),
            "--strategy", "sprint", "--trials", "50",
        ])
        assert result.exit_code == 0
        assert "sprint" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py::TestCLI -v`
Expected: ImportError

- [ ] **Step 3: Implement CLI**

```python
# src/bts/simulate/cli.py
"""CLI commands for BTS strategy simulation."""

import json
import sys
from pathlib import Path

import click

from bts.simulate.strategies import ALL_STRATEGIES


@click.group()
def simulate():
    """Strategy simulation and backtesting."""
    pass


@simulate.command()
@click.option("--seasons", default="2021,2022,2023,2024,2025",
              help="Comma-separated seasons to backtest")
@click.option("--data-dir", default="data/processed", type=click.Path(),
              help="Processed parquet directory")
@click.option("--output-dir", default="data/simulation", type=click.Path(),
              help="Output directory for profile parquets")
@click.option("--retrain-every", default=7, type=int,
              help="Retrain blend models every N days")
def backtest(seasons: str, data_dir: str, output_dir: str, retrain_every: int):
    """Run blend walk-forward backtest and save daily profiles."""
    from bts.simulate.backtest_blend import run_backtest

    season_list = [int(s.strip()) for s in seasons.split(",")]
    click.echo(f"Running blend backtest for seasons: {season_list}")
    run_backtest(data_dir, output_dir, season_list, retrain_every)
    click.echo("Done.")


@simulate.command(name="run")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest profile parquets")
@click.option("--trials", default=10_000, type=int,
              help="Number of Monte Carlo trials per strategy")
@click.option("--season-length", default=180, type=int,
              help="Days per simulated season")
@click.option("--strategy", "strategy_name", default=None,
              help="Run only this strategy (default: all)")
@click.option("--replay-only", is_flag=True,
              help="Only replay actual seasons, no Monte Carlo")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--save-json", default=None, type=click.Path(),
              help="Save raw results to JSON")
def run_sim(profiles_dir: str, trials: int, season_length: int,
            strategy_name: str | None, replay_only: bool, seed: int,
            save_json: str | None):
    """Run Monte Carlo strategy simulation."""
    from rich.console import Console
    from rich.table import Table
    from bts.simulate.monte_carlo import (
        load_all_profiles, load_season_profiles, run_monte_carlo, run_replay,
    )

    profiles_path = Path(profiles_dir)
    console = Console()

    strategies = ALL_STRATEGIES
    if strategy_name:
        if strategy_name not in ALL_STRATEGIES:
            click.echo(f"Unknown strategy: {strategy_name}. "
                       f"Options: {', '.join(ALL_STRATEGIES.keys())}", err=True)
            raise SystemExit(1)
        strategies = {strategy_name: ALL_STRATEGIES[strategy_name]}

    if replay_only:
        season_data = load_season_profiles(profiles_path)
        if not season_data:
            click.echo("No profile parquets found.", err=True)
            raise SystemExit(1)

        table = Table(title=f"Replay Results ({len(season_data)} seasons)")
        table.add_column("Strategy")
        for s in sorted(season_data.keys()):
            table.add_column(str(s), justify="right")
        table.add_column("Best", justify="right")

        for name, strategy in strategies.items():
            results = run_replay(season_data, strategy)
            streaks = [results[s].max_streak for s in sorted(results.keys())]
            row = [name] + [str(s) for s in streaks] + [str(max(streaks))]
            table.add_row(*row)

        console.print(table)
        return

    # Monte Carlo mode
    profiles = load_all_profiles(profiles_path)
    if not profiles:
        click.echo("No profile parquets found.", err=True)
        raise SystemExit(1)

    table = Table(title=f"Strategy Comparison ({trials:,} seasons, {len(profiles)} daily profiles)")
    table.add_column("Strategy")
    table.add_column("P(57)", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("P(30+)", justify="right")
    table.add_column("P(20+)", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("95th", justify="right")
    table.add_column("Play Days", justify="right")

    json_results = {}

    for name, strategy in strategies.items():
        result = run_monte_carlo(profiles, strategy, n_trials=trials,
                                  season_length=season_length, seed=seed)
        table.add_row(
            name,
            f"{result.p_57:.2%}",
            f"[{result.ci_95_lower:.2%}, {result.ci_95_upper:.2%}]",
            f"{result.p_30:.1%}",
            f"{result.p_20:.1%}",
            str(result.median_streak),
            str(result.p95_streak),
            f"{result.mean_play_days:.0f}",
        )
        json_results[name] = {
            "p_57": result.p_57, "p_30": result.p_30, "p_20": result.p_20,
            "median": result.median_streak, "p95": result.p95_streak,
            "mean_play_days": result.mean_play_days,
            "ci_95": [result.ci_95_lower, result.ci_95_upper],
        }

    console.print(table)

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(save_json).write_text(json.dumps(json_results, indent=2))
        click.echo(f"Results saved to {save_json}")
```

- [ ] **Step 4: Wire into main CLI**

Add to `src/bts/cli.py`, after the existing imports at the top of the file:

```python
from bts.simulate.cli import simulate
cli.add_command(simulate)
```

Place after the `cli` group definition (after line 8) but before the `data` group.

- [ ] **Step 5: Run CLI tests**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/simulate/test_monte_carlo.py::TestCLI -v`
Expected: All 3 PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/stone/projects/bts && UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`
Expected: All tests pass (existing 106 + new ~20)

- [ ] **Step 7: Commit**

```bash
cd /Users/stone/projects/bts
git add src/bts/simulate/cli.py src/bts/cli.py
git commit -m "feat: CLI commands for strategy simulation"
```

---

### Task 6: Run Backtests and Validate

**Files:**
- No new files. Uses existing `bts simulate backtest` command.

- [ ] **Step 1: Run blend backtest for all 5 seasons**

This is a long-running command (~2-3 hours). Run from the bts project directory:

```bash
cd /Users/stone/projects/bts
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2021,2022,2023,2024,2025
```

Monitor progress via stderr output. Each season prints day progress every 30 days.

- [ ] **Step 2: Validate output files exist**

```bash
ls -la data/simulation/backtest_*.parquet
```

Expected: 5 files, one per season.

- [ ] **Step 3: Spot-check P@1 for 2024 and 2025**

Run a quick validation — the blend backtest P@1 for 2024 should be ~84.9% and 2025 ~87.5%:

```bash
cd /Users/stone/projects/bts
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
import pandas as pd
for season in [2024, 2025]:
    df = pd.read_parquet(f'data/simulation/backtest_{season}.parquet')
    r1 = df[df['rank'] == 1]
    p_at_1 = r1['actual_hit'].mean()
    print(f'{season}: P@1 = {p_at_1:.1%} ({len(r1)} days)')
"
```

Expected: P@1 within ~1% of known values (84.9% for 2024, 87.5% for 2025).

- [ ] **Step 4: Run Monte Carlo simulation**

```bash
cd /Users/stone/projects/bts
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate run --trials 10000 --save-json data/simulation/results.json
```

Review the output table. Compare strategies.

- [ ] **Step 5: Run replay mode**

```bash
cd /Users/stone/projects/bts
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate run --replay-only
```

Verify that replay max streaks are reasonable (known: 47 in 2022, 44 with doubling in 2025).

- [ ] **Step 6: Commit results metadata**

```bash
cd /Users/stone/projects/bts
git add data/simulation/results.json
git commit -m "results: initial strategy simulation comparison"
```

Do NOT commit the parquet files (they're large and derived data — add `data/simulation/*.parquet` to `.gitignore` if not already covered).
