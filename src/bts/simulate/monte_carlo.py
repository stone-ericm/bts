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
    from pathlib import Path as _Path
    profiles_dir = _Path(profiles_dir)
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
    from pathlib import Path as _Path
    profiles_dir = _Path(profiles_dir)
    seasons = {}
    for p in sorted(profiles_dir.glob("backtest_*.parquet")):
        season = int(p.stem.split("_")[1])
        df = pd.read_parquet(p)
        seasons[season] = load_profiles(df)
    return seasons
