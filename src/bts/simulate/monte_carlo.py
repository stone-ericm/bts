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
