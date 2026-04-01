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
    "combined": Strategy(name="combined", streak_config=(
        (9, 0.80, 0.55),     # skip bad days + aggressive doubling
        (15, 0.80, 0.60),    # saver phase
        (30, 0.80, 0.65),    # mid — selective doubling
        (45, 0.80, 0.65),    # lockdown — keep doubling (counter-intuitive but optimal)
        (56, 0.80, None),    # sprint — singles only, don't risk near-win
    )),
}
