"""Pick strategy: MDP-optimal with heuristic fallback.

Uses the MDP policy table (if available) for provably optimal
skip/single/double decisions based on (streak, days_remaining,
saver, quality_bin). Falls back to heuristic thresholds if no
policy file exists.

Extracted from cli.py so both `bts run` (local) and the Pi5 orchestrator
share the same decision logic.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from bts.picks import (
    DailyPick, Pick, pick_from_row, load_pick, get_game_statuses,
)

OVERRIDE_THRESHOLD = 0.78

# --- MDP policy (preferred) ---
# Loaded once on first use. Falls back to heuristic if not available.
_mdp_cache: dict | None = None

# Approximate season end — used to compute days_remaining for MDP lookup.
# Updated each season. The MDP is robust to ±5 days.
SEASON_END_DATE = "2026-09-28"


def _load_mdp():
    """Load MDP policy table, caching on first call. Returns None if not available."""
    global _mdp_cache
    if _mdp_cache is not None:
        return _mdp_cache

    try:
        from bts.simulate.mdp import load_policy, DEFAULT_POLICY_PATH
        policy_table, boundaries, season_length = load_policy(DEFAULT_POLICY_PATH)
        _mdp_cache = {
            "policy_table": policy_table,
            "boundaries": boundaries,
            "season_length": season_length,
        }
        return _mdp_cache
    except (FileNotFoundError, ImportError):
        _mdp_cache = {}  # empty dict = tried but failed
        return None


def _mdp_action(p_game_hit: float, streak: int, date: str, saver: bool = True) -> str | None:
    """Look up optimal action from MDP policy. Returns None if MDP not available."""
    mdp = _load_mdp()
    if not mdp:
        return None

    from bts.simulate.mdp import lookup_action

    end = datetime.strptime(SEASON_END_DATE, "%Y-%m-%d")
    today = datetime.strptime(date, "%Y-%m-%d")
    days_remaining = max(0, (end - today).days)

    return lookup_action(
        mdp["policy_table"], mdp["boundaries"],
        streak, days_remaining, saver, p_game_hit, mdp["season_length"],
    )


# --- Heuristic fallback ---
SKIP_THRESHOLD = 0.80

_DOUBLE_BY_STREAK = (
    (9, 0.55),    # aggressive — little to lose
    (15, 0.60),   # saver phase — moderate
    (45, 0.65),   # mid + lockdown — selective doubling
    (56, None),   # sprint — singles only, don't risk a near-win
)


def _double_threshold(streak: int) -> float | None:
    """Return the P(both hit) threshold for doubling at this streak."""
    for max_streak, threshold in _DOUBLE_BY_STREAK:
        if streak <= max_streak:
            return threshold
    return None


@dataclass
class PickResult:
    """Result of pick strategy.

    daily: the selected DailyPick (new or existing locked)
    locked: True if pick was already locked (game started or posted)
    """
    daily: DailyPick
    locked: bool = False


def _classify_et_hour(game_time_utc: str) -> int:
    """Convert UTC game time to ET hour (UTC - 4)."""
    try:
        utc = datetime.fromisoformat(str(game_time_utc).replace("Z", "+00:00"))
        return (utc - timedelta(hours=4)).hour
    except Exception:
        return 18  # default to prime


def _apply_densest_bucket(valid: pd.DataFrame) -> pd.DataFrame:
    """Apply densest bucket + override strategy.

    Returns the filtered DataFrame to pick from.
    """
    if "game_time" not in valid.columns:
        return valid

    valid = valid.copy()
    valid["_et_hour"] = valid["game_time"].apply(_classify_et_hour)

    early = valid[valid["_et_hour"] < 16]
    prime = valid[(valid["_et_hour"] >= 16) & (valid["_et_hour"] < 20)]
    west = valid[valid["_et_hour"] >= 20]

    buckets = {"early": early, "prime": prime, "west": west}
    densest_name = max(buckets, key=lambda k: len(buckets[k]))

    top_overall = valid.iloc[0]
    top_hour = top_overall["_et_hour"]
    top_window = "early" if top_hour < 16 else ("prime" if top_hour < 20 else "west")

    if top_window == densest_name:
        return buckets[densest_name]
    elif top_overall["p_game_hit"] > OVERRIDE_THRESHOLD:
        return valid  # override — top pick from non-densest beats threshold
    else:
        return buckets[densest_name]


def select_pick(
    predictions: pd.DataFrame,
    date: str,
    picks_dir: Path,
    streak: int = 0,
) -> PickResult | None:
    """Apply densest bucket + override strategy to predictions.

    Returns PickResult with the selected DailyPick, or None if there's
    nothing to pick (no games, all started, empty predictions).
    """
    if predictions.empty:
        return None

    current = load_pick(date, picks_dir)
    statuses = get_game_statuses(date)

    # Check if current pick is locked
    if current and (
        statuses.get(current.pick.game_pk) != "P" or current.bluesky_posted
    ):
        return PickResult(daily=current, locked=True)

    # Filter to games not yet started
    not_started = predictions["game_pk"].map(lambda pk: statuses.get(pk) == "P")
    available = predictions[not_started]

    if available.empty:
        if current:
            return PickResult(daily=current, locked=True)
        return None

    # Filter to valid predictions
    valid = available[available["p_game_hit"].notna()]
    if valid.empty:
        return None

    # Apply densest bucket + override
    valid = _apply_densest_bucket(valid)

    best_row = valid.iloc[0]

    # Determine action: MDP policy (preferred) or heuristic fallback
    action = _mdp_action(best_row["p_game_hit"], streak, date)
    if action is None:
        # Heuristic fallback
        if best_row["p_game_hit"] < SKIP_THRESHOLD:
            action = "skip"
        elif _double_threshold(streak) is not None and len(valid) >= 2:
            second = valid.iloc[1]
            p_both = best_row["p_game_hit"] * second["p_game_hit"]
            action = "double" if p_both >= _double_threshold(streak) else "single"
        else:
            action = "single"

    if action == "skip":
        return None

    new_pick = pick_from_row(best_row)

    # Double-down
    double_pick = None
    if action == "double" and len(valid) >= 2:
        double_pick = pick_from_row(valid.iloc[1])

    # Runner-up
    runner_up = None
    if len(valid) >= 2:
        ru = valid.iloc[1]
        runner_up = {"batter_name": ru["batter_name"], "p_game_hit": float(ru["p_game_hit"])}

    daily = DailyPick(
        date=date,
        run_time=datetime.now(timezone.utc).isoformat(),
        pick=new_pick,
        double_down=double_pick,
        runner_up=runner_up,
        bluesky_posted=False,
        bluesky_uri=None,
    )

    return PickResult(daily=daily, locked=False)
