"""Pick strategy: MDP-optimal with heuristic fallback.

Uses the MDP policy table (if available) for provably optimal
skip/single/double decisions based on (streak, days_remaining,
saver, quality_bin). Falls back to heuristic thresholds if no
policy file exists.

Extracted from cli.py so both `bts run` (local) and the Pi5 orchestrator
share the same decision logic.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bts.picks import (
    DailyPick, Pick, pick_from_row, load_pick, get_game_statuses,
    get_game_statuses_detailed, load_saver_available, classify_pick_lock_state,
    pick_candidate_status_is_available,
)


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


def _mdp_action_from(mdp: dict | None, p_game_hit: float, streak: int, date: str, saver: bool) -> str | None:
    """Look up the optimal MDP action from an *injected* policy.

    Returns None when there is no policy, so the caller falls back to the heuristic.
    Pure given ``mdp`` — the policy is resolved once into the DecisionContext rather
    than loaded here, so ``decide_action`` can be evaluated repeatedly (Phase 2b)
    without re-doing IO.
    """
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


_UNSET = object()


def effective_pick_bar(streak: int, date: str, saver: bool, mdp=_UNSET) -> float | None:
    """The smallest p_game_hit the policy will play at (streak, date, saver): the
    lower edge of the first non-skip quality bin (0.0 when even the bottom bin
    plays). Heuristic path (no policy) -> SKIP_THRESHOLD. None when every bin
    skips or the policy is unreadable. Display-only (skip messages) — never
    raises and never feeds the pick path."""
    if mdp is _UNSET:
        mdp = _load_mdp()
    if not mdp:
        return SKIP_THRESHOLD
    try:
        boundaries = [float(b) for b in mdp["boundaries"]]
        # One representative p per bin: just under the first boundary (bin 0),
        # then each boundary (entering bins 1..n). The bar reported for bin i>0
        # is its lower edge = boundaries[i-1]; for bin 0 it is 0.0.
        reps = ([boundaries[0] - 1e-9] if boundaries else [0.0]) + boundaries
        floors = [0.0] + boundaries
        for rep, floor in zip(reps, floors):
            if _mdp_action_from(mdp, rep, streak, date, saver) != "skip":
                return floor
        return None
    except Exception:
        return None


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
class DecisionContext:
    """Impure prep for one pick decision; ``decide_action`` is pure over (streak, saver).

    All file/cache/network IO and candidate selection happen while building this in
    ``select_pick``; the action decision itself reads only these fields plus the
    scalar (streak, saver). This is the seam Phase 2b evaluates over a plausible set
    of (streak, saver) states.
    """
    primary_p: float            # best candidate's p_game_hit
    second_p: float | None      # executable different-game second pick's p_game_hit (or None)
    has_diff_game: bool         # a valid different-game second pick exists
    date: str                   # YYYY-MM-DD (for MDP days_remaining)
    allow_double: bool          # global operational clamp (NOT uncertainty logic)
    mdp: dict | None            # injected MDP policy (None -> heuristic)


def decide_action(ctx: DecisionContext, streak: int, saver: bool) -> tuple[str, str]:
    """Pure skip/single/double decision given a prepared context + (streak, saver).

    MDP policy (preferred) or heuristic fallback, then the ``allow_double`` clamp and
    the "double must be executable (a different-game second pick exists)" guard. No IO.
    Mirrors the action branch that previously lived inline in ``select_pick``.

    Returns (action, source) where source is "mdp" or "heuristic".
    """
    mdp_action = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)
    if mdp_action is not None:
        action, source = mdp_action, "mdp"
    else:
        source = "heuristic"
        if ctx.primary_p < SKIP_THRESHOLD:
            action = "skip"
        elif _double_threshold(streak) is not None and ctx.has_diff_game and ctx.second_p is not None:
            p_both = ctx.primary_p * ctx.second_p
            action = "double" if p_both >= _double_threshold(streak) else "single"
        else:
            action = "single"
    if action == "double" and not ctx.allow_double:
        action = "single"
    if action == "double" and not ctx.has_diff_game:
        action = "single"
    return action, source


@dataclass
class PickResult:
    """Result of pick strategy.

    daily: the selected DailyPick (new or existing locked)
    locked: True if pick was already locked (game started or posted)
    """
    daily: DailyPick
    locked: bool = False


@dataclass
class SelectionResult:
    """Authoritative output of select_pick.

    Always returned (never None) so callers can inspect the action and source
    even on a skip day. Use pick_result to check if a pick was actually made.
    """
    pick_result: "PickResult | None"
    action: "str | None"           # "skip"|"single"|"double", or None if no action reached
    source: "str | None"           # "mdp"|"heuristic", or None
    primary_candidate: "dict | None"   # executable best_row
    double_candidate: "dict | None"
    no_pick_reason: "str | None"   # "no_eligible"|"status_failure"|"no_valid_predictions"|None
    streak: int = 0
    saver_available: "bool | None" = None


def _row_to_candidate(row) -> "dict | None":
    """Convert a predictions row to a native-typed candidate dict."""
    if row is None:
        return None
    gpk = pd.to_numeric(pd.Series([row.get("game_pk")]), errors="coerce").iloc[0]
    return {
        "batter_id": int(row["batter_id"]),
        "batter_name": row.get("batter_name"),
        "team": row.get("team"),
        "game_pk": (int(gpk) if pd.notna(gpk) else None),
        "p_game_hit": float(row["p_game_hit"]),
    }


def should_lock(
    top_pick: dict,
    all_picks: list[dict],
    early_lock_gap: float,
    double_down: dict | None = None,
) -> bool:
    """Decide if the current top pick should be locked (posted to Bluesky).

    Locks when:
    1. The top pick has a confirmed (not projected) lineup, AND
    2. The selected double-down (if any) has a confirmed lineup, AND
    3. Either all picks have confirmed lineups, OR the gap between
       the top pick and the best projected-lineup pick exceeds early_lock_gap.

    Condition 2 is audit F2: a double commits BOTH slots, so the gap rule —
    valid for a single pick racing projected alternatives — must not carry a
    projected DD through on the primary's confirmation. The T-35 final
    fallback deliberately bypasses this function (delivering on projected
    data beats missing the day), so the gate can only delay a lock, never
    lose one.
    """
    if top_pick.get("projected_lineup", True):
        return False

    if double_down is not None and double_down.get("projected_lineup", True):
        return False

    # Find the best projected-lineup pick (excluding the top pick's game)
    best_projected = None
    for p in all_picks:
        if p.get("projected_lineup", False) and p["game_pk"] != top_pick["game_pk"]:
            if best_projected is None or p["p_game_hit"] > best_projected["p_game_hit"]:
                best_projected = p

    if best_projected is None:
        # All confirmed — safe to lock
        return True

    return (top_pick["p_game_hit"] - best_projected["p_game_hit"]) >= early_lock_gap


def select_pick(
    predictions: pd.DataFrame,
    date: str,
    picks_dir: Path,
    streak: int = 0,
    saver_available: bool | None = None,
    allow_double: bool = True,
    for_shadow: bool = False,
    game_statuses_detailed: dict[int, dict[str, str]] | None = None,
    require_detailed_statuses: bool = False,
) -> "SelectionResult":
    """Select the best pick from available predictions.

    Always returns a SelectionResult. Check pick_result to see if a pick was made.
    Use action/source to inspect the decision even on a skip day.

    Existing production picks are classified before reuse. Posted picks and
    picks whose committed games have started remain locked; unposted picks
    whose committed games are postponed, cancelled, or missing from today's
    schedule are treated as stale and can be replaced by fresh predictions.

    When ``for_shadow=True``, production pick reuse is bypassed so the shadow
    model always computes its own pick from its own predictions. (Without this,
    shadow calls made after production locks would silently return production's
    DailyPick and corrupt {date}.shadow.json.)

    ``game_statuses_detailed`` lets live callers inject detailed MLB statuses
    for candidate filtering without making offline/backtest-shaped calls depend
    on a live detailed-status lookup. When omitted, the legacy coarse
    ``get_game_statuses`` path is preserved unless
    ``require_detailed_statuses`` is true. Live pick-generation callers should
    set that strict mode so postponed/cancelled/missing game protection cannot
    silently degrade to abstract MLB statuses.
    """
    if predictions.empty:
        return SelectionResult(None, None, None, None, None, "no_valid_predictions")

    if require_detailed_statuses and game_statuses_detailed is None:
        try:
            game_statuses_detailed = get_game_statuses_detailed(date)
        except Exception:
            game_statuses_detailed = None

    current = None
    if not for_shadow:
        current = load_pick(date, picks_dir)
        if require_detailed_statuses and game_statuses_detailed is None and current:
            return SelectionResult(PickResult(daily=current, locked=True), None, None, None, None, None)
        if current:
            lock_state = classify_pick_lock_state(current, date)
            if lock_state.stale:
                current = None
            elif lock_state.locked:
                return SelectionResult(PickResult(daily=current, locked=True), None, None, None, None, None)

    if game_statuses_detailed is None:
        if require_detailed_statuses:
            return SelectionResult(None, None, None, None, None, "status_failure")
        try:
            statuses = get_game_statuses(date)
        except Exception:
            if current:
                return SelectionResult(PickResult(daily=current, locked=True), None, None, None, None, None)
            return SelectionResult(None, None, None, None, None, "status_failure")

        # Filter to games not yet started, preserving legacy coarse behavior.
        not_started = predictions["game_pk"].map(lambda pk: statuses.get(pk) == "P")
    else:
        def is_available(game_pk) -> bool:
            try:
                status = game_statuses_detailed.get(int(game_pk))
            except (TypeError, ValueError):
                return False
            return pick_candidate_status_is_available(status)

        not_started = predictions["game_pk"].map(is_available)
    available = predictions[not_started]

    if available.empty:
        if current:
            return SelectionResult(PickResult(daily=current, locked=True), None, None, None, None, None)
        return SelectionResult(None, None, None, None, None, "no_eligible")

    # Filter to valid predictions
    valid = available[available["p_game_hit"].notna()]
    if valid.empty:
        return SelectionResult(None, None, None, None, None, "no_valid_predictions")

    best_row = valid.iloc[0]

    # Different-game candidates, computed once and shared by the doubling decision
    # and the executed double-down / runner-up. game_pk is normalized so a NaN or
    # type-mismatched value (e.g. "778899" vs 778899) can't be mistaken for a
    # different game and trigger a same-game (correlated) or junk double-down.
    best_game_pk = pd.to_numeric(pd.Series([best_row["game_pk"]]), errors="coerce").iloc[0]
    game_pk_num = pd.to_numeric(valid["game_pk"], errors="coerce")
    if pd.isna(best_game_pk):
        diff_game = valid.iloc[0:0]
    else:
        diff_game = valid[game_pk_num.notna() & (game_pk_num != best_game_pk)]

    # Determine action: build the decision context (impure prep) then decide.
    # decide_action owns the MDP-or-heuristic branch, the allow_double clamp, and the
    # "double must be executable (a different-game second pick exists)" guard. The
    # context carries the resolved MDP policy, the selected primary/second candidates,
    # and allow_double. second_p stays a raw pandas value so p_both is computed
    # exactly as the inline branch did (best_row * second, no dtype coercion).
    saver = load_saver_available(picks_dir) if saver_available is None else saver_available
    second_row = diff_game.iloc[0] if len(diff_game) >= 1 else None
    ctx = DecisionContext(
        primary_p=best_row["p_game_hit"],
        second_p=second_row["p_game_hit"] if second_row is not None else None,
        has_diff_game=len(diff_game) >= 1,
        date=date,
        allow_double=allow_double,
        mdp=_load_mdp(),
    )
    action, source = decide_action(ctx, streak, saver)

    primary_candidate = _row_to_candidate(best_row)
    double_candidate = _row_to_candidate(second_row) if second_row is not None else None

    if action == "skip":
        return SelectionResult(None, "skip", source, primary_candidate, double_candidate, None,
                               streak=streak, saver_available=saver)

    new_pick = pick_from_row(best_row)

    # Double-down: must be from a different game to avoid correlated outcomes
    double_pick = None
    if action == "double" and len(diff_game) >= 1:
        double_pick = pick_from_row(diff_game.iloc[0])

    # Runner-up (also from different game)
    runner_up = None
    if len(diff_game) >= 1:
        ru = diff_game.iloc[0]
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

    return SelectionResult(PickResult(daily=daily, locked=False), action, source,
                           primary_candidate, double_candidate, None,
                           streak=streak, saver_available=saver)
