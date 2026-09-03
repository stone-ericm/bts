"""Pick strategy: MDP-optimal with heuristic fallback.

Uses the MDP policy table (if available) for provably optimal
skip/single/double decisions based on (streak, days_remaining,
saver, quality_bin). Falls back to heuristic thresholds if no
policy file exists.

Extracted from cli.py so both `bts run` (local) and the Pi5 orchestrator
share the same decision logic.
"""

import io
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd

from bts.picks import (
    DailyPick, Pick, pick_from_row, load_pick, get_game_statuses,
    get_game_statuses_detailed, load_saver_available, classify_pick_lock_state,
    pick_candidate_status_is_available,
)
from bts.simulate.tail_policy import (
    OBJECTIVE_REACH57, OBJECTIVE_TAIL, effective_best, effective_days,
    forced_tail_action, lookup_tail_action, mdp_objective,
)

log = logging.getLogger(__name__)


# --- MDP policy (preferred) ---
# Loaded once on first use. Falls back to heuristic if not available.
_mdp_cache: dict | None = None

# Approximate season end — used to compute days_remaining for MDP lookup.
# Updated each season. The MDP is robust to ±5 days.
SEASON_END_DATE = "2026-09-28"


def _load_mdp():
    """Load the reach-57 policy AND the tail policy, caching on first call.

    The two artifacts are loaded independently with their errors isolated
    (Codex r2): a tail error must not discard a valid reach-57 table, and a base
    error (missing OR corrupt) must not bypass the tail — in the tail regime the
    decision path never consults the base table, so a dict with
    ``policy_table=None`` and a valid ``tail`` is a fully working state.
    Returns None only when NEITHER artifact loaded (legacy: heuristic while 57 is
    reachable; the forced tail rule once it is not).
    """
    global _mdp_cache
    if _mdp_cache is not None:
        return _mdp_cache or None
    from bts.simulate import mdp as _mdp_mod
    from bts.simulate import tail_policy as _tail_mod

    cache = {"policy_table": None, "boundaries": None, "season_length": 180,
             "base_sha256": None, "tail": None, "tail_error": None, "base_error": None}
    base_sha = None
    try:
        # ONE read of the base bytes: the hash the tail is bound to is the hash
        # of exactly the table we loaded (no second open to race a redeploy).
        raw = Path(_mdp_mod.DEFAULT_POLICY_PATH).read_bytes()
        if len(raw) > _tail_mod.MAX_ARTIFACT_BYTES:
            raise ValueError(f"base policy artifact too large ({len(raw)} bytes)")
        with np.load(io.BytesIO(raw)) as data:
            cache.update(policy_table=data["policy_table"],
                         boundaries=data["boundaries"].tolist(),
                         season_length=int(data["season_length"]))
        base_sha = hashlib.sha256(raw).hexdigest()
        cache["base_sha256"] = base_sha
    except Exception as exc:   # missing, corrupt, unreadable — all isolated
        cache["base_error"] = f"{type(exc).__name__}: {exc}"
        log.error("MDP base policy unavailable (%s)", cache["base_error"])
    if base_sha is None:
        # Codex r3 P1: without the base hash the tail's pairing cannot be
        # verified, and an unverified tail is NOT actionable — the decision path
        # takes the forced rule (skip iff the season best is unbeatable, else single).
        cache["tail_error"] = (f"base policy unavailable ({cache['base_error']}): "
                               f"tail pairing unverifiable")
        log.error("MDP tail policy not usable (%s)", cache["tail_error"])
    else:
        try:
            cache["tail"] = _tail_mod.load_tail_policy(
                _tail_mod.DEFAULT_TAIL_POLICY_PATH, expected_base_sha=base_sha)
        except _tail_mod.TailPolicyError as exc:
            cache["tail_error"] = str(exc)
            log.error("MDP tail policy unavailable (%s)", cache["tail_error"])
    if (cache["policy_table"] is None and cache["tail"] is None
            and not Path(_mdp_mod.DEFAULT_POLICY_PATH).exists()
            and not Path(_tail_mod.DEFAULT_TAIL_POLICY_PATH).exists()):
        _mdp_cache = {}  # nothing on disk at all (legacy: heuristic while 57 is reachable)
        return None
    # An artifact EXISTS but failed: keep the diagnostic dict so the decision path
    # reports WHY (base corrupt / tail unverifiable) instead of "no artifacts".
    _mdp_cache = cache
    return cache


def _mdp_action_from(mdp: dict | None, p_game_hit: float, streak: int, date: str, saver: bool) -> str | None:
    """Look up the optimal MDP action from an *injected* policy.

    Returns None when there is no policy, so the caller falls back to the heuristic.
    Pure given ``mdp`` — the policy is resolved once into the DecisionContext rather
    than loaded here, so ``decide_action`` can be evaluated repeatedly (Phase 2b)
    without re-doing IO.
    """
    if not mdp or mdp.get("policy_table") is None:
        return None

    from bts.simulate.mdp import lookup_action

    days_remaining = _days_remaining(date)

    return lookup_action(
        mdp["policy_table"], mdp["boundaries"],
        streak, days_remaining, saver, p_game_hit, mdp["season_length"],
    )


_UNSET = object()


def effective_pick_bar(streak: int, date: str, saver: bool, mdp=_UNSET,
                       best_streak: int | None = None, best_status: str | None = None) -> float | None:
    """The smallest p_game_hit the policy will play at (streak, date, saver): the
    lower edge of the first non-skip quality bin (0.0 when even the bottom bin
    plays). Heuristic path (no base policy while 57 is reachable) -> SKIP_THRESHOLD.
    In the tail regime the bins are the TAIL's (one bin in production: 0.0 while
    the season best is beatable, None at the terminal stop). None when every bin
    skips or the policy is unreadable. Display-only (skip messages) — never
    raises and never feeds the pick path."""
    if mdp is _UNSET:
        mdp = _load_mdp()
    try:
        objective, d_eff = _regime(mdp, streak, date)
        if objective == OBJECTIVE_TAIL:
            _, best_for_lookup = _normalize_best(best_streak, best_status)
            tail = mdp.get("tail") if mdp else None
            boundaries = [float(b) for b in tail.boundaries] if tail is not None else []
            reps = ([boundaries[0] - 1e-9] if boundaries else [0.0]) + boundaries
            floors = [0.0] + boundaries
            for rep_p, floor in zip(reps, floors):
                if _tail_action(mdp, rep_p, streak, d_eff, saver, best_for_lookup)[0] != "skip":
                    return floor
            return None
        if not mdp or mdp.get("policy_table") is None:
            return SKIP_THRESHOLD
        boundaries = [float(b) for b in mdp["boundaries"]]
        # One representative p per bin: just under the first boundary (bin 0),
        # then each boundary (entering bins 1..n). The bar reported for bin i>0
        # is its lower edge = boundaries[i-1]; for bin 0 it is 0.0.
        reps = ([boundaries[0] - 1e-9] if boundaries else [0.0]) + boundaries
        floors = [0.0] + boundaries
        for rep_p, floor in zip(reps, floors):
            if _mdp_action_from(mdp, rep_p, streak, date, saver) != "skip":
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
    """Impure prep for one pick decision; ``resolve_policy_decision`` is pure over
    (streak, saver).

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
    mdp: dict | None            # injected MDP policy (None -> heuristic / forced tail rule)
    best_streak: int | None = None   # contest season-best as supplied (None = unknown)
    best_status: str | None = None   # "trusted" | "untrusted" | None (contest_state decides)


@dataclass
class PolicyDecision:
    """One structured policy outcome, produced ONCE by ``resolve_policy_decision``
    and reused by the log line, skip summary/DM/dashboard, fallback classification
    and decision.json — never re-derived from a second state read (Codex r2)."""
    action: str                  # executed action after the operational clamps
    policy_action: str           # the policy's raw action before the clamps
    source: str                  # "mdp" | "heuristic"
    objective: str               # OBJECTIVE_REACH57 | OBJECTIVE_TAIL
    streak: int
    days_effective: int
    best_supplied: int | None
    best_status: str             # "trusted" | "untrusted" | "missing"
    effective_best: int | None   # m the tail lookup used; None under reach57
    tail_sha256: str | None      # sha256 of the tail artifact that chose the action
    degraded_reason: str | None  # set when the tail resolved via the forced rule
    reason: str | None           # human-readable skip explanation (tail stop rule)
    pick_bar: float | None       # display-only bar when the action is a skip


def _days_remaining(date: str) -> int:
    end = datetime.strptime(SEASON_END_DATE, "%Y-%m-%d")
    today = datetime.strptime(date, "%Y-%m-%d")
    return max(0, (end - today).days)


def _regime(mdp: dict | None, streak: int, date: str) -> tuple[str, int]:
    """(objective, effective days) from STATE ALONE — resolved before any artifact
    is consulted, so an artifact failure can never fall through to the 0.80
    heuristic once 57 is unreachable (Codex r2 P0)."""
    season_length = int((mdp or {}).get("season_length") or 180)
    days = _days_remaining(date)
    return mdp_objective(streak, days, season_length), effective_days(days, season_length)


def _normalize_best(best_streak: int | None, best_status: str | None) -> tuple[str, int | None]:
    """Trust contract: only a best the caller marked ``"trusted"`` may authorise the
    tail's terminal stop. Missing/untrusted degrade to best = streak (keeps picking)."""
    if best_streak is None:
        return "missing", None
    if best_status == "trusted":
        return "trusted", int(best_streak)
    return "untrusted", None


def _tail_action(mdp: dict | None, p_game_hit: float, streak: int, days_effective: int,
                 saver: bool, best_for_lookup: int | None) -> tuple[str, str | None]:
    """(action, degraded_reason). EVERY failure shape — no artifacts at all, tail
    absent/invalid, lookup exception — resolves to ``forced_tail_action`` (skip iff
    the season best can't be beaten, else single). Never the base table, never the
    heuristic."""
    tail = mdp.get("tail") if mdp else None
    if tail is not None and mdp.get("base_sha256") != getattr(tail, "base_policy_sha256", None):
        tail, why = None, "base policy sha unavailable or mismatched: tail pairing unverifiable"
    else:
        why = (mdp.get("tail_error") if mdp else None) or "no MDP artifacts loaded"
    if tail is None:
        action = forced_tail_action(streak, best_for_lookup or 0, days_effective)
        reason = f"tail policy unavailable ({why}); forced fallback"
        log.error("tail policy: %s -> %s", reason, action)
        return action, reason
    try:
        return lookup_tail_action(tail, streak, best_for_lookup, days_effective, saver, p_game_hit), None
    except Exception as exc:
        action = forced_tail_action(streak, best_for_lookup or 0, days_effective)
        reason = f"tail lookup failed ({exc}); forced fallback"
        log.error("tail policy: %s -> %s", reason, action)
        return action, reason


def resolve_policy_decision(ctx: DecisionContext, streak: int, saver: bool) -> PolicyDecision:
    """Pure skip/single/double decision given a prepared context + (streak, saver).

    Regime first: while 57 is reachable this is the shipped reach-57 table (or the
    heuristic when no base policy loaded); once it is not, the tail policy — exact
    E[season-best] with the account's best carried as state — or its forced rule.
    Then the ``allow_double`` clamp and the "double must be executable (a
    different-game second pick exists)" guard. No IO.
    """
    objective, d_eff = _regime(ctx.mdp, streak, ctx.date)
    best_status, best_for_lookup = _normalize_best(ctx.best_streak, ctx.best_status)
    degraded = None
    effective_m = None
    tail_sha = None
    reason = None
    if objective == OBJECTIVE_TAIL:
        effective_m = effective_best(streak, best_for_lookup)
        policy_action, degraded = _tail_action(
            ctx.mdp, ctx.primary_p, streak, d_eff, saver, best_for_lookup)
        source = "mdp"
        tail = ctx.mdp.get("tail") if ctx.mdp else None
        if tail is not None and degraded is None:
            tail_sha = getattr(tail, "sha256", None)
        if policy_action == "skip":
            reason = (f"season-best {effective_m} can't be beaten with {d_eff} days left "
                      f"(max reachable {min(57, streak + 2 * d_eff)}); no pick")
    else:
        mdp_action = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)
        if mdp_action is not None:
            policy_action, source = mdp_action, "mdp"
        else:
            source = "heuristic"
            if ctx.primary_p < SKIP_THRESHOLD:
                policy_action = "skip"
            elif _double_threshold(streak) is not None and ctx.has_diff_game and ctx.second_p is not None:
                p_both = ctx.primary_p * ctx.second_p
                policy_action = "double" if p_both >= _double_threshold(streak) else "single"
            else:
                policy_action = "single"
    action = policy_action
    if action == "double" and not ctx.allow_double:
        action = "single"
    if action == "double" and not ctx.has_diff_game:
        action = "single"
    pick_bar = None
    if action == "skip":
        pick_bar = effective_pick_bar(streak, ctx.date, saver, mdp=ctx.mdp,
                                      best_streak=ctx.best_streak, best_status=ctx.best_status)
    return PolicyDecision(
        action=action, policy_action=policy_action, source=source, objective=objective,
        streak=int(streak), days_effective=d_eff, best_supplied=ctx.best_streak,
        best_status=best_status, effective_best=effective_m, tail_sha256=tail_sha,
        degraded_reason=degraded, reason=reason, pick_bar=pick_bar,
    )


def decide_action(ctx: DecisionContext, streak: int, saver: bool) -> tuple[str, str]:
    """(action, source) — thin wrapper over ``resolve_policy_decision`` kept for the
    existing call sites and tests; source is "mdp" or "heuristic"."""
    dec = resolve_policy_decision(ctx, streak, saver)
    return dec.action, dec.source


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
    # bts_daily_decision_v2 provenance — stamped by the orchestrator from the
    # DecisionStreakState that fed this selection; None on paths without one
    # (shadow, direct test callers, classification recoveries).
    state_source: "str | None" = None
    state_status: "str | None" = None
    allow_double: "bool | None" = None
    contest_source_date: "str | None" = None
    # The structured policy outcome that produced ``action`` (None on paths that
    # never reached the policy: locked/current pick, no-eligible, status failure).
    decision: "PolicyDecision | None" = None


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
    unavailable_game_pks: "set[int] | None" = None,
    best_streak: int | None = None,
    best_status: str | None = None,
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
        if current and unavailable_game_pks:
            committed = {current.pick.game_pk} | (
                {current.double_down.game_pk} if current.double_down else set())
            if committed & set(unavailable_game_pks):
                current = None   # a committed slot is past its cutoff: re-select
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
    if unavailable_game_pks:
        # Live-only (scheduler passes games whose submission cutoff has already
        # passed — NOT a margin; 2026-08-30). Offline callers pass nothing.
        available = available[
            ~available["game_pk"].astype(int).isin(list(unavailable_game_pks))]

    if available.empty:
        if current:
            # Every candidate game is unavailable (started, or warmup — which
            # the fresh pool deliberately excludes) but the existing pick itself
            # classified deliverable above. Return it UNLOCKED so the
            # fallback-deadline path can still deliver it; returning it locked
            # recreates the 2026-08-13 silent pass when the pick's own games
            # are merely in warmup.
            return SelectionResult(PickResult(daily=current, locked=False), None, None, None, None, None)
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
        best_streak=best_streak,
        best_status=best_status,
    )
    decision = resolve_policy_decision(ctx, streak, saver)
    action, source = decision.action, decision.source

    primary_candidate = _row_to_candidate(best_row)
    double_candidate = _row_to_candidate(second_row) if second_row is not None else None

    if action == "skip":
        return SelectionResult(None, "skip", source, primary_candidate, double_candidate, None,
                               streak=streak, saver_available=saver, decision=decision)

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
        # Codex r3 P1: the cached-fallback / restart-recovery commit paths see
        # selection=None — the decision must travel WITH the pick it chose.
        policy_decision=asdict(decision),
    )

    return SelectionResult(PickResult(daily=daily, locked=False), action, source,
                           primary_candidate, double_candidate, None,
                           streak=streak, saver_available=saver, decision=decision)
