# src/bts/daily_decision.py
"""Authoritative end-of-day decision record (data/picks/<date>/decision.json).

The SINGLE source of truth for "what did production finally do on <date>". Written only by the
scheduler at true finalization points; read by check-results and the skip-policy shadow. See
docs/superpowers/specs/2026-06-21-daily-decision-record-design.md.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bts.util import atomic_write_text

DECISION_SCHEMA = "bts_daily_decision_v3"
# v1 records (through 2026-08-09) persist state only on MDP skips and never
# the second candidate; v2 (through 2026-09-03) has no objective. Readers accept
# all three so legacy files stay authoritative; a record without ``objective``
# was decided under reach57 (the only objective before v3).
ACCEPTED_SCHEMAS = ("bts_daily_decision_v1", "bts_daily_decision_v2", "bts_daily_decision_v3")
OBJECTIVE_REACH57 = "reach57"
OBJECTIVES = ("reach57", "emax_season_best")
_LEGACY_SCHEMAS = ("bts_daily_decision_v1", "bts_daily_decision_v2")
_RANK_FIELDS = ("batter_id", "batter_name", "team", "game_pk", "p_game_hit")


def _utc_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _summary(cand: dict | None) -> dict | None:
    return None if cand is None else {k: cand.get(k) for k in _RANK_FIELDS}


def decision_path(date: str, picks_dir) -> Path:
    return Path(picks_dir) / date / "decision.json"


def decision_objective(rec: dict | None) -> str:
    """The objective a record was decided under. Pre-v3 records (no objective
    field existed) are reach57. A v3 record MUST carry a valid objective; a
    null/invalid one is "unknown" — never silently reach57 (Codex r3), so it is
    excluded from every reach-57 consumer (skip shadow, census, alignment)."""
    if not rec:
        return OBJECTIVE_REACH57   # no record at all (pre-decision.json picks): legacy reach57
    obj = rec.get("objective")
    if rec.get("schema_version") in _LEGACY_SCHEMAS:
        return obj if obj in OBJECTIVES else OBJECTIVE_REACH57
    return obj if obj in OBJECTIVES else "unknown"


def is_reach57_mdp_skip(rec: dict | None) -> bool:
    """The skip-policy shadow / boundary census estimand: an MDP skip decided under
    the reach-57 objective. Tail-objective skips (season best unbeatable) come from
    a different rule and must NOT feed those pre-registered checkpoints."""
    return bool(rec) and rec.get("action") == "skip" and rec.get("source") == "mdp" \
        and decision_objective(rec) == OBJECTIVE_REACH57


def write_decision(date, picks_dir, *, action, source, primary=None, double_down=None,
                   streak=None, saver_available=None, delivery_status, scoreable,
                   second_candidate=None, state_source=None, state_status=None,
                   allow_double=None, contest_source_date=None, now=None,
                   objective=None, best_streak=None, best_status=None, effective_best=None,
                   tail_policy_sha256=None, degraded_reason=None) -> dict | None:
    """Best-effort atomic write of the day's decision record. Returns the record, or None on any
    failure (must never raise into the live pick path).

    v2 fields (2026-08-09, boundary-census follow-up): state provenance on
    every record — (streak, saver_available, state_source, state_status,
    allow_double, contest_source_date) from the DecisionStreakState that fed
    the action — and second_candidate, the executable different-game runner-up
    at skip time. All default None so legacy call paths stay valid.

    v3 fields (2026-09-03, tail policy): objective ("reach57" | "emax_season_best"),
    best_streak (as supplied) + best_status (trust), effective_best (the m the tail
    lookup used), tail_policy_sha256 (the artifact that chose the action) and
    degraded_reason (set when the forced fallback decided). An action is not
    reproducible without them once the objective can switch."""
    try:
        objective = objective or OBJECTIVE_REACH57   # legacy callers: explicit, not null
        if objective not in OBJECTIVES:
            raise ValueError(f"objective {objective!r} not in {OBJECTIVES}")
        record = {
            "schema_version": DECISION_SCHEMA, "date": date,
            "action": action, "source": source,
            "primary": _summary(primary), "double_down": _summary(double_down),
            "second_candidate": _summary(second_candidate),
            "streak": streak,
            "saver_available": (None if saver_available is None else bool(saver_available)),
            "state_source": state_source, "state_status": state_status,
            "allow_double": (None if allow_double is None else bool(allow_double)),
            "contest_source_date": contest_source_date,
            "delivery_status": delivery_status, "scoreable": bool(scoreable),
            "objective": objective, "best_streak": best_streak, "best_status": best_status,
            "effective_best": effective_best, "tail_policy_sha256": tail_policy_sha256,
            "degraded_reason": degraded_reason,
            "finalized_at": _utc_iso(now),
        }
        path = decision_path(date, picks_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(record, indent=2))
        return record
    except Exception:
        return None


def load_decision(date: str, picks_dir) -> dict | None:
    path = decision_path(date, picks_dir)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
        if not isinstance(rec, dict) or rec.get("schema_version") not in ACCEPTED_SCHEMAS:
            return None
        # Reject partial / wrong-shape records that carry the schema tag but lack the
        # core fields (post-review Fix 3): accepting e.g. {schema_version, scoreable}
        # would treat a stale preview as authoritative and could mis-authorize scoring.
        # write_decision always writes all of these, so genuine records are unaffected.
        if (rec.get("action") not in {"skip", "single", "double"}
                or not isinstance(rec.get("scoreable"), bool)
                or "date" not in rec):
            return None
        return rec
    except (json.JSONDecodeError, OSError):
        return None


def is_scoreable_commit(date: str, picks_dir, daily) -> bool:
    """Single source of truth for "should this pick advance the streak / be polled."

    If a decision record exists for *date*, its ``scoreable`` field is authoritative.
    When no record exists (legacy picks pre-dating decision.json), falls back to
    ``pick_was_delivered(daily)``.  The ``picks`` import is local to avoid a circular
    dependency (picks.py is heavier and imports from daily_decision indirectly).
    """
    from bts.picks import pick_was_delivered
    dec = load_decision(date, picks_dir)
    if dec is not None:
        return bool(dec.get("scoreable"))
    return bool(daily is not None and pick_was_delivered(daily))
