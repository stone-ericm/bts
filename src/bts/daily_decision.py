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

DECISION_SCHEMA = "bts_daily_decision_v1"
_RANK_FIELDS = ("batter_id", "batter_name", "team", "game_pk", "p_game_hit")


def _utc_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _summary(cand: dict | None) -> dict | None:
    return None if cand is None else {k: cand.get(k) for k in _RANK_FIELDS}


def decision_path(date: str, picks_dir) -> Path:
    return Path(picks_dir) / date / "decision.json"


def write_decision(date, picks_dir, *, action, source, primary=None, double_down=None,
                   streak=None, saver_available=None, delivery_status, scoreable, now=None) -> dict | None:
    """Best-effort atomic write of the day's decision record. Returns the record, or None on any
    failure (must never raise into the live pick path)."""
    try:
        record = {
            "schema_version": DECISION_SCHEMA, "date": date,
            "action": action, "source": source,
            "primary": _summary(primary), "double_down": _summary(double_down),
            "streak": streak,
            "saver_available": (None if saver_available is None else bool(saver_available)),
            "delivery_status": delivery_status, "scoreable": bool(scoreable),
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
        if not isinstance(rec, dict) or rec.get("schema_version") != DECISION_SCHEMA:
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
