"""Contest-account state validation for live recommendation safety."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bts.contest_state import (
    ContestStateError,
    contest_state_is_fresh,
    latest_resolved_pick_date,
    load_contest_streak_state,
    _parse_dt,
)
from bts.health.alert import Alert

SOURCE = "contest_state"


def check(picks_dir: Path, *, expected: bool = False, now: datetime | None = None) -> list[Alert]:
    """Alert on contest-account state problems that affect live picks.

    CRITICAL: missing/invalid when expected, or STALE when expected — the gap that
    let a frozen observation silently drive picks for a week. WARN: a legacy/expired
    manual override file is present (archive it; auto contest_streak.json is the source).
    """
    now = now or datetime.now(timezone.utc)
    try:
        state = load_contest_streak_state(picks_dir, now=now)
    except ContestStateError as exc:
        return [Alert(level="CRITICAL", source=SOURCE, message=str(exc))]

    alerts: list[Alert] = []
    if state is None:
        if expected:
            alerts.append(Alert(
                level="CRITICAL",
                source=SOURCE,
                message=("contest-account streak state expected but missing at "
                         f"{picks_dir / 'account_state'}"),
            ))
        return alerts

    if expected and not contest_state_is_fresh(state, picks_dir):
        latest = latest_resolved_pick_date(picks_dir)
        alerts.append(Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(f"contest state is STALE: {state.path} source_date={state.source_date} "
                     f"< latest resolved pick {latest}; live picks are frozen conservatively"),
        ))

    manual_path = picks_dir / "account_state" / "contest_streak.manual.json"
    if manual_path.exists():
        try:
            exp = json.loads(manual_path.read_text()).get("override_expires_at")
        except (json.JSONDecodeError, OSError):
            exp = None
        exp_dt = _parse_dt(exp)
        if exp_dt is None or exp_dt <= now:
            alerts.append(Alert(
                level="WARN",
                source=SOURCE,
                message=(f"legacy/expired manual override present at {manual_path}; "
                         "archive it (auto contest_streak.json is the live source)"),
            ))

    return alerts
