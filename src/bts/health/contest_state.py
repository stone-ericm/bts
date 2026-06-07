"""Contest-account state validation for live recommendation safety."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bts.contest_state import (
    ContestStateError,
    latest_resolved_pick_date,
    load_contest_streak_state,
    _parse_dt,
)
from bts.health.alert import Alert

SOURCE = "contest_state"

# A contest observation legitimately lags our own settlement by up to one day:
# the scheduler settles day D in the evening (latest_resolved -> D) before the
# contest account settles D and the next fetch (next morning) advances
# source_date to D. That one-day overnight gap is expected and harmless (no picks
# are made overnight; the afternoon pick sees a refreshed source_date), so it is
# surfaced as INFO. A gap beyond this is genuine staleness (the week-long-freeze
# incident class) and escalates to CRITICAL.
EXPECTED_OVERNIGHT_LAG_DAYS = 1


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

    if expected:
        latest = latest_resolved_pick_date(picks_dir)
        if state.source_date is None:
            alerts.append(Alert(
                level="CRITICAL",
                source=SOURCE,
                message=(f"contest state has no source_date ({state.path}); freshness "
                         "cannot be verified — live picks are frozen conservatively"),
            ))
        elif state.source_date > now.date() + timedelta(days=1):
            # A contest source_date can't be in the future (US contest dates trail
            # UTC). One day of grace absorbs any TZ skew; beyond that the file is
            # corrupt/fat-fingered and would otherwise pass the freshness check.
            alerts.append(Alert(
                level="CRITICAL",
                source=SOURCE,
                message=(f"contest state source_date={state.source_date} is in the FUTURE "
                         f"(now {now.date()}); file is corrupt/untrusted — investigate"),
            ))
        elif latest is not None:
            # gap==1 stays INFO regardless of time of day: a *persistent daytime*
            # lag means fetches are failing, which is owned by the separate
            # throttled fetch-failure DM (cli `_contest_fetch_alert`); it also
            # escalates here to CRITICAL once the next settlement makes gap>=2.
            gap_days = (latest - state.source_date).days
            if gap_days > EXPECTED_OVERNIGHT_LAG_DAYS:
                alerts.append(Alert(
                    level="CRITICAL",
                    source=SOURCE,
                    message=(f"contest state is STALE by {gap_days}d: {state.path} "
                             f"source_date={state.source_date} < latest resolved pick {latest}; "
                             "live picks are frozen conservatively"),
                ))
            elif gap_days == EXPECTED_OVERNIGHT_LAG_DAYS:
                alerts.append(Alert(
                    level="INFO",
                    source=SOURCE,
                    message=(f"contest state lags {gap_days}d (expected overnight window): "
                             f"source_date={state.source_date}, latest resolved {latest}; "
                             "refreshes on the next scheduled fetch"),
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
