"""Contest-account state validation for live recommendation safety."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bts.contest_state import (
    ContestStateError,
    latest_resolved_pick_date,
    load_contest_streak_state,
    resolved_pick_settlement_gap,
    _parse_dt,
)
from bts.health.alert import Alert

SOURCE = "contest_state"

# A contest observation legitimately lags our own settlement by exactly one
# *settled pick*: the scheduler settles day D in the evening (latest_resolved
# -> D) before the contest account settles D and the next fetch advances
# source_date. That one-pick lag is expected and harmless, so it is surfaced as
# INFO. The gap is counted in settled picks, NOT calendar days, so multi-day
# off-day stretches (the All-Star break) cannot inflate it into a false CRITICAL.
# A gap of >= 2 settled picks is genuine staleness (the week-long-freeze incident
# class, where picks resolve daily while source_date is frozen) and is CRITICAL.
EXPECTED_OVERNIGHT_LAG_STEPS = 1


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
            # Count of *settled picks* newer than source_date, not calendar days:
            # off-days (All-Star break) have no picks and must not fire a false
            # CRITICAL. gap==1 before noon ET is the expected overnight
            # settlement window (INFO). gap==1 PERSISTING past noon ET is not —
            # the 2026-06-12 incident showed the fetch can succeed all day
            # while the contest never advances (a delivered pick was never
            # entered in the MLB app; the fetch-failure DM never fires because
            # nothing fails). That case escalates to WARN, not CRITICAL: the
            # pick path is already conservative (doubles frozen).
            gap_steps = resolved_pick_settlement_gap(picks_dir, state.source_date)
            if gap_steps > EXPECTED_OVERNIGHT_LAG_STEPS:
                alerts.append(Alert(
                    level="CRITICAL",
                    source=SOURCE,
                    message=(f"contest state is STALE: {gap_steps} settled picks are newer "
                             f"than source_date={state.source_date} ({state.path}); latest "
                             f"resolved pick {latest}; live picks are frozen conservatively"),
                ))
            elif gap_steps == EXPECTED_OVERNIGHT_LAG_STEPS:
                from zoneinfo import ZoneInfo
                now_et = now.astimezone(ZoneInfo("America/New_York"))
                if now_et.hour >= 12:
                    alerts.append(Alert(
                        level="WARN",
                        source=SOURCE,
                        message=(f"contest state lag persisting past noon ET: "
                                 f"source_date={state.source_date}, latest resolved {latest}. "
                                 "Fetches are succeeding but the contest never advanced — "
                                 "check the pick was actually entered in the MLB app "
                                 "(check-pick-entered) or MLB settlement is delayed"),
                    ))
                else:
                    alerts.append(Alert(
                        level="INFO",
                        source=SOURCE,
                        message=(f"contest state lags 1 settled pick (expected overnight window): "
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
