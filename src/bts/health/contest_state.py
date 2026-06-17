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

# Under the Phase-1 snapshot/coverage split, source_date is derived from the
# per-round predictions array, which trails the live activeStreak counter by ~one
# settled pick BY DESIGN (the counter is current; the per-round ledger lags). So a
# 1-pick gap is the normal coverage lag, surfaced as INFO at any time of day. The
# gap is counted in settled picks, NOT calendar days, so off-day stretches (the
# All-Star break) cannot inflate it. A gap of >= 2 settled picks is unusual (the
# predictions array shouldn't trail by two) but is NOT a reliable staleness signal
# under the coverage split (it can be a transient ledger lag) -> WARN, logged not
# DM'd. A genuinely broken fetch DMs via the auth-failure path; data corruption
# (no/future source_date, malformed file) stays CRITICAL.
EXPECTED_OVERNIGHT_LAG_STEPS = 1


def check(picks_dir: Path, *, expected: bool = False, now: datetime | None = None) -> list[Alert]:
    """Alert on contest-account state problems that affect live picks.

    CRITICAL: contest state missing/invalid/corrupt when expected (no source_date, a
    future source_date, or a malformed file). WARN: a >=2 settled-pick coverage gap
    (the predictions array trailing the live activeStreak counter -- informational,
    not DM'd), or a legacy/expired manual override (archive it; auto
    contest_streak.json is the source).
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
                message=(f"contest state has no source_date ({state.path}); no settled "
                         "rounds in the profile, so coverage/freshness cannot be verified "
                         "— investigate the fetch"),
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
            # CRITICAL. Under the Phase-1 snapshot/coverage split, source_date (from
            # the per-round predictions array) trails the live activeStreak counter
            # by ~one settled pick BY DESIGN, so gap==1 is the normal coverage lag ->
            # INFO at any time of day. gap>=2 is unusual but NOT a reliable staleness
            # signal under the coverage split (it can be a transient ledger lag), so
            # it is WARN (logged, not DM'd), not CRITICAL.
            gap_steps = resolved_pick_settlement_gap(picks_dir, state.source_date)
            if gap_steps > EXPECTED_OVERNIGHT_LAG_STEPS:
                alerts.append(Alert(
                    level="WARN",
                    source=SOURCE,
                    message=(f"contest coverage lag: {gap_steps} settled picks are newer than "
                             f"source_date={state.source_date} ({state.path}); latest resolved "
                             f"pick {latest}. The predictions array trails the live activeStreak "
                             "counter by >=2 -- usually a transient settlement lag (a broken "
                             "fetch DMs separately)"),
                ))
            elif gap_steps == EXPECTED_OVERNIGHT_LAG_STEPS:
                alerts.append(Alert(
                    level="INFO",
                    source=SOURCE,
                    message=(f"contest state lags 1 settled pick (expected coverage lag — the "
                             f"predictions array trails the live activeStreak counter): "
                             f"source_date={state.source_date}, latest resolved {latest}; "
                             "refreshes on the next fetch"),
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
