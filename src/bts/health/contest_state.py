"""Contest-account state validation for live recommendation safety."""

from __future__ import annotations

import json
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.contest_state import (
    ContestStateError,
    DATED_MANUAL_SCHEMA,
    dated_observation_in_effect,
    latest_resolved_pick_date,
    load_contest_streak_state,
    resolved_pick_settlement_gap,
    _parse_dt,
)
from bts.health.alert import Alert

SOURCE = "contest_state"
SAVER_SOURCE = "saver_state"

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
            manual = json.loads(manual_path.read_text())
            exp = manual.get("override_expires_at")
            dated = manual.get("schema_version") == DATED_MANUAL_SCHEMA
        except (json.JSONDecodeError, OSError):
            exp = None
            dated = False
        exp_dt = _parse_dt(exp)
        if not dated and (exp_dt is None or exp_dt <= now):
            alerts.append(Alert(
                level="WARN",
                source=SOURCE,
                message=(f"legacy/expired manual override present at {manual_path}; "
                         "archive it (auto contest_streak.json is the live source)"),
            ))

    if dated_observation_in_effect(state, now):
        if state.saver_state == "not_earned" and 10 <= state.streak <= 15:
            season = state.effective_date.year
            base = (f"bts set-contest-streak --date {state.effective_date.isoformat()} "
                    f"--streak {state.streak} --best-streak {state.best_streak}")
            directory = f" --picks-dir {shlex.quote(str(picks_dir))}"
            correction = (f"bts set-contest-streak --date {state.effective_date.isoformat()} "
                          "--streak N --best-streak B --saver not_earned" + directory)
            alerts.append(Alert(
                level="WARN", source=SAVER_SOURCE,
                message=(f"Streak Saver not_earned in dated observation {state.path}, season "
                         f"{season}, contradicts streak {state.streak}; counting the saver as spent. "
                         "Check MLB and run ONE correction to the observation. Earned and unused: "
                         f"`{base} --saver active{directory}`. Already used: "
                         f"`{base} --saver used{directory}`. Not earned: "
                         f"`{correction}`, replacing N and B with the active and best streaks MLB "
                         "shows. Changing the saver flag does not correct this observation."),
            ))
        return alerts

    # The flag belongs to the caller's current ET season, even when ledger coverage lags.
    from bts.saver_state import load_saver_state
    season = now.astimezone(ZoneInfo("America/New_York")).year
    saver = load_saver_state(picks_dir, season=season)
    in_zone = 10 <= state.streak <= 15
    if saver.state == "uninitialized" and (expected or in_zone):
        detail = (f"stale-season flag, season {saver.season}" if saver.season is not None
                  else "no saver_state.json")
        alerts.append(Alert(
            level="WARN", source=SAVER_SOURCE,
            message=(f"Streak Saver flag uninitialized for season {season} ({detail}) "
                     f"while streak is {state.streak}; counting the saver as spent. "
                     "Check the saver in MLB and run ONE of these. Not earned yet this season: "
                     f"`bts saver-state --init not_earned --season {season}`. Earned and unused: "
                     f"`bts saver-state --init active --season {season}`. Already used: "
                     f"`bts saver-state --init used --season {season}`"),
        ))
    elif saver.state == "not_earned" and in_zone:
        alerts.append(Alert(
            level="WARN", source=SAVER_SOURCE,
            message=(f"Streak Saver flag not_earned for season {season} contradicts streak "
                     f"{state.streak}; counting the saver as spent. Check the saver in MLB and "
                     "run ONE of these. Earned and unused: "
                     f"`bts saver-state --init active --force --season {season}`. Already used: "
                     f"`bts saver-state --use --season {season}`. If MLB shows the saver not earned "
                     f"yet, streak {state.streak} is a stale contest reading (dated "
                     f"{state.source_date}): leave the flag as it is and correct the contest streak "
                     "with bts set-contest-streak"),
        ))

    return alerts
