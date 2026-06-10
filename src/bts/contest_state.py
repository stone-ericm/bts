"""Contest-account streak state for live recommendations.

``streak.json`` remains the model/replay state derived from local pick files.
Contest state is a separate operator or external-source observation of the
actual MLB BTS account. Live recommendations should prefer fresh contest state;
simulation, shadow, and replay paths should keep using ``streak.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from bts.picks import load_saver_available, load_streak


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CONTEST_STATE_NAMES = (
    "contest_streak.manual.json",
    "contest_streak.json",
)
_RESOLVED_RESULTS = {"hit", "miss", "void"}


class ContestStateError(RuntimeError):
    """Contest-account state is missing or invalid for live decisions."""


@dataclass(frozen=True)
class ContestStreakState:
    streak: int
    saver_available: bool | None
    source: str
    source_date: date | None
    recorded_at: str | None
    path: Path
    best_streak: int | None = None
    override_expires_at: datetime | None = None


@dataclass(frozen=True)
class DecisionStreakState:
    """The streak state used for live pick action selection."""

    streak: int
    saver_available: bool
    allow_double: bool
    source: str
    status: str
    model_streak: int
    model_saver_available: bool
    contest_streak: int | None = None
    contest_saver_available: bool | None = None
    contest_source_date: date | None = None
    message: str | None = None


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _contest_state_paths(picks_dir: Path) -> list[Path]:
    state_dir = picks_dir / "account_state"
    return [state_dir / name for name in _CONTEST_STATE_NAMES]


def _parse_state_file(path: Path) -> ContestStreakState:
    """Parse one contest-streak observation file. Raises ContestStateError if malformed."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ContestStateError(f"contest streak state malformed at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContestStateError(f"contest streak state malformed at {path}: expected object")
    streak = data.get("active_streak", data.get("streak"))
    if type(streak) is not int or streak < 0:        # type() not isinstance: reject bool
        raise ContestStateError(f"contest streak invalid in {path}: {streak!r}")

    saver = data.get("saver_available")
    if saver is not None and not isinstance(saver, bool):
        raise ContestStateError(f"contest saver_available invalid in {path}: {saver!r}")

    best = data.get("best_streak")
    if best is not None and type(best) is not int:
        best = None

    return ContestStreakState(
        streak=streak,
        saver_available=saver,
        source=str(data.get("source") or path.stem),
        source_date=_parse_date(data.get("source_date") or data.get("as_of")),
        recorded_at=data.get("recorded_at") if isinstance(data.get("recorded_at"), str) else None,
        path=path,
        best_streak=best,
        override_expires_at=_parse_dt(data.get("override_expires_at")),
    )


def load_contest_streak_state(
    picks_dir: Path,
    *,
    now: datetime | None = None,
) -> ContestStreakState | None:
    """Select the active contest-streak observation.

    Precedence: an UNEXPIRED manual override wins; else the auto observation
    (``contest_streak.json``); else a legacy/expired manual file is used as a
    fallback so production keeps working until the first auto-fetch (health alerts
    on that case). Returns None when no observation exists.
    """
    now = now or datetime.now(timezone.utc)
    state_dir = picks_dir / "account_state"
    manual_path = state_dir / "contest_streak.manual.json"
    auto_path = state_dir / "contest_streak.json"

    manual = _parse_state_file(manual_path) if manual_path.exists() else None
    # An unexpired manual override wins; return it BEFORE parsing the auto file so
    # a corrupt auto file can't block the emergency override (audit D2). A corrupt
    # manual file still raises (surfaced as CRITICAL), as does a corrupt auto file
    # when no override is active — both fail closed.
    if (manual is not None and manual.override_expires_at is not None
            and manual.override_expires_at > now):
        return manual

    auto = _parse_state_file(auto_path) if auto_path.exists() else None
    if auto is not None:
        return auto
    return manual


def _resolved_pick_dates(picks_dir: Path) -> list[date]:
    """All root-level production pick dates with a settled result."""
    dates: list[date] = []
    for path in picks_dir.glob("*.json"):
        if not _ISO_DATE_RE.match(path.stem):
            continue
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if body.get("result") not in _RESOLVED_RESULTS:
            continue
        dates.append(date.fromisoformat(path.stem))
    return dates


def latest_resolved_pick_date(picks_dir: Path) -> date | None:
    """Return the latest root-level production pick date with a settled result."""
    return max(_resolved_pick_dates(picks_dir), default=None)


def resolved_pick_settlement_gap(picks_dir: Path, source_date: date) -> int:
    """Number of settled production picks dated strictly after ``source_date``.

    This is the contest's settlement lag measured in *picks*, not calendar days:
    off-days (the All-Star break, league off-days) have no picks, so they do not
    inflate it. Exactly 1 == the expected overnight lag (we settle day D before
    the contest does); >= 2 == genuine staleness (the week-long-freeze incident
    class, where picks resolve daily while source_date is frozen).
    """
    return sum(1 for d in _resolved_pick_dates(picks_dir) if d > source_date)


def contest_state_is_fresh(
    contest: ContestStreakState,
    picks_dir: Path,
) -> bool:
    """Return whether contest state was observed after the last settled pick.

    A contest observation with no ``source_date`` cannot be freshness-verified,
    so it is treated as stale (conservative) rather than failing open to fresh —
    checked first so a missing date is conservative even when there are no
    resolved picks (e.g. a corrupt picks dir where ``latest`` collapses to None).
    """
    if contest.source_date is None:
        return False
    latest = latest_resolved_pick_date(picks_dir)
    if latest is None:
        return True
    return contest.source_date >= latest


def load_decision_streak_state(
    picks_dir: Path,
    *,
    require_contest_state: bool = False,
    now: datetime | None = None,
) -> DecisionStreakState:
    """Return the streak/saver state for live user-facing recommendations.

    Fresh contest state wins. If a contest observation exists but is stale, keep
    the effective streak at least as high as the last-known contest streak and
    forbid automatic doubles. This avoids the dangerous under-count path where
    live recommendations double against a higher real account streak.
    """
    model_streak = load_streak(picks_dir)
    model_saver = load_saver_available(picks_dir)
    contest = load_contest_streak_state(picks_dir, now=now)
    if contest is None:
        if require_contest_state:
            paths = ", ".join(str(path) for path in _contest_state_paths(picks_dir))
            raise ContestStateError(
                f"contest-account streak state required but missing; expected one of: {paths}"
            )
        return DecisionStreakState(
            streak=model_streak,
            saver_available=model_saver,
            allow_double=True,
            source="model",
            status="model_only",
            model_streak=model_streak,
            model_saver_available=model_saver,
            message="no contest-account streak state found",
        )

    # The profile API can't observe the mulligan (contest.saver_available is
    # always None). When our model streak AGREES with the contest streak (the
    # normal 4x/day auto path) the locally-tracked model saver is a reliable
    # proxy, so use it to keep the saver-aware MDP line reachable (audit D3).
    # When they diverge (e.g. a manual override set a different streak) the model
    # saver no longer describes the contest account, so stay conservative.
    if contest.saver_available is not None:
        contest_saver = contest.saver_available
    elif contest.streak == model_streak:
        contest_saver = model_saver
    else:
        contest_saver = False
    if contest_state_is_fresh(contest, picks_dir):
        return DecisionStreakState(
            streak=contest.streak,
            saver_available=contest_saver,
            allow_double=True,
            source="contest",
            status="fresh",
            model_streak=model_streak,
            model_saver_available=model_saver,
            contest_streak=contest.streak,
            contest_saver_available=contest.saver_available,
            contest_source_date=contest.source_date,
            message=f"using fresh contest streak from {contest.source}",
        )

    return DecisionStreakState(
        streak=max(model_streak, contest.streak),
        saver_available=False,
        allow_double=False,
        source="contest",
        status="stale",
        model_streak=model_streak,
        model_saver_available=model_saver,
        contest_streak=contest.streak,
        contest_saver_available=contest.saver_available,
        contest_source_date=contest.source_date,
        message="contest streak is stale; freezing at conservative effective streak and disabling doubles",
    )
