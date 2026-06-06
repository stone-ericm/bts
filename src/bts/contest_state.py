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
    if not isinstance(streak, int) or streak < 0:
        raise ContestStateError(f"contest streak invalid in {path}: {streak!r}")

    saver = data.get("saver_available")
    if saver is not None and not isinstance(saver, bool):
        raise ContestStateError(f"contest saver_available invalid in {path}: {saver!r}")

    best = data.get("best_streak")
    if best is not None and not isinstance(best, int):
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
    auto = _parse_state_file(auto_path) if auto_path.exists() else None

    if manual is not None and manual.override_expires_at is not None and manual.override_expires_at > now:
        return manual
    if auto is not None:
        return auto
    return manual


def latest_resolved_pick_date(picks_dir: Path) -> date | None:
    """Return the latest root-level production pick date with a settled result."""
    latest: date | None = None
    for path in picks_dir.glob("*.json"):
        if not _ISO_DATE_RE.match(path.stem):
            continue
        try:
            body = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if body.get("result") not in _RESOLVED_RESULTS:
            continue
        pick_date = date.fromisoformat(path.stem)
        if latest is None or pick_date > latest:
            latest = pick_date
    return latest


def contest_state_is_fresh(
    contest: ContestStreakState,
    picks_dir: Path,
) -> bool:
    """Return whether contest state was observed after the last settled pick."""
    latest = latest_resolved_pick_date(picks_dir)
    if latest is None:
        return True
    if contest.source_date is None:
        return False
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

    contest_saver = contest.saver_available if contest.saver_available is not None else False
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
