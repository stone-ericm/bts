"""Contest-account streak state for live recommendations.

``streak.json`` remains the model/replay state derived from local pick files.
Contest state is a separate operator or external-source observation of the
actual MLB BTS account. Live recommendations should prefer fresh contest state;
simulation, shadow, and replay paths should keep using ``streak.json``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.saver_state import SaverState, load_saver_state, policy_saver_available


logger = logging.getLogger(__name__)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CONTEST_STATE_NAMES = (
    "contest_streak.manual.json",
    "contest_streak.json",
)
_RESOLVED_RESULTS = {"hit", "miss", "void"}
# What reading and decoding a state or pick file can raise. ValueError covers malformed JSON,
# bytes that are not UTF-8, and a JSON integer longer than the interpreter's conversion limit
# (a plain ValueError, not JSONDecodeError); RecursionError is a body nested too deeply.
_UNREADABLE = (OSError, ValueError, RecursionError)


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
    schema_version: str | None = None
    effective_date: date | None = None
    saver_state: str | None = None
    season: int | None = None
    activation_id: str | None = None


@dataclass(frozen=True)
class DecisionStreakState:
    """The streak state used for live pick action selection."""

    streak: int
    saver_available: bool
    allow_double: bool
    source: str
    status: str
    model_streak: int | None
    model_saver_available: bool | None
    contest_streak: int | None = None
    contest_saver_available: bool | None = None
    contest_source_date: date | None = None
    message: str | None = None
    # Season-best streak as supplied by the selected observation, and whether the
    # tail policy may TRUST it to authorise its terminal stop (2026-09-03). Only a
    # trusted best can stop the account; "untrusted"/"missing" degrade to
    # best = streak downstream (strategy._normalize_best), which keeps picking.
    best_streak: int | None = None
    best_status: str = "missing"   # "trusted" | "untrusted" | "missing"
    model_state_error: str | None = None
    saver_flag_state: str | None = None


TARGET_STREAK = 57
_AUTO_SCHEMA = "bts_contest_streak_auto_v1"
_AUTO_SOURCE = "mlb_bts_profile"
_MANUAL_SCHEMA_PREFIX = "bts_contest_streak_manual"
DATED_MANUAL_SCHEMA = "bts_contest_streak_manual_v3"


def dated_observation_in_effect(state: ContestStreakState, now: datetime) -> bool:
    """A dated observation applies only to the caller's current contest date.

    No elapsed/expired state is remembered: stepping the clock back restores it.
    """
    return (state.effective_date is not None
            and state.effective_date == now.astimezone(ZoneInfo("America/New_York")).date())


def validate_dated_observation(data: object, path: Path) -> ContestStreakState:
    """Validate v3 contents in contract order, independently of today's date.

    Both the file reader and the command's raw-snapshot comparison use this
    validator. Other schemas retain the legacy parser and its timestamp grammar.
    """
    def invalid(number: int, field: str, requirement: str) -> None:
        raise ContestStateError(
            f"dated contest observation invalid at {path}: check {number}, "
            f"{field}: {requirement}")

    if not isinstance(data, dict):
        invalid(1, "file", "expected a JSON object")

    raw_date = data.get("effective_date")
    if not isinstance(raw_date, str):
        invalid(2, "effective_date", "must be a padded YYYY-MM-DD string")
    try:
        effective_date = date.fromisoformat(raw_date)
    except ValueError:
        invalid(2, "effective_date", "must be a padded YYYY-MM-DD string")
    if raw_date != effective_date.isoformat():
        invalid(2, "effective_date", "must be a padded YYYY-MM-DD string")

    season = data.get("season")
    if type(season) is not int or season != effective_date.year:
        invalid(3, "season", "must be an integer equal to effective_date's year")

    def timestamp(field: str, number: int) -> datetime:
        raw = data.get(field)
        if not isinstance(raw, str):
            invalid(number, field, "must be a timezone-aware ISO timestamp")
        try:
            parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
        except ValueError:
            invalid(number, field, "must be a timezone-aware ISO timestamp")
        if parsed.tzinfo is None:
            invalid(number, field, "must be a timezone-aware ISO timestamp")
        return parsed

    expires_at = timestamp("expires_at", 4)
    end_of_day = datetime.combine(effective_date, time.max, ZoneInfo("America/New_York"))
    if expires_at != end_of_day:
        invalid(4, "expires_at", "must equal the last microsecond of effective_date in America/New_York")
    active = data.get("active_streak")
    if type(active) is not int or not 0 <= active <= TARGET_STREAK:
        invalid(5, "active_streak", "must be an integer from 0 to 57")
    best = data.get("best_streak")
    if type(best) is not int or not active <= best <= TARGET_STREAK:
        invalid(6, "best_streak", "must be an integer from active_streak to 57")
    saver = data.get("saver_state")
    if not isinstance(saver, str) or saver not in {"not_earned", "active", "used"}:
        invalid(7, "saver_state", "must be not_earned, active or used")
    timestamp("recorded_at", 8)
    if not isinstance(data.get("source"), str):
        invalid(9, "source", "must be a string")
    if "reason" in data and not isinstance(data["reason"], str):
        invalid(10, "reason", "must be a string when present")
    return ContestStreakState(
        streak=active, best_streak=best, saver_available=None, source=data["source"],
        source_date=effective_date, recorded_at=data["recorded_at"], path=path,
        override_expires_at=expires_at, schema_version=DATED_MANUAL_SCHEMA,
        effective_date=effective_date, saver_state=saver,
    )


def classify_best_streak(contest: "ContestStreakState | None", *, now: datetime,
                         now_year: int) -> tuple[int | None, str]:
    """(best_streak as supplied, trust status) for the selected observation.

    Trusted iff: an integer; ``streak <= best <= 57`` (the auto fetch enforces
    best >= active, a hand-written manual file may not; >57 is impossible without
    having won, so it is a typo); the observation carries a source_date in the
    CURRENT season and not in the future (ET); and its CONTENTS prove provenance —
    the auto schema + the profile source, a dated manual observation in effect,
    or a legacy manual schema whose override is unexpired (an explicit operator
    statement). Provenance is never inferred from
    the filename (Codex r3: copied/symlinked files are trust-elevating boundaries
    otherwise). (Codex r2 P0: ``max(streak, best)`` is an algebraic clamp, not a
    trust boundary — an inflated best would otherwise stop the account for the
    rest of the season.)
    """
    if contest is None or contest.best_streak is None:
        return None, "missing"
    best = int(contest.best_streak)
    if best < 0 or best < contest.streak or best > TARGET_STREAK:
        return best, "untrusted"
    if dated_observation_in_effect(contest, now):
        return best, "trusted"
    today_et = now.astimezone(ZoneInfo("America/New_York")).date()
    if (contest.source_date is None or contest.source_date.year != now_year
            or contest.source_date > today_et):
        return best, "untrusted"
    schema = contest.schema_version or ""
    if schema in (_AUTO_SCHEMA, "bts_contest_streak_auto_v2") and contest.source == _AUTO_SOURCE:
        return best, "trusted"
    if schema.startswith(_MANUAL_SCHEMA_PREFIX) and (
            contest.override_expires_at is not None and contest.override_expires_at > now):
        return best, "trusted"
    return best, "untrusted"


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


def _parse_state_file(path: Path, *, require_dated: bool = False) -> ContestStreakState:
    """Parse one contest-streak observation file. Raises ContestStateError if malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except _UNREADABLE as exc:
        raise ContestStateError(f"contest streak state malformed at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContestStateError(f"contest streak state malformed at {path}: expected object")
    if path.name == "contest_streak.manual.json" and data.get("schema_version") == DATED_MANUAL_SCHEMA:
        return validate_dated_observation(data, path)
    streak = data.get("active_streak", data.get("streak"))
    if type(streak) is not int or streak < 0:        # type() not isinstance: reject bool
        raise ContestStateError(f"contest streak invalid in {path}: {streak!r}")

    saver = data.get("saver_available")
    if saver is not None and not isinstance(saver, bool):
        raise ContestStateError(f"contest saver_available invalid in {path}: {saver!r}")

    best = data.get("best_streak")
    if best is not None and type(best) is not int:
        best = None

    if path.name == "contest_streak.json" and (
        streak > TARGET_STREAK
        or (best is not None and (best < streak or best > TARGET_STREAK))
    ):
        raise ContestStateError(
            f"automatic contest streak invalid in {path}: active_streak={streak}, best_streak={best}"
        )

    source_date = _parse_date(data.get("source_date") or data.get("as_of"))
    if (require_dated and source_date is None and _parse_date(data.get("as_of")) is None
            and _parse_dt(data.get("recorded_at")) is None):
        raise ContestStateError(f"automatic contest observation is undated at {path}")

    season = activation_id = None
    if path.name == "contest_streak.json" and data.get("schema_version") == "bts_contest_streak_auto_v2":
        from bts.season_contract import valid_activation_id
        season = data.get("season")
        if type(season) is not int:
            raise ContestStateError(f"automatic contest season invalid in {path}: must be an integer")
        activation_id = data.get("activation_id")
        if not valid_activation_id(activation_id):
            raise ContestStateError(f"automatic contest activation_id invalid in {path}: must be 32 lowercase hexadecimal characters")

    return ContestStreakState(
        streak=streak,
        saver_available=saver,
        source=str(data.get("source") or path.stem),
        source_date=source_date,
        recorded_at=data.get("recorded_at") if isinstance(data.get("recorded_at"), str) else None,
        path=path,
        best_streak=best,
        override_expires_at=_parse_dt(data.get("override_expires_at")),
        schema_version=(data.get("schema_version") if isinstance(data.get("schema_version"), str) else None),
        season=season,
        activation_id=activation_id,
    )


def load_contest_streak_state(
    picks_dir: Path,
    *,
    now: datetime | None = None,
    require_dated_auto: bool = False,
) -> ContestStreakState | None:
    """Select the active contest-streak observation.

    A dated manual observation wins on its effective date and is ignored on all
    other dates. For legacy files, an UNEXPIRED manual override wins; else the auto observation
    (``contest_streak.json``); else a legacy/expired manual file is used as a
    fallback so production keeps working until the first auto-fetch (health alerts
    on that case). Returns None when no observation exists.

    Required decision reads set require_dated_auto; other callers keep legacy
    acceptance of undated automatic files during the CS1 sequencing step.
    """
    now = now or datetime.now(timezone.utc)
    state_dir = picks_dir / "account_state"
    manual_path = state_dir / "contest_streak.manual.json"
    auto_path = state_dir / "contest_streak.json"

    manual = _parse_state_file(manual_path) if manual_path.exists() else None
    if manual is not None and manual.effective_date is not None:
        if dated_observation_in_effect(manual, now):
            return manual
        # A dated observation for another day is absent, including as a fallback.
        manual = None
    # An unexpired manual override wins; return it BEFORE parsing the auto file so
    # a corrupt auto file can't block the emergency override (audit D2). A corrupt
    # manual file still raises (surfaced as CRITICAL), as does a corrupt auto file
    # when no override is active — both fail closed.
    if (manual is not None and manual.override_expires_at is not None
            and manual.override_expires_at > now):
        return manual

    auto = (_parse_state_file(auto_path, require_dated=require_dated_auto)
            if auto_path.exists() else None)
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
            pick_date = date.fromisoformat(path.stem)
        except ValueError:        # ISO-shaped but invalid (e.g. 2026-99-99)
            continue
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except _UNREADABLE:
            continue
        if not isinstance(body, dict):
            continue
        result = body.get("result")
        if not isinstance(result, str) or result not in _RESOLVED_RESULTS:
            continue
        dates.append(pick_date)
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


def _has_unconfirmed_miss(picks_dir: Path, source_date: date) -> bool:
    """True if a settled local pick dated strictly after ``source_date`` is a MISS.

    The bot resolves a pick locally before the contest posts it; a local miss the
    contest hasn't confirmed means the real streak may have reset (stale-high risk).
    The bot only *recommends*, so this is an uncertainty signal, not proof of a reset.
    """
    for path in picks_dir.glob("*.json"):
        if not _ISO_DATE_RE.match(path.stem):
            continue
        try:
            pick_date = date.fromisoformat(path.stem)
        except ValueError:        # ISO-shaped but invalid (e.g. 2026-99-99)
            continue
        if pick_date <= source_date:
            continue
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except _UNREADABLE:
            continue
        if not isinstance(body, dict):
            continue
        if body.get("result") == "miss":
            return True
    return False


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


def _load_model_diagnostics(picks_dir: Path, now_year: int) -> tuple[int | None, bool | None, str | None]:
    """Read one model snapshot without changing the scorer loaders' behavior.

    Missing files and unmatched seasons return fresh model diagnostics. A
    matching season keeps the legacy key defaults. Corrupt files make BOTH
    diagnostics unavailable, never a zero.
    """
    path = picks_dir / "streak.json"
    try:
        if not path.exists():
            return 0, True, None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("expected object")
        if type(data.get("season")) is not int or data["season"] != now_year:
            return 0, True, None
        return data.get("streak", 0), data.get("saver_available", True), None
    except (*_UNREADABLE, TypeError) as exc:
        error = f"{path}: {type(exc).__name__}: {exc}"
        logger.warning("model_state_corrupt: %s", error)
        return None, None, error


def load_decision_streak_state(
    picks_dir: Path,
    *,
    require_contest_state: bool = False,
    now: datetime | None = None,
) -> DecisionStreakState:
    """Return the streak/saver state for live user-facing recommendations.

    The decision streak is the real contest (MLB) streak; the model replay can
    NEVER raise it (the 2026-06-17 inflation bug). ``status`` in {fresh, lagged,
    stale} reflects settlement freshness. Doubles are not frozen on staleness —
    Phase 1 surfaces stale-high via ``status``; Phase 2 makes strategy act on the
    uncertainty.
    """
    now_dt = now or datetime.now(timezone.utc)
    now_year = now_dt.astimezone(ZoneInfo("America/New_York")).year
    model_streak, model_saver, model_error = _load_model_diagnostics(picks_dir, now_year)
    contest = load_contest_streak_state(
        picks_dir, now=now_dt, require_dated_auto=require_contest_state)
    if require_contest_state and not (
            contest is not None and dated_observation_in_effect(contest, now_dt)):
        from bts.season_contract import live_automatic_refusal
        refusal = live_automatic_refusal(picks_dir, now_dt, contest)
        if refusal is not None:
            raise ContestStateError(refusal)
    if contest is None:
        if require_contest_state:
            paths = ", ".join(str(path) for path in _contest_state_paths(picks_dir))
            raise ContestStateError(
                f"contest-account streak state required but missing; expected one of: {paths}"
            )
        if model_error is not None:
            raise ContestStateError(f"model state is corrupt; cannot use model fallback: {model_error}")
        # Here the model streak IS the decision streak, and the policy saver compares it with 10.
        # A readable streak.json can hold any JSON value in that field; only a count is usable.
        if type(model_streak) is not int or model_streak < 0:
            raise ContestStateError(
                "model state is corrupt; cannot use model fallback: "
                f"{picks_dir / 'streak.json'}: streak is not a non-negative integer")
        saver = load_saver_state(picks_dir, season=now_year)
        return DecisionStreakState(
            streak=model_streak,
            # Model-only fallback still reads the saver from the flag (not streak.json), so the
            # live saver has one authority whether or not a contest observation exists.
            saver_available=policy_saver_available(saver, model_streak),
            saver_flag_state=saver.state,
            allow_double=True,
            source="model",
            status="model_only",
            model_streak=model_streak,
            model_saver_available=model_saver,
            message="no contest-account streak state found",
        )

    dated_in_effect = dated_observation_in_effect(contest, now_dt)
    if dated_in_effect or contest_state_is_fresh(contest, picks_dir):
        status = "fresh"
        message = f"using fresh contest streak from {contest.source}"
    elif contest.source_date is None:
        status = "stale"
        message = "contest streak has no source_date; treating as last-confirmed (stale)"
    else:
        gap = resolved_pick_settlement_gap(picks_dir, contest.source_date)
        if gap >= 2 or _has_unconfirmed_miss(picks_dir, contest.source_date):
            status = "stale"
            message = "contest streak stale; using last confirmed value (current may be lower)"
        else:
            status = "lagged"
            message = "contest streak lagged by expected overnight settlement; using last confirmed value"

    # On a dated observation's day its recorded saver is the authority. Otherwise
    # the live saver comes from saver_state.json (the manual Streak Saver flag) --
    # replaces the unsound infer_saver/best_streak inference (the streak-preserving save can
    # vanish from the windowed ledger, so available-vs-used is not observable from it). See the
    # 2026-06-18 spec. Read-only here; the flag is written by the fetch-path auto-earn and the
    # CLI/dashboard. `contest.saver_available` is retired from the decision (it is still surfaced
    # as the `contest_saver_available` diagnostic below).
    saver = (SaverState(state=contest.saver_state, season=contest.effective_date.year)
             if dated_in_effect else load_saver_state(picks_dir, season=now_year))
    contest_saver = policy_saver_available(saver, contest.streak)
    best_streak, best_status = classify_best_streak(contest, now=now_dt, now_year=now_year)

    # The decision streak is ALWAYS the contest (real MLB) value. The model is a
    # research replay of the bot's own suggestions and can NEVER raise it (the
    # 2026-06-17 inflation bug). Doubles are no longer frozen on staleness — Phase 1
    # surfaces stale-high via `status`; Phase 2 makes strategy act on the uncertainty.
    return DecisionStreakState(
        streak=contest.streak,
        saver_available=contest_saver,
        saver_flag_state=saver.state,
        allow_double=True,
        source="contest",
        status=status,
        model_streak=model_streak,
        model_saver_available=model_saver,
        contest_streak=contest.streak,
        contest_saver_available=contest.saver_available,
        contest_source_date=contest.source_date,
        message=message,
        best_streak=best_streak,
        best_status=best_status,
        model_state_error=model_error,
    )
