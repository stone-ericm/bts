"""Pick persistence, streak tracking, and MLB API helpers for BTS automation."""

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from collections.abc import Mapping
from pathlib import Path

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

from bts.util import atomic_write_text, retry_urlopen
from bts.data.schema import HIT_EVENTS, PA_ENDING_EVENTS  # pandas-free constants

API_BASE = "https://statsapi.mlb.com"
log = logging.getLogger(__name__)

FEATURE_ENV_SCHEMA_VERSION = "bts_feature_env_v1"

# Scale-affecting env keys only. Do not hash all of os.environ: unrelated
# host/process churn would create false production-scale discontinuities. This
# does not capture non-env runtime composition such as a future TOML-driven
# blend/model set; that would need separate provenance if it becomes variable.
FEATURE_ENV_DEFAULTS = {
    # Rookie shrinkage changes batter rolling-HR features before prediction.
    "BTS_ROOKIE_GATE_K": "20",
    # Pitcher rolling-window support changes pitcher HR feature scale.
    "BTS_PITCHER_HR_30G_MIN_PERIODS": "7",
    # LightGBM seed changes the fitted model when a daily blend is trained.
    "BTS_LGBM_RANDOM_STATE": "42",
    # Deterministic mode changes LightGBM training/reduction behavior.
    "BTS_LGBM_DETERMINISTIC": "0",
    # Optional post-model calibration directly rewrites p_game_hit.
    "BTS_USE_CALIBRATION": "0",
}


def _git_head_sha(cwd: Path | str = ".") -> str | None:
    """Return git rev-parse HEAD for ``cwd``, or None on any failure.

    Provenance helper. Failures (cwd is not a git repo, git binary missing,
    timeout) MUST be non-fatal — a failed sha read should never block a
    pick save. Per Codex bus #168.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _sha256_file(path: Path | str | None) -> str | None:
    """Return hex sha256 of the file at ``path``, or None on any failure.

    Returns None when ``path`` is None, doesn't exist, or any I/O error
    occurs. Failures MUST be non-fatal per Codex bus #168. Used only as
    a content-identity hash over already-existing artifact files; this
    helper does not deserialize the content.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def compute_feature_env_fingerprint(
    env: Mapping[str, str] | None = None,
) -> dict[str, str | dict[str, str]]:
    """Return a stable fingerprint of scale-affecting runtime config.

    The hash is over resolved values, including defaults when variables are
    unset. That makes "unset in production" and "explicitly set to the default"
    the same scale state, while deliberate changes to listed keys move the hash.
    """
    source = os.environ if env is None else env
    values = {
        key: str(source.get(key, default))
        for key, default in sorted(FEATURE_ENV_DEFAULTS.items())
    }
    payload = {
        "schema_version": FEATURE_ENV_SCHEMA_VERSION,
        "values": values,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "feature_env_schema_version": FEATURE_ENV_SCHEMA_VERSION,
        "feature_env": values,
        "feature_env_hash": hashlib.sha256(encoded).hexdigest(),
    }


def _safe_feature_env_fingerprint() -> dict[str, str | dict[str, str] | None]:
    try:
        return compute_feature_env_fingerprint()
    except Exception as exc:
        log.warning("feature-env fingerprint failed; continuing without it: %s", exc)
        return {
            "feature_env_schema_version": FEATURE_ENV_SCHEMA_VERSION,
            "feature_env": None,
            "feature_env_hash": None,
        }


def compute_provenance(
    blend_path: Path | str | None = None,
    policy_path: Path | str | None = None,
    cwd: Path | str = ".",
) -> dict[str, str | dict[str, str] | None]:
    """Bundle provenance fields for a DailyPick.

    Returns a dict with artifact hashes plus feature-env fingerprint fields.
    Each hash value is either a hex string or None. None values reflect "the
    artifact/fingerprint is genuinely unavailable" or "the git/hash call
    failed" — they MUST NOT cause callers to error out (per Codex #168).

    ``blend_path`` is the path of the cached blend artifact written by
    ``bts.model.predict.run_pipeline``; the field name on DailyPick
    follows the existing on-disk convention (``model_pickle_sha256``).
    """
    feature_env = _safe_feature_env_fingerprint()
    return {
        "model_git_sha": _git_head_sha(cwd),
        "model_pickle_sha256": _sha256_file(blend_path),
        "policy_npz_sha256": _sha256_file(policy_path),
        "feature_env_schema_version": feature_env["feature_env_schema_version"],
        "feature_env": feature_env["feature_env"],
        "feature_env_hash": feature_env["feature_env_hash"],
    }


def attach_provenance(
    daily: "DailyPick",
    blend_path: Path | str | None = None,
    policy_path: Path | str | None = None,
    cwd: Path | str = ".",
) -> "DailyPick":
    """Attach provenance v1 fields to a freshly-predicted DailyPick.

    Mutates and returns ``daily`` (callable as either an effect or an
    expression). Wraps :func:`compute_provenance` and writes the three
    fields directly onto the dataclass. Failure modes are inherited from
    the helpers — None values are silently accepted; this never raises.

    Use only when ``daily`` represents a fresh prediction run; re-saves
    of an already-saved DailyPick should preserve the existing provenance
    (it round-trips through load_pick).
    """
    prov = compute_provenance(blend_path=blend_path, policy_path=policy_path, cwd=cwd)
    daily.model_git_sha = prov["model_git_sha"]
    daily.model_pickle_sha256 = prov["model_pickle_sha256"]
    daily.policy_npz_sha256 = prov["policy_npz_sha256"]
    daily.feature_env_schema_version = prov["feature_env_schema_version"]
    daily.feature_env = prov["feature_env"]
    daily.feature_env_hash = prov["feature_env_hash"]
    return daily


@dataclass
class Pick:
    batter_name: str
    batter_id: int
    team: str
    lineup_position: int
    pitcher_name: str
    pitcher_id: int | None
    p_game_hit: float
    flags: list[str]
    projected_lineup: bool
    game_pk: int
    game_time: str  # ISO 8601 UTC
    pitcher_team: str | None = None


@dataclass
class DailyPick:
    date: str
    run_time: str
    pick: Pick
    double_down: Pick | None
    runner_up: dict | None  # {"batter_name": str, "p_game_hit": float}
    bluesky_posted: bool = False
    bluesky_uri: str | None = None
    notification_sent: bool = False
    notification_channel: str | None = None
    notification_id: str | None = None
    # E2 idempotency: persisted True *before* a network delivery (post/DM) and
    # cleared on a caught failure. If it survives as True while the pick is not
    # delivered, the daemon crashed mid-send → don't re-send (avoid a duplicate).
    delivery_attempted: bool = False
    result: str | None = None  # "hit", "miss", "void", "suspended", "unresolved", or None (pending)
    slot_results: dict[str, str] | None = None  # {"pick": "hit|miss|void", "double_down": ...}
    # Shadow-stack identity: stamped by save_shadow_pick with the current
    # SHADOW_MODEL_NAME so shadow_eval can filter history by feature-stack
    # version (v1 files must not count toward v2 review thresholds).
    # None on production picks and on legacy (pre-v2) shadow files.
    shadow_model_version: str | None = None
    # Provenance v1 (added 2026-05-04, per Codex bus #168). All optional;
    # old picks lack these fields and load_pick backfills via .get(...).
    # See bts.picks.compute_provenance for the helper that populates them.
    model_git_sha: str | None = None  # git rev-parse HEAD at predict/save time
    model_pickle_sha256: str | None = None  # sha256 of blend artifact actually used
    policy_npz_sha256: str | None = None  # sha256 of mdp_policy.npz if loaded
    feature_env_schema_version: str | None = None  # fingerprint schema id
    feature_env: dict[str, str] | None = None  # resolved scale-affecting config
    feature_env_hash: str | None = None  # sha256 of resolved feature_env payload


@dataclass(frozen=True)
class PickLockState:
    """Classification for whether an existing pick can be reused."""

    locked: bool
    stale: bool = False
    reason: str = ""
    game_pk: int | None = None
    abstract: str | None = None
    detailed: str | None = None


def pick_from_row(row) -> Pick:
    """Create a Pick from a prediction DataFrame row."""
    flags_str = row.get("flags", "")
    flags = [f.strip() for f in flags_str.split(",") if f.strip()] if flags_str else []
    return Pick(
        batter_name=row["batter_name"],
        batter_id=int(row["batter_id"]),
        team=row["team"],
        lineup_position=int(row["lineup"]),
        pitcher_name=row["pitcher_name"],
        pitcher_id=_optional_int(row.get("pitcher_id")),
        p_game_hit=float(row["p_game_hit"]),
        flags=flags,
        projected_lineup="PROJECTED" in flags_str,
        game_pk=int(row["game_pk"]),
        game_time=row["game_time"],
        pitcher_team=row.get("pitcher_team"),
    )


def _optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return None
        return int(stripped)
    try:
        if math.isnan(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        if str(value) in {"<NA>", "NaT"}:
            return None
        raise


def save_pick(daily: DailyPick, picks_dir: Path) -> Path:
    """Save daily pick to JSON file (overwrite-on-write).

    Also appends a lightweight observation to lineup_evolution_{date}.jsonl so
    we have an audit trail of how the pick changed across the day's lineup
    checks. Each save_pick call corresponds to one prediction run; the JSONL
    captures projected-vs-confirmed evolution so we can later analyze whether
    morning projected-lineup picks underperform confirmed-lineup picks.
    Audit-log failures must not block the save.
    """
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / f"{daily.date}.json"
    atomic_write_text(path, json.dumps(asdict(daily), indent=2))
    try:
        append_lineup_evolution(daily, picks_dir)
    except Exception:
        pass
    return path


def pick_was_delivered(daily: DailyPick) -> bool:
    """Return True when today's pick has been durably delivered to a human.

    Historically ``bluesky_posted`` doubled as both "public feed post exists"
    and "this pick is locked for the day." Private delivery needs a separate
    persisted signal so a scheduler restart after a DM does not regenerate or
    resend the pick before first pitch.
    """
    return bool(daily.bluesky_posted or (daily.notification_sent and daily.notification_id))


def append_lineup_evolution(daily: DailyPick, picks_dir: Path) -> Path:
    """Append one observation row to data/picks/lineup_evolution_{date}.jsonl.

    Emits one line per save_pick call. Through the day, this file accumulates
    the trajectory of {primary_pick, double_down} choices across lineup
    confirmations. Comparing the first row (often projected_lineup=True) to
    the last row (often projected_lineup=False) reveals whether the pick
    changed at confirm time.
    """
    log_path = picks_dir / f"lineup_evolution_{daily.date}.jsonl"

    def _slot(p) -> dict | None:
        if p is None:
            return None
        return {
            "batter_id": p.batter_id,
            "batter_name": p.batter_name,
            "team": p.team,
            "p_game_hit": p.p_game_hit,
            "projected_lineup": p.projected_lineup,
            "game_pk": p.game_pk,
        }

    entry = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "date": daily.date,
        "run_time": daily.run_time,
        "primary": _slot(daily.pick),
        "double_down": _slot(daily.double_down),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return log_path


def save_shadow_pick(daily: DailyPick, picks_dir: Path) -> Path:
    """Save shadow model pick to {date}.shadow.json."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / f"{daily.date}.shadow.json"
    # NOTE: save does NOT auto-stamp shadow_model_version — grading re-saves
    # of legacy v1 files must keep version=None so shadow_eval keeps excluding
    # them. Fresh shadow picks are stamped at creation by the scheduler via
    # shadow_eval.stamp_shadow_version.
    atomic_write_text(path, json.dumps(asdict(daily), indent=2))
    return path


def load_shadow_pick(date: str, picks_dir: Path) -> DailyPick | None:
    """Load shadow model pick. Returns None if not found.

    Honors the file's bluesky_posted/bluesky_uri fields verbatim so any
    corruption (e.g., shadow pipeline accidentally writing production data)
    stays visible on disk rather than being silently masked on save-back.
    In normal operation a shadow file should always have bluesky_posted=False.
    """
    path = picks_dir / f"{date}.shadow.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["pick"].setdefault("pitcher_team", None)
    if data.get("double_down"):
        data["double_down"].setdefault("pitcher_team", None)
    pick = Pick(**data["pick"])
    dd = Pick(**data["double_down"]) if data.get("double_down") else None
    return DailyPick(
        date=data["date"], run_time=data["run_time"], pick=pick,
        double_down=dd, runner_up=data.get("runner_up"),
        bluesky_posted=data.get("bluesky_posted", False),
        bluesky_uri=data.get("bluesky_uri"),
        notification_sent=data.get("notification_sent", False),
        notification_channel=data.get("notification_channel"),
        notification_id=data.get("notification_id"),
        delivery_attempted=data.get("delivery_attempted", False),
        result=data.get("result"),
        slot_results=data.get("slot_results"),
        model_git_sha=data.get("model_git_sha"),
        model_pickle_sha256=data.get("model_pickle_sha256"),
        policy_npz_sha256=data.get("policy_npz_sha256"),
        feature_env_schema_version=data.get("feature_env_schema_version"),
        feature_env=data.get("feature_env"),
        feature_env_hash=data.get("feature_env_hash"),
        shadow_model_version=data.get("shadow_model_version"),
    )


def load_pick(date: str, picks_dir: Path) -> DailyPick | None:
    """Load daily pick from JSON file. Returns None if not found."""
    path = picks_dir / f"{date}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    # Backfill pitcher_team for picks saved before this field existed
    data["pick"].setdefault("pitcher_team", None)
    if data["double_down"]:
        data["double_down"].setdefault("pitcher_team", None)
    return DailyPick(
        date=data["date"],
        run_time=data["run_time"],
        pick=Pick(**data["pick"]),
        double_down=Pick(**data["double_down"]) if data["double_down"] else None,
        runner_up=data["runner_up"],
        bluesky_posted=data.get("bluesky_posted", False),
        bluesky_uri=data.get("bluesky_uri"),
        notification_sent=data.get("notification_sent", False),
        notification_channel=data.get("notification_channel"),
        notification_id=data.get("notification_id"),
        delivery_attempted=data.get("delivery_attempted", False),
        result=data.get("result"),
        slot_results=data.get("slot_results"),
        # Provenance v1 — defaults to None for picks saved before these fields existed.
        model_git_sha=data.get("model_git_sha"),
        model_pickle_sha256=data.get("model_pickle_sha256"),
        policy_npz_sha256=data.get("policy_npz_sha256"),
        feature_env_schema_version=data.get("feature_env_schema_version"),
        feature_env=data.get("feature_env"),
        feature_env_hash=data.get("feature_env_hash"),
    )


def load_streak(picks_dir: Path) -> int:
    """Load current streak count. Returns 0 if no streak file."""
    path = picks_dir / "streak.json"
    if not path.exists():
        return 0
    return json.loads(path.read_text()).get("streak", 0)


def load_saver_available(picks_dir: Path) -> bool:
    """Load streak saver status. True if not yet consumed this season."""
    path = picks_dir / "streak.json"
    if not path.exists():
        return True
    return json.loads(path.read_text()).get("saver_available", True)


def save_streak(streak: int, picks_dir: Path, saver_available: bool | None = None) -> None:
    """Save current streak count and saver status."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / "streak.json"
    # Preserve existing saver state if not explicitly set
    existing_saver = True
    if path.exists() and saver_available is None:
        existing_saver = json.loads(path.read_text()).get("saver_available", True)
    atomic_write_text(path, json.dumps({
        "streak": streak,
        "saver_available": saver_available if saver_available is not None else existing_saver,
        "updated": datetime.now(timezone.utc).isoformat(),
    }))


@contextmanager
def scoring_lock(picks_dir: Path):
    """Serialize result scoring across the daemon and the 1am cron scorer.

    Both do load daily -> resolve -> update_streak -> save_pick around 01:00 ET
    on the same date (review F13); an unlocked interleave can double-apply or
    lose a streak update. Callers MUST re-load the pick and re-check its result
    inside the lock before updating the streak. Scoring is sub-second, so
    blocking on the peer is fine (mirrors saver_state's flock)."""
    lock_path = Path(picks_dir) / ".scoring.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def update_streak(results: list[bool], picks_dir: Path) -> int:
    """Update streak based on pick results.

    Single pick: [True] -> +1, [False] -> 0
    Double-down: [True, True] -> +2, anything else -> 0
    Voided postponed/cancelled slots must be omitted by callers; voids do not
    advance the streak, reset it, or consume the streak saver.

    Handles streak saver: if miss at streak 10-15 with saver available,
    streak holds and saver is consumed.
    """
    current = load_streak(picks_dir)
    saver = load_saver_available(picks_dir)

    if all(results):
        new = current + len(results)
        save_streak(new, picks_dir)
        return new

    # Miss — check saver
    if saver and 10 <= current <= 15:
        save_streak(current, picks_dir, saver_available=False)
        return current  # streak preserved, saver consumed

    save_streak(0, picks_dir)
    return 0


def _apply_streak_day(streak: int, saver: bool, is_hit: bool, increment: int) -> tuple[int, bool]:
    """One day of the streak state machine (matches update_streak's saver rule).

    A miss is forgiven (streak preserved, saver consumed) ONLY when the saver is
    available AND the streak is in [10, 15] at that point; otherwise it resets.
    """
    if is_hit:
        return streak + increment, saver
    if saver and 10 <= streak <= 15:
        return streak, False
    return 0, saver


def _replay_season_streak(
    picks_dir: Path, season: int, today_iso: str
) -> tuple[int, bool] | None:
    """Forward-replay the model streak + saver over the season's resolved picks.

    Returns ``(streak, saver_available)``, or ``None`` if the history is
    incomplete or unreadable. A backward suffix can't safely reconstruct saver
    forgiveness (it depends on the forward streak at the miss and whether the
    saver was already used), so forward replay is the sound approach — and on any
    unresolved/corrupt past pick we fail closed (return None) and let the caller
    keep the live-tracked streak rather than risk an over-count (audit D4).
    """
    streak, saver = 0, True
    for f in sorted(picks_dir.glob(f"{season}-*.json")):
        if not _ISO_DATE_RE.match(f.stem) or f.stem >= today_iso:
            continue  # skip non-date files and today/future (unplayed) previews
        try:
            data = json.loads(f.read_text())
        except Exception:
            return None
        r = data.get("result")
        if r == "void":
            continue
        if r not in ("hit", "miss"):
            return None  # unresolved past pick -> history not reconstructible
        if r == "hit":
            daily = load_pick(f.stem, picks_dir)
            if daily is None:
                return None
            streak, saver = _apply_streak_day(
                streak, saver, True, streak_increment_for_resolved_hit(daily)
            )
        else:
            streak, saver = _apply_streak_day(streak, saver, False, 0)
    return streak, saver


def is_resume_date_game(game: dict, date) -> bool:
    """True if this MLB schedule game is a suspended game being *resumed* on ``date``.

    A suspended game keeps its original ``officialDate``; when resumed on a later day it
    still appears on that day's schedule (officialDate earlier than the queried date).
    Per BTS rules the resumed portion is never evaluated, so a pick on the resume day can
    never score -- callers must not offer such a game as a candidate. A missing
    officialDate defensively returns False (keep the game). Both sides are compared on a
    ``YYYY-MM-DD`` prefix, which is correct for the ISO date / Timestamp /
    ``"YYYY-MM-DD 00:00:00"`` forms the MLB schedule and callers use. See
    docs/audit/2026-06-29-skip-threshold-and-discrimination.md (the 2026-06-17
    live_forward_resolution stall, game 824912 resumed from 06-16).
    """
    official = game.get("officialDate")
    if not official:
        return False
    # Resumed games carry an EARLIER officialDate; use `<` (not `!=`) so a future or
    # otherwise-odd official date is never silently dropped from the slate.
    return str(official)[:10] < str(date)[:10]


def get_game_statuses(date: str) -> dict[int, str]:
    """Get game statuses for all games on a date.

    Returns {game_pk: abstractGameCode} where codes are:
        P = Preview (not started), L = Live, F = Final
    """
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
        timeout=15,
    ).read())
    statuses = {}
    for d in resp.get("dates", []):
        for g in d.get("games", []):
            statuses[g["gamePk"]] = g["status"]["abstractGameCode"]
    return statuses


def get_game_statuses_detailed(date: str) -> dict[int, dict[str, str]]:
    """Get detailed game statuses for all games on a date.

    Returns {game_pk: {"abstract": code, "detailed": state}} where:
        abstract: P = Preview, L = Live, F = Final
        detailed: e.g. "Suspended", "Delayed Start", "Final", "In Progress"
    """
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
        timeout=15,
    ).read())
    statuses = {}
    for d in resp.get("dates", []):
        for g in d.get("games", []):
            statuses[g["gamePk"]] = {
                "abstract": g["status"]["abstractGameCode"],
                "detailed": g["status"].get("detailedState", ""),
            }
    return statuses


_STALE_DETAILED_STATES = {"Postponed", "Cancelled", "Canceled"}
_VOID_DETAILED_STATES = {state.lower() for state in _STALE_DETAILED_STATES}


def _is_void_detailed_state(detailed: str | None) -> bool:
    return (detailed or "").strip().lower() in _VOID_DETAILED_STATES


def _classify_unposted_game_status(
    status: dict[str, str] | None,
    *,
    game_pk: int | None = None,
) -> PickLockState:
    """Classify one unposted committed game status for lock/candidate logic."""
    if status is None:
        return PickLockState(
            stale=True,
            locked=False,
            reason="missing_from_schedule",
            game_pk=game_pk,
        )

    detailed = status.get("detailed", "")
    abstract = status.get("abstract")
    if _is_void_detailed_state(detailed):
        return PickLockState(
            stale=True,
            locked=False,
            reason="stale_game_status",
            game_pk=game_pk,
            abstract=abstract,
            detailed=detailed,
        )

    if abstract != "P":
        return PickLockState(
            locked=True,
            reason="game_started_or_final",
            game_pk=game_pk,
            abstract=abstract,
            detailed=detailed,
        )

    return PickLockState(
        locked=False,
        reason="all_preview",
        game_pk=game_pk,
        abstract=abstract,
        detailed=detailed,
    )


def pick_candidate_status_is_available(status: dict[str, str] | None) -> bool:
    """Return whether a prediction row's game is eligible for a fresh pick."""
    lock_state = _classify_unposted_game_status(status)
    return not lock_state.stale and not lock_state.locked


def iter_daily_pick_slots(daily: DailyPick) -> list[tuple[str, Pick]]:
    """Return score-bearing pick slots in stable contest order."""
    slots = [("pick", daily.pick)]
    if daily.double_down:
        slots.append(("double_down", daily.double_down))
    return slots


def _slot_is_voided(pick: Pick, detailed_statuses: dict[int, dict[str, str]]) -> bool:
    status = detailed_statuses.get(pick.game_pk)
    return bool(status and _is_void_detailed_state(status.get("detailed")))


def resolve_pick_slot_result(
    pick: Pick,
    date: str,
    detailed_statuses: dict[int, dict[str, str]] | None = None,
) -> str | None:
    """Resolve one locked BTS slot as "hit", "miss", "void", or pending.

    A postponed/cancelled locked game is void for that slot only. It does not
    count as a miss and does not wait for the future makeup game.
    """
    if detailed_statuses is None:
        try:
            detailed_statuses = get_game_statuses_detailed(date)
        except Exception:
            detailed_statuses = {}

    if _slot_is_voided(pick, detailed_statuses):
        return "void"

    result = check_hit(
        pick.game_pk,
        pick.batter_id,
        batter_name=pick.batter_name,
        date=date,
        team=pick.team,
        return_status=True,
    )
    if result is None:
        return None
    # check_hit(return_status=True) yields a status string ("hit"/"miss"/"void"); tolerate
    # a bool so callers/test doubles on the legacy True/False contract still map correctly.
    if isinstance(result, bool):
        return "hit" if result else "miss"
    return result  # "hit" | "miss" | "void" (suspended game with no pre-suspension PA)


def resolve_daily_slot_results(daily: DailyPick, date: str) -> dict[str, str] | None:
    """Resolve all score-bearing slots, or return None if any active slot is pending."""
    try:
        detailed_statuses = get_game_statuses_detailed(date)
    except Exception:
        detailed_statuses = {}

    slot_results: dict[str, str] = {}
    for slot_key, pick in iter_daily_pick_slots(daily):
        result = resolve_pick_slot_result(pick, date, detailed_statuses=detailed_statuses)
        if result is None:
            return None
        slot_results[slot_key] = result
    return slot_results


def active_streak_results(slot_results: dict[str, str]) -> list[bool]:
    """Convert resolved slot results into the non-void booleans used for streaks."""
    return [result == "hit" for result in slot_results.values() if result != "void"]


def effective_daily_result(slot_results: dict[str, str]) -> str:
    """Return the day-level BTS result after removing voided slots."""
    active_results = active_streak_results(slot_results)
    if not active_results:
        return "void"
    return "hit" if all(active_results) else "miss"


def streak_increment_for_resolved_hit(daily: DailyPick) -> int:
    """Return how many streak days a resolved hit should add.

    Legacy double-down hit files lack slot_results and still count as +2.
    Partial-void files with slot_results count only non-void hit slots.
    """
    if daily.result != "hit":
        return 0
    if daily.slot_results is not None:
        return sum(1 for result in daily.slot_results.values() if result == "hit")
    return 2 if daily.double_down else 1


def _committed_pick_game_pks(daily: DailyPick) -> list[int]:
    game_pks = [daily.pick.game_pk]
    if daily.double_down and daily.double_down.game_pk not in game_pks:
        game_pks.append(daily.double_down.game_pk)
    return game_pks


def classify_pick_lock_state(daily: DailyPick, date: str) -> PickLockState:
    """Classify an existing pick as locked, stale, or refreshable.

    Delivered picks are always locked. Undelivered picks become stale when
    any committed pick game is missing from today's schedule or is explicitly
    postponed/cancelled. Status lookup failures fail closed to avoid duplicate
    or incorrect public/private pick delivery.
    """
    if pick_was_delivered(daily):
        reason = "bluesky_posted" if daily.bluesky_posted else "notification_sent"
        return PickLockState(locked=True, reason=reason)

    game_pks = _committed_pick_game_pks(daily)
    try:
        detailed_statuses = get_game_statuses_detailed(date)
    except Exception:
        try:
            abstract_statuses = get_game_statuses(date)
        except Exception:
            return PickLockState(locked=True, reason="status_lookup_failed")
        for game_pk in game_pks:
            abstract = abstract_statuses.get(game_pk)
            if abstract != "P":
                return PickLockState(
                    locked=True,
                    reason="fallback_status_locked",
                    game_pk=game_pk,
                    abstract=abstract,
                )
        return PickLockState(locked=False, reason="fallback_all_preview")

    game_states = [
        _classify_unposted_game_status(detailed_statuses.get(game_pk), game_pk=game_pk)
        for game_pk in game_pks
    ]

    for game_state in game_states:
        if game_state.stale:
            return game_state

    for game_state in game_states:
        if game_state.locked:
            return game_state

    return PickLockState(locked=False, reason="all_preview")


def _parse_feed_timestamp(value: str | None):
    """Parse an MLB feed ISO-8601 UTC timestamp (e.g. '2026-06-17T18:00:00Z')."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _play_matches_batter(play: dict, batter_id: int, batter_name: str | None = None) -> bool:
    batter = play.get("matchup", {}).get("batter", {})
    if batter.get("id") == batter_id:
        return True
    name = batter.get("fullName")
    return bool(batter_name and name and name.lower() == str(batter_name).lower())


def _grade_pick_pre_suspension(
    resp: dict, batter_id: int, batter_name: str | None = None
) -> str | None:
    """Grade a pick in a SUSPENDED-and-resumed game from pre-suspension PA only.

    Per BTS rules the resumed/rescheduled portion of a suspended game is never evaluated,
    so the cumulative final boxscore must not be used. Returns:
      "hit"  - a hit recorded before suspension,
      "miss" - >=1 pre-suspension PA but no pre-suspension hit,
      "void" - the batter appears only in the resumed portion (no evaluable PA),
      None   - not a suspended game, OR the batter is absent from this game's PA entirely
               (caller then grades from the boxscore / falls back to other games).
    """
    datetime_block = resp.get("gameData", {}).get("datetime", {})
    resume_dt = _parse_feed_timestamp(datetime_block.get("resumeDateTime"))
    if resume_dt is None:
        return None  # not a suspended game
    plays = resp.get("liveData", {}).get("plays", {}).get("allPlays", [])
    saw_batter = saw_pre_pa = saw_pre_hit = False
    for play in plays:
        event_type = play.get("result", {}).get("eventType")
        if event_type not in PA_ENDING_EVENTS:
            continue
        if not _play_matches_batter(play, batter_id, batter_name):
            continue
        saw_batter = True
        start_dt = _parse_feed_timestamp(play.get("about", {}).get("startTime"))
        if start_dt is None or start_dt >= resume_dt:
            continue  # resumed portion (or unknown time) -> never evaluated
        saw_pre_pa = True
        if event_type in HIT_EVENTS:
            saw_pre_hit = True
    if not saw_batter:
        return None  # batter not in this game -> let caller fall back to other games
    if not saw_pre_pa:
        return "void"  # only in the resumed portion -> no evaluable original-day PA
    return "hit" if saw_pre_hit else "miss"


def _boxscore_hit(resp: dict, batter_id: int, batter_name: str | None = None) -> bool | None:
    """Whether a batter got a hit per the (cumulative) final boxscore. Correct for normal
    games; suspended games are graded by _grade_pick_pre_suspension instead.

    Looks up by ID first, falls back to name match. Returns True/False/None (not found).
    """
    for side in ("away", "home"):
        players = resp["liveData"]["boxscore"]["teams"][side]["players"]
        # Try by ID first
        key = f"ID{batter_id}"
        if key in players:
            hits = players[key].get("stats", {}).get("batting", {}).get("hits", 0)
            return hits > 0
        # Fallback: search by name
        if batter_name:
            for pid, pdata in players.items():
                if pdata["person"]["fullName"].lower() == batter_name.lower():
                    hits = pdata.get("stats", {}).get("batting", {}).get("hits", 0)
                    return hits > 0
    return None


def _check_hit_in_game(resp: dict, batter_id: int, batter_name: str | None = None) -> bool | None:
    """Suspension-aware hit check. A suspended-resumed game is graded from pre-suspension
    PA (a resumed-portion hit does NOT count); otherwise the final boxscore is used.
    Returns True (hit), False (no hit, incl. unevaluable resumed-only PA), None (not found).
    """
    grade = _grade_pick_pre_suspension(resp, batter_id, batter_name)
    if grade is not None:
        return grade == "hit"
    return _boxscore_hit(resp, batter_id, batter_name)


def grade_pick_in_feed(resp: dict, batter_id: int, batter_name: str | None = None) -> str | None:
    """Grade a pick from one game-feed response: "hit" / "miss" / "void" / None.

    Suspended-and-resumed games are graded from pre-suspension PA only (void = the batter
    has no evaluable pre-suspension PA); normal games from the final boxscore. None means
    the batter was not found in this feed (caller may fall back). This is the
    status-returning sibling of _check_hit_in_game, which collapses void/miss to a bool --
    use this wherever the void/miss distinction must survive (streak, shadow backfill).
    """
    grade = _grade_pick_pre_suspension(resp, batter_id, batter_name)
    if grade is not None:
        return grade
    hit = _boxscore_hit(resp, batter_id, batter_name)
    return None if hit is None else ("hit" if hit else "miss")


def check_hit(game_pk: int | None, batter_id: int, batter_name: str | None = None,
              date: str | None = None, team: str | None = None,
              *, return_status: bool = False):
    """Check if a batter got a hit in a game.

    Default: returns True (hit), False (no hit), or None (game not final OR batter not
    found). With return_status=True returns "hit" / "miss" / "void" / None, where "void"
    is a suspended game in which the batter has no pre-suspension (evaluable) PA.

    Suspended-and-resumed games are graded from pre-suspension PA only (the resumed
    portion is never evaluated for BTS); normal games use the final boxscore. If game_pk
    is None or the batter is not found, falls back to searching all Final games on date.
    """
    def _emit(grade):
        return grade if return_status else (grade == "hit")

    if game_pk is not None:
        resp = json.loads(retry_urlopen(
            f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        ).read())
        if resp["gameData"]["status"]["abstractGameCode"] != "F":
            return None
        grade = grade_pick_in_feed(resp, batter_id, batter_name)
        if grade is not None:
            return _emit(grade)

    # Batter not found (or no game_pk) — try every Final game on that date
    if date:
        sched = json.loads(retry_urlopen(
            f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
            timeout=15,
        ).read())
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                if g["gamePk"] == game_pk:
                    continue  # Already tried this one
                if g["status"]["abstractGameCode"] != "F":
                    continue
                alt_resp = json.loads(retry_urlopen(
                    f"{API_BASE}/api/v1.1/game/{g['gamePk']}/feed/live",
                    timeout=15,
                ).read())
                grade = grade_pick_in_feed(alt_resp, batter_id, batter_name)
                if grade is not None:
                    return _emit(grade)

    return None


def save_pick_shadow(pick_data, shadow_dir, source: str) -> Path:
    """Save a pick record to the shadow directory (not authoritative).

    Shadow dirs are used during Phase 2 of the cloud migration to
    compare Fly's output against Pi5's real state without affecting
    production. source is 'fly' or 'pi5' to distinguish writers.
    """
    shadow_dir = Path(shadow_dir)
    date = pick_data["date"] if isinstance(pick_data, dict) else pick_data.date
    date_dir = shadow_dir / date
    date_dir.mkdir(parents=True, exist_ok=True)

    out_path = date_dir / f"{source}.json"
    if isinstance(pick_data, dict):
        payload = pick_data
    else:
        payload = pick_data.__dict__ if hasattr(pick_data, "__dict__") else dict(pick_data)
    atomic_write_text(out_path, json.dumps(payload, indent=2, default=str))
    return out_path


def reconcile_results(
    picks_dir: Path,
    lookback_days: int = 8,
) -> list[dict]:
    """Re-check recent picks against current boxscore data.

    Catches scoring changes (hit -> error) that happened after the original
    check-results. Returns list of corrections made.
    """
    from datetime import date as date_cls, timedelta as td
    today = date_cls.today()
    corrections = []

    for i in range(1, lookback_days + 1):
        d = (today - td(days=i)).isoformat()
        daily = load_pick(d, picks_dir)
        if not daily or daily.result not in ("hit", "miss", "void"):
            continue

        slot_results = resolve_daily_slot_results(daily, d)
        if slot_results is None:
            continue

        current_result = effective_daily_result(slot_results)
        if current_result != daily.result:
            corrections.append({
                "date": d,
                "batter": daily.pick.batter_name,
                "old_result": daily.result,
                "new_result": current_result,
            })
            daily.result = current_result
            daily.slot_results = slot_results
            save_pick(daily, picks_dir)
        elif slot_results != daily.slot_results:
            daily.slot_results = slot_results
            save_pick(daily, picks_dir)

    # Recompute the streak by FORWARD replay over the season — catches result
    # corrections AND streak-increment bugs, and (unlike the old backward walk)
    # correctly replays the saver: forgiveness depends on the forward streak at
    # the miss and whether the saver was already used, which a backward suffix
    # can't reconstruct (see _replay_season_streak / audit D4). On incomplete or
    # unreadable history the replay returns None and we keep the live-tracked
    # streak.json rather than risk a mis-count (fail closed).
    today_iso = date_cls.today().isoformat()
    replay = _replay_season_streak(picks_dir, today.year, today_iso)
    if replay is not None:
        streak, saver = replay
        save_streak(streak, picks_dir, saver_available=saver)

    return corrections
