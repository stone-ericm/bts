"""Pick persistence, streak tracking, and MLB API helpers for BTS automation."""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

from bts.util import retry_urlopen

API_BASE = "https://statsapi.mlb.com"


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


def compute_provenance(
    blend_path: Path | str | None = None,
    policy_path: Path | str | None = None,
    cwd: Path | str = ".",
) -> dict[str, str | None]:
    """Bundle provenance fields for a DailyPick.

    Returns a dict with keys ``model_git_sha``, ``model_pickle_sha256``,
    ``policy_npz_sha256``. Each value is either a hex string or None.
    None values reflect "the artifact is genuinely unavailable" or "the
    git/hash call failed" — they MUST NOT cause callers to error out
    (per Codex #168).

    ``blend_path`` is the path of the cached blend artifact written by
    ``bts.model.predict.run_pipeline``; the field name on DailyPick
    follows the existing on-disk convention (``model_pickle_sha256``).
    """
    return {
        "model_git_sha": _git_head_sha(cwd),
        "model_pickle_sha256": _sha256_file(blend_path),
        "policy_npz_sha256": _sha256_file(policy_path),
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
    result: str | None = None  # "hit", "miss", "suspended", "unresolved", or None (pending)
    # Provenance v1 (added 2026-05-04, per Codex bus #168). All optional;
    # old picks lack these fields and load_pick backfills via .get(...).
    # See bts.picks.compute_provenance for the helper that populates them.
    model_git_sha: str | None = None  # git rev-parse HEAD at predict/save time
    model_pickle_sha256: str | None = None  # sha256 of blend artifact actually used
    policy_npz_sha256: str | None = None  # sha256 of mdp_policy.npz if loaded


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
        pitcher_id=int(row["pitcher_id"]) if row.get("pitcher_id") else None,
        p_game_hit=float(row["p_game_hit"]),
        flags=flags,
        projected_lineup="PROJECTED" in flags_str,
        game_pk=int(row["game_pk"]),
        game_time=row["game_time"],
        pitcher_team=row.get("pitcher_team"),
    )


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
    path.write_text(json.dumps(asdict(daily), indent=2))
    try:
        append_lineup_evolution(daily, picks_dir)
    except Exception:
        pass
    return path


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
    path.write_text(json.dumps(asdict(daily), indent=2))
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
    pick = Pick(**data["pick"])
    dd = Pick(**data["double_down"]) if data.get("double_down") else None
    return DailyPick(
        date=data["date"], run_time=data["run_time"], pick=pick,
        double_down=dd, runner_up=data.get("runner_up"),
        bluesky_posted=data.get("bluesky_posted", False),
        bluesky_uri=data.get("bluesky_uri"),
        result=data.get("result"),
        model_git_sha=data.get("model_git_sha"),
        model_pickle_sha256=data.get("model_pickle_sha256"),
        policy_npz_sha256=data.get("policy_npz_sha256"),
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
        result=data.get("result"),
        # Provenance v1 — defaults to None for picks saved before these fields existed.
        model_git_sha=data.get("model_git_sha"),
        model_pickle_sha256=data.get("model_pickle_sha256"),
        policy_npz_sha256=data.get("policy_npz_sha256"),
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
    path.write_text(json.dumps({
        "streak": streak,
        "saver_available": saver_available if saver_available is not None else existing_saver,
        "updated": datetime.now(timezone.utc).isoformat(),
    }))


def update_streak(results: list[bool], picks_dir: Path) -> int:
    """Update streak based on pick results.

    Single pick: [True] -> +1, [False] -> 0
    Double-down: [True, True] -> +2, anything else -> 0

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


def _check_hit_in_game(resp: dict, batter_id: int, batter_name: str | None = None) -> bool | None:
    """Check if a batter got a hit in a game feed response.

    Looks up by ID first, falls back to name match if ID not found.
    Returns True (hit), False (no hit), or None (not found).
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


def check_hit(game_pk: int | None, batter_id: int, batter_name: str | None = None,
              date: str | None = None, team: str | None = None) -> bool | None:
    """Check if a batter got a hit in a game.

    Returns True (hit), False (no hit), or None (game not final OR batter
    not found in boxscore, e.g. scratched).

    If game_pk is None or batter not found, falls back to searching all
    Final games on the given date.
    """
    if game_pk is not None:
        resp = json.loads(retry_urlopen(
            f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        ).read())
        status = resp["gameData"]["status"]["abstractGameCode"]
        if status != "F":
            return None

        result = _check_hit_in_game(resp, batter_id, batter_name)
        if result is not None:
            return result

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
                result = _check_hit_in_game(alt_resp, batter_id, batter_name)
                if result is not None:
                    return result

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
    out_path.write_text(json.dumps(payload, indent=2, default=str))
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
        if not daily or daily.result not in ("hit", "miss"):
            continue

        # Re-check primary pick
        primary = check_hit(
            daily.pick.game_pk, daily.pick.batter_id,
            batter_name=daily.pick.batter_name,
            date=d, team=daily.pick.team,
        )
        if primary is None:
            continue

        results = [primary]
        if daily.double_down:
            double = check_hit(
                daily.double_down.game_pk, daily.double_down.batter_id,
                batter_name=daily.double_down.batter_name,
                date=d, team=daily.double_down.team,
            )
            if double is not None:
                results.append(double)

        current_result = "hit" if all(results) else "miss"
        if current_result != daily.result:
            corrections.append({
                "date": d,
                "batter": daily.pick.batter_name,
                "old_result": daily.result,
                "new_result": current_result,
            })
            daily.result = current_result
            save_pick(daily, picks_dir)

    # Always recalculate streak from scratch — catches both result corrections
    # and streak increment bugs (e.g., cross-game double-down counted as +1).
    # Only iterate strict YYYY-MM-DD files so streak.json, automation.json, and
    # *.shadow.json are all skipped automatically. Also skip any pick file dated
    # today or later: `bts preview` pre-generates tomorrow's pick before games
    # are played, so its result=None doesn't mean "miss" — it means "not played
    # yet" and we must not let it break the backward walk.
    today_iso = date_cls.today().isoformat()
    streak = 0
    dates = sorted(picks_dir.glob("*.json"))
    for f in reversed(dates):
        if not _ISO_DATE_RE.match(f.stem):
            continue
        if f.stem >= today_iso:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        r = data.get("result")
        if r == "hit":
            dd = data.get("double_down")
            streak += 2 if dd else 1
        elif r == "miss":
            break
        else:
            break
    save_streak(streak, picks_dir)

    return corrections
