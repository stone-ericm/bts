"""Dynamic lineup scheduler for BTS.

Replaces fixed cron runs with game-time-aware lineup checks.
Checks lineups 45 min before each game, clusters nearby checks,
and commits picks only when confirmed lineup + gap threshold met.
"""

import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.util import retry_urlopen
from bts.picks import API_BASE
from bts.heartbeat import write_heartbeat, HeartbeatState, heartbeat_watchdog
from bts.sd_notify import notify_ready, notify_watchdog
from bts.orchestrator import predict_local_shadow, run_and_pick
from bts.picks import save_shadow_pick
from bts.strategy import select_pick

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _optional_health_path(value) -> Path | None:
    """Resolve optional health-check paths with a simple disable escape hatch."""
    if value is False or value is None or value == "":
        return None
    return Path(value)


def _contest_state_required(config: dict) -> bool:
    sched_config = config.get("scheduler", {})
    health_config = config.get("health_checks", {})
    return bool(
        sched_config.get("contest_state_required", False)
        or health_config.get("contest_state_expected", False)
    )


def _alert_contest_state_failure(config: dict, error: Exception) -> None:
    """Best-effort immediate alert for fail-closed contest-state decisions."""
    from bts.health.alert import Alert, dispatch_dm_for_health_alerts

    dm_recipient = config.get("bluesky", {}).get("dm_recipient")
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    status_path = (
        picks_dir.parent / "health_state" / "health_dm_delivery_status.json"
    )
    dispatch_dm_for_health_alerts(
        [Alert("CRITICAL", "contest_state", str(error))],
        dm_recipient,
        status_path=status_path,
    )


def _alert_missed_pick(config: dict, daily, mins_to_game: float) -> None:
    """E3: DM the operator that no pick was delivered as first pitch nears, so
    they can post manually (the EOD post_failure DM is hours too late)."""
    from bts.health.alert import Alert, dispatch_dm_for_health_alerts

    dm_recipient = config.get("bluesky", {}).get("dm_recipient")
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    status_path = picks_dir.parent / "health_state" / "health_dm_delivery_status.json"
    name = daily.pick.batter_name if daily and getattr(daily, "pick", None) else "?"
    msg = (f"NO PICK DELIVERED with ~{mins_to_game:.0f} min to first pitch "
           f"(top pick: {name}). Deliver manually if you want to keep the streak alive.")
    dispatch_dm_for_health_alerts(
        [Alert("CRITICAL", "missed_pick", msg)],
        dm_recipient,
        status_path=status_path,
    )


def _maybe_alert_missed_pick(
    config: dict, date: str, picks_dir: Path, missed_pick_alert_min: int,
    heartbeat_path: "Path | None",
) -> None:
    """E3: if no pick is delivered by ``missed_pick_alert_min`` minutes before the
    earliest first pitch, DM a one-shot warning. Waits (watchdog-fed) to the alert
    window, then re-checks — a late delivery during the wait suppresses the alert."""
    from bts.picks import load_pick, pick_was_delivered

    daily = load_pick(date, picks_dir)
    if not daily or pick_was_delivered(daily):
        return
    earliest_game_et = _earliest_pick_game_et(daily)
    alert_at = earliest_game_et - timedelta(minutes=missed_pick_alert_min)
    wait_s = (alert_at - _now_et()).total_seconds()
    if wait_s > 0:
        if heartbeat_path:
            write_heartbeat(
                heartbeat_path, state=HeartbeatState.SLEEPING,
                sleeping_until=alert_at.astimezone(UTC),
            )
            notify_watchdog()
        _watchdog_ping_sleep(wait_s)
        if heartbeat_path:
            write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
            notify_watchdog()
    daily = load_pick(date, picks_dir)  # a late delivery may have landed during the wait
    if daily and not pick_was_delivered(daily):
        mins = (earliest_game_et - _now_et()).total_seconds() / 60
        print(f"  MISSED-PICK ALERT — no delivery with ~{mins:.0f}min to first pitch.",
              file=sys.stderr)
        _alert_missed_pick(config, daily, mins)


def fetch_schedule(date: str) -> list[dict]:
    """Fetch today's MLB schedule. Returns list of game dicts."""
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher",
        timeout=15,
    ).read())
    games = []
    for d in resp.get("dates", []):
        games.extend(d.get("games", []))
    return games


def game_time_et(game: dict) -> datetime:
    """Extract game time as ET datetime."""
    utc = datetime.fromisoformat(game["gameDate"].replace("Z", "+00:00"))
    return utc.astimezone(ET)


def compute_run_times(
    games: list[dict],
    offset_min: int = 45,
    cluster_min: int = 10,
) -> list[dict]:
    """Compute clustered lineup check times from game schedule.

    For each game, the check time is game_time - offset_min.
    Checks within cluster_min of each other are merged into one run.

    Returns list of {"time_et": datetime, "game_pks": [int, ...]}
    sorted by time.
    """
    if not games:
        return []

    checks = []
    for g in games:
        et = game_time_et(g)
        check_time = et - timedelta(minutes=offset_min)
        checks.append({"time_et": check_time, "game_pk": g["gamePk"]})

    checks.sort(key=lambda c: c["time_et"])

    clusters = []
    current = {"time_et": checks[0]["time_et"], "game_pks": [checks[0]["game_pk"]]}

    for c in checks[1:]:
        if (c["time_et"] - current["time_et"]) <= timedelta(minutes=cluster_min):
            current["game_pks"].append(c["game_pk"])
        else:
            clusters.append(current)
            current = {"time_et": c["time_et"], "game_pks": [c["game_pk"]]}

    clusters.append(current)
    return clusters


def detect_doubleheader_game2s(games: list[dict]) -> set[int]:
    """Detect game 2 of doubleheaders (fluid start time).

    Returns set of game_pks that are doubleheader game 2s.
    Detected by finding two games with the same away+home team pair.
    """
    team_games = {}
    for g in games:
        away = g["teams"]["away"]["team"]["name"]
        home = g["teams"]["home"]["team"]["name"]
        key = (away, home)
        team_games.setdefault(key, []).append(g)

    game2s = set()
    for key, team_g in team_games.items():
        if len(team_g) >= 2:
            team_g.sort(key=lambda x: game_time_et(x))
            for g in team_g[1:]:
                game2s.add(g["gamePk"])

    return game2s


def compute_wakeup_time(
    games: list[dict],
    default_hour_et: int = 10,
    early_buffer_min: int = 60,
) -> datetime:
    """Compute scheduler wake-up time based on earliest game.

    If any game starts before the default init hour, wakes up
    early_buffer_min before the earliest game.
    """
    if not games:
        return datetime.now(ET).replace(hour=default_hour_et, minute=0, second=0, microsecond=0)

    earliest = min(game_time_et(g) for g in games)
    default_wakeup = earliest.replace(hour=default_hour_et, minute=0, second=0, microsecond=0)
    early_wake = earliest - timedelta(minutes=early_buffer_min)

    if early_wake < default_wakeup:
        return early_wake

    return default_wakeup


def _next_day_wakeup(date: str, sched_config: dict) -> datetime:
    """A FUTURE wake time for the day after ``date``, used to idle through
    off-days instead of thrashing systemd Restart=always (audit E1).

    Uses tomorrow's earliest game if any, else the default morning hour. Always
    returns a time strictly in the future — on a multi-day break compute_wakeup_time
    would return today's hour (already past), so we bump to tomorrow morning.
    """
    default_hour = sched_config.get("default_init_hour_et", 10)
    tomorrow = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        wakeup = compute_wakeup_time(
            fetch_schedule(tomorrow),
            default_hour_et=default_hour,
            early_buffer_min=sched_config.get("early_game_buffer_min", 60),
        )
    except Exception as e:
        print(f"  Failed to fetch tomorrow's schedule: {e}", file=sys.stderr)
        wakeup = _now_et().replace(hour=default_hour, minute=0, second=0, microsecond=0)
    now = _now_et()
    if wakeup <= now:
        wakeup = (now + timedelta(days=1)).replace(
            hour=default_hour, minute=0, second=0, microsecond=0
        )
    return wakeup


def resolve_fallback_deadline_min(
    earliest_game_et: datetime,
    standard_min: int = 35,
    morning_min: int = 25,
    morning_cutoff_hour: int = 11,
) -> int:
    """Return fallback_deadline_min adjusted for morning games.

    For games with first pitch strictly before morning_cutoff_hour (in ET),
    use morning_min instead of standard_min. This gives morning games
    (London Series, July 4 morning starts) more lineup-wait tolerance
    before force-picking with projected lineups.

    Normal-time games (first pitch at or after morning_cutoff_hour) keep
    the standard buffer unchanged.

    Note: Caller is responsible for passing an ET-localized datetime;
    the function reads `.hour` directly without timezone conversion.
    """
    if morning_min > standard_min:
        raise ValueError(
            f"morning_min ({morning_min}) must be <= standard_min ({standard_min}); "
            "morning buffer should be shorter, not longer"
        )
    if earliest_game_et.hour < morning_cutoff_hour:
        return morning_min
    return standard_min


def check_confirmed_lineups(game_pks: list[int]) -> dict[int, set[str]]:
    """Check which teams in which games have confirmed lineups posted.

    A team's lineup is confirmed when any of its players have `battingOrder`
    set in the boxscore. Returns `{game_pk: {confirmed_sides}}` where the
    value is a subset of `{"home", "away"}` (empty set = no lineups yet).

    Team-level tracking matters: the prediction pipeline uses per-side data,
    so a game that flips from one confirmed side to two still represents
    new information even though the game was already "seen" at game level.
    """
    results: dict[int, set[str]] = {}
    for pk in game_pks:
        confirmed_sides: set[str] = set()
        try:
            resp = json.loads(retry_urlopen(
                f"{API_BASE}/api/v1.1/game/{pk}/feed/live",
                timeout=15,
            ).read())
            for side in ("away", "home"):
                players = resp["liveData"]["boxscore"]["teams"][side]["players"]
                for pid, pdata in players.items():
                    if pdata.get("battingOrder"):
                        confirmed_sides.add(side)
                        break
        except Exception:
            pass
        results[pk] = confirmed_sides

    return results


@dataclass
class SchedulerState:
    """Daily scheduler state, persisted to JSON."""
    date: str
    schedule_fetched_at: str
    games: list[dict]  # [{game_pk, game_time_et, lineup_confirmed, is_doubleheader_game2}]
    confirmed_game_pks: list[int]
    runs_completed: list[dict]  # [{time, new_lineups, skipped, pick_name, pick_p}]
    pick_locked: bool
    pick_locked_at: str | None
    result_status: str | None  # "final", "suspended", "unresolved", None
    next_wakeup: str | None  # ISO for next day's wake-up
    analytics_jobs: dict | None = None  # shadow/capture attempt status by job name
    skip_summary: dict | None = None    # {best_batter, best_team, best_p, streak} on a skip day
    skip_notified_at: str | None = None  # ISO; set once the skip notice is delivered (idempotency)
    # Finalization state (formerly the run-local FinalizationState). Persisted to
    # scheduler_state.json + carried forward by carry_forward_skip_state so a
    # same-day daemon restart (deploys; Restart=always) after an MDP skip but
    # before end-of-day no longer loses the captured skip (#3 / GH #144).
    # Defaulted so an old scheduler_state.json still deserializes via
    # SchedulerState(**data); asdict round-trips the dict cleanly.
    final_skip_candidate: dict | None = None  # declined MDP-skip candidate to record at EOD
    committed_pick_written: bool = False       # True once a scoreable commit record was written


@dataclass
class FallbackRefreshResult:
    daily: object
    should_post: bool | None
    # The SelectionResult behind the refreshed pick (Task 3). None on the
    # cached / no-refresh paths; the fresh-pick paths carry the real selection
    # so the fallback delivery can record source + candidate metadata.
    selection: "SelectionResult | None" = None


def _row_from_daily(pick) -> dict | None:
    """Map a Pick (or None) to the candidate dict shape ``_row_to_candidate``
    produces, so commit/classification records match the skip-path records."""
    if pick is None:
        return None
    return {"batter_id": pick.batter_id, "batter_name": pick.batter_name,
            "team": pick.team, "game_pk": pick.game_pk,
            "p_game_hit": pick.p_game_hit}


def _write_commit_decision(picks_dir, date, *, action, source, primary, double_down, delivery_status, state):
    """Record an authoritative, scoreable decision at a real commit/lock point.

    Sets ``state.committed_pick_written`` and persists state AFTER the decision
    write (save-ordering fix #2a): the flag now lives on the persisted state, so
    a restart between the decision write and the next save must not lose it. The
    on-disk decision.json is written first, then the flag, then save_state — so
    every ordering leaves the EOD skip non-clobbering (the decision.json record
    backstops the flag even if this save fails).
    """
    from bts.daily_decision import write_decision
    write_decision(date, picks_dir, action=action, source=(source or "unknown"),
                   primary=primary, double_down=double_down,
                   delivery_status=delivery_status, scoreable=True)
    state.committed_pick_written = True
    save_state(state, picks_dir)


def _write_classification_decision(picks_dir, date, *, action, delivered, primary, double_down, state):
    """Record a classification-lock only when the existing pick was genuinely delivered.

    A genuinely DELIVERED existing pick recovered via classification-lock -> scoreable.
    A non-delivered classification-lock (stale preview locked by game-start/status) -> nothing,
    so the earlier MDP-skip record can still be written at end-of-day (the GH #144 case).
    """
    if not delivered:
        return
    _write_commit_decision(picks_dir, date, action=action, source="unknown",
                           primary=primary, double_down=double_down, delivery_status="delivered", state=state)


def _write_endofday_skip(picks_dir, date, state):
    """Record the day's MDP skip — only if no pick was committed and a candidate was captured.

    Overwrite-guard (#2b): the authoritative "committed today?" is the on-disk
    decision.json, not just the in-memory flag. If a crash landed between a
    commit's decision write and its state save, ``committed_pick_written`` may be
    stale (False) on the rebuilt state while a real scoreable record already
    exists on disk — so never clobber a scoreable commit with an EOD skip.
    """
    from bts.daily_decision import load_decision, write_decision
    if state.committed_pick_written or not state.final_skip_candidate:
        return
    existing = load_decision(date, picks_dir)
    if existing is not None and existing.get("scoreable"):
        return
    c = state.final_skip_candidate
    write_decision(date, picks_dir, action="skip", source="mdp", primary=c.get("primary"),
                   streak=c.get("streak"), saver_available=c.get("saver_available"),
                   delivery_status="not_applicable", scoreable=False)


def save_state(state: SchedulerState, picks_dir: Path) -> Path:
    """Save scheduler state to JSON."""
    date_dir = picks_dir / state.date
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / "scheduler_state.json"
    # Analytics job helpers update disk directly while run_day holds an older
    # in-memory SchedulerState, so preserve those status writes on later saves.
    if path.exists():
        try:
            prior = json.loads(path.read_text())
            prior_jobs = prior.get("analytics_jobs")
            if isinstance(prior_jobs, dict):
                merged_jobs = dict(prior_jobs)
                merged_jobs.update(state.analytics_jobs or {})
                state.analytics_jobs = merged_jobs
        except Exception:
            pass
    path.write_text(json.dumps(asdict(state), indent=2))
    return path


def load_state(date: str, picks_dir: Path) -> SchedulerState | None:
    """Load scheduler state from JSON. Returns None if not found."""
    path = picks_dir / date / "scheduler_state.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return SchedulerState(**data)


def _update_analytics_job_status(
    config: dict,
    date: str,
    job: str,
    status: str,
    **extra,
) -> None:
    """Best-effort status write for observational analytics jobs."""
    try:
        picks_dir = Path(config["orchestrator"]["picks_dir"])
        state = load_state(date, picks_dir)
        if state is None:
            return
        jobs = dict(state.analytics_jobs or {})
        payload = {
            "status": status,
            "updated_at": _now_et().isoformat(),
        }
        payload.update({k: v for k, v in extra.items() if v is not None})
        jobs[job] = payload
        state.analytics_jobs = jobs
        save_state(state, picks_dir)
    except Exception as exc:
        print(
            f"  analytics job status update failed for {job}: {exc}",
            file=sys.stderr,
        )


def _analytics_job_status(config: dict, date: str, job: str) -> dict:
    try:
        picks_dir = Path(config["orchestrator"]["picks_dir"])
        state = load_state(date, picks_dir)
    except Exception:
        return {}
    if state is None or not isinstance(state.analytics_jobs, dict):
        return {}
    status = state.analytics_jobs.get(job)
    return status if isinstance(status, dict) else {}


def _pick_delivery_mode(config: dict) -> str:
    """Resolve how the scheduler should deliver a locked pick.

    ``pick_delivery`` is the new explicit control. ``posting_mode`` is accepted
    as a plain-English alias, and legacy ``private_mode`` still disables public
    feed posting when no explicit mode is configured.
    """
    sched_config = config.get("scheduler", {})
    raw = sched_config.get("pick_delivery", sched_config.get("posting_mode"))
    if raw is None:
        return "private" if sched_config.get("private_mode", False) else "public"

    mode = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "bluesky": "public",
        "feed": "public",
        "post": "public",
        "bluesky_dm": "dm",
        "direct_message": "dm",
        "none": "private",
        "off": "private",
        "local": "private",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"public", "dm", "private"}:
        raise ValueError(
            "scheduler.pick_delivery must be one of: public, dm, private"
        )
    return mode


def _format_pick_delivery_text(daily, streak: int) -> str:
    from bts.posting import format_post

    return format_post(
        daily.pick.batter_name, daily.pick.team,
        daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
        daily.double_down.batter_name if daily.double_down else None,
        daily.double_down.p_game_hit if daily.double_down else None,
        daily.double_down.team if daily.double_down else None,
        daily.double_down.pitcher_name if daily.double_down else None,
    )


def _deliver_and_lock_pick(
    daily,
    config: dict,
    picks_dir: Path,
    state: SchedulerState,
    date: str,
    label: str,
    *,
    selection: "SelectionResult | None" = None,
) -> bool:
    """Deliver a pick through the configured channel and persist the lock.

    Each branch that locks after a real delivery writes an authoritative,
    scoreable decision record onto ``state`` (the single finalization object;
    ``FinalizationState`` was dropped in favor of persisted SchedulerState
    fields). ``selection`` defaults to None so direct test callers keep building;
    when present its source/candidate metadata are recorded, else they fall back
    to the DailyPick.
    """
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts.picks import pick_was_delivered, save_pick

    mode = _pick_delivery_mode(config)

    def _record_commit(delivery_status: str) -> None:
        # Single chokepoint for the 5 lock branches — they differ only in
        # delivery_status. Source/candidate metadata come from the captured
        # SelectionResult when available, else fall back to the DailyPick.
        try:
            # Decision-record writes must never affect pick delivery: isolate
            # any failure here so it cannot be caught by the delivery try/except.
            _write_commit_decision(
                picks_dir, date,
                action=("double" if daily.double_down else "single"),
                source=(selection.source if selection is not None else "unknown"),
                primary=(selection.primary_candidate if selection is not None
                         else _row_from_daily(daily.pick)),
                double_down=(selection.double_candidate if selection is not None
                             else _row_from_daily(daily.double_down)),
                delivery_status=delivery_status, state=state,
            )
        except Exception:
            pass

    if pick_was_delivered(daily):
        save_pick(daily, picks_dir)
        state.pick_locked = True
        state.pick_locked_at = _now_et().isoformat()
        save_state(state, picks_dir)
        _trigger_live_forward_capture_on_lock(config, date)
        _record_commit("delivered")
        print(f"  LOCKED ({label}) — pick already delivered.", file=sys.stderr)
        return True

    if daily.delivery_attempted:
        # A delivery was persisted as "attempted" but the pick is not delivered →
        # the daemon crashed in the gap between sending and recording success. We
        # cannot know whether it posted, so do NOT re-send (a duplicate on the
        # public feed is worse than a missed post, which the EOD post_failure check
        # surfaces). Lock to stop further attempts. (Caught send failures clear the
        # marker before returning, so this only fires on an uncaught crash.)
        print(f"  DELIVERY OUTCOME UNKNOWN ({label}) — prior attempt unconfirmed; "
              f"NOT re-sending to avoid a duplicate. Locking; verify manually.",
              file=sys.stderr)
        state.pick_locked = True
        state.pick_locked_at = _now_et().isoformat()
        save_state(state, picks_dir)
        _record_commit("locked_unconfirmed")
        return False

    if mode == "private":
        save_pick(daily, picks_dir)
        state.pick_locked = True
        state.pick_locked_at = _now_et().isoformat()
        save_state(state, picks_dir)
        _trigger_live_forward_capture_on_lock(config, date)
        print(
            f"  [PRIVATE] LOCKED ({label}) — {daily.pick.batter_name} "
            f"({daily.pick.team}) {daily.pick.p_game_hit:.1%} — NOT delivered",
            file=sys.stderr,
        )
        _record_commit("private_locked")
        return True

    try:
        decision_state = load_decision_streak_state(
            picks_dir,
            require_contest_state=_contest_state_required(config),
        )
    except ContestStateError as e:
        print(f"  CONTEST STATE ERROR — pick delivery blocked: {e}", file=sys.stderr)
        _alert_contest_state_failure(config, e)
        return False
    text = _format_pick_delivery_text(daily, decision_state.streak)

    if mode == "dm":
        recipient = config.get("bluesky", {}).get("dm_recipient")
        if not recipient:
            print("  Pick DM failed: bluesky.dm_recipient is not configured", file=sys.stderr)
            return False
        daily.delivery_attempted = True  # persist BEFORE the network call (E2 idempotency)
        save_pick(daily, picks_dir)
        try:
            from bts.dm import send_dm
            msg_id = send_dm(recipient, text)
            daily.notification_sent = True
            daily.notification_channel = "bluesky_dm"
            daily.notification_id = msg_id
            save_pick(daily, picks_dir)
            state.pick_locked = True
            state.pick_locked_at = _now_et().isoformat()
            save_state(state, picks_dir)
            _trigger_live_forward_capture_on_lock(config, date)
            _record_commit("delivered")
            print(f"  LOCKED ({label}) — Pick DM sent: {msg_id}", file=sys.stderr)
            return True
        except Exception as e:
            daily.delivery_attempted = False  # known failure → clear so a later cycle retries
            save_pick(daily, picks_dir)
            print(f"  Pick DM failed: {e}", file=sys.stderr)
            return False

    daily.delivery_attempted = True  # persist BEFORE the network call (E2 idempotency)
    save_pick(daily, picks_dir)
    try:
        from bts.posting import post_to_bluesky
        uri = post_to_bluesky(text)
        daily.bluesky_posted = True
        daily.bluesky_uri = uri
        save_pick(daily, picks_dir)
        state.pick_locked = True
        state.pick_locked_at = _now_et().isoformat()
        save_state(state, picks_dir)
        _trigger_live_forward_capture_on_lock(config, date)
        _record_commit("delivered")
        print(f"  LOCKED ({label}) — Posted to Bluesky: {uri}", file=sys.stderr)
        return True
    except Exception as e:
        daily.delivery_attempted = False  # known failure → clear so a later cycle retries
        save_pick(daily, picks_dir)
        print(f"  Bluesky post failed: {e}", file=sys.stderr)
        return False


def count_new_confirmations(
    game_pks: list[int],
    previously_confirmed: set[tuple[int, str]],
) -> int:
    """Check for new lineup confirmations since last check (team-level).

    `previously_confirmed` is a set of `(game_pk, side)` tuples. Updates it
    in place with any newly confirmed sides and returns the count of new
    entries. Game-level counting would hide the case where one team in a
    game is already confirmed but the other side only just posted — that
    still represents new information for the prediction pipeline, so we
    track sides independently.
    """
    statuses = check_confirmed_lineups(game_pks)
    new_count = 0
    for pk, sides in statuses.items():
        for side in sides:
            key = (pk, side)
            if key not in previously_confirmed:
                previously_confirmed.add(key)
                new_count += 1
    return new_count


def _now_et() -> datetime:
    """Current time in ET. Extracted for testability."""
    return datetime.now(ET)


def _earliest_pick_game_et(daily) -> datetime:
    """Return the earliest game start time among primary and double-down picks (ET).

    BTS-app deadline = first game in the slate to start, since the user has to
    submit BOTH picks before either game begins. The fallback-post deadline must
    therefore use this earlier time, NOT the primary pick's game time, in cases
    where the double-down's game starts first.
    """
    times = [datetime.fromisoformat(daily.pick.game_time).astimezone(ET)]
    if daily.double_down:
        times.append(
            datetime.fromisoformat(daily.double_down.game_time).astimezone(ET)
        )
    return min(times)


def _compute_result_poll_start(daily) -> datetime:
    """Return the ET datetime when result polling should start: 10 minutes
    after the earliest of primary or double-down game start.

    Uses `_earliest_pick_game_et` so a double-down game that starts before the
    primary doesn't get skipped — `run_result_polling` already handles both
    game_pks internally, but only if the scheduler wakes in time.
    """
    return _earliest_pick_game_et(daily) + timedelta(minutes=10)


def _watchdog_ping_sleep(seconds: float, interval_sec: float = 60) -> None:
    """Sleep `seconds` while pinging notify_watchdog every `interval_sec`.

    Does NOT touch the heartbeat file. Use during SLEEPING-state waits where
    the pre-sleep heartbeat already encodes state+sleeping_until and the
    external check_heartbeat monitor relies on that metadata.

    Systemd's WatchdogSec=1800 requires notify_watchdog() at least every
    30 min; any time.sleep(>1800) in a SLEEPING state will SIGABRT-kill the
    daemon without these pings. Observed live 2026-04-23 overnight during
    the idle_end_of_day → next-wake sleep (NRestarts=21 before discovery).
    """
    from threading import Event, Thread

    from bts.sd_notify import notify_watchdog

    stop = Event()

    def _pulse() -> None:
        while not stop.is_set():
            try:
                notify_watchdog()
            except Exception:
                pass
            stop.wait(interval_sec)

    thread = Thread(target=_pulse, daemon=True)
    thread.start()
    try:
        time.sleep(seconds)
    finally:
        stop.set()
        thread.join(timeout=2)


def _idle_until_next_wakeup(
    next_wakeup_iso: str | None, heartbeat_path: Path | None
) -> None:
    """Sleep until ``next_wakeup_iso`` to prevent post-work Restart=always thrash.

    After the daily run_day reaches IDLE_END_OF_DAY, it must stay alive until
    the next day's scheduled wake. Without this sleep, run_day returns, the
    process exits, and systemd's Restart=always re-launches within 30s — then
    run_day cycles again in ~3 min because all its post-lock branches
    short-circuit (pick already locked, results already polled, games already
    final). Observed live 2026-04-23 post-games: 7 restarts in 25 min before
    discovery.

    No-op if ``next_wakeup_iso`` is None, malformed, tz-naive, or in the past.
    """
    if not next_wakeup_iso:
        return
    try:
        wakeup = datetime.fromisoformat(next_wakeup_iso)
    except (ValueError, TypeError):
        return
    if wakeup.tzinfo is None:
        return
    now = datetime.now(UTC)
    if wakeup <= now:
        return
    wait_secs = (wakeup - now).total_seconds()
    if heartbeat_path:
        write_heartbeat(
            heartbeat_path,
            state=HeartbeatState.SLEEPING,
            sleeping_until=wakeup.astimezone(UTC),
        )
        notify_watchdog()
    print(
        f"  Idle until tomorrow's wakeup "
        f"{wakeup.astimezone(ET).strftime('%H:%M ET')} "
        f"({wait_secs / 3600:.1f}h)...",
        file=sys.stderr,
    )
    _watchdog_ping_sleep(wait_secs)


def _poll_interval_sleep(
    heartbeat_path: Path | None,
    seconds: float,
    watchdog_interval_sec: float = 60,
) -> None:
    """Sleep `seconds` while keeping the heartbeat fresh via heartbeat_watchdog.

    Wraps the inter-iteration pause in `run_result_polling` so the external
    check_heartbeat monitor does not trip its 5-minute `running` threshold
    during normal 15-min poll intervals. Without this wrap, every polling
    cycle produces 2-3 HC /fail pings.

    If `heartbeat_path` is None (e.g., caller that doesn't care about the
    external monitor), just sleeps plain.
    """
    if heartbeat_path is None:
        time.sleep(seconds)
        return
    with heartbeat_watchdog(heartbeat_path, interval_sec=watchdog_interval_sec):
        time.sleep(seconds)


def _lock_decision_from_predictions(
    predictions,
    daily,
    date: str,
    early_lock_gap: float,
) -> tuple[bool, float | None]:
    """Return should_lock plus the best projected contender, if any."""
    from bts.picks import (
        get_game_statuses_detailed,
        pick_candidate_status_is_available,
    )
    from bts.strategy import should_lock

    try:
        detailed_statuses = get_game_statuses_detailed(date)
    except Exception:
        print(
            "  Detailed game-status lookup failed; should_lock=False.",
            file=sys.stderr,
        )
        return False, None

    pick_data = {
        "p_game_hit": daily.pick.p_game_hit,
        "projected_lineup": daily.pick.projected_lineup,
        "game_pk": daily.pick.game_pk,
    }
    all_pick_data = []
    best_projected = None
    for _, row in predictions.iterrows():
        if row.get("p_game_hit") and row["p_game_hit"] == row["p_game_hit"]:  # not NaN
            game_pk = int(row["game_pk"])
            if not pick_candidate_status_is_available(detailed_statuses.get(game_pk)):
                continue
            is_proj = "PROJECTED" in str(row.get("flags", ""))
            all_pick_data.append({
                "p_game_hit": float(row["p_game_hit"]),
                "projected_lineup": is_proj,
                "game_pk": game_pk,
            })
            if is_proj and game_pk != pick_data["game_pk"]:
                if best_projected is None or float(row["p_game_hit"]) > best_projected:
                    best_projected = float(row["p_game_hit"])

    return should_lock(pick_data, all_pick_data, early_lock_gap), best_projected


def _has_pending_future_confirmation_window(
    future_runs: list[dict],
    confirmed_sides: set[tuple[int, str]],
) -> bool:
    """Return True if a future scheduled check can still add lineup data."""
    for run in future_runs:
        for game_pk in run["game_pks"]:
            if (
                (game_pk, "away") not in confirmed_sides
                or (game_pk, "home") not in confirmed_sides
            ):
                return True
    return False


def build_skip_summary(predictions, streak) -> dict | None:
    """Summarize a skip for log/DM/dashboard: the model's best candidate today and
    the streak being protected. Returns None on unusable data — this runs in the
    daemon loop and must never raise. Emits JSON-native types only (no NaN/numpy)."""
    try:
        cols = getattr(predictions, "columns", [])
        if "p_game_hit" not in cols or "batter_name" not in cols:
            return None
        import pandas as pd
        p = pd.to_numeric(predictions["p_game_hit"], errors="coerce")
        finite = p[p.notna() & (p != float("inf")) & (p != float("-inf"))]
        if finite.empty:
            return None
        idx = finite.idxmax()
        row = predictions.loc[idx]
        name, team = row.get("batter_name"), row.get("team")
        return {
            "best_batter": str(name) if name is not None and name == name else "?",
            "best_team": str(team) if team is not None and team == team else "?",
            "best_p": float(finite.loc[idx]),
            "streak": int(streak) if streak is not None else None,
        }
    except Exception:
        return None


def format_skip_dm(date: str, summary: dict) -> str:
    """One-line operator notice for a skip. Phrased tentatively — an early cycle's
    skip may still flip to a pick if a confirmed lineup clears the bar, and a DM
    can't be retracted, so it must not claim a finality it doesn't have."""
    return (
        f"BTS {date}: No pick yet — model's best is {summary['best_batter']} "
        f"({summary['best_team']}) {summary['best_p']:.0%}, below the ~80% bar "
        f"(streak holds at {summary['streak']}). Will pick if a confirmed lineup clears it."
    )


def maybe_notify_skip(
    state: "SchedulerState",
    summary: dict,
    config: dict,
    *,
    now_iso: str,
    send=None,
) -> bool:
    """Deliver a one-time skip notice for the day. Idempotent via
    ``state.skip_notified_at`` so the operator is told once, not every cycle.
    Returns True iff a notice was sent on this call."""
    if state.skip_notified_at is not None:
        return False
    if _pick_delivery_mode(config) != "dm":
        return False
    if send is None:
        from bts.dm import send_dm
        send = send_dm
    recipient = config["bluesky"]["dm_recipient"]
    try:
        send(recipient, format_skip_dm(state.date, summary))
    except Exception as e:
        print(f"  Skip DM failed ({e}); will retry next cycle.", file=sys.stderr)
        return False
    state.skip_notified_at = now_iso
    return True


def carry_forward_skip_state(state: "SchedulerState", previous_state) -> "SchedulerState":
    """Preserve once-per-day skip fields across a same-day scheduler restart.
    run_day rebuilds SchedulerState fresh each startup (runs_completed reset), so
    without this the skip notice would re-fire on every restart (deploys, systemd
    Restart=always). skip_notified_at is the SOLE idempotency guard for a skip —
    unlike pick delivery, there is no pick file to backstop it."""
    if previous_state is not None and previous_state.date == state.date:
        state.skip_summary = previous_state.skip_summary
        state.skip_notified_at = previous_state.skip_notified_at
        # Finalization state (#3): a same-day restart rebuilds SchedulerState
        # fresh, so without carrying these the captured MDP skip (or the
        # committed-pick suppression) would be lost between the skip cycle and
        # end-of-day. Mirror the skip_summary/skip_notified_at copy above.
        state.final_skip_candidate = previous_state.final_skip_candidate
        state.committed_pick_written = previous_state.committed_pick_written
    return state


def run_single_check(
    date: str,
    all_game_pks: list[int],
    confirmed_sides: set[tuple[int, str]],
    config: dict,
    early_lock_gap: float,
) -> dict:
    """Run a single lineup check cycle.

    Short-circuits if the pick is already locked (game started or posted).
    Otherwise runs the prediction cascade and applies strategy.

    `confirmed_sides` is a mutable set of `(game_pk, side)` tuples tracking
    team-level confirmations across runs; `count_new_confirmations` updates
    it in place.

    Returns {"skipped": bool, "new_lineups": int, "should_post": bool,
             "pick_result": PickResult | None, "pick_name": str | None,
             "pick_p": float | None}.
    """
    from bts.contest_state import ContestStateError
    from bts.orchestrator import run_and_pick
    from bts.picks import save_pick, load_pick, classify_pick_lock_state
    from bts.strategy import PickResult

    new_count = count_new_confirmations(all_game_pks, confirmed_sides)

    # Short-circuit: if pick is already locked, skip the expensive cascade
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    existing = load_pick(date, picks_dir)
    if existing:
        lock_state = classify_pick_lock_state(existing, date)
        if lock_state.stale:
            print(
                f"  Existing pick stale ({lock_state.reason}); regenerating.",
                file=sys.stderr,
            )
        elif lock_state.locked:
            print(
                f"  Pick already locked ({lock_state.reason}) — skipping cascade.",
                file=sys.stderr,
            )
            return {"skipped": False, "new_lineups": new_count, "should_post": False,
                    "pick_result": PickResult(daily=existing, locked=True),
                    "pick_name": existing.pick.batter_name,
                    "pick_p": existing.pick.p_game_hit,
                    # Pre-cascade lock returns BEFORE run_and_pick — no selection this cycle.
                    "selection": None}

    print(f"  {new_count} new confirmed lineup(s). Running predictions...", file=sys.stderr)

    heartbeat_path = Path(config.get("orchestrator", {}).get("heartbeat_path", "data/.heartbeat"))
    stall_after = float(config.get("scheduler", {}).get("heartbeat_stall_after_sec", 900))
    durations_path = Path("data/health_state/cascade_stage_durations.jsonl")
    try:
        with heartbeat_watchdog(
            heartbeat_path, interval_sec=60,
            kind="primary", date=date,
            stall_after_sec=stall_after, durations_path=durations_path,
        ):
            predictions, sel, tier = run_and_pick(
                config,
                date,
                require_detailed_statuses=False,
            )
            pick_result = sel.pick_result if sel is not None else None
    except ContestStateError as e:
        print(f"  CONTEST STATE ERROR — no pick made: {e}", file=sys.stderr)
        _alert_contest_state_failure(config, e)
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": None, "pick_name": None, "pick_p": None,
                # run_and_pick raised — no selection produced this cycle.
                "selection": None}

    if predictions is None or pick_result is None:
        skip_summary = None
        if predictions is not None and not predictions.empty:
            # Skip day: predictions ran but the policy declined to pick (best
            # candidate below the pick bar). Surface it instead of staying silent
            # — the 2026-06-18 incident, where a legit skip looked like a hang.
            from bts.contest_state import load_decision_streak_state
            try:
                streak = load_decision_streak_state(
                    picks_dir, require_contest_state=False).streak
            except Exception:
                streak = None
            skip_summary = build_skip_summary(predictions, streak)
            if skip_summary is not None:
                print(f"  SKIP — best {skip_summary['best_batter']} "
                      f"({skip_summary['best_team']}) {skip_summary['best_p']:.1%} below the "
                      f"pick bar; streak holds at {skip_summary['streak']}.", file=sys.stderr)
            else:
                print("  No pick this cycle (no usable candidate in predictions).",
                      file=sys.stderr)
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result, "pick_name": None, "pick_p": None,
                "skip_summary": skip_summary, "selection": sel}

    if pick_result.locked:
        print(f"  Pick locked: {pick_result.daily.pick.batter_name} "
              f"({pick_result.daily.pick.team}) {pick_result.daily.pick.p_game_hit:.1%}",
              file=sys.stderr)
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result,
                "pick_name": pick_result.daily.pick.batter_name,
                "pick_p": pick_result.daily.pick.p_game_hit,
                "selection": sel}

    # Save candidate pick — attach provenance v1 fields first (per Codex #168).
    from bts.picks import attach_provenance
    from bts.simulate.mdp import DEFAULT_POLICY_PATH
    models_dir = config["orchestrator"].get("models_dir", "data/models")
    attach_provenance(
        pick_result.daily,
        blend_path=Path(models_dir) / f"blend_{date}.pkl",
        policy_path=DEFAULT_POLICY_PATH,
    )
    save_pick(pick_result.daily, picks_dir)

    do_post, best_projected = _lock_decision_from_predictions(
        predictions,
        pick_result.daily,
        date,
        early_lock_gap,
    )

    # Log the decision
    pick = pick_result.daily.pick
    gap_info = ""
    if best_projected is not None:
        gap = pick.p_game_hit - best_projected
        gap_info = f", gap={gap:.1%} vs projected {best_projected:.1%}"
    print(f"  Pick: {pick.batter_name} ({pick.team}) {pick.p_game_hit:.1%}"
          f"{gap_info} → should_lock={do_post}", file=sys.stderr)

    return {"skipped": False, "new_lineups": new_count, "should_post": do_post,
            "pick_result": pick_result,
            "pick_name": pick.batter_name, "pick_p": pick.p_game_hit,
            "selection": sel}


def _run_shadow_prediction(
    config: dict,
    date: str,
    production_pick_name: str,
    *,
    allow_prior_dispatched: bool = False,
    attempt_reason: str = "scheduler_inline_shadow_attempt",
    unit: str | None = None,
) -> None:
    """Run shadow model prediction and save result. Never raises.

    Threads ``data_dir`` and ``models_dir`` from the orchestrator config
    into both the prediction call and provenance attachment so the
    blend-artifact hash reflects the artifact actually loaded (per
    Codex bus #170/#172). A non-default TOML must not produce a path
    that loads one blend artifact while hashing another.
    """
    from bts.picks import get_game_statuses_detailed, load_shadow_pick, load_streak

    picks_dir = Path(config["orchestrator"]["picks_dir"])
    data_dir = config["orchestrator"].get("data_dir", "data/processed")
    models_dir = config["orchestrator"].get("models_dir", "data/models")
    heartbeat_path = Path(
        config["orchestrator"].get("heartbeat_path", picks_dir.parent / ".heartbeat")
    )

    try:
        # Shadow picks are single-shot daily artifacts; skipping an existing
        # file prevents scheduler restart loops after a successful write.
        existing = load_shadow_pick(date, picks_dir)
        if existing is not None:
            _update_analytics_job_status(
                config,
                date,
                "shadow",
                "completed",
                reason="existing_shadow_artifact",
            )
            print("  [SHADOW MODEL] Existing shadow pick found; skipping.", file=sys.stderr)
            return

        prior_status = _analytics_job_status(config, date, "shadow")
        prior = prior_status.get("status")
        if prior in {"completed", "failed"}:
            print(
                "  [SHADOW MODEL] Prior shadow attempt recorded "
                f"({prior_status.get('status')}); skipping retry.",
                file=sys.stderr,
            )
            return
        if prior == "dispatched" and not allow_prior_dispatched:
            _update_analytics_job_status(
                config,
                date,
                "shadow",
                "failed",
                reason="prior_dispatched_without_artifact",
                dispatched_at=(
                    prior_status.get("dispatched_at") or prior_status.get("updated_at")
                ),
            )
            print(
                "  [SHADOW MODEL] Prior shadow attempt was left dispatched "
                "without an artifact; marking failed and skipping retry.",
                file=sys.stderr,
            )
            return

        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "dispatched",
            reason=attempt_reason,
            unit=unit,
        )

        shadow_stall_after = float(
            config.get("scheduler", {}).get("heartbeat_stall_after_sec", 900)
        )
        with heartbeat_watchdog(
            heartbeat_path, interval_sec=60,
            kind="shadow", date=date,
            stall_after_sec=shadow_stall_after,
            durations_path=Path("data/health_state/cascade_stage_durations.jsonl"),
        ):
            predictions = predict_local_shadow(
                date, data_dir=data_dir, models_dir=models_dir
            )
        if predictions is None:
            _update_analytics_job_status(
                config,
                date,
                "shadow",
                "failed",
                reason="prediction_failed_or_none",
            )
            print("  [SHADOW MODEL] No predictions returned.", file=sys.stderr)
            return

        streak = load_streak(picks_dir)
        try:
            game_statuses_detailed = get_game_statuses_detailed(date)
        except Exception:
            game_statuses_detailed = None
        result = select_pick(
            predictions,
            date,
            picks_dir,
            streak=streak,
            for_shadow=True,
            game_statuses_detailed=game_statuses_detailed,
            require_detailed_statuses=True,
        ).pick_result
        if result is None or result.daily is None:
            _update_analytics_job_status(
                config,
                date,
                "shadow",
                "failed",
                reason="select_pick_returned_none",
            )
            print("  [SHADOW MODEL] Skip (below threshold).", file=sys.stderr)
            return

        # Shadow picks are also fresh DailyPick JSONs — attach provenance v1
        # using the same models_dir as the prediction call (per Codex #172).
        from bts.picks import attach_provenance
        from bts.simulate.mdp import DEFAULT_POLICY_PATH
        attach_provenance(
            result.daily,
            blend_path=Path(models_dir) / f"blend_{date}_shadow.pkl",
            policy_path=DEFAULT_POLICY_PATH,
        )
        save_shadow_pick(result.daily, picks_dir)
        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "completed",
            reason="shadow_artifact_written",
        )
        shadow_name = result.daily.pick.batter_name
        shadow_team = result.daily.pick.team
        shadow_p = result.daily.pick.p_game_hit
        agreed = shadow_name == production_pick_name
        tag = "AGREES" if agreed else f"DISAGREES (prod: {production_pick_name})"
        print(f"  [SHADOW MODEL] {shadow_name} ({shadow_team}) "
              f"{shadow_p:.1%} — {tag}", file=sys.stderr)
    except Exception as e:
        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "failed",
            reason=f"exception: {e}",
        )
        print(f"  [SHADOW MODEL] Failed: {e}", file=sys.stderr)


def _trigger_shadow_prediction_on_lock(
    config: dict,
    date: str,
    production_pick_name: str,
) -> None:
    """Run or queue shadow prediction after a pick is locked.

    By default this preserves the historical inline behavior. Production can
    opt into an out-of-process unit with ``scheduler.shadow_model_unit`` once
    the matching systemd unit is installed.
    """
    from bts.picks import load_shadow_pick

    sched_config = config.get("scheduler", {})
    command = sched_config.get("shadow_model_command")
    unit = sched_config.get("shadow_model_unit")
    if command == "":
        command = None
    if unit == "":
        unit = None

    if command is None and unit is None:
        _run_shadow_prediction(config, date, production_pick_name)
        return

    picks_dir = Path(config["orchestrator"]["picks_dir"])
    try:
        existing = load_shadow_pick(date, picks_dir)
    except Exception:
        existing = None
    if existing is not None:
        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "completed",
            reason="existing_shadow_artifact",
            unit=unit,
        )
        print("  [SHADOW MODEL] Existing shadow pick found; skipping trigger.",
              file=sys.stderr)
        return

    prior_status = _analytics_job_status(config, date, "shadow")
    if prior_status.get("status") in {"dispatched", "completed", "failed"}:
        print(
            "  [SHADOW MODEL] Prior shadow attempt recorded "
            f"({prior_status.get('status')}); skipping trigger.",
            file=sys.stderr,
        )
        return

    if command is None:
        args = [
            "systemctl",
            "--user",
            "start",
            "--no-block",
            unit or "bts-shadow-prediction.service",
        ]
    elif isinstance(command, str):
        args = shlex.split(command.format(date=date))
    else:
        args = [str(part).format(date=date) for part in command]

    timeout = float(sched_config.get("shadow_model_trigger_timeout_sec", 10))
    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except Exception as exc:
        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "failed",
            reason=f"trigger_exception: {exc}",
            unit=unit,
        )
        print(f"  [SHADOW MODEL] Trigger failed (suppressed): {exc}",
              file=sys.stderr)
        return

    if result.returncode == 0:
        _update_analytics_job_status(
            config,
            date,
            "shadow",
            "dispatched",
            reason="trigger_queued",
            unit=unit,
        )
        print(f"  [SHADOW MODEL] Trigger queued for {date}.", file=sys.stderr)
        return

    detail = (result.stderr or result.stdout or "").strip()
    _update_analytics_job_status(
        config,
        date,
        "shadow",
        "failed",
        reason=f"trigger_returned_{result.returncode}: {detail}",
        unit=unit,
    )
    if detail:
        detail = f": {detail}"
    print(f"  [SHADOW MODEL] Trigger returned {result.returncode}{detail}",
          file=sys.stderr)


def _refresh_pick_at_fallback_decision(
    config: dict,
    date: str,
    cached_daily,
    early_lock_gap: float,
) -> FallbackRefreshResult:
    """Re-run predictions right before fallback delivery so late-arriving
    lineups can update the pick. If the refreshed pick differs from the cached
    one, log the swap and persist the fresh daily before delivery.

    Returns the DailyPick plus a fresh should_lock decision. Falls back to
    ``cached_daily`` with should_post=None on any error (cascade failure,
    locked result) so the fallback path stays robust — we always have
    *something* to deliver if the loop reaches here.
    """
    from bts.contest_state import ContestStateError
    from bts.picks import save_pick

    picks_dir = Path(config["orchestrator"]["picks_dir"])
    heartbeat_path = Path(config.get("orchestrator", {}).get("heartbeat_path", "data/.heartbeat"))

    try:
        with heartbeat_watchdog(heartbeat_path, interval_sec=60):
            predictions, sel, _ = run_and_pick(
                config,
                date,
                require_detailed_statuses=False,
            )
            pick_result = sel.pick_result if sel is not None else None
    except ContestStateError:
        raise
    except Exception as e:
        print(f"  FALLBACK REFRESH: re-predict failed ({e}), using cached pick",
              file=sys.stderr)
        return FallbackRefreshResult(cached_daily, None)

    if pick_result is None or pick_result.daily is None:
        print("  FALLBACK REFRESH: no fresh pick available, using cached",
              file=sys.stderr)
        return FallbackRefreshResult(cached_daily, None)

    fresh = pick_result.daily

    if cached_daily and fresh.pick.batter_id != cached_daily.pick.batter_id:
        print(
            f"  FALLBACK REFRESH: pick CHANGED "
            f"{cached_daily.pick.batter_name} ({cached_daily.pick.p_game_hit:.1%}) "
            f"→ {fresh.pick.batter_name} ({fresh.pick.p_game_hit:.1%})",
            file=sys.stderr,
        )
    else:
        print(
            f"  FALLBACK REFRESH: pick unchanged "
            f"({fresh.pick.batter_name} {fresh.pick.p_game_hit:.1%})",
            file=sys.stderr,
        )

    # Fresh pick from a re-prediction — attach provenance v1 fields per Codex #168.
    from bts.picks import attach_provenance
    from bts.simulate.mdp import DEFAULT_POLICY_PATH
    models_dir = config["orchestrator"].get("models_dir", "data/models")
    attach_provenance(
        fresh,
        blend_path=Path(models_dir) / f"blend_{date}.pkl",
        policy_path=DEFAULT_POLICY_PATH,
    )
    save_pick(fresh, picks_dir)
    if predictions is None:
        print(
            "  FALLBACK REFRESH: no prediction frame available; "
            "should_lock unknown",
            file=sys.stderr,
        )
        return FallbackRefreshResult(fresh, None, selection=sel)

    should_post, best_projected = _lock_decision_from_predictions(
        predictions,
        fresh,
        date,
        early_lock_gap,
    )
    gap_info = ""
    if best_projected is not None:
        gap = fresh.pick.p_game_hit - best_projected
        gap_info = f", gap={gap:.1%} vs projected {best_projected:.1%}"
    print(
        f"  FALLBACK REFRESH: should_lock={should_post}{gap_info}",
        file=sys.stderr,
    )
    return FallbackRefreshResult(fresh, should_post, selection=sel)


def _refresh_pick_at_fallback(config: dict, date: str, cached_daily):
    """Compatibility wrapper returning only the refreshed DailyPick."""
    return _refresh_pick_at_fallback_decision(
        config,
        date,
        cached_daily,
        early_lock_gap=0.03,
    ).daily


def _defer_pick_at_fallback(picks_dir: Path, date: str, daily, reason: str) -> Path:
    """Archive and remove an unsafe fallback candidate so later checks refresh."""
    source = picks_dir / f"{date}.json"
    archive_dir = picks_dir / date
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_et().strftime("%Y%m%dT%H%M%S%z")
    archive = archive_dir / f"deferred_fallback_{stamp}.json"
    payload = asdict(daily)
    payload["deferred_fallback"] = {
        "reason": reason,
        "deferred_at": _now_et().isoformat(),
    }
    archive.write_text(json.dumps(payload, indent=2))
    if source.exists():
        source.unlink()
    return archive


def _trigger_live_forward_capture_on_lock(config: dict, date: str) -> None:
    """Start a non-blocking live-forward capture after a pick is locked.

    Capture is observational and must not block or undo the production pick
    lifecycle. The default production path uses the existing systemd one-shot
    service; local/dev callers can disable it or provide an explicit command in
    scheduler config.
    """
    sched_config = config.get("scheduler", {})
    if not sched_config.get("live_forward_capture_on_lock", True):
        _update_analytics_job_status(
            config,
            date,
            "live_forward_capture",
            "disabled",
            reason="live_forward_capture_on_lock=false",
        )
        return

    command = sched_config.get("live_forward_capture_command")
    if command is None:
        unit = sched_config.get(
            "live_forward_capture_unit",
            "bts-live-forward-capture.service",
        )
        args = ["systemctl", "--user", "start", "--no-block", unit]
    elif isinstance(command, str):
        unit = None
        args = shlex.split(command.format(date=date))
    else:
        unit = None
        args = [str(part).format(date=date) for part in command]

    timeout = float(sched_config.get("live_forward_capture_trigger_timeout_sec", 10))
    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except Exception as exc:
        _update_analytics_job_status(
            config,
            date,
            "live_forward_capture",
            "failed",
            reason=f"trigger_exception: {exc}",
            unit=unit,
        )
        print(
            f"  live-forward capture trigger failed (suppressed): {exc}",
            file=sys.stderr,
        )
        return

    if result.returncode == 0:
        _update_analytics_job_status(
            config,
            date,
            "live_forward_capture",
            "dispatched",
            reason="trigger_queued",
            unit=unit,
        )
        print(
            f"  live-forward capture trigger queued for {date}.",
            file=sys.stderr,
        )
        return

    detail = (result.stderr or result.stdout or "").strip()
    _update_analytics_job_status(
        config,
        date,
        "live_forward_capture",
        "failed",
        reason=f"trigger_returned_{result.returncode}: {detail}",
        unit=unit,
    )
    if detail:
        detail = f": {detail}"
    print(
        f"  live-forward capture trigger returned {result.returncode}{detail}",
        file=sys.stderr,
    )


def poll_game_result(game_pk: int) -> str:
    """Check a game's current status.

    Returns one of: "final", "live", "suspended", "preview", "unknown".
    """
    try:
        resp = json.loads(retry_urlopen(
            f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        ).read())
    except Exception:
        return "unknown"

    abstract = resp["gameData"]["status"]["abstractGameCode"]
    detailed = resp["gameData"]["status"].get("detailedState", "")

    if detailed.strip().lower() in {"postponed", "cancelled", "canceled"}:
        return "final"
    if abstract == "F":
        return "final"
    if "suspend" in detailed.lower():
        return "suspended"
    if abstract == "L":
        return "live"
    if abstract == "P":
        return "preview"
    return "unknown"


def _check_hits_midgame(daily, date: str) -> list[bool | None]:
    """Check if picked batters have hits in a live or final game.

    Returns list of True/False/None per pick (primary + optional double).
    None = batter not yet in boxscore or no AB yet.
    """
    from bts.picks import API_BASE
    results = []
    for pick in [daily.pick] + ([daily.double_down] if daily.double_down else []):
        try:
            resp = json.loads(retry_urlopen(
                f"{API_BASE}/api/v1.1/game/{pick.game_pk}/feed/live",
                timeout=15,
            ).read())
            for side in ("away", "home"):
                players = resp["liveData"]["boxscore"]["teams"][side]["players"]
                key = f"ID{pick.batter_id}"
                if key in players:
                    hits = players[key].get("stats", {}).get("batting", {}).get("hits", 0)
                    results.append(hits > 0)
                    break
            else:
                results.append(None)
        except Exception:
            results.append(None)
    return results


def run_result_polling(
    game_pk: int,
    date: str,
    picks_dir: Path,
    poll_interval_min: int = 15,
    cap_hour_et: int = 5,
    heartbeat_path: Path | None = None,
) -> str:
    """Poll for pick results, checking for hits mid-game.

    Posts reply as soon as all picks have hits (early exit) or when
    game goes Final/Suspended. Returns "final", "suspended", or "unresolved".
    """
    from bts.picks import (
        _is_void_detailed_state,
        active_streak_results,
        effective_daily_result,
        get_game_statuses_detailed,
        load_pick,
        load_streak,
        resolve_daily_slot_results,
        save_pick,
        update_streak,
    )

    early_replied = False

    # Determine all game PKs to track (primary + double-down if different game)
    daily = load_pick(date, picks_dir)
    all_game_pks = {game_pk}
    if daily and daily.double_down and daily.double_down.game_pk != game_pk:
        all_game_pks.add(daily.double_down.game_pk)

    while True:
        if heartbeat_path:
            write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING,
                           extra={"phase": "result_polling"})
            notify_watchdog()
        now = _now_et()
        if now.hour >= cap_hour_et and now.hour < 10:
            print(f"  Result polling capped at {cap_hour_et}am ET. Flagging as unresolved.",
                  file=sys.stderr)
            daily = load_pick(date, picks_dir)
            if daily and daily.result in ("hit", "miss", "void"):
                return "final"
            if daily:
                daily.result = "unresolved"
                save_pick(daily, picks_dir)
            return "unresolved"

        daily = load_pick(date, picks_dir)
        if not daily:
            return "unresolved"
        if daily.result in ("hit", "miss", "void"):
            return "final"

        # Check status of ALL games involved in today's picks
        statuses = {pk: poll_game_result(pk) for pk in all_game_pks}
        try:
            detailed_statuses = get_game_statuses_detailed(date)
        except Exception:
            detailed_statuses = {}
        for pk in all_game_pks:
            detailed = detailed_statuses.get(pk, {}).get("detailed")
            if detailed and _is_void_detailed_state(detailed):
                statuses[pk] = "final"
        status_summary = ", ".join(f"{pk}: {s}" for pk, s in statuses.items())
        print(f"  [{now.strftime('%H:%M ET')}] Games: {status_summary}", file=sys.stderr)

        any_live_or_final = any(s in ("live", "final") for s in statuses.values())
        all_final = all(s == "final" for s in statuses.values())
        any_suspended = any(s == "suspended" for s in statuses.values())

        # Check for mid-game hits (even if games are still live)
        if not early_replied and any_live_or_final and not all_final:
            hit_checks = _check_hits_midgame(daily, date)
            n_picks = 1 + (1 if daily.double_down else 0)
            confirmed_hits = [h for h in hit_checks[:n_picks] if h is True]

            if len(confirmed_hits) == n_picks:
                # All picks have hits — post early reply
                new_streak = update_streak([True] * n_picks, picks_dir)
                daily.result = "hit"
                save_pick(daily, picks_dir)
                print(f"  All picks have hits! Streak: {new_streak}.", file=sys.stderr)

                if daily.bluesky_uri:
                    try:
                        from bts.posting import format_result_reply, reply_to_bluesky
                        reply_text = format_result_reply("hit", new_streak)
                        reply_uri = reply_to_bluesky(reply_text, daily.bluesky_uri)
                        print(f"  Result reply posted (mid-game): {reply_uri}", file=sys.stderr)
                    except Exception as e:
                        print(f"  Result reply failed: {e}", file=sys.stderr)
                early_replied = True

        if all_final:
            if not early_replied:
                # All games over, haven't replied yet — do final check
                slot_results = resolve_daily_slot_results(daily, date)
                if slot_results is None:
                    daily.result = "unresolved"
                    save_pick(daily, picks_dir)
                    return "unresolved"

                results = active_streak_results(slot_results)
                new_streak = update_streak(results, picks_dir) if results else load_streak(picks_dir)
                daily.slot_results = slot_results
                daily.result = effective_daily_result(slot_results)
                save_pick(daily, picks_dir)
                print(f"  Result: {daily.result}. Streak: {new_streak}.", file=sys.stderr)

                if daily.bluesky_uri:
                    try:
                        from bts.posting import format_result_reply, reply_to_bluesky
                        reply_text = format_result_reply(daily.result, new_streak)
                        reply_uri = reply_to_bluesky(reply_text, daily.bluesky_uri)
                        print(f"  Result reply posted: {reply_uri}", file=sys.stderr)
                    except Exception as e:
                        print(f"  Result reply failed: {e}", file=sys.stderr)
            return "final"

        if any_suspended:
            daily = load_pick(date, picks_dir)
            if daily and not early_replied:
                daily.result = "suspended"
                save_pick(daily, picks_dir)
            return "suspended"

        # Still live — wait and retry. Use _poll_interval_sleep so the
        # external heartbeat monitor stays fresh across the 15-min gap.
        _poll_interval_sleep(heartbeat_path, poll_interval_min * 60)


def run_day(
    date: str,
    config: dict,
    dry_run: bool = False,
) -> None:
    """Run the scheduler for a single day.

    Orchestrates the full daily lifecycle:
    1. Fetch MLB schedule
    2. Compute lineup check times (game_time - offset)
    3. Sleep between checks, run predictions when lineups confirm
    4. Deliver and lock the pick when lock conditions are met
    5. Fallback delivery if close to first pitch
    6. Doubleheader game 2 re-checks
    7. Next-day lookahead for wake-up time
    8. Result polling after games finish
    """
    from bts.contest_state import ContestStateError
    from bts.daily_decision import is_scoreable_commit
    from bts.picks import load_pick, pick_was_delivered

    sched_config = config.get("scheduler", {})
    delivery_mode = _pick_delivery_mode(config)
    if delivery_mode == "private":
        print("  [PRIVATE MODE] Bluesky posting disabled — picks saved locally only.", file=sys.stderr)
    elif delivery_mode == "dm":
        print("  [DM MODE] Public Bluesky posting disabled — picks sent by DM.", file=sys.stderr)
    shadow_model_enabled = sched_config.get("shadow_model", False)
    if shadow_model_enabled:
        print("  [SHADOW MODEL] Context stack shadow model enabled.", file=sys.stderr)
    offset_min = sched_config.get("lineup_check_offset_min", 45)
    cluster_min = sched_config.get("cluster_min", 10)
    dh_recheck_min = sched_config.get("doubleheader_recheck_min", 15)
    early_lock_gap = sched_config.get("early_lock_gap", 0.03)
    fallback_deadline_min_standard = sched_config.get("fallback_deadline_min", 35)
    fallback_deadline_min_morning = sched_config.get("fallback_deadline_min_morning", 25)
    morning_cutoff_hour = sched_config.get("morning_cutoff_hour", 11)
    missed_pick_alert_min = sched_config.get("missed_pick_alert_min", 10)
    poll_interval_min = sched_config.get("results_poll_interval_min", 15)
    cap_hour_et = sched_config.get("results_cap_hour_et", 5)
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    heartbeat_path = Path(config.get("orchestrator", {}).get("heartbeat_path", "data/.heartbeat"))
    write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
    notify_ready()
    notify_watchdog()

    # 1. Fetch schedule
    print(f"[{_now_et().strftime('%H:%M ET')}] Fetching schedule for {date}...", file=sys.stderr)
    games = fetch_schedule(date)
    if not games:
        print(f"No games scheduled for {date}.", file=sys.stderr)
        # Idle until tomorrow's wake instead of returning — returning lets systemd
        # Restart=always relaunch within ~30s and thrash all day on an off-day
        # (the All-Star break is ~4 days), spiking NRestarts into a false
        # restart_spike CRITICAL. (audit E1)
        write_heartbeat(heartbeat_path, state=HeartbeatState.IDLE_END_OF_DAY)
        notify_watchdog()
        _idle_until_next_wakeup(
            _next_day_wakeup(date, sched_config).isoformat(), heartbeat_path
        )
        return

    all_game_pks = [g["gamePk"] for g in games]
    dh_game2s = detect_doubleheader_game2s(games)

    # 2. Compute run times
    runs = compute_run_times(games, offset_min=offset_min, cluster_min=cluster_min)

    print(f"  {len(games)} games, {len(runs)} scheduled checks:", file=sys.stderr)
    for r in runs:
        print(f"    {r['time_et'].strftime('%H:%M ET')} — {len(r['game_pks'])} game(s)", file=sys.stderr)
    if dh_game2s:
        print(f"  Doubleheader game 2s (fluid time): {dh_game2s}", file=sys.stderr)

    if dry_run:
        print("  (--dry-run: not executing checks)", file=sys.stderr)
        return

    # 3. Initialize state
    # Team-level confirmation tracking: set of (game_pk, side) tuples. A game
    # with one side confirmed differs from a game with both sides confirmed,
    # and the prediction pipeline notices — so we count both independently.
    confirmed_sides: set[tuple[int, str]] = set()
    previous_state = load_state(date, picks_dir)
    state = SchedulerState(
        date=date,
        schedule_fetched_at=_now_et().isoformat(),
        games=[{
            "game_pk": g["gamePk"],
            "game_time_et": game_time_et(g).isoformat(),
            "lineup_confirmed": False,
            "is_doubleheader_game2": g["gamePk"] in dh_game2s,
        } for g in games],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
        analytics_jobs=previous_state.analytics_jobs if previous_state else None,
    )
    # Finalization state (captured MDP-skip candidate + committed_pick_written)
    # now lives ON `state` and is carried forward across a same-day restart by
    # carry_forward_skip_state, so it survives a deploy/Restart=always between an
    # MDP skip and end-of-day (#3 / GH #144).
    carry_forward_skip_state(state, previous_state)
    save_state(state, picks_dir)

    # 4. Main loop — sleep until each check time, then run
    for run_info in runs:
        target = run_info["time_et"]
        now = _now_et()

        if now < target:
            write_heartbeat(
                heartbeat_path,
                state=HeartbeatState.SLEEPING,
                sleeping_until=target.astimezone(UTC),
            )
            notify_watchdog()
            wait_secs = (target - now).total_seconds()
            print(f"  Sleeping until {target.strftime('%H:%M ET')} "
                  f"({wait_secs / 60:.0f} min)...", file=sys.stderr)
            _watchdog_ping_sleep(wait_secs)
            write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
            notify_watchdog()

        now = _now_et()
        if now < target:
            continue

        print(f"\n[{_now_et().strftime('%H:%M ET')}] Running lineup check...", file=sys.stderr)
        result = run_single_check(
            date=date,
            all_game_pks=all_game_pks,
            confirmed_sides=confirmed_sides,
            config=config,
            early_lock_gap=early_lock_gap,
        )

        # Track the day's finalization intent off THIS cycle's selection (Task 3).
        # Narrow lifecycle (Codex r2 P0): SET on a genuine MDP skip; CLEAR only on a
        # genuine pick attempt. Everything else — sel is None (no-predictions /
        # ContestStateError), a non-delivered classification (sel.action is None,
        # the #144 stale preview), or sel.no_pick_reason set — LEAVES the captured
        # skip unchanged so it can still be recorded at end-of-day. End-of-day
        # suppression for real picks is handled by committed_pick_written.
        sel = result.get("selection")
        if sel is not None and sel.action == "skip" and sel.source == "mdp":
            state.final_skip_candidate = {
                "primary": sel.primary_candidate,
                "streak": sel.streak,
                "saver_available": sel.saver_available,
            }
            save_state(state, picks_dir)  # persist so a same-day restart inherits the skip (#3)
        elif sel is not None and sel.action in {"single", "double"}:
            state.final_skip_candidate = None
            save_state(state, picks_dir)

        state.runs_completed.append({
            "time": _now_et().isoformat(),
            "new_lineups": result["new_lineups"],
            "skipped": result["skipped"],
            "pick_name": result.get("pick_name"),
            "pick_p": round(result["pick_p"], 4) if result.get("pick_p") else None,
        })
        confirmed_game_pks_derived = {pk for pk, _ in confirmed_sides}
        state.confirmed_game_pks = sorted(confirmed_game_pks_derived)
        for g in state.games:
            g["lineup_confirmed"] = g["game_pk"] in confirmed_game_pks_derived
        skip_summary = result.get("skip_summary")
        if skip_summary:
            state.skip_summary = skip_summary
            try:
                maybe_notify_skip(state, skip_summary, config, now_iso=_now_et().isoformat())
            except Exception as e:  # a skip notice must never take down the daemon loop
                print(f"  Skip notify failed ({e}); continuing.", file=sys.stderr)
        save_state(state, picks_dir)

        if result["pick_result"] and result["pick_result"].locked:
            state.pick_locked = True
            state.pick_locked_at = _now_et().isoformat()
            save_state(state, picks_dir)
            print(f"  Pick already locked (game started or previously delivered).",
                  file=sys.stderr)
            # Single chokepoint for classification-lock records (Task 3): both the
            # pre-cascade locked-existing return AND any select_pick-locked result
            # arrive here as result["pick_result"].locked. Writes a scoreable record
            # only when the existing pick was genuinely delivered; a non-delivered
            # stale-preview lock writes nothing and does not set committed_pick_written
            # (so the captured MDP skip still records at end-of-day — GH #144).
            ld = result["pick_result"].daily
            _write_classification_decision(
                picks_dir, date,
                action=("double" if ld.double_down else "single"),
                delivered=pick_was_delivered(ld),
                primary=_row_from_daily(ld.pick),
                double_down=_row_from_daily(ld.double_down),
                state=state,
            )

        if result["should_post"] and result["pick_result"] and not result["pick_result"].locked:
            daily = result["pick_result"].daily
            _deliver_and_lock_pick(daily, config, picks_dir, state, date, "lineup",
                                   selection=result.get("selection"))

        if state.pick_locked:
            # Run shadow model if enabled (after production pick is resolved)
            if shadow_model_enabled and result.get("pick_result") and result["pick_result"].daily:
                prod_name = result["pick_result"].daily.pick.batter_name
                _trigger_shadow_prediction_on_lock(config, date, prod_name)
            print(f"  Pick locked. Stopping lineup checks.", file=sys.stderr)
            break

        # If the earliest game in the slate starts before the next scheduled
        # check, wake up for forced delivery. Use earliest of primary + double-down
        # because BTS app rejects submissions once the FIRST game has started.
        if not state.pick_locked and result.get("pick_result") and result["pick_result"].daily:
            earliest_game_et = _earliest_pick_game_et(result["pick_result"].daily)
            fallback_min = resolve_fallback_deadline_min(
                earliest_game_et,
                standard_min=fallback_deadline_min_standard,
                morning_min=fallback_deadline_min_morning,
                morning_cutoff_hour=morning_cutoff_hour,
            )
            fallback_deadline = earliest_game_et - timedelta(minutes=fallback_min)
            now = _now_et()

            # Is there a later check that fires before the deadline?
            run_idx = runs.index(run_info)
            future_runs = runs[run_idx + 1:]
            next_checks = [r["time_et"] for r in future_runs]
            has_earlier_check = any(t <= fallback_deadline for t in next_checks)
            has_pending_future_window = _has_pending_future_confirmation_window(
                future_runs,
                confirmed_sides,
            )

            if not has_earlier_check:
                if now < fallback_deadline:
                    write_heartbeat(
                        heartbeat_path,
                        state=HeartbeatState.SLEEPING,
                        sleeping_until=fallback_deadline.astimezone(UTC),
                    )
                    notify_watchdog()
                    wait = (fallback_deadline - now).total_seconds()
                    print(f"  Earliest pick game at {earliest_game_et.strftime('%H:%M ET')}, "
                          f"no check before then — fallback at "
                          f"{fallback_deadline.strftime('%H:%M ET')} "
                          f"({wait / 60:.0f} min)...", file=sys.stderr)
                    _watchdog_ping_sleep(wait)
                    write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
                    notify_watchdog()

                # Force-deliver current pick (waited to deadline, or past it).
                # Re-run predictions first in case late-arriving lineups
                # changed the top pick since the last scheduled check.
                daily = load_pick(date, picks_dir)
                if daily and not pick_was_delivered(daily):
                    try:
                        refresh = _refresh_pick_at_fallback_decision(
                            config,
                            date,
                            daily,
                            early_lock_gap,
                        )
                    except ContestStateError as e:
                        print(
                            f"  FALLBACK BLOCKED — contest state invalid: {e}",
                            file=sys.stderr,
                        )
                        _alert_contest_state_failure(config, e)
                        continue
                    daily = refresh.daily
                    if refresh.should_post is False and has_pending_future_window:
                        archive = _defer_pick_at_fallback(
                            picks_dir,
                            date,
                            daily,
                            reason="should_lock_false_future_checks_remain",
                        )
                        print(
                            "  FALLBACK DEFERRED — should_lock=False and "
                            f"{len(next_checks)} future check(s) with pending "
                            "lineup data remain; "
                            f"archived {archive.name}.",
                            file=sys.stderr,
                        )
                        continue
                    print(f"  FALLBACK — delivering before game starts.", file=sys.stderr)
                    _deliver_and_lock_pick(daily, config, picks_dir, state, date, "fallback",
                                           selection=refresh.selection)

                if state.pick_locked:
                    if shadow_model_enabled and daily:
                        _trigger_shadow_prediction_on_lock(
                            config,
                            date,
                            daily.pick.batter_name,
                        )
                    print(f"  Pick locked. Stopping lineup checks.", file=sys.stderr)
                    break

    # 5. Fallback — if not yet locked, check for deadline (use earliest of
    # primary + double-down so we never miss the BTS submission window).
    if not state.pick_locked:
        daily = load_pick(date, picks_dir)
        if daily and not pick_was_delivered(daily):
            earliest_game_et = _earliest_pick_game_et(daily)
            now = _now_et()
            mins_to_game = (earliest_game_et - now).total_seconds() / 60
            fallback_min = resolve_fallback_deadline_min(
                earliest_game_et,
                standard_min=fallback_deadline_min_standard,
                morning_min=fallback_deadline_min_morning,
                morning_cutoff_hour=morning_cutoff_hour,
            )
            if mins_to_game <= fallback_min:
                # Re-run predictions first in case late-arriving lineups
                # changed the top pick since the last scheduled check.
                try:
                    refresh = _refresh_pick_at_fallback_decision(
                        config,
                        date,
                        daily,
                        early_lock_gap,
                    )
                    daily = refresh.daily
                except ContestStateError as e:
                    print(
                        f"  FALLBACK BLOCKED — contest state invalid: {e}",
                        file=sys.stderr,
                    )
                    _alert_contest_state_failure(config, e)
                    daily = None
                else:
                    print(
                        f"  FALLBACK — {fallback_min}min to first pitch, delivering on projected data.",
                        file=sys.stderr,
                    )
                    _deliver_and_lock_pick(daily, config, picks_dir, state, date, "final fallback",
                                           selection=refresh.selection)

        if state.pick_locked and shadow_model_enabled and daily:
            _trigger_shadow_prediction_on_lock(config, date, daily.pick.batter_name)

    # 5b. Missed-pick early alert (audit E3): if delivery failed, warn the
    # operator in-window — while they can still post manually — instead of only
    # the hours-late EOD post_failure DM.
    if not state.pick_locked:
        _maybe_alert_missed_pick(config, date, picks_dir, missed_pick_alert_min, heartbeat_path)

    # 6. Doubleheader game 2 re-checks
    for pk in dh_game2s:
        if any(cs_pk == pk for cs_pk, _ in confirmed_sides):
            continue
        if state.pick_locked:
            break
        print(f"  DH game 2 ({pk}): re-checking every {dh_recheck_min}min...", file=sys.stderr)
        for _ in range(10):
            # Watchdog-fed sleep + SLEEPING heartbeat so this long inter-check
            # wait neither SIGABRTs the daemon (WatchdogSec=1800) nor trips the
            # external check_heartbeat monitor (audit O1).
            if heartbeat_path:
                write_heartbeat(
                    heartbeat_path,
                    state=HeartbeatState.SLEEPING,
                    sleeping_until=(
                        _now_et() + timedelta(minutes=dh_recheck_min)
                    ).astimezone(UTC),
                )
                notify_watchdog()
            _watchdog_ping_sleep(dh_recheck_min * 60)
            new = count_new_confirmations([pk], confirmed_sides)
            if new > 0:
                print(f"  DH game 2 ({pk}): lineup confirmed.", file=sys.stderr)
                break
        if heartbeat_path:
            write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
            notify_watchdog()

    # 7. Next-day lookahead for wake-up time
    tomorrow = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        tomorrow_games = fetch_schedule(tomorrow)
        wakeup = compute_wakeup_time(
            tomorrow_games,
            default_hour_et=sched_config.get("default_init_hour_et", 10),
            early_buffer_min=sched_config.get("early_game_buffer_min", 60),
        )
        state.next_wakeup = wakeup.isoformat()
        save_state(state, picks_dir)
        print(f"  Tomorrow's wake-up: {wakeup.strftime('%H:%M ET')}", file=sys.stderr)
    except Exception as e:
        print(f"  Failed to fetch tomorrow's schedule: {e}", file=sys.stderr)

    # 8. Result polling (start 10 min after game start, check for hits mid-game).
    # Gate on a GENUINE commit, not merely state.pick_locked (C2 / GH #144): a
    # NON-delivered classification-lock of a stale projected-preview <date>.json
    # sets pick_locked=True on a skip day, and run_result_polling -> update_streak
    # has no internal scoreable guard. is_scoreable_commit defers to decision.json
    # (decision.scoreable) and falls back to pick_was_delivered when no record
    # exists, so a stale preview is locked (for other purposes) but never scored.
    daily_for_poll = load_pick(date, picks_dir)
    if state.pick_locked and daily_for_poll is not None and is_scoreable_commit(date, picks_dir, daily_for_poll):
        daily = load_pick(date, picks_dir)
        if daily and daily.result is None:
            # Wait until earliest pick game (primary or double-down) + 10 min.
            # run_result_polling tracks both game_pks once it starts, but only
            # if the scheduler is awake when the earlier game begins.
            poll_start = _compute_result_poll_start(daily)
            now = _now_et()
            if now < poll_start:
                write_heartbeat(
                    heartbeat_path,
                    state=HeartbeatState.SLEEPING,
                    sleeping_until=poll_start.astimezone(UTC),
                )
                notify_watchdog()
                wait = (poll_start - now).total_seconds()
                print(f"  Waiting until {poll_start.strftime('%H:%M ET')} "
                      f"(game start + 10min, {wait / 60:.0f} min)...", file=sys.stderr)
                _watchdog_ping_sleep(wait)
                write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)
                notify_watchdog()

            game_pk = daily.pick.game_pk
            status = run_result_polling(
                game_pk, date, picks_dir,
                poll_interval_min=poll_interval_min,
                cap_hour_et=cap_hour_et,
                heartbeat_path=heartbeat_path,
            )
            state.result_status = status
            save_state(state, picks_dir)
            print(f"  Day complete. Result: {status}", file=sys.stderr)

    # End-of-day skip record (Task 3): the scheduler is the SINGLE writer of the
    # authoritative decision.json. If the day finalized as a genuine MDP skip with
    # no committed pick, record it here — immediately before the end-of-day health
    # checks (run_day idles after this; it does not return). It is NOT written in
    # the early no-games / dry-run returns, and no-ops if there is nothing to record.
    _write_endofday_skip(picks_dir, date, state)

    # End-of-day health checks. Pure observation — never modifies picks.
    # Each check is failure-isolated; the runner catches per-check errors.
    # Sends a single Bluesky DM if any CRITICAL alerts triggered.
    health_config = config.get("health_checks", {})
    if health_config.get("enabled", True):
        from bts.health.runner import (
            run_all_checks, read_systemd_nrestarts, get_self_pid,
        )
        dm_recipient = config.get("bluesky", {}).get("dm_recipient")
        models_dir = Path(
            config.get("orchestrator", {}).get("models_dir", "data/models")
        )
        shadow_model_command = sched_config.get("shadow_model_command")
        shadow_model_unit = sched_config.get("shadow_model_unit")
        if shadow_model_command == "":
            shadow_model_command = None
        if shadow_model_unit == "":
            shadow_model_unit = None
        pooled_dir = _optional_health_path(health_config.get("pooled_dir"))
        leaderboard_dir = _optional_health_path(health_config.get("leaderboard_dir"))
        try:
            run_all_checks(
                picks_dir=picks_dir,
                models_dir=models_dir,
                # The day being processed (NOT date.today()): a post-midnight or
                # early-finish EOD run must evaluate THIS date's picks/artifacts.
                today=datetime.strptime(date, "%Y-%m-%d").date(),
                # PA-frame ground truth for realized_calibration; without it the
                # check falls back to the biased streak-proxy attribution path.
                data_dir=Path(
                    config.get("orchestrator", {}).get("data_dir", "data/processed")
                ),
                dm_recipient=dm_recipient,
                scheduler_pid=get_self_pid(),
                current_nrestarts=read_systemd_nrestarts(),
                thresholds_overrides=health_config.get("thresholds"),
                pooled_dir=pooled_dir,
                leaderboard_dir=leaderboard_dir,
                shadow_model_enabled=shadow_model_enabled,
                live_forward_capture_enabled=sched_config.get(
                    "live_forward_capture_on_lock", True
                ),
                live_forward_capture_artifact_root=Path(
                    sched_config.get(
                        "live_forward_capture_artifact_root",
                        "data/validation/decision_weighted_lgbm_v0_live_forward",
                    )
                ),
                live_forward_resolve_status_root=Path(
                    health_config.get(
                        "live_forward_resolve_status_root",
                        "data/validation/decision_weighted_lgbm_v0_live_forward_resolved_status",
                    )
                ),
                live_forward_capture_unit=sched_config.get(
                    "live_forward_capture_unit",
                    "bts-live-forward-capture.service",
                ) if sched_config.get("live_forward_capture_command") is None else None,
                shadow_unit=shadow_model_unit if shadow_model_command is None else None,
                contest_state_expected=_contest_state_required(config),
            )
        except Exception as e:
            print(f"  health_checks: unexpected error (suppressed): {e}", file=sys.stderr)

    write_heartbeat(heartbeat_path, state=HeartbeatState.IDLE_END_OF_DAY)
    notify_watchdog()

    # Stay alive until tomorrow's scheduled wake — returning here would cause
    # systemd Restart=always to re-launch within 30s and cycle through the
    # short-circuit post-lock branches every ~3 min overnight.
    _idle_until_next_wakeup(state.next_wakeup, heartbeat_path)
