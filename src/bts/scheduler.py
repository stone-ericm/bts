"""Dynamic lineup scheduler for BTS.

Replaces fixed cron runs with game-time-aware lineup checks.
Checks lineups 45 min before each game, clusters nearby checks,
and commits picks only when confirmed lineup + gap threshold met.
"""

import json
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
    today_et = datetime.now(ET).replace(hour=default_hour_et, minute=0, second=0, microsecond=0)

    if not games:
        return today_et

    earliest = min(game_time_et(g) for g in games)
    early_wake = earliest - timedelta(minutes=early_buffer_min)

    if early_wake < today_et:
        return early_wake

    return today_et


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


def save_state(state: SchedulerState, picks_dir: Path) -> Path:
    """Save scheduler state to JSON."""
    date_dir = picks_dir / state.date
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / "scheduler_state.json"
    path.write_text(json.dumps(asdict(state), indent=2))
    return path


def load_state(date: str, picks_dir: Path) -> SchedulerState | None:
    """Load scheduler state from JSON. Returns None if not found."""
    path = picks_dir / date / "scheduler_state.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return SchedulerState(**data)


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
    from bts.orchestrator import run_and_pick
    from bts.picks import save_pick, get_game_statuses, load_pick
    from bts.strategy import should_lock, PickResult

    new_count = count_new_confirmations(all_game_pks, confirmed_sides)

    # Short-circuit: if pick is already locked, skip the expensive cascade
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    existing = load_pick(date, picks_dir)
    if existing:
        statuses = get_game_statuses(date)
        if statuses.get(existing.pick.game_pk) != "P" or existing.bluesky_posted:
            print(f"  Pick already locked — skipping cascade.", file=sys.stderr)
            return {"skipped": False, "new_lineups": new_count, "should_post": False,
                    "pick_result": PickResult(daily=existing, locked=True),
                    "pick_name": existing.pick.batter_name,
                    "pick_p": existing.pick.p_game_hit}

    print(f"  {new_count} new confirmed lineup(s). Running predictions...", file=sys.stderr)

    heartbeat_path = Path(config.get("orchestrator", {}).get("heartbeat_path", "data/.heartbeat"))
    with heartbeat_watchdog(heartbeat_path, interval_sec=60):
        predictions, pick_result, tier = run_and_pick(config, date)

    if predictions is None or pick_result is None:
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result, "pick_name": None, "pick_p": None}

    if pick_result.locked:
        print(f"  Pick locked: {pick_result.daily.pick.batter_name} "
              f"({pick_result.daily.pick.team}) {pick_result.daily.pick.p_game_hit:.1%}",
              file=sys.stderr)
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result,
                "pick_name": pick_result.daily.pick.batter_name,
                "pick_p": pick_result.daily.pick.p_game_hit}

    # Save candidate pick
    save_pick(pick_result.daily, picks_dir)

    # Check if we should lock — only consider picks from pickable games
    statuses = get_game_statuses(date)
    pick_data = {
        "p_game_hit": pick_result.daily.pick.p_game_hit,
        "projected_lineup": pick_result.daily.pick.projected_lineup,
        "game_pk": pick_result.daily.pick.game_pk,
    }
    all_pick_data = []
    best_projected = None
    for _, row in predictions.iterrows():
        if row.get("p_game_hit") and row["p_game_hit"] == row["p_game_hit"]:  # not NaN
            game_pk = int(row["game_pk"])
            if statuses.get(game_pk) != "P":
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

    do_post = should_lock(pick_data, all_pick_data, early_lock_gap)

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
            "pick_name": pick.batter_name, "pick_p": pick.p_game_hit}


def _run_shadow_prediction(config: dict, date: str, production_pick_name: str) -> None:
    """Run shadow model prediction and save result. Never raises."""
    from bts.picks import load_streak

    picks_dir = Path(config["orchestrator"]["picks_dir"])

    try:
        predictions = predict_local_shadow(date)
        if predictions is None:
            print("  [SHADOW MODEL] No predictions returned.", file=sys.stderr)
            return

        streak = load_streak(picks_dir)
        result = select_pick(predictions, date, picks_dir, streak=streak, for_shadow=True)
        if result is None or result.daily is None:
            print("  [SHADOW MODEL] Skip (below threshold).", file=sys.stderr)
            return

        save_shadow_pick(result.daily, picks_dir)
        shadow_name = result.daily.pick.batter_name
        shadow_team = result.daily.pick.team
        shadow_p = result.daily.pick.p_game_hit
        agreed = shadow_name == production_pick_name
        tag = "AGREES" if agreed else f"DISAGREES (prod: {production_pick_name})"
        print(f"  [SHADOW MODEL] {shadow_name} ({shadow_team}) "
              f"{shadow_p:.1%} — {tag}", file=sys.stderr)
    except Exception as e:
        print(f"  [SHADOW MODEL] Failed: {e}", file=sys.stderr)


def _refresh_pick_at_fallback(config: dict, date: str, cached_daily):
    """Re-run predictions right before the fallback post so any late-arriving
    lineups can update the pick. If the refreshed pick differs from the cached
    one, log the swap and persist the fresh daily to disk before posting.

    Returns the DailyPick to post. Falls back to ``cached_daily`` on any error
    (cascade failure, empty predictions, locked result) so the fallback path
    stays robust — we always have *something* to post if the loop reaches here.
    """
    from bts.picks import save_pick

    picks_dir = Path(config["orchestrator"]["picks_dir"])
    heartbeat_path = Path(config.get("orchestrator", {}).get("heartbeat_path", "data/.heartbeat"))

    try:
        with heartbeat_watchdog(heartbeat_path, interval_sec=60):
            _, pick_result, _ = run_and_pick(config, date)
    except Exception as e:
        print(f"  FALLBACK REFRESH: re-predict failed ({e}), using cached pick",
              file=sys.stderr)
        return cached_daily

    if pick_result is None or pick_result.daily is None:
        print("  FALLBACK REFRESH: no fresh pick available, using cached",
              file=sys.stderr)
        return cached_daily

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

    save_pick(fresh, picks_dir)
    return fresh


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
    from bts.picks import load_pick, check_hit, update_streak, save_pick

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
            if daily:
                daily.result = "unresolved"
                save_pick(daily, picks_dir)
            return "unresolved"

        daily = load_pick(date, picks_dir)
        if not daily:
            return "unresolved"

        # Check status of ALL games involved in today's picks
        statuses = {pk: poll_game_result(pk) for pk in all_game_pks}
        status_summary = ", ".join(f"{pk}: {s}" for pk, s in statuses.items())
        print(f"  [{now.strftime('%H:%M ET')}] Games: {status_summary}", file=sys.stderr)

        any_live_or_final = any(s in ("live", "final") for s in statuses.values())
        all_final = all(s == "final" for s in statuses.values())
        any_suspended = any(s == "suspended" for s in statuses.values())

        # Check for mid-game hits (even if games are still live)
        if not early_replied and any_live_or_final:
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
                primary_result = check_hit(
                    daily.pick.game_pk, daily.pick.batter_id,
                    batter_name=daily.pick.batter_name,
                    date=date, team=daily.pick.team,
                )
                if primary_result is None:
                    daily.result = "unresolved"
                    save_pick(daily, picks_dir)
                    return "unresolved"

                results = [primary_result]
                if daily.double_down:
                    double_result = check_hit(
                        daily.double_down.game_pk, daily.double_down.batter_id,
                        batter_name=daily.double_down.batter_name,
                        date=date, team=daily.double_down.team,
                    )
                    if double_result is None:
                        daily.result = "unresolved"
                        save_pick(daily, picks_dir)
                        return "unresolved"
                    results.append(double_result)

                new_streak = update_streak(results, picks_dir)
                daily.result = "hit" if all(results) else "miss"
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
    4. Post to Bluesky when lock conditions met
    5. Fallback posting if close to first pitch
    6. Doubleheader game 2 re-checks
    7. Next-day lookahead for wake-up time
    8. Result polling after games finish
    """
    from bts.picks import save_pick, load_streak, load_pick
    from bts.posting import format_post, format_skip_post, post_to_bluesky

    sched_config = config.get("scheduler", {})
    private_mode = sched_config.get("private_mode", False)
    if private_mode:
        print("  [PRIVATE MODE] Bluesky posting disabled — picks saved locally only.", file=sys.stderr)
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
    )
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
        save_state(state, picks_dir)

        if result["pick_result"] and result["pick_result"].locked:
            state.pick_locked = True
            state.pick_locked_at = _now_et().isoformat()
            save_state(state, picks_dir)
            print(f"  Pick already locked (game started or previously posted).",
                  file=sys.stderr)

        if result["should_post"] and result["pick_result"] and not result["pick_result"].locked:
            daily = result["pick_result"].daily
            streak = load_streak(picks_dir)
            if private_mode:
                save_pick(daily, picks_dir)
                state.pick_locked = True
                state.pick_locked_at = _now_et().isoformat()
                save_state(state, picks_dir)
                print(f"  [PRIVATE] LOCKED — {daily.pick.batter_name} ({daily.pick.team}) "
                      f"{daily.pick.p_game_hit:.1%} — NOT posted (private mode)", file=sys.stderr)
            else:
                text = format_post(
                    daily.pick.batter_name, daily.pick.team,
                    daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
                    daily.double_down.batter_name if daily.double_down else None,
                    daily.double_down.p_game_hit if daily.double_down else None,
                    daily.double_down.team if daily.double_down else None,
                    daily.double_down.pitcher_name if daily.double_down else None,
                )
                try:
                    uri = post_to_bluesky(text)
                    daily.bluesky_posted = True
                    daily.bluesky_uri = uri
                    save_pick(daily, picks_dir)
                    state.pick_locked = True
                    state.pick_locked_at = _now_et().isoformat()
                    save_state(state, picks_dir)
                    print(f"  LOCKED — Posted to Bluesky: {uri}", file=sys.stderr)
                except Exception as e:
                    print(f"  Bluesky post failed: {e}", file=sys.stderr)

        if state.pick_locked:
            # Run shadow model if enabled (after production pick is resolved)
            if shadow_model_enabled and result.get("pick_result") and result["pick_result"].daily:
                prod_name = result["pick_result"].daily.pick.batter_name
                _run_shadow_prediction(config, date, prod_name)
            print(f"  Pick locked. Stopping lineup checks.", file=sys.stderr)
            break

        # If the earliest game in the slate starts before the next scheduled
        # check, wake up to force-post. Use earliest of primary + double-down
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
            next_checks = [r["time_et"] for r in runs[run_idx + 1:]]
            has_earlier_check = any(t <= fallback_deadline for t in next_checks)

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

                # Force-post current pick (waited to deadline, or past it).
                # Re-run predictions first in case late-arriving lineups
                # changed the top pick since the last scheduled check.
                daily = load_pick(date, picks_dir)
                if daily and not daily.bluesky_posted:
                    daily = _refresh_pick_at_fallback(config, date, daily)
                    if private_mode:
                        state.pick_locked = True
                        state.pick_locked_at = _now_et().isoformat()
                        save_state(state, picks_dir)
                        print(f"  [PRIVATE] FALLBACK LOCKED — {daily.pick.batter_name} — NOT posted", file=sys.stderr)
                    else:
                        print(f"  FALLBACK — posting before game starts.", file=sys.stderr)
                        streak = load_streak(picks_dir)
                        text = format_post(
                            daily.pick.batter_name, daily.pick.team,
                            daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
                            daily.double_down.batter_name if daily.double_down else None,
                            daily.double_down.p_game_hit if daily.double_down else None,
                            daily.double_down.team if daily.double_down else None,
                            daily.double_down.pitcher_name if daily.double_down else None,
                        )
                        try:
                            uri = post_to_bluesky(text)
                            daily.bluesky_posted = True
                            daily.bluesky_uri = uri
                            save_pick(daily, picks_dir)
                            state.pick_locked = True
                            state.pick_locked_at = _now_et().isoformat()
                            save_state(state, picks_dir)
                            print(f"  LOCKED (fallback) — Posted to Bluesky: {uri}",
                                  file=sys.stderr)
                        except Exception as e:
                            print(f"  Bluesky fallback post failed: {e}", file=sys.stderr)

                if state.pick_locked:
                    if shadow_model_enabled and daily:
                        _run_shadow_prediction(config, date, daily.pick.batter_name)
                    print(f"  Pick locked. Stopping lineup checks.", file=sys.stderr)
                    break

    # 5. Fallback — if not yet locked, check for deadline (use earliest of
    # primary + double-down so we never miss the BTS submission window).
    if not state.pick_locked:
        daily = load_pick(date, picks_dir)
        if daily and not daily.bluesky_posted:
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
                daily = _refresh_pick_at_fallback(config, date, daily)
                if private_mode:
                    state.pick_locked = True
                    state.pick_locked_at = _now_et().isoformat()
                    save_state(state, picks_dir)
                    print(f"  [PRIVATE] FINAL FALLBACK LOCKED — {daily.pick.batter_name} — NOT posted", file=sys.stderr)
                else:
                    print(f"  FALLBACK — {fallback_min}min to first pitch, posting on projected data.",
                          file=sys.stderr)
                    streak = load_streak(picks_dir)
                    text = format_post(
                        daily.pick.batter_name, daily.pick.team,
                        daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
                        daily.double_down.batter_name if daily.double_down else None,
                        daily.double_down.p_game_hit if daily.double_down else None,
                        daily.double_down.team if daily.double_down else None,
                        daily.double_down.pitcher_name if daily.double_down else None,
                    )
                    try:
                        uri = post_to_bluesky(text)
                        daily.bluesky_posted = True
                        daily.bluesky_uri = uri
                        save_pick(daily, picks_dir)
                        state.pick_locked = True
                        state.pick_locked_at = _now_et().isoformat()
                        save_state(state, picks_dir)
                    except Exception as e:
                        print(f"  Bluesky fallback post failed: {e}", file=sys.stderr)

        if state.pick_locked and shadow_model_enabled and daily:
            _run_shadow_prediction(config, date, daily.pick.batter_name)

    # 6. Doubleheader game 2 re-checks
    for pk in dh_game2s:
        if any(cs_pk == pk for cs_pk, _ in confirmed_sides):
            continue
        if state.pick_locked:
            break
        print(f"  DH game 2 ({pk}): re-checking every {dh_recheck_min}min...", file=sys.stderr)
        for _ in range(10):
            time.sleep(dh_recheck_min * 60)
            new = count_new_confirmations([pk], confirmed_sides)
            if new > 0:
                print(f"  DH game 2 ({pk}): lineup confirmed.", file=sys.stderr)
                break

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

    # 8. Result polling (start 10 min after game start, check for hits mid-game)
    if state.pick_locked:
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

    # End-of-day calibration drift check. Pure observation — never modifies
    # picks. Wrapped in try/except so an alerting bug can't break the
    # scheduler's pick lifecycle. Sends Bluesky DM only on CRITICAL alerts.
    cal_config = config.get("calibration_check", {})
    if cal_config.get("enabled", True):
        from bts.calibration_check import run_calibration_check
        dm_recipient = config.get("bluesky", {}).get("dm_recipient")
        try:
            run_calibration_check(picks_dir=picks_dir, dm_recipient=dm_recipient)
        except Exception as e:
            print(f"  calibration_check: unexpected error (suppressed): {e}", file=sys.stderr)

    write_heartbeat(heartbeat_path, state=HeartbeatState.IDLE_END_OF_DAY)
    notify_watchdog()

    # Stay alive until tomorrow's scheduled wake — returning here would cause
    # systemd Restart=always to re-launch within 30s and cycle through the
    # short-circuit post-lock branches every ~3 min overnight.
    _idle_until_next_wakeup(state.next_wakeup, heartbeat_path)
