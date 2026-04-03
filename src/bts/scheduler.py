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


def _game_time_et(game: dict) -> datetime:
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
        et = _game_time_et(g)
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
            team_g.sort(key=lambda x: _game_time_et(x))
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

    earliest = min(_game_time_et(g) for g in games)
    early_wake = earliest - timedelta(minutes=early_buffer_min)

    if early_wake < today_et:
        return early_wake

    return today_et


def check_confirmed_lineups(game_pks: list[int]) -> dict[int, bool]:
    """Check which games have confirmed lineups posted.

    A lineup is confirmed if the boxscore has players with battingOrder set.
    Returns {game_pk: has_confirmed_lineup}.
    """
    results = {}
    for pk in game_pks:
        try:
            resp = json.loads(retry_urlopen(
                f"{API_BASE}/api/v1.1/game/{pk}/feed/live",
                timeout=15,
            ).read())
            has_lineup = False
            for side in ("away", "home"):
                players = resp["liveData"]["boxscore"]["teams"][side]["players"]
                for pid, pdata in players.items():
                    if pdata.get("battingOrder"):
                        has_lineup = True
                        break
                if has_lineup:
                    break
            results[pk] = has_lineup
        except Exception:
            results[pk] = False

    return results


@dataclass
class SchedulerState:
    """Daily scheduler state, persisted to JSON."""
    date: str
    schedule_fetched_at: str
    games: list[dict]  # [{game_pk, game_time_et, lineup_confirmed, is_doubleheader_game2}]
    confirmed_game_pks: list[int]
    runs_completed: list[dict]  # [{time, new_lineups, skipped}]
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
    previously_confirmed: set[int],
) -> int:
    """Check for new lineup confirmations since last check.

    Updates previously_confirmed in place. Returns count of newly confirmed games.
    """
    statuses = check_confirmed_lineups(game_pks)
    new_count = 0
    for pk, confirmed in statuses.items():
        if confirmed and pk not in previously_confirmed:
            previously_confirmed.add(pk)
            new_count += 1
    return new_count


def _now_et() -> datetime:
    """Current time in ET. Extracted for testability."""
    return datetime.now(ET)


def run_single_check(
    date: str,
    all_game_pks: list[int],
    confirmed_game_pks: set[int],
    config: dict,
    early_lock_gap: float,
) -> dict:
    """Run a single lineup check cycle.

    1. Check for new confirmed lineups.
    2. If new confirmations, run prediction cascade.
    3. Evaluate should_lock().

    Returns {"skipped": bool, "new_lineups": int, "should_post": bool,
             "pick_result": PickResult | None}.
    """
    from bts.orchestrator import run_and_pick
    from bts.picks import save_pick
    from bts.strategy import should_lock

    new_count = count_new_confirmations(all_game_pks, confirmed_game_pks)

    if new_count == 0:
        return {"skipped": True, "new_lineups": 0, "should_post": False, "pick_result": None}

    print(f"  {new_count} new confirmed lineup(s). Running predictions...", file=sys.stderr)

    predictions, pick_result, tier = run_and_pick(config, date)

    if predictions is None or pick_result is None:
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result}

    if pick_result.locked:
        return {"skipped": False, "new_lineups": new_count, "should_post": False,
                "pick_result": pick_result}

    # Save candidate pick
    picks_dir = Path(config["orchestrator"]["picks_dir"])
    save_pick(pick_result.daily, picks_dir)

    # Check if we should lock
    pick_data = {
        "p_game_hit": pick_result.daily.pick.p_game_hit,
        "projected_lineup": pick_result.daily.pick.projected_lineup,
        "game_pk": pick_result.daily.pick.game_pk,
    }
    all_pick_data = []
    for _, row in predictions.iterrows():
        if row.get("p_game_hit") and row["p_game_hit"] == row["p_game_hit"]:  # not NaN
            all_pick_data.append({
                "p_game_hit": float(row["p_game_hit"]),
                "projected_lineup": "PROJECTED" in str(row.get("flags", "")),
                "game_pk": int(row["game_pk"]),
            })

    do_post = should_lock(pick_data, all_pick_data, early_lock_gap)

    return {"skipped": False, "new_lineups": new_count, "should_post": do_post,
            "pick_result": pick_result}


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


def run_result_polling(
    game_pk: int,
    date: str,
    picks_dir: Path,
    poll_interval_min: int = 15,
    cap_hour_et: int = 5,
) -> str:
    """Poll a game until it reaches a terminal state.

    Returns the final status: "final", "suspended", or "unresolved".
    On "final", runs check-results logic to update streak.
    """
    from bts.picks import load_pick, check_hit, update_streak, save_pick

    while True:
        now = _now_et()
        if now.hour >= cap_hour_et and now.hour < 10:
            # Past the cap — give up
            print(f"  Result polling capped at {cap_hour_et}am ET. Flagging as unresolved.",
                  file=sys.stderr)
            daily = load_pick(date, picks_dir)
            if daily:
                daily.result = "unresolved"
                save_pick(daily, picks_dir)
            return "unresolved"

        status = poll_game_result(game_pk)
        print(f"  [{now.strftime('%H:%M ET')}] Game {game_pk}: {status}", file=sys.stderr)

        if status == "final":
            # Run check-results logic
            daily = load_pick(date, picks_dir)
            if daily:
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
                    if double_result is not None:
                        results.append(double_result)

                update_streak(results, picks_dir)
                daily.result = "hit" if all(results) else "miss"
                save_pick(daily, picks_dir)
                print(f"  Result: {daily.result}. Streak updated.", file=sys.stderr)
            return "final"

        if status == "suspended":
            daily = load_pick(date, picks_dir)
            if daily:
                daily.result = "suspended"
                save_pick(daily, picks_dir)
            return "suspended"

        # Still live — wait and retry
        time.sleep(poll_interval_min * 60)


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
    offset_min = sched_config.get("lineup_check_offset_min", 45)
    cluster_min = sched_config.get("cluster_min", 10)
    dh_recheck_min = sched_config.get("doubleheader_recheck_min", 15)
    early_lock_gap = sched_config.get("early_lock_gap", 0.03)
    poll_interval_min = sched_config.get("results_poll_interval_min", 15)
    cap_hour_et = sched_config.get("results_cap_hour_et", 5)
    picks_dir = Path(config["orchestrator"]["picks_dir"])

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
    confirmed_pks: set[int] = set()
    state = SchedulerState(
        date=date,
        schedule_fetched_at=_now_et().isoformat(),
        games=[{
            "game_pk": g["gamePk"],
            "game_time_et": _game_time_et(g).isoformat(),
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
            wait_secs = (target - now).total_seconds()
            print(f"  Sleeping until {target.strftime('%H:%M ET')} "
                  f"({wait_secs / 60:.0f} min)...", file=sys.stderr)
            time.sleep(wait_secs)

        now = _now_et()
        if now < target:
            continue

        print(f"\n[{_now_et().strftime('%H:%M ET')}] Running lineup check...", file=sys.stderr)
        result = run_single_check(
            date=date,
            all_game_pks=all_game_pks,
            confirmed_game_pks=confirmed_pks,
            config=config,
            early_lock_gap=early_lock_gap,
        )

        state.runs_completed.append({
            "time": _now_et().isoformat(),
            "new_lineups": result["new_lineups"],
            "skipped": result["skipped"],
        })
        state.confirmed_game_pks = list(confirmed_pks)
        for g in state.games:
            g["lineup_confirmed"] = g["game_pk"] in confirmed_pks
        save_state(state, picks_dir)

        if result["should_post"] and result["pick_result"] and not result["pick_result"].locked:
            daily = result["pick_result"].daily
            streak = load_streak(picks_dir)
            text = format_post(
                daily.pick.batter_name, daily.pick.team,
                daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
                daily.double_down.batter_name if daily.double_down else None,
                daily.double_down.p_game_hit if daily.double_down else None,
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
            print(f"  Pick locked. Stopping lineup checks.", file=sys.stderr)
            break

    # 5. Fallback — if not yet locked, check for deadline
    if not state.pick_locked:
        daily = load_pick(date, picks_dir)
        if daily and not daily.bluesky_posted:
            game_et = datetime.fromisoformat(daily.pick.game_time).astimezone(ET)
            now = _now_et()
            mins_to_game = (game_et - now).total_seconds() / 60
            if mins_to_game <= 15:
                print(f"  FALLBACK — 15min to first pitch, posting on projected data.",
                      file=sys.stderr)
                streak = load_streak(picks_dir)
                text = format_post(
                    daily.pick.batter_name, daily.pick.team,
                    daily.pick.pitcher_name, daily.pick.p_game_hit, streak,
                    daily.double_down.batter_name if daily.double_down else None,
                    daily.double_down.p_game_hit if daily.double_down else None,
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

    # 6. Doubleheader game 2 re-checks
    for pk in dh_game2s:
        if pk in confirmed_pks:
            continue
        if state.pick_locked:
            break
        print(f"  DH game 2 ({pk}): re-checking every {dh_recheck_min}min...", file=sys.stderr)
        for _ in range(10):
            time.sleep(dh_recheck_min * 60)
            new = count_new_confirmations([pk], confirmed_pks)
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

    # 8. Result polling (wait until 1am ET, then poll)
    if state.pick_locked:
        daily = load_pick(date, picks_dir)
        if daily and daily.result is None:
            now = _now_et()
            target_1am = now.replace(hour=1, minute=0, second=0, microsecond=0)
            if now.hour >= 1:
                target_1am += timedelta(days=1)
            wait = (target_1am - now).total_seconds()
            if wait > 0:
                print(f"  Waiting until 1am ET for result check ({wait / 3600:.1f}h)...",
                      file=sys.stderr)
                time.sleep(wait)

            game_pk = daily.pick.game_pk
            status = run_result_polling(
                game_pk, date, picks_dir,
                poll_interval_min=poll_interval_min,
                cap_hour_et=cap_hour_et,
            )
            state.result_status = status
            save_state(state, picks_dir)
            print(f"  Day complete. Result: {status}", file=sys.stderr)
