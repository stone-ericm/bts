# src/bts/leaderboard/scraper.py
"""Scraping orchestration for the BTS leaderboard watcher.

Each top-level scrape function takes session cookies + a freshly minted
xSid token and returns typed model rows. HTTP errors propagate as exceptions;
the orchestrator (`run`) catches per-call failures so one bad user doesn't
abort the whole scrape.

Static lookups (rounds, players, units, squads) come from the BTS app's
static JSON files; the scraper fetches them once per run for resolution
of names + teams + opponents on PickRow.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Literal

import httpx

from bts.leaderboard.endpoints import (
    LEADERBOARD_URL_TEMPLATE,
    LEADERBOARD_ROUND_URL_TEMPLATE,
    USER_PROFILE_URL_TEMPLATE,
    ROUNDS_URL,
    RANKS_TYPE_BY_TAB,
    browser_headers,
)
from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats
from bts.leaderboard.ratelimit import rate_limited, next_gap
from bts.leaderboard.storage import (
    write_leaderboard_snapshot, append_user_picks, write_season_stats,
    safe_filename_component,
)

log = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL_S = 2.0
# Human-scale jitter: a real person clicking through the board doesn't fetch on
# a fixed 2.000s metronome. Draw each gap from [MIN, MIN+JITTER].
DEFAULT_JITTER_S = 2.5
# HTTP statuses that mean "you are being throttled / blocked" — a careful human
# stops immediately rather than hammering, which is what escalates to a ban.
RATE_LIMIT_STATUSES = frozenset({403, 429})

TabName = Literal["active_streak", "all_season", "all_time", "yesterday"]


class RateLimitedError(Exception):
    """Raised when MLB signals throttling/blocking (403/429). Aborts the scrape
    so we back off instead of hammering the account into a harder block."""

    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        self.url = url
        super().__init__(f"rate-limited: HTTP {status_code} on {url}")


# Module-level RNG for inter-page jitter (not seeded — variance IS the point;
# tests monkeypatch _deep_page_pause, so determinism isn't needed here).
_PAGE_RNG = random.Random()


@dataclass
class StaticLookups:
    """Static-JSON lookups, fetched once per scrape run for name resolution."""
    rounds: dict[int, date] = field(default_factory=dict)
    players: dict[int, dict] = field(default_factory=dict)   # bts_player_id -> player record
    units: dict[int, dict] = field(default_factory=dict)     # unit_id -> unit record
    squads: dict[int, dict] = field(default_factory=dict)    # squad_id -> squad record

    def player(self, bts_player_id: int) -> dict | None:
        return self.players.get(bts_player_id)

    def unit(self, unit_id: int) -> dict | None:
        return self.units.get(unit_id)

    def squad_abbrev(self, squad_id: int | None) -> str | None:
        if squad_id is None:
            return None
        s = self.squads.get(squad_id)
        return s.get("abbreviation") if s else None


def parse_rounds_lookup(body: dict) -> dict[int, date]:
    """Parse rounds.json into round_id -> date dict."""
    out: dict[int, date] = {}
    for r in body.get("rounds", []):
        # date format: '2026-03-25T08:00:00-04:00'
        d_str = r["date"][:10]
        out[int(r["id"])] = date.fromisoformat(d_str)
    return out


def parse_leaderboard_response(
    body: dict, tab: TabName, captured_at: datetime,
) -> list[LeaderboardRow]:
    """Parse a leaderboard JSON body into typed rows."""
    raw_rows = body.get("success", {}).get("ranks", [])
    out: list[LeaderboardRow] = []
    for r in raw_rows:
        out.append(LeaderboardRow(
            captured_at=captured_at,
            tab=tab,
            rank=int(r["rank"]),
            username=str(r["username"]),
            streak=int(r["activeStreak"]) if r.get("activeStreak") is not None else (
                int(r["streak"]) if r.get("streak") is not None else None
            ),
            hits_today=None,  # 'yesterday' tab doesn't expose explicit hits_today in the rank list
            user_id=int(r["userId"]) if r.get("userId") is not None else None,
        ))
    return out


def parse_user_profile_response(
    body: dict,
    captured_at: datetime,
    user_id_unused: int,
    lookups: StaticLookups,
    username: str = "unknown",
) -> tuple[list[PickRow], SeasonStats]:
    """Parse the combined profile response into (picks_list, season_stats).

    The profile endpoint returns `predictions[]` (one entry per round) where
    each entry has a `roundPredictions[]` list (1 or 2 entries: primary + DD).
    We emit one PickRow per roundPrediction.

    username is not in the profile API response; pass it if known or leave
    as "unknown" — the orchestrator (run()) backfills it via model_copy.
    """
    success = body.get("success", {})
    # API returns None for these fields on users with no picks (e.g. users
    # appearing on All-Time leaderboard for past streaks but inactive this season).
    # Coerce None -> 0 before pydantic validation.
    stats = SeasonStats(
        captured_at=captured_at,
        username=username,
        best_streak=int(success.get("seasonBestStreak") or 0),
        active_streak=int(success.get("activeStreak") or 0),
        pick_accuracy_pct=float(success.get("accuracy") or 0),
    )
    picks: list[PickRow] = []
    for pred in success.get("predictions", []):
        round_id = int(pred["roundId"])
        pick_date = lookups.rounds.get(round_id)
        if pick_date is None:
            log.warning(f"no rounds_lookup entry for round_id={round_id}; skipping pick")
            continue
        # API may return None for streak / atBats / hits on yet-to-resolve picks
        # or for users with no recent activity. Coerce None -> 0 throughout.
        streak_after = int(pred.get("streak") or 0)
        for rp in pred.get("roundPredictions", []):
            unit_id = int(rp.get("unitId") or 0)
            bts_player_id = int(rp.get("playerId") or 0)
            player = lookups.player(bts_player_id) or {}
            unit = lookups.unit(unit_id) or {}
            player_squad_id = player.get("squadId")
            home_squad_id = unit.get("homeSquadId")
            away_squad_id = unit.get("awaySquadId")
            home_or_away: str | None = None
            opponent_squad_id: int | None = None
            if player_squad_id is not None and home_squad_id is not None:
                if player_squad_id == home_squad_id:
                    home_or_away = "home"
                    opponent_squad_id = away_squad_id
                elif player_squad_id == away_squad_id:
                    home_or_away = "away"
                    opponent_squad_id = home_squad_id
            picks.append(PickRow(
                captured_at=captured_at,
                round_id=round_id,
                pick_date=pick_date,
                pick_number=int(rp.get("number") or 1),
                unit_id=unit_id,
                bts_player_id=bts_player_id,
                result=str(rp.get("result") or ""),
                at_bats=int(rp.get("atBats") or 0),
                hits=int(rp.get("hits") or 0),
                streak_after=streak_after,
                batter_id=int(player["feedId"]) if player.get("feedId") is not None else None,
                batter_name=player.get("name"),
                batter_team=lookups.squad_abbrev(player_squad_id),
                opponent_team=lookups.squad_abbrev(opponent_squad_id),
                home_or_away=home_or_away,  # type: ignore[arg-type]  # validated by Literal
            ))
    return picks, stats


# --- HTTP wrappers ---

def _get_json(url: str, cookies: dict[str, str], timeout: float = 30.0) -> dict:
    r = httpx.get(url, cookies=cookies, timeout=timeout, headers=browser_headers())
    if r.status_code in RATE_LIMIT_STATUSES:
        raise RateLimitedError(r.status_code, url)
    r.raise_for_status()
    return r.json()


@rate_limited(min_interval_s=DEFAULT_MIN_INTERVAL_S, jitter_s=DEFAULT_JITTER_S)
def scrape_leaderboard(
    tab: TabName, cookies: dict[str, str], xsid: str,
    season: int = 2026, page: int = 1, limit: int = 100,
    round_id: int | None = None,
) -> list[LeaderboardRow]:
    """Fetch + parse one leaderboard tab. For 'yesterday', supply round_id."""
    ranks_type = RANKS_TYPE_BY_TAB[tab]
    if tab == "yesterday":
        if round_id is None:
            raise ValueError("round_id is required for the 'yesterday' tab")
        url = LEADERBOARD_ROUND_URL_TEMPLATE.format(
            round_id=round_id, page=page, limit=limit,
            ranks_type=ranks_type, xsid=xsid,
        )
    else:
        url = LEADERBOARD_URL_TEMPLATE.format(
            season=season, page=page, limit=limit,
            ranks_type=ranks_type, xsid=xsid,
        )
    body = _get_json(url, cookies=cookies)
    return parse_leaderboard_response(body, tab=tab, captured_at=datetime.now(timezone.utc).replace(tzinfo=None))


@rate_limited(min_interval_s=DEFAULT_MIN_INTERVAL_S, jitter_s=DEFAULT_JITTER_S)
def scrape_user_profile(
    user_id: int, cookies: dict[str, str], xsid: str, lookups: StaticLookups,
) -> tuple[list[PickRow], SeasonStats]:
    url = USER_PROFILE_URL_TEMPLATE.format(user_id=user_id, xsid=xsid)
    body = _get_json(url, cookies=cookies)
    return parse_user_profile_response(
        body, captured_at=datetime.now(timezone.utc).replace(tzinfo=None), user_id_unused=user_id, lookups=lookups,
    )


def scrape_static_lookups(cookies: dict[str, str]) -> StaticLookups:
    """Fetch all four static JSON files; build name-resolution lookups."""
    base = "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json"
    rounds_body = _get_json(ROUNDS_URL, cookies=cookies)
    players_body = _get_json(f"{base}/players.json", cookies=cookies)
    units_body = _get_json(f"{base}/units.json", cookies=cookies)
    squads_body = _get_json(f"{base}/squads.json", cookies=cookies)
    return StaticLookups(
        rounds=parse_rounds_lookup(rounds_body),
        players={int(p["id"]): p for p in players_body.get("players", [])},
        units={int(u["id"]): u for u in units_body.get("units", [])},
        squads={int(s["id"]): s for s in squads_body.get("squads", [])},
    )


def _yesterday_round_id(rounds_lookup: dict[int, date], today: date) -> int | None:
    """Find the round_id whose date == today - 1d. Returns None if not present."""
    target = date.fromordinal(today.toordinal() - 1)
    for rid, d in rounds_lookup.items():
        if d == target:
            return rid
    return None


def _deep_page_pause() -> None:
    """Jittered human-scale gap between deep leaderboard pages (test seam)."""
    time.sleep(next_gap(DEFAULT_MIN_INTERVAL_S, DEFAULT_JITTER_S, _PAGE_RNG))


def _scrape_active_streak_deep(
    cookies: dict[str, str],
    xsid: str,
    season: int,
    deep_limit: int,
    deep_max_pages: int,
    deep_min_streak: int,
) -> tuple[list[LeaderboardRow], list[dict], bool]:
    """Paginate the season active-streak board well past the top-100.

    Verified 2026-07-03: the leaderboard endpoint honors page/limit far beyond
    rank 100 (allParticipantsCount ~57k; limit=300 accepted). Rank ties make
    page boundaries overlap or shift between requests, so pages can repeat or
    skip a user near a boundary — rows are deduped by userId across pages and
    the result is an approximation of the board, not a census.

    Stops when: a page comes back short (end of board), the page's minimum
    streak drops below `deep_min_streak` (the long uninformative tail), or
    `deep_max_pages` is hit (runaway backstop) — these are CLEAN stops
    (`complete=True`). A transient page fetch/parse error stops deep paging but
    KEEPS the rows already collected and reports `complete=False` so callers
    don't mistake a truncated board for the whole field. A `RateLimitedError`
    propagates (it must abort the whole scrape, not degrade to partial).

    Returns (parsed snapshot rows, deduped raw rank entries, complete flag).
    """
    seen: set[int] = set()
    parsed: list[LeaderboardRow] = []
    raw_entries: list[dict] = []
    complete = True
    for page in range(1, deep_max_pages + 1):
        if page > 1:
            _deep_page_pause()
        url = LEADERBOARD_URL_TEMPLATE.format(
            season=season, page=page, limit=deep_limit,
            ranks_type=RANKS_TYPE_BY_TAB["active_streak"], xsid=xsid,
        )
        try:
            body = _get_json(url, cookies=cookies)
        except RateLimitedError:
            raise  # throttled — abort the whole scrape, don't keep hammering
        except Exception as e:  # noqa: BLE001 - keep partial board on transient failure
            log.warning(f"deep active_streak page {page} failed; "
                        f"keeping {len(parsed)} rows (INCOMPLETE): {e}")
            complete = False
            break
        ranks = body.get("success", {}).get("ranks", [])
        fresh: list[dict] = []
        for r in ranks:
            uid = r.get("userId")
            if uid is not None:
                if int(uid) in seen:
                    continue
                seen.add(int(uid))
            fresh.append(r)
        parsed.extend(parse_leaderboard_response(
            {"success": {"ranks": fresh}}, tab="active_streak",
            captured_at=datetime.now(timezone.utc).replace(tzinfo=None)))
        raw_entries.extend(fresh)
        if len(ranks) < deep_limit:
            break
        page_streaks = [r.get("activeStreak", r.get("streak")) for r in ranks]
        page_streaks = [s for s in page_streaks if s is not None]
        if page_streaks and min(page_streaks) < deep_min_streak:
            break
    return parsed, raw_entries, complete


def _write_scrape_status(status_path: Path, payload: dict) -> None:
    """Persist scrape completeness/throttle metadata (never raises)."""
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        prior = {}
        if status_path.exists():
            try:
                prior = json.loads(status_path.read_text())
            except (json.JSONDecodeError, OSError):
                prior = {}
        merged = {**prior, **payload}
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.rename(status_path)
    except OSError as e:
        log.warning(f"could not write scrape_status.json: {e}")


def _rate_limit_alert(status_path: Path, dm_recipient: str | None,
                      err: RateLimitedError, now_iso: str,
                      cooldown_hours: float = 6.0) -> bool:
    """DM Eric that MLB threw a 403/429 so he can back off before a hard block.
    Throttled via last_alert_at in scrape_status.json. Returns whether DM'd."""
    from datetime import datetime as _dt, timedelta
    last = None
    if status_path.exists():
        try:
            last = json.loads(status_path.read_text()).get("last_alert_at")
        except (json.JSONDecodeError, OSError):
            last = None
    should = True
    if last:
        try:
            should = (_dt.fromisoformat(now_iso) - _dt.fromisoformat(last)) >= timedelta(hours=cooldown_hours)
        except ValueError:
            should = True
    sent = False
    if should and dm_recipient:
        try:
            from bts.dm import send_dm
            send_dm(dm_recipient, f"⚠️ BTS leaderboard scrape hit HTTP "
                    f"{err.status_code} (throttled) — deep scraping aborted and "
                    f"backed off. Check before it escalates to a block.")
            sent = True
        except Exception as dm_err:  # noqa: BLE001
            log.warning(f"rate-limit DM failed: {dm_err}")
    return sent


def run(
    cookies: dict[str, str], xsid: str, output_dir: Path, top_n: int = 100,
    tabs: tuple[TabName, ...] = ("active_streak", "all_season", "all_time", "yesterday"),
    today: date | None = None,
    deep_limit: int = 300,
    deep_max_pages: int = 100,
    deep_min_streak: int = 3,
    profile_top_n: int = 300,
    dm_recipient: str | None = None,
) -> None:
    """Full daily scrape: leaderboards + per-user profiles.

    Since 2026-07-03 the active_streak tab is paginated DEEP (down to streaks
    of `deep_min_streak`, ~10-30k rows) so users stay visible in snapshots
    after a reset instead of vanishing off the top-100 cliff — the censoring
    fix for field-level analyses. The full deep board lands in the SNAPSHOT;
    pick-log PROFILES are capped at `profile_top_n` TOTAL (prioritizing the
    deep active board, then the other tabs) to bound the authenticated
    footprint. `deep_max_pages=0` restores the legacy single-page behavior.

    Throttle discipline: an HTTP 403/429 anywhere raises RateLimitedError,
    which ABORTS the scrape (a careful client backs off rather than hammering
    the account into a hard block), records `rate_limited` in scrape_status.json,
    and DMs Eric (throttled). Per-user/per-tab transient errors are logged and
    skipped as before. Completeness of the deep board is recorded so downstream
    analyses can tell a truncated snapshot from the whole field.
    """
    today = today or date.today()
    snapshot_path = output_dir / "leaderboard_snapshots" / f"{today.isoformat()}.parquet"
    stats_path = output_dir / "season_stats" / f"{today.isoformat()}.parquet"
    status_path = output_dir / "scrape_status.json"
    now_iso = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

    all_rows: list[LeaderboardRow] = []
    season_rows: list[SeasonStats] = []
    tracked: dict[str, int] = {}  # username -> bts_user_id, deduped across tabs
    active_complete = True
    rate_limited = False
    alert_sent = False

    try:
        log.info("fetching static lookups (rounds + players + units + squads)")
        lookups = scrape_static_lookups(cookies)
        log.info(
            f"  rounds: {len(lookups.rounds)}, players: {len(lookups.players)}, "
            f"units: {len(lookups.units)}, squads: {len(lookups.squads)}"
        )

        yesterday_rid = _yesterday_round_id(lookups.rounds, today)
        if "yesterday" in tabs and yesterday_rid is None:
            log.warning(f"no rounds_lookup entry for {today.toordinal() - 1}; skipping 'yesterday' tab")

        for tab in tabs:
            try:
                if tab == "active_streak" and deep_max_pages > 0:
                    deep_rows, deep_entries, active_complete = _scrape_active_streak_deep(
                        cookies=cookies, xsid=xsid, season=today.year,
                        deep_limit=deep_limit, deep_max_pages=deep_max_pages,
                        deep_min_streak=deep_min_streak,
                    )
                    all_rows.extend(deep_rows)
                    for entry in deep_entries[:max(top_n, profile_top_n)]:
                        username, user_id = entry.get("username"), entry.get("userId")
                        if username and user_id is not None:
                            tracked.setdefault(str(username), int(user_id))
                    continue
                if tab == "yesterday":
                    if yesterday_rid is None:
                        continue
                    url = LEADERBOARD_ROUND_URL_TEMPLATE.format(
                        round_id=yesterday_rid, page=1, limit=top_n,
                        ranks_type=RANKS_TYPE_BY_TAB[tab], xsid=xsid,
                    )
                else:
                    url = LEADERBOARD_URL_TEMPLATE.format(
                        season=today.year, page=1, limit=top_n,
                        ranks_type=RANKS_TYPE_BY_TAB[tab], xsid=xsid,
                    )
                raw = _get_json(url, cookies=cookies)
                rows = parse_leaderboard_response(raw, tab=tab, captured_at=datetime.now(timezone.utc).replace(tzinfo=None))
                all_rows.extend(rows[:top_n])
                for entry in raw.get("success", {}).get("ranks", []):
                    tracked.setdefault(entry["username"], int(entry["userId"]))
            except RateLimitedError:
                raise  # abort the whole scrape; don't try more tabs
            except httpx.HTTPError as e:
                log.exception(f"failed to scrape {tab}: {e}")
                continue
            except Exception as e:
                # Schema drift (JSON/key/validation) on one tab must not abort the
                # whole scrape — matches the per-user resilience below (audit G).
                log.exception(f"failed to parse {tab} (schema drift?): {e}")
                continue

        write_leaderboard_snapshot(snapshot_path, all_rows)
        log.info(f"wrote {len(all_rows)} leaderboard rows to {snapshot_path} "
                 f"(active_streak complete={active_complete})")

        # Cap total profile fetches at profile_top_n to bound the authenticated
        # footprint (Eric: be careful). tracked is insertion-ordered with the
        # deep ACTIVE board first (processed first in the tab loop), so the cap
        # keeps the useful pick-logs — active players — and sheds the tail of the
        # all_season/all_time/yesterday top-100s. The full deep board still lands
        # in the SNAPSHOT (that's the censoring fix); only pick-log fetches are
        # capped. Shuffle only the capped set — a real user doesn't page profiles
        # in strict rank order at a fixed cadence.
        profile_order = list(tracked.items())[:profile_top_n]
        random.shuffle(profile_order)
        for username, user_id in profile_order:
            try:
                picks, stats = scrape_user_profile(user_id, cookies=cookies, xsid=xsid, lookups=lookups)
                # backfill username on stats (parser doesn't know it from API response)
                stats = stats.model_copy(update={"username": username})
                # Sanitize the arbitrary public username before using it as a filename
                # (path-traversal write guard, audit G).
                user_path = output_dir / "user_picks" / f"{safe_filename_component(username)}.parquet"
                append_user_picks(user_path, picks)
                season_rows.append(stats)
            except RateLimitedError:
                raise  # abort now; season_stats is dropped and retried next run
            except httpx.HTTPError as e:
                log.warning(f"skipping user {username} (id={user_id}): {e}")
                continue
            except Exception as e:
                log.warning(f"skipping user {username} (id={user_id}) on parse/write error: {e}")
                continue

        write_season_stats(stats_path, season_rows)
        log.info(f"wrote {len(season_rows)} season-stats rows to {stats_path}")
    except RateLimitedError as e:
        rate_limited = True
        active_complete = False
        log.error(f"scrape ABORTED (throttled): {e}")
        # Only consume the alert cooldown when a DM actually went out, so a run
        # with a missing/failed recipient can still alert on the next attempt.
        alert_sent = _rate_limit_alert(status_path, dm_recipient, e, now_iso)

    _write_scrape_status(status_path, {
        "last_run_utc": now_iso,
        "date": today.isoformat(),
        "rate_limited": rate_limited,
        "active_streak_complete": active_complete,
        "n_leaderboard_rows": len(all_rows),
        "n_profiles": len(season_rows),
        **({"last_alert_at": now_iso} if rate_limited and alert_sent else {}),
    })
