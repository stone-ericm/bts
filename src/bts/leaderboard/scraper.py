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

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Literal

import httpx

from bts.leaderboard.endpoints import (
    LEADERBOARD_URL_TEMPLATE,
    LEADERBOARD_ROUND_URL_TEMPLATE,
    USER_PROFILE_URL_TEMPLATE,
    ROUNDS_URL,
    RANKS_TYPE_BY_TAB,
    USER_AGENT,
)
from bts.leaderboard.models import LeaderboardRow, PickRow, SeasonStats
from bts.leaderboard.ratelimit import rate_limited
from bts.leaderboard.storage import (
    write_leaderboard_snapshot, append_user_picks, write_season_stats,
)

log = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL_S = 2.0

TabName = Literal["active_streak", "all_season", "all_time", "yesterday"]


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
    r = httpx.get(
        url, cookies=cookies, timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    r.raise_for_status()
    return r.json()


@rate_limited(min_interval_s=DEFAULT_MIN_INTERVAL_S)
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
    return parse_leaderboard_response(body, tab=tab, captured_at=datetime.utcnow())


@rate_limited(min_interval_s=DEFAULT_MIN_INTERVAL_S)
def scrape_user_profile(
    user_id: int, cookies: dict[str, str], xsid: str, lookups: StaticLookups,
) -> tuple[list[PickRow], SeasonStats]:
    url = USER_PROFILE_URL_TEMPLATE.format(user_id=user_id, xsid=xsid)
    body = _get_json(url, cookies=cookies)
    return parse_user_profile_response(
        body, captured_at=datetime.utcnow(), user_id_unused=user_id, lookups=lookups,
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


def run(
    cookies: dict[str, str], xsid: str, output_dir: Path, top_n: int = 100,
    tabs: tuple[TabName, ...] = ("active_streak", "all_season", "all_time", "yesterday"),
    today: date | None = None,
) -> None:
    """Full daily scrape: 4 leaderboards + per-user profiles for top_n users.

    Failures during per-user iteration are logged but don't abort the run.
    """
    today = today or date.today()
    snapshot_path = output_dir / "leaderboard_snapshots" / f"{today.isoformat()}.parquet"
    stats_path = output_dir / "season_stats" / f"{today.isoformat()}.parquet"

    log.info("fetching static lookups (rounds + players + units + squads)")
    lookups = scrape_static_lookups(cookies)
    log.info(
        f"  rounds: {len(lookups.rounds)}, players: {len(lookups.players)}, "
        f"units: {len(lookups.units)}, squads: {len(lookups.squads)}"
    )

    yesterday_rid = _yesterday_round_id(lookups.rounds, today)
    if "yesterday" in tabs and yesterday_rid is None:
        log.warning(f"no rounds_lookup entry for {today.toordinal() - 1}; skipping 'yesterday' tab")

    all_rows: list[LeaderboardRow] = []
    tracked: dict[str, int] = {}  # username -> bts_user_id, deduped across tabs

    for tab in tabs:
        try:
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
        except httpx.HTTPError as e:
            log.exception(f"failed to scrape {tab}: {e}")
            continue
        rows = parse_leaderboard_response(raw, tab=tab, captured_at=datetime.utcnow())
        all_rows.extend(rows[:top_n])
        for entry in raw.get("success", {}).get("ranks", []):
            tracked.setdefault(entry["username"], int(entry["userId"]))

    write_leaderboard_snapshot(snapshot_path, all_rows)
    log.info(f"wrote {len(all_rows)} leaderboard rows to {snapshot_path}")

    season_rows: list[SeasonStats] = []
    for username, user_id in sorted(tracked.items()):
        try:
            picks, stats = scrape_user_profile(user_id, cookies=cookies, xsid=xsid, lookups=lookups)
        except httpx.HTTPError as e:
            log.warning(f"skipping user {username} (id={user_id}): {e}")
            continue
        # backfill username on stats (parser doesn't know it from API response)
        stats = stats.model_copy(update={"username": username})
        user_path = output_dir / "user_picks" / f"{username}.parquet"
        append_user_picks(user_path, picks)
        season_rows.append(stats)

    write_season_stats(stats_path, season_rows)
    log.info(f"wrote {len(season_rows)} season-stats rows to {stats_path}")
