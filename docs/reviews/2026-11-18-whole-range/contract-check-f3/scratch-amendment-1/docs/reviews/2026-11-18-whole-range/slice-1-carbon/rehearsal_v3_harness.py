"""Self-contained end-to-end rehearsal harness (v3) for the 2027 season-start first days.

Drives the real ``bts.scheduler.run_day`` and ``bts.cli.cli`` commands (plus
``scripts/cron-setup-hetzner.sh``) against an isolated data directory seeded from
``BASESHA = 619abda7`` writers when available (with an identical schema fallback
when copied standalone into ``tests/``), captured transports (DM, Bluesky post/reply,
MLB Stats API, MLB BTS Contest API, predictions), deterministic disk usage, a pinned
event-driven ET clock (including function-local ``from datetime import datetime``),
and a hard socket block.

Regression-test hygiene:
- No module-level import of ``bts.model.predict`` or ``lightgbm``.
- No process-wide state (cwd, env, caches, mocks) left modified on exit.
- No network sockets permitted while active.
"""
from __future__ import annotations

import collections
import contextlib
import datetime as _dt_mod
import gzip
import io
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd

_REAL_DATETIME = _dt_mod.datetime
_REAL_DATE = _dt_mod.date
_REAL_TIMEDELTA = _dt_mod.timedelta
_REAL_TIME = _dt_mod.time
_REAL_TIMEZONE = _dt_mod.timezone
_DiskUsage = collections.namedtuple("usage", ["total", "used", "free"])

ET = ZoneInfo("America/New_York")
UTC = _REAL_TIMEZONE.utc

REPO_ROOT = Path(__file__).resolve().parents[2]
if not (REPO_ROOT / "src" / "bts").exists():
    REPO_ROOT = Path(__file__).resolve().parents[1]

BASESHA_SRC = Path(__file__).resolve().parent / "basesha" / "src"
_BASESHA_SEED_CACHE: dict[str, bytes] | None = None


class SimulatedBoxRestart(BaseException):
    """Unwinds run_day at an exact simulated instant (like SIGTERM/OOM/reboot)."""


def et_dt(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> _REAL_DATETIME:
    return _REAL_DATETIME(year, month, day, hour, minute, second, tzinfo=ET)


class PinnedClock:
    """Deterministic ET/UTC clock that fires scheduled callbacks during sleeps."""

    def __init__(self, initial_et: _REAL_DATETIME):
        if initial_et.tzinfo is None:
            initial_et = initial_et.replace(tzinfo=ET)
        self.current_utc: _REAL_DATETIME = initial_et.astimezone(UTC)
        self.mono: float = 10_000.0
        self._events: list[tuple[_REAL_DATETIME, int, str, Callable[[], None]]] = []
        self._seq: int = 0
        self.event_log: list[dict[str, Any]] = []
        self.sleep_log: list[dict[str, Any]] = []
        self.stop_before_utc: _REAL_DATETIME | None = None

    def now_et(self) -> _REAL_DATETIME:
        return self.current_utc.astimezone(ET)

    def now(self, tz=None) -> _REAL_DATETIME:
        if tz is None:
            return self.current_utc.replace(tzinfo=None)
        return self.current_utc.astimezone(tz)

    def set_et(self, target_et: _REAL_DATETIME) -> None:
        if target_et.tzinfo is None:
            target_et = target_et.replace(tzinfo=ET)
        target_utc = target_et.astimezone(UTC)
        delta = (target_utc - self.current_utc).total_seconds()
        self.current_utc = target_utc
        if delta > 0:
            self.mono += delta

    def schedule_at(self, when_et: _REAL_DATETIME, label: str, fn: Callable[[], None]) -> None:
        if when_et.tzinfo is None:
            when_et = when_et.replace(tzinfo=ET)
        self._seq += 1
        self._events.append((when_et.astimezone(UTC), self._seq, label, fn))
        self._events.sort(key=lambda item: (item[0], item[1]))

    def advance_seconds(self, seconds: float, *, source: str = "sleep") -> None:
        if seconds <= 0:
            self._drain_due_events()
            return
        target_utc = self.current_utc + _REAL_TIMEDELTA(seconds=float(seconds))
        if self.stop_before_utc is not None and target_utc > self.stop_before_utc:
            target_utc = max(self.current_utc, self.stop_before_utc)
        self.sleep_log.append({
            "source": source,
            "from_et": self.now_et().isoformat(),
            "requested_sec": round(float(seconds), 2),
            "target_et": target_utc.astimezone(ET).isoformat(),
        })
        while self._events and self._events[0][0] <= target_utc:
            ev_utc, _, label, fn = self._events.pop(0)
            if ev_utc > self.current_utc:
                self.mono += (ev_utc - self.current_utc).total_seconds()
                self.current_utc = ev_utc
            self.event_log.append({"at_et": self.now_et().isoformat(), "label": label})
            fn()
        if target_utc > self.current_utc:
            self.mono += (target_utc - self.current_utc).total_seconds()
            self.current_utc = target_utc

    def _drain_due_events(self) -> None:
        while self._events and self._events[0][0] <= self.current_utc:
            _, _, label, fn = self._events.pop(0)
            self.event_log.append({"at_et": self.now_et().isoformat(), "label": label})
            fn()


def make_stdlib_safe_clock_types(clock: PinnedClock):
    class _DateMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            return isinstance(instance, _REAL_DATE) and not isinstance(instance, _REAL_DATETIME)

        def __subclasscheck__(cls, subclass: type) -> bool:
            return subclass is cls or issubclass(subclass, _REAL_DATE)

        def __getattr__(cls, name: str) -> Any:
            return getattr(_REAL_DATE, name)

    class _DateTimeMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            return isinstance(instance, _REAL_DATETIME)

        def __subclasscheck__(cls, subclass: type) -> bool:
            return subclass is cls or issubclass(subclass, _REAL_DATETIME)

        def __getattr__(cls, name: str) -> Any:
            return getattr(_REAL_DATETIME, name)

    class FakeDate(metaclass=_DateMeta):
        min = _REAL_DATE.min
        max = _REAL_DATE.max
        resolution = _REAL_DATE.resolution

        def __new__(cls, year: int, month: int, day: int) -> _REAL_DATE:
            return _REAL_DATE(year, month, day)

        @classmethod
        def today(cls) -> _REAL_DATE:
            return clock.now_et().date()

        @classmethod
        def fromisoformat(cls, s: str) -> _REAL_DATE:
            return _REAL_DATE.fromisoformat(s)

    class FakeDateTime(metaclass=_DateTimeMeta):
        min = _REAL_DATETIME.min
        max = _REAL_DATETIME.max
        resolution = _REAL_DATETIME.resolution

        def __new__(cls, *args: Any, **kwargs: Any) -> _REAL_DATETIME:
            return _REAL_DATETIME(*args, **kwargs)

        @classmethod
        def now(cls, tz=None) -> _REAL_DATETIME:
            return clock.now(tz)

        @classmethod
        def utcnow(cls) -> _REAL_DATETIME:
            return clock.current_utc.replace(tzinfo=None)

        @classmethod
        def fromisoformat(cls, s: str) -> _REAL_DATETIME:
            return _REAL_DATETIME.fromisoformat(s)

        @classmethod
        def strptime(cls, date_string: str, fmt: str) -> _REAL_DATETIME:
            return _REAL_DATETIME.strptime(date_string, fmt)

        @classmethod
        def combine(cls, d: _REAL_DATE, t: _REAL_TIME, tzinfo: Any = None) -> _REAL_DATETIME:
            if tzinfo is None:
                return _REAL_DATETIME.combine(d, t)
            return _REAL_DATETIME.combine(d, t, tzinfo=tzinfo)

        @classmethod
        def fromtimestamp(cls, ts: float, tz=None) -> _REAL_DATETIME:
            return _REAL_DATETIME.fromtimestamp(ts, tz=tz)

    return FakeDate, FakeDateTime


@dataclass
class GameSpec:
    game_pk: int
    official_date: str
    start_et: _REAL_DATETIME
    away_name: str
    away_abbr: str
    home_name: str
    home_abbr: str
    away_batter_id: int
    away_batter_name: str
    away_p_hit: float
    home_batter_id: int
    home_batter_name: str
    home_p_hit: float
    away_pitcher_id: int = 400101
    away_pitcher_name: str = "Away Starter"
    home_pitcher_id: int = 400202
    home_pitcher_name: str = "Home Starter"
    away_confirm_et: _REAL_DATETIME | None = None
    home_confirm_et: _REAL_DATETIME | None = None
    final_et: _REAL_DATETIME | None = None
    away_hits: int = 1
    away_at_bats: int = 4
    home_hits: int = 1
    home_at_bats: int = 4
    detailed_override: str | None = None
    game_type: str = "R"


def default_opening_days_schedules() -> dict[str, list[GameSpec]]:
    """Schedules for 2027-03-24 (singleton), 2027-03-25 (4 games), 2027-03-26 (3 games), 2027-03-27."""
    return {
        "2027-03-24": [
            GameSpec(
                game_pk=777001,
                official_date="2027-03-24",
                start_et=et_dt(2027, 3, 24, 20, 5),
                away_name="New York Yankees",
                away_abbr="NYY",
                home_name="San Francisco Giants",
                home_abbr="SF",
                away_batter_id=592450,
                away_batter_name="Aaron Judge",
                away_p_hit=0.835,
                home_batter_id=671218,
                home_batter_name="Heliot Ramos",
                home_p_hit=0.790,
                away_confirm_et=et_dt(2027, 3, 24, 17, 45),
                home_confirm_et=et_dt(2027, 3, 24, 18, 45),
                final_et=et_dt(2027, 3, 24, 22, 50),
                away_hits=1,
                home_hits=1,
            ),
        ],
        "2027-03-25": [
            GameSpec(
                game_pk=777101,
                official_date="2027-03-25",
                start_et=et_dt(2027, 3, 25, 16, 10),
                away_name="Baltimore Orioles",
                away_abbr="BAL",
                home_name="Toronto Blue Jays",
                home_abbr="TOR",
                away_batter_id=683002,
                away_batter_name="Gunnar Henderson",
                away_p_hit=0.842,
                home_batter_id=665489,
                home_batter_name="Vladimir Guerrero Jr.",
                home_p_hit=0.815,
                away_confirm_et=et_dt(2027, 3, 25, 14, 30),
                home_confirm_et=et_dt(2027, 3, 25, 14, 40),
                final_et=et_dt(2027, 3, 25, 19, 0),
                away_hits=2,
                home_hits=1,
            ),
            GameSpec(
                game_pk=777102,
                official_date="2027-03-25",
                start_et=et_dt(2027, 3, 25, 16, 15),
                away_name="Boston Red Sox",
                away_abbr="BOS",
                home_name="Texas Rangers",
                home_abbr="TEX",
                away_batter_id=680776,
                away_batter_name="Jarren Duran",
                away_p_hit=0.828,
                home_batter_id=608369,
                home_batter_name="Corey Seager",
                home_p_hit=0.804,
                away_confirm_et=et_dt(2027, 3, 25, 14, 35),
                home_confirm_et=et_dt(2027, 3, 25, 14, 45),
                final_et=et_dt(2027, 3, 25, 19, 10),
                away_hits=1,
                home_hits=1,
            ),
            GameSpec(
                game_pk=777103,
                official_date="2027-03-25",
                start_et=et_dt(2027, 3, 25, 19, 10),
                away_name="Atlanta Braves",
                away_abbr="ATL",
                home_name="Philadelphia Phillies",
                home_abbr="PHI",
                away_batter_id=660670,
                away_batter_name="Ronald Acuna Jr.",
                away_p_hit=0.795,
                home_batter_id=547180,
                home_batter_name="Bryce Harper",
                home_p_hit=0.785,
                away_confirm_et=et_dt(2027, 3, 25, 17, 30),
                home_confirm_et=et_dt(2027, 3, 25, 17, 45),
                final_et=et_dt(2027, 3, 25, 22, 0),
                away_hits=1,
                home_hits=0,
            ),
            GameSpec(
                game_pk=777104,
                official_date="2027-03-25",
                start_et=et_dt(2027, 3, 25, 22, 10),
                away_name="Chicago Cubs",
                away_abbr="CHC",
                home_name="Los Angeles Dodgers",
                home_abbr="LAD",
                away_batter_id=663538,
                away_batter_name="Nico Hoerner",
                away_p_hit=0.780,
                home_batter_id=660271,
                home_batter_name="Shohei Ohtani",
                home_p_hit=0.801,
                away_confirm_et=et_dt(2027, 3, 25, 20, 30),
                home_confirm_et=et_dt(2027, 3, 25, 20, 40),
                final_et=et_dt(2027, 3, 26, 0, 50),
                away_hits=1,
                home_hits=1,
            ),
        ],
        "2027-03-26": [
            GameSpec(
                game_pk=777201,
                official_date="2027-03-26",
                start_et=et_dt(2027, 3, 26, 19, 5),
                away_name="Kansas City Royals",
                away_abbr="KC",
                home_name="Cleveland Guardians",
                home_abbr="CLE",
                away_batter_id=677951,
                away_batter_name="Bobby Witt Jr.",
                away_p_hit=0.848,
                home_batter_id=680757,
                home_batter_name="Steven Kwan",
                home_p_hit=0.832,
                away_confirm_et=et_dt(2027, 3, 26, 17, 20),
                home_confirm_et=et_dt(2027, 3, 26, 17, 30),
                final_et=et_dt(2027, 3, 26, 21, 55),
                away_hits=1,
                home_hits=1,
            ),
            GameSpec(
                game_pk=777202,
                official_date="2027-03-26",
                start_et=et_dt(2027, 3, 26, 19, 10),
                away_name="San Diego Padres",
                away_abbr="SD",
                home_name="Arizona Diamondbacks",
                home_abbr="AZ",
                away_batter_id=650333,
                away_batter_name="Luis Arraez",
                away_p_hit=0.839,
                home_batter_id=682998,
                home_batter_name="Corbin Carroll",
                home_p_hit=0.792,
                away_confirm_et=et_dt(2027, 3, 26, 17, 25),
                home_confirm_et=et_dt(2027, 3, 26, 17, 35),
                final_et=et_dt(2027, 3, 26, 22, 10),
                away_hits=1,
                home_hits=1,
            ),
            GameSpec(
                game_pk=777203,
                official_date="2027-03-26",
                start_et=et_dt(2027, 3, 26, 20, 10),
                away_name="Houston Astros",
                away_abbr="HOU",
                home_name="Milwaukee Brewers",
                home_abbr="MIL",
                away_batter_id=670541,
                away_batter_name="Yordan Alvarez",
                away_p_hit=0.805,
                home_batter_id=661388,
                home_batter_name="William Contreras",
                home_p_hit=0.788,
                away_confirm_et=et_dt(2027, 3, 26, 18, 30),
                home_confirm_et=et_dt(2027, 3, 26, 18, 40),
                final_et=et_dt(2027, 3, 26, 23, 0),
                away_hits=1,
                home_hits=1,
            ),
        ],
        "2027-03-27": [
            GameSpec(
                game_pk=777301,
                official_date="2027-03-27",
                start_et=et_dt(2027, 3, 27, 13, 5),
                away_name="New York Mets",
                away_abbr="NYM",
                home_name="Washington Nationals",
                home_abbr="WSH",
                away_batter_id=596019,
                away_batter_name="Francisco Lindor",
                away_p_hit=0.825,
                home_batter_id=682928,
                home_batter_name="CJ Abrams",
                home_p_hit=0.790,
                away_confirm_et=et_dt(2027, 3, 27, 11, 30),
                home_confirm_et=et_dt(2027, 3, 27, 11, 40),
                final_et=et_dt(2027, 3, 27, 16, 0),
                away_hits=1,
                home_hits=1,
            ),
        ],
    }


class FakeMLBServer:
    """In-memory MLB Stats API responder driven by PinnedClock."""

    def __init__(self, clock: PinnedClock, schedules: dict[str, list[GameSpec]]):
        self.clock = clock
        self.schedules = schedules
        self.requests: list[str] = []

    def _by_pk(self, game_pk: int) -> GameSpec | None:
        for games in self.schedules.values():
            for g in games:
                if g.game_pk == game_pk:
                    return g
        return None

    def _game_status(self, g: GameSpec, now_et: _REAL_DATETIME) -> tuple[str, str, str]:
        if g.detailed_override is not None:
            return ("F", g.detailed_override, "DR")
        if g.final_et is not None and now_et >= g.final_et:
            return ("F", "Final", "F")
        if now_et >= g.start_et:
            return ("L", "In Progress", "I")
        return ("P", "Scheduled", "S")

    def urlopen(self, req_or_url: Any, timeout: float = 15) -> io.BytesIO:
        url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
        self.requests.append(url)
        parsed = urlparse(url)
        now_et = self.clock.now_et()

        if parsed.path == "/api/v1/schedule":
            qs = parse_qs(parsed.query)
            date_str = (qs.get("date") or [""])[0]
            games_out = []
            for g in self.schedules.get(date_str, []):
                abstract, detailed, code = self._game_status(g, now_et)
                games_out.append({
                    "gamePk": g.game_pk,
                    "gameType": g.game_type,
                    "officialDate": g.official_date,
                    "gameDate": g.start_et.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "status": {
                        "abstractGameCode": abstract,
                        "detailedState": detailed,
                        "statusCode": code,
                    },
                    "teams": {
                        "away": {
                            "team": {"id": 101, "name": g.away_name, "abbreviation": g.away_abbr},
                            "probablePitcher": {"id": g.away_pitcher_id, "fullName": g.away_pitcher_name},
                        },
                        "home": {
                            "team": {"id": 102, "name": g.home_name, "abbreviation": g.home_abbr},
                            "probablePitcher": {"id": g.home_pitcher_id, "fullName": g.home_pitcher_name},
                        },
                    },
                })
            body = {"dates": [{"date": date_str, "games": games_out}] if games_out else []}
            return io.BytesIO(json.dumps(body).encode("utf-8"))

        if parsed.path.startswith("/api/v1.1/game/") and parsed.path.endswith("/feed/live"):
            pk = int(parsed.path.split("/")[4])
            g = self._by_pk(pk)
            if g is None:
                raise OSError(f"unknown gamePk {pk}")
            abstract, detailed, code = self._game_status(g, now_et)
            away_conf = g.away_confirm_et is not None and now_et >= g.away_confirm_et
            home_conf = g.home_confirm_et is not None and now_et >= g.home_confirm_et
            is_live_or_final = abstract in ("L", "F")
            away_stats = {"hits": g.away_hits, "atBats": g.away_at_bats, "sacFlies": 0} if is_live_or_final else {}
            home_stats = {"hits": g.home_hits, "atBats": g.home_at_bats, "sacFlies": 0} if is_live_or_final else {}
            away_player: dict[str, Any] = {
                "person": {"id": g.away_batter_id, "fullName": g.away_batter_name},
                "stats": {"batting": away_stats},
            }
            if away_conf:
                away_player["battingOrder"] = "100"
            home_player: dict[str, Any] = {
                "person": {"id": g.home_batter_id, "fullName": g.home_batter_name},
                "stats": {"batting": home_stats},
            }
            if home_conf:
                home_player["battingOrder"] = "100"
            feed = {
                "gameData": {
                    "game": {"pk": g.game_pk, "type": g.game_type, "season": g.official_date[:4]},
                    "datetime": {
                        "officialDate": g.official_date,
                        "dateTime": g.start_et.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    },
                    "status": {
                        "abstractGameCode": abstract,
                        "detailedState": detailed,
                        "statusCode": code,
                    },
                },
                "liveData": {
                    "boxscore": {
                        "teams": {
                            "away": {"players": {f"ID{g.away_batter_id}": away_player}},
                            "home": {"players": {f"ID{g.home_batter_id}": home_player}},
                        }
                    },
                    "plays": {"allPlays": []},
                },
            }
            return io.BytesIO(json.dumps(feed).encode("utf-8"))

        raise RuntimeError(f"Unhandled MLB URL in FakeMLBServer: {url}")

    def predictions_for(self, date_str: str, *, include_unconfirmed_first_game: bool = True) -> pd.DataFrame:
        now_et = self.clock.now_et()
        rows: list[dict[str, Any]] = []
        for g in self.schedules.get(date_str, []):
            away_conf = g.away_confirm_et is not None and now_et >= g.away_confirm_et
            home_conf = g.home_confirm_et is not None and now_et >= g.home_confirm_et
            game_time_iso = g.start_et.astimezone(UTC).isoformat().replace("+00:00", "Z")
            if away_conf or include_unconfirmed_first_game:
                rows.append({
                    "batter_name": g.away_batter_name,
                    "batter_id": g.away_batter_id,
                    "team": g.away_abbr,
                    "lineup": 1,
                    "pitcher_name": g.home_pitcher_name,
                    "pitcher_id": g.home_pitcher_id,
                    "pitcher_team": g.home_abbr,
                    "p_hit_pa": 0.32,
                    "p_game_hit": g.away_p_hit,
                    "flags": "" if away_conf else "PROJECTED",
                    "game_pk": g.game_pk,
                    "game_time": game_time_iso,
                })
            if home_conf or include_unconfirmed_first_game:
                rows.append({
                    "batter_name": g.home_batter_name,
                    "batter_id": g.home_batter_id,
                    "team": g.home_abbr,
                    "lineup": 1,
                    "pitcher_name": g.away_pitcher_name,
                    "pitcher_id": g.away_pitcher_id,
                    "pitcher_team": g.away_abbr,
                    "p_hit_pa": 0.30,
                    "p_game_hit": g.home_p_hit,
                    "flags": "" if home_conf else "PROJECTED",
                    "game_pk": g.game_pk,
                    "game_time": game_time_iso,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("p_game_hit", ascending=False).reset_index(drop=True)


@dataclass
class FakeContestAccount:
    """In-memory MLB Beat-the-Streak contest account."""
    user_id: int = 771001
    username: str = "stonehengee"
    xsid: str = "xsid-synthetic-2027"
    active_streak: int = 0
    season_best_streak: int = 0
    rounds: dict[int, _REAL_DATE] = field(default_factory=lambda: {
        2026180: _REAL_DATE(2026, 9, 14),
        2027001: _REAL_DATE(2027, 3, 24),
        2027002: _REAL_DATE(2027, 3, 25),
        2027003: _REAL_DATE(2027, 3, 26),
        2027004: _REAL_DATE(2027, 3, 27),
    })
    bts_to_mlb: dict[int, int] = field(default_factory=dict)
    settled_predictions: list[dict[str, Any]] = field(default_factory=list)
    pending_predictions: list[dict[str, Any]] = field(default_factory=list)
    auth_error: Exception | None = None

    def register_schedules(self, schedules: dict[str, list[GameSpec]]) -> None:
        for games in schedules.values():
            for g in games:
                self.bts_to_mlb[9_000_000 + g.away_batter_id] = g.away_batter_id
                self.bts_to_mlb[9_000_000 + g.home_batter_id] = g.home_batter_id

    def round_id_for(self, d: _REAL_DATE) -> int:
        for rid, rdate in self.rounds.items():
            if rdate == d:
                return rid
        raise KeyError(f"no roundId configured for {d}")

    def enter_pick(self, d: _REAL_DATE, slots: list[tuple[int, int]]) -> None:
        rid = self.round_id_for(d)
        self.pending_predictions = [r for r in self.pending_predictions if r.get("roundId") != rid]
        for idx, (mlb_id, game_pk) in enumerate(slots, start=1):
            self.pending_predictions.append({
                "roundId": rid,
                "unitId": 8_000_000 + game_pk,
                "playerId": 9_000_000 + mlb_id,
                "number": idx,
                "result": None,
            })

    def settle_day(self, d: _REAL_DATE, result: str, new_active: int, new_best: int) -> None:
        rid = self.round_id_for(d)
        day_pending = [r for r in self.pending_predictions if r.get("roundId") == rid]
        self.pending_predictions = [r for r in self.pending_predictions if r.get("roundId") != rid]
        self.settled_predictions.append({
            "roundId": rid,
            "result": result,
            "streak": new_active,
            "roundPredictions": [
                {
                    "unitId": r["unitId"],
                    "playerId": r["playerId"],
                    "number": r["number"],
                    "result": result,
                }
                for r in day_pending
            ],
        })
        self.active_streak = new_active
        self.season_best_streak = new_best


def write_units_snapshot(
    box_root: Path, schedules: dict[str, list[GameSpec]], contest: FakeContestAccount
) -> Path:
    units_dir = box_root / "data" / "leaderboard" / "static_snapshots" / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    units_list = []
    for date_str, games in sorted(schedules.items()):
        if not date_str.startswith("2027-"):
            continue
        rid = contest.round_id_for(_REAL_DATE.fromisoformat(date_str))
        for g in games:
            units_list.append({
                "id": 8_000_000 + g.game_pk,
                "feedId": g.game_pk,
                "roundId": rid,
            })
    out = units_dir / "20270324T120000Z.json.gz"
    out.write_bytes(gzip.compress(json.dumps({"units": units_list}).encode("utf-8")))
    return out


def _build_basesha_picks_files() -> tuple[dict[str, bytes], bool]:
    global _BASESHA_SEED_CACHE
    if _BASESHA_SEED_CACHE is not None:
        return _BASESHA_SEED_CACHE, BASESHA_SRC.exists()

    if BASESHA_SRC.exists():
        seed_code = """
import json, sys
from datetime import date, datetime, timezone
from pathlib import Path
from bts.contest_fetch import build_observation
from bts.saver_state import transition_saver_state
from bts.picks import DailyPick, Pick, save_pick, save_shadow_pick, save_streak
from bts.daily_decision import write_decision
from bts.scheduler import SchedulerState, save_state

picks = Path(sys.argv[1])
acct = picks / "account_state"
acct.mkdir(parents=True, exist_ok=True)

save_streak(4, picks, saver_available=False)
obs = build_observation(
    {"activeStreak": 12, "seasonBestStreak": 18, "predictions": []},
    date(2026, 9, 14),
    771001,
    "stonehengee",
    datetime(2026, 10, 20, 17, 30, tzinfo=timezone.utc),
)
(acct / "contest_streak.json").write_text(json.dumps(obs, indent=2, sort_keys=True) + "\\n")
transition_saver_state(picks, expected_prior="uninitialized", new_state="used", season=2026, source="cli")

p914 = DailyPick(
    date="2026-09-14",
    run_time="2026-09-14T21:05:00+00:00",
    pick=Pick("Bobby Witt Jr.", 677951, "KC", 2, "Starter A", 400001, 0.84, [], False, 745001, "2026-09-14T23:10:00Z", "CLE"),
    double_down=None,
    runner_up={"batter_name": "Steven Kwan", "p_game_hit": 0.82},
    notification_sent=True,
    notification_channel="bluesky_dm",
    notification_id="dm-2026-0914",
    delivered_at="2026-09-14T17:12:00-04:00",
    result="hit",
    slot_results={"pick": "hit"},
)
save_pick(p914, picks)
write_decision(
    "2026-09-14", picks, action="single", source="mdp",
    primary={"batter_id": 677951, "batter_name": "Bobby Witt Jr.", "team": "KC", "game_pk": 745001, "p_game_hit": 0.84},
    streak=11, saver_available=False, state_source="contest", state_status="fresh",
    allow_double=True, contest_source_date="2026-09-13",
    delivery_status="delivered", scoreable=True,
    now=datetime(2026, 9, 14, 21, 12, tzinfo=timezone.utc),
)
save_state(SchedulerState(
    date="2026-09-14", schedule_fetched_at="2026-09-14T10:00:00-04:00",
    games=[{"game_pk": 745001, "game_time_et": "2026-09-14T19:10:00-04:00", "lineup_confirmed": True, "is_doubleheader_game2": False}],
    confirmed_game_pks=[745001], runs_completed=[], pick_locked=True,
    pick_locked_at="2026-09-14T17:12:00-04:00", result_status="final",
    next_wakeup="2026-09-15T10:00:00-04:00", committed_pick_written=True,
), picks)

p918 = DailyPick(
    date="2026-09-18",
    run_time="2026-09-18T21:05:00+00:00",
    pick=Pick("Luis Arraez", 650333, "SD", 1, "Starter B", 400002, 0.83, [], False, 745099, "2026-09-18T23:10:00Z", "AZ"),
    double_down=None,
    runner_up=None,
    result="hit",
    slot_results={"pick": "hit"},
)
save_pick(p918, picks)
sh918 = DailyPick(
    date="2026-09-18",
    run_time="2026-09-18T21:06:00+00:00",
    pick=Pick("Luis Arraez", 650333, "SD", 1, "Starter B", 400002, 0.83, [], False, 745099, "2026-09-18T23:10:00Z", "AZ"),
    double_down=None,
    runner_up=None,
    result="hit",
    slot_results={"pick": "hit"},
    shadow_model_version="context_stack_shadow_v2",
)
save_shadow_pick(sh918, picks)
write_decision(
    "2026-09-18", picks, action="single", source="mdp",
    primary={"batter_id": 650333, "batter_name": "Luis Arraez", "team": "SD", "game_pk": 745099, "p_game_hit": 0.83},
    streak=12, saver_available=False, state_source="contest", state_status="fresh",
    allow_double=True, contest_source_date="2026-09-14",
    delivery_status="private_locked", scoreable=True,
    objective="emax_season_best", best_streak=18, best_status="trusted", effective_best=18,
    now=datetime(2026, 9, 18, 21, 10, tzinfo=timezone.utc),
)
"""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(BASESHA_SRC)
            subprocess.run([sys.executable, "-c", seed_code, str(td_path)], env=env, check=True)
            files = {
                str(p.relative_to(td_path)): p.read_bytes()
                for p in sorted(td_path.rglob("*"))
                if p.is_file()
            }
        _BASESHA_SEED_CACHE = files
        return files, True

    # Standalone fallback when copied into tests/ without .codex-review/whole-range/basesha/
    fallback: dict[str, bytes] = {
        "streak.json": json.dumps({
            "streak": 4,
            "saver_available": False,
            "updated": "2026-09-18T23:45:00+00:00",
        }, indent=2).encode("utf-8"),
        "account_state/contest_streak.json": (json.dumps({
            "active_streak": 12,
            "best_streak": 18,
            "recorded_at": "2026-10-20T17:30:00Z",
            "saver_available": None,
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "source_date": "2026-09-14",
            "user_id": 771001,
            "username": "stonehengee",
        }, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "account_state/saver_state.json": json.dumps({
            "season": 2026,
            "state": "used",
            "updated_at": "2026-08-02T14:00:00+00:00",
            "source": "cli",
        }, indent=2).encode("utf-8"),
        "2026-09-14.json": json.dumps({
            "date": "2026-09-14",
            "run_time": "2026-09-14T21:05:00+00:00",
            "pick": {
                "batter_name": "Bobby Witt Jr.", "batter_id": 677951, "team": "KC",
                "lineup_pos": 2, "pitcher_name": "Starter A", "pitcher_id": 400001,
                "p_game_hit": 0.84, "flags": [], "projected_lineup": False,
                "game_pk": 745001, "game_time": "2026-09-14T23:10:00Z", "pitcher_team": "CLE",
            },
            "double_down": None,
            "runner_up": {"batter_name": "Steven Kwan", "p_game_hit": 0.82},
            "bluesky_posted": False,
            "bluesky_uri": None,
            "notification_sent": True,
            "notification_channel": "bluesky_dm",
            "notification_id": "dm-2026-0914",
            "delivered_at": "2026-09-14T17:12:00-04:00",
            "delivery_attempted": False,
            "result": "hit",
            "slot_results": {"pick": "hit"},
        }, indent=2).encode("utf-8"),
        "2026-09-14/decision.json": json.dumps({
            "schema_version": "bts_daily_decision_v3",
            "date": "2026-09-14",
            "recorded_at": "2026-09-14T21:12:00+00:00",
            "action": "single",
            "source": "mdp",
            "primary": {"batter_id": 677951, "batter_name": "Bobby Witt Jr.", "team": "KC", "game_pk": 745001, "p_game_hit": 0.84},
            "double_down": None,
            "second_candidate": None,
            "streak": 11,
            "saver_available": False,
            "state_source": "contest",
            "state_status": "fresh",
            "allow_double": True,
            "contest_source_date": "2026-09-13",
            "delivery_status": "delivered",
            "scoreable": True,
        }, indent=2).encode("utf-8"),
    }
    _BASESHA_SEED_CACHE = fallback
    return fallback, False


def seed_restored_box_from_basesha(box_root: Path) -> dict[str, Any]:
    """Seed ``box_root`` with END-OF-2026 snapshot state."""
    data_dir = box_root / "data"
    picks_dir = data_dir / "picks"
    health_dir = data_dir / "health_state"
    models_dir = data_dir / "models"
    proc_dir = data_dir / "processed"
    park_dir = data_dir / "external" / "park_drag"
    home_dir = box_root / "home"
    for d in (picks_dir, health_dir, models_dir, proc_dir, park_dir / "producer", home_dir):
        d.mkdir(parents=True, exist_ok=True)

    for art in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        src = REPO_ROOT / "data" / "models" / art
        if src.exists():
            shutil.copy2(src, models_dir / art)
    for d_str in ("2027-03-24", "2027-03-25", "2027-03-26", "2027-03-27", "2027-03-28"):
        (models_dir / f"blend_{d_str}.pkl").write_bytes(b"synthetic-blend-placeholder")

    seed_files, used_basesha_writers = _build_basesha_picks_files()
    for rel, content in seed_files.items():
        dst = picks_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)

    (health_dir / "health_dm_delivery_status.json").write_text(json.dumps({
        "schema_version": "bts_health_dm_delivery_status_v1",
        "status": "sent",
        "updated_at": "2026-09-28T23:55:00+00:00",
        "recipient_configured": True,
        "recipient": "stonehengee.bsky.social",
        "critical_count": 1,
        "warn_attention_count": 0,
        "body_sha256": "0" * 64,
        "body_first_line": "BTS health CRITICAL alert(s):",
        "sent_day": "2026-09-28",
        "sent_sources": ["CRITICAL:park_drag_freshness"],
    }, indent=2) + "\n")
    (health_dir / "warn_attention_state.json").write_text(json.dumps({
        "schema_version": "bts_warn_attention_state_v1",
        "sources": {
            "saver_state": {
                "last_seen": "2026-09-28",
                "streak": 1,
                "last_message": "2026 end-of-season note",
            }
        },
    }, indent=2) + "\n")
    (health_dir / "contest_streak_fetch_status.json").write_text(json.dumps({
        "last_success_at": "2026-10-20T17:30:00Z",
        "last_error": None,
    }, indent=2) + "\n")
    (health_dir / "pick_entry_check.json").write_text(json.dumps({
        "date": "2026-09-14",
        "status": "confirmed",
        "reason": "match",
        "checked_at": "2026-09-14T18:30:00-04:00",
    }, indent=2) + "\n")

    (park_dir / "park_drag_export.csv").write_text("venue_id,date,park_drag_delta\n3289,2026-09-27,0.001\n")
    (park_dir / "park_drag_manifest.json").write_text(json.dumps({
        "max_source_game_date": "2026-09-27",
        "generated_at": "2026-09-28T07:45:12",
        "export_rows": 1,
    }, indent=2) + "\n")
    pitches_2026 = "game_pk,game_date,home_team,release_speed,vx0,vy0,vz0,ax,ay,az\n745099,2026-09-27,NYM,95.0,1.0,-138.0,-5.0,-10.0,28.0,-15.0\n"
    (park_dir / "producer" / "pitches_current.csv.gz").write_bytes(gzip.compress(pitches_2026.encode("utf-8")))
    (park_dir / "producer" / "games_meta.csv").write_text("game_pk,venue_id,venue_name,temp_f,condition\n745099,3289,Citi Field,72,Clear\n")
    (park_dir / "producer" / "game_level_static.csv").write_text("game_pk,season,game_date,venue_id,cd_delta,n_pitches\n")

    toml_path = home_dir / ".bts-orchestrator.toml"
    write_orchestrator_toml(
        toml_path,
        picks_dir=picks_dir,
        data_dir=proc_dir,
        models_dir=models_dir,
        heartbeat_path=data_dir / ".heartbeat",
        pick_delivery="private",
        entry_intent="research",
        contest_state_required=None,
        contest_state_expected=None,
    )
    return {
        "box_root": box_root,
        "picks_dir": picks_dir,
        "health_dir": health_dir,
        "models_dir": models_dir,
        "proc_dir": proc_dir,
        "park_dir": park_dir,
        "home_dir": home_dir,
        "toml_path": toml_path,
        "used_basesha_writers": used_basesha_writers,
    }


def write_orchestrator_toml(
    toml_path: Path,
    *,
    picks_dir: Path,
    data_dir: Path,
    models_dir: Path,
    heartbeat_path: Path,
    pick_delivery: str = "dm",
    entry_intent: str = "enter",
    contest_state_required: bool | None = True,
    contest_state_expected: bool | None = True,
    dm_recipient: str = "stonehengee.bsky.social",
    lineup_check_offset_min: int = 60,
    live_forward_capture_on_lock: bool = False,
) -> Path:
    req_line = (
        f"contest_state_required = {'true' if contest_state_required else 'false'}\n"
        if contest_state_required is not None else ""
    )
    exp_line = (
        f"contest_state_expected = {'true' if contest_state_expected else 'false'}\n"
        if contest_state_expected is not None else ""
    )
    cap_flag = "true" if live_forward_capture_on_lock else "false"
    content = (
        "[orchestrator]\n"
        f'picks_dir = "{picks_dir.as_posix()}"\n'
        f'data_dir = "{data_dir.as_posix()}"\n'
        f'models_dir = "{models_dir.as_posix()}"\n'
        f'heartbeat_path = "{heartbeat_path.as_posix()}"\n\n'
        "[bluesky]\n"
        f'dm_recipient = "{dm_recipient}"\n\n'
        "[scheduler]\n"
        f'pick_delivery = "{pick_delivery}"\n'
        f'entry_intent = "{entry_intent}"\n'
        f"{req_line}"
        "early_lock_gap = 0.03\n"
        f"lineup_check_offset_min = {lineup_check_offset_min}\n"
        "cluster_min = 10\n"
        "doubleheader_recheck_min = 15\n"
        "fallback_deadline_min = 35\n"
        "fallback_deadline_min_morning = 25\n"
        "morning_cutoff_hour = 11\n"
        "missed_pick_alert_min = 10\n"
        "results_poll_interval_min = 15\n"
        "results_cap_hour_et = 5\n"
        "default_init_hour_et = 10\n"
        "early_game_buffer_min = 60\n"
        f"live_forward_capture_on_lock = {cap_flag}\n"
        'live_forward_capture_command = ["true"]\n'
        "shadow_model = false\n\n"
        "[health_checks]\n"
        "enabled = true\n"
        f"{exp_line}"
        'pooled_dir = ""\n'
        'leaderboard_dir = ""\n\n'
        "[[tiers]]\n"
        'name = "local"\n'
        'type = "local"\n'
    )
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(content, encoding="utf-8")
    return toml_path


def install_cron_shims(home_dir: Path) -> dict[str, Path]:
    local_bin = home_dir / ".local" / "bin"
    shim_bin = home_dir / "shim_bin"
    bts_proj = home_dir / "projects" / "bts"
    for d in (local_bin, shim_bin, bts_proj):
        d.mkdir(parents=True, exist_ok=True)

    (bts_proj / ".env").write_text(
        "HEALTHCHECKS_PING_URL=https://hc-ping.example.invalid/0000\n"
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.example.invalid/1111\n",
        encoding="utf-8",
    )

    bts_exe = Path(sys.executable).parent / "bts"
    uv_script = local_bin / "uv"
    uv_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [ "${1:-}" = "run" ] && [ "${2:-}" = "bts" ]; then\n'
        "  shift 2\n"
        f'  exec "{bts_exe}" "$@"\n'
        "fi\n"
        'echo "unexpected uv args: $*" >&2\n'
        "exit 2\n",
        encoding="utf-8",
    )
    uv_script.chmod(uv_script.stat().st_mode | stat.S_IXUSR)

    crontab_store = home_dir / ".mock_crontab"
    crontab_store.write_text(
        "# BTS-HETZNER\n"
        "0 3 * * * cd /home/bts/projects/bts && bts data build --seasons 2026 # BTS-HETZNER\n",
        encoding="utf-8",
    )
    crontab_script = shim_bin / "crontab"
    crontab_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'STORE="{crontab_store}"\n'
        'if [ "${1:-}" = "-l" ]; then\n'
        '  if [ ! -f "$STORE" ]; then echo "no crontab for bts" >&2; exit 1; fi\n'
        '  cat "$STORE"\n'
        "  exit 0\n"
        'elif [ "${1:-}" = "-" ]; then\n'
        '  cat > "$STORE"\n'
        "  exit 0\n"
        "fi\n"
        'echo "unexpected crontab args: $*" >&2\n'
        "exit 2\n",
        encoding="utf-8",
    )
    crontab_script.chmod(crontab_script.stat().st_mode | stat.S_IXUSR)
    return {"shim_bin": shim_bin, "crontab_store": crontab_store, "bts_proj": bts_proj}


class RehearsalWorld:
    """Isolated runtime harnessing real scheduler + CLI + captured transports + pinned clock."""

    def __init__(
        self,
        box_root: Path,
        *,
        initial_et: _REAL_DATETIME | None = None,
        schedules: dict[str, list[GameSpec]] | None = None,
    ):
        self.paths = seed_restored_box_from_basesha(box_root)
        self.box_root = box_root
        self.picks_dir: Path = self.paths["picks_dir"]
        self.health_dir: Path = self.paths["health_dir"]
        self.models_dir: Path = self.paths["models_dir"]
        self.proc_dir: Path = self.paths["proc_dir"]
        self.park_dir: Path = self.paths["park_dir"]
        self.home_dir: Path = self.paths["home_dir"]
        self.toml_path: Path = self.paths["toml_path"]

        self.clock = PinnedClock(initial_et or et_dt(2027, 3, 24, 9, 0))
        self.schedules = schedules if schedules is not None else default_opening_days_schedules()
        self.mlb = FakeMLBServer(self.clock, self.schedules)
        self.contest = FakeContestAccount()
        self.contest.register_schedules(self.schedules)

        self.dms: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []
        self.replies: list[dict[str, Any]] = []
        self.cascade_calls: list[dict[str, Any]] = []

        self.cascade_duration_sec: float = 5.0
        self.disk_used_pct: float = 0.20
        self.prediction_override: Callable[[str], pd.DataFrame | None] | None = None
        self.dm_hook: Callable[[str, str], str | None] | None = None
        self._dm_seq: int = 0
        self._post_seq: int = 0
        self._stack: contextlib.ExitStack | None = None
        self._prev_cwd: Path | None = None

    def configure_live_dm(
        self,
        *,
        contest_state_required: bool | None = True,
        contest_state_expected: bool | None = True,
        lineup_check_offset_min: int = 60,
    ) -> Path:
        return write_orchestrator_toml(
            self.toml_path,
            picks_dir=self.picks_dir,
            data_dir=self.proc_dir,
            models_dir=self.models_dir,
            heartbeat_path=self.box_root / "data" / ".heartbeat",
            pick_delivery="dm",
            entry_intent="enter",
            contest_state_required=contest_state_required,
            contest_state_expected=contest_state_expected,
            lineup_check_offset_min=lineup_check_offset_min,
        )

    def _send_dm(self, recipient: str, text: str) -> str:
        if self.dm_hook is not None:
            hook_res = self.dm_hook(recipient, text)
            if hook_res is not None:
                return hook_res
        self._dm_seq += 1
        msg_id = f"dm-msg-{self._dm_seq:04d}"
        self.dms.append({
            "seq": self._dm_seq,
            "at_et": self.clock.now_et().isoformat(),
            "recipient": recipient,
            "text": text,
            "msg_id": msg_id,
        })
        return msg_id

    def _post_bluesky(self, text: str) -> str:
        self._post_seq += 1
        uri = f"at://did:plc:bot/app.bsky.feed.post/{self._post_seq:04d}"
        self.posts.append({
            "seq": self._post_seq,
            "at_et": self.clock.now_et().isoformat(),
            "text": text,
            "uri": uri,
        })
        return uri

    def _reply_bluesky(self, text: str, parent_uri: str) -> str:
        self._post_seq += 1
        uri = f"at://did:plc:bot/app.bsky.feed.post/reply-{self._post_seq:04d}"
        self.replies.append({
            "seq": self._post_seq,
            "at_et": self.clock.now_et().isoformat(),
            "text": text,
            "parent_uri": parent_uri,
            "uri": uri,
        })
        return uri

    def _predict_local(self, date: str, *args: Any, **kwargs: Any) -> pd.DataFrame | None:
        started = self.clock.now_et().isoformat()
        if self.cascade_duration_sec > 0:
            self.clock.advance_seconds(self.cascade_duration_sec, source="cascade")
        if self.prediction_override is not None:
            df = self.prediction_override(date)
        else:
            df = self.mlb.predictions_for(date)
        self.cascade_calls.append({
            "date": date,
            "started_et": started,
            "finished_et": self.clock.now_et().isoformat(),
            "rows": None if df is None else len(df),
        })
        return df

    def __enter__(self) -> "RehearsalWorld":
        import bts.cli
        import bts.contest_fetch
        import bts.contest_state
        import bts.daily_decision
        import bts.data.pull
        import bts.dm
        import bts.features.park_drag_producer
        import bts.health.alert
        import bts.health.backup_freshness
        import bts.health.blend_training
        import bts.health.contest_state
        import bts.health.disk_fill
        import bts.health.fallback_defer
        import bts.health.late_delivery
        import bts.health.park_drag_freshness
        import bts.health.pick_entry
        import bts.health.post_failure
        import bts.health.result_resolution
        import bts.health.runner
        import bts.health.tail_policy
        import bts.heartbeat
        import bts.leaderboard.auth
        import bts.orchestrator
        import bts.picks
        import bts.posting
        import bts.saver_state
        import bts.scheduler
        import bts.season_contract
        import bts.strategy
        import bts.util

        stack = contextlib.ExitStack()
        self._stack = stack
        self._prev_cwd = Path.cwd()
        os.chdir(self.box_root)
        stack.callback(os.chdir, self._prev_cwd)

        bts.strategy._mdp_cache = None

        def _reset_mdp_cache() -> None:
            bts.strategy._mdp_cache = None

        stack.callback(_reset_mdp_cache)

        stack.enter_context(patch.dict(os.environ, {
            "HOME": str(self.home_dir),
            "HEALTHCHECKS_PING_URL": "https://hc-ping.example.invalid/0000",
            "BTS_SCHEDULER_HEARTBEAT_PING_URL": "https://hc-ping.example.invalid/1111",
        }))

        def _blocked_socket(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("Network socket creation is forbidden in RehearsalWorld")

        stack.enter_context(patch.object(socket, "socket", _blocked_socket))
        stack.enter_context(patch.object(socket, "create_connection", _blocked_socket))

        # Synthetic disk usage so host disk fill never contaminates health checks
        def _fake_disk_usage(_path: Any) -> _DiskUsage:
            total = 80 * (1024 ** 3)
            used = int(total * self.disk_used_pct)
            return _DiskUsage(total=total, used=used, free=total - used)

        stack.enter_context(patch.object(bts.health.disk_fill.shutil, "disk_usage", _fake_disk_usage))

        fake_date, fake_datetime = make_stdlib_safe_clock_types(self.clock)
        stack.enter_context(patch.object(_dt_mod, "datetime", fake_datetime))
        stack.enter_context(patch.object(_dt_mod, "date", fake_date))

        for mod in (
            bts.scheduler,
            bts.cli,
            bts.contest_state,
            bts.season_contract,
            bts.saver_state,
            bts.picks,
            bts.daily_decision,
            bts.heartbeat,
            bts.orchestrator,
            bts.health.alert,
            bts.health.contest_state,
            bts.health.late_delivery,
            bts.health.post_failure,
            bts.health.fallback_defer,
            bts.health.pick_entry,
            bts.health.backup_freshness,
            bts.health.park_drag_freshness,
            bts.health.blend_training,
            bts.features.park_drag_producer,
        ):
            if hasattr(mod, "datetime"):
                stack.enter_context(patch.object(mod, "datetime", fake_datetime))
        for mod in (
            bts.health.runner,
            bts.health.result_resolution,
            bts.health.tail_policy,
            bts.health.post_failure,
            bts.health.pick_entry,
            bts.health.blend_training,
            bts.health.late_delivery,
            bts.health.fallback_defer,
            bts.features.park_drag_producer,
        ):
            if hasattr(mod, "date"):
                stack.enter_context(patch.object(mod, "date", fake_date))

        stack.enter_context(patch.object(bts.scheduler, "_now_et", self.clock.now_et))
        stack.enter_context(patch.object(bts.cli, "_now_et", self.clock.now_et))
        stack.enter_context(patch.object(
            bts.cli, "_sleep", lambda s: self.clock.advance_seconds(s, source="cli_sleep")
        ))
        stack.enter_context(patch.object(bts.scheduler.time, "monotonic", lambda: self.clock.mono))
        stack.enter_context(patch.object(bts.cli.time, "monotonic", lambda: self.clock.mono))
        stack.enter_context(patch.object(
            bts.scheduler, "_watchdog_ping_sleep",
            lambda secs, interval_sec=60: self.clock.advance_seconds(secs, source="watchdog_sleep"),
        ))
        stack.enter_context(patch.object(
            bts.scheduler, "_poll_interval_sleep",
            lambda hb, secs, watchdog_interval_sec=60: self.clock.advance_seconds(secs, source="poll_sleep"),
        ))
        stack.enter_context(patch.object(
            bts.scheduler, "read_systemd_nrestarts", lambda unit="bts-scheduler": None, create=True
        ))
        stack.enter_context(patch.object(
            bts.health.runner, "read_systemd_nrestarts", lambda unit="bts-scheduler": 0
        ))

        stack.enter_context(patch.object(bts.dm, "send_dm", self._send_dm))
        stack.enter_context(patch.object(bts.health.alert, "send_dm", self._send_dm))
        stack.enter_context(patch.object(bts.posting, "post_to_bluesky", self._post_bluesky))
        stack.enter_context(patch.object(bts.posting, "reply_to_bluesky", self._reply_bluesky))

        stack.enter_context(patch.object(bts.util, "retry_urlopen", self.mlb.urlopen))
        stack.enter_context(patch.object(bts.picks, "retry_urlopen", self.mlb.urlopen))
        stack.enter_context(patch.object(bts.scheduler, "retry_urlopen", self.mlb.urlopen))
        stack.enter_context(patch.object(bts.data.pull, "retry_urlopen", self.mlb.urlopen))
        stack.enter_context(patch.object(bts.data.pull, "urlopen", self.mlb.urlopen))
        stack.enter_context(patch.object(
            bts.scheduler, "_delivery_game_statuses", bts.picks.get_game_statuses_detailed,
        ))

        def _load_cookies() -> dict[str, str]:
            if self.contest.auth_error is not None:
                raise self.contest.auth_error
            return {"uid": str(self.contest.user_id), "session": "ok"}

        def _extract_uid(cookies: dict[str, str]) -> str:
            return cookies["uid"]

        def _fetch_login_session(uid: str, cookies: dict[str, str], attempts: int = 3) -> Any:
            if self.contest.auth_error is not None:
                raise self.contest.auth_error
            return types.SimpleNamespace(
                user_id=self.contest.user_id,
                username=self.contest.username,
                xsid=self.contest.xsid,
            )

        def _fetch_profile(user_id: int, cookies: dict[str, str], xsid: str, **_: Any) -> dict[str, Any]:
            if self.contest.auth_error is not None:
                raise self.contest.auth_error
            return {
                "activeStreak": self.contest.active_streak,
                "seasonBestStreak": self.contest.season_best_streak,
                "predictions": list(self.contest.settled_predictions),
            }

        def _fetch_pending(cookies: dict[str, str], xsid: str, **_: Any) -> list[dict[str, Any]]:
            if self.contest.auth_error is not None:
                raise self.contest.auth_error
            return list(self.contest.pending_predictions)

        stack.enter_context(patch.object(bts.leaderboard.auth, "load_session_cookies", _load_cookies))
        stack.enter_context(patch.object(bts.leaderboard.auth, "extract_uid", _extract_uid))
        stack.enter_context(patch.object(bts.leaderboard.auth, "fetch_login_session", _fetch_login_session))
        stack.enter_context(patch.object(bts.contest_fetch, "fetch_profile", _fetch_profile))
        stack.enter_context(patch.object(bts.contest_fetch, "fetch_pending_predictions", _fetch_pending))
        stack.enter_context(patch.object(bts.cli, "_fetch_rounds", lambda client=None: dict(self.contest.rounds)))
        stack.enter_context(patch.object(bts.cli, "_fetch_bts_to_mlb", lambda client=None: dict(self.contest.bts_to_mlb)))

        stack.enter_context(patch.object(bts.picks, "_git_head_sha", lambda cwd=".": "a" * 40))
        stack.enter_context(patch.object(bts.orchestrator, "predict_local", self._predict_local))

        def _fake_run_pipeline(date: str, data_dir: str = "data/processed", **kw: Any) -> pd.DataFrame:
            res = self._predict_local(date)
            return res if res is not None else pd.DataFrame()

        fake_predict_mod = types.ModuleType("bts.model.predict")
        fake_predict_mod.run_pipeline = _fake_run_pipeline
        fake_predict_mod.load_blend = lambda path: {"_model": "stub"}
        fake_predict_mod.save_blend = lambda obj, path: Path(path).write_bytes(b"stub-blend")
        stack.enter_context(patch.dict(sys.modules, {"bts.model.predict": fake_predict_mod}))

        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._stack is not None:
            self._stack.__exit__(exc_type, exc, tb)
            self._stack = None

    def invoke_cli(self, args: list[str], *, catch_exceptions: bool = False):
        from click.testing import CliRunner
        from bts.cli import cli

        runner = CliRunner()
        return runner.invoke(cli, args, catch_exceptions=catch_exceptions)

    def run_scheduler_day(self, date_str: str, *, stop_at_et: _REAL_DATETIME | None = None) -> None:
        from bts.orchestrator import load_config
        from bts.scheduler import run_day

        prev_stop = self.clock.stop_before_utc
        if stop_at_et is not None:
            self.clock.stop_before_utc = stop_at_et.astimezone(UTC)
        try:
            cfg = load_config(self.toml_path)
            run_day(date=date_str, config=cfg, dry_run=False)
        finally:
            self.clock.stop_before_utc = prev_stop

    def read_pick_json(self, date_str: str) -> dict[str, Any] | None:
        p = self.picks_dir / f"{date_str}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def read_decision_json(self, date_str: str) -> dict[str, Any] | None:
        p = self.picks_dir / date_str / "decision.json"
        return json.loads(p.read_text()) if p.exists() else None

    def read_streak_json(self) -> dict[str, Any] | None:
        p = self.picks_dir / "streak.json"
        return json.loads(p.read_text()) if p.exists() else None
