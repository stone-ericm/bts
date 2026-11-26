"""Shared offline test harness for whole-range review slice 2.

Drives real `bts.scheduler.run_day` and `bts.cli.cli` against a temporary data
directory with captured transports (DM, Bluesky post/reply, MLB schedule/feeds,
predictions, sd_notify), a pinned advancing ET clock, and socket-level network
blocking. Leaves zero process-wide state changed and never imports
`bts.model.predict` at module level.
"""
from __future__ import annotations

import contextlib
import datetime as _dt_mod
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

RealDatetime = _dt_mod.datetime
RealDate = _dt_mod.date
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parent.parent
BASE_SRC = REVIEW_DIR / "base_619abda7" / "src"
HEAD_SRC = REPO_ROOT / "src"


@contextlib.contextmanager
def isolated_tmp_root(prefix: str = "run_"):
    """Create a temporary directory strictly inside `.carbon-review/whole-range/tmp/`
    and point TMPDIR there, restoring environment and tempfile cache on exit."""
    base_tmp = REVIEW_DIR / "tmp"
    base_tmp.mkdir(parents=True, exist_ok=True)
    old_tmpdir = os.environ.get("TMPDIR")
    old_tempdir_cache = tempfile.tempdir
    os.environ["TMPDIR"] = str(base_tmp)
    tempfile.tempdir = str(base_tmp)
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_tmp)))
    try:
        yield run_dir
    finally:
        os.environ.pop("TMPDIR", None)
        if old_tmpdir is not None:
            os.environ["TMPDIR"] = old_tmpdir
        tempfile.tempdir = old_tempdir_cache
        shutil.rmtree(run_dir, ignore_errors=True)


@contextlib.contextmanager
def block_network():
    """Fail immediately if any code tries to open a real network socket."""
    def _deny(*args, **kwargs):
        raise AssertionError(f"Unexpected network call: args={args!r}")

    with patch.object(socket.socket, "connect", _deny), \
         patch.object(socket.socket, "connect_ex", _deny), \
         patch.object(socket, "create_connection", _deny):
        yield


class VirtualClock:
    """Deterministic virtual clock supporting pinned instants and sleep advances."""

    def __init__(self, start_et_iso: str):
        parsed = RealDatetime.fromisoformat(start_et_iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ET)
        self._utc = parsed.astimezone(UTC)
        self.sleeps: list[tuple[str, float]] = []
        self.idle_seconds: float | None = None

    def now(self, tz=None) -> _dt_mod.datetime:
        if tz is None:
            return self._utc.astimezone(ET).replace(tzinfo=None)
        return self._utc.astimezone(tz)

    def now_et(self) -> _dt_mod.datetime:
        return self._utc.astimezone(ET)

    def now_utc(self) -> _dt_mod.datetime:
        return self._utc

    def advance(self, seconds: float, kind: str = "sleep") -> None:
        self.sleeps.append((kind, float(seconds)))
        if seconds > 0:
            self._utc = self._utc + _dt_mod.timedelta(seconds=float(seconds))

    def set_et(self, et_iso: str) -> None:
        parsed = RealDatetime.fromisoformat(et_iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ET)
        self._utc = parsed.astimezone(UTC)


def make_pinned_datetime_cls(clock: VirtualClock):
    """Return a datetime subclass whose constructors return real `datetime.datetime`
    instances while `.now()` / `.today()` read `clock`, and `isinstance(dt, PinnedDatetime)`
    holds for all real `datetime.datetime` instances."""

    class _PinnedDatetimeMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, RealDatetime)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, RealDatetime)

    class PinnedDatetime(RealDatetime, metaclass=_PinnedDatetimeMeta):
        def __new__(cls, *args, **kwargs):
            return RealDatetime(*args, **kwargs)

        @classmethod
        def now(cls, tz=None):
            return clock.now(tz)

        @classmethod
        def utcnow(cls):
            return clock.now_utc().replace(tzinfo=None)

        @classmethod
        def today(cls):
            return clock.now_et().replace(tzinfo=None)

        @classmethod
        def fromisoformat(cls, s):
            return RealDatetime.fromisoformat(s)

        @classmethod
        def strptime(cls, date_string, fmt):
            return RealDatetime.strptime(date_string, fmt)

        @classmethod
        def combine(cls, date, time, tzinfo=True):
            if tzinfo is True:
                return RealDatetime.combine(date, time)
            return RealDatetime.combine(date, time, tzinfo=tzinfo)

        @classmethod
        def fromtimestamp(cls, t, tz=None):
            return RealDatetime.fromtimestamp(t, tz=tz)

    return PinnedDatetime


def make_pinned_date_cls(clock: VirtualClock):
    class _PinnedDateMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, RealDate)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, RealDate)

    class PinnedDate(RealDate, metaclass=_PinnedDateMeta):
        def __new__(cls, *args, **kwargs):
            return RealDate(*args, **kwargs)

        @classmethod
        def today(cls):
            d = clock.now_et().date()
            return RealDate(d.year, d.month, d.day)

        @classmethod
        def fromisoformat(cls, s):
            return RealDate.fromisoformat(s)

    return PinnedDate


@dataclass
class GameSpec:
    game_pk: int
    official_date: str
    game_date_utc: str
    away_name: str = "New York Yankees"
    away_abbrev: str = "NYY"
    away_id: int = 147
    home_name: str = "Boston Red Sox"
    home_abbrev: str = "BOS"
    home_id: int = 111
    game_type: str = "R"
    double_header: str = "N"
    # Optional dynamic overrides: fn(now_et) -> (abstract, detailed, status_code)
    status_override: object = None
    # When lineup confirms (ET ISO); None = never confirms before game, "always" = always confirmed
    away_confirm_at_et: str | None = "always"
    home_confirm_at_et: str | None = "always"
    # Batters in game: list of dicts {id, name, side: "away"|"home", batting: {hits, atBats, baseOnBalls, sacFlies}}
    batters: list[dict] = field(default_factory=list)
    # Optional suspended-game fields
    resume_datetime_utc: str | None = None
    plays: list[dict] | None = None
    duration_min: int = 150


class FakeHTTPResponse:
    def __init__(self, payload: dict | list):
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class SyntheticMLBWorld:
    """Time-aware synthetic MLB Stats API schedule and live game feeds."""

    def __init__(self, clock: VirtualClock, games_by_date: dict[str, list[GameSpec]]):
        self.clock = clock
        self.games_by_date = games_by_date
        self.requests: list[str] = []

    def _game_start_et(self, g: GameSpec) -> _dt_mod.datetime:
        return RealDatetime.fromisoformat(g.game_date_utc.replace("Z", "+00:00")).astimezone(ET)

    def _status_at(self, g: GameSpec, now_et: _dt_mod.datetime) -> tuple[str, str, str]:
        if callable(g.status_override):
            return g.status_override(now_et)
        if isinstance(g.status_override, tuple):
            return g.status_override
        start_et = self._game_start_et(g)
        end_et = start_et + _dt_mod.timedelta(minutes=g.duration_min)
        if now_et < start_et:
            return ("P", "Scheduled", "S")
        if now_et < end_et:
            return ("L", "In Progress", "I")
        return ("F", "Final", "F")

    def _side_confirmed(self, confirm_spec: str | None, now_et: _dt_mod.datetime) -> bool:
        if confirm_spec == "always":
            return True
        if confirm_spec is None:
            return False
        at = RealDatetime.fromisoformat(confirm_spec)
        if at.tzinfo is None:
            at = at.replace(tzinfo=ET)
        return now_et >= at

    def schedule_payload(self, date_str: str) -> dict:
        now_et = self.clock.now_et()
        games = self.games_by_date.get(date_str, [])
        if not games:
            return {"dates": []}
        out_games = []
        for g in games:
            abstract, detailed, code = self._status_at(g, now_et)
            out_games.append({
                "gamePk": g.game_pk,
                "gameType": g.game_type,
                "doubleHeader": g.double_header,
                "officialDate": g.official_date,
                "gameDate": g.game_date_utc,
                "status": {
                    "abstractGameCode": abstract,
                    "detailedState": detailed,
                    "statusCode": code,
                },
                "teams": {
                    "away": {"team": {"id": g.away_id, "name": g.away_name}},
                    "home": {"team": {"id": g.home_id, "name": g.home_name}},
                },
            })
        return {"dates": [{"date": date_str, "games": out_games}]}

    def feed_payload(self, game_pk: int) -> dict:
        now_et = self.clock.now_et()
        found = None
        for glist in self.games_by_date.values():
            for g in glist:
                if g.game_pk == game_pk:
                    found = g
                    break
        if found is None:
            raise AssertionError(f"Unknown synthetic game_pk {game_pk}")

        abstract, detailed, code = self._status_at(found, now_et)
        away_conf = self._side_confirmed(found.away_confirm_at_et, now_et)
        home_conf = self._side_confirmed(found.home_confirm_at_et, now_et)

        away_players = {}
        home_players = {}
        if away_conf:
            away_players["ID900001"] = {
                "person": {"id": 900001, "fullName": "Away Placeholder"},
                "battingOrder": "100",
                "stats": {"batting": {"atBats": 4, "hits": 1}},
            }
        if home_conf:
            home_players["ID900002"] = {
                "person": {"id": 900002, "fullName": "Home Placeholder"},
                "battingOrder": "100",
                "stats": {"batting": {"atBats": 4, "hits": 1}},
            }

        for idx, b in enumerate(found.batters, start=2):
            side = b.get("side", "away")
            conf = away_conf if side == "away" else home_conf
            # Populate final batting stats when Final, or live_batting when Live
            if abstract == "F" and "batting" in b:
                stats = {"batting": dict(b["batting"])}
            elif abstract == "L" and "live_batting" in b:
                stats = {"batting": dict(b["live_batting"])}
            else:
                stats = {"batting": {}}
            entry = {
                "person": {"id": b["id"], "fullName": b["name"]},
                "stats": stats,
            }
            if conf and b.get("in_lineup", True):
                entry["battingOrder"] = f"{idx}00"
            target = away_players if side == "away" else home_players
            target[f"ID{b['id']}"] = entry

        dt_block = {"officialDate": found.official_date, "dateTime": found.game_date_utc}
        if found.resume_datetime_utc:
            dt_block["resumeDateTime"] = found.resume_datetime_utc

        plays = found.plays if found.plays is not None else []
        return {
            "gameData": {
                "game": {
                    "pk": found.game_pk,
                    "type": found.game_type,
                    "doubleHeader": found.double_header,
                    "season": found.official_date[:4],
                },
                "datetime": dt_block,
                "status": {
                    "abstractGameCode": abstract,
                    "detailedState": detailed,
                    "statusCode": code,
                },
                "teams": {
                    "away": {"id": found.away_id, "name": found.away_name},
                    "home": {"id": found.home_id, "name": found.home_name},
                },
                "venue": {"id": 3313, "name": "Fenway Park", "fieldInfo": {"roofType": "Open"}},
                "weather": {"condition": "Clear", "temp": "75", "wind": "5 mph, Out To CF"},
            },
            "liveData": {
                "boxscore": {
                    "teams": {
                        "away": {"players": away_players},
                        "home": {"players": home_players},
                    },
                    "officials": [],
                },
                "plays": {"allPlays": plays},
            },
        }

    def urlopen(self, req_or_url, timeout=15, **kwargs):
        url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
        self.requests.append(url)
        parsed = urllib.parse.urlparse(url)
        if "/api/v1/schedule" in parsed.path:
            qs = urllib.parse.parse_qs(parsed.query)
            date_str = qs["date"][0]
            return FakeHTTPResponse(self.schedule_payload(date_str))
        if "/api/v1.1/game/" in parsed.path and parsed.path.endswith("/feed/live"):
            parts = parsed.path.split("/")
            game_pk = int(parts[parts.index("game") + 1])
            return FakeHTTPResponse(self.feed_payload(game_pk))
        raise AssertionError(f"Unhandled URL in SyntheticMLBWorld: {url}")


@dataclass
class CapturedTransports:
    dms: list[dict] = field(default_factory=list)
    posts: list[str] = field(default_factory=list)
    replies: list[dict] = field(default_factory=list)
    sd_events: list[str] = field(default_factory=list)
    stderr_text: str = ""


def stage_models(workdir: Path) -> Path:
    """Link tracked MDP policy artifacts into workdir/data/models."""
    models_dir = workdir / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        src = REPO_ROOT / "data" / "models" / name
        dst = models_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
    return models_dir


def write_covered_auto_state(
    picks_dir: Path,
    *,
    season: int,
    streak: int,
    best_streak: int,
    source_date: str | None,
    recorded_at_utc: str,
    saver_state: str = "active",
    saver_season: int | None = None,
    activation_id: str = "a" * 32,
    switched_at_utc: str = "2027-04-01T12:00:00Z",
    auto_schema: str = "bts_contest_streak_auto_v2",
    write_switch: bool = True,
    switch_status: str = "automatic",
    model_streak: int | None = None,
    model_saver: bool = True,
    model_season: int | None = 2027,
    applied_dates: list[str] | None = None,
) -> None:
    """Populate account_state/ and streak.json for synthetic runs."""
    acc = picks_dir / "account_state"
    acc.mkdir(parents=True, exist_ok=True)
    if write_switch:
        contract = {
            "schema_version": "bts_season_contract_v1",
            "season": season,
            "status": switch_status,
            "activation_id": activation_id,
            "switched_at": switched_at_utc if switch_status == "automatic" else None,
            "revoked_at": switched_at_utc if switch_status == "revoked" else None,
            "note": None,
        }
        (acc / "season_contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    obs = {
        "schema_version": auto_schema,
        "active_streak": streak,
        "best_streak": best_streak,
        "source": "mlb_bts_profile",
        "source_date": source_date,
        "recorded_at": recorded_at_utc,
        "user_id": 12345,
        "username": "stonehengee",
        "saver_available": None,
    }
    if auto_schema == "bts_contest_streak_auto_v2":
        obs["season"] = season
        obs["activation_id"] = activation_id
    (acc / "contest_streak.json").write_text(json.dumps(obs, indent=2) + "\n")

    if saver_state != "absent":
        sv_season = season if saver_season is None else saver_season
        (acc / "saver_state.json").write_text(json.dumps({
            "season": sv_season,
            "state": saver_state,
            "updated_at": recorded_at_utc,
            "source": "cli",
        }, indent=2) + "\n")

    ms = streak if model_streak is None else model_streak
    streak_doc: dict = {
        "streak": ms,
        "saver_available": model_saver,
        "updated": recorded_at_utc,
    }
    if applied_dates is not None:
        streak_doc["applied_dates"] = list(applied_dates)
    if model_season is not None:
        streak_doc["season"] = model_season
    (picks_dir / "streak.json").write_text(json.dumps(streak_doc, indent=2) + "\n")


def _toml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    raise TypeError(f"Unsupported TOML value: {v!r}")


def write_toml_config(path: Path, cfg: dict) -> Path:
    """Write an orchestrator config dict as valid TOML without external deps."""
    lines: list[str] = []
    for section, body in cfg.items():
        if isinstance(body, dict):
            lines.append(f"[{section}]")
            for k, v in body.items():
                if v is None:
                    continue
                if isinstance(v, dict):
                    lines.append(f"[{section}.{k}]")
                    for sk, sv in v.items():
                        if sv is not None:
                            lines.append(f"{sk} = {_toml_val(sv)}")
                else:
                    lines.append(f"{k} = {_toml_val(v)}")
            lines.append("")
        elif isinstance(body, list):
            for item in body:
                lines.append(f"[[{section}]]")
                for k, v in item.items():
                    if v is not None:
                        lines.append(f"{k} = {_toml_val(v)}")
                lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def base_live_config(workdir: Path, *, pick_delivery: str = "dm", entry_intent: str = "enter",
                     contest_state_required: bool = True, contest_state_expected: bool = True,
                     offset_min: int = 60, shadow_model: bool = False) -> dict:
    return {
        "orchestrator": {
            "picks_dir": str(workdir / "data" / "picks"),
            "data_dir": str(workdir / "data" / "processed"),
            "models_dir": str(workdir / "data" / "models"),
            "heartbeat_path": str(workdir / "data" / ".heartbeat"),
        },
        "bluesky": {
            "dm_recipient": "stonehengee.bsky.social",
        },
        "scheduler": {
            "pick_delivery": pick_delivery,
            "entry_intent": entry_intent,
            "contest_state_required": contest_state_required,
            "early_lock_gap": 0.03,
            "lineup_check_offset_min": offset_min,
            "cluster_min": 10,
            "doubleheader_recheck_min": 15,
            "fallback_deadline_min": 35,
            "fallback_deadline_min_morning": 25,
            "morning_cutoff_hour": 11,
            "missed_pick_alert_min": 10,
            "results_poll_interval_min": 15,
            "results_cap_hour_et": 5,
            "default_init_hour_et": 10,
            "early_game_buffer_min": 60,
            "live_forward_capture_on_lock": False,
            "shadow_model": shadow_model,
        },
        "health_checks": {
            "enabled": True,
            "contest_state_expected": contest_state_expected,
            "pooled_dir": "",
            "leaderboard_dir": "",
        },
        "tiers": [
            {"name": "local", "type": "local"},
        ],
    }


@contextlib.contextmanager
def harness_env(
    workdir: Path,
    clock: VirtualClock,
    world: SyntheticMLBWorld,
    predict_fn,
    *,
    dm_fail: bool = False,
):
    """Patch clocks, sleeps, network, MLB HTTP, predictions, and transports cleanly."""
    import bts.cli as cli_mod
    import bts.contest_state as cs_mod
    import bts.daily_decision as dd_mod
    import bts.dm as dm_mod
    import bts.health.alert as alert_mod
    import bts.health.contest_state as h_cs_mod
    import bts.health.late_delivery as h_ld_mod
    import bts.health.post_failure as h_pf_mod
    import bts.health.runner as h_run_mod
    import bts.health.tail_policy as h_tp_mod
    import bts.health.unit_drift as h_ud_mod
    import bts.heartbeat as hb_mod
    import bts.orchestrator as orch_mod
    import bts.picks as picks_mod
    import bts.posting as post_mod
    import bts.saver_state as sv_mod
    import bts.scheduler as sch_mod
    import bts.sd_notify as sd_mod
    import bts.strategy as strat_mod
    import bts.util as util_mod

    stage_models(workdir)
    (workdir / "data" / "picks").mkdir(parents=True, exist_ok=True)
    (workdir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (workdir / "data" / "health_state").mkdir(parents=True, exist_ok=True)

    PinnedDT = make_pinned_datetime_cls(clock)
    PinnedD = make_pinned_date_cls(clock)
    captured = CapturedTransports()
    old_cwd = Path.cwd()
    old_mdp_cache = strat_mod._mdp_cache

    def _fake_send_dm(recipient: str, text: str) -> str:
        if dm_fail:
            raise RuntimeError("synthetic DM transport failure")
        msg_id = f"dm-{len(captured.dms) + 1:03d}"
        captured.dms.append({
            "id": msg_id,
            "at_et": clock.now_et().isoformat(),
            "recipient": recipient,
            "text": text,
        })
        return msg_id

    def _fake_post(text: str) -> str:
        uri = f"at://did:plc:bot/app.bsky.feed.post/{len(captured.posts) + 1}"
        captured.posts.append(text)
        return uri

    def _fake_reply(text: str, parent_uri: str) -> str:
        uri = f"at://did:plc:bot/app.bsky.feed.post/reply-{len(captured.replies) + 1}"
        captured.replies.append({"text": text, "parent_uri": parent_uri})
        return uri

    def _fake_watchdog_sleep(seconds: float, interval_sec: float = 60) -> None:
        clock.advance(seconds, kind="watchdog_sleep")

    def _fake_poll_sleep(hb_path, seconds: float, watchdog_interval_sec: float = 60) -> None:
        if hb_path is not None:
            hb_mod.write_heartbeat(hb_path, state=hb_mod.HeartbeatState.RUNNING)
        clock.advance(seconds, kind="poll_sleep")

    real_idle = sch_mod._idle_until_next_wakeup

    def _fake_idle(next_wakeup_iso: str | None, heartbeat_path: Path | None) -> None:
        # Execute real validation/heartbeat logic, intercepting only the final long sleep
        def _record_idle_sleep(seconds: float, interval_sec: float = 60) -> None:
            clock.idle_seconds = float(seconds)
            clock.sleeps.append(("idle_until_next_wakeup", float(seconds)))

        with patch.object(sch_mod, "_watchdog_ping_sleep", _record_idle_sleep):
            real_idle(next_wakeup_iso, heartbeat_path)

    stack = contextlib.ExitStack()
    try:
        os.chdir(workdir)
        strat_mod._mdp_cache = None
        stack.enter_context(block_network())
        stack.enter_context(patch.object(_dt_mod, "datetime", PinnedDT))
        stack.enter_context(patch.object(_dt_mod, "date", PinnedD))

        for mod in (
            sch_mod, cs_mod, picks_mod, dd_mod, sv_mod, post_mod, cli_mod,
            alert_mod, h_run_mod, h_cs_mod, h_pf_mod, h_ld_mod, hb_mod, orch_mod, strat_mod,
        ):
            if hasattr(mod, "datetime"):
                stack.enter_context(patch.object(mod, "datetime", PinnedDT))
        for mod in (cs_mod, h_run_mod, h_cs_mod, h_pf_mod, h_ld_mod, h_tp_mod):
            if hasattr(mod, "date"):
                stack.enter_context(patch.object(mod, "date", PinnedD))
        if hasattr(picks_mod, "date_type"):
            stack.enter_context(patch.object(picks_mod, "date_type", PinnedD))
        if hasattr(sch_mod, "date_type"):
            stack.enter_context(patch.object(sch_mod, "date_type", PinnedD))
        # season_contract exists only on HEAD, not BASESHA
        try:
            import bts.season_contract as sc_mod
            if hasattr(sc_mod, "datetime"):
                stack.enter_context(patch.object(sc_mod, "datetime", PinnedDT))
        except ImportError:
            pass

        stack.enter_context(patch.object(sch_mod, "_now_et", clock.now_et))
        stack.enter_context(patch.object(post_mod, "_now_et", clock.now_et))
        stack.enter_context(patch.object(cli_mod, "_now_et", clock.now_et))
        stack.enter_context(patch.object(cli_mod, "_sleep", lambda s: clock.advance(s, "cli_sleep")))

        stack.enter_context(patch.object(sch_mod, "_watchdog_ping_sleep", _fake_watchdog_sleep))
        stack.enter_context(patch.object(sch_mod, "_poll_interval_sleep", _fake_poll_sleep))
        stack.enter_context(patch.object(sch_mod, "_idle_until_next_wakeup", _fake_idle))

        stack.enter_context(patch.object(util_mod, "retry_urlopen", world.urlopen))
        stack.enter_context(patch.object(sch_mod, "retry_urlopen", world.urlopen))
        stack.enter_context(patch.object(picks_mod, "retry_urlopen", world.urlopen))
        if hasattr(sch_mod, "_delivery_game_statuses"):
            stack.enter_context(patch.object(
                sch_mod, "_delivery_game_statuses", picks_mod.get_game_statuses_detailed
            ))

        stack.enter_context(patch.object(dm_mod, "send_dm", _fake_send_dm))
        stack.enter_context(patch.object(alert_mod, "send_dm", _fake_send_dm))
        stack.enter_context(patch.object(post_mod, "post_to_bluesky", _fake_post))
        stack.enter_context(patch.object(post_mod, "reply_to_bluesky", _fake_reply))

        stack.enter_context(patch.object(sd_mod, "notify_ready", lambda: captured.sd_events.append("READY=1")))
        stack.enter_context(patch.object(sd_mod, "notify_watchdog", lambda: captured.sd_events.append("WATCHDOG=1")))
        stack.enter_context(patch.object(sch_mod, "notify_ready", lambda: captured.sd_events.append("READY=1")))
        stack.enter_context(patch.object(sch_mod, "notify_watchdog", lambda: captured.sd_events.append("WATCHDOG=1")))

        stack.enter_context(patch.object(orch_mod, "predict_local", predict_fn))
        stack.enter_context(patch.object(h_run_mod, "read_systemd_nrestarts", lambda *a, **k: 0))
        stack.enter_context(patch.object(h_ud_mod, "check", lambda **k: []))
        # Prevent git subprocess calls in provenance/calibration from slowing down runs
        stack.enter_context(patch.object(picks_mod, "_git_head_sha", lambda cwd=".": "synthetic_sha"))

        yield captured
    finally:
        stack.close()
        strat_mod._mdp_cache = old_mdp_cache
        os.chdir(old_cwd)


def run_worker_subprocess(script_path: Path, code_tree: str, scenario: str, extra_args: list[str] | None = None) -> dict:
    """Execute `script_path --worker <code_tree> <scenario>` under clean PYTHONPATH
    (`HEAD_SRC` or `BASE_SRC`) and return the decoded JSON result."""
    src_path = HEAD_SRC if code_tree == "head" else BASE_SRC
    with isolated_tmp_root(prefix=f"sub_{code_tree}_{scenario}_") as tmp_dir:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{src_path}:{REVIEW_DIR}:{REPO_ROOT}"
        env["TMPDIR"] = str(tmp_dir)
        env["TZ"] = "America/New_York"
        cmd = [sys.executable, str(script_path), "--worker", code_tree, scenario, *(extra_args or [])]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=55)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker {code_tree}/{scenario} failed (rc={proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return json.loads(proc.stdout)
