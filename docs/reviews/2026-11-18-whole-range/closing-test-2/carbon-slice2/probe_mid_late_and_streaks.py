"""Slice 2 Probe 1: Mid-season (2027-07-15), late-season (2027-09-20), and finale (2027-09-26)
automatic-path days, streaks 55..400, all streak readers, and BASESHA vs HEAD comparison.

Runnable standalone (`nice -n 19 .venv/bin/python .carbon-review/whole-range/probe_mid_late_and_streaks.py`)
and adoptable as a pytest regression module (no module-level `bts.model.predict` import,
no process-wide state left mutated, no network).
"""
from __future__ import annotations

import contextlib
import datetime as dt_stdlib
import gzip
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace
from datetime import date as real_date, datetime as real_datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from click.testing import CliRunner

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "bts").is_dir():
            return parent
    raise RuntimeError("Could not locate BTS repository root")


REPO_ROOT = _find_repo_root()


def _ensure_basesha_tree(tmp_path: Path) -> Path:
    """Return path to extracted BASESHA (619abda7) tree, extracting into tmp_path if needed."""
    pre_extracted = REPO_ROOT / ".carbon-review" / "whole-range" / "basesha"
    if (pre_extracted / "src" / "bts" / "scheduler.py").exists():
        return pre_extracted
    dest = tmp_path / "basesha"
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "619abda7", "src", "scripts"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    return dest


class ClockController:
    """Deterministic advancing ET/UTC clock shared across all bts modules."""

    def __init__(self, initial_et: real_datetime):
        assert initial_et.tzinfo is not None
        self.now_utc: real_datetime = initial_et.astimezone(UTC)
        self.sleeps: list[float] = []

    @property
    def now_et(self) -> real_datetime:
        return self.now_utc.astimezone(ET)

    def set_et(self, dt_et: real_datetime) -> None:
        self.now_utc = dt_et.astimezone(UTC)

    def advance(self, seconds: float) -> None:
        secs = max(0.0, float(seconds))
        self.sleeps.append(secs)
        self.now_utc = self.now_utc + timedelta(seconds=secs)


@contextlib.contextmanager
def pinned_environment(monkeypatch: pytest.MonkeyPatch, clock: ClockController):
    """Block all sockets and pin wall clocks + sleep hooks across stdlib + bts modules."""

    real_sock = socket.socket

    class _BlockedSocket(real_sock):
        def __init__(self, *args, **kwargs):
            raise AssertionError(f"Network socket creation blocked in probe: args={args} kwargs={kwargs}")

    def _blocked_conn(*args, **kwargs):
        raise AssertionError(f"Network connection blocked in probe: args={args} kwargs={kwargs}")

    monkeypatch.setattr(socket, "socket", _BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", _blocked_conn)

    class PinnedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return clock.now_et.replace(tzinfo=None)
            return clock.now_utc.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return clock.now_utc.replace(tzinfo=None)

    class PinnedDate(real_date):
        @classmethod
        def today(cls):
            return clock.now_et.date()

    monkeypatch.setattr(dt_stdlib, "datetime", PinnedDatetime)
    monkeypatch.setattr(dt_stdlib, "date", PinnedDate)

    import bts.cli as cli_mod
    import bts.contest_fetch as cf_mod
    import bts.contest_state as cs_mod
    import bts.daily_decision as dd_mod
    import bts.day_outcome as do_mod
    import bts.entry_receipt as er_mod
    import bts.heartbeat as hb_mod
    import bts.picks as picks_mod
    import bts.posting as post_mod
    import bts.reconcile_receipt as rr_mod
    import bts.saver_state as ss_mod
    import bts.scheduler as sch_mod
    import bts.season_contract as sc_mod
    import bts.skip_policy_shadow as sps_mod
    import bts.strategy as strat_mod
    import bts.web as web_mod
    import bts.health.alert as halert_mod
    import bts.health.contest_state as hcs_mod
    import bts.health.fallback_defer as hfd_mod
    import bts.health.late_delivery as hld_mod
    import bts.health.pick_entry as hpe_mod
    import bts.health.post_failure as hpf_mod
    import bts.health.postponed_pick as hpp_mod
    import bts.health.result_resolution as hrr_mod
    import bts.health.runner as hrunner_mod
    import bts.health.tail_policy as htp_mod

    dt_modules = [
        cli_mod, cf_mod, cs_mod, dd_mod, do_mod, er_mod, hb_mod, picks_mod,
        post_mod, rr_mod, ss_mod, sch_mod, sc_mod, sps_mod, strat_mod, web_mod,
        halert_mod, hcs_mod, hfd_mod, hld_mod, hpe_mod, hpf_mod, hrunner_mod,
    ]
    for mod in dt_modules:
        if hasattr(mod, "datetime"):
            monkeypatch.setattr(mod, "datetime", PinnedDatetime)

    date_modules = [
        cs_mod, do_mod, er_mod, hcs_mod, hfd_mod, hld_mod, hpe_mod, hpf_mod,
        hpp_mod, hrr_mod, hrunner_mod, htp_mod,
    ]
    for mod in date_modules:
        if hasattr(mod, "date"):
            monkeypatch.setattr(mod, "date", PinnedDate)
    if hasattr(web_mod, "_date"):
        monkeypatch.setattr(web_mod, "_date", PinnedDate)

    monkeypatch.setattr(sch_mod, "_now_et", lambda: clock.now_et)
    monkeypatch.setattr(post_mod, "_now_et", lambda: clock.now_et)
    monkeypatch.setattr(cli_mod, "_now_et", lambda: clock.now_et)
    monkeypatch.setattr(sch_mod, "notify_ready", lambda: False)
    monkeypatch.setattr(sch_mod, "notify_watchdog", lambda: False)
    monkeypatch.setattr(sch_mod, "_watchdog_ping_sleep", lambda sec, **kw: clock.advance(sec))
    monkeypatch.setattr(sch_mod, "_poll_interval_sleep", lambda sec: clock.advance(sec))

    def _fast_idle(target_et, heartbeat_path=None):
        if isinstance(target_et, str):
            target_et = real_datetime.fromisoformat(target_et)
        delta = (target_et - clock.now_et).total_seconds()
        if delta > 0:
            clock.advance(delta)

    monkeypatch.setattr(sch_mod, "_idle_until_next_wakeup", _fast_idle)

    @contextlib.contextmanager
    def _noop_watchdog(*args, **kwargs):
        yield

    monkeypatch.setattr(sch_mod, "heartbeat_watchdog", _noop_watchdog)
    yield


def make_live_config(base_dir: Path, *, delivery: str = "dm", intent: str = "enter",
                     required: bool = True, expected: bool = True) -> tuple[dict, Path]:
    picks_dir = base_dir / "data" / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "health_state").mkdir(parents=True, exist_ok=True)
    models_dir = base_dir / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for npz in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        dst = models_dir / npz
        if not dst.exists():
            dst.symlink_to(REPO_ROOT / "data" / "models" / npz)
    cfg = {
        "orchestrator": {
            "picks_dir": str(picks_dir),
            "models_dir": str(models_dir),
            "heartbeat_path": str(base_dir / "data" / ".heartbeat"),
        },
        "bluesky": {
            "dm_recipient": "stonehengee.bsky.social",
        },
        "scheduler": {
            "pick_delivery": delivery,
            "entry_intent": intent,
            "contest_state_required": required,
            "shadow_model": False,
            "live_forward_capture_on_lock": False,
            "default_init_hour_et": 10,
            "lineup_check_offset_min": 150,
            "cluster_min": 30,
            "early_lock_gap": 0.03,
            "fallback_deadline_min": 55,
            "cascade_budget_min": 15,
            "operator_reserve_min": 15,
            "missed_pick_alert_min": 20,
            "results_poll_interval_min": 15,
        },
        "health_checks": {
            "enabled": False,
            "contest_state_expected": expected,
            "thresholds": {
                "dm_delivery_status": str(base_dir / "data" / "health_state" / "health_dm_delivery_status.json"),
                "warn_attention_state": str(base_dir / "data" / "health_state" / "warn_attention.json"),
                "disk_fill_warn_pct": 99.8,
                "disk_fill_crit_pct": 99.9,
            },
        },
        "tiers": [{"name": "local", "type": "local"}],
    }
    toml_lines = [
        "[orchestrator]",
        f'picks_dir = "{picks_dir}"',
        f'models_dir = "{models_dir}"',
        f'heartbeat_path = "{base_dir / "data" / ".heartbeat"}"',
        "",
        "[bluesky]",
        'dm_recipient = "stonehengee.bsky.social"',
        "",
        "[scheduler]",
        f'pick_delivery = "{delivery}"',
        f'entry_intent = "{intent}"',
        f"contest_state_required = {'true' if required else 'false'}",
        "shadow_model = false",
        "live_forward_capture_on_lock = false",
        "",
        "[health_checks]",
        "enabled = false",
        f"contest_state_expected = {'true' if expected else 'false'}",
        "",
        "[health_checks.thresholds]",
        f'dm_delivery_status = "{base_dir / "data" / "health_state" / "health_dm_delivery_status.json"}"',
        f'warn_attention_state = "{base_dir / "data" / "health_state" / "warn_attention.json"}"',
        "",
        "[[tiers]]",
        'name = "local"',
        'type = "local"',
        "",
    ]
    cfg_path = base_dir / "orchestrator.toml"
    cfg_path.write_text("\n".join(toml_lines))
    return cfg, cfg_path


def write_covered_auto_state(
    picks_dir: Path,
    now_et: real_datetime,
    *,
    streak: int,
    best_streak: int,
    saver_state: str = "active",
    source_date: str | None = None,
    recorded_at: real_datetime | None = None,
    activation_id: str = "a" * 32,
) -> None:
    acc = picks_dir / "account_state"
    acc.mkdir(parents=True, exist_ok=True)
    year = now_et.year
    rec_dt = (recorded_at or now_et).astimezone(UTC)
    src_date = source_date if source_date is not None else (now_et.date() - timedelta(days=1)).isoformat()
    (acc / "season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1",
        "season": year,
        "status": "automatic",
        "activation_id": activation_id,
        "switched_at": rec_dt.isoformat().replace("+00:00", "Z"),
        "revoked_at": None,
        "note": None,
    }, indent=2))
    (acc / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": year,
        "activation_id": activation_id,
        "active_streak": streak,
        "best_streak": best_streak,
        "source_date": src_date,
        "recorded_at": rec_dt.isoformat().replace("+00:00", "Z"),
        "source": "mlb_bts_profile",
        "user_id": 12345,
        "username": "stonehengee",
        "saver_available": None,
    }, indent=2))
    (acc / "saver_state.json").write_text(json.dumps({
        "state": saver_state,
        "season": year,
        "updated_at": rec_dt.isoformat(),
        "source": "probe",
    }, indent=2))


def make_two_game_schedule(date_str: str, *, g1_state: str = "Scheduled", g2_state: str = "Scheduled",
                           doubleheader: bool = False) -> list[dict]:
    """Two evening games at 19:10 ET and 20:10 ET (or a same-matchup doubleheader at 16:10 / 19:40 ET)."""
    d = real_date.fromisoformat(date_str)
    if doubleheader:
        t1 = real_datetime(d.year, d.month, d.day, 16, 10, tzinfo=ET).astimezone(UTC)
        t2 = real_datetime(d.year, d.month, d.day, 19, 40, tzinfo=ET).astimezone(UTC)
        return [
            {
                "game_pk": 700101,
                "game_date": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "game_time_et": t1.astimezone(ET),
                "away": "NYY", "home": "BOS",
                "away_id": 147, "home_id": 111,
                "status": g1_state,
                "doubleheader": "Y", "game_num": 1,
            },
            {
                "game_pk": 700102,
                "game_date": t2.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "game_time_et": t2.astimezone(ET),
                "away": "NYY", "home": "BOS",
                "away_id": 147, "home_id": 111,
                "status": g2_state,
                "doubleheader": "Y", "game_num": 2,
            },
        ]
    t1 = real_datetime(d.year, d.month, d.day, 19, 10, tzinfo=ET).astimezone(UTC)
    t2 = real_datetime(d.year, d.month, d.day, 20, 10, tzinfo=ET).astimezone(UTC)
    return [
        {
            "game_pk": 700001,
            "game_date": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "game_time_et": t1.astimezone(ET),
            "away": "NYY", "home": "BOS",
            "away_id": 147, "home_id": 111,
            "status": g1_state,
            "doubleheader": "N", "game_num": 1,
        },
        {
            "game_pk": 700002,
            "game_date": t2.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "game_time_et": t2.astimezone(ET),
            "away": "LAD", "home": "SF",
            "away_id": 119, "home_id": 137,
            "status": g2_state,
            "doubleheader": "N", "game_num": 1,
        },
    ]


def make_predictions_df(
    date_str: str,
    games: list[dict],
    *,
    p1: float = 0.84,
    p2: float = 0.82,
    lineup_confirmed: bool = True,
) -> pd.DataFrame:
    g1, g2 = games[0], games[1]
    flags_val = "" if lineup_confirmed else "PROJECTED"
    rows = [
        {
            "date": date_str,
            "batter_id": 100001,
            "batter_name": "Juan Soto",
            "team": g1["away"],
            "opp_team": g1["home"],
            "pitcher_id": 300001,
            "pitcher_name": "Garrett Crochet",
            "pitcher_team": g1["home"],
            "p_hit": p1 / 4.0,
            "p_game_hit": p1,
            "est_pa": 4.5,
            "lineup": 2,
            "lineup_pos": 2,
            "flags": flags_val,
            "lineup_confirmed": lineup_confirmed,
            "game_pk": g1["game_pk"],
            "game_time": g1["game_date"],
        },
        {
            "date": date_str,
            "batter_id": 100002,
            "batter_name": "Shohei Ohtani",
            "team": g2["away"],
            "opp_team": g2["home"],
            "pitcher_id": 300002,
            "pitcher_name": "Logan Webb",
            "pitcher_team": g2["home"],
            "p_hit": p2 / 4.0,
            "p_game_hit": p2,
            "est_pa": 4.5,
            "lineup": 1,
            "lineup_pos": 1,
            "flags": flags_val,
            "lineup_confirmed": lineup_confirmed,
            "game_pk": g2["game_pk"],
            "game_time": g2["game_date"],
        },
    ]
    return pd.DataFrame(rows)


def make_final_feed(
    game_pk: int,
    date_str: str,
    batter_outcomes: dict[int, tuple[str, str]],
    *,
    abstract_code: str = "F",
    detailed_state: str = "Final",
) -> dict:
    away_players = {}
    away_batters = []
    plays = []
    for idx, (bid, (bname, outcome)) in enumerate(batter_outcomes.items(), start=1):
        away_batters.append(bid)
        if outcome == "hit":
            stats = {"batting": {"atBats": 4, "hits": 1, "baseOnBalls": 0, "sacFlies": 0, "plateAppearances": 4}}
            ev_type = "single"
        elif outcome == "miss":
            stats = {"batting": {"atBats": 4, "hits": 0, "baseOnBalls": 0, "sacFlies": 0, "plateAppearances": 4}}
            ev_type = "strikeout"
        elif outcome == "walk_void":
            stats = {"batting": {"atBats": 0, "hits": 0, "baseOnBalls": 4, "sacFlies": 0, "plateAppearances": 4}}
            ev_type = "walk"
        elif outcome == "dnp_void":
            stats = {"batting": {"atBats": 0, "hits": 0, "baseOnBalls": 0, "sacFlies": 0, "plateAppearances": 0}}
            ev_type = None
        else:
            raise ValueError(outcome)
        away_players[f"ID{bid}"] = {
            "person": {"id": bid, "fullName": bname},
            "battingOrder": str(idx * 100),
            "stats": stats,
        }
        if ev_type is not None:
            plays.append({
                "result": {"eventType": ev_type, "description": f"{bname} {ev_type}."},
                "about": {"halfInning": "top", "inning": 1, "isComplete": True},
                "matchup": {"batter": {"id": bid, "fullName": bname}, "pitcher": {"id": 300001}},
            })
    return {
        "gameData": {
            "datetime": {"officialDate": date_str},
            "game": {"pk": game_pk, "season": date_str[:4], "type": "R"},
            "status": {"abstractGameCode": abstract_code, "detailedState": detailed_state},
            "teams": {"away": {"id": 147, "abbreviation": "NYY"}, "home": {"id": 111, "abbreviation": "BOS"}},
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"batters": away_batters, "players": away_players},
                    "home": {"batters": [], "players": {}},
                }
            },
            "plays": {"allPlays": plays},
        },
    }


def make_raw_schedule_json(date_str: str, games: list[dict]) -> dict:
    return {
        "dates": [
            {
                "date": date_str,
                "games": [
                    {
                        "gamePk": g["game_pk"],
                        "gameType": "R",
                        "gameDate": g["game_date"],
                        "officialDate": date_str,
                        "doubleHeader": g.get("doubleheader", "N"),
                        "gameNumber": g.get("game_num", 1),
                        "status": {
                            "detailedState": g["status"],
                            "abstractGameState": ("Final" if g["status"] in ("Final", "Postponed", "Cancelled") else "Preview"),
                            "abstractGameCode": ("F" if g["status"] in ("Final", "Postponed", "Cancelled") else "P"),
                            "statusCode": ("DR" if g["status"] == "Postponed" else "F" if g["status"] == "Final" else "S"),
                        },
                        "teams": {
                            "away": {"team": {"id": g["away_id"], "name": g["away"], "abbreviation": g["away"]}},
                            "home": {"team": {"id": g["home_id"], "name": g["home"], "abbreviation": g["home"]}},
                        },
                    }
                    for g in games
                ],
            }
        ]
    }


class StubTransports:
    """Captures DMs, public posts, and stubs MLB schedule/feed HTTP endpoints."""

    def __init__(self):
        self.dms: list[tuple[str, str, str]] = []
        self.posts: list[str] = []
        self.schedules_by_date: dict[str, list[dict]] = {}
        self.feeds_by_pk: dict[int, dict] = {}
        self._msg_seq = 0

    def install(self, monkeypatch: pytest.MonkeyPatch, pred_fn) -> None:
        import bts.cli as cli_mod
        import bts.day_outcome as do_mod
        import bts.dm as dm_mod
        import bts.health.alert as halert_mod
        import bts.orchestrator as orch_mod
        import bts.picks as picks_mod
        import bts.posting as post_mod
        import bts.scheduler as sch_mod

        def _send_dm(recipient: str, text: str, **kwargs) -> str:
            self._msg_seq += 1
            mid = f"dm-msg-{self._msg_seq}"
            self.dms.append((recipient, text, mid))
            return mid

        def _post(text: str, **kwargs) -> str:
            self._msg_seq += 1
            uri = f"at://did:plc:test/app.bsky.feed.post/{self._msg_seq}"
            self.posts.append(text)
            return uri

        for mod in (dm_mod, sch_mod, orch_mod, halert_mod, cli_mod, do_mod):
            if hasattr(mod, "send_dm"):
                monkeypatch.setattr(mod, "send_dm", _send_dm)
        for mod in (post_mod, sch_mod, orch_mod, cli_mod):
            if hasattr(mod, "post_to_bluesky"):
                monkeypatch.setattr(mod, "post_to_bluesky", _post)

        def _fetch_schedule(d_str: str) -> list[dict]:
            return list(self.schedules_by_date.get(d_str, []))

        def _statuses_detailed(d_str: str) -> dict[int, dict]:
            games = self.schedules_by_date.get(d_str, [])
            return {
                g["game_pk"]: {
                    "state": g["status"],
                    "away_id": g["away_id"],
                    "home_id": g["home_id"],
                }
                for g in games
            }

        def _count_confirmations(game_pks, confirmed_sides):
            added = 0
            for pk in game_pks:
                for side in ("away", "home"):
                    if (pk, side) not in confirmed_sides:
                        confirmed_sides.add((pk, side))
                        added += 1
            return added

        class _FakeHTTPResp:
            def __init__(self, payload: dict):
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self) -> bytes:
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _fake_urlopen(req, *args, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "/game/" in url and "/feed/live" in url:
                pk_str = url.split("/game/")[1].split("/")[0]
                pk = int(pk_str)
                if pk in self.feeds_by_pk:
                    return _FakeHTTPResp(self.feeds_by_pk[pk])
                raise RuntimeError(f"Stub feed missing for pk={pk}")
            if "api/v1/schedule" in url:
                d_str = None
                if "date=" in url:
                    d_str = url.split("date=")[1].split("&")[0]
                games = self.schedules_by_date.get(d_str or "", [])
                return _FakeHTTPResp(make_raw_schedule_json(d_str or "2027-07-15", games))
            raise AssertionError(f"Unexpected URL in _fake_urlopen: {url}")

        import bts.util as util_mod

        monkeypatch.setattr(sch_mod, "retry_urlopen", _fake_urlopen)
        monkeypatch.setattr(sch_mod, "count_new_confirmations", _count_confirmations)
        monkeypatch.setattr(sch_mod, "_delivery_game_statuses", picks_mod.get_game_statuses_detailed)
        monkeypatch.setattr(picks_mod, "retry_urlopen", _fake_urlopen)
        monkeypatch.setattr(util_mod, "retry_urlopen", _fake_urlopen)
        monkeypatch.setattr(orch_mod, "run_cascade", lambda tiers, d_str: (pred_fn(d_str), "local"))


def stub_bts_contest_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rounds: dict[int, real_date],
    pending_rows: list[dict],
    active_streak: int = 12,
    best_streak: int = 18,
    settled_predictions: list[dict] | None = None,
    bts_to_mlb: dict[int, int] | None = None,
) -> None:
    """Stub MLB BTS auth + profile + pending predictions + rounds + player crosswalk."""
    import bts.cli as cli_mod
    import bts.contest_fetch as cf_mod
    import bts.leaderboard.auth as auth_mod

    monkeypatch.setattr(auth_mod, "load_session_cookies", lambda *a, **k: {"ipid": "12345", "session": "ok"})
    monkeypatch.setattr(auth_mod, "extract_uid", lambda c: 12345)
    monkeypatch.setattr(
        auth_mod,
        "fetch_login_session",
        lambda **kw: SimpleNamespace(user_id=12345, username="stonehengee", xsid="xsid-stub"),
    )
    monkeypatch.setattr(
        cf_mod,
        "fetch_profile",
        lambda uid, cookies, xsid: {
            "activeStreak": active_streak,
            "seasonBestStreak": best_streak,
            "predictions": (settled_predictions if settled_predictions is not None else []),
        },
    )
    monkeypatch.setattr(cf_mod, "fetch_pending_predictions", lambda cookies, xsid: list(pending_rows))
    monkeypatch.setattr(cli_mod, "_fetch_rounds", lambda: dict(rounds))
    monkeypatch.setattr(
        cli_mod,
        "_fetch_bts_to_mlb",
        lambda: (dict(bts_to_mlb) if bts_to_mlb is not None else {9001: 100001, 9002: 100002}),
    )


def write_units_snapshot(
    base_dir: Path,
    year: int,
    mapping: dict[int, tuple[int, int]],
    stamp: str = "20270715T140000Z",
) -> Path:
    """Write units snapshot {unit_id: (feedId, roundId)}."""
    units_dir = base_dir / "data" / "leaderboard" / "static_snapshots" / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    path = units_dir / f"{stamp}.json.gz"
    assert str(year) in stamp, "units snapshot filename must match season year"
    units_payload = {
        "units": [{"id": int(uid), "feedId": int(feed_id), "roundId": int(rnd_id)} for uid, (feed_id, rnd_id) in mapping.items()]
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(units_payload, fh)
    return path


def test_mid_late_finale_automatic_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Exercise 2027-07-15, 2027-09-20, and 2027-09-26 across single, double-down,
    skip (reach57 + tail stop), postponed game (single, 1-leg double, all-void),
    doubleheader (right game vs wrong game + units missing), and void (walk-only Pass)."""
    import bts.cli as cli_mod
    import bts.daily_decision as dd_mod
    import bts.day_outcome as do_mod
    import bts.picks as picks_mod
    import bts.scheduler as sch_mod
    import bts.skip_policy_shadow as sps_mod

    summary: dict[str, object] = {}
    runner = CliRunner()

    # 1. Single, Double, Reach57 Skip, Tail-Stop Skip across 2027-07-15, 2027-09-20, 2027-09-26
    day_cases = [
        # (label, date_str, streak, best, saver, p1, p2, expected_action, expected_obj, feed_outcomes)
        ("mid_single_saver_earned", "2027-07-15", 12, 18, "active", 0.82, 0.72, "single", "reach57",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "miss")}}),
        ("mid_double_saver_rescue", "2027-07-15", 12, 18, "active", 0.835, 0.83, "double", "reach57",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "miss")}}),
        ("mid_skip_reach57", "2027-07-15", 12, 18, "active", 0.72, 0.70, "skip", "reach57",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "hit")}}),
        ("late_tail_single", "2027-09-20", 12, 14, "active", 0.84, 0.82, "single", "emax_season_best",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "hit")}}),
        ("late_tail_double", "2027-09-20", 12, 25, "active", 0.84, 0.82, "double", "emax_season_best",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "hit")}}),
        ("late_tail_stop_skip", "2027-09-20", 5, 25, "active", 0.85, 0.84, "skip", "emax_season_best",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "hit")}}),
        ("late_reach57_streak51", "2027-09-20", 51, 51, "active", 0.85, 0.84, "double", "reach57",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "hit")}}),
        ("finale_streak55_double", "2027-09-26", 55, 55, "active", 0.85, 0.84, "double", "reach57",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "hit")}}),
        ("finale_streak56_single", "2027-09-26", 56, 56, "active", 0.85, 0.84, "single", "reach57",
         {700001: {100001: ("Juan Soto", "hit")}, 700002: {100002: ("Shohei Ohtani", "hit")}}),
        ("mid_void_walk_only", "2027-07-15", 12, 18, "active", 0.82, 0.72, "single", "reach57",
         {700001: {100001: ("Juan Soto", "walk_void")}, 700002: {100002: ("Shohei Ohtani", "miss")}}),
    ]

    for label, d_str, streak, best, saver, p1, p2, exp_action, exp_obj, outcomes in day_cases:
        case_dir = tmp_path / label
        d = real_date.fromisoformat(d_str)
        start_et = real_datetime(d.year, d.month, d.day, 10, 0, tzinfo=ET)
        clock = ClockController(start_et)
        with monkeypatch.context() as mp, pinned_environment(mp, clock):
            cfg, cfg_path = make_live_config(case_dir)
            picks_dir = Path(cfg["orchestrator"]["picks_dir"])
            write_covered_auto_state(picks_dir, start_et, streak=streak, best_streak=best, saver_state=saver)
            picks_mod.save_streak(streak, picks_dir, saver_available=(saver == "active"), season=2027)

            transports = StubTransports()
            games = make_two_game_schedule(d_str)
            next_d = (d + timedelta(days=1)).isoformat()
            transports.schedules_by_date[d_str] = games
            transports.schedules_by_date[next_d] = make_two_game_schedule(next_d)
            for pk, bmap in outcomes.items():
                transports.feeds_by_pk[pk] = make_final_feed(pk, d_str, bmap)

            transports.install(mp, lambda ds: make_predictions_df(ds, games, p1=p1, p2=p2))

            sch_mod.run_day(d_str, cfg)

            dec = dd_mod.load_decision(d_str, picks_dir)
            assert dec is not None, f"{label}: missing decision.json"
            assert dec["action"] == exp_action, f"{label}: got action={dec['action']} expected={exp_action}"
            assert dd_mod.decision_objective(dec) == exp_obj, f"{label}: obj={dd_mod.decision_objective(dec)}"

            if exp_action in ("single", "double"):
                daily = picks_mod.load_pick(d_str, picks_dir)
                assert daily is not None and picks_mod.pick_was_delivered(daily)
                assert daily.chosen_on is not None and daily.chosen_on["required"] is True
                assert daily.chosen_on["kind"] == "automatic"
                assert daily.automatic_reading is not None and daily.automatic_reading["stale"] is False

                stamp = d_str.replace("-", "") + "T140000Z"
                write_units_snapshot(case_dir, 2027, {101: (700001, 1), 102: (700002, 1)}, stamp=stamp)
                pending = [{"roundId": 1, "unitId": 101, "playerId": 9001, "number": 1}]
                if exp_action == "double":
                    pending.append({"roundId": 1, "unitId": 102, "playerId": 9002, "number": 2})

                clock.set_et(real_datetime(d.year, d.month, d.day, 18, 30, tzinfo=ET))
                stub_bts_contest_endpoints(
                    mp, rounds={1: d}, pending_rows=pending, active_streak=streak, best_streak=best
                )
                res_cpe = runner.invoke(cli_mod.cli, [
                    "check-pick-entered", "--cron",
                    "--picks-dir", str(picks_dir),
                    "--expected-username", "stonehengee",
                    "--dm-recipient", "stonehengee.bsky.social",
                ])
                assert res_cpe.exit_code == 0, f"{label} check-pick-entered failed: {res_cpe.output}"

                nd = d + timedelta(days=1)
                clock.set_et(real_datetime(nd.year, nd.month, nd.day, 8, 30, tzinfo=ET))
                dms_before_do = len(transports.dms)
                res_do = runner.invoke(cli_mod.cli, [
                    "day-outcome", "--date", d_str,
                    "--picks-dir", str(picks_dir), "--config", str(cfg_path),
                    "--expected-username", "stonehengee",
                    "--dm-recipient", "stonehengee.bsky.social",
                ])
                assert res_do.exit_code == 0, f"{label} day-outcome failed: {res_do.output}"
                assert len(transports.dms) == dms_before_do, f"{label} unexpected day-outcome DM"
                out_json = json.loads(res_do.output)
                assert out_json["state"] == "entered" and out_json["ok"] is True
                if exp_action == "single":
                    assert "Shohei Ohtani" not in (out_json.get("pick") or ""), "W8: runner-up leaked into single"
                if label == "mid_void_walk_only":
                    assert daily.result == "void", f"Expected void on walk-only, got {daily.result}"
                    s_data = json.loads((picks_dir / "streak.json").read_text())
                    assert s_data["streak"] == 12 and s_data["saver_available"] is True
                elif label == "mid_double_saver_rescue":
                    s_data = json.loads((picks_dir / "streak.json").read_text())
                    assert daily.result == "miss" and s_data["saver_available"] is False
                    assert s_data["streak"] > 0, f"Saver should rescue streak>=10: {s_data}"
            else:
                skip_dms = [txt for _, txt, _ in transports.dms if "No pick at this check." in txt]
                assert len(skip_dms) == 1, f"{label}: expected 1 skip DM, got {transports.dms}"
                assert f"Streak used at this check: {streak}." in skip_dms[0]
                sps_rec = sps_mod.record_skip_from_decision(d_str, picks_dir, now=clock.now_utc)
                if exp_obj == "reach57":
                    assert sps_rec is not None, "reach57 skip should record policy_shadow"
                else:
                    assert sps_rec is None, "tail skip must NOT record reach57 policy_shadow"

                nd = d + timedelta(days=1)
                clock.set_et(real_datetime(nd.year, nd.month, nd.day, 8, 30, tzinfo=ET))
                dms_before_do = len(transports.dms)
                res_do = runner.invoke(cli_mod.cli, [
                    "day-outcome", "--date", d_str,
                    "--picks-dir", str(picks_dir), "--config", str(cfg_path),
                    "--dm-recipient", "stonehengee.bsky.social",
                ])
                assert res_do.exit_code == 0 and len(transports.dms) == dms_before_do
                out_json = json.loads(res_do.output)
                assert out_json["state"] == "deliberate_skip" and out_json["ok"] is True

            summary[label] = {
                "action": dec["action"],
                "objective": dd_mod.decision_objective(dec),
                "dms_sent": len(transports.dms),
            }

    # 2. Postponed game scenarios on 2027-07-15:
    pp_dir = tmp_path / "postponed_single"
    d_str = "2027-07-15"
    start_et = real_datetime(2027, 7, 15, 10, 0, tzinfo=ET)
    clock = ClockController(start_et)
    with monkeypatch.context() as mp, pinned_environment(mp, clock):
        cfg, cfg_path = make_live_config(pp_dir)
        cfg["health_checks"]["enabled"] = True
        picks_dir = Path(cfg["orchestrator"]["picks_dir"])
        models_dir = Path(cfg["orchestrator"]["models_dir"])
        (models_dir / "blend_2027-07-16.pkl").write_bytes(b"stub")
        write_covered_auto_state(picks_dir, start_et, streak=12, best_streak=18, saver_state="active")
        picks_mod.save_streak(12, picks_dir, saver_available=True, season=2027)

        transports = StubTransports()
        games_initial = make_two_game_schedule(d_str, g1_state="Scheduled", g2_state="Scheduled")
        games_g1_ppd = make_two_game_schedule(d_str, g1_state="Postponed", g2_state="Scheduled")
        transports.schedules_by_date[d_str] = games_initial
        transports.schedules_by_date["2027-07-16"] = make_two_game_schedule("2027-07-16")
        transports.feeds_by_pk[700001] = make_final_feed(700001, d_str, {}, abstract_code="F", detailed_state="Postponed")
        transports.feeds_by_pk[700002] = make_final_feed(700002, d_str, {100002: ("Shohei Ohtani", "hit")})
        transports.install(mp, lambda ds: make_predictions_df(ds, games_initial, p1=0.85, p2=0.83))

        def _dynamic_delivery_statuses(ds: str) -> dict[int, dict]:
            transports.schedules_by_date[d_str] = games_g1_ppd
            return picks_mod.get_game_statuses_detailed(ds)

        mp.setattr(sch_mod, "_delivery_game_statuses", _dynamic_delivery_statuses)
        sch_mod.run_day(d_str, cfg)

        refusal_dms = [txt for _, txt, _ in transports.dms if "DELIVERY REFUSED" in txt or "postponed" in txt.lower()]
        assert len(refusal_dms) >= 1, f"Expected postponed refusal DM, got {transports.dms}"
        assert "This pick was withheld." in refusal_dms[0]
        # Note: in-loop refusal dispatches source='postponed_pick', while EOD late_delivery.check emits source='late_delivery',
        # producing a second EOD CRITICAL DM even after Game 2 re-picked, delivered, and hit!
        eod_dup_dms = [txt for _, txt, _ in transports.dms if "[late_delivery]" in txt and "game 700001 is Postponed" in txt]
        assert len(eod_dup_dms) == 1, f"Expected EOD late_delivery second CRITICAL DM, got {transports.dms}"
        final_daily = picks_mod.load_pick(d_str, picks_dir)
        assert final_daily is not None and final_daily.pick.game_pk == 700002
        assert picks_mod.pick_was_delivered(final_daily) and final_daily.result == "hit"
        summary["postponed_single_repick"] = "refused_g1_then_delivered_g2_hit_plus_eod_late_delivery_dm"

    # (b) Double-down where leg 1 turns Postponed before delivery -> delivered UNCHANGED (R13/L03), graded void+hit -> hit!
    pp_dd_dir = tmp_path / "postponed_double_one_leg"
    clock = ClockController(start_et)
    with monkeypatch.context() as mp, pinned_environment(mp, clock):
        cfg, _ = make_live_config(pp_dd_dir)
        picks_dir = Path(cfg["orchestrator"]["picks_dir"])
        write_covered_auto_state(picks_dir, start_et, streak=0, best_streak=18, saver_state="active")
        picks_mod.save_streak(0, picks_dir, saver_available=True, season=2027)
        transports = StubTransports()
        games_initial = make_two_game_schedule(d_str, g1_state="Scheduled", g2_state="Scheduled")
        transports.schedules_by_date[d_str] = games_initial
        transports.schedules_by_date["2027-07-16"] = make_two_game_schedule("2027-07-16")
        transports.feeds_by_pk[700001] = make_final_feed(700001, d_str, {}, abstract_code="F", detailed_state="Postponed")
        transports.feeds_by_pk[700002] = make_final_feed(700002, d_str, {100002: ("Shohei Ohtani", "hit")})
        transports.install(mp, lambda ds: make_predictions_df(ds, games_initial, p1=0.85, p2=0.84))
        mp.setattr(
            sch_mod,
            "_delivery_game_statuses",
            lambda ds: {
                700001: {"abstract": "F", "detailed": "Postponed", "code": "DR"},
                700002: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
            },
        )
        orig_deliver = sch_mod._deliver_and_lock_pick

        def _wrapped_deliver(*a, **kw):
            ok = orig_deliver(*a, **kw)
            transports.schedules_by_date[d_str] = make_two_game_schedule(d_str, g1_state="Postponed", g2_state="Final")
            return ok

        mp.setattr(sch_mod, "_deliver_and_lock_pick", _wrapped_deliver)
        sch_mod.run_day(d_str, cfg)

        dd_daily = picks_mod.load_pick(d_str, picks_dir)
        assert dd_daily is not None and dd_daily.double_down is not None
        assert picks_mod.pick_was_delivered(dd_daily)
        assert dd_daily.slot_results == {"pick": "void", "double_down": "hit"}
        assert dd_daily.result == "hit"
        assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 1
        summary["postponed_double_one_leg"] = dd_daily.slot_results

    # (c) All-postponed slate -> day-outcome classifies as no_games (ok=True)
    all_ppd_dir = tmp_path / "all_postponed"
    clock = ClockController(real_datetime(2027, 7, 16, 8, 30, tzinfo=ET))
    with monkeypatch.context() as mp, pinned_environment(mp, clock):
        cfg, cfg_path = make_live_config(all_ppd_dir)
        picks_dir = Path(cfg["orchestrator"]["picks_dir"])
        transports = StubTransports()
        transports.schedules_by_date[d_str] = make_two_game_schedule(d_str, g1_state="Postponed", g2_state="Postponed")
        transports.install(mp, lambda ds: pd.DataFrame())
        out = do_mod.gather(d_str, picks_dir, cfg, expected_username="stonehengee")
        assert out.state == "no_games" and out.ok is True
        summary["all_postponed_day_outcome"] = out.state

    # 3. Doubleheader: right batter entered in Game 2 (wrong game) -> F1/F1b alert + entry_unverifiable,
    #    then switched to Game 1 (right game) -> all-clear + entered; plus missing units -> entered_game_unverified!
    dh_dir = tmp_path / "doubleheader"
    d_dh = real_date(2027, 7, 15)
    clock = ClockController(real_datetime(2027, 7, 15, 10, 0, tzinfo=ET))
    with monkeypatch.context() as mp, pinned_environment(mp, clock):
        cfg, cfg_path = make_live_config(dh_dir)
        picks_dir = Path(cfg["orchestrator"]["picks_dir"])
        write_covered_auto_state(picks_dir, clock.now_et, streak=12, best_streak=18, saver_state="active")
        picks_mod.save_streak(12, picks_dir, saver_available=True, season=2027)
        transports = StubTransports()
        dh_games = make_two_game_schedule(d_str, doubleheader=True)
        transports.schedules_by_date[d_str] = dh_games
        transports.schedules_by_date["2027-07-16"] = make_two_game_schedule("2027-07-16")
        transports.feeds_by_pk[700101] = make_final_feed(700101, d_str, {100001: ("Juan Soto", "hit")})
        transports.feeds_by_pk[700102] = make_final_feed(700102, d_str, {100001: ("Juan Soto", "hit")})
        transports.install(mp, lambda ds: make_predictions_df(ds, dh_games, p1=0.82, p2=0.72))
        sch_mod.run_day(d_str, cfg)

        dh_daily = picks_mod.load_pick(d_str, picks_dir)
        assert dh_daily.pick.game_pk == 700101 and dh_daily.double_down is None

        # Units snapshot binds unit 201 -> (700101, round 1), unit 202 -> (700102, round 1)
        write_units_snapshot(dh_dir, 2027, {201: (700101, 1), 202: (700102, 1)}, stamp="20270715T120000Z")

        # (a) Operator entered Juan Soto (bts id 9001 -> mlb 100001) in Game 2 (unitId=202 -> 700102) at 15:15 ET
        clock.set_et(real_datetime(2027, 7, 15, 15, 15, tzinfo=ET))
        stub_bts_contest_endpoints(
            mp,
            rounds={1: d_dh},
            pending_rows=[{"roundId": 1, "unitId": 202, "playerId": 9001, "number": 1}],
        )
        dms_before = len(transports.dms)
        res_wg = runner.invoke(cli_mod.cli, [
            "check-pick-entered", "--cron",
            "--picks-dir", str(picks_dir),
            "--expected-username", "stonehengee",
            "--dm-recipient", "stonehengee.bsky.social",
        ])
        assert res_wg.exit_code == 0 or res_wg.exit_code == 1
        assert len(transports.dms) == dms_before + 1, f"Expected wrong_game DM, output={res_wg.output}"
        wg_dm = transports.dms[-1][1]
        assert "BTS entry game needs verification" in wg_dm
        marker = json.loads((dh_dir / "data" / "health_state" / "pick_entry_check.json").read_text())
        assert marker["status"] == "alerted" and marker["game_unresolved"] is True
        assert marker["escalations"] == ["wrong_game"]

        clock.set_et(real_datetime(2027, 7, 16, 8, 30, tzinfo=ET))
        out_wg = do_mod.gather(d_str, picks_dir, cfg, expected_username="stonehengee")
        assert out_wg.state == "entry_unverifiable" and out_wg.ok is False

        # (b) Operator fixes entry to Game 1 (unitId=201 -> 700101) at 15:30 ET -> all-clear + entered!
        clock.set_et(real_datetime(2027, 7, 15, 15, 30, tzinfo=ET))
        stub_bts_contest_endpoints(
            mp,
            rounds={1: d_dh},
            pending_rows=[{"roundId": 1, "unitId": 201, "playerId": 9001, "number": 1}],
        )
        dms_before = len(transports.dms)
        res_ok = runner.invoke(cli_mod.cli, [
            "check-pick-entered", "--cron",
            "--picks-dir", str(picks_dir),
            "--expected-username", "stonehengee",
            "--dm-recipient", "stonehengee.bsky.social",
        ])
        assert res_ok.exit_code == 0
        assert len(transports.dms) == dms_before + 1 and "✅" in transports.dms[-1][1]
        marker_ok = json.loads((dh_dir / "data" / "health_state" / "pick_entry_check.json").read_text())
        assert marker_ok["status"] == "confirmed" and "game_unresolved" not in marker_ok

        clock.set_et(real_datetime(2027, 7, 16, 8, 30, tzinfo=ET))
        out_ok = do_mod.gather(d_str, picks_dir, cfg, expected_username="stonehengee")
        assert out_ok.state == "entered" and out_ok.ok is True
        summary["doubleheader"] = {"wrong_game": out_wg.state, "right_game": out_ok.state}

    # (c) Doubleheader without units snapshot -> confirmed (match_game_unchecked) + entered_game_unverified (ok=True)
    dh_nounits_dir = tmp_path / "doubleheader_no_units"
    clock = ClockController(real_datetime(2027, 7, 15, 10, 0, tzinfo=ET))
    with monkeypatch.context() as mp, pinned_environment(mp, clock):
        cfg, cfg_path = make_live_config(dh_nounits_dir)
        picks_dir = Path(cfg["orchestrator"]["picks_dir"])
        write_covered_auto_state(picks_dir, clock.now_et, streak=12, best_streak=18, saver_state="active")
        picks_mod.save_streak(12, picks_dir, saver_available=True, season=2027)
        transports = StubTransports()
        dh_games = make_two_game_schedule(d_str, doubleheader=True)
        transports.schedules_by_date[d_str] = dh_games
        transports.schedules_by_date["2027-07-16"] = make_two_game_schedule("2027-07-16")
        transports.feeds_by_pk[700101] = make_final_feed(700101, d_str, {100001: ("Juan Soto", "hit")})
        transports.feeds_by_pk[700102] = make_final_feed(700102, d_str, {100001: ("Juan Soto", "hit")})
        transports.install(mp, lambda ds: make_predictions_df(ds, dh_games, p1=0.82, p2=0.72))
        sch_mod.run_day(d_str, cfg)

        clock.set_et(real_datetime(2027, 7, 15, 15, 15, tzinfo=ET))
        stub_bts_contest_endpoints(
            mp,
            rounds={1: d_dh},
            pending_rows=[{"roundId": 1, "unitId": 202, "playerId": 9001, "number": 1}],
        )
        res_nu = runner.invoke(cli_mod.cli, [
            "check-pick-entered", "--cron",
            "--picks-dir", str(picks_dir),
            "--expected-username", "stonehengee",
            "--dm-recipient", "stonehengee.bsky.social",
        ])
        assert res_nu.exit_code == 0
        clock.set_et(real_datetime(2027, 7, 16, 8, 30, tzinfo=ET))
        out_nu = do_mod.gather(d_str, picks_dir, cfg, expected_username="stonehengee")
        assert out_nu.state == "entered_game_unverified" and out_nu.ok is True
        summary["doubleheader"]["no_units"] = out_nu.state

    return summary


def test_streaks_55_to_400_and_all_readers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Exercise streaks 55, 56, 57, 58, 75, 120, 400 (and 401 boundary refusal) across multiple day counts
    and verify every streak reader after hitting at 57 and advancing to MLB's 58+."""
    import bts.cli as cli_mod
    import bts.contest_state as cs_mod
    import bts.day_outcome as do_mod
    import bts.health.runner as hrunner_mod
    import bts.picks as picks_mod
    import bts.scheduler as sch_mod
    import bts.strategy as strat_mod
    import bts.web as web_mod

    runner = CliRunner()
    results: dict[str, object] = {}

    dates_for_d = {
        180: "2027-03-31",
        74: "2027-07-15",
        28: "2027-08-30",
        7: "2027-09-20",
        4: "2027-09-23",
        3: "2027-09-24",
        2: "2027-09-25",
        1: "2027-09-26",
        0: "2027-09-27",
    }
    matrix_dir = tmp_path / "policy_matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    matrix_summary = {}
    for d_rem, d_str in dates_for_d.items():
        assert strat_mod._days_remaining(d_str) == d_rem
        games = make_two_game_schedule(d_str)
        preds = make_predictions_df(d_str, games, p1=0.84, p2=0.82)
        gstat = {
            700001: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
            700002: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        }
        for s in (55, 56, 57, 58, 75, 120, 400):
            sel = strat_mod.select_pick(
                preds, d_str, matrix_dir,
                streak=s, saver_available=False, allow_double=True,
                game_statuses_detailed=gstat,
                best_streak=max(s, 58), best_status="trusted",
            )
            if d_rem == 0:
                assert sel.action == "skip"
            elif s >= 57:
                assert sel.decision.objective == "longest_streak"
                expected_act = "double" if d_rem <= 3 else "single"
                assert sel.action == expected_act, f"s={s} d={d_rem}: got {sel.action} expected {expected_act}"
            elif s == 56:
                assert sel.decision.objective == "reach57"
            elif s == 55:
                assert sel.decision.objective == "reach57"
            matrix_summary[f"s{s}_d{d_rem}"] = (sel.action, sel.decision.objective)
    results["matrix_count"] = len(matrix_summary)

    # Full 57 -> hit -> MLB fetch 58 -> play & grade 58 -> 59 (mid-season) AND finale 58 -> 60 double!
    prog_dir = tmp_path / "progression_57_58"
    d1_str, d2_str = "2027-07-14", "2027-07-15"
    clock = ClockController(real_datetime(2027, 7, 14, 10, 0, tzinfo=ET))
    with monkeypatch.context() as mp, pinned_environment(mp, clock):
        cfg, cfg_path = make_live_config(prog_dir)
        picks_dir = Path(cfg["orchestrator"]["picks_dir"])
        write_covered_auto_state(
            picks_dir, clock.now_et, streak=57, best_streak=57, saver_state="used", source_date="2027-07-13"
        )
        picks_mod.save_streak(57, picks_dir, saver_available=False, applied_date="2027-07-13", season=2027)

        transports = StubTransports()
        for ds in (d1_str, d2_str, "2027-07-16"):
            transports.schedules_by_date[ds] = make_two_game_schedule(ds)
        transports.feeds_by_pk[700001] = make_final_feed(700001, d1_str, {100001: ("Juan Soto", "hit")})
        transports.feeds_by_pk[700002] = make_final_feed(700002, d1_str, {100002: ("Shohei Ohtani", "hit")})
        transports.install(mp, lambda ds: make_predictions_df(ds, transports.schedules_by_date[ds], p1=0.84, p2=0.82))

        sch_mod.run_day(d1_str, cfg)
        d1_pick = picks_mod.load_pick(d1_str, picks_dir)
        assert d1_pick.result == "hit"
        assert d1_pick.policy_decision["objective"] == "longest_streak"
        assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 58

        # Overnight 01:10 ET on 2027-07-15: `bts fetch-contest-streak` installs MLB's 58/58!
        clock.set_et(real_datetime(2027, 7, 15, 1, 10, tzinfo=ET))
        stub_bts_contest_endpoints(
            mp,
            rounds={10: real_date(2027, 7, 14)},
            pending_rows=[],
            active_streak=58,
            best_streak=58,
            settled_predictions=[{"roundId": 10, "result": "hit", "roundPredictions": [{"playerId": 9001, "unitId": 101}]}],
        )
        res_fetch = runner.invoke(cli_mod.cli, [
            "fetch-contest-streak", "--picks-dir", str(picks_dir), "--dm-recipient", "stonehengee.bsky.social"
        ])
        assert res_fetch.exit_code == 0, f"fetch-contest-streak failed on 58: {res_fetch.output}"
        auto_58 = json.loads((picks_dir / "account_state" / "contest_streak.json").read_text())
        assert auto_58["active_streak"] == 58 and auto_58["best_streak"] == 58
        assert auto_58["schema_version"] == "bts_contest_streak_auto_v2"

        res_status = runner.invoke(cli_mod.cli, ["season-contract", "status", "--picks-dir", str(picks_dir)])
        assert res_status.exit_code == 0
        assert "live decision now: accepted; source=contest; streak=58;" in res_status.output
        assert "automatic path alone: accepted (streak 58," in res_status.output

        ds_58 = cs_mod.load_decision_streak_state(picks_dir, require_contest_state=True, now=clock.now_utc)
        assert ds_58.streak == 58 and ds_58.best_streak == 58 and ds_58.best_status == "trusted"

        # Day 2 (2027-07-15) at 10:00 ET: morning check + run_day picks, delivers, and grades 58 -> 59!
        clock.set_et(real_datetime(2027, 7, 15, 10, 0, tzinfo=ET))
        transports.feeds_by_pk[700001] = make_final_feed(700001, d2_str, {100001: ("Juan Soto", "hit")})
        transports.feeds_by_pk[700002] = make_final_feed(700002, d2_str, {100002: ("Shohei Ohtani", "hit")})
        sch_mod.run_day(d2_str, cfg)

        d2_pick = picks_mod.load_pick(d2_str, picks_dir)
        assert d2_pick is not None and picks_mod.pick_was_delivered(d2_pick)
        assert d2_pick.chosen_on["streak"] == 58
        assert picks_mod.saved_pick_refusal(d2_pick, ds_58) is None
        assert d2_pick.result == "hit"
        assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 59

        res_rec = runner.invoke(cli_mod.cli, ["reconcile", "--picks-dir", str(picks_dir), "--lookback", "3"])
        assert res_rec.exit_code == 0, f"reconcile failed at streak 59: {res_rec.output}"

        write_units_snapshot(prog_dir, 2027, {101: (700001, 11), 102: (700002, 11)}, stamp="20270715T140000Z")
        clock.set_et(real_datetime(2027, 7, 15, 18, 30, tzinfo=ET))
        stub_bts_contest_endpoints(
            mp,
            rounds={10: real_date(2027, 7, 14), 11: real_date(2027, 7, 15)},
            pending_rows=[{"roundId": 11, "unitId": 101, "playerId": 9001, "number": 1}],
            active_streak=58,
            best_streak=58,
        )
        assert runner.invoke(cli_mod.cli, [
            "check-pick-entered", "--cron", "--picks-dir", str(picks_dir), "--expected-username", "stonehengee",
        ]).exit_code == 0

        clock.set_et(real_datetime(2027, 7, 15, 23, 30, tzinfo=ET))
        alerts_58 = hrunner_mod.run_all_checks(
            picks_dir=picks_dir,
            models_dir=Path(cfg["orchestrator"]["models_dir"]),
            dm_recipient=None,
            contest_state_expected=True,
            today=real_date(2027, 7, 15),
        )
        cs_alerts = [
            a for a in alerts_58
            if a.source in ("contest_state", "saver_state", "streak_validation", "tail_policy")
            and a.level in ("WARN", "CRITICAL")
        ]
        assert cs_alerts == [], f"Unexpected contest/streak health alerts at 58->59: {cs_alerts}"

        web_ctx = web_mod.saver_dashboard_context(picks_dir, now=clock.now_utc)
        assert web_ctx.state == "used" and web_ctx.warning is False

        clock.set_et(real_datetime(2027, 7, 16, 8, 30, tzinfo=ET))
        out_58 = do_mod.gather(d2_str, picks_dir, cfg, expected_username="stonehengee")
        assert out_58.state == "entered" and out_58.ok is True and out_58.streak_problem is None

        # Test well above 57 (120 and 400) on BOTH dated observation and automatic fetch, plus 401 refusal!
        clock.set_et(real_datetime(2027, 7, 16, 9, 30, tzinfo=ET))
        for valid_s, valid_b in ((120, 150), (400, 400)):
            res_set = runner.invoke(cli_mod.cli, [
                "set-contest-streak", "--date", "2027-07-16",
                "--streak", str(valid_s), "--best-streak", str(valid_b),
                "--saver", "used", "--picks-dir", str(picks_dir),
            ])
            assert res_set.exit_code == 0, f"set-contest-streak rejected {valid_s}/{valid_b}: {res_set.output}"
            ds_high = cs_mod.load_decision_streak_state(picks_dir, require_contest_state=True, now=clock.now_utc)
            assert ds_high.streak == valid_s and ds_high.best_streak == valid_b and ds_high.best_status == "trusted"
            d2_pick.chosen_on = picks_mod.chosen_on_record(ds_high, required=True)
            assert picks_mod.saved_pick_refusal(d2_pick, ds_high) is None

            stub_bts_contest_endpoints(
                mp,
                rounds={12: real_date(2027, 7, 15)},
                pending_rows=[],
                active_streak=valid_s,
                best_streak=valid_b,
                settled_predictions=[{"roundId": 12, "result": "hit", "roundPredictions": []}],
            )
            assert runner.invoke(cli_mod.cli, ["fetch-contest-streak", "--picks-dir", str(picks_dir)]).exit_code == 0

        res_401 = runner.invoke(cli_mod.cli, [
            "set-contest-streak", "--date", "2027-07-16",
            "--streak", "401", "--best-streak", "401",
            "--saver", "used", "--picks-dir", str(picks_dir),
        ])
        assert res_401.exit_code != 0
        d2_pick.chosen_on["streak"] = 401
        d2_pick.chosen_on["dated"]["active_streak"] = 401
        d2_pick.chosen_on["dated"]["best_streak"] = 401
        assert picks_mod._valid_chosen_on(d2_pick.chosen_on) is None

        stub_bts_contest_endpoints(
            mp, rounds={12: real_date(2027, 7, 15)}, pending_rows=[], active_streak=401, best_streak=401
        )
        res_fetch_401 = runner.invoke(cli_mod.cli, ["fetch-contest-streak", "--picks-dir", str(picks_dir)])
        assert res_fetch_401.exit_code != 0

    # Also test 2027-09-26 finale at streak 58 -> plays DOUBLE (d=1 <= 3) and grades 58 -> 60!
    fin_dir = tmp_path / "finale_58_double"
    clock = ClockController(real_datetime(2027, 9, 26, 10, 0, tzinfo=ET))
    with monkeypatch.context() as mp, pinned_environment(mp, clock):
        cfg, _ = make_live_config(fin_dir)
        cfg["scheduler"]["health_checks_enabled"] = False
        picks_dir = Path(cfg["orchestrator"]["picks_dir"])
        write_covered_auto_state(
            picks_dir, clock.now_et, streak=58, best_streak=58, saver_state="used", source_date="2027-09-25"
        )
        picks_mod.save_streak(58, picks_dir, saver_available=False, season=2027)
        transports = StubTransports()
        transports.schedules_by_date["2027-09-26"] = make_two_game_schedule("2027-09-26")
        transports.feeds_by_pk[700001] = make_final_feed(700001, "2027-09-26", {100001: ("Juan Soto", "hit")})
        transports.feeds_by_pk[700002] = make_final_feed(700002, "2027-09-26", {100002: ("Shohei Ohtani", "hit")})
        transports.install(mp, lambda ds: make_predictions_df(ds, transports.schedules_by_date[ds], p1=0.84, p2=0.82))
        sch_mod.run_day("2027-09-26", cfg)
        fin_pick = picks_mod.load_pick("2027-09-26", picks_dir)
        assert fin_pick.double_down is not None and fin_pick.result == "hit"
        assert json.loads((picks_dir / "streak.json").read_text())["streak"] == 60

    results["progression_57_58_59"] = "clean"
    results["finale_58_to_60_double"] = "clean"
    results["cap_400_accepted_401_rejected"] = "verified"
    return results


def _run_basesha_comparison_subprocess(basesha_root: Path, work_dir: Path) -> dict:
    """Run BASESHA (619abda7) in a clean subprocess on both exact 2027 dates and matched-day-count 2026 dates."""
    work_dir.mkdir(parents=True, exist_ok=True)
    driver = work_dir / "_basesha_driver.py"
    driver.write_text(
        """
import contextlib, datetime as dt_stdlib, json, socket, sys
from datetime import date as real_date, datetime as real_datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

_real_sock = socket.socket
class _BlockedSock(_real_sock):
    def __init__(self, *a, **kw):
        raise AssertionError("network blocked")
socket.socket = _BlockedSock
socket.create_connection = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network blocked"))

import bts.cli as cli_mod
import bts.daily_decision as dd_mod
import bts.dm as dm_mod
import bts.health.alert as halert_mod
import bts.orchestrator as orch_mod
import bts.picks as picks_mod
import bts.posting as post_mod
import bts.scheduler as sch_mod
import bts.strategy as strat_mod

out = {"dates_2027": {}, "matched_2026": {}}
root = Path(sys.argv[1])

for d_str in ("2027-07-15", "2027-09-20", "2027-09-26"):
    d_rem = strat_mod._days_remaining(d_str)
    preds = pd.DataFrame([
        {"date": d_str, "batter_id": 100001, "batter_name": "Juan Soto", "team": "NYY", "opp_team": "BOS",
         "pitcher_id": 300001, "pitcher_name": "P1", "pitcher_team": "BOS", "p_hit": 0.21, "p_game_hit": 0.85,
         "est_pa": 4.5, "lineup": 2, "flags": "", "lineup_confirmed": True, "game_pk": 700001, "game_time": f"{d_str}T23:10:00Z"},
        {"date": d_str, "batter_id": 100002, "batter_name": "Shohei Ohtani", "team": "LAD", "opp_team": "SF",
         "pitcher_id": 300002, "pitcher_name": "P2", "pitcher_team": "SF", "p_hit": 0.21, "p_game_hit": 0.84,
         "est_pa": 4.5, "lineup": 1, "flags": "", "lineup_confirmed": True, "game_pk": 700002, "game_time": f"{d_str}T23:40:00Z"},
    ])
    pdir = root / f"p_{d_str}"
    pdir.mkdir(parents=True, exist_ok=True)
    gstat = {700001: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
             700002: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
    sel_12 = strat_mod.select_pick(preds, d_str, pdir, streak=12, saver_available=True, allow_double=True,
                                   game_statuses_detailed=gstat, best_streak=18, best_status="trusted")
    sel_57 = strat_mod.select_pick(preds, d_str, pdir, streak=57, saver_available=False, allow_double=True,
                                   game_statuses_detailed=gstat, best_streak=57, best_status="trusted")
    out["dates_2027"][d_str] = {
        "days_remaining": d_rem,
        "streak12_action": sel_12.action,
        "streak12_obj": sel_12.decision.objective,
        "streak57_action": sel_57.action,
        "streak57_obj": sel_57.decision.objective,
    }

for d_str in ("2026-07-16", "2026-09-21", "2026-09-27"):
    d = real_date.fromisoformat(d_str)
    now_holder = {"utc": real_datetime(d.year, d.month, d.day, 10, 0, tzinfo=ET).astimezone(UTC)}
    class PinnedDT(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return now_holder["utc"].astimezone(tz) if tz else now_holder["utc"].astimezone(ET).replace(tzinfo=None)
    dt_stdlib.datetime = PinnedDT
    for m in (sch_mod, post_mod, cli_mod, dd_mod, picks_mod, halert_mod):
        if hasattr(m, "datetime"):
            m.datetime = PinnedDT
    sch_mod._now_et = lambda: now_holder["utc"].astimezone(ET)
    post_mod._now_et = lambda: now_holder["utc"].astimezone(ET)
    sch_mod.notify_ready = lambda: False
    sch_mod.notify_watchdog = lambda: False
    sch_mod._watchdog_ping_sleep = lambda s, **k: now_holder.update(utc=now_holder["utc"] + timedelta(seconds=max(0, s)))
    sch_mod._poll_interval_sleep = lambda s: now_holder.update(utc=now_holder["utc"] + timedelta(seconds=max(0, s)))
    sch_mod._idle_until_next_wakeup = lambda t, heartbeat_path=None: None
    @contextlib.contextmanager
    def noop_wd(*a, **k):
        yield
    sch_mod.heartbeat_watchdog = noop_wd

    cdir = root / f"run_{d_str}"
    pdir = cdir / "data" / "picks"
    acc = pdir / "account_state"
    acc.mkdir(parents=True, exist_ok=True)
    (cdir / "data" / "health_state").mkdir(parents=True, exist_ok=True)
    (acc / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 12, "best_streak": 18,
        "source_date": (d - timedelta(days=1)).isoformat(),
        "recorded_at": now_holder["utc"].isoformat().replace("+00:00", "Z"),
        "source": "mlb_bts_profile",
    }))
    (acc / "saver_state.json").write_text(json.dumps({
        "state": "active", "season": 2026, "updated_at": now_holder["utc"].isoformat(), "source": "probe"
    }))
    picks_mod.save_streak(12, pdir, saver_available=True)

    dms = []
    def send_dm(rec, txt, **kw):
        dms.append(txt)
        return f"id-{len(dms)}"
    for m in (dm_mod, sch_mod, orch_mod, halert_mod, cli_mod):
        if hasattr(m, "send_dm"):
            m.send_dm = send_dm

    t1 = real_datetime(d.year, d.month, d.day, 19, 10, tzinfo=ET).astimezone(UTC)
    t2 = real_datetime(d.year, d.month, d.day, 20, 10, tzinfo=ET).astimezone(UTC)
    raw_sched = {"dates": [{"date": d_str, "games": [
        {"gamePk": 700001, "gameType": "R", "gameDate": t1.strftime("%Y-%m-%dT%H:%M:%SZ"), "officialDate": d_str,
         "doubleHeader": "N", "gameNumber": 1,
         "status": {"detailedState": "Scheduled", "abstractGameState": "Preview", "abstractGameCode": "P", "statusCode": "S"},
         "teams": {"away": {"team": {"id": 147, "name": "NYY", "abbreviation": "NYY"}},
                   "home": {"team": {"id": 111, "name": "BOS", "abbreviation": "BOS"}}}},
        {"gamePk": 700002, "gameType": "R", "gameDate": t2.strftime("%Y-%m-%dT%H:%M:%SZ"), "officialDate": d_str,
         "doubleHeader": "N", "gameNumber": 1,
         "status": {"detailedState": "Scheduled", "abstractGameState": "Preview", "abstractGameCode": "P", "statusCode": "S"},
         "teams": {"away": {"team": {"id": 119, "name": "LAD", "abbreviation": "LAD"}},
                   "home": {"team": {"id": 137, "name": "SF", "abbreviation": "SF"}}}},
    ]}]}
    sch_mod.count_new_confirmations = lambda pks, cs: sum(1 for pk in pks for s in ("away", "home") if (pk, s) not in cs and not cs.add((pk, s)))
    preds = pd.DataFrame([
        {"date": d_str, "batter_id": 100001, "batter_name": "Juan Soto", "team": "NYY", "opp_team": "BOS",
         "pitcher_id": 300001, "pitcher_name": "Garrett Crochet", "pitcher_team": "BOS", "p_hit": 0.21, "p_game_hit": 0.82,
         "est_pa": 4.5, "lineup": 2, "flags": "", "lineup_confirmed": True, "game_pk": 700001, "game_time": t1.strftime("%Y-%m-%dT%H:%M:%SZ")},
        {"date": d_str, "batter_id": 100002, "batter_name": "Shohei Ohtani", "team": "LAD", "opp_team": "SF",
         "pitcher_id": 300002, "pitcher_name": "Logan Webb", "pitcher_team": "SF", "p_hit": 0.205, "p_game_hit": 0.72,
         "est_pa": 4.5, "lineup": 1, "flags": "", "lineup_confirmed": True, "game_pk": 700002, "game_time": t2.strftime("%Y-%m-%dT%H:%M:%SZ")},
    ])
    orch_mod.run_cascade = lambda tiers, ds: (preds, "local")

    walk_only = (d_str == "2026-07-16")
    feed_700001 = {
        "gameData": {"datetime": {"officialDate": d_str}, "game": {"pk": 700001, "season": "2026", "type": "R"},
                     "status": {"abstractGameCode": "F", "detailedState": "Final"},
                     "teams": {"away": {"id": 147, "abbreviation": "NYY"}, "home": {"id": 111, "abbreviation": "BOS"}}},
        "liveData": {
            "boxscore": {"teams": {"away": {"batters": [100001], "players": {"ID100001": {
                "person": {"id": 100001, "fullName": "Juan Soto"}, "battingOrder": "200",
                "stats": {"batting": {"atBats": 0 if walk_only else 4, "hits": 0 if walk_only else 1,
                                      "baseOnBalls": 4 if walk_only else 0, "sacFlies": 0, "plateAppearances": 4}}
            }}}, "home": {"batters": [], "players": {}}}},
            "plays": {"allPlays": [{"result": {"eventType": "walk" if walk_only else "single"},
                                    "about": {"halfInning": "top", "inning": 1, "isComplete": True},
                                    "matchup": {"batter": {"id": 100001, "fullName": "Juan Soto"}, "pitcher": {"id": 300001}}}]}
        }
    }
    class R:
        def __init__(self, p): self._b = json.dumps(p).encode()
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *e): return False
    def _base_urlopen(req, *a, **k):
        u = str(getattr(req, "full_url", req))
        if "/game/" in u:
            return R(feed_700001)
        if f"date={d_str}" in u:
            return R(raw_sched)
        return R({"dates": []})
    picks_mod.retry_urlopen = _base_urlopen
    sch_mod.retry_urlopen = _base_urlopen

    cfg = {
        "orchestrator": {"picks_dir": str(pdir), "models_dir": str(Path.cwd() / "data" / "models"),
                         "heartbeat_path": str(cdir / "data" / ".heartbeat")},
        "bluesky": {"dm_recipient": "stonehengee.bsky.social"},
        "scheduler": {"pick_delivery": "dm", "entry_intent": "enter", "shadow_model": False,
                      "live_forward_capture_on_lock": False},
        "health_checks": {"enabled": False},
        "tiers": [{"name": "local", "type": "local"}],
    }
    sch_mod.run_day(d_str, cfg)
    pick_path = pdir / f"{d_str}.json"
    pick_doc = json.loads(pick_path.read_text()) if pick_path.exists() else None
    dec_doc = json.loads((pdir / d_str / "decision.json").read_text())
    streak_doc = json.loads((pdir / "streak.json").read_text())
    out["matched_2026"][d_str] = {
        "days_remaining": strat_mod._days_remaining(d_str),
        "action": dec_doc["action"],
        "objective": dd_mod.decision_objective(dec_doc),
        "pick_keys": sorted(pick_doc.keys()) if pick_doc else [],
        "dec_keys": sorted(dec_doc.keys()),
        "streak_keys": sorted(streak_doc.keys()),
        "graded_result": pick_doc.get("result") if pick_doc else None,
        "dms": dms,
    }

pdir57 = root / "s57"
pdir57.mkdir(parents=True, exist_ok=True)
preds57 = pd.DataFrame([
    {"date": "2026-07-16", "batter_id": 100001, "batter_name": "Juan Soto", "team": "NYY", "opp_team": "BOS",
     "pitcher_id": 300001, "pitcher_name": "P1", "pitcher_team": "BOS", "p_hit": 0.21, "p_game_hit": 0.85,
     "est_pa": 4.5, "lineup": 2, "flags": "", "lineup_confirmed": True, "game_pk": 700001, "game_time": "2026-07-16T23:10:00Z"},
])
out["basesha_s57_74d"] = strat_mod.select_pick(preds57, "2026-07-16", pdir57, streak=57, saver_available=False, game_statuses_detailed=gstat).action
out["basesha_s58_74d"] = strat_mod.select_pick(preds57, "2026-07-16", pdir57, streak=58, saver_available=False, game_statuses_detailed=gstat).action
out["has_day_outcome_cmd"] = ("day-outcome" in cli_mod.cli.commands)

print(json.dumps(out))
"""
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(basesha_root / "src")
    proc = subprocess.run(
        [sys.executable, str(driver), str(work_dir)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"BASESHA comparison failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_basesha_vs_head_differences_all_documented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Compare BASESHA (619abda7) vs HEAD on 2027 dates and matched-day-count 2026 dates,
    verifying every sent or written difference matches CLAUDE.md."""
    import bts.picks as picks_mod
    import bts.scheduler as sch_mod
    import bts.simulate.mdp as mdp_mod
    import bts.strategy as strat_mod

    basesha_root = _ensure_basesha_tree(tmp_path)
    base_out = _run_basesha_comparison_subprocess(basesha_root, tmp_path / "basesha_runs")

    for d_str in ("2027-07-15", "2027-09-20", "2027-09-26"):
        row = base_out["dates_2027"][d_str]
        assert row["days_remaining"] == 0
        assert row["streak12_action"] == "skip" and row["streak57_action"] == "skip"

    # Run HEAD on matched 2026-07-16, 2026-09-21, 2026-09-27 and compare against BASESHA!
    matched_diffs = {}
    for d_str in ("2026-07-16", "2026-09-21", "2026-09-27"):
        head_dir = tmp_path / f"head_{d_str}"
        d = real_date.fromisoformat(d_str)
        start_et = real_datetime(d.year, d.month, d.day, 10, 0, tzinfo=ET)
        clock = ClockController(start_et)
        with monkeypatch.context() as mp, pinned_environment(mp, clock):
            cfg, _ = make_live_config(head_dir)
            cfg["scheduler"]["health_checks_enabled"] = False
            picks_dir = Path(cfg["orchestrator"]["picks_dir"])
            write_covered_auto_state(picks_dir, start_et, streak=12, best_streak=18, saver_state="active")
            picks_mod.save_streak(12, picks_dir, saver_available=True, season=2026)
            transports = StubTransports()
            games = make_two_game_schedule(d_str)
            transports.schedules_by_date[d_str] = games
            walk_only = (d_str == "2026-07-16")
            transports.feeds_by_pk[700001] = make_final_feed(
                700001, d_str, {100001: ("Juan Soto", "walk_void" if walk_only else "hit")}
            )
            transports.feeds_by_pk[700002] = make_final_feed(700002, d_str, {100002: ("Shohei Ohtani", "hit")})
            transports.install(mp, lambda ds: make_predictions_df(ds, games, p1=0.82, p2=0.72))
            sch_mod.run_day(d_str, cfg)

            head_pick_path = picks_dir / f"{d_str}.json"
            head_pick = json.loads(head_pick_path.read_text()) if head_pick_path.exists() else None
            head_dec = json.loads((picks_dir / d_str / "decision.json").read_text())
            head_streak = json.loads((picks_dir / "streak.json").read_text())
            head_dms = [txt for _, txt, _ in transports.dms]

        base_row = base_out["matched_2026"][d_str]
        assert base_row["action"] == head_dec["action"]
        assert base_row["objective"] == head_dec["objective"]
        matched_diffs[d_str] = {
            "action": head_dec["action"],
            "objective": head_dec["objective"],
            "dm_equal": (base_row["dms"] == head_dms),
            "base_dms": base_row["dms"],
            "head_dms": head_dms,
        }
        if d_str == "2026-09-27":
            # Skip day (streak=12, d=1): Ea intentionally changed skip DM wording from "skipping today" to "No pick at this check."
            assert len(base_row["dms"]) == 1 and len(head_dms) == 1
            assert "No pick at this check." in head_dms[0] and "Streak used at this check: 12." in head_dms[0]
        if d_str == "2026-07-16":
            added_pick_keys = sorted(set(head_pick.keys()) - set(base_row["pick_keys"]))
            added_dec_keys = sorted(set(head_dec.keys()) - set(base_row["dec_keys"]))
            added_streak_keys = sorted(set(head_streak.keys()) - set(base_row["streak_keys"]))
            assert added_pick_keys == ["automatic_reading", "chosen_on"]
            assert added_dec_keys == ["policy_guard"]
            assert added_streak_keys == ["applied_dates", "season"]
            assert base_row["graded_result"] == "miss" and head_pick["result"] == "void"
            assert base_row["dms"] == head_dms, f"BASE DMS={base_row['dms']!r}\nHEAD DMS={head_dms!r}"

    # Also exercise Change Set G (monotone skip guard) on a real guarded cell in the shipped table!
    policy_table, bounds, season_len = mdp_mod.load_policy(REPO_ROOT / "data" / "models" / "mdp_policy.npz")
    guarded_example = None
    for s in range(0, 57):
        for d_rem in (74, 28, 7, 1):
            for sav in (0, 1):
                for b_idx in range(1, len(bounds) + 1):
                    mid_p = (bounds[b_idx - 1] + (bounds[b_idx] if b_idx < len(bounds) else 0.95)) / 2.0
                    raw_a = mdp_mod.lookup_action(policy_table, list(bounds), s, d_rem, bool(sav), mid_p, int(season_len))
                    g_a, g_meta = mdp_mod.lookup_action_guarded(
                        policy_table, list(bounds), s, d_rem, bool(sav), mid_p, int(season_len)
                    )
                    if raw_a == "skip" and g_a != "skip":
                        guarded_example = (s, d_rem, bool(sav), mid_p, g_a, g_meta)
                        break
                if guarded_example:
                    break
            if guarded_example:
                break
        if guarded_example:
            break
    assert guarded_example is not None, "Expected at least one guarded cell in shipped mdp_policy.npz"
    gs, gd, gsav, gp, gact, gmeta = guarded_example
    g_date = (real_date(2027, 9, 27) - timedelta(days=gd)).isoformat()
    g_games = make_two_game_schedule(g_date)
    g_preds = make_predictions_df(g_date, g_games, p1=gp, p2=gp - 0.01)
    g_sel = strat_mod.select_pick(
        g_preds, g_date, tmp_path / "guard_test", streak=gs, saver_available=gsav,
        game_statuses_detailed={
            700001: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
            700002: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        },
    )
    assert g_sel.decision.guard is not None and g_sel.decision.guard["rule"] == "monotone_skip_v1"

    return {
        "matched_2026": matched_diffs,
        "walk_only_basesha_vs_head": ("miss", "void"),
        "basesha_2027_all_skip": True,
        "basesha_s57_s58_skip": (base_out["basesha_s57_74d"], base_out["basesha_s58_74d"]),
        "guarded_cell_verified": {"streak": gs, "days": gd, "saver": gsav, "action": gact, "guard": gmeta},
    }


def main() -> int:
    import tempfile

    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))

    log_path = REPO_ROOT / ".carbon-review" / "whole-range" / "probe_mid_late_and_streaks.log"
    results = {}
    with tempfile.TemporaryDirectory(prefix="wr2_probe1_") as tmp:
        tmp_path = Path(tmp)
        mp = pytest.MonkeyPatch()
        try:
            results["days"] = test_mid_late_finale_automatic_days(tmp_path / "t1", mp)
            results["streaks"] = test_streaks_55_to_400_and_all_readers(tmp_path / "t2", mp)
            results["basesha_vs_head"] = test_basesha_vs_head_differences_all_documented(tmp_path / "t3", mp)
        finally:
            mp.undo()
    assert "bts.model.predict" not in sys.modules, "bts.model.predict must not be imported"
    log_path.write_text(json.dumps(results, indent=2) + "\n")
    print("PROBE 1 PASS:", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
