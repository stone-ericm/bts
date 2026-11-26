"""Slice 2 Probe 2: Pre-season deploy behavior on END-OF-2026 snapshot state
(old private/research config, half-edited configs, systemd restart + canary behavior)
and automatic rollback to BASESHA (619abda7) on a day HEAD already wrote files.

Runnable standalone (`.venv/bin/python .carbon-review/whole-range/probe_deploy_and_rollback.py`)
and adoptable as a pytest regression module (no module-level `bts.model.predict` import,
no process-wide state left mutated, no network).
"""
from __future__ import annotations

import contextlib
import datetime as dt_stdlib
import json
import os
import socket
import subprocess
import sys
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
    def __init__(self, initial_et: real_datetime):
        self.now_utc: real_datetime = initial_et.astimezone(UTC)

    @property
    def now_et(self) -> real_datetime:
        return self.now_utc.astimezone(ET)

    def set_et(self, dt_et: real_datetime) -> None:
        self.now_utc = dt_et.astimezone(UTC)

    def advance(self, seconds: float) -> None:
        self.now_utc = self.now_utc + timedelta(seconds=max(0.0, float(seconds)))


@contextlib.contextmanager
def pinned_env(monkeypatch: pytest.MonkeyPatch, clock: ClockController, events: dict):
    real_sock = socket.socket

    class _BlockedSocket(real_sock):
        def __init__(self, *a, **kw):
            raise AssertionError("network socket blocked")

    monkeypatch.setattr(socket, "socket", _BlockedSocket)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network blocked")))

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
    import bts.contest_state as cs_mod
    import bts.daily_decision as dd_mod
    import bts.day_outcome as do_mod
    import bts.dm as dm_mod
    import bts.heartbeat as hb_mod
    import bts.orchestrator as orch_mod
    import bts.picks as picks_mod
    import bts.posting as post_mod
    import bts.saver_state as ss_mod
    import bts.scheduler as sch_mod
    import bts.season_contract as sc_mod
    import bts.strategy as strat_mod
    import bts.util as util_mod
    import bts.health.alert as halert_mod
    import bts.health.contest_state as hcs_mod
    import bts.health.runner as hrunner_mod

    for mod in (cli_mod, cs_mod, dd_mod, do_mod, hb_mod, picks_mod, post_mod, ss_mod, sch_mod, sc_mod, strat_mod, halert_mod, hcs_mod, hrunner_mod):
        if hasattr(mod, "datetime"):
            monkeypatch.setattr(mod, "datetime", PinnedDatetime)
    for mod in (cs_mod, do_mod, hcs_mod, hrunner_mod):
        if hasattr(mod, "date"):
            monkeypatch.setattr(mod, "date", PinnedDate)

    monkeypatch.setattr(sch_mod, "_now_et", lambda: clock.now_et)
    monkeypatch.setattr(post_mod, "_now_et", lambda: clock.now_et)
    monkeypatch.setattr(cli_mod, "_now_et", lambda: clock.now_et)

    def _ready():
        events["ready_notifications"] += 1
        return True

    monkeypatch.setattr(sch_mod, "notify_ready", _ready)
    monkeypatch.setattr(sch_mod, "notify_watchdog", lambda: True)
    monkeypatch.setattr(sch_mod, "_watchdog_ping_sleep", lambda s, **kw: clock.advance(s))
    monkeypatch.setattr(sch_mod, "_poll_interval_sleep", lambda s: clock.advance(s))

    def _fast_idle(target_et, heartbeat_path=None):
        events["idled_to"].append(str(target_et))
        if isinstance(target_et, str):
            target_et = real_datetime.fromisoformat(target_et)
        delta = (target_et - clock.now_et).total_seconds()
        if delta > 0:
            clock.advance(delta)

    monkeypatch.setattr(sch_mod, "_idle_until_next_wakeup", _fast_idle)

    @contextlib.contextmanager
    def _noop_wd(*a, **kw):
        yield

    monkeypatch.setattr(sch_mod, "heartbeat_watchdog", _noop_wd)

    def _send_dm(recipient: str, text: str, **kw) -> str:
        events["dms"].append((recipient, text))
        return f"dm-{len(events['dms'])}"

    def _post(text: str, **kw) -> str:
        events["posts"].append(text)
        return f"post-{len(events['posts'])}"

    for mod in (dm_mod, sch_mod, orch_mod, halert_mod, cli_mod, do_mod):
        if hasattr(mod, "send_dm"):
            monkeypatch.setattr(mod, "send_dm", _send_dm)
    for mod in (post_mod, sch_mod, orch_mod, cli_mod):
        if hasattr(mod, "post_to_bluesky"):
            monkeypatch.setattr(mod, "post_to_bluesky", _post)

    class _FakeResp:
        def __init__(self, obj):
            self._b = json.dumps(obj).encode("utf-8")

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *e):
            return False

    def _fake_urlopen(req, *a, **kw):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "/game/" in url and "/feed/live" in url:
            pk = int(url.split("/game/")[1].split("/")[0])
            return _FakeResp(events["feeds"][pk])
        if "api/v1/schedule" in url:
            ds = url.split("date=")[1].split("&")[0] if "date=" in url else ""
            return _FakeResp(events["schedules"].get(ds, {"dates": []}))
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(sch_mod, "retry_urlopen", _fake_urlopen)
    monkeypatch.setattr(picks_mod, "retry_urlopen", _fake_urlopen)
    monkeypatch.setattr(util_mod, "retry_urlopen", _fake_urlopen)
    monkeypatch.setattr(sch_mod, "_delivery_game_statuses", picks_mod.get_game_statuses_detailed)
    monkeypatch.setattr(
        sch_mod,
        "count_new_confirmations",
        lambda pks, cs: sum(1 for pk in pks for s in ("away", "home") if (pk, s) not in cs and not cs.add((pk, s))),
    )
    monkeypatch.setattr(orch_mod, "run_cascade", lambda tiers, ds: (events["pred_fn"](ds), "local"))
    yield


def populate_end_of_2026_snapshot(box_dir: Path) -> Path:
    """Create a realistic END-OF-2026 restored snapshot directory."""
    picks_dir = box_dir / "data" / "picks"
    acc_dir = picks_dir / "account_state"
    health_dir = box_dir / "data" / "health_state"
    models_dir = box_dir / "data" / "models"
    for d in (picks_dir, acc_dir, health_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    for npz in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        dst = models_dir / npz
        if not dst.exists():
            dst.symlink_to(REPO_ROOT / "data" / "models" / npz)
    # Create dummy blend files so unrelated blend_training health check is quiet when testing contest health
    for ds in ("2027-03-24", "2027-03-25", "2027-07-15", "2027-07-16"):
        (models_dir / f"blend_{ds}.pkl").write_bytes(b"stub")

    # 2026 streak.json (no season tag, no applied_dates)
    (picks_dir / "streak.json").write_text(json.dumps({
        "streak": 4,
        "saver_available": False,
        "updated": "2026-09-28T06:00:00+00:00",
    }, indent=2))
    # 2026 auto_v1 contest_streak.json
    (acc_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 4,
        "best_streak": 18,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-14",
        "recorded_at": "2026-09-15T06:10:00Z",
        "user_id": 12345,
        "username": "stonehengee",
        "saver_available": None,
    }, indent=2))
    # 2026 saver_state.json
    (acc_dir / "saver_state.json").write_text(json.dumps({
        "state": "used",
        "season": 2026,
        "updated_at": "2026-07-02T01:10:00Z",
        "source": "cli",
    }, indent=2))
    # 2026 historical pick + scheduler_state + heartbeat + health status
    (picks_dir / "2026-09-14.json").write_text(json.dumps({
        "date": "2026-09-14",
        "pick": {
            "batter_name": "Bobby Witt Jr.", "batter_id": 677951, "team": "KC",
            "lineup_position": 2, "pitcher_name": "P", "pitcher_id": 1,
            "p_game_hit": 0.83, "flags": [], "projected_lineup": False,
            "game_pk": 600001, "game_time": "2026-09-14T23:10:00Z", "pitcher_team": "NYY",
        },
        "double_down": None,
        "runner_up": None,
        "streak_before": 3,
        "result": "hit",
        "run_time": "2026-09-14T20:00:00Z",
    }, indent=2))
    (picks_dir / "scheduler_state.json").write_text(json.dumps({
        "date": "2026-10-20",
        "games": [],
        "runs_planned": [],
        "runs_completed": [],
        "pick_locked": False,
        "locked_at": None,
        "post_uri": None,
        "result_status": None,
        "next_wakeup": "2026-10-21T10:00:00-04:00",
    }, indent=2))
    (box_dir / "data" / ".heartbeat").write_text(json.dumps({
        "pid": 999,
        "state": "sleeping",
        "updated_at": "2026-10-20T14:00:00+00:00",
        "sleeping_until": "2026-10-21T14:00:00+00:00",
    }))
    (health_dir / "health_dm_delivery_status.json").write_text(json.dumps({
        "date": "2026-10-20",
        "status": "sent",
        "sent_sources": [],
    }))
    return picks_dir


def write_toml_config(
    box_dir: Path,
    *,
    delivery: str,
    intent: str,
    required: bool | None = None,
    expected: bool | None = None,
    health_enabled: bool = True,
) -> Path:
    picks_dir = box_dir / "data" / "picks"
    models_dir = box_dir / "data" / "models"
    hb_path = box_dir / "data" / ".heartbeat"
    health_dir = box_dir / "data" / "health_state"
    lines = [
        "[orchestrator]",
        f'picks_dir = "{picks_dir}"',
        f'models_dir = "{models_dir}"',
        f'heartbeat_path = "{hb_path}"',
        "",
        "[bluesky]",
        'dm_recipient = "stonehengee.bsky.social"',
        "",
        "[scheduler]",
        f'pick_delivery = "{delivery}"',
        f'entry_intent = "{intent}"',
        "shadow_model = false",
        "live_forward_capture_on_lock = false",
    ]
    if required is not None:
        lines.append(f"contest_state_required = {'true' if required else 'false'}")
    lines += [
        "",
        "[health_checks]",
        f"enabled = {'true' if health_enabled else 'false'}",
    ]
    if expected is not None:
        lines.append(f"contest_state_expected = {'true' if expected else 'false'}")
    lines += [
        "",
        "[health_checks.thresholds]",
        f'dm_delivery_status = "{health_dir / "health_dm_delivery_status.json"}"',
        f'warn_attention_state = "{health_dir / "warn_attention.json"}"',
        "disk_fill_warn_pct = 99.8",
        "disk_fill_crit_pct = 99.9",
        "",
        "[[tiers]]",
        'name = "local"',
        'type = "local"',
        "",
    ]
    p = box_dir / "orchestrator.toml"
    p.write_text("\n".join(lines))
    return p


def make_single_game_schedule_and_preds(d_str: str):
    d = real_date.fromisoformat(d_str)
    t1 = real_datetime(d.year, d.month, d.day, 19, 10, tzinfo=ET).astimezone(UTC)
    sched = {
        "dates": [{
            "date": d_str,
            "games": [{
                "gamePk": 700001, "gameType": "R", "gameDate": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "officialDate": d_str, "doubleHeader": "N", "gameNumber": 1,
                "status": {"detailedState": "Scheduled", "abstractGameState": "Preview", "abstractGameCode": "P", "statusCode": "S"},
                "teams": {
                    "away": {"team": {"id": 147, "name": "NYY", "abbreviation": "NYY"}},
                    "home": {"team": {"id": 111, "name": "BOS", "abbreviation": "BOS"}},
                },
            }],
        }]
    }
    preds = pd.DataFrame([{
        "date": d_str, "batter_id": 100001, "batter_name": "Juan Soto", "team": "NYY", "opp_team": "BOS",
        "pitcher_id": 300001, "pitcher_name": "Garrett Crochet", "pitcher_team": "BOS",
        "p_hit": 0.21, "p_game_hit": 0.85, "est_pa": 4.5, "lineup": 2, "flags": "",
        "lineup_confirmed": True, "game_pk": 700001, "game_time": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }])
    feed = {
        "gameData": {
            "datetime": {"officialDate": d_str},
            "game": {"pk": 700001, "season": d_str[:4], "type": "R"},
            "status": {"abstractGameCode": "F", "detailedState": "Final"},
            "teams": {"away": {"id": 147, "abbreviation": "NYY"}, "home": {"id": 111, "abbreviation": "BOS"}},
        },
        "liveData": {
            "boxscore": {"teams": {"away": {"batters": [100001], "players": {"ID100001": {
                "person": {"id": 100001, "fullName": "Juan Soto"}, "battingOrder": "200",
                "stats": {"batting": {"atBats": 4, "hits": 1, "baseOnBalls": 0, "sacFlies": 0, "plateAppearances": 4}},
            }}}, "home": {"batters": [], "players": {}}}},
            "plays": {"allPlays": [{"result": {"eventType": "single"}, "about": {"halfInning": "top", "inning": 1, "isComplete": True},
                                    "matchup": {"batter": {"id": 100001, "fullName": "Juan Soto"}, "pitcher": {"id": 300001}}}]},
        },
    }
    return sched, preds, feed


def test_head_first_start_and_half_edited_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Test HEAD `bts schedule` on END-OF-2026 snapshot state under:
    1) Old config (private, research, neither contest_state flag set)
    2) Half-edited A (dm delivery, contest state NOT required) + systemd restart loop + check_heartbeat
    3) Half-edited B (dm delivery, contest state required, NO season_contract.json switch file) + same-day restarts
    """
    import bts.cli as cli_mod

    runner = CliRunner()
    out: dict[str, object] = {}

    # Case 1: Old snapshot config (private, research, no contest_state_required/expected) on a game day (2027-03-24)
    box1 = tmp_path / "box_old_private_research"
    picks1 = populate_end_of_2026_snapshot(box1)
    cfg1 = write_toml_config(box1, delivery="private", intent="research", required=None, expected=None, health_enabled=False)
    sched_json, preds_df, feed_json = make_single_game_schedule_and_preds("2027-03-24")
    clock1 = ClockController(real_datetime(2027, 3, 24, 10, 0, tzinfo=ET))
    ev1 = {"dms": [], "posts": [], "ready_notifications": 0, "idled_to": [],
           "schedules": {"2027-03-24": sched_json, "2027-03-25": sched_json},
           "feeds": {700001: feed_json}, "pred_fn": lambda ds: preds_df}
    with monkeypatch.context() as mp, pinned_env(mp, clock1, ev1):
        res1 = runner.invoke(cli_mod.cli, ["schedule", "--config", str(cfg1), "--date", "2027-03-24"])
        assert res1.exit_code == 0, f"Old config failed: {res1.output}"
        assert ev1["ready_notifications"] >= 1, "Expected systemd READY=1 notification"
        assert ev1["dms"] == [] and ev1["posts"] == []
        # In private research mode without contest_state_required, 2026 contest_streak.json (streak=4) drove selection!
        p24 = json.loads((picks1 / "2027-03-24.json").read_text())
        d24 = json.loads((picks1 / "2027-03-24" / "decision.json").read_text())
        s24 = json.loads((picks1 / "streak.json").read_text())
        assert d24["streak"] == 4, f"Expected 2026 contest_streak=4 in private mode, got {d24['streak']}"
        assert d24["delivery_status"] == "private_locked"
        # CS2: grading stamped streak.json for 2027 starting from 0 -> 1!
        assert s24["season"] == 2027 and s24["streak"] == 1
    out["old_private_research"] = {
        "exit_code": res1.exit_code,
        "ready_sent": ev1["ready_notifications"] >= 1,
        "dms_sent": len(ev1["dms"]),
        "used_2026_contest_streak": d24["streak"],
        "graded_model_streak_2027": s24["streak"],
    }

    # Case 2: Half-edited A (pick_delivery = "dm", entry_intent = "enter" or "research", contest_state_required unset/false)
    box2 = tmp_path / "box_half_dm_unrequired"
    populate_end_of_2026_snapshot(box2)
    hb2 = box2 / "data" / ".heartbeat"
    hb_before = hb2.read_text()
    for sub_intent in ("enter", "research"):
        cfg2 = write_toml_config(box2, delivery="dm", intent=sub_intent, required=None, expected=None)
        clock2 = ClockController(real_datetime(2027, 3, 24, 10, 0, tzinfo=ET))
        ev2 = {"dms": [], "posts": [], "ready_notifications": 0, "idled_to": [],
               "schedules": {"2027-03-24": sched_json}, "feeds": {}, "pred_fn": lambda ds: preds_df}
        with monkeypatch.context() as mp, pinned_env(mp, clock2, ev2):
            # Simulate 5 consecutive systemd Restart=always cycles (every 30s)
            for _ in range(5):
                r = runner.invoke(cli_mod.cli, ["schedule", "--config", str(cfg2)])
                assert r.exit_code == 2
                assert "bts schedule: refusing to start:" in r.output
                clock2.advance(30)
            assert ev2["dms"] == [], "Refusal must send ZERO DMs across restarts!"
            assert ev2["ready_notifications"] == 0, "Refusal must exit before READY=1 (so deploy canary fails!)"
            assert hb2.read_text() == hb_before, "Refusal must exit before writing heartbeat!"
            # Verify entry-intent CLI also exits 1
            r_ei = runner.invoke(cli_mod.cli, ["entry-intent", "--config", str(cfg2)])
            assert r_ei.exit_code == 1
    with monkeypatch.context() as mp, pinned_env(mp, ClockController(real_datetime(2027, 3, 24, 10, 0, tzinfo=ET)), ev2):
        r_dry = runner.invoke(cli_mod.cli, ["schedule", "--config", str(cfg2), "--date", "2027-03-24", "--dry-run"])
        assert r_dry.exit_code == 0 and "WARNING (a real run would refuse to start)" in r_dry.output
    out["half_edited_dm_not_required"] = {
        "exit_code": 2,
        "dms_across_5_systemd_restarts": 0,
        "ready_sent": False,
        "heartbeat_untouched": True,
        "deploy_canary_would_fail_and_rollback": True,
    }

    # Case 3: Half-edited B (pick_delivery = "dm", entry_intent = "enter", contest_state_required = true,
    #         contest_state_expected = true, NO season_contract.json switch file!)
    box3 = tmp_path / "box_half_required_no_switch"
    picks3 = populate_end_of_2026_snapshot(box3)
    cfg3 = write_toml_config(box3, delivery="dm", intent="enter", required=True, expected=True, health_enabled=True)

    # 3a) Pre-season OFF-DAY (2027-03-03): starts, sends READY=1 (deploy canary PASSES!), 0 DMs, idles!
    clock3_off = ClockController(real_datetime(2027, 3, 3, 10, 0, tzinfo=ET))
    ev3_off = {"dms": [], "posts": [], "ready_notifications": 0, "idled_to": [],
               "schedules": {}, "feeds": {}, "pred_fn": lambda ds: pd.DataFrame()}
    with monkeypatch.context() as mp, pinned_env(mp, clock3_off, ev3_off):
        res3_off = runner.invoke(cli_mod.cli, ["schedule", "--config", str(cfg3), "--date", "2027-03-03"])
        assert res3_off.exit_code == 0
        assert ev3_off["ready_notifications"] >= 1
        assert ev3_off["dms"] == []

    # 3b) GAME DAY (2027-03-24): starts, sends READY=1, sends 1 morning CRITICAL DM + 1 check CRITICAL DM +
    #     1 EOD missed_pick/post_failure/saver_state DMs, does NOT crash-loop, and a same-day restart deduplicates!
    clock3_game = ClockController(real_datetime(2027, 3, 24, 10, 0, tzinfo=ET))
    ev3_game = {"dms": [], "posts": [], "ready_notifications": 0, "idled_to": [],
                "schedules": {"2027-03-24": sched_json, "2027-03-25": sched_json},
                "feeds": {700001: feed_json}, "pred_fn": lambda ds: preds_df}
    with monkeypatch.context() as mp, pinned_env(mp, clock3_game, ev3_game):
        res3_g1 = runner.invoke(cli_mod.cli, ["schedule", "--config", str(cfg3), "--date", "2027-03-24"])
        assert res3_g1.exit_code == 0
        assert not (picks3 / "2027-03-24.json").exists(), "Must not pick or deliver without switch!"
        dms_run1 = [txt for _, txt in ev3_game["dms"]]
        assert any("Morning check for 2027-03-24: no usable contest state" in m for m in dms_run1)
        assert any("Live decision refused: season 2027 switch is absent." in m for m in dms_run1)

        # Simulate a same-day systemd restart at 14:00 ET on 2027-03-24 -> verify contest_state DMs do NOT repeat!
        clock3_game.set_et(real_datetime(2027, 3, 24, 14, 0, tzinfo=ET))
        dms_count_before_restart = len(ev3_game["dms"])
        res3_g2 = runner.invoke(cli_mod.cli, ["schedule", "--config", str(cfg3), "--date", "2027-03-24"])
        assert res3_g2.exit_code == 0
        dms_after_restart = [txt for _, txt in ev3_game["dms"][dms_count_before_restart:]]
        assert not any("Morning check for 2027-03-24" in m for m in dms_after_restart), "Morning DM repeated on restart!"
        assert not any("contest_state" in m and "CRITICAL" in m for m in dms_after_restart), "contest_state CRITICAL repeated!"

    out["half_edited_required_no_switch"] = {
        "off_day_exit": res3_off.exit_code,
        "off_day_dms": len(ev3_off["dms"]),
        "deploy_canary_passes": ev3_off["ready_notifications"] >= 1,
        "game_day_exit": res3_g1.exit_code,
        "game_day_dms_first_run": len(dms_run1),
        "same_day_restart_extra_dms": len(dms_after_restart),
    }
    return out


def test_rollback_to_basesha_after_head_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Create every file HEAD can write on an active 2027 day (and matched 2026 day to isolate schema vs date effects),
    then run BASESHA (619abda7) readers, scheduler, check-results, reconcile, check-pick-entered, fetch-contest-streak,
    and health.runner against those exact files."""
    basesha_root = _ensure_basesha_tree(tmp_path)
    work = tmp_path / "rollback_work"
    work.mkdir(parents=True, exist_ok=True)

    driver = work / "_rollback_basesha_probe.py"
    driver.write_text(
        """
import contextlib, datetime as dt_stdlib, json, socket, sys
from types import SimpleNamespace
from datetime import date as real_date, datetime as real_datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from click.testing import CliRunner

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

_real_sock = socket.socket
class _BlockedSock(_real_sock):
    def __init__(self, *a, **kw):
        raise AssertionError("network blocked")
socket.socket = _BlockedSock
socket.create_connection = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network blocked"))

import bts.cli as cli_mod
import bts.contest_fetch as cf_mod
import bts.contest_state as cs_mod
import bts.daily_decision as dd_mod
import bts.dm as dm_mod
import bts.leaderboard.auth as auth_mod
import bts.orchestrator as orch_mod
import bts.picks as picks_mod
import bts.posting as post_mod
import bts.scheduler as sch_mod
import bts.strategy as strat_mod
import bts.health.alert as halert_mod
import bts.health.contest_state as hcs_mod
import bts.health.pick_entry as hpe_mod
import bts.health.runner as hrunner_mod

root = Path(sys.argv[1])
models_dir = Path.cwd() / "data" / "models"
report = {}

# Helper to pin clock in BASESHA
now_holder = {"utc": real_datetime(2027, 7, 15, 17, 0, tzinfo=ET).astimezone(UTC)}
class PinnedDT(real_datetime):
    @classmethod
    def now(cls, tz=None):
        return now_holder["utc"].astimezone(tz) if tz else now_holder["utc"].astimezone(ET).replace(tzinfo=None)
class PinnedD(real_date):
    @classmethod
    def today(cls):
        return now_holder["utc"].astimezone(ET).date()
dt_stdlib.datetime = PinnedDT
dt_stdlib.date = PinnedD
for m in (cli_mod, cs_mod, dd_mod, picks_mod, post_mod, sch_mod, strat_mod, halert_mod, hcs_mod, hpe_mod, hrunner_mod):
    if hasattr(m, "datetime"): m.datetime = PinnedDT
    if hasattr(m, "date"): m.date = PinnedD
sch_mod._now_et = lambda: now_holder["utc"].astimezone(ET)
post_mod._now_et = lambda: now_holder["utc"].astimezone(ET)
sch_mod.notify_ready = lambda: False
sch_mod.notify_watchdog = lambda: False
sch_mod._watchdog_ping_sleep = lambda s, **k: now_holder.update(utc=now_holder["utc"] + timedelta(seconds=max(0, s)))
sch_mod._poll_interval_sleep = lambda s: now_holder.update(utc=now_holder["utc"] + timedelta(seconds=max(0, s)))
sch_mod._idle_until_next_wakeup = lambda t, heartbeat_path=None: None
@contextlib.contextmanager
def noop_wd(*a, **k): yield
sch_mod.heartbeat_watchdog = noop_wd

dms = []
def send_dm(rec, txt, **kw):
    dms.append(txt)
    return f"id-{len(dms)}"
for m in (dm_mod, sch_mod, orch_mod, halert_mod, cli_mod):
    if hasattr(m, "send_dm"): m.send_dm = send_dm

# --- TEST A: HEAD pick file (<date>.json) with chosen_on, automatic_reading, guard, and longest_streak ---
pdir_a = root / "case_a" / "data" / "picks"
pdir_a.mkdir(parents=True, exist_ok=True)
head_pick_doc = {
    "date": "2027-07-15",
    "pick": {
        "batter_name": "Juan Soto", "batter_id": 100001, "team": "NYY", "lineup_position": 2,
        "pitcher_name": "P1", "pitcher_id": 300001, "p_game_hit": 0.84, "flags": [],
        "projected_lineup": False, "game_pk": 700001, "game_time": "2027-07-15T23:10:00Z", "pitcher_team": "BOS"
    },
    "double_down": None,
    "runner_up": None,
    "result": None,
    "run_time": "2027-07-15T20:10:00Z",
    "notification_sent": True,
    "notification_channel": "dm",
    "notification_id": "msg-head-1",
    "delivered_at": "2027-07-15T20:10:05Z",
    "delivery_attempted": True,
    "policy_decision": {
        "action": "single", "source": "mdp", "objective": "longest_streak",
        "best_streak": 58, "effective_best": 58, "best_status": "trusted",
        "tail_policy_sha256": "dc5d0c99", "degraded_reason": None,
        "guard": {"rule": "monotone_skip_v1", "bin": 2, "from_bin": 1, "table_action": "skip"}
    },
    "chosen_on": {
        "schema": "bts_pick_chosen_on_v1", "required": True, "kind": "automatic",
        "streak": 58, "saver_available": False, "dated": None
    },
    "automatic_reading": {
        "schema": "bts_pick_automatic_reading_v1", "recorded_at": "2027-07-15T17:30:00Z",
        "age_hours": 2.67, "stale": False, "reason": None
    }
}
(pdir_a / "2027-07-15.json").write_text(json.dumps(head_pick_doc, indent=2))
loaded_daily = picks_mod.load_pick("2027-07-15", pdir_a)
assert loaded_daily is not None and loaded_daily.pick.batter_id == 100001
# Re-save via BASESHA save_pick (as check-results / run_result_polling does):
loaded_daily.result = "hit"
picks_mod.save_pick(loaded_daily, pdir_a)
resaved_doc = json.loads((pdir_a / "2027-07-15.json").read_text())
# Also test HEAD shadow pick file (<date>.shadow.json) on BASESHA load_shadow_pick:
shadow_doc = dict(head_pick_doc, shadow_model_version="v2")
(pdir_a / "2027-07-15.shadow.json").write_text(json.dumps(shadow_doc, indent=2))
loaded_shadow = picks_mod.load_shadow_pick("2027-07-15", pdir_a)
assert loaded_shadow is not None and loaded_shadow.pick.batter_id == 100001
report["pick_file"] = {
    "basesha_loads_head_pick": True,
    "basesha_loads_head_shadow_pick": True,
    "resave_drops_keys": sorted(set(head_pick_doc.keys()) - set(resaved_doc.keys())),
}

# --- TEST B: HEAD decision.json (already written with longest_streak & policy_guard) vs unwritten mid-commit H pick ---
(pdir_a / "2027-07-15").mkdir(parents=True, exist_ok=True)
head_dec_doc = {
    "schema_version": "bts_daily_decision_v3",
    "date": "2027-07-15",
    "action": "single",
    "source": "mdp",
    "scoreable": True,
    "delivery_status": "delivered",
    "finalized_at": "2027-07-15T20:10:06Z",
    "primary": {"batter_id": 100001, "batter_name": "Juan Soto", "team": "NYY", "game_pk": 700001, "p_game_hit": 0.84},
    "double_down": None,
    "streak": 58, "saver_available": False, "allow_double": True,
    "state_source": "contest", "state_status": "fresh", "contest_source_date": "2027-07-14",
    "objective": "longest_streak", "best_streak": 58, "effective_best": 58, "best_status": "trusted",
    "tail_policy_sha256": "dc5d0c99", "degraded_reason": None,
    "policy_guard": None,
}
(pdir_a / "2027-07-15" / "decision.json").write_text(json.dumps(head_dec_doc, indent=2))
parsed_dec = dd_mod.load_decision("2027-07-15", pdir_a)
# Now test BASESHA trying to WRITE a decision for an outstanding H pick (objective='longest_streak'):
pdir_h_unwritten = root / "case_h_unwritten" / "data" / "picks"
pdir_h_unwritten.mkdir(parents=True, exist_ok=True)
write_h_ret = dd_mod.write_decision(
    "2027-07-15", pdir_h_unwritten, action="single", source="mdp",
    primary={"batter_id": 100001, "batter_name": "Juan Soto", "team": "NYY", "game_pk": 700001, "p_game_hit": 0.84},
    scoreable=True, delivery_status="locked_unconfirmed", objective="longest_streak"
)
report["decision_file"] = {
    "existing_longest_streak_loads": (parsed_dec is not None),
    "existing_is_scoreable_commit": dd_mod.is_scoreable_commit("2027-07-15", pdir_a, loaded_daily),
    "existing_decision_objective_normalized_to": dd_mod.decision_objective(parsed_dec),
    "writing_outstanding_h_pick_returns": write_h_ret,
    "writing_outstanding_h_pick_leaves_file_absent": not (pdir_h_unwritten / "2027-07-15" / "decision.json").exists(),
}

# --- TEST C: HEAD streak.json (season=2027, applied_dates) when BASESHA updates streak ---
(pdir_a / "streak.json").write_text(json.dumps({
    "streak": 58, "saver_available": False, "updated": "2027-07-14T23:00:00+00:00",
    "applied_dates": ["2027-07-14"], "season": 2027
}))
picks_mod.update_streak([True], pdir_a)
streak_after_basesha = json.loads((pdir_a / "streak.json").read_text())
report["streak_file"] = {
    "basesha_reads_head_streak_and_increments_to": streak_after_basesha["streak"],
    "basesha_save_drops_keys": sorted({"season", "applied_dates"} - set(streak_after_basesha.keys())),
}

# --- TEST D: HEAD season_contract.json + auto_v2 contest_streak.json + v3 contest_streak.manual.json ---
# D1: auto_v2 contest_streak.json (at streak=12/18 AND at streak=58/60) + revoked season_contract.json!
pdir_d1 = root / "case_d1" / "data" / "picks"
acc_d1 = pdir_d1 / "account_state"
acc_d1.mkdir(parents=True, exist_ok=True)
(acc_d1 / "season_contract.json").write_text(json.dumps({
    "schema_version": "bts_season_contract_v1", "season": 2027, "status": "revoked",
    "activation_id": "a" * 32, "switched_at": None, "revoked_at": "2027-07-15T12:00:00Z", "note": "revoked"
}))
(acc_d1 / "contest_streak.json").write_text(json.dumps({
    "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "a" * 32,
    "active_streak": 58, "best_streak": 60, "source": "mlb_bts_profile",
    "source_date": "2027-07-14", "recorded_at": "2027-07-15T17:30:00Z", "user_id": 12345, "username": "stonehengee"
}))
(acc_d1 / "saver_state.json").write_text(json.dumps({
    "state": "used", "season": 2027, "updated_at": "2027-07-15T17:30:00Z", "source": "cli"
}))
ds_d1 = cs_mod.load_decision_streak_state(pdir_d1, require_contest_state=True, now=now_holder["utc"])

# D2: v3 dated manual file (streak=5, best=10, saver='not_earned') ALONGSIDE stale 2026 auto_v1 contest_streak.json (streak=4)!
pdir_d2 = root / "case_d2" / "data" / "picks"
acc_d2 = pdir_d2 / "account_state"
acc_d2.mkdir(parents=True, exist_ok=True)
(acc_d2 / "contest_streak.json").write_text(json.dumps({
    "schema_version": "bts_contest_streak_auto_v1",
    "active_streak": 4, "best_streak": 18, "source": "mlb_bts_profile",
    "source_date": "2026-09-14", "recorded_at": "2026-09-15T06:10:00Z", "user_id": 12345, "username": "stonehengee"
}))
(acc_d2 / "contest_streak.manual.json").write_text(json.dumps({
    "schema_version": "bts_contest_streak_manual_v3",
    "effective_date": "2027-07-15", "season": 2027,
    "expires_at": "2027-07-16T04:00:00Z",
    "active_streak": 5, "best_streak": 10, "saver_state": "not_earned",
    "recorded_at": "2027-07-15T13:00:00Z", "source": "manual_cli_dated", "reason": "morning check"
}))
ds_d2_with_auto = cs_mod.load_decision_streak_state(pdir_d2, require_contest_state=True, now=now_holder["utc"])

# D3: v3 dated manual file WITHOUT contest_streak.json alongside it!
(acc_d2 / "contest_streak.json").unlink()
ds_d3_manual_only = cs_mod.load_decision_streak_state(pdir_d2, require_contest_state=True, now=now_holder["utc"])
h_d3_alerts = hcs_mod.check(pdir_d2, expected=True, now=now_holder["utc"])

report["contest_and_manual_files"] = {
    "auto_v2_with_revoked_switch": {
        "basesha_ignores_revoked_switch_and_reads_streak": ds_d1.streak,
        "auto_v2_best_status_degrades_to": ds_d1.best_status,
        "auto_v2_best_streak_returned": ds_d1.best_streak,
    },
    "v3_manual_alongside_2026_auto": {
        "v3_manual_streak_was": 5,
        "basesha_selected_streak": ds_d2_with_auto.streak,
        "basesha_selected_source_date": str(ds_d2_with_auto.contest_source_date),
        "v3_manual_ignored_because_override_expires_at_missing": (ds_d2_with_auto.streak == 4),
    },
    "v3_manual_alone": {
        "basesha_selected_streak": ds_d3_manual_only.streak,
        "status": ds_d3_manual_only.status,
        "allow_double": ds_d3_manual_only.allow_double,
        "best_status": ds_d3_manual_only.best_status,
        "health_alerts": [(a.level, a.source, a.message[:90]) for a in h_d3_alerts],
    },
}

# --- TEST E: HEAD status files + HEAD wrong_game pick_entry_check.json under BASESHA check-pick-entered & health ---
pdir_e = root / "case_e" / "data" / "picks"
hdir_e = root / "case_e" / "data" / "health_state"
pdir_e.mkdir(parents=True, exist_ok=True)
hdir_e.mkdir(parents=True, exist_ok=True)
(hdir_e / "delivery_outcome_unknown").mkdir(parents=True, exist_ok=True)
(hdir_e / "probable_pitcher_cache").mkdir(parents=True, exist_ok=True)
(hdir_e / "pick_entry_check.early").mkdir(parents=True, exist_ok=True)

# Write every HEAD status file
(hdir_e / "health_dm_delivery_status.json").write_text(json.dumps({
    "date": "2027-07-15", "status": "sent", "sent_sources": ["saver_state", "probable_pitcher_cache:2027-07-15:abc"]
}))
(hdir_e / "health_dm_delivery_status.json.dated_used_mismatch.json").write_text(json.dumps({
    "date": "2027-07-15", "status": "sent", "sent_sources": ["saver_state:dated_used_mismatch"]
}))
(hdir_e / "contest_morning_check_dm_delivery_status.json").write_text(json.dumps({
    "date": "2027-07-15", "status": "sent", "sent_sources": ["CRITICAL:contest_state:morning"]
}))
(hdir_e / "contest_stale_reading_dm_delivery_status.json").write_text(json.dumps({
    "date": "2027-07-15", "status": "sent", "sent_sources": ["WARN_ATTENTION:contest_state:stale_reading"]
}))
(hdir_e / "delivery_outcome_unknown" / "2027-07-15.json").write_text(json.dumps({"status": "sent"}))
(hdir_e / "probable_pitcher_cache" / "2027-07-15.invalid_entries.0123456789ab.json").write_text(json.dumps({"reason": "invalid_entries"}))
(hdir_e / "contest_streak_fetch_status.json").write_text(json.dumps({
    "last_success_at": "2027-07-15T17:30:00Z",
    "last_alert_at_by_category": {"actionable": "2027-07-15T05:10:00+00:00"}
}))
# HEAD wrong_game marker on a doubleheader pick (game 700101, while batter is STILL in wrong game 700102!)
dh_pick = dict(head_pick_doc)
dh_pick["pick"] = dict(head_pick_doc["pick"], game_pk=700101, game_time="2027-07-15T22:00:00Z")
(pdir_e / "2027-07-15.json").write_text(json.dumps(dh_pick, indent=2))
(pdir_e / "2027-07-15").mkdir(parents=True, exist_ok=True)
(pdir_e / "2027-07-15" / "decision.json").write_text(json.dumps(head_dec_doc, indent=2))
(hdir_e / "pick_entry_check.json").write_text(json.dumps({
    "date": "2027-07-15", "status": "alerted", "reason": "wrong_game",
    "checked_at": "2027-07-15T16:45:00-04:00", "escalations": ["wrong_game"],
    "game_unresolved": True, "receipt": "attempt-1"
}))

# Run BASESHA health.pick_entry BEFORE check-pick-entered runs, then run BASESHA check-pick-entered while batter is STILL in wrong game!
pe_alerts_before = hpe_mod.check(pdir_e, today=real_date(2027, 7, 15), now=now_holder["utc"])
auth_mod.load_session_cookies = lambda *a, **k: {"ipid": "12345"}
auth_mod.extract_uid = lambda c: 12345
auth_mod.fetch_login_session = lambda **kw: SimpleNamespace(user_id=12345, username="stonehengee", xsid="xs")
cf_mod.fetch_profile = lambda u, c, x: {"activeStreak": 12, "seasonBestStreak": 18, "predictions": []}
cf_mod.fetch_pending_predictions = lambda c, x: [{"roundId": 1, "unitId": 202, "playerId": 9001, "number": 1}]
cli_mod._fetch_rounds = lambda: {1: real_date(2027, 7, 15)}
cli_mod._fetch_bts_to_mlb = lambda: {9001: 100001}

dms.clear()
res_cpe_base = CliRunner().invoke(cli_mod.cli, [
    "check-pick-entered", "--picks-dir", str(pdir_e),
    "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
    "--now-et", "2027-07-15T17:00:00"
])
marker_after_base = json.loads((hdir_e / "pick_entry_check.json").read_text())

# Also test BASESHA run_day on 2027-07-15 when HEAD had NOT yet committed a pick vs AFTER HEAD committed a pick!
# Uncommitted day on 2027-07-15 -> BASESHA skips due to days_remaining=0!
pdir_f = root / "case_f" / "data" / "picks"
(pdir_f / "account_state").mkdir(parents=True, exist_ok=True)
(pdir_f / "account_state" / "contest_streak.json").write_text(json.dumps({
    "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "a" * 32,
    "active_streak": 12, "best_streak": 18, "source": "mlb_bts_profile",
    "source_date": "2027-07-14", "recorded_at": "2027-07-15T13:30:00Z"
}))
# Suppose HEAD preview/check wrote an undelivered 2027-07-15.json before rollback:
undelivered_pick = dict(head_pick_doc, notification_sent=False, notification_channel=None, notification_id=None, delivered_at=None, delivery_attempted=False)
(pdir_f / "2027-07-15.json").write_text(json.dumps(undelivered_pick, indent=2))
now_holder["utc"] = real_datetime(2027, 7, 15, 10, 0, tzinfo=ET).astimezone(UTC)
t1 = real_datetime(2027, 7, 15, 19, 10, tzinfo=ET).astimezone(UTC)
sched_2027 = {"dates": [{"date": "2027-07-15", "games": [{
    "gamePk": 700001, "gameType": "R", "gameDate": t1.strftime("%Y-%m-%dT%H:%M:%SZ"), "officialDate": "2027-07-15",
    "doubleHeader": "N", "gameNumber": 1,
    "status": {"detailedState": "Scheduled", "abstractGameState": "Preview", "abstractGameCode": "P", "statusCode": "S"},
    "teams": {"away": {"team": {"id": 147, "name": "NYY", "abbreviation": "NYY"}},
              "home": {"team": {"id": 111, "name": "BOS", "abbreviation": "BOS"}}}
}]}]}
class R:
    def __init__(self, p): self._b = json.dumps(p).encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *e): return False
sch_mod.retry_urlopen = lambda req, *a, **k: R(sched_2027 if "date=2027-07-15" in str(getattr(req, "full_url", req)) else {"dates": []})
picks_mod.retry_urlopen = sch_mod.retry_urlopen
orch_mod.run_cascade = lambda tiers, ds: (pd.DataFrame([{
    "date": "2027-07-15", "batter_id": 100001, "batter_name": "Juan Soto", "team": "NYY", "opp_team": "BOS",
    "pitcher_id": 300001, "pitcher_name": "P1", "pitcher_team": "BOS", "p_hit": 0.21, "p_game_hit": 0.85,
    "est_pa": 4.5, "lineup": 2, "flags": "", "lineup_confirmed": True, "game_pk": 700001, "game_time": t1.strftime("%Y-%m-%dT%H:%M:%SZ")
}]), "local")
dms.clear()
sch_mod.run_day("2027-07-15", {
    "orchestrator": {"picks_dir": str(pdir_f), "models_dir": str(models_dir), "heartbeat_path": str(root / "case_f" / ".hb")},
    "bluesky": {"dm_recipient": "stonehengee.bsky.social"},
    "scheduler": {"pick_delivery": "dm", "entry_intent": "enter", "shadow_model": False, "live_forward_capture_on_lock": False},
    "health_checks": {"enabled": False},
    "tiers": [{"name": "local", "type": "local"}],
})
dec_f = json.loads((pdir_f / "2027-07-15" / "decision.json").read_text())

report["status_and_day_effects"] = {
    "pe_alerts_on_wrong_game_marker": [(a.level, a.source, a.message[:80]) for a in pe_alerts_before],
    "basesha_cpe_exit_on_wrong_game_dh": res_cpe_base.exit_code,
    "basesha_cpe_sent_false_all_clear_dm": list(dms_after_cpe := []) or None,
    "basesha_marker_after_wrong_game": marker_after_base,
    "basesha_uncommitted_2027_day_action": dec_f["action"],
    "basesha_uncommitted_2027_day_dms": list(dms),
}
print(json.dumps(report))
"""
    )
    # Fix small variable capture for dms_after_cpe in driver before running:
    txt = driver.read_text().replace(
        'dms.clear()\nres_cpe_base',
        'dms.clear()\nres_cpe_base'
    ).replace(
        'marker_after_base = json.loads((hdir_e / "pick_entry_check.json").read_text())',
        'marker_after_base = json.loads((hdir_e / "pick_entry_check.json").read_text())\ncpe_dms = list(dms)'
    ).replace(
        '"basesha_cpe_sent_false_all_clear_dm": list(dms_after_cpe := []) or None,',
        '"basesha_cpe_sent_false_all_clear_dm": cpe_dms,'
    )
    driver.write_text(txt)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(basesha_root / "src")
    proc = subprocess.run(
        [sys.executable, str(driver), str(work)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"BASESHA rollback probe failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    rep = json.loads(proc.stdout.strip().splitlines()[-1])

    # Assertions proving each rollback hazard:
    assert rep["pick_file"]["basesha_loads_head_pick"] is True
    assert rep["pick_file"]["resave_drops_keys"] == ["automatic_reading", "chosen_on"], f"got: {rep['pick_file']}"
    assert rep["decision_file"]["existing_longest_streak_loads"] is True
    assert rep["decision_file"]["existing_is_scoreable_commit"] is True
    assert rep["decision_file"]["existing_decision_objective_normalized_to"] == "unknown"
    assert rep["decision_file"]["writing_outstanding_h_pick_returns"] is None
    assert rep["decision_file"]["writing_outstanding_h_pick_leaves_file_absent"] is True
    assert rep["streak_file"]["basesha_save_drops_keys"] == ["applied_dates", "season"]
    assert rep["contest_and_manual_files"]["auto_v2_with_revoked_switch"]["basesha_ignores_revoked_switch_and_reads_streak"] == 58
    assert rep["contest_and_manual_files"]["auto_v2_with_revoked_switch"]["auto_v2_best_status_degrades_to"] == "untrusted"
    assert rep["contest_and_manual_files"]["v3_manual_alongside_2026_auto"]["v3_manual_ignored_because_override_expires_at_missing"] is True
    assert rep["contest_and_manual_files"]["v3_manual_alone"]["status"] == "stale"
    assert rep["contest_and_manual_files"]["v3_manual_alone"]["best_status"] == "untrusted"
    assert any(lvl == "CRITICAL" for lvl, _, _ in rep["contest_and_manual_files"]["v3_manual_alone"]["health_alerts"])
    assert rep["status_and_day_effects"]["basesha_cpe_exit_on_wrong_game_dh"] == 0
    assert any("✅" in m for m in rep["status_and_day_effects"]["basesha_cpe_sent_false_all_clear_dm"])
    assert rep["status_and_day_effects"]["basesha_uncommitted_2027_day_action"] == "skip"
    return rep


def main() -> int:
    import tempfile

    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))

    log_path = REPO_ROOT / ".carbon-review" / "whole-range" / "probe_deploy_and_rollback.log"
    results = {}
    with tempfile.TemporaryDirectory(prefix="wr2_probe2_") as tmp:
        tmp_path = Path(tmp)
        mp = pytest.MonkeyPatch()
        try:
            results["head_first_start"] = test_head_first_start_and_half_edited_configs(tmp_path / "p1", mp)
            results["rollback_to_basesha"] = test_rollback_to_basesha_after_head_writes(tmp_path / "p2", mp)
        finally:
            mp.undo()
    assert "bts.model.predict" not in sys.modules
    log_path.write_text(json.dumps(results, indent=2) + "\n")
    print("PROBE 2 PASS:", json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
