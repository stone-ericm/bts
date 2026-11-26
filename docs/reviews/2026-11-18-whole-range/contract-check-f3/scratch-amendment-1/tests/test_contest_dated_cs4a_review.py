"""Adversarial review regression tests for CS4a (dated manual contest observations).

All clocks are pinned explicitly; no network or external account access is used.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from click.testing import CliRunner

import bts.contest_state as cs
import bts.health.contest_state as health
import bts.orchestrator as orch
import bts.saver_state as flags
import bts.scheduler as sched
import bts.strategy as strat
from bts.cli import cli
from bts.health.alert import dispatch_dm_for_health_alerts
from bts.health.attention import apply_warn_attention_policy

ET = ZoneInfo("America/New_York")
SCHEMA = "bts_contest_streak_manual_v3"
NOW = dt.datetime(2027, 7, 15, 13, 0, tzinfo=ET)


def _v3(day: str = "2027-07-15", **overrides) -> dict:
    d = dt.date.fromisoformat(day)
    body = {
        "schema_version": SCHEMA,
        "effective_date": day,
        "season": d.year,
        "expires_at": dt.datetime.combine(d, dt.time.max, ET).isoformat(),
        "active_streak": 12,
        "best_streak": 14,
        "saver_state": "not_earned",
        "recorded_at": "2027-07-15T16:00:00Z",
        "source": "manual_cli",
    }
    body.update(overrides)
    return body


def _pin_clock(monkeypatch, instant: dt.datetime):
    class PinnedClock(dt.datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    monkeypatch.setattr(dt, "datetime", PinnedClock)
    monkeypatch.setattr(cs, "datetime", PinnedClock)
    monkeypatch.setattr(health, "datetime", PinnedClock)
    monkeypatch.setattr(flags, "datetime", PinnedClock)
    return PinnedClock


@pytest.mark.parametrize("flag_opt", ["--saver-available", "--saver-unavailable"])
def test_saver_boolean_rejection_names_legacy_boolean_option_specifically(tmp_path: Path, monkeypatch, flag_opt: str):
    _pin_clock(monkeypatch, NOW)
    res = CliRunner().invoke(cli, [
        "set-contest-streak", "--picks-dir", str(tmp_path),
        "--date", "2027-07-15", "--streak", "1", "--best-streak", "1",
        "--saver", "active", flag_opt,
    ])
    assert res.exit_code == 2
    assert "--saver-available/--saver-unavailable" in res.output
    assert not (tmp_path / "account_state").exists()


@pytest.mark.parametrize("day,overrides,expected_check,expected_field", [
    # On 0001-01-01, effective_date.year == 1 == True: only `type(season) is int` rejects bool!
    ("0001-01-01", {"season": True}, 3, "season"),
    # With active_streak=0 or 1, `active <= True <= 57` holds numerically: only `type(best) is int` rejects bool!
    ("2027-07-15", {"active_streak": 0, "best_streak": True}, 6, "best_streak"),
    ("2027-07-15", {"active_streak": 1, "best_streak": True}, 6, "best_streak"),
    ("2027-07-15", {"active_streak": False, "best_streak": 0}, 5, "active_streak"),
    ("2027-07-15", {"season": 2027.0}, 3, "season"),
    ("2027-07-15", {"active_streak": 0.0, "best_streak": 0}, 5, "active_streak"),
    ("2027-07-15", {"active_streak": 0, "best_streak": 0.0}, 6, "best_streak"),
])
def test_bool_and_float_for_integer_fields_rejected_where_bounds_match(
    tmp_path: Path, day: str, overrides: dict, expected_check: int, expected_field: str,
):
    manual = tmp_path / "account_state" / "contest_streak.manual.json"
    manual.parent.mkdir(parents=True)
    manual.write_text(json.dumps(_v3(day, **overrides)), encoding="utf-8")
    with pytest.raises(cs.ContestStateError) as exc_info:
        cs.load_contest_streak_state(tmp_path, now=NOW)
    msg = str(exc_info.value)
    assert f"check {expected_check}," in msg
    assert expected_field in msg


_PAIR_FAILURES = {
    2: ("effective_date", "20270715"),
    3: ("season", 2026),
    4: ("expires_at", "2027-07-15T23:59:59.999999"),
    5: ("active_streak", 401),
    6: ("best_streak", 11),
    7: ("saver_state", "uninitialized"),
    8: ("recorded_at", "2027-07-15T16:00:00"),
    9: ("source", None),
    10: ("reason", None),
}


@pytest.mark.parametrize(
    "i,j",
    [(i, j) for i in range(2, 11) for j in range(i + 1, 11)],
)
def test_all_36_pairwise_invalid_v3_checks_report_lower_numbered_check(tmp_path: Path, i: int, j: int):
    fi, vi = _PAIR_FAILURES[i]
    fj, vj = _PAIR_FAILURES[j]
    manual = tmp_path / "account_state" / "contest_streak.manual.json"
    manual.parent.mkdir(parents=True)
    manual.write_text(json.dumps(_v3(**{fi: vi, fj: vj})), encoding="utf-8")
    for probe_now in (NOW, NOW + dt.timedelta(days=30)):
        with pytest.raises(cs.ContestStateError) as exc_info:
            cs.load_contest_streak_state(tmp_path, now=probe_now)
        msg = str(exc_info.value)
        assert f"check {i}," in msg and fi in msg


def test_unmocked_overlong_saver_flag_never_read_through_orchestrator_scheduler_or_health(
    tmp_path: Path, monkeypatch,
):
    """A 5,000-digit integer in saver_state.json raises ValueError in load_saver_state without mocks."""
    sdir = tmp_path / "account_state"
    sdir.mkdir(parents=True)
    (sdir / "saver_state.json").write_text('{"state":"used","season":' + ("9" * 5000) + "}", encoding="utf-8")
    (sdir / "contest_streak.json").write_text(
        json.dumps({"active_streak": 401, "source_date": "2027-07-14"}), encoding="utf-8"
    )
    (tmp_path / "streak.json").write_text("{corrupt", encoding="utf-8")
    (sdir / "contest_streak.manual.json").write_text(
        json.dumps(_v3("2027-07-15", active_streak=6, best_streak=6, saver_state="not_earned",
                       override_expires_at="2020-01-01T00:00:00Z")),
        encoding="utf-8",
    )

    _pin_clock(monkeypatch, NOW)
    assert health.check(tmp_path, expected=True, now=NOW) == []

    df_preds = pd.DataFrame([
        {
            "batter_name": "Bobby Witt Jr.", "batter_id": 677951, "team": "KC",
            "lineup": 2, "pitcher_name": "Gerrit Cole", "pitcher_id": 543037,
            "pitcher_team": "NYY", "p_game_hit": 0.85, "p_hit_pa": 0.34,
            "flags": "", "game_pk": 7001, "game_time": "2027-07-15T23:10:00Z",
        },
        {
            "batter_name": "Luis Arraez", "batter_id": 650333, "team": "SD",
            "lineup": 1, "pitcher_name": "Logan Webb", "pitcher_id": 657277,
            "pitcher_team": "SF", "p_game_hit": 0.84, "p_hit_pa": 0.33,
            "flags": "", "game_pk": 7002, "game_time": "2027-07-16T01:40:00Z",
        },
    ])
    statuses = {
        7001: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        7002: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    }
    cfg = {
        "orchestrator": {"picks_dir": str(tmp_path), "models_dir": "data/models"},
        "tiers": [{"name": "local", "type": "local"}],
        "scheduler": {
            "pick_delivery": "dm",
            "contest_state_required": True,
            "live_forward_capture_on_lock": False,
        },
        "bluesky": {"dm_recipient": "operator.bsky.social"},
    }
    dms: list[str] = []
    with (
        patch("bts.orchestrator.run_cascade", return_value=(df_preds, "local")),
        patch("bts.picks.get_game_statuses_detailed", return_value=statuses),
        patch("bts.scheduler._delivery_game_statuses", return_value=statuses),
        patch("bts.scheduler._now_et", return_value=NOW),
        patch("bts.dm.send_dm", side_effect=lambda _r, txt: (dms.append(txt) or "dm-1")),
    ):
        _, sel, _ = orch.run_and_pick(cfg, "2027-07-15")
        assert sel is not None and sel.pick_result is not None
        sched_state = sched.SchedulerState(
            date="2027-07-15", schedule_fetched_at="2027-07-15T10:00:00-04:00",
            games=[], confirmed_game_pks=[7001, 7002], runs_completed=[],
            pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        ok = sched._deliver_and_lock_pick(
            sel.pick_result.daily, cfg, tmp_path, sched_state, "2027-07-15", "lineup_lock", selection=sel
        )

    assert ok is True
    assert len(dms) == 1 and "Streak: 6" in dms[0]
    dec_json = json.loads((tmp_path / "2027-07-15" / "decision.json").read_text(encoding="utf-8"))
    assert dec_json["streak"] == 6
    assert dec_json["saver_available"] is True
    assert dec_json["best_status"] == "trusted"
    assert dec_json["contest_source_date"] == "2027-07-15"


def test_tail_policy_honours_trusted_best_at_final_microsecond_of_day(tmp_path: Path):
    sdir = tmp_path / "account_state"
    sdir.mkdir(parents=True)
    day = "2027-09-22"
    end_of_day = dt.datetime(2027, 9, 22, 23, 59, 59, 999999, tzinfo=ET)
    (sdir / "saver_state.json").write_text('{"state":"used","season":' + ("9" * 5000) + "}", encoding="utf-8")
    (sdir / "contest_streak.manual.json").write_text(
        json.dumps(_v3(day, active_streak=2, best_streak=30, saver_state="active")), encoding="utf-8"
    )
    ds = cs.load_decision_streak_state(tmp_path, require_contest_state=True, now=end_of_day)
    assert ds.best_status == "trusted" and ds.best_streak == 30
    ctx = strat.DecisionContext(
        primary_p=0.88, second_p=0.87, has_diff_game=True, date=day,
        allow_double=ds.allow_double, mdp=strat._load_mdp(),
        best_streak=ds.best_streak, best_status=ds.best_status,
    )
    dec = strat.resolve_policy_decision(ctx, ds.streak, ds.saver_available)
    assert dec.action == "skip"


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory 0555 permission")
def test_readonly_directory_distinguishes_append_failure_from_replacement_failure(tmp_path: Path, monkeypatch):
    _pin_clock(monkeypatch, NOW)
    runner = CliRunner()

    # Case 1: read-only account_state with NO existing log -> step (a) open fails; observation not written
    d1 = tmp_path / "no_log"
    s1 = d1 / "account_state"
    s1.mkdir(parents=True)
    (s1 / "contest_streak.manual.json").write_text(json.dumps(_v3(active_streak=3, best_streak=5)), encoding="utf-8")
    s1.chmod(0o555)
    try:
        r1 = runner.invoke(cli, [
            "set-contest-streak", "--picks-dir", str(d1),
            "--date", "2027-07-15", "--streak", "4", "--best-streak", "5", "--saver", "not_earned",
        ])
        assert r1.exit_code == 1
        assert "not written" in r1.output and "partial" in r1.output
        assert json.loads((s1 / "contest_streak.manual.json").read_text(encoding="utf-8"))["active_streak"] == 3
    finally:
        s1.chmod(0o755)

    # Case 2: read-only account_state WITH pre-existing writable log -> step (a) appends; step (b) mkstemp fails!
    d2 = tmp_path / "with_log"
    s2 = d2 / "account_state"
    s2.mkdir(parents=True)
    (s2 / "contest_streak.manual.json").write_text(json.dumps(_v3(active_streak=3, best_streak=5)), encoding="utf-8")
    (s2 / "contest_manual_requests.jsonl").write_text("", encoding="utf-8")
    s2.chmod(0o555)
    try:
        r2 = runner.invoke(cli, [
            "set-contest-streak", "--picks-dir", str(d2),
            "--date", "2027-07-15", "--streak", "4", "--best-streak", "5", "--saver", "not_earned",
        ])
        assert r2.exit_code == 1
        assert "NOT saved" in r2.output and "appended" in r2.output
        assert len((s2 / "contest_manual_requests.jsonl").read_text(encoding="utf-8").splitlines()) == 1
        assert json.loads((s2 / "contest_streak.manual.json").read_text(encoding="utf-8"))["active_streak"] == 3
    finally:
        s2.chmod(0o755)


def test_all_three_row11_warning_commands_and_fetch_auto_earn_isolation(tmp_path: Path, monkeypatch):
    picks = tmp_path / "dir with 'quotes' & spaces"
    sdir = picks / "account_state"
    sdir.mkdir(parents=True)
    (sdir / "saver_state.json").write_text(json.dumps({"season": 2027, "state": "not_earned"}), encoding="utf-8")

    t_day = dt.datetime(2027, 4, 5, 11, 0, tzinfo=ET)
    _pin_clock(monkeypatch, t_day)
    runner = CliRunner()

    # Trigger row-11 warning at streak 12, saver not_earned
    assert runner.invoke(cli, [
        "set-contest-streak", "--picks-dir", str(picks),
        "--date", "2027-04-05", "--streak", "12", "--best-streak", "14", "--saver", "not_earned",
    ]).exit_code == 0
    alerts = health.check(picks, expected=True, now=t_day)
    assert len(alerts) == 1 and alerts[0].source == "saver_state"
    cmds = [s for s in alerts[0].message.split("`")[1::2] if s.startswith("bts set-contest-streak")]
    assert len(cmds) == 3

    # Paste third command with concrete N=8, B=9 -> saver becomes available (<10) and warning clears
    cmd3 = cmds[2].replace("--streak N --best-streak B", "--streak 8 --best-streak 9")
    assert runner.invoke(cli, shlex.split(cmd3)[1:]).exit_code == 0
    ds_after_3 = cs.load_decision_streak_state(picks, now=t_day)
    assert ds_after_3.streak == 8 and ds_after_3.saver_available is True
    assert health.check(picks, expected=True, now=t_day) == []

    # Paste second command (--saver used at streak 12) -> saver spent; then run fetch-contest-streak
    # which promotes saver_state.json to 'active', verifying today's dated decision stays 'used'!
    assert runner.invoke(cli, shlex.split(cmds[1])[1:]).exit_code == 0
    from tests.test_season_contract_cs4b1 import contract
    (sdir / "season_contract.json").write_text(json.dumps(contract()), encoding="utf-8")

    @dataclass
    class FakeSession:
        user_id: int = 4242
        username: str = "operator_bts"
        xsid: str = "fake-xsid"

    with (
        patch("bts.leaderboard.auth.load_session_cookies", return_value={"uid": "4242"}),
        patch("bts.leaderboard.auth.extract_uid", return_value=4242),
        patch("bts.leaderboard.auth.fetch_login_session", return_value=FakeSession()),
        patch("bts.contest_fetch.fetch_profile", return_value={
            "activeStreak": 12, "seasonBestStreak": 14,
            "predictions": [{"roundId": 101, "result": "hit"}],
        }),
        patch("bts.cli._fetch_rounds", return_value={101: dt.date(2027, 4, 4)}),
    ):
        r_fetch = runner.invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
    assert r_fetch.exit_code == 0, r_fetch.output
    assert json.loads((sdir / "saver_state.json").read_text(encoding="utf-8"))["state"] == "active"

    ds_same_day = cs.load_decision_streak_state(picks, require_contest_state=True, now=t_day)
    assert ds_same_day.saver_flag_state == "used" and ds_same_day.saver_available is False
