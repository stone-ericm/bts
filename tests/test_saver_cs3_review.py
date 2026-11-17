# Adopted from the Carbon review of change set 3, round 1 (marker CS3RG-d5e3275c).
# Source: docs/reviews/2026-11-14-season-rollover/build/cs3-review/carbon-r1/test_cs3_adversarial_probes.py
# Its three strict expected-failure markers (the review's blocking finding B1) are removed: the fix
# makes those cases pass, and they are now plain regression tests. Three tests were changed by the
# lead where the probe itself was at fault or pinned the old message; each says so where it is.
"""Adversarial regression probes for CS3 (Streak Saver policy mapping, season binding, health, web, CLI).

All clocks are pinned or derived from the active ET clock; no network or external account access is used.
"""
from __future__ import annotations

import json
import os
import shlex
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

import bts.contest_state as cs
import bts.scheduler as sched
import bts.web as web
from bts.cli import cli
from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.health.alert import dispatch_dm_for_health_alerts
from bts.health.attention import apply_warn_attention_policy
from bts.health.contest_state import check as check_contest_health
from bts.picks import DailyPick, Pick, save_pick
from bts.saver_state import (
    SaverState,
    load_saver_state,
    maybe_auto_earn_saver,
    policy_saver_available,
    transition_saver_state,
)
from bts.strategy import DecisionContext, _load_mdp, resolve_policy_decision

ET = ZoneInfo("America/New_York")
NOW_2027 = datetime(2027, 7, 15, 16, 0, tzinfo=timezone.utc)


def _seed_contest_auto(picks_dir: Path, streak: int, best: int | None = None,
                       source_date: str | None = "2027-07-15") -> Path:
    d = picks_dir / "account_state"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "contest_streak.json"
    p.write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": streak,
        "best_streak": streak if best is None else best,
        "source": "mlb_bts_profile",
        "source_date": source_date,
    }))
    return p


def _seed_saver(picks_dir: Path, state: str, season: int = 2027) -> Path:
    d = picks_dir / "account_state"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "saver_state.json"
    p.write_text(json.dumps({"season": season, "state": state}))
    return p


@pytest.mark.parametrize("bad_streak", [None, "7", "12", [7], {"n": 7}, True, False, -1, -40, 7.5])
def test_block_model_only_invalid_json_streak_fails_closed_as_contest_state_error(
    tmp_path: Path, bad_streak,
):
    _seed_saver(tmp_path, "not_earned", season=2027)
    (tmp_path / "streak.json").write_text(json.dumps({"season": 2027, "streak": bad_streak}))

    with pytest.raises(ContestStateError):
        load_decision_streak_state(tmp_path, require_contest_state=False, now=NOW_2027)


def test_block_deliver_and_lock_pick_catches_corrupt_model_streak_without_type_error(tmp_path: Path):
    # Changed by the lead: _deliver_and_lock_pick reads the wall clock for the model file's season, so
    # a file tagged 2027 read in another year is "another season" (streak 0) and nothing is corrupt.
    # With the fixed year the test failed for that reason, not for the TypeError it was written for.
    cur_yr = datetime.now(ET).year
    _seed_saver(tmp_path, "not_earned", season=cur_yr)
    (tmp_path / "streak.json").write_text(json.dumps({"season": cur_yr, "streak": None}))

    dp = DailyPick(
        date=f"{cur_yr}-07-15",
        run_time=f"{cur_yr}-07-15T14:00:00+00:00",
        pick=Pick(
            "Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
            0.82, [], False, 1, f"{cur_yr}-07-15T23:10:00Z", "NYY",
        ),
        double_down=None,
        runner_up=None,
    )
    save_pick(dp, tmp_path)
    st_sched = sched.SchedulerState(
        date=f"{cur_yr}-07-15",
        schedule_fetched_at="",
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
    )
    cfg = {
        "orchestrator": {"picks_dir": str(tmp_path)},
        "scheduler": {"pick_delivery": "dm", "contest_state_required": False},
        "bluesky": {"dm_recipient": "eric.test"},
    }
    alerts_called: list[Exception] = []
    with patch("bts.dm.send_dm", return_value="msg-1"), \
         patch.object(sched, "_now_et", return_value=datetime(cur_yr, 7, 15, 12, 0, tzinfo=ET)), \
         patch.object(sched, "_void_committed_games", return_value={}), \
         patch.object(sched, "_alert_contest_state_failure", side_effect=lambda _c, e: alerts_called.append(e)):
        ok = sched._deliver_and_lock_pick(dp, cfg, tmp_path, st_sched, f"{cur_yr}-07-15", "check-1")

    assert ok is False
    assert len(alerts_called) == 1
    assert isinstance(alerts_called[0], ContestStateError)


@pytest.mark.parametrize("bad_streak", [True, -40])
def test_block_bool_and_negative_model_streak_is_rejected(
    tmp_path: Path, bad_streak,
):
    # Changed by the lead: the reviewer's search for a real policy flip with this value raised
    # IndexError inside the policy tables for a bool streak (which the strict expected-failure
    # marker hid). What matters is kept: the value never reaches the policy.
    _seed_saver(tmp_path, "not_earned", season=2027)
    (tmp_path / "streak.json").write_text(json.dumps({"season": 2027, "streak": bad_streak}))

    # Should reject corrupt model streak rather than returning saver_available=True:
    with pytest.raises(ContestStateError):
        load_decision_streak_state(tmp_path, require_contest_state=False, now=NOW_2027)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod 000")
def test_health_unreadable_saver_file_warns_and_pasted_init_command_repairs_it(tmp_path: Path):
    _seed_contest_auto(tmp_path, streak=0, best=0, source_date="2027-03-24")
    saver_path = _seed_saver(tmp_path, "not_earned", season=2026)
    saver_path.chmod(0o000)
    try:
        alerts = check_contest_health(
            tmp_path, expected=True, now=datetime(2027, 3, 24, 23, 0, tzinfo=ET),
        )
        saver_alerts = [a for a in alerts if a.source == "saver_state"]
        assert len(saver_alerts) == 1 and saver_alerts[0].level == "WARN"
        cmd_str = saver_alerts[0].message.split("`")[1]
        tokens = shlex.split(cmd_str)
        assert tokens[0] == "bts"
        res = CliRunner().invoke(cli, tokens[1:] + ["--picks-dir", str(tmp_path)])
        assert res.exit_code == 0, res.output
        # Changed by the lead: the message now lists a command per case, not-earned first
        # (the contract's row 5 amendment); the reviewer's probe expected the only command, `used`.
        assert load_saver_state(tmp_path, season=2027).state == "not_earned"
        after = check_contest_health(
            tmp_path, expected=True, now=datetime(2027, 3, 24, 23, 0, tzinfo=ET),
        )
        assert [a for a in after if a.source == "saver_state"] == []
    finally:
        if saver_path.exists():
            saver_path.chmod(0o600)


def test_eod_health_dm_day1_day2_day30_saver_attention_vs_coverage_lag(tmp_path: Path):
    _seed_contest_auto(tmp_path, streak=4, best=9, source_date="2026-09-27")
    _seed_saver(tmp_path, "used", season=2026)
    # Two resolved 2027 picks create a >=2 coverage-lag WARN (source='contest_state') alongside saver_state WARN
    (tmp_path / "2027-03-24.json").write_text(json.dumps({"result": "hit"}))
    (tmp_path / "2027-03-25.json").write_text(json.dumps({"result": "hit"}))

    attn_path = tmp_path / "health_state" / "warn_attention_state.json"
    dm_status_path = tmp_path / "health_state" / "health_dm_delivery_status.json"
    sent: list[tuple[int, str]] = []

    for day_num in range(1, 31):
        cur_date = date.fromordinal(date(2027, 3, 24).toordinal() + day_num - 1)
        now_dt = datetime(cur_date.year, cur_date.month, cur_date.day, 23, 0, tzinfo=ET)
        raw = check_contest_health(tmp_path, expected=True, now=now_dt)
        assert {(a.level, a.source) for a in raw} == {("WARN", "contest_state"), ("WARN", "saver_state")}
        pol_crit, warn_attn = apply_warn_attention_policy(raw, state_path=attn_path, today=cur_date)
        assert pol_crit == []
        assert [a.source for a in warn_attn] == ["saver_state"]

        with patch("bts.health.alert.send_dm", side_effect=lambda _r, b, dn=day_num: sent.append((dn, b))):
            ok1 = dispatch_dm_for_health_alerts(
                raw + pol_crit, "eric.test", warn_attention=warn_attn,
                status_path=dm_status_path, now_et_date=cur_date,
            )
            # Second run on the same ET date is suppressed by per-day source-set dedup:
            ok2 = dispatch_dm_for_health_alerts(
                raw + pol_crit, "eric.test", warn_attention=warn_attn,
                status_path=dm_status_path, now_et_date=cur_date,
            )
        assert ok1 is True and ok2 is False

    assert len(sent) == 30
    assert "1st observed day" in sent[0][1] and "[contest_state]" not in sent[0][1]
    assert "2nd consecutive day" in sent[1][1]
    assert "30th consecutive day" in sent[29][1]


@pytest.mark.parametrize("bad_manual", ["unreadable", "non_utf8", "non_object", "directory", "malformed"])
def test_dashboard_survives_corrupt_manual_contest_file_and_midnight_et_posts(
    tmp_path: Path, monkeypatch, bad_manual: str,
):
    if bad_manual == "unreadable" and os.geteuid() == 0:
        pytest.skip("root bypasses chmod 000")

    _seed_saver(tmp_path, "active", season=2026)
    manual_path = tmp_path / "account_state" / "contest_streak.manual.json"
    if bad_manual == "directory":
        manual_path.mkdir(parents=True)
    else:
        payload = {
            "unreadable": b'{"active_streak": 7}',
            "non_utf8": b"\xff\xfe",
            "non_object": b"[1, 2, 3]",
            "malformed": b"{truncated",
        }[bad_manual]
        manual_path.write_bytes(payload)
        if bad_manual == "unreadable":
            manual_path.chmod(0o000)

    try:
        dec31_et = datetime(2027, 1, 1, 4, 59, tzinfo=timezone.utc)   # 2026-12-31 23:59 ET
        jan1_et = datetime(2027, 1, 1, 5, 1, tzinfo=timezone.utc)     # 2027-01-01 00:01 ET

        ctx_2026 = web.saver_dashboard_context(tmp_path, now=dec31_et)
        assert (ctx_2026.state, ctx_2026.button, ctx_2026.warning, ctx_2026.nudge) == (
            "active", "mark_used", False, False,
        )

        # POST at 2027-01-01 00:01 ET targets season 2027 (uninitialized -> 409) and leaves 2026 file intact
        before_bytes = (tmp_path / "account_state" / "saver_state.json").read_bytes()
        code_2027, _ = web.saver_transition_response(
            tmp_path, expected_prior="active", new_state="used", same_origin=True, now=jan1_et,
        )
        assert code_2027 == 409
        assert (tmp_path / "account_state" / "saver_state.json").read_bytes() == before_bytes

        # POST at 2026-12-31 23:59 ET targets season 2026 -> 200 used
        code_2026, _ = web.saver_transition_response(
            tmp_path, expected_prior="active", new_state="used", same_origin=True, now=dec31_et,
        )
        assert code_2026 == 200
        assert load_saver_state(tmp_path, season=2026).state == "used"
    finally:
        if bad_manual == "unreadable" and manual_path.exists():
            manual_path.chmod(0o600)


def test_cli_use_succeeds_when_auto_earn_promotes_not_earned_to_active_mid_command(tmp_path: Path):
    _seed_saver(tmp_path, "not_earned", season=2027)
    calls: list[tuple[str, str]] = []
    real_transition = transition_saver_state

    def racing_transition(p_dir, *, expected_prior, new_state, season, source, force=False, peer=None):
        calls.append((expected_prior, new_state))
        if len(calls) == 1 and expected_prior == "not_earned":
            # Simulate fetch auto-earn promoting not_earned -> active just before the CLI lock
            real_transition(p_dir, expected_prior="not_earned", new_state="active", season=season, source="auto_earn")
        return real_transition(
            p_dir, expected_prior=expected_prior, new_state=new_state,
            season=season, source=source, force=force, peer=peer,
        )

    with patch("bts.saver_state.transition_saver_state", side_effect=racing_transition):
        res = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(tmp_path), "--season", "2027", "--use"])

    assert res.exit_code == 0, res.output
    assert calls == [("not_earned", "used"), ("active", "used")]
    assert load_saver_state(tmp_path, season=2027).state == "used"
    assert res.output.splitlines()[-1] == "saver_state: used (season 2027)"


def test_mistaken_use_then_undo_never_earned_behavior_at_streaks_5_10_16(tmp_path: Path):
    _seed_saver(tmp_path, "not_earned", season=2027)
    assert CliRunner().invoke(
        cli, ["saver-state", "--picks-dir", str(tmp_path), "--season", "2027", "--use"],
    ).exit_code == 0
    assert CliRunner().invoke(
        cli, ["saver-state", "--picks-dir", str(tmp_path), "--season", "2027", "--undo"],
    ).exit_code == 0
    assert load_saver_state(tmp_path, season=2027).state == "active"

    mdp = _load_mdp()
    for streak, expected_warn_flag in [(5, False), (10, False), (16, True)]:
        _seed_contest_auto(tmp_path, streak=streak, best=streak, source_date="2027-08-23")
        dec = load_decision_streak_state(tmp_path, now=datetime(2027, 8, 23, 16, 0, tzinfo=timezone.utc))
        assert dec.saver_flag_state == "active"
        assert dec.saver_available is True
        assert check_contest_health(
            tmp_path, expected=True, now=datetime(2027, 8, 23, 16, 0, tzinfo=timezone.utc),
        ) == []
        dctx = web.saver_dashboard_context(tmp_path, now=datetime(2027, 8, 23, 16, 0, tzinfo=timezone.utc))
        assert dctx.button == "mark_used"
        assert dctx.warning is expected_warn_flag
        if streak == 16:
            pctx = DecisionContext(0.82, 0.82, True, "2027-08-23", True, mdp, best_streak=16, best_status="trusted")
            assert resolve_policy_decision(pctx, 16, False).action == "single"
            assert resolve_policy_decision(pctx, 16, dec.saver_available).action == "double"
