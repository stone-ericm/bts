"""Item 4 probe: R64 operator-facing text rendering, B8a verification, and all 5 worst ordinary states via real CLI, dashboard, and scheduler."""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import unicodedata
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner
from pytest import MonkeyPatch

import bts
from bts import cli, contest_state, orchestrator, scheduler, strategy, web
from bts.health import alert, attention, contest_state as health, runner
from bts.saver_state import load_saver_state

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 30, tzinfo=ET)
NYE = REAL_DT(2027, 12, 31, 23, 52, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"
ACT_ID = "c" * 32


def make_clock(fixed_now: dt.datetime):
    class Clock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    return Clock


def invoke_cli(now: dt.datetime, *args: str):
    clk = make_clock(now)
    with patch.object(dt, "datetime", clk), \
         patch.object(cli, "datetime", clk), \
         patch.object(contest_state, "datetime", clk), \
         patch.object(health, "datetime", clk), \
         patch.object(runner, "datetime", clk), \
         patch.object(alert, "datetime", clk):
        return CliRunner().invoke(cli.cli, list(args))


def write_auto(picks: Path, now: dt.datetime, streak: int = 12, best: int = 15) -> None:
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    yr = now.astimezone(ET).year
    (acct / "season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1",
        "season": yr,
        "status": "automatic",
        "activation_id": ACT_ID,
        "switched_at": f"{yr}-04-01T12:00:00Z",
        "revoked_at": None,
        "note": None,
    }), encoding="utf-8")
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": yr,
        "activation_id": ACT_ID,
        "active_streak": streak,
        "best_streak": best,
        "source": "mlb_bts_profile",
        "source_date": (now.astimezone(ET).date() - dt.timedelta(days=1)).isoformat(),
        "recorded_at": now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }), encoding="utf-8")


def test_render_live_and_historical_messages_and_check_r64_invariants(tmp_path):
    picks = tmp_path / "oom" / "path with 'quotes'" / "picks"
    assert invoke_cli(
        DAY, "set-contest-streak", "--date", "2027-07-15",
        "--streak", "14", "--best-streak", "15", "--saver", "used",
        "--picks-dir", str(picks),
    ).exit_code == 0

    for flag_val, exp_raw_len, exp_dm_len in (("active", 439, 501), ("not_earned", 443, 505)):
        (picks / "account_state" / "saver_state.json").write_text(
            json.dumps({"season": 2027, "state": flag_val}), encoding="utf-8",
        )
        live = health.check(picks, expected=True, now=DAY)
        assert len(live) == 1 and live[0].incident_key == INCIDENT
        expected_raw = (
            f"Streak Saver on 2027-07-15 ET. The dated observation says the saver was used. "
            f"The flag for season 2027 read {flag_val} at this check. A later automatic decision may count "
            f"a saver as available. Verify the current season's saver and prior entries in MLB before "
            f"any correction. Use the `bts season-contract status` command with the scheduler's picks "
            f"directory as shown in checklist B8a. Follow B8a's saver reconciliation steps for any correction."
        )
        assert live[0].message == expected_raw
        assert len(live[0].message) == exp_raw_len
        assert str(picks) not in live[0].message
        assert re.findall(r"`([^`]+)`", live[0].message) == ["bts season-contract status"]
        assert "--use" not in live[0].message and "--init" not in live[0].message
        assert attention.OOM_EVIDENCE_RE.search(f"{live[0].source} {live[0].message}") is None

        dm_live = alert.format_health_dm_body([], [alert.Alert("WARN", "saver_state", live[0].message + " (1st observed day)", incident_key=INCIDENT)])
        assert len(dm_live) == exp_dm_len
        assert sum(not unicodedata.combining(c) for c in dm_live) == exp_dm_len

        hist = health.check(picks, expected=True, now=DAY, historical_day=DAY.date())
        assert len(hist) == 1 and hist[0].incident_key == INCIDENT
        expected_hist = (
            f"Prior slate 2027-07-15 ET. The dated observation recorded the saver as used. "
            f"At this check the flag for season 2027 read {flag_val}. {health.HISTORICAL_POINTER}"
        )
        assert hist[0].message == expected_hist
        assert "`" not in hist[0].message and str(picks) not in hist[0].message
        assert attention.OOM_EVIDENCE_RE.search(f"{hist[0].source} {hist[0].message}") is None


def test_five_worst_ordinary_states_following_dm_and_b8a_verbatim(tmp_path, monkeypatch):
    # State 1: Dated observation was the mistaken side (both active and not_earned flags when MLB shows not used).
    for true_mlb_saver, streak in (("active", 12), ("not_earned", 6)):
        p1 = tmp_path / f"state1_{true_mlb_saver}" / "picks"
        write_auto(p1, DAY, streak=streak, best=max(streak, 8 if true_mlb_saver == "not_earned" else 15))
        assert invoke_cli(DAY, "saver-state", "--init", true_mlb_saver, "--season", "2027", "--picks-dir", str(p1)).exit_code == 0
        # Mistakenly filed --saver used on D:
        assert invoke_cli(DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", str(streak), "--best-streak", str(max(streak, 15)), "--saver", "used", "--picks-dir", str(p1)).exit_code == 0
        assert len(health.check(p1, expected=True, now=DAY)) == 1
        # Follow B8a F2 bullet: run `bts season-contract status --picks-dir DIR`:
        st = invoke_cli(DAY, "season-contract", "status", "--picks-dir", str(p1))
        assert st.exit_code == 0
        assert "saver=used in effect=True" in st.output
        assert f"saver: {true_mlb_saver}; season: 2027" in st.output
        # MLB shows saver NOT used -> B8a says: "leave the flag as it is (a dated observation applies only on its own date)."
        assert load_saver_state(p1, season=2027).state == true_mlb_saver
        # Next day automatic decision reads the untouched flag accurately:
        next_day = DAY.replace(hour=11) + dt.timedelta(days=1)
        write_auto(p1, next_day, streak=streak, best=max(streak, 8 if true_mlb_saver == "not_earned" else 15))
        dec = contest_state.load_decision_streak_state(p1, require_contest_state=True, now=next_day)
        assert dec.saver_available is True

    # State 2: Observation is corrected to `active` later that day before operator acts on the warning.
    p2 = tmp_path / "state2" / "picks"
    write_auto(p2, DAY, streak=12, best=15)
    assert invoke_cli(DAY, "saver-state", "--init", "active", "--season", "2027", "--picks-dir", str(p2)).exit_code == 0
    assert invoke_cli(DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p2)).exit_code == 0
    assert len(health.check(p2, expected=True, now=DAY)) == 1
    # Observation corrected to `active` at 22:40 ET:
    later = DAY + dt.timedelta(minutes=10)
    assert invoke_cli(later, "set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "15", "--saver", "active", "--picks-dir", str(p2)).exit_code == 0
    st2 = invoke_cli(later + dt.timedelta(minutes=5), "season-contract", "status", "--picks-dir", str(p2))
    assert "saver=active in effect=True" in st2.output and "saver: active; season: 2027" in st2.output
    # MLB shows not used -> leave flag as active; rerun EOD produces 0 warnings:
    assert health.check(p2, expected=True, now=later) == []
    assert load_saver_state(p2, season=2027).state == "active"

    # State 3: Dashboard button changes the flag between the warning and the operator's action (including concurrent Undo).
    p3 = tmp_path / "state3" / "picks"
    write_auto(p3, DAY, streak=14, best=15)
    assert invoke_cli(DAY, "saver-state", "--init", "active", "--season", "2027", "--picks-dir", str(p3)).exit_code == 0
    assert invoke_cli(DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "14", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p3)).exit_code == 0
    # 3a: Dashboard already marked used while MLB shows used -> operator runs `bts saver-state --picks-dir DIR`, sees `used`, runs nothing:
    assert web.saver_transition_response(p3, expected_prior="active", new_state="used", same_origin=True, now=DAY + dt.timedelta(minutes=1))[0] == 200
    assert "saver_state: used (season 2027)" in invoke_cli(DAY + dt.timedelta(minutes=2), "saver-state", "--picks-dir", str(p3)).output
    # 3b: Reset to active; Operator A verifies MLB=used & flag=active; Operator B marks used then Undo after MLB scoring change to not-used;
    # Operator A obeys B8a's immediate-before-write refresh of MLB & flag -> sees MLB changed -> stops without writing `used`!
    assert web.saver_transition_response(p3, expected_prior="used", new_state="active", same_origin=True, now=DAY + dt.timedelta(minutes=3))[0] == 200
    assert "saver_state: active (season 2027)" in invoke_cli(DAY + dt.timedelta(minutes=4), "saver-state", "--picks-dir", str(p3)).output
    assert web.saver_transition_response(p3, expected_prior="active", new_state="used", same_origin=True, now=DAY + dt.timedelta(minutes=5))[0] == 200
    assert web.saver_transition_response(p3, expected_prior="used", new_state="active", same_origin=True, now=DAY + dt.timedelta(minutes=6))[0] == 200
    # Immediate-before-write refresh sees MLB changed -> Operator A stops; flag stays `active`!
    assert load_saver_state(p3, season=2027).state == "active"

    # State 4: Warning read next morning after automatic path has ALREADY saved a pick (both active and not_earned flags).
    from tests.test_saved_pick_paths_cs5a import world
    for init_flag in ("active", "not_earned"):
        with MonkeyPatch.context() as mp:
            sub = tmp_path / f"state4_{init_flag}"
            config, p4, now_box, _ = world.__wrapped__(sub, mp)
            mp.setattr(strategy, "_mdp_cache", None)
            assert invoke_cli(DAY, "saver-state", "--init", init_flag, "--season", "2027", "--picks-dir", str(p4)).exit_code == 0
            assert invoke_cli(DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "14", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p4)).exit_code == 0
            manual_path = p4 / "account_state" / "contest_streak.manual.json"
            manual_bytes = manual_path.read_bytes()

            morning = REAL_DT(2027, 7, 16, 10, 30, tzinfo=ET)
            now_box[0] = morning
            write_auto(p4, morning, streak=8, best=15)
            frame, tier = orchestrator.run_cascade.return_value
            d_str = "2027-07-16"
            orchestrator.run_cascade.return_value = (
                frame.assign(p_game_hit=[0.99, 0.96], game_time=[f"{d_str}T23:10:00Z", f"{d_str}T23:40:00Z"]),
                tier,
            )
            clk = make_clock(morning)
            with patch.object(dt, "datetime", clk), patch.object(contest_state, "datetime", clk):
                selected = scheduler._refresh_pick_at_fallback_decision(config, d_str, None, 0.03)
            assert selected.daily is not None and selected.daily.chosen_on["kind"] == "automatic"
            saved_pick_path = p4 / f"{d_str}.json"
            saved_pick_before = saved_pick_path.read_bytes()

            # Operator reads warning at 10:45 ET, runs B8a steps (`status --picks-dir`, `saver-state --picks-dir`, `saver-state --use --picks-dir`):
            t_act = morning + dt.timedelta(minutes=15)
            assert "saver: " + init_flag in invoke_cli(t_act, "season-contract", "status", "--picks-dir", str(p4)).output
            assert f"saver_state: {init_flag} (season 2027)" in invoke_cli(t_act, "saver-state", "--picks-dir", str(p4)).output
            use_out = invoke_cli(t_act, "saver-state", "--use", "--picks-dir", str(p4))
            assert use_out.exit_code == 0
            assert load_saver_state(p4, season=2027).state == "used"
            assert saved_pick_path.read_bytes() == saved_pick_before
            assert manual_path.read_bytes() == manual_bytes
            assert contest_state.load_decision_streak_state(p4, require_contest_state=True, now=t_act).saver_available is False

    # State 5: December 31 late evening (23:52 ET, already 2028 UTC) + default-dir vs --picks-dir check + read on Jan 1.
    monkeypatch.chdir(tmp_path)
    default_picks = tmp_path / "data" / "picks"
    default_picks.mkdir(parents=True)
    assert invoke_cli(NYE, "saver-state", "--init", "used", "--season", "2027", "--picks-dir", str(default_picks)).exit_code == 0

    p5 = tmp_path / "box_nye" / "picks"
    assert invoke_cli(NYE, "saver-state", "--init", "active", "--season", "2027", "--picks-dir", str(p5)).exit_code == 0
    assert invoke_cli(NYE, "set-contest-streak", "--date", "2027-12-31", "--streak", "14", "--best-streak", "15", "--saver", "used", "--picks-dir", str(p5)).exit_code == 0
    nye_alerts = health.check(p5, expected=True, now=NYE)
    assert len(nye_alerts) == 1 and "Streak Saver on 2027-12-31 ET." in nye_alerts[0].message and "season 2027" in nye_alerts[0].message

    # Following B8a (`bts season-contract status --picks-dir DIR`) inspects p5 (`active`), NOT default_picks (`used`):
    st_nye = invoke_cli(NYE, "season-contract", "status", "--picks-dir", str(p5))
    assert "saver: active; season: 2027" in st_nye.output

    # Read on Jan 1, 2028 09:00 ET with a 2028 `active` flag already initialized:
    jan1 = REAL_DT(2028, 1, 1, 9, 0, tzinfo=ET)
    (p5 / "account_state" / "saver_state.json").write_text('{"season": 2028, "state": "active"}', encoding="utf-8")
    p5_bytes_before = (p5 / "account_state" / "saver_state.json").read_bytes()
    # B8a's **The saver** requires "In D's ET year away from midnight" (D=2027-12-31 -> year 2027 != 2028), so operator does NOT run --use!
    st_jan1 = invoke_cli(jan1, "season-contract", "status", "--picks-dir", str(p5))
    assert "ET date: 2028-01-01; year: 2028; inspected season: 2028" in st_jan1.output
    assert (p5 / "account_state" / "saver_state.json").read_bytes() == p5_bytes_before


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
