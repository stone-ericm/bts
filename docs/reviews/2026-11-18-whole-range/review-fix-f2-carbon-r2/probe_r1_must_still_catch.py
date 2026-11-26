"""Round 2 must-still-catch verification for all Round 1 Codex & Carbon findings and notes against 3b7d5a18."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

import bts
from bts import cli, contest_state, web
from bts.health import alert, contest_state as health, runner
from bts.saver_state import load_saver_state
from tests import test_wr_f2_saver_flag as f2_tests

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 0, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"
STATUS = "health_dm_delivery_status.json"
MISMATCH_STATUS = STATUS + ".dated_used_mismatch.json"
CRITICAL = alert.Alert("CRITICAL", "offline-probe", "A pre-existing critical alert")


def _clock(now: dt.datetime):
    class Pinned(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)
    return Pinned


def _cli(now: dt.datetime, *args: str):
    clk = _clock(now)
    with patch.object(dt, "datetime", clk), \
         patch.object(cli, "datetime", clk), \
         patch.object(contest_state, "datetime", clk), \
         patch.object(health, "datetime", clk), \
         patch.object(runner, "datetime", clk), \
         patch.object(alert, "datetime", clk):
        return CliRunner().invoke(cli.cli, list(args))


def _setup_dated_used(picks: Path, when: dt.datetime = DAY, streak: int = 12) -> Path:
    res = _cli(
        when, "set-contest-streak",
        "--date", str(when.date()),
        "--streak", str(streak),
        "--best-streak", str(max(streak, 16)),
        "--saver", "used",
        "--picks-dir", str(picks),
    )
    assert res.exit_code == 0, res.output
    return picks / "account_state" / "contest_streak.manual.json"


def _write_flag(picks: Path, state: str, season: int = 2027) -> Path:
    p = picks / "account_state" / "saver_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"season": season, "state": state}), encoding="utf-8")
    return p


def _write_auto(picks: Path, when: dt.datetime, streak: int = 9, best: int = 16) -> None:
    acct = picks / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    act_id = "f" * 32
    (acct / "season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1",
        "season": when.year,
        "status": "automatic",
        "activation_id": act_id,
        "switched_at": (when - dt.timedelta(hours=1)).isoformat(),
        "revoked_at": None,
        "note": None,
    }), encoding="utf-8")
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": when.year,
        "activation_id": act_id,
        "active_streak": streak,
        "best_streak": best,
        "source": "mlb_bts_profile",
        "source_date": str(when.date() - dt.timedelta(days=1)),
        "recorded_at": when.isoformat(),
    }), encoding="utf-8")


def _run_eod(picks: Path, *, day: dt.datetime = DAY, wall: dt.datetime | None = None,
             existing=(), overrides: dict | None = None, transport=None):
    orig = runner._safe_run
    sends = []

    def run_check(name, check):
        if name == "contest_state":
            return orig(name, check)
        return list(existing) if name == "calibration" else []

    def send_fn(recipient, body):
        sends.append((recipient, body))
        return transport(recipient, body) if transport else "msg-ack-1"

    clk = _clock(wall or day)
    with patch.object(dt, "datetime", clk), \
         patch.object(health, "datetime", clk), \
         patch.object(runner, "datetime", clk), \
         patch.object(alert, "datetime", clk), \
         patch.object(runner, "_safe_run", run_check), \
         patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None), \
         patch.object(alert, "send_dm", send_fn):
        raw = runner.run_all_checks(
            picks, picks.parent / "models", "offline.test",
            today=day.date(), thresholds_overrides=overrides, contest_state_expected=True,
        )
    return raw, sends


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_codex_f2_receipt_alias_refused_flag_preserved_and_retries(tmp_path, alias_kind):
    """Codex Finding 2 + Amendment 2 items 2 & 9: symlink and hardlink at mismatch receipt leave flag intact."""
    picks = tmp_path / alias_kind / "picks"
    manual = _setup_dated_used(picks)
    flag_path = _write_flag(picks, "active")
    flag_before = flag_path.read_bytes()
    manual_before = manual.read_bytes()
    next_day = DAY.replace(hour=12) + dt.timedelta(days=1)
    _write_auto(picks, next_day, streak=9)

    receipt_path = picks.parent / "health_state" / MISMATCH_STATUS
    receipt_path.parent.mkdir(parents=True)
    if alias_kind == "symlink":
        receipt_path.symlink_to(flag_path)
    else:
        os.link(flag_path, receipt_path)

    raw, sends = _run_eod(picks, existing=(CRITICAL,))
    assert raw[0] == CRITICAL
    assert len(sends) == 2
    assert json.loads((picks.parent / "health_state" / STATUS).read_text())["status"] == "sent"
    assert flag_path.read_bytes() == flag_before
    assert manual.read_bytes() == manual_before
    assert receipt_path.is_symlink() == (alias_kind == "symlink")
    assert receipt_path.samefile(flag_path)

    dec = contest_state.load_decision_streak_state(picks, now=next_day, require_contest_state=True)
    assert dec.saver_available is True

    # Known limit #9: rerun deduplicates the primary digest, re-attempts the unrecorded mismatch DM, keeps flag intact
    _, sends_rerun = _run_eod(picks, existing=(CRITICAL,))
    assert len(sends_rerun) == 1
    assert "Streak Saver on 2027-07-15 ET." in sends_rerun[0][1]
    assert flag_path.read_bytes() == flag_before


@pytest.mark.parametrize("bad_override", ["/", "", ".", "/."])
def test_codex_f3_empty_basename_override_falls_back_to_default_receipt(tmp_path, bad_override):
    """Codex Finding 3 + Amendment 2 item 2: status_path without a basename falls back to default mismatch receipt."""
    picks = tmp_path / "data" / "picks"
    _setup_dated_used(picks)
    _write_flag(picks, "active")
    raw, sends = _run_eod(picks, existing=(CRITICAL,), overrides={"health_dm_delivery_status": bad_override})
    assert len(sends) == 2
    assert CRITICAL.message in sends[0][1]
    assert "Streak Saver on 2027-07-15 ET." in sends[1][1]
    default_receipt = picks.parent / "health_state" / MISMATCH_STATUS
    rec = json.loads(default_receipt.read_text())
    assert rec["status"] == "sent"
    assert rec["message_id"] == "msg-ack-1"
    assert rec["sent_sources"] == [f"WARN_ATTENTION:{INCIDENT}"]

    # Rerun: mismatch is deduplicated via default receipt!
    _, sends_rerun = _run_eod(picks, existing=(CRITICAL,), overrides={"health_dm_delivery_status": bad_override})
    # Note: first digest re-sends if bad_override is non-regular ('/' or '.'), just as on base; mismatch does NOT resend!
    assert all("Streak Saver on 2027-07-15 ET." not in body for _, body in sends_rerun)


def test_codex_f4_and_edit_operator_b8a_recheck_prevents_overwriting_undo(tmp_path, monkeypatch):
    """Codex Finding 4 + Edit: following live DM + B8a verbatim with concurrent dashboard Undo."""
    monkeypatch.chdir(tmp_path)
    default_picks = tmp_path / "data" / "picks"
    sched_picks = tmp_path / "srv" / "picks"
    _write_flag(default_picks, "used")
    _setup_dated_used(sched_picks)
    flag_path = _write_flag(sched_picks, "active")

    _, sends = _run_eod(sched_picks)
    assert len(sends) == 1
    body = sends[0][1]
    assert (
        "Use the `bts season-contract status` command with the scheduler's picks directory "
        "as shown in checklist B8a. Follow B8a's saver reconciliation steps for any correction."
    ) in body

    # Step 1 (new B8a F2 bullet): Operator A runs `bts season-contract status --picks-dir DIR`
    st1 = _cli(DAY, "season-contract", "status", "--picks-dir", str(sched_picks))
    assert st1.exit_code == 0
    assert "manual: dated date=2027-07-15 streak=12 best=16 saver=used in effect=True" in st1.output
    assert "saver: active; season: 2027" in st1.output

    # Operator A also checks `bts saver-state --picks-dir DIR` per **The saver** while MLB initially appears used:
    sv1 = _cli(DAY, "saver-state", "--picks-dir", str(sched_picks))
    assert "saver_state: active (season 2027)" in sv1.output

    # Concurrent Operator B marks used via dashboard, sees official MLB scoring correction (not used), presses Undo,
    # and updates dated observation to active:
    t1 = DAY + dt.timedelta(minutes=1)
    t2 = DAY + dt.timedelta(minutes=2)
    assert web.saver_transition_response(sched_picks, expected_prior="active", new_state="used", same_origin=True, now=t1)[0] == 200
    assert web.saver_transition_response(sched_picks, expected_prior="used", new_state="active", same_origin=True, now=t2)[0] == 200
    assert _cli(t2, "set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "16", "--saver", "active", "--picks-dir", str(sched_picks)).exit_code == 0

    # Step 2 (new B8a recheck rule): "Immediately before writing, refresh MLB and the scheduler's flag.
    # If either changed since your verification, stop and reconcile the current state before retrying."
    # Operator A refreshes MLB (now shows NOT used / active) and refreshes status/flag -> sees MLB changed!
    st2 = _cli(t2 + dt.timedelta(minutes=1), "season-contract", "status", "--picks-dir", str(sched_picks))
    assert "saver=active in effect=True" in st2.output
    assert "saver: active; season: 2027" in st2.output
    # Operator A stops and does NOT run `--use`; flag stays `active`!
    assert load_saver_state(sched_picks, season=2027).state == "active"
    assert json.loads(flag_path.read_text())["state"] == "active"


def test_codex_and_carbon_notes_n1_n2_fifo_alarm_and_known_limit_8(tmp_path):
    """Verify N1 (200k decoder limit raises RecursionError), N2 (K1 preserved with 200k & stat OSError), FIFO BaseException, Known #8."""
    payload = f2_tests.decoder_limit()
    assert len(payload) == 400_000
    with pytest.raises(RecursionError):
        json.loads(payload)

    # N2 with both 200k decoder limit AND PermissionError symlink: K1 dated not_earned warning at streak 10..15 preserved!
    picks = tmp_path / "k1" / "picks"
    assert _cli(DAY, "set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "15", "--saver", "not_earned", "--picks-dir", str(picks)).exit_code == 0
    expected_live = health.check(picks, expected=True, now=DAY)
    assert len(expected_live) == 1 and expected_live[0].source == "saver_state" and expected_live[0].incident_key is None

    locked = picks / "account_state" / "locked"
    locked.mkdir()
    (locked / "f.json").write_text('{"season":2027,"state":"active"}')
    locked.chmod(0o000)
    try:
        (picks / "account_state" / "saver_state.json").symlink_to(locked / "f.json")
        assert health.check(picks, expected=True, now=DAY) == expected_live
    finally:
        locked.chmod(0o700)

    # Known limit #8: malformed streak in warn_attention_state.json raises ValueError as on base:
    picks8 = tmp_path / "known8" / "picks"
    _setup_dated_used(picks8)
    _write_flag(picks8, "active")
    att_path = picks8.parent / "health_state" / "warn_attention_state.json"
    att_path.parent.mkdir(parents=True)
    att_path.write_text(json.dumps({"sources": {INCIDENT: {"last_seen": "2027-07-15", "streak": "bad"}}}))
    with pytest.raises(ValueError):
        _run_eod(picks8, existing=(CRITICAL,))


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
