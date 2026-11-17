"""Adversarial regression tests for CS4b-2 (closing false-green gaps in existing tests).

All clocks are explicitly pinned; DNS and socket connect are forbidden via audit
hook or offline fixtures; fetch/DM/posting transports are captured offline.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import re
import shlex
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import httpx
import pytest
from click.testing import CliRunner

import bts.cli as cli
import bts.contest_fetch as cf
import bts.contest_state as cs
import bts.health.contest_state as hc
import bts.leaderboard.auth as auth
import bts.saver_state as ss
import bts.season_contract as sc

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
ID = "a" * 32


def _sw(**kw):
    return {
        "schema_version": sc.SCHEMA,
        "season": 2027,
        "status": "automatic",
        "activation_id": ID,
        "switched_at": "2027-07-15T16:00:00Z",
        "revoked_at": None,
        "note": None,
    } | kw


def _auto(**kw):
    return {
        "schema_version": sc.AUTO_SCHEMA,
        "season": 2027,
        "activation_id": ID,
        "active_streak": 7,
        "best_streak": 9,
        "source": "mlb_bts_profile",
        "source_date": "2027-07-14",
        "recorded_at": "2027-07-15T14:00:00Z",
        "user_id": 50311,
        "username": "stonehengee",
    } | kw


def _put(picks: Path, name: str, payload) -> Path:
    p = picks / "account_state" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        p.write_bytes(payload)
    else:
        p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture
def offline_env(monkeypatch):
    import pandas  # noqa: F401

    now_box = [NOW]

    class Clock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            cur = now_box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

    monkeypatch.setattr(dt, "datetime", Clock)
    monkeypatch.setattr(httpx, "get", Mock(side_effect=AssertionError("uncaptured httpx.get")))
    monkeypatch.setattr(auth, "load_session_cookies", lambda: {})
    monkeypatch.setattr(auth, "extract_uid", lambda _: "u1")
    monkeypatch.setattr(
        auth, "fetch_login_session",
        lambda **_: SimpleNamespace(user_id=50311, username="stonehengee", xsid="x"),
    )
    monkeypatch.setattr(
        cf, "fetch_profile",
        lambda *_: {"activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}]},
    )
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: now_box[0].astimezone(ET).date() - dt.timedelta(days=1)})
    monkeypatch.setattr("bts.dm.send_dm", Mock(return_value="dm-id"))
    return now_box


@pytest.mark.parametrize(
    "check_num,bad_switch,expected_phrase",
    [
        (1, [], "JSON object"),
        (2, _sw(schema_version="bad"), "schema_version must be"),
        (3, _sw(season=True), "season must be an integer"),
        (4, _sw(status="paused"), "status must be automatic or revoked"),
        (5, _sw(activation_id="A" * 32), "activation_id must be 32 lowercase hexadecimal"),
        (6, _sw(switched_at=None), "switched_at must be present"),
        (7, _sw(revoked_at="2027-07-15T12:00:00"), "revoked_at must be present"),
        (8, _sw(note=["bad"]), "note must be null or a string"),
    ],
)
def test_corrupt_refusal_carries_exact_check_number_and_error_text(
    tmp_path: Path, check_num: int, bad_switch, expected_phrase: str,
):
    """Row 1(1): existing matrix only checked 'corrupt' substring, missing check number and text."""
    picks = tmp_path / "picks"
    _put(picks, "season_contract.json", bad_switch)
    _put(picks, "contest_streak.json", _auto())

    with pytest.raises(cs.ContestStateError) as exc:
        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)

    first_line = str(exc.value).splitlines()[0]
    assert f"check {check_num}:" in first_line
    assert expected_phrase in first_line
    alerts = hc.check(picks, expected=True, now=NOW)
    assert alerts[-1].message == str(exc.value)


def test_other_season_refusal_names_the_files_season(tmp_path: Path):
    """Row 1(3): existing matrix checked 'other_season' without asserting the file's season."""
    picks = tmp_path / "picks"
    _put(picks, "season_contract.json", _sw(season=2026))
    _put(picks, "contest_streak.json", _auto())

    with pytest.raises(cs.ContestStateError) as exc:
        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)

    first_line = str(exc.value).splitlines()[0]
    assert "season 2027 switch is other_season (file season 2026)." in first_line


def test_covered_timestamp_only_auto_reaches_health_missing_source_date_critical(
    tmp_path: Path,
):
    """Row 5: test_missing_source_date_is_critical in health/test_contest_state.py had no
    recorded_at, so require_dated_auto=True turned it into a loader error and stopped
    exercising the post-loader `if state.source_date is None:` CRITICAL branch."""
    picks = tmp_path / "picks"
    _put(picks, "season_contract.json", _sw())
    _put(picks, "contest_streak.json", _auto(
        active_streak=0,
        best_streak=0,
        source_date=None,
        recorded_at="2027-07-15T15:00:00Z",
    ))
    _put(picks, "saver_state.json", {"season": 2027, "state": "not_earned"})

    # Live decision accepts (stale, trusted=False/untrusted):
    dec = cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
    assert dec.streak == 0 and dec.status == "stale"

    # Expected health emits ONLY the missing-source_date CRITICAL (no refusal CRITICAL):
    alerts = hc.check(picks, expected=True, now=NOW)
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL" and alerts[0].source == "contest_state"
    assert "no source_date" in alerts[0].message


def test_corrupt_recovery_runs_status_before_removal_and_checks_intermediate_refusals(
    tmp_path: Path, offline_env,
):
    """Row 2: execute status while corrupt (exit 1), test both unlink->evidence->enable->fetch
    and direct enable->fetch paths on a path containing spaces, single/double quotes and '&'."""
    runner = CliRunner()
    picks = tmp_path / "my 'single' \"double\" & picks"
    picks.mkdir()
    _put(picks, "season_contract.json", [])
    _put(picks, "contest_streak.json", _auto(schema_version="bts_contest_streak_auto_v1"))

    with pytest.raises(cs.ContestStateError) as exc:
        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
    auto_cmds = [
        shlex.split(raw)
        for raw in re.findall(r"`(bts [^`]+)`", str(exc.value))
        if not raw.startswith("bts set-contest-streak")
    ]
    # Step 1: status while switch is still corrupt exits 1 and prints refusal
    r_status1 = runner.invoke(cli.cli, auto_cmds[0][1:])
    assert r_status1.exit_code == 1 and "switch: corrupt" in r_status1.output

    # Step 2: remove corrupt switch -> exposes reason 3 (absent)
    (picks / "account_state/season_contract.json").unlink()
    with pytest.raises(cs.ContestStateError, match="season 2027 switch is absent"):
        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)

    # Step 3: unswitched evidence fetch -> ledger only, still reason 3 (absent)
    assert runner.invoke(cli.cli, auto_cmds[1][1:]).exit_code == 0
    with pytest.raises(cs.ContestStateError, match="season 2027 switch is absent"):
        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)

    # Step 4: status -> exit 0; Step 5: enable -> exposes reason 5 (schema_version)
    assert runner.invoke(cli.cli, auto_cmds[2][1:]).exit_code == 0
    assert runner.invoke(cli.cli, auto_cmds[3][1:]).exit_code == 0
    with pytest.raises(cs.ContestStateError, match="schema_version is 'bts_contest_streak_auto_v1'"):
        cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW)

    # Step 6: switched fetch -> accepted!
    assert runner.invoke(cli.cli, auto_cmds[4][1:]).exit_code == 0
    assert cs.load_decision_streak_state(picks, require_contest_state=True, now=NOW).streak == 7


def test_revoke_preserves_real_fetch_status_path_and_executes_saver_template(
    tmp_path: Path, offline_env,
):
    """Rows 7 & 8: check contest_streak_fetch_status.json at picks.parent/health_state/,
    carry non-null switched_at across repeated revokes, and run printed saver template."""
    runner = CliRunner()
    picks = tmp_path / "space ' \" & picks"
    picks.mkdir()
    _put(picks, "season_contract.json", _sw(switched_at="2027-03-25T14:00:00Z"))
    auto_p = _put(picks, "contest_streak.json", _auto())
    ledger_p = _put(picks, "contest_ledger.jsonl", {"row": 1})
    fetch_status_p = picks.parent / "health_state/contest_streak_fetch_status.json"
    fetch_status_p.parent.mkdir(parents=True, exist_ok=True)
    fetch_status_p.write_bytes(b'{"last_success_at":"2027-07-15T14:00:00Z"}\n')
    before = {p: p.read_bytes() for p in (auto_p, ledger_p, fetch_status_p)}

    r1 = runner.invoke(
        cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "first", "--picks-dir", str(picks)],
    )
    assert r1.exit_code == 0, r1.output
    offline_env[0] = NOW + dt.timedelta(seconds=1)
    r2 = runner.invoke(
        cli.cli, ["season-contract", "revoke", "--season", "2027", "--reason", "second", "--picks-dir", str(picks)],
    )
    assert r2.exit_code == 0, r2.output

    snap = sc.switch_state(picks, offline_env[0]).snapshot
    assert snap["activation_id"] == ID
    assert snap["switched_at"] == "2027-03-25T14:00:00Z"
    for p, content in before.items():
        assert p.read_bytes() == content

    raw_cmds = re.findall(r"`(bts [^`]+)`", r2.output)
    saver_argv = shlex.split(raw_cmds[1].replace("<state read from MLB>", "used"))
    assert runner.invoke(cli.cli, saver_argv[1:]).exit_code == 0
    assert ss.load_saver_state(picks, season=2027).state == "used"
