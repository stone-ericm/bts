"""Pytest regressions for F5 early-gate boundaries where 10:00 ET is the sole base slot (10:06..10:20 ET)."""
from __future__ import annotations

import gzip
import json
import shlex
from datetime import date, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import entry_receipt, picks
from bts.cli import cli
from bts.daily_decision import write_decision
from tests.test_entry_intent import ENTRY_CMD, box
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row

DAY = "2027-03-24"
ET = ZoneInfo("America/New_York")
ENTER = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'


def _installed_args(box_fixture):
    box_fixture.config.write_text(ENTER)
    res = box_fixture.run("install")
    assert res.returncode == 0, res.stderr
    line, = [ln for ln in box_fixture.lines() if ENTRY_CMD in ln and not ln.startswith("#")]
    tokens = shlex.split(line, comments=True)
    return tokens[tokens.index("check-pick-entered"):tokens.index(">>")]


@pytest.mark.parametrize("pitch_hm", ["10:06", "10:15", "10:20"])
@pytest.mark.parametrize("wrong_game", [True, False], ids=["wrong-game", "removed-entry"])
def test_sole_1000_base_slot_defers_early_runs_and_preserves_1000_warning(tmp_path, monkeypatch, box, pitch_hm, wrong_game):
    """Kills the `range(15, 14 * 60, 15)` (start at 10:15) mutation: for 10:06..10:20 pitches, 10:00 is the ONLY
    10:00-23:45 slot in (cutoff, window], so 09:00..09:45 must still return base_slot_available."""
    args = _installed_args(box)
    pdir = tmp_path / "picks"
    pdir.mkdir()
    args[args.index("--picks-dir") + 1] = str(pdir)
    uname = args[args.index("--expected-username") + 1]
    recip = args[args.index("--dm-recipient") + 1]
    cfg = tmp_path / "config.toml"
    cfg.write_text(ENTER)
    marker = tmp_path / "health_state" / "pick_entry_check.json"
    udir = entry_receipt.units_snapshot_dir(pdir)
    udir.mkdir(parents=True)

    slot = picks.Pick("Batter 1010", 1010, "NYY", 1, "Pitcher", 500, 0.80, [], False, 101, f"{DAY}T{pitch_hm}:00-04:00")
    commit_iso = f"{DAY}T08:40:00-04:00"
    picks.save_pick(
        picks.DailyPick(DAY, commit_iso, slot, None, None, notification_sent=True,
                        notification_channel="bluesky_dm", notification_id="m1",
                        delivery_attempted=True, delivered_at=commit_iso),
        pdir,
    )
    write_decision(DAY, pdir, action="single", source="mdp", delivery_status="delivered", scoreable=True,
                   primary={"batter_name": "Batter 1010", "batter_id": 1010, "team": "NYY", "game_pk": 101, "p_game_hit": 0.80},
                   now=datetime.fromisoformat(commit_iso))

    if not wrong_game:
        (udir / f"{DAY}T030000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [RIGHT]}).encode(), mtime=0))

    calls, dms = [], []
    monkeypatch.setattr("bts.leaderboard.auth.load_session_cookies", lambda: {})
    monkeypatch.setattr("bts.leaderboard.auth.extract_uid", lambda _: "u1")
    monkeypatch.setattr("bts.leaderboard.auth.fetch_login_session",
                        lambda *a, **k: calls.append("session") or SimpleNamespace(user_id="u1", username=uname, xsid="x1"))
    monkeypatch.setattr("bts.contest_fetch.fetch_profile", lambda *a, **k: calls.append("profile") or {"predictions": []})
    monkeypatch.setattr("bts.cli._fetch_rounds", lambda: {1: date.fromisoformat(DAY)})
    monkeypatch.setattr("bts.cli._fetch_bts_to_mlb", lambda: {1: 1010})
    monkeypatch.setattr("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}])
    monkeypatch.setattr("bts.dm.send_dm", lambda _, msg: dms.append(msg))
    monkeypatch.setattr("bts.picks._git_head_sha", lambda *a, **k: "fixed-head")
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)

    runner = CliRunner()
    for early in ("09:00", "09:15", "09:30", "09:45"):
        calls.clear()
        dms.clear()
        before_ids = {r["attempt_id"] for r in entry_receipt.discover(pdir, DAY)}
        monkeypatch.setattr("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: [row()])
        res = runner.invoke(cli, [*args, "--now-et", f"{DAY}T{early}:00"])
        assert res.exit_code == 0
        rec, = [r for r in entry_receipt.discover(pdir, DAY) if r["attempt_id"] not in before_ids]
        assert rec["outcome"] == "base_slot_available"
        assert calls == [] and dms == [] and not marker.exists()

    if wrong_game:
        (udir / f"{DAY}T140000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [WRONG]}).encode(), mtime=0))

    calls.clear()
    dms.clear()
    monkeypatch.setattr("bts.contest_fetch.fetch_pending_predictions",
                        lambda *a, **k: [row()] if wrong_game else [])
    res_1000 = runner.invoke(cli, [*args, "--now-et", f"{DAY}T10:00:00"])
    assert res_1000.exit_code == 1
    assert "session" in calls and len(dms) == 1
    expected_dm = "the entry is recorded in game 102" if wrong_game else "BTS pick NOT entered"
    assert expected_dm in dms[0]

    dms.clear()
    res_morning = runner.invoke(cli, ["day-outcome", "--date", DAY, "--config", str(cfg),
                                      "--picks-dir", str(pdir), "--expected-username", uname, "--dm-recipient", recip])
    assert res_morning.exit_code == 1 and len(dms) == 1
    assert ("ENTRY UNVERIFIABLE" if wrong_game else "NOT ENTERED") in dms[0]
