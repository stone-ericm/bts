"""Offline regression tests for the whole-range operator paths."""
import hashlib
import json
import re
import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner

import bts.cli as cli
from bts import contest_fetch, contest_state, daily_decision
from bts.health import contest_state as health
from tests.test_day_outcome import _receipt
from tests.test_day_outcome_entry_sequence import _setup_committed_day
from tests.test_season_contract_cs4b1 import NOW, contract, fetch, files, offline, put, run, tmp_path
from tests.test_season_contract_cs4b2 import automatic, dated


def test_saver_flag_warning_command_repairs_live_state(tmp_path):
    picks = tmp_path / "operator's picks"
    put(picks, "season_contract.json", contract())
    put(picks, "contest_streak.json", automatic(active_streak=11))
    flag = put(picks, "saver_state.json", {"season": NOW.year, "state": "not_earned"})
    before = flag.read_bytes()
    warning, = [a for a in health.check(picks, expected=True, now=NOW)
                if "flag not_earned" in a.message]
    assert (warning.level, warning.source) == ("WARN", "saver_state")
    command = re.search(r"`(bts set-contest-streak [^`]+)`", warning.message)
    assert command, warning.message
    args = shlex.split(command[1])
    assert args[args.index("--picks-dir") + 1] == str(picks)
    args = [{"N": "0", "B": "0", "S": "not_earned"}.get(arg, arg) for arg in args]
    result = CliRunner().invoke(cli.cli, args[1:])
    assert result.exit_code == 0, result.output
    live = contest_state.load_decision_streak_state(picks, require_contest_state=True, now=NOW)
    assert (live.observation_kind, live.streak, flag.read_bytes()) == ("dated", 0, before)


@pytest.mark.parametrize("delivery", ["locked_unconfirmed", "delivered", "private_locked",
                                      "missing", "rejected", "directory"])
def test_unconfirmed_nags_verify_delivery_at_each_tier(tmp_path, monkeypatch, delivery):
    from bts.dm import send_dm
    from bts.entry_receipt import discover

    day = NOW.date().isoformat()
    picks = _setup_committed_day(tmp_path, day, game_time_utc=f"{day}T23:10:00Z")
    decision_path = picks / day / "decision.json"
    if delivery in {"missing", "rejected", "directory"}:
        decision_path.unlink()
        if delivery == "rejected":
            decision_path.write_text("[]")
        if delivery == "directory":
            decision_path.mkdir()
        from bts.picks import load_pick, save_pick
        daily = load_pick(day, picks)
        daily.notification_id = "confirmed-delivery"
        save_pick(daily, picks)
    else:
        decision = json.loads(decision_path.read_bytes())
        decision["delivery_status"] = delivery
        decision_path.write_text(json.dumps(decision))
    put(picks, "contest_streak.manual.json", dated(active_streak=46, best_streak=46))
    monkeypatch.setattr(contest_fetch, "fetch_pending_predictions", lambda *_: [])
    monkeypatch.setattr(cli, "_fetch_bts_to_mlb", lambda: {})
    marker_path = picks.parent / "health_state" / "pick_entry_check.json"
    for index, (at, minutes) in enumerate((("18:20", 45), ("18:40", 25), ("18:55", 10)), 1):
        result = run(picks, "check-pick-entered", "--now-et", f"{day}T{at}:00",
                     "--dm-recipient", "offline.test", "--expected-username", "offline")
        assert result.exit_code == 1, result.output
        assert send_dm.call_count == index
        body = send_dm.call_args.args[1]
        if delivery == "locked_unconfirmed":
            assert "Delivery of this pick was NOT confirmed" in body
            assert "Aaron Judge" in body and "the pick the scheduler tried to send" in body
            assert "DMs and MLB entry before entering anything" in body
            assert f"bts season-contract status --picks-dir {shlex.quote(str(picks))}" in body
            assert str(minutes) in body and "Fix it now!" not in body
        elif index == 1:
            assert body == "⚠️ BTS pick NOT entered in MLB app: Aaron Judge — first pitch 7:10 PM ET (45 min to submit). Fix it now!"
        else:
            assert body == f"⏰ STILL NOT entered (no_pick): Aaron Judge — {minutes} min left to submit!"
        marker = json.loads(marker_path.read_text())
        assert marker["date"] == day and marker["status"] == "alerted"
        assert marker["escalations"] == sorted(["initial", "t30", "t15"][:index])
        assert any(rec["attempt_id"] == marker["receipt"] for rec in discover(picks, day))


@pytest.mark.parametrize(("action", "runner_up", "names"), [
    ("single", {"batter_name": "Runner Up"}, "Selected"),
    ("single", ["malformed"], "Selected"),
    ("double", {"batter_name": "Second"}, "Selected + Second"),
    ("double", {"batter_name": 42}, "Selected"),
])
def test_single_outcome_names_only_the_executed_batter(tmp_path, monkeypatch, action, runner_up, names):
    from bts import entry_receipt, receipt_io
    from bts.dm import send_dm

    day = NOW.date().isoformat()
    daily_decision.write_decision(day, tmp_path, action=action, source="mdp",
                                  primary={"batter_name": "Selected"},
                                  double_down=runner_up if isinstance(runner_up, dict) else None,
                                  delivery_status="delivered", scoreable=True)
    path = tmp_path / day / "decision.json"
    if isinstance(runner_up, list):
        path.write_text(json.dumps(json.loads(path.read_text()) | {"double_down": runner_up}))
    before = path.read_bytes()
    cfg = tmp_path / "config.toml"
    cfg.write_text('[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n')
    monkeypatch.setattr("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 1}])
    for confirmed in (False, True):
        if confirmed:
            receipt = _receipt(date=day, sha=hashlib.sha256(before).hexdigest())
            receipt_io.publish(entry_receipt.receipts_root(tmp_path) / day / "bound.json",
                               json.dumps(receipt).encode())
        result = run(tmp_path, "day-outcome", "--date", day, "--config", str(cfg),
                     "--expected-username", "stonehengee", "--dm-recipient", "offline.test")
        assert result.exit_code == (0 if confirmed else 1), (result.output, result.exception)
        outcome = json.loads(result.stdout)
        assert outcome["state"] == ("entered" if confirmed else "entry_unverifiable")
        assert outcome["pick"] == names and path.read_bytes() == before
        assert send_dm.call_count == 1
        assert send_dm.call_args.args[1].endswith(f"Pick: {names}.")


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_status_exposes_automatic_refusal_masked_by_dated_observation(tmp_path, manual, ready):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json",
        automatic(schema_version="bts_contest_streak_auto_v2" if ready else "bts_contest_streak_auto_v1"))
    if manual:
        put(tmp_path, "contest_streak.manual.json", dated(active_streak=1))
    before = files(tmp_path)
    result = run(tmp_path, "season-contract", "status")
    assert result.exit_code == 0, result.output
    assert ("live decision now: accepted" in result.output) == (manual or ready)
    verdict = next((line for line in result.output.splitlines() if line.startswith("automatic path alone:")), "")
    if ready:
        assert verdict == "automatic path alone: accepted (streak 12, recorded 2027-07-15T16:00:00Z)"
    else:
        assert verdict.startswith("automatic path alone: refused: Live decision refused: schema_version")
    assert ("expires at midnight ET" in result.output) == (manual and not ready)
    assert files(tmp_path) == before


@pytest.mark.parametrize("body", [b"[]", b"null", b"\xff", b'{"last_alert_at_by_category":{"actionable":42}}'])
@pytest.mark.parametrize("failure", [False, True])
def test_fetch_repairs_bad_status_on_success_and_failure(tmp_path, monkeypatch, body, failure):
    from bts.dm import send_dm

    put(tmp_path, "season_contract.json", contract())
    auto = put(tmp_path, "contest_streak.json", automatic())
    before = auto.read_bytes()
    status = tmp_path.parent / "health_state" / "contest_streak_fetch_status.json"
    status.parent.mkdir()
    status.write_bytes(body)
    if failure:
        monkeypatch.setattr(contest_fetch, "fetch_profile", lambda *_: {
            "activeStreak": 7, "seasonBestStreak": 9, "predictions": {}})
    result = fetch(tmp_path)
    assert result.exit_code == (2 if failure else 0), (result.output, result.exception)
    payload = json.loads(status.read_bytes())
    ledger = auto.with_name("contest_ledger.jsonl")
    flag = auto.with_name("saver_state.json")
    if failure:
        send_dm.assert_called_once()
        assert "predictions must be a list" in payload["last_error"]
        assert auto.read_bytes() == before and not ledger.exists() and not flag.exists()
    else:
        send_dm.assert_not_called()
        assert payload["last_error"] is None and json.loads(ledger.read_text())["switched"]
        assert json.loads(flag.read_text())["state"] == "not_earned"


def test_b8a_requires_successful_fetch_automatic_readiness_and_saver_read():
    docs = Path(__file__).resolve().parents[1] / "docs/ops/2027-season-start.md"
    row = next(line for line in docs.read_text().splitlines() if line.startswith("| B8a |"))
    assert re.search(r"fetch-contest-streak` succeeds.*automatic path alone: accepted.*"
                     r"read `bts saver-state`.*Initialize only if it reports `uninitialized`", row)
