"""Clock, file and restart boundaries for the whole-range fixes."""
import ast
import json
import os
import re
import shlex
import signal
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from click.testing import CliRunner

import bts.cli as cli
from bts import contest_fetch, saver_state, season_contract
from bts.health import alert
from bts.health import contest_state as health
from bts.leaderboard import auth
from tests.test_day_outcome_entry_sequence import _setup_committed_day
from tests.test_season_contract_cs4b1 import (
    ET, NOW, REAL_DATETIME, contract, fetch, files, offline, pin, put, run, tmp_path,
)
from tests.test_season_contract_cs4b2 import automatic, dated


@contextmanager
def deadline():
    def timed_out(*_):
        raise SystemExit("operation blocked on special file")

    old = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(5)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def test_dated_repair_templates_cover_operator_messages(tmp_path):
    root = Path(__file__).resolve().parents[1] / "src/bts"
    actual = {}

    def assemble(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(map(assemble, node.values))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return assemble(node.left) + assemble(node.right)
        return "<value>"

    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(parents.get(node), (ast.JoinedStr, ast.BinOp)):
                continue
            template = assemble(node)
            if "bts set-contest-streak" not in template:
                continue
            assert "bts set-contest-streak --date" in template, (path, node.lineno, template)
            name = path.relative_to(root).as_posix()
            actual.setdefault(name, []).append(template)
            assert "--streak" in template and "--best-streak" in template
            if name != "health/contest_state.py":
                assert "--saver" in template

    assert actual.keys() >= {"health/contest_state.py", "scheduler.py", "season_contract.py"}
    put(tmp_path, "contest_streak.manual.json", dated(active_streak=11, saver_state="not_earned"))
    warning, = [item for item in health.check(tmp_path, expected=True, now=NOW) if item.source == "saver_state"]
    assert warning.level == "WARN"
    commands = re.findall(r"`(bts set-contest-streak [^`]+)`", warning.message)
    assert [shlex.split(command)[9] for command in commands] == ["active", "used", "not_earned"]
    for command in [*commands, season_contract.dated_template(tmp_path, NOW)]:
        fields = shlex.split(command)
        assert fields[2::2] == ["--date", "--streak", "--best-streak", "--saver", "--picks-dir"]
        assert fields[-1] == str(tmp_path)


def test_warning_uses_check_clock_and_keeps_health_delivery_identity(tmp_path, monkeypatch):
    before = REAL_DATETIME(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    after = before + timedelta(seconds=2)
    picks = tmp_path / "operator's picks"
    put(picks, "contest_streak.json", automatic(active_streak=11, source_date="2027-09-26"))
    flag = put(picks, "saver_state.json", {"season": before.year, "state": "not_earned"})
    warning, = [a for a in health.check(picks, now=before) if a.source == "saver_state"]
    assert "--date 2027-12-31" in warning.message
    cmd = re.search(r"`(bts set-contest-streak [^`]+)`", warning.message)[1]
    pin(monkeypatch, after)
    args = [{"N": "0", "B": "0", "S": "not_earned"}.get(arg, arg) for arg in shlex.split(cmd)[1:]]
    assert CliRunner().invoke(cli.cli, args).exit_code == 2
    flag.write_text(json.dumps({"season": after.year, "state": "not_earned"}))
    new_warning, = [a for a in health.check(picks, now=after) if a.source == "saver_state"]
    assert "--date 2028-01-01" in new_warning.message
    args[args.index("--date") + 1] = after.date().isoformat()
    assert CliRunner().invoke(cli.cli, args).exit_code == 0

    send = Mock(side_effect=[KeyboardInterrupt, TimeoutError("lost reply"), "sent", "sent"])
    monkeypatch.setattr(alert, "send_dm", send)
    dispatch = alert.dispatch_dm_for_health_alerts
    status = tmp_path / "health.json"
    kwargs = dict(warn_attention=[warning], status_path=status, now_et_date=before.date())
    with pytest.raises(KeyboardInterrupt):
        dispatch([], "offline.test", **kwargs)
    assert not status.exists()
    dispatch([], "offline.test", **kwargs)
    assert json.loads(status.read_text())["status"] == "failed"
    dispatch([], "offline.test", **kwargs)
    saved = json.loads(status.read_text())
    assert saved["sent_day"] == before.date().isoformat()
    assert saved["sent_sources"] == ["WARN_ATTENTION:saver_state"]
    assert not dispatch([], "offline.test", **kwargs)
    dispatch([alert.Alert("CRITICAL", "other", "Synthetic incident")], "offline.test", **kwargs)
    assert json.loads(status.read_text())["sent_sources"] == ["CRITICAL:other", "WARN_ATTENTION:saver_state"]


@pytest.mark.parametrize(("filename", "data", "reason"), [
    ("contest_streak.manual.json", {"active_streak": 5, "override_expires_at": "2090-01-01T00:00:00Z"},
     "legacy manual override"),
    ("contest_streak.manual.json", None, "expected object"),
    ("contest_streak.json", [], "expected object"),
    ("season_contract.json", [], "corrupt season switch"),
    ("contest_streak.json", automatic(activation_id="b" * 32), "not the switch's"),
])
def test_status_keeps_authority_refusals(tmp_path, filename, data, reason):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic())
    put(tmp_path, filename, data)
    before = files(tmp_path)
    result = run(tmp_path, "season-contract", "status")
    assert result.exit_code == int(filename == "season_contract.json"), result.output
    verdict, = [line for line in result.output.splitlines() if line.startswith("automatic path alone:")]
    assert verdict.startswith("automatic path alone: refused:") and reason in verdict
    assert "expires at midnight ET" not in result.output and files(tmp_path) == before


@pytest.mark.parametrize(("name", "accepted"), [
    ("season_contract.json", False), ("contest_streak.json", False),
    ("contest_streak.manual.json", False), ("saver_state.json", True),
    ("contest_ledger.jsonl", True), ("2027-07-15.json", True), ("streak.json", True),
])
def test_status_never_blocks_on_special_files(tmp_path, name, accepted):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic())
    base = tmp_path if name in {"streak.json", "2027-07-15.json"} else tmp_path / "account_state"
    path = base / name
    path.unlink(missing_ok=True)
    os.mkfifo(path)
    before = files(tmp_path)
    with deadline():
        result = run(tmp_path, "season-contract", "status")
    assert result.exit_code == int(name == "season_contract.json"), result.output
    assert ("automatic path alone: accepted" in result.output) == accepted
    if not accepted or name in {"saver_state.json", "contest_ledger.jsonl"}:
        assert "not a regular file" in result.output
    assert files(tmp_path) == before


@pytest.mark.parametrize("recorded", [None, "", "2027-12-31T12:00:00", "2027-10-01T12:00:00Z",
                                     "2028-01-02T10:00:00Z"])
def test_status_uses_one_et_snapshot_across_new_year(tmp_path, monkeypatch, recorded):
    entry = REAL_DATETIME(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    next_year = entry + timedelta(seconds=2)
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic(recorded_at=recorded))
    put(tmp_path, "contest_streak.manual.json", dated("2027-12-31"))
    original = season_contract.read_regular_file_text
    calls = []

    def cross(path):
        calls.append(path.name)
        pin(monkeypatch, next_year)
        return original(path)

    pin(monkeypatch, entry)
    monkeypatch.setattr(season_contract, "read_regular_file_text", cross)
    before = files(tmp_path)
    result = run(tmp_path, "season-contract", "status", "--season", "2026")
    assert "inspected season: 2026" in result.output and "in effect=True" in result.output
    expected = recorded if recorded and recorded.endswith("Z") else "unavailable"
    assert f"automatic path alone: accepted (streak 12, recorded {expected})" in result.output
    assert calls.count("contest_streak.manual.json") == 2  # inspection and live selection only
    assert calls.count("season_contract.json") == 0  # captured before diagnostics
    result = run(tmp_path, "season-contract", "status", "--season", "2027")
    assert "in effect=False" in result.output and "switch: automatic" in result.output
    assert "automatic path alone: refused: Live decision refused: season 2028 switch is other_season" in result.output
    assert "expires at midnight" not in result.output and files(tmp_path) == before


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("section", ["contest_streak.manual.json", "saver_state.json",
                                     "contest_ledger.jsonl", "contest_streak.json"])
def test_status_restart_after_interrupted_read(tmp_path, monkeypatch, interrupt, section):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic())
    put(tmp_path, "contest_streak.manual.json", dated())
    put(tmp_path, "saver_state.json", {})
    put(tmp_path, "contest_ledger.jsonl", {})
    before = files(tmp_path)
    original = season_contract.read_regular_file_text

    def interrupted(path):
        if path.name == section:
            raise interrupt(4)
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(season_contract, "read_regular_file_text", interrupted)
        result = run(tmp_path, "season-contract", "status")
        assert result.exit_code and "automatic path alone:" not in result.output
    assert files(tmp_path) == before
    assert "automatic path alone: accepted" in run(tmp_path, "season-contract", "status").output


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_fetch_replaces_special_status_or_diagnoses_path(tmp_path, monkeypatch, failure, kind):
    from bts.dm import send_dm

    put(tmp_path, "season_contract.json", contract())
    auto = put(tmp_path, "contest_streak.json", automatic())
    prior = auto.read_bytes()
    status = tmp_path.parent / "health_state/contest_streak_fetch_status.json"
    status.parent.mkdir()
    if kind == "fifo":
        os.mkfifo(status)
    else:
        status.mkdir()
    if failure:
        monkeypatch.setattr(contest_fetch, "fetch_profile", Mock(side_effect=httpx.ConnectError("offline")))
    with deadline():
        result = fetch(tmp_path)
    assert result.exit_code == (2 if failure or kind == "directory" else 0), (result.output, result.exception)
    assert (auto.read_bytes() == prior) == failure
    assert send_dm.call_count == int(failure)
    if kind == "directory":
        assert str(status) in result.output and "cannot replace" in result.output
        assert status.is_dir() and not auto.with_name("contest_ledger.jsonl").exists()
    else:
        assert status.is_file()
        assert (json.loads(status.read_text())["last_error"] is not None) == failure


@pytest.mark.parametrize("checkpoint", ["observation", "status", "ledger"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_fetch_restart_after_success_checkpoint(tmp_path, monkeypatch, checkpoint, interrupt):
    put(tmp_path, "season_contract.json", contract())
    auto = tmp_path / "account_state/contest_streak.json"
    status = tmp_path.parent / "health_state/contest_streak_fetch_status.json"
    ledger = auto.with_name("contest_ledger.jsonl")
    flag = auto.with_name("saver_state.json")
    write = cli._atomic_write_json

    def stop_after_write(path, data):
        write(path, data)
        if path == {"observation": auto, "status": status}.get(checkpoint):
            raise interrupt(4)

    with monkeypatch.context() as patch:
        patch.setattr(cli, "_atomic_write_json", stop_after_write)
        if checkpoint == "ledger":
            patch.setattr(saver_state, "maybe_auto_earn_saver", Mock(side_effect=interrupt(4)))
        assert fetch(tmp_path).exit_code != 0
    assert json.loads(auto.read_text())["active_streak"] == 7
    assert status.exists() == (checkpoint != "observation")
    assert ledger.exists() == (checkpoint == "ledger") and not flag.exists()
    assert fetch(tmp_path).exit_code == 0
    assert len(ledger.read_text().splitlines()) == (2 if checkpoint == "ledger" else 1)
    assert json.loads(flag.read_text())["state"] == "not_earned"


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_fetch_retries_failure_dm_until_checkpoint(tmp_path, monkeypatch, interrupt):
    from bts.dm import send_dm

    auto = put(tmp_path, "contest_streak.json", automatic())
    before = auto.read_bytes()
    monkeypatch.setattr(contest_fetch, "fetch_profile", Mock(side_effect=httpx.ConnectError("offline")))
    status = tmp_path.parent / "health_state/contest_streak_fetch_status.json"
    with monkeypatch.context() as patch:
        patch.setattr(cli, "_atomic_write_json", Mock(side_effect=interrupt(4)))
        assert fetch(tmp_path).exit_code != 2
    assert send_dm.call_count == 1 and not status.exists()
    for _ in range(2):
        assert fetch(tmp_path).exit_code == 2
    assert send_dm.call_count == 2
    assert auto.read_bytes() == before and not auto.with_name("contest_ledger.jsonl").exists()


def test_fetch_unconfirmed_dm_does_not_start_cooldown(tmp_path, monkeypatch):
    from bts.dm import send_dm

    monkeypatch.setattr(auth, "fetch_login_session", Mock(side_effect=auth.AuthError("offline")))
    send_dm.side_effect = [None, "confirmed"]
    status = tmp_path.parent / "health_state" / "contest_streak_fetch_status.json"
    assert fetch(tmp_path).exit_code == 2
    assert json.loads(status.read_text())["last_alert_at_by_category"] == {}
    assert fetch(tmp_path).exit_code == 2
    assert set(json.loads(status.read_text())["last_alert_at_by_category"]) == {"actionable"}
    assert fetch(tmp_path).exit_code == 2 and send_dm.call_count == 2


@pytest.mark.parametrize(("error", "category"), [
    (auth.AuthError("offline"), "actionable"),
    (auth.TransientAuthError("offline"), "transient"),
    (auth.RateLimitedLoginError("offline"), "rate_limited"),
])
def test_fetch_category_map_ignores_only_invalid_stamps(tmp_path, monkeypatch, error, category):
    from bts.dm import send_dm

    stamp = NOW.isoformat()
    status = tmp_path.parent / "health_state" / "contest_streak_fetch_status.json"
    status.parent.mkdir()
    status.write_text(json.dumps({"last_alert_at": stamp, "last_alert_at_by_category": {
        "other": stamp, category: "2027-07-15T12:00:00",
    }}))
    monkeypatch.setattr(auth, "fetch_login_session", Mock(side_effect=error))
    for _ in range(2):
        assert fetch(tmp_path, dry=True).exit_code == 2
    send_dm.assert_called_once()
    payload = json.loads(status.read_text())
    assert payload["last_error_category"] == category
    assert payload["last_alert_at_by_category"]["other"] == stamp
    assert payload["last_alert_at_by_category"][category].endswith("Z")
    assert not (tmp_path / "account_state").exists()


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, TimeoutError])
def test_unconfirmed_entry_restart_binds_original_et_day(tmp_path, monkeypatch, failure):
    from bts.dm import send_dm
    from bts.entry_receipt import discover

    start = REAL_DATETIME(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    later = start + timedelta(seconds=2)
    day = start.date().isoformat()
    picks = _setup_committed_day(tmp_path, day, game_time_utc="2028-01-01T05:45:00Z")
    decision = picks / day / "decision.json"
    decision.write_text(json.dumps(json.loads(decision.read_text()) | {"delivery_status": "locked_unconfirmed"}))
    pending = [{"roundId": 1, "playerId": 10}]
    crosswalk = {10: 42}

    def cookies():
        pin(monkeypatch, later)
        return {}

    monkeypatch.setattr(auth, "load_session_cookies", cookies)
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: start.date()})
    monkeypatch.setattr(cli, "_fetch_bts_to_mlb", lambda: crosswalk)
    monkeypatch.setattr(contest_fetch, "fetch_pending_predictions", lambda *_: pending)
    args = ("check-pick-entered", "--dm-recipient", "offline.test", "--expected-username", "offline")
    marker = picks.parent / "health_state" / "pick_entry_check.json"
    pin(monkeypatch, start)
    send_dm.side_effect = failure("lost")
    assert run(picks, *args).exit_code != 0
    if marker.exists():
        assert json.loads(marker.read_text())["status"] == "dm_failed"

    pin(monkeypatch, start)
    send_dm.side_effect = None
    assert run(picks, *args).exit_code == 1
    receipt = json.loads(marker.read_text())
    assert receipt["date"] == day and receipt["escalations"] == ["initial"]
    assert "The MLB entry differs" in send_dm.call_args.args[1]
    assert any(row["attempt_id"] == receipt["receipt"] for row in discover(picks, day))
    assert not discover(picks, later.date().isoformat())

    pin(monkeypatch, start)
    crosswalk[10] = 592450
    send_dm.side_effect = TimeoutError("lost all-clear")
    result = run(picks, *args)
    assert result.exit_code == 0 and "confirmation DM failed" in result.output
    assert send_dm.call_args.args[1] == f"✅ BTS pick entry confirmed for {day} (match)."
    assert json.loads(marker.read_text())["status"] == "confirmed"
    send_dm.reset_mock()
    pin(monkeypatch, start)
    assert "already confirmed" in run(picks, *args).output
    send_dm.assert_not_called()


@pytest.mark.parametrize("settled", [False, True])
def test_cutover_rehearses_masked_and_matching_authority(tmp_path, monkeypatch, settled):
    from bts.dm import send_dm

    put(tmp_path, "contest_streak.json", automatic(schema_version="bts_contest_streak_auto_v1"))
    put(tmp_path, "contest_streak.manual.json", dated(active_streak=1))
    assert run(tmp_path, "season-contract", "enable", "--season", str(NOW.year)).exit_code == 0
    with monkeypatch.context() as patch:
        patch.setattr(contest_fetch, "fetch_profile", Mock(side_effect=httpx.ConnectError("offline")))
        assert fetch(tmp_path).exit_code == 2
    result = run(tmp_path, "season-contract", "status")
    assert "live decision now: accepted" in result.output
    assert "automatic path alone: refused" in result.output and "expires at midnight ET" in result.output
    if not settled:
        monkeypatch.setattr(contest_fetch, "fetch_profile", lambda *_: {
            "activeStreak": 7, "seasonBestStreak": 9, "predictions": [],
        })
    assert fetch(tmp_path).exit_code == 0
    result = run(tmp_path, "season-contract", "status")
    assert "automatic path alone: accepted" in result.output and "expires at midnight ET" not in result.output
    saver = run(tmp_path, "saver-state", "--season", str(NOW.year))
    assert saver.exit_code == 0
    assert f"saver_state: {'not_earned' if settled else 'uninitialized'}" in saver.output
    if not settled:
        assert run(tmp_path, "saver-state", "--init", "not_earned", "--season", str(NOW.year)).exit_code == 0
    assert "saver_state: not_earned" in run(tmp_path, "saver-state", "--season", str(NOW.year)).output
    send_dm.assert_called_once()


def test_crash_then_corrected_real_scheduler_replay(tmp_path, monkeypatch):
    import datetime as calendar

    # This replay has its own clock and transport guard.
    monkeypatch.setattr(calendar, "datetime", REAL_DATETIME)
    review = Path(__file__).resolve().parents[1] / "docs/reviews/2026-11-18-whole-range/slice-1-codex"
    monkeypatch.syspath_prepend(str(review))
    import opening_edges
    import opening_rehearsal

    season = opening_rehearsal.Season(tmp_path)
    put(season.picks, "contest_streak.json", automatic(
        schema_version="bts_contest_streak_auto_v1", active_streak=17, best_streak=26,
        source_date=opening_rehearsal.OLD_DAY, recorded_at="2026-10-21T14:00:00Z",
    ))

    def verify(world):
        day = opening_rehearsal.DAYS[1]
        for at in ("15:29", "15:40", "15:52"):
            world.command("check-pick-entered", "--expected-username", opening_rehearsal.ACCOUNT,
                          "--dm-recipient", opening_rehearsal.DM, "--now-et", f"{day}T{at}:00",
                          expected=1)

    monkeypatch.setattr(opening_edges, "verify_unconfirmed_entry", verify)
    try:
        with season.run_offline():
            season.restore()
            evidence = opening_edges.restart_at_send(season, "corrected_after_marker")
    except AssertionError:
        print(season.journal.getvalue())
        raise
    nags = [entry["body"] for entry in evidence["messages"]
            if "min to submit" in entry["body"] or "min left to submit" in entry["body"]]
    assert len(nags) == 3
    for minutes, nag in zip((36, 25, 13), nags, strict=True):
        assert "Delivery of this pick was NOT confirmed" in nag
        assert f"({minutes} min to submit)" in nag
        assert "Batter 27250 + DD Batter 27251" in nag
        assert "the pick the scheduler tried to send" in nag
        assert f"bts season-contract status --picks-dir {shlex.quote(str(season.picks))}" in nag
        assert "DMs and MLB entry before entering anything" in nag
    assert evidence["snapshot"]["decision"]["delivery_status"] == "locked_unconfirmed"
    assert evidence["next_morning"]["state"] == "not_entered"


@pytest.mark.parametrize("failure", [TimeoutError, KeyboardInterrupt, SystemExit])
def test_day_outcome_restart_uses_taken_date_and_retries_dm(tmp_path, monkeypatch, failure):
    from bts.dm import send_dm
    from bts.picks import load_pick

    entry = REAL_DATETIME(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    later = entry + timedelta(seconds=2)
    day = entry.date().isoformat()
    picks = _setup_committed_day(tmp_path, day)
    assert load_pick(day, picks) is not None
    cfg = tmp_path / "config.toml"
    cfg.write_text('[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n')
    checks = []

    def schedule(at):
        checks.append(at)
        pin(monkeypatch, later)
        return [{"gamePk": 1}]

    monkeypatch.setattr("bts.scheduler.fetch_schedule", schedule)
    args = ("day-outcome", "--config", str(cfg), "--dm-recipient", "offline.test")
    before = files(tmp_path)
    pin(monkeypatch, entry)
    send_dm.side_effect = failure(4)
    first = run(picks, *args)
    assert first.exit_code == (4 if failure is SystemExit else 1)
    assert json.loads(first.stdout)["date"] == day
    if failure is TimeoutError:
        assert "DM failed" in first.stderr
    assert files(tmp_path) == before

    send_dm.side_effect = None
    result = run(picks, *args, "--date", day)
    assert result.exit_code == 1 and json.loads(result.stdout)["pick"] == "Aaron Judge"
    assert checks == [day, day] and send_dm.call_count == 2
    assert files(tmp_path) == before
