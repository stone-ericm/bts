"""Recovery from the durable recommendation-attempt marker."""

import hashlib
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from bts import dm, picks, scheduler, strategy
from bts.daily_decision import decision_path, load_decision, write_decision
from bts.health import alert, post_failure, runner
from tests.test_pick_chosen_on_cs5a import DAY, daily_pick, record, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


@pytest.fixture(autouse=True)
def restore_policy_cache(monkeypatch):
    monkeypatch.setattr(strategy, "_mdp_cache", None)


def own(directory, day=DAY):
    return directory.parent / "health_state" / "delivery_outcome_unknown" / f"{day}.json"


def shared(directory):
    return directory.parent / "health_state" / "health_dm_delivery_status.json"


def direct(config, directory, daily=None):
    if daily is None:
        daily = daily_pick()
        daily.delivery_attempted = True
    picks.save_pick(daily, directory)
    return scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "C2")


def copy(day):
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


@pytest.mark.parametrize("phase", ["before_remote_accept", "after_remote_accept", "corrected_after_marker"])
def test_restart_pages_unknown_delivery_without_resending(tmp_path, monkeypatch, phase):
    review = Path(__file__).resolve().parents[1] / "docs/reviews/2026-11-18-whole-range/slice-1-codex"
    monkeypatch.syspath_prepend(str(review))
    import opening_edges as edges
    import opening_rehearsal as opening

    season = opening.Season(tmp_path)
    opening.write_json(season.account_dir / "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v1", "season": 2026,
        "active_streak": 17, "best_streak": 26, "saver_available": False,
        "source_date": opening.OLD_DAY, "recorded_at": "2026-10-21T14:00:00Z",
        "source": "manual_cli",
    })
    finish = edges.finish

    def restart_then_finish(season, day, outcome_exit):
        path = decision_path(day, season.picks)
        original = path.read_bytes()
        season.command("schedule", "--config", season.config, "--date", day)
        assert path.read_bytes() == original
        return finish(season, day, outcome_exit)

    monkeypatch.setattr(edges, "finish", restart_then_finish)
    with season.run_offline():
        season.restore()
        evidence = edges.restart_at_send(season, phase)
        day = evidence["snapshot"]["day"]
    assert len(edges.picks_sent(season)) == int(phase == "after_remote_accept")
    assert evidence["snapshot"]["scheduler"]["pick_locked"]
    assert evidence["snapshot"]["decision"]["scoreable"]
    assert evidence["snapshot"]["decision"]["delivery_status"] == "locked_unconfirmed"
    assert evidence["next_morning"]["state"] == "not_entered"
    assert any("Delivery of this pick was NOT confirmed" in msg["body"] for msg in season.messages)
    assert [msg["body"] for msg in season.messages if "DELIVERY OUTCOME UNKNOWN" in msg["body"]] == [copy(day)]


def test_direct_unconfirmed_uses_own_receipt_after_commit(world):
    config, directory, _, _ = world
    shared(directory).parent.mkdir()
    shared(directory).write_bytes(b"[]")
    checkpoints = []

    def send(_recipient, _body):
        checkpoints.append((scheduler.load_state(DAY, directory), load_decision(DAY, directory)))
        return "ack"

    alert.send_dm.side_effect = send
    assert direct(config, directory) is False
    alert.send_dm.assert_called_once_with("offline.test", copy(DAY))
    (state, decision), = checkpoints
    assert state.pick_locked and state.committed_pick_written
    assert decision["delivery_status"] == "locked_unconfirmed"
    receipt = json.loads(own(directory).read_bytes())
    assert receipt["status"] == "sent" and receipt["message_id"] == "ack"
    assert shared(directory).read_bytes() == b"[]"
    dm.send_dm.assert_not_called()


@pytest.mark.parametrize("prior", [None, "locked_unconfirmed", "delivered", "terminal", "write_none"])
def test_classified_restart_preserves_existing_commit_and_terminal_result(world, monkeypatch, prior):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    if prior == "terminal":
        daily.result = "hit"
    picks.save_pick(daily, directory)
    path = decision_path(DAY, directory)
    if prior in {"locked_unconfirmed", "delivered", "terminal"}:
        write_decision(DAY, directory, action="single", source="mdp",
                       delivery_status=prior if prior != "terminal" else "locked_unconfirmed",
                       primary={"batter_id": daily.pick.batter_id, "batter_name": daily.pick.batter_name},
                       scoreable=True)
    writer = Mock(return_value=None) if prior == "write_none" else None
    if writer is not None:
        monkeypatch.setattr("bts.daily_decision.write_decision", writer)
    original = path.read_bytes() if path.exists() else None
    checkpoints = []

    def send(_recipient, body):
        if "DELIVERY OUTCOME UNKNOWN" in body:
            checkpoints.append((
                scheduler.load_state(DAY, directory),
                path.read_bytes() if path.exists() else None,
                writer.call_count if writer is not None else None,
            ))
        return "ack"

    alert.send_dm.side_effect = send
    scheduler.run_day(DAY, config)
    (state, decision_bytes, write_attempts), = checkpoints
    assert state.pick_locked
    if original is not None:
        assert decision_bytes == path.read_bytes() == original and state.committed_pick_written
    elif prior == "write_none":
        assert decision_bytes is None and not state.committed_pick_written and write_attempts == 1
    else:
        assert json.loads(decision_bytes)["delivery_status"] == "locked_unconfirmed"
        assert state.committed_pick_written
    receipt = json.loads(own(directory).read_bytes())
    assert receipt["status"] == "sent" and receipt["message_id"] == "ack"
    bodies = [call.args[1] for call in alert.send_dm.call_args_list]
    assert [body for body in bodies if "DELIVERY OUTCOME UNKNOWN" in body] == [copy(DAY)]
    assert any("[missed_pick]" in body for body in bodies) is (prior == "write_none")
    assert scheduler.run_result_polling.called is (prior not in {"terminal", "write_none"})
    dm.send_dm.assert_not_called()


@pytest.mark.parametrize("reply", [None, "", " \t ", 17, RuntimeError("offline"), " ack "])
def test_only_unknown_alert_requires_message_id(world, reply):
    config, directory, _, _ = world
    alert.send_dm.side_effect = [reply, "retry-ack"]
    assert direct(config, directory) is False
    acknowledged = isinstance(reply, str) and bool(reply.strip())
    checkpoint = json.loads(own(directory).read_bytes())
    if acknowledged:
        assert checkpoint["status"] == "sent" and checkpoint["message_id"] == reply
        assert checkpoint["sent_day"] == DAY
        assert checkpoint["sent_sources"] == ["CRITICAL:late_delivery"]
    else:
        assert checkpoint["status"] == "failed"
        assert checkpoint["error"] == ("offline" if isinstance(reply, Exception)
                                      else "send_dm returned no usable message id")
        assert not {"message_id", "sent_day", "sent_sources"} & checkpoint.keys()
    assert direct(config, directory) is False
    assert [call.args[1] for call in alert.send_dm.call_args_list] == [copy(DAY)] * (1 if acknowledged else 2)


@pytest.mark.parametrize("case,source", [
    ("eod", "pick_delivery"), ("4c", "contest_state"),
    ("6b-1", "contest_state"), ("cutoff", "late_delivery"), ("void", "postponed_pick"),
    ("no_recipient", "contest_state"), ("duplicate", "contest_state"),
    ("exception", "contest_state"), ("missing_id", "contest_state"),
])
def test_legacy_health_dispatch_matches_base(world, tmp_path, caplog, case, source):
    _, _, now, _ = world
    recipient = None if case == "no_recipient" else "offline.test"
    warn = case == "4c"
    incident = {"4c": "contest_state:stale_reading", "6b-1": "contest_state:morning"}.get(case)
    key = f"{'WARN_ATTENTION' if warn else 'CRITICAL'}:{incident or source}"
    headline = "BTS health WARN attention:" if warn else "BTS health CRITICAL alert(s):"
    body = f"{headline}\n- [{source}] offline fixture"
    fixture_alert = alert.Alert("WARN" if warn else "CRITICAL", source, "offline fixture", incident)
    expected = {
        "schema_version": "bts_health_dm_delivery_status_v1",
        "status": "sent", "updated_at": now[0].astimezone(ZoneInfo("UTC")).isoformat(),
        "recipient_configured": bool(recipient), "recipient": recipient,
        "critical_count": int(not warn), "warn_attention_count": int(warn),
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "body_first_line": headline,
        "sent_day": DAY, "sent_sources": [key],
    }
    logs = [f"sent health DM to offline.test ({int(not warn)} CRITICAL, {int(warn)} WARN attention)"]
    if case in {"no_recipient", "exception"}:
        del expected["sent_day"], expected["sent_sources"]
        if case == "no_recipient":
            expected.update(status="skipped_no_recipient", error="dm_recipient is not configured")
            logs = ["[health_dm_delivery CRITICAL] health alert DM skipped: no recipient"]
        else:
            expected.update(status="failed", error="offline")
            logs = ["send_dm failed for health alerts; CRITICAL/WARN visibility may be lost: offline",
                    "[health_dm_delivery CRITICAL] health alert DM delivery failed"]
    if case == "duplicate":
        logs = [f"health DM suppressed: {[key]} already covered by today's DM (day={DAY})"]
    frozen_bytes = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()
    for name, opt in [("omitted", {}), ("false", {"require_message_id": False})]:
        path = tmp_path / f"{name}.json"
        if case == "duplicate":
            path.write_bytes(frozen_bytes)
        alert.send_dm.reset_mock()
        alert.send_dm.side_effect = OSError("offline") if case == "exception" else None
        alert.send_dm.return_value = None if case == "missing_id" else "legacy-id"
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=alert.__name__):
            attempted = alert.dispatch_dm_for_health_alerts(
                [] if warn else [fixture_alert], recipient, [fixture_alert] if warn else None,
                status_path=path, now_et_date=now[0].date(), **opt,
            )
        assert attempted is (case not in {"no_recipient", "duplicate"})
        if attempted:
            alert.send_dm.assert_called_once_with(recipient, body)
        else:
            alert.send_dm.assert_not_called()
        assert path.read_bytes() == frozen_bytes
        assert [entry.message for entry in caplog.records] == logs


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("injury", [
    "truncated", "array", "null", "bad_schema_type", "wrong_schema", "missing_schema",
    "bad_status", "bad_sent_day", "bad_sent_sources", "bad_message_id",
    "directory", "fifo", "unreadable", "symlink", "hardlink", "dir_link", "blocked_parent",
])
def test_unknown_receipt_ignores_unusable_path(world, monkeypatch, route, injury):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)
    path = own(directory)
    health = shared(directory)
    health.parent.mkdir()
    valid = {
        "schema_version": alert.HEALTH_DM_STATUS_SCHEMA_VERSION, "status": "sent", "sent_day": DAY,
        "sent_sources": ["CRITICAL:late_delivery"], "message_id": "old",
    }
    shared_bytes = json.dumps(valid).encode()
    health.write_bytes(shared_bytes)
    malformed = {
        "truncated": b"{", "array": b"[]", "null": b"null",
        "bad_schema_type": json.dumps(valid | {"schema_version": ["damaged"]}).encode(),
        "wrong_schema": json.dumps(valid | {"schema_version": "unknown"}).encode(),
        "missing_schema": json.dumps({k: v for k, v in valid.items() if k != "schema_version"}).encode(),
        "bad_status": json.dumps(valid | {"status": ["sent"]}).encode(),
        "bad_sent_day": json.dumps(valid | {"sent_day": [DAY]}).encode(),
        "bad_sent_sources": json.dumps(valid | {"sent_sources": "CRITICAL:late_delivery"}).encode(),
        "bad_message_id": json.dumps(valid | {"message_id": 17}).encode(),
    }
    if injury == "blocked_parent":
        path.parent.touch()
    elif injury == "dir_link":
        path.parent.symlink_to(health.parent, target_is_directory=True)
    else:
        path.parent.mkdir()
        if injury == "directory":
            path.mkdir()
        elif injury == "fifo":
            os.mkfifo(path)
        elif injury == "symlink":
            path.symlink_to(health)
        elif injury == "hardlink":
            os.link(health, path)
        else:
            path.write_bytes(shared_bytes if injury == "unreadable" else malformed[injury])
            if injury == "unreadable":
                path.chmod(0)

    def timed_out(*_):
        raise BaseException("receipt I/O timed out")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(5)
    try:
        if route == "direct":
            assert direct(config, directory, daily) is False
        else:
            assert scheduler.run_day(DAY, config) is None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    alert.send_dm.assert_called_once_with("offline.test", copy(DAY))
    assert health.read_bytes() == shared_bytes
    if injury in malformed:
        assert json.loads(path.read_bytes()) == valid | {"message_id": "health-dm"}
        scheduler.run_day(DAY, config)
        alert.send_dm.assert_called_once()
    elif injury in {"fifo", "directory", "symlink", "hardlink"}:
        assert os.path.lexists(path)


def test_no_recipient_retries_when_configured(world):
    config, directory, _, _ = world
    config["bluesky"].pop("dm_recipient")
    assert direct(config, directory) is False
    assert json.loads(own(directory).read_bytes())["status"] == "skipped_no_recipient"
    alert.send_dm.assert_not_called()
    config["bluesky"]["dm_recipient"] = "offline.test"
    assert direct(config, directory) is False
    alert.send_dm.assert_called_once_with("offline.test", copy(DAY))


@pytest.mark.parametrize("order", [("unknown", "cutoff"), ("cutoff", "unknown")])
def test_unknown_receipt_isolated_from_cutoff_and_eod(world, monkeypatch, order):
    config, directory, now, _ = world
    cutoff_dir = directory.with_name("cutoff-picks")
    set_dated(cutoff_dir)
    now[0] = now[0].replace(hour=19)
    for route in order:
        assert (direct(config, directory) if route == "unknown"
                else direct(config, cutoff_dir, daily_pick())) is False
    bodies = [call.args[1] for call in alert.send_dm.call_args_list]
    assert len(bodies) == 2 and sum(body == copy(DAY) for body in bodies) == 1
    original = own(directory).read_bytes()
    unrelated = [alert.Alert("CRITICAL", "pick_delivery", "Unrelated check")]
    alert.send_dm.side_effect = [OSError("offline"), "eod-ack"]
    assert alert.dispatch_dm_for_health_alerts(
        unrelated, "offline.test", status_path=shared(directory), now_et_date=now[0].date(),
    )
    assert json.loads(shared(directory).read_bytes())["status"] == "failed"
    assert direct(config, directory) is False
    assert own(directory).read_bytes() == original

    eod_day = now[0].date() - timedelta(days=1)
    monkeypatch.setattr(runner, "_safe_run", lambda name, _: unrelated if name == "post_failure" else [])
    monkeypatch.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
    runner.run_all_checks(directory, directory.parent / "models", "offline.test", today=eod_day)
    assert json.loads(shared(directory).read_bytes())["sent_day"] == eod_day.isoformat()
    assert direct(config, directory) is False
    assert own(directory).read_bytes() == original
    assert alert.send_dm.call_count == 4


@pytest.mark.parametrize("day,following", [("2027-03-25", "2027-03-26"), ("2027-12-31", "2028-01-01")])
def test_unknown_receipt_uses_slate_across_midnight(world, monkeypatch, day, following):
    config, directory, now, _ = world
    prepare_run_day(monkeypatch)
    et = ZoneInfo("America/New_York")
    games = {}
    for slate in (day, following):
        daily = daily_pick()
        daily.date = slate
        daily.pick.game_time = f"{slate}T23:10:00Z"
        if daily.double_down:
            daily.double_down.game_time = f"{slate}T23:40:00Z"
        daily.delivery_attempted = True
        picks.save_pick(daily, directory)
        games[slate] = [{"gamePk": 101, "gameDate": daily.pick.game_time,
                         "teams": {"away": {"team": {"name": "Away"}},
                                   "home": {"team": {"name": "Home"}}}}]
    scheduler.fetch_schedule.side_effect = games.get
    shared(directory).parent.mkdir()
    shared(directory).write_bytes(b"[]")
    now[0] = datetime.fromisoformat(f"{day}T18:30").replace(tzinfo=et)
    scheduler.run_day(day, config)
    alert.send_dm.assert_called_once_with("offline.test", copy(day))
    original = own(directory, day).read_bytes()
    now[0] = datetime.fromisoformat(f"{following}T00:01").replace(tzinfo=et)
    scheduler.run_day(day, config)
    alert.send_dm.assert_called_once()
    assert own(directory, day).read_bytes() == original
    now[0] = now[0].replace(hour=18, minute=30)
    scheduler.run_day(following, config)
    assert [call.args[1] for call in alert.send_dm.call_args_list] == [copy(day), copy(following)]
    assert json.loads(own(directory, following).read_bytes())["sent_day"] == following
    assert shared(directory).read_bytes() == b"[]"


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("durable", [False, True])
def test_restart_retries_unknown_alert_after_interrupt(world, monkeypatch, interrupt, durable):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    dispatch = alert.dispatch_dm_for_health_alerts
    with monkeypatch.context() as crash:
        if durable:
            def checkpoint(*args, **kwargs):
                dispatch(*args, **kwargs)
                raise interrupt()

            crash.setattr(alert, "dispatch_dm_for_health_alerts", checkpoint)
        else:
            alert.send_dm.side_effect = interrupt()
        with pytest.raises(interrupt):
            direct(config, directory)
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "locked_unconfirmed"
    assert own(directory).exists() is durable
    alert.send_dm.side_effect = None
    scheduler.run_day(DAY, config)
    scheduler.run_day(DAY, config)
    assert alert.send_dm.call_count == (1 if durable else 2)
    dm.send_dm.assert_not_called()


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("stage", ["dispatch", "formatter", "directory", "write", "stderr", "unprintable"])
def test_unknown_alert_failure_is_contained_after_commit(world, monkeypatch, capsys, route, stage):
    config, directory, _, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)
    error = OSError("first line\nsecond line" + "x" * 500)
    if stage in {"dispatch", "stderr", "unprintable"}:
        def fail(*args, **kwargs):
            if stage == "unprintable":
                class UnprintableError(Exception):
                    def __str__(self):
                        raise ValueError("cannot format")

                raise UnprintableError()
            if stage == "stderr":
                class Closed:
                    def write(self, text):
                        if text.startswith("Unknown-outcome alert failed: "):
                            raise ValueError("closed")
                        return len(text)

                monkeypatch.setattr(sys, "stderr", Closed())
            raise error

        monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", fail)
    elif stage == "formatter":
        monkeypatch.setattr(alert, "format_health_dm_body", Mock(side_effect=error))
    elif stage == "directory":
        own(directory).parent.parent.mkdir()
        own(directory).parent.touch()
    else:
        replace = os.replace

        def fail_receipt_replace(source, target, **kwargs):
            if kwargs.get("dst_dir_fd") is not None and target == f"{DAY}.json":
                raise error
            return replace(source, target, **kwargs)

        monkeypatch.setattr(os, "replace", fail_receipt_replace)
    if route == "direct":
        assert direct(config, directory, daily) is False
    else:
        scheduler.run_day(DAY, config)
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "locked_unconfirmed"
    prefix = "Unknown-outcome alert failed: "
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith(prefix)]
    assert len(lines) == (0 if stage == "stderr" else 1)
    if lines:
        assert len(lines[0].removeprefix(prefix)) <= 200
        if stage in {"directory", "write"}:
            assert str(own(directory)) in lines[0]
            alert.send_dm.assert_called_once()
        elif stage == "unprintable":
            assert "UnprintableError" in lines[0]
        else:
            assert r"first line\nsecond line" in lines[0]


@pytest.mark.parametrize("route", ["direct", "classification"])
@pytest.mark.parametrize("scenario,hour,minute,day_offset,entered", [
    ("never_left", 18, 35, 0, False),
    ("accepted", 18, 35, 0, True),
    ("cutoff", 19, 6, 0, False),
    ("first_pitch", 19, 14, 0, False),
    ("overnight", 0, 1, 1, False),
    ("switched_private", 18, 35, 0, False),
])
def test_unknown_alert_operator_copy(world, monkeypatch, route, scenario, hour, minute, day_offset, entered):
    config, directory, now, _ = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    daily.delivery_attempted = True
    picks.save_pick(daily, directory)
    now[0] = now[0].replace(hour=hour, minute=minute) + timedelta(days=day_offset)
    cutoff = datetime.fromisoformat(daily.pick.game_time) - timedelta(minutes=5)
    entry = {DAY: daily.pick.batter_id} if entered else {}
    if scenario == "switched_private":
        config["scheduler"].update(pick_delivery="private", entry_intent="research")

    checkpoints = []

    def send(_recipient, body):
        checkpoints.append((now[0], entry.get(DAY), body))
        return None if len(checkpoints) == 1 else "ack"

    alert.send_dm.side_effect = send
    for _ in range(2):
        if route == "direct":
            assert direct(config, directory, daily) is False
        else:
            scheduler.run_day(DAY, config)
    assert [call.args[1] for call in alert.send_dm.call_args_list] == [copy(DAY)] * 2
    for observed_at, entered_batter, body in checkpoints:
        assert f"the MLB entry for {DAY} ET" in body
        assert "Do not enter or change a pick based on this alert." in body
        assert f"If {DAY} is still today in ET and its submission window is open" in body
        window_closed = observed_at.date().isoformat() != DAY or observed_at >= cutoff
        assert window_closed is (scenario in {"overnight", "cutoff", "first_pitch"})
        assert entered_batter == (daily.pick.batter_id if entered else None)
    receipt = json.loads(own(directory).read_bytes())
    assert receipt["status"] == "sent" and receipt["message_id"] == "ack"


@pytest.mark.parametrize("control", ["dm", "public", "private", "stale", "pre_attempt", "eod",
                                     "dry_run", "off_day", "schedule_error", "void"])
def test_unknown_alert_leaves_pick_lifecycle_and_backstops_unchanged(world, monkeypatch, control):
    config, directory, now, statuses = world
    prepare_run_day(monkeypatch)
    daily = daily_pick()
    if control == "eod":
        daily.delivery_attempted = True
        picks.save_pick(daily, directory)
        alerts = post_failure.check(directory, today=now[0].date(), now=now[0].replace(hour=22))
        assert any(a.source == "pick_delivery" for a in alerts)
    elif control in {"dm", "public", "private", "void"}:
        config["scheduler"]["pick_delivery"] = "dm" if control == "void" else control
        set_dated(directory)
        daily.chosen_on = record()
        if control == "void":
            statuses[101] = {"abstract": "P", "detailed": "Cancelled", "code": "C"}
            monkeypatch.setattr(scheduler, "_delivery_game_statuses", lambda _: statuses)
        assert direct(config, directory, daily) is (control != "void")
        stored = picks.load_pick(DAY, directory)
        if control in {"dm", "public", "private"}:
            assert picks.pick_was_delivered(stored) is (control != "private")
            scheduler.run_day(DAY, config)
    else:
        picks.save_pick(daily, directory)
        if control == "stale":
            statuses[101] = {"abstract": "L", "detailed": "In Progress", "code": "I"}
        if control in {"dry_run", "off_day", "schedule_error"}:
            daily.delivery_attempted = True
            picks.save_pick(daily, directory)
        if control == "off_day":
            scheduler.fetch_schedule.side_effect = lambda _: []
        if control == "schedule_error":
            scheduler.fetch_schedule.side_effect = OSError("offline")
            with pytest.raises(OSError, match="offline"):
                scheduler.run_day(DAY, config)
        else:
            scheduler.run_day(DAY, config, dry_run=control == "dry_run")
    assert all("DELIVERY OUTCOME UNKNOWN" not in call.args[1] for call in alert.send_dm.call_args_list)
    assert not own(directory).exists()
