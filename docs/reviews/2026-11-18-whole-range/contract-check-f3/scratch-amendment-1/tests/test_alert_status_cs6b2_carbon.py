"""Adversarial review regression tests for CS6b-2:
- Special paths: FIFO with active writer/bytes, symlink to /dev/null, real chmod 0o000 & 0o200,
  parent-is-regular-file, and non-blocking stat->read_text races.
- Extreme JSON tokens: 200,000-deep array/object nesting at default recursion limit, >4300-digit ints,
  NaN/Infinity/-Infinity, lone surrogates, unhashable status, non-string sent_day, dedup-before-recipient gate.
- Real callers with 20,000-deep nested JSON and [] across shared, morning, and stale-reading status lanes.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from bts import contest_state, dm, orchestrator, picks, posting, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


PROBE_DAY = date(2027, 7, 14)
CRIT = alert.Alert("CRITICAL", "probe", "critical probe")
WARN = alert.Alert("WARN", "probe", "warn probe")
KEY = "CRITICAL:probe"


@contextlib.contextmanager
def bounded_alarm(seconds: float = 2.0):
    def _timeout(*_):
        pytest.fail(f"dispatcher blocked for >= {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _timeout)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)


def test_fifo_with_preloaded_covering_writer_is_untouched_and_logged(tmp_path, monkeypatch, caplog):
    fifo = tmp_path / "health_dm_delivery_status.json"
    os.mkfifo(fifo)
    rw_fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    payload = json.dumps({
        "status": "sent",
        "sent_day": PROBE_DAY.isoformat(),
        "sent_sources": [KEY],
    }).encode("utf-8") + b"\n"
    os.write(rw_fd, payload)
    send = Mock(return_value="ok")
    monkeypatch.setattr(alert, "send_dm", send)
    try:
        with caplog.at_level(logging.WARNING, logger="bts.health.alert"), bounded_alarm(2.0):
            sent = alert.dispatch_dm_for_health_alerts(
                [CRIT], "offline.test", status_path=fifo, now_et_date=PROBE_DAY,
            )
        assert sent is True
        send.assert_called_once()
        assert "not a regular file" in caplog.text
        assert os.read(rw_fd, 4096) == payload
    finally:
        os.close(rw_fd)


def test_devnull_symlink_and_parent_file_log_persistence_failure(tmp_path, monkeypatch, caplog):
    send = Mock(return_value="ok")
    monkeypatch.setattr(alert, "send_dm", send)

    devnull_link = tmp_path / "devnull_status.json"
    devnull_link.symlink_to("/dev/null")
    with caplog.at_level(logging.WARNING, logger="bts.health.alert"), bounded_alarm(2.0):
        assert alert.dispatch_dm_for_health_alerts(
            [CRIT], "offline.test", status_path=devnull_link, now_et_date=PROBE_DAY,
        ) is True
    assert "not a regular file" in caplog.text

    caplog.clear()
    reg_parent = tmp_path / "regular_parent"
    reg_parent.write_text("file")
    with caplog.at_level(logging.WARNING, logger="bts.health.alert"), bounded_alarm(2.0):
        assert alert.dispatch_dm_for_health_alerts(
            [CRIT], "offline.test", status_path=reg_parent / "status.json", now_et_date=PROBE_DAY,
        ) is True
    assert "could not write health DM delivery status" in caplog.text
    assert send.call_count == 2


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses DAC read permissions")
@pytest.mark.parametrize("mode_bits,writable", [(0o000, False), (0o200, True)])
def test_real_denied_read_permissions(tmp_path, monkeypatch, caplog, mode_bits, writable):
    path = tmp_path / "perm_status.json"
    path.write_text(json.dumps({
        "status": "sent",
        "sent_day": PROBE_DAY.isoformat(),
        "sent_sources": [KEY],
    }))
    path.chmod(mode_bits)
    send = Mock(return_value="ok")
    monkeypatch.setattr(alert, "send_dm", send)
    try:
        with pytest.raises(PermissionError):
            path.read_text()
        with caplog.at_level(logging.WARNING, logger="bts.health.alert"), bounded_alarm(2.0):
            assert alert.dispatch_dm_for_health_alerts(
                [CRIT], "offline.test", status_path=path, now_et_date=PROBE_DAY,
            ) is True
        send.assert_called_once()
        path.chmod(0o600)
        if writable:
            assert json.loads(path.read_bytes())["status"] == "sent"
        else:
            assert "could not write health DM delivery status" in caplog.text
    finally:
        if path.exists():
            path.chmod(0o600)


@pytest.mark.parametrize("swap_kind", ["to_dir", "to_missing", "to_loop"])
def test_stat_to_read_non_fifo_races_recover(tmp_path, monkeypatch, swap_kind):
    path = tmp_path / "race_status.json"
    path.write_text(json.dumps({
        "status": "sent",
        "sent_day": PROBE_DAY.isoformat(),
        "sent_sources": [KEY],
    }))
    real_stat = Path.stat
    swapped = False

    def swap_after_stat(self, *args, **kwargs):
        nonlocal swapped
        st = real_stat(self, *args, **kwargs)
        if self == path and not swapped:
            swapped = True
            path.unlink()
            if swap_kind == "to_dir":
                path.mkdir()
            elif swap_kind == "to_loop":
                path.symlink_to(path.name)
        return st

    send = Mock(return_value="ok")
    monkeypatch.setattr(alert, "send_dm", send)
    monkeypatch.setattr(Path, "stat", swap_after_stat)
    with bounded_alarm(2.0):
        assert alert.dispatch_dm_for_health_alerts(
            [CRIT], "offline.test", status_path=path, now_et_date=PROBE_DAY,
        ) is True
    assert swapped is True
    send.assert_called_once()
    if swap_kind == "to_missing":
        assert json.loads(path.read_bytes())["status"] == "sent"


@pytest.mark.parametrize("raw", [
    b"[" * 200_000 + b"0" + b"]" * 200_000,
    b'{"a":' * 200_000 + b"0" + b"}" * 200_000,
    b"9" * 5000,
    b'{"status":"failed","critical_count":' + b"9" * 5000 + b"}",
    b"NaN",
    b"Infinity",
    b'{"status":"failed","critical_count":NaN}',
    b'{"status":"failed","critical_count":Infinity}',
    b'{"status":"failed","critical_count":-Infinity}',
    b'{"status":[]}',
    b'{"status":"sent","sent_day":20270714,"sent_sources":["CRITICAL:probe"]}',
    b'{"status":"sent","sent_day":"2027-07-14","sent_sources":["\\ud800"]}',
])
def test_extreme_json_tokens_never_raise_or_banner(tmp_path, monkeypatch, raw):
    path = tmp_path / "extreme.json"
    path.write_bytes(raw)
    send = Mock(return_value="ok")
    monkeypatch.setattr(alert, "send_dm", send)
    with bounded_alarm(2.0):
        assert alert.dispatch_dm_for_health_alerts(
            [], "offline.test", status_path=path, now_et_date=PROBE_DAY,
        ) is False
        send.assert_not_called()

        assert alert.dispatch_dm_for_health_alerts(
            [CRIT], "offline.test", status_path=path, now_et_date=PROBE_DAY,
        ) is True
    send.assert_called_once_with("offline.test", alert.format_health_dm_body([CRIT], []))
    persisted = json.loads(path.read_bytes())
    assert persisted["status"] == "sent"
    if b"\\ud800" in raw:
        assert persisted["sent_sources"] == [KEY, "\ud800"]
    else:
        assert persisted["sent_sources"] == [KEY]


def test_dedup_precedes_missing_recipient_gate(tmp_path, monkeypatch):
    path = tmp_path / "dedup_gate.json"
    original = json.dumps({
        "status": "sent",
        "sent_day": PROBE_DAY.isoformat(),
        "sent_sources": [KEY, 1, {}],
    }).encode("utf-8")
    path.write_bytes(original)
    monkeypatch.setattr(alert, "send_dm", Mock())
    assert alert.dispatch_dm_for_health_alerts(
        [CRIT], None, status_path=path, now_et_date=PROBE_DAY,
    ) is False
    assert path.read_bytes() == original


@pytest.mark.parametrize("payload_kind", ["array", "nested"])
@pytest.mark.parametrize("lane_case", ["morning_critical", "morning_stale_warn", "delivery_stale_warn", "check_nested"])
def test_lane_callers_recover_corrupt_and_nested_status(
    world, monkeypatch, tmp_path, payload_kind, lane_case,
):
    config, directory, now, _ = world
    monkeypatch.setattr(strategy, "_mdp_cache", strategy._mdp_cache)
    monkeypatch.chdir(tmp_path)
    prepare_run_day(monkeypatch)
    monkeypatch.setattr(scheduler, "notify_ready", Mock())
    monkeypatch.setattr(scheduler, "notify_watchdog", Mock())
    monkeypatch.setattr(scheduler, "time", Mock(monotonic=lambda: 0))

    raw = b"[]" if payload_kind == "array" else (b"[" * 20_000 + b"0" + b"]" * 20_000)
    health_dir = directory.parent / "health_state"
    health_dir.mkdir(parents=True, exist_ok=True)

    if lane_case == "morning_critical":
        status_file = health_dir / "contest_morning_check_dm_delivery_status.json"
        status_file.write_bytes(raw)
        now[0] = NOW.replace(hour=10, minute=0)

        def advance_and_repair(secs):
            now[0] += timedelta(seconds=secs)
            if not (directory / "account_state" / "contest_streak.manual.json").exists():
                set_dated(directory, now=now[0], streak=30, best=30)

        monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", advance_and_repair)
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
        with bounded_alarm(4.0):
            scheduler.run_day(DAY, config)
        assert alert.send_dm.call_count == 1
        assert json.loads(status_file.read_bytes())["sent_sources"] == ["CRITICAL:contest_state:morning"]
        assert dm.send_dm.call_count == 1
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"

    elif lane_case == "morning_stale_warn":
        status_file = health_dir / "contest_stale_reading_dm_delivery_status.json"
        status_file.write_bytes(raw)
        start = NOW.replace(hour=10, minute=0)
        now[0] = start
        acc = directory / "account_state"
        acc.mkdir(parents=True, exist_ok=True)
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2",
            "season": start.year,
            "activation_id": "a" * 32,
            "active_streak": 30,
            "best_streak": 30,
            "source": "manual_cli",
            "source_date": start.date().isoformat(),
            "recorded_at": (start - timedelta(hours=20)).isoformat(),
        }))
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": start.year,
            "status": "automatic",
            "activation_id": "a" * 32,
            "switched_at": start.isoformat(),
            "revoked_at": None,
            "note": None,
        }))
        monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", lambda secs: now.__setitem__(0, now[0] + timedelta(seconds=secs)))
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
        with bounded_alarm(4.0):
            scheduler.run_day(DAY, config)
        # Morning stale warning repairs the file to 'sent', so delivery's same-day stale warning is deduped!
        assert alert.send_dm.call_count == 1
        assert json.loads(status_file.read_bytes())["sent_sources"] == ["WARN_ATTENTION:contest_state:stale_reading"]
        assert dm.send_dm.call_count == 1
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"

    elif lane_case == "delivery_stale_warn":
        status_file = health_dir / "contest_stale_reading_dm_delivery_status.json"
        status_file.write_bytes(raw)
        acc = directory / "account_state"
        acc.mkdir(parents=True, exist_ok=True)
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2",
            "season": NOW.year,
            "activation_id": "a" * 32,
            "active_streak": 7,
            "best_streak": 7,
            "source": "manual_cli",
            "source_date": NOW.date().isoformat(),
            "recorded_at": (NOW - timedelta(hours=20)).isoformat(),
        }))
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": NOW.year,
            "status": "automatic",
            "activation_id": "a" * 32,
            "switched_at": NOW.isoformat(),
            "revoked_at": None,
            "note": None,
        }))
        daily = daily_pick()
        chosen = contest_state.load_decision_streak_state(
            directory, now=now[0], require_contest_state=True,
        )
        daily.chosen_on = picks.chosen_on_record(chosen, required=True)
        picks.save_pick(daily, directory)
        policy = strategy.PolicyDecision(
            "single", "single", "heuristic", "reach57", 7, 80, None, "missing",
            None, None, None, None, None,
        )
        sel = strategy.SelectionResult(
            strategy.PickResult(daily), "single", "heuristic", None, None, None,
            streak=7, decision=policy,
        )
        state = scheduler_state()
        with bounded_alarm(4.0):
            assert scheduler._deliver_and_lock_pick(
                daily, config, directory, state, DAY, "fixture", selection=sel,
            ) is True
        assert dm.send_dm.call_count == 1
        assert posting.post_to_bluesky.call_count == 0
        assert alert.send_dm.call_count == 1
        assert json.loads(status_file.read_bytes())["sent_sources"] == ["WARN_ATTENTION:contest_state:stale_reading"]
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"

    else:
        status_file = health_dir / "health_dm_delivery_status.json"
        status_file.write_bytes(raw)
        with bounded_alarm(4.0):
            res = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
        assert res["pick_result"] is None
        assert alert.send_dm.call_count == 1
        assert json.loads(status_file.read_bytes())["sent_sources"] == ["CRITICAL:contest_state"]
