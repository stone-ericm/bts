"""Compare scheduler state and delivery effects for each real status-file consumer."""

import inspect
import json
import os
import signal
from dataclasses import asdict
from datetime import timedelta
from unittest.mock import Mock

import pytest

from bts import contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
from bts.health import alert, runner
from tests.test_morning_state_cs6b1 import FIRST, MORNING_FILE, START
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world
from tests.test_stale_delivery_cs4c import WARNING_FILE, fresh, seed_automatic


NESTED = b"[" * 20_000 + b"0" + b"]" * 20_000
SOURCES = {
    "check": "contest_state", "state_read": "contest_state", "saved_pick": "contest_state",
    "in_loop": "contest_state", "final": "contest_state", "missed": "missed_pick",
    "late": "late_delivery", "void": "postponed_pick", "unconfirmed": "late_delivery",
    "classified_unconfirmed": None, "eod": "contest_state", "morning": "contest_state",
    "stale_delivery": "contest_state",
}
CONTAINED = {"saved_pick", "eod", "morning", "stale_delivery", "classified_unconfirmed"}


def replay(case, payload, world_data, monkeypatch, root):
    config, directory, now, statuses = world_data
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    directory = root / "picks"
    config["orchestrator"].update(picks_dir=str(directory), heartbeat_path=str(root / "heartbeat"),
                                  models_dir=str(root / "models"))
    file = {"morning": MORNING_FILE, "stale_delivery": WARNING_FILE}.get(case, "health_dm_delivery_status.json")
    path = root / "health_state" / file
    if payload is not None:
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)

    now[0] = START if case == "morning" else NOW
    with monkeypatch.context() as test:
        test.chdir(root)
        test.setattr(strategy, "_mdp_cache", None)
        for module in (daily_decision, picks, scheduler, strategy):
            test.setattr(module, "datetime", contest_state.datetime)
        prediction = orchestrator.run_cascade.return_value
        test.setattr(orchestrator, "run_cascade", Mock(return_value=prediction))
        test.setattr(scheduler, "fetch_schedule", Mock())
        test.setattr(scheduler, "notify_ready", Mock())
        test.setattr(scheduler, "notify_watchdog", Mock())
        for module, attr, value in [(dm, "send_dm", "pick-dm"), (alert, "send_dm", "health-dm"),
                                    (posting, "post_to_bluesky", "at://post")]:
            test.setattr(module, attr, Mock(return_value=value))
        prepare_run_day(test)
        idle = Mock()
        test.setattr(scheduler, "_idle_until_next_wakeup", idle)
        if case == "morning":
            def advance(seconds):
                now[0] += timedelta(seconds=seconds)
            test.setattr(scheduler, "_watchdog_ping_sleep", advance)
            orchestrator.run_cascade.return_value = (None, None)

        state, daily = scheduler_state(), daily_pick()
        actual_dispatch = alert.dispatch_dm_for_health_alerts
        seen = []

        def observe(*args, **kwargs):
            frame = inspect.currentframe().f_back
            route = frame.f_back.f_lineno if frame.f_code.co_name == "_alert_contest_state_failure" else frame.f_lineno
            seen.append((frame.f_code.co_name, route, kwargs["status_path"].name,
                         [(a.level, a.source) for a in args[0]],
                         [(a.level, a.source) for a in kwargs.get("warn_attention", ())]))
            return actual_dispatch(*args, **kwargs)

        test.setattr(alert, "dispatch_dm_for_health_alerts", observe)
        test.setattr(runner, "dispatch_dm_for_health_alerts", observe)

        def execute():
            if case == "check":
                return scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)["pick_result"]
            if case in {"in_loop", "final", "eod", "classified_unconfirmed", "morning"}:
                if case == "eod":
                    config["health_checks"]["enabled"] = True
                    orchestrator.run_cascade.return_value = (None, None)
                    test.setattr(runner, "get_self_pid", lambda: None)
                    test.setattr(runner, "read_systemd_nrestarts", lambda: None)
                    test.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
                elif case == "classified_unconfirmed":
                    daily.delivery_attempted = True
                    picks.save_pick(daily, directory)
                elif case in {"in_loop", "final"}:
                    picks.save_pick(daily, directory)
                    if case == "in_loop":
                        set_dated(directory)
                        test.setattr(scheduler, "_lock_decision_from_predictions",
                                     lambda *a, **kw: scheduler.LockDecision(False, None, False))
                        now[0] -= timedelta(minutes=10)

                        def predict(*_):
                            if orchestrator.run_cascade.call_count > 1:
                                (directory / "account_state" / "contest_streak.manual.json").unlink(missing_ok=True)
                            return prediction

                        orchestrator.run_cascade.side_effect = predict
                    else:
                        orchestrator.run_cascade.side_effect = [(None, None), prediction]
                return scheduler.run_day(DAY, config)
            if case == "missed":
                picks.save_pick(daily, directory)
                return scheduler._maybe_alert_missed_pick(config, DAY, directory, 10, None, state)
            if case == "stale_delivery":
                seed_automatic(directory)
                return scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, case, selection=fresh(daily))
            if case != "state_read":
                set_dated(directory)
            if case == "late":
                now[0] += timedelta(hours=1)
            elif case == "void":
                test.setattr(scheduler, "_delivery_game_statuses",
                             lambda _: statuses | {101: {"abstract": "P", "detailed": "Cancelled", "code": "C"}})
            elif case == "unconfirmed":
                daily.delivery_attempted = True
            picks.save_pick(daily, directory)
            return scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, case)

        def timeout(*_):
            pytest.fail(f"caller {case} blocked")

        handler = signal.signal(signal.SIGALRM, timeout)
        prior_timer = signal.setitimer(signal.ITIMER_REAL, 10)
        try:
            try:
                result, error = execute(), None
            except (AttributeError, RecursionError) as exc:
                result, error = None, type(exc).__name__
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, *prior_timer)

        health = [args for args, _ in alert.send_dm.call_args_list]
        pick = [args for args, _ in dm.send_dm.call_args_list]
        assert not posting.post_to_bluesky.called
        route = [entry for entry in seen if entry[2] == file]
        persisted = path.read_bytes() if path.exists() else None
        state_on_disk = {str(file.relative_to(directory)): json.loads(file.read_bytes())
                         for file in sorted(directory.rglob("*.json"))}
        summary = dict(result=result, error=error, health=health, pick=pick, seen=seen, route=route,
                       in_memory=asdict(state), disk=state_on_disk, idle=idle.call_count,
                       predictions=orchestrator.run_cascade.call_count, schedule=scheduler.fetch_schedule.call_count)
        summary = json.loads(json.dumps(summary).replace(str(root), "<scratch>"))
        return summary, persisted


@pytest.mark.parametrize("case", SOURCES)
@pytest.mark.parametrize("body", [b"[]", NESTED], ids=["list", "nested"])
def test_caller_status_replay(case, body, world, monkeypatch, tmp_path):
    expected_error = "AttributeError" if body == b"[]" else "RecursionError"
    clean, corrupt = tmp_path / "clean", tmp_path / "corrupt"
    clean.mkdir()
    corrupt.mkdir()
    control, clean_status = replay(case, None, world, monkeypatch, clean)
    outcome, status = replay(case, body, world, monkeypatch, corrupt)

    assert control["error"] is None
    assert len(control["pick"]) == int(case == "stale_delivery")
    if case in {"in_loop", "final", "eod", "classified_unconfirmed", "morning"}:
        assert control["idle"] == 1
    if SOURCES[case] is None:
        assert not control["route"] and clean_status is None and status == body
        assert outcome == control
        return
    assert control["route"] and clean_status and json.loads(clean_status)["status"] == "sent"
    expected_level = "WARN" if case == "stale_delivery" else "CRITICAL"
    assert any([expected_level, SOURCES[case]] in [*entry[3], *entry[4]] for entry in control["route"])
    if case in {"in_loop", "final"}:
        lines, first = inspect.getsourcelines(scheduler.run_day)
        sites = [first + i for i, line in enumerate(lines) if "_alert_contest_state_failure(config, e)" in line]
        reached = {entry[1] for entry in outcome["route"] if entry[0] == "_alert_contest_state_failure"}
        assert sites[0 if case == "in_loop" else 1] in reached
        if case == "final":
            assert sites[0] not in reached
    if os.environ.get("CS6B2_SOURCE") == "base":
        assert outcome["route"] and status == body
        assert outcome["error"] == (None if case in CONTAINED else expected_error)
        target = "BTS health WARN attention:" if case == "stale_delivery" else (
            "Morning check for" if case == "morning" else f"[{SOURCES[case]}]"
        )
        assert not any(target in message for _, message in outcome["health"])
        return
    assert outcome == control
    assert status == clean_status
