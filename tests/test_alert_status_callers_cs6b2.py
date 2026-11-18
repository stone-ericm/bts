"""Scheduler health alerts continue through corrupt shared status."""

import inspect
import json
from datetime import timedelta
from unittest.mock import Mock

import pytest

from bts import dm, orchestrator, picks, posting, scheduler, strategy
from bts.health import alert, runner
from tests.test_pick_chosen_on_cs5a import DAY, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


@pytest.mark.parametrize("corrupt", [False, True], ids=["green_missing", "corrupt"])
@pytest.mark.parametrize("case,source", [
    ("check", "contest_state"),
    ("state_read", "contest_state"),
    ("saved_pick", "contest_state"),
    ("in_loop", "contest_state"),
    ("final", "contest_state"),
    ("missed", "missed_pick"),
    ("late", "late_delivery"),
    ("void", "postponed_pick"),
    ("unconfirmed", "late_delivery"),
    ("classified_unconfirmed", None),
    ("eod", "contest_state"),
])
def test_shared_alert_callers(case, source, corrupt, world, monkeypatch, tmp_path):
    config, directory, now, statuses = world
    # The policy's first lookup can use the working directory. Restore its
    # module cache after the isolated scheduler run.
    monkeypatch.setattr(strategy, "_mdp_cache", None)
    monkeypatch.chdir(tmp_path)
    path = directory.parent / "health_state" / "health_dm_delivery_status.json"
    if corrupt:
        path.parent.mkdir()
        path.write_text("[]")
    prepare_run_day(monkeypatch)
    idle = Mock()
    monkeypatch.setattr(scheduler, "_idle_until_next_wakeup", idle)
    monkeypatch.setattr(scheduler, "notify_ready", Mock())
    monkeypatch.setattr(scheduler, "notify_watchdog", Mock())
    state = scheduler_state()
    daily = daily_pick()
    dispatch = alert.dispatch_dm_for_health_alerts
    observed = []
    fallback_calls = []

    def observe(*args, **kwargs):
        frame = inspect.currentframe().f_back
        observed.append((frame.f_code.co_name, frame.f_lineno, args[0]))
        if frame.f_code.co_name == "_alert_contest_state_failure":
            parent = frame.f_back
            if parent.f_code.co_name == "run_day":
                fallback_calls.append(parent.f_lineno)
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", observe)
    monkeypatch.setattr(runner, "dispatch_dm_for_health_alerts", observe)
    if case == "check":
        result = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
        assert result["pick_result"] is None
    elif case in {"in_loop", "final", "eod", "classified_unconfirmed"}:
        if case == "eod":
            config["health_checks"]["enabled"] = True
            orchestrator.run_cascade.return_value = (None, None)
            monkeypatch.setattr(runner, "get_self_pid", lambda: None)
            monkeypatch.setattr(runner, "read_systemd_nrestarts", lambda: None)
            monkeypatch.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        elif case == "classified_unconfirmed":
            daily.delivery_attempted = True
            picks.save_pick(daily, directory)
        else:
            picks.save_pick(daily, directory)
            if case == "in_loop":
                set_dated(directory)
                monkeypatch.setattr(
                    scheduler, "_lock_decision_from_predictions",
                    lambda *a, **kw: scheduler.LockDecision(False, None, False),
                )
                # Ensure the first fallback refresh runs after a wait and uses
                # the real contest-state reader after the record is withdrawn.
                now[0] -= timedelta(minutes=10)
                frame, tier = orchestrator.run_cascade.return_value

                def predict(*_):
                    if orchestrator.run_cascade.call_count > 1:
                        (directory / "account_state" / "contest_streak.manual.json").unlink(missing_ok=True)
                    return frame, tier

                orchestrator.run_cascade.side_effect = predict
            else:
                frame, tier = orchestrator.run_cascade.return_value
                orchestrator.run_cascade.side_effect = [(None, None), (frame, tier)]
        scheduler.run_day(DAY, config)
        idle.assert_called_once()
        assert scheduler.load_state(DAY, directory).pick_locked is (case == "classified_unconfirmed")
        if case in {"in_loop", "final"}:
            lines, first = inspect.getsourcelines(scheduler.run_day)
            fallback_sites = [
                first + index for index, line in enumerate(lines)
                if "_alert_contest_state_failure(config, e)" in line
            ]
            in_loop, final = fallback_sites
            routes = [line for name, line, _ in observed if name == "_alert_contest_state_failure"]
            assert routes  # The dispatcher is called through the alert helper.
            assert (in_loop if case == "in_loop" else final) in fallback_calls
            if case == "final":
                assert in_loop not in fallback_calls
    elif case == "missed":
        pick_path = picks.save_pick(daily, directory)
        original = pick_path.read_bytes()
        scheduler._maybe_alert_missed_pick(config, DAY, directory, 10, None, state)
        assert pick_path.read_bytes() == original
    else:
        if case != "state_read":
            set_dated(directory)
        if case == "late":
            now[0] += timedelta(hours=1)
        elif case == "void":
            statuses[101] = {"abstract": "P", "detailed": "Cancelled", "code": "C"}
            monkeypatch.setattr(scheduler, "_delivery_game_statuses", lambda _: statuses)
        elif case == "unconfirmed":
            daily.delivery_attempted = True
        pick_path = picks.save_pick(daily, directory)
        original = pick_path.read_bytes()
        assert scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, case) is False
        assert state.pick_locked is (case == "unconfirmed")
        if case in {"late", "void"}:
            reason = "past_submission_cutoff" if case == "late" else "game_not_playable"
            refusal, = scheduler.load_state(DAY, directory).delivery_refusals
            assert refusal["reason"] == reason
            assert not pick_path.exists() and (directory / DAY / refusal["archive"]).is_file()
        else:
            assert pick_path.read_bytes() == original

    if source is None:
        assert not observed
        alert.send_dm.assert_not_called()
        if corrupt:
            assert path.read_text() == "[]"
        else:
            assert not path.exists()
    else:
        assert any(a.source == source and a.level == "CRITICAL" for _, _, alerts in observed for a in alerts)
        assert sum(f"[{source}]" in call.args[1] for call in alert.send_dm.call_args_list) == 1
        assert json.loads(path.read_bytes())["status"] == "sent"
    dm.send_dm.assert_not_called()
    posting.post_to_bluesky.assert_not_called()
