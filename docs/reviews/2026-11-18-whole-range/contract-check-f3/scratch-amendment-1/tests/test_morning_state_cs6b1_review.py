"""Round-1 review of change set 6b-1 (Codex BLOCK B1 to B3 and its surviving mutants).

B1: an interrupt (KeyboardInterrupt, SystemExit) during the morning work is the
operator's or the process's and must leave ``run_day`` as it would on the base;
only ordinary exceptions from the work are contained.
B2: a malformed decision or pick file is "nothing found", never the end of the check.
B3: the refusal text reaches the journal once, bounded; a failed send must not
republish it in a traceback.
"""

import json
import logging
from datetime import timedelta
from unittest.mock import Mock

import pytest

from bts import contest_state, scheduler
from bts.health import alert
from tests.test_morning_state_cs6b1 import (  # noqa: F401
    FAILURE, FIRST, MORNING_FILE, START, morning, refusal,
)
from tests.test_pick_chosen_on_cs5a import DAY, set_dated
from tests.test_saved_pick_paths_cs5a import world  # noqa: F401
from tests.test_stale_delivery_cs4c import seed_automatic


def run_until_first_check(config, orchestrator_module, now):
    """Run the day; report what escaped and whether the first prediction was reached."""
    reached = []

    def predict(*args, **kwargs):
        reached.append(now[0])
        return None, None

    orchestrator_module.run_cascade.side_effect = predict
    escaped = None
    try:
        scheduler.run_day(DAY, config)
    except BaseException as caught:  # an interrupt must fail a test, not end the session
        escaped = caught
    return escaped, reached


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("site", ["read", "dispatcher", "transport", "advisory_print", "failure_print"])
def test_an_interrupt_during_the_morning_work_propagates(morning, monkeypatch, interrupt, site):
    from bts import orchestrator

    config, directory, now, _ = morning
    error = interrupt("operator")
    if site == "read":
        monkeypatch.setattr(contest_state, "load_decision_streak_state", Mock(side_effect=error))
    elif site == "dispatcher":
        monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", Mock(side_effect=error))
    elif site == "transport":
        alert.send_dm.side_effect = error
    else:
        if site == "failure_print":
            monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts",
                                Mock(side_effect=RuntimeError("dispatch")))
        blocked = "  MORNING STATE CHECK" if site == "advisory_print" else FAILURE

        def printed(*args, **kwargs):
            if str(args[0]).startswith(blocked):
                raise error
            print(*args, **kwargs)
        monkeypatch.setattr(scheduler, "print", printed, raising=False)
    escaped, reached = run_until_first_check(config, orchestrator, now)
    assert escaped is error
    assert reached == []


def _write_decision(directory, body):
    folder = directory / DAY
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "decision.json").write_text(body)


NESTED = "[" * 20000 + "]" * 20000


@pytest.mark.parametrize("record", ["unhashable_action", "nested_decision", "nested_pick"])
def test_a_malformed_commit_record_is_nothing_found(morning, capsys, record):
    from bts import orchestrator

    config, directory, now, _ = morning
    if record == "unhashable_action":
        _write_decision(directory, json.dumps({
            "schema_version": "bts_daily_decision_v3", "date": DAY, "scoreable": True, "action": [],
        }))
    elif record == "nested_decision":
        _write_decision(directory, NESTED)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{DAY}.json").write_text(NESTED)
        # The lineup checks would meet the same file; this test is about the morning read.
        orchestrator.run_cascade.side_effect = None
    escaped = None
    try:
        scheduler.run_day(DAY, config)
    except BaseException as caught:
        escaped = caught
    # What the rest of the day does with the same malformed file is the base's
    # behaviour and not this change set's (the lineup checks raise on it: a note
    # for a follow-up); the morning check must have reported first.
    del escaped
    assert alert.send_dm.call_args_list[0].args == ("offline.test", refusal())
    assert FAILURE not in capsys.readouterr().err


def test_a_failed_send_does_not_republish_the_refusal(morning, capsys, caplog):
    config, directory, now, _ = morning
    alert.send_dm.side_effect = OSError("transport down")
    with caplog.at_level(logging.DEBUG):
        scheduler.run_day(DAY, config)
    assert json.loads((directory.parent / "health_state" / MORNING_FILE).read_bytes())["status"] == "failed"
    journal = [line for line in capsys.readouterr().err.splitlines()
               if line.startswith("  MORNING STATE CHECK — no pick can be made yet: ")]
    assert len(journal) == 1 and len(journal[0]) <= 49 + 800
    rendered = "\n".join(
        record.getMessage() + (logging.Formatter().formatException(record.exc_info) if record.exc_info else "")
        for record in caplog.records
    )
    assert "transport down" in rendered
    assert "ContestStateError" not in rendered
    assert "set-contest-streak" not in rendered and "season-contract enable" not in rendered


@pytest.mark.parametrize("lane", ["refusal", "stale"])
def test_the_dispatch_day_is_passed_explicitly(morning, monkeypatch, lane):
    config, directory, now, _ = morning
    if lane == "stale":
        seed_automatic(directory, (START - timedelta(hours=20)).isoformat())
    real = alert.dispatch_dm_for_health_alerts
    days = []

    def spy(*args, **kwargs):
        days.append(kwargs.get("now_et_date"))
        return real(*args, **kwargs)
    monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", spy)
    scheduler.run_day(DAY, config)
    assert days[0] == START.date()


def test_an_accepted_morning_reads_the_state_once(morning, monkeypatch):
    config, directory, now, _ = morning
    set_dated(directory)
    real = contest_state.load_decision_streak_state
    reads = []

    def spy(*args, **kwargs):
        reads.append(now[0])
        return real(*args, **kwargs)
    monkeypatch.setattr(contest_state, "load_decision_streak_state", spy)
    scheduler.run_day(DAY, config)
    assert reads.count(START) == 1
