"""Round-1 review of change set 4c (Codex BLOCK B1 and B2, and its two false greens).

B1: nothing the stale-reading work raises, a ``BaseException`` included, may stop
the pick's send or interrupt the return after it.
B2: the warning says "the pick was delivered", so it is sent only when the
transport returned a non-empty identifier.
"""

import json
from datetime import timedelta
from unittest.mock import Mock

import pytest

from bts import contest_state, picks, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_automatic_reading_cs4c import OLD
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick
from tests.test_saved_pick_paths_cs5a import world  # noqa: F401
from tests.test_stale_delivery_cs4c import (
    PREFIX, WARNING_FILE, envelope, fresh, pick_transport, seed_automatic, send,
)


class Custom(BaseException):
    pass


INTERRUPTS = [KeyboardInterrupt, SystemExit, GeneratorExit, Custom]


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("kind", INTERRUPTS)
@pytest.mark.parametrize("where", ["calculation", "dispatcher", "stderr", "diagnostic"])
def test_base_exceptions_in_the_stale_work_cannot_affect_the_pick(
        world, monkeypatch, capsys, mode, kind, where):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    seed_automatic(directory)
    error = kind("interrupt")
    if where == "calculation":
        monkeypatch.setattr(contest_state, "automatic_reading_record", Mock(side_effect=error))
    elif where == "dispatcher":
        monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", Mock(side_effect=error))
    else:
        # "stderr": the warning line cannot be printed. "diagnostic": the dispatcher
        # fails AND the failure line cannot be printed either.
        if where == "diagnostic":
            monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts",
                                Mock(side_effect=RuntimeError("dispatch")))
        blocked = PREFIX if where == "stderr" else "  Stale-reading warning failed:"

        def printed(*args, **kwargs):
            if str(args[0]).startswith(blocked):
                raise error
            print(*args, **kwargs)
        monkeypatch.setattr(scheduler, "print", printed, raising=False)
    daily = daily_pick()
    escaped = None
    try:
        result = send(daily, world, fresh(daily))
    except BaseException as caught:  # an escaping interrupt would otherwise end the test session
        escaped = caught
    assert escaped is None, f"the stale-reading work let {escaped!r} escape the delivery"
    assert result is True
    assert pick_transport(mode).call_count == 1
    final = picks.load_pick(DAY, directory)
    assert final.delivery_attempted and picks.pick_was_delivered(final)
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "delivered"
    assert final.automatic_reading == (None if where == "calculation" else OLD)
    if where == "stderr":
        # The dispatch is attempted independently of the stderr write.
        alert.send_dm.assert_called_once_with("offline.test", envelope())


@pytest.mark.parametrize("mode", ["dm", "public"])
# 19 and True: a truthy value that is not a str is no confirmation either (round-2 note).
@pytest.mark.parametrize("identifier", [None, "", 0, 19, True])
def test_no_warning_without_a_confirmation_identifier(world, capsys, mode, identifier):
    """The base's outcome is kept (return, markers, lock, decision); only the warning is withheld."""
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    seed_automatic(directory)
    pick_transport(mode).return_value = identifier
    daily = daily_pick()
    assert send(daily, world, fresh(daily)) is True
    assert pick_transport(mode).call_count == 1
    assert scheduler.load_state(DAY, directory).pick_locked
    assert picks.load_pick(DAY, directory).automatic_reading == OLD
    alert.send_dm.assert_not_called()
    assert PREFIX not in capsys.readouterr().err
    assert not (directory.parent / "health_state" / WARNING_FILE).exists()


def test_warning_day_is_passed_to_the_dispatcher_explicitly(world, monkeypatch):
    """False green 1: the dispatcher's own clock must not decide the dedup day."""
    _, directory, _, _ = world
    seed_automatic(directory)
    real = alert.dispatch_dm_for_health_alerts
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("now_et_date"))
        return real(*args, **kwargs)
    monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", spy)
    daily = daily_pick()
    assert send(daily, world, fresh(daily))
    assert seen == [NOW.date()]


@pytest.mark.parametrize("age", [True, False, 1, 0])
def test_validator_rejects_a_non_float_age_with_nothing_else_wrong(tmp_path, age):
    """False green 2: a bool (or int) age on an otherwise consistent, not-stale record."""
    recorded = (NOW - timedelta(hours=1)).isoformat()
    record = dict(OLD, recorded_at=recorded, age_hours=age, stale=False, reason=None)
    daily = daily_pick()
    path = picks.save_pick(daily, tmp_path)
    path.write_text(json.dumps(dict(json.loads(path.read_text()), automatic_reading=record)))
    assert picks.load_pick(DAY, tmp_path).automatic_reading is None
    assert picks.load_pick_bytes(DAY, tmp_path)[0].automatic_reading is None
    control = dict(record, age_hours=1.0)
    path.write_text(json.dumps(dict(json.loads(path.read_text()), automatic_reading=control)))
    assert picks.load_pick(DAY, tmp_path).automatic_reading == control
