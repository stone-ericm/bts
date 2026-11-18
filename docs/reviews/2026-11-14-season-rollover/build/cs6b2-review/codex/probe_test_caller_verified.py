"""Re-run the caller pairs with identical paths and a pinned slate clock."""

import inspect
import json
import os

import pytest

from bts import contest_state, scheduler, slate
from test_caller_replay import CONTAINED, NESTED, SOURCES, replay, world


@pytest.mark.parametrize("case", SOURCES)
@pytest.mark.parametrize("body", [b"[]", NESTED], ids=["list", "nested"])
def test_caller_status_replay(case, body, world, monkeypatch, tmp_path):
    monkeypatch.setattr(slate, "datetime", contest_state.datetime)
    root = tmp_path / "run"
    root.mkdir()
    control, clean_status = replay(case, None, world, monkeypatch, root)
    # Both alert text and its persisted hash can legitimately include the paths.
    root.rename(tmp_path / "control-saved")
    root.mkdir()
    outcome, status = replay(case, body, world, monkeypatch, root)

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
        expected_error = "AttributeError" if body == b"[]" else "RecursionError"
        assert outcome["error"] == (None if case in CONTAINED else expected_error)
        target = "BTS health WARN attention:" if case == "stale_delivery" else (
            "Morning check for" if case == "morning" else f"[{SOURCES[case]}]"
        )
        assert not any(target in message for _, message in outcome["health"])
        return
    assert outcome == control
    assert status == clean_status
