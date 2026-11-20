"""Readiness CLI uses the scheduler's config and state without starting it."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import contest_state, scheduler
from bts.cli import cli
from tests.test_contest_dated_cs4a import observation, write


@pytest.mark.parametrize(
    "scheduler_config,expected_exit,classification",
    [
        ('entry_intent = "research"\npick_delivery = "private"\n', 0, "contest state skipped"),
        ('entry_intent = "research"\npick_delivery = "dm"\n', 1, "configuration refused"),
        (
            'entry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n',
            1,
            "contest state refused",
        ),
    ],
)
def test_deploy_canary_classifies_without_scheduler_start(
    tmp_path, monkeypatch, scheduler_config, expected_exit, classification,
):
    picks = tmp_path / "picks"
    picks.mkdir()
    sentinel = picks / "existing.json"
    sentinel.write_bytes(b'{"unchanged":true}\n')
    config = tmp_path / "scheduler.toml"
    config.write_text(
        f"[orchestrator]\npicks_dir = {json.dumps(str(picks))}\n"
        f"[scheduler]\n{scheduler_config}"
        "[health_checks]\ncontest_state_expected = true\n"
    )

    def unexpected_start(*args, **kwargs):
        pytest.fail("readiness invoked scheduler startup")

    monkeypatch.setattr(scheduler, "run_day", unexpected_start)
    reads = []

    def refused(picks_dir, *, require_contest_state, **kwargs):
        reads.append((picks_dir, require_contest_state))
        raise contest_state.ContestStateError("fixture private detail")

    monkeypatch.setattr(contest_state, "load_decision_streak_state", refused)
    result = CliRunner().invoke(cli, ["deploy-canary", "--config", str(config)])
    assert result.exit_code == expected_exit, result.output
    assert classification in result.output
    assert "fixture private detail" not in result.output
    assert reads == ([(picks, True)] if classification == "contest state refused" else [])
    assert {path.name: path.read_bytes() for path in picks.iterdir()} == {
        "existing.json": b'{"unchanged":true}\n',
    }


@pytest.mark.parametrize("day,next_stamp", [
    ("2027-07-15", "2027-07-16T00:00:01"),
    ("2027-12-31", "2028-01-01T00:00:01"),
])
def test_live_canary_uses_real_dated_observation_at_et_boundary(tmp_path, monkeypatch, day, next_stamp):
    from importlib import import_module

    et = ZoneInfo("America/New_York")
    before = datetime.fromisoformat(day + "T23:59:59").replace(tzinfo=et)
    after = datetime.fromisoformat(next_stamp).replace(tzinfo=et)
    picks = tmp_path / "picks"
    source = write(picks, observation(day, recorded_at=day + "T12:00:00-05:00"))
    original = source.read_bytes()
    config = tmp_path / "scheduler.toml"
    config.write_text(
        f"[orchestrator]\npicks_dir = {json.dumps(str(picks))}\n"
        '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
        "[health_checks]\ncontest_state_expected = true\n"
    )
    clocks = iter((before, after))
    monkeypatch.setattr(import_module("bts.cli"), "_now_et", lambda: next(clocks))
    runner = CliRunner()
    passed = runner.invoke(cli, ["deploy-canary", "--config", str(config)])
    refused = runner.invoke(cli, ["deploy-canary", "--config", str(config)])
    assert (passed.exit_code, passed.output) == (
        0, f"deploy canary {before.isoformat()}: live contest state accepted (dated, streak 12)\n",
    )
    assert (refused.exit_code, refused.output) == (
        1, f"deploy canary {after.isoformat()}: contest state refused\n",
    )
    assert source.read_bytes() == original
