"""Readiness CLI uses the scheduler's config and state without starting it."""

import json

import pytest
from click.testing import CliRunner

from bts import contest_state, scheduler
from bts.cli import cli


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
