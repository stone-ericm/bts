"""Reviewers' probes for audit finding F6, adopted as tests (2026-11-18).

Sources: docs/reviews/2026-11-14-season-rollover/build/batch-2026-11-18/
(`review_probes.py`, `confirm2/revision_probes.py`). Only the cases the change's own
tests do not already cover: the cron installer's `show` under both intents, and
fetch failures with no DM recipient.
"""
import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

import bts.cli as cli_module
from tests.test_contest_fetch import _mock_auth_success
from tests.test_entry_intent import ENTRY_CMD, box  # noqa: F401  (fixture)

RECIPIENT = " --dm-recipient stonehengee.bsky.social"
FETCH_CMD = "bts fetch-contest-streak"


# ---- F6: the cron installer's `show`, and fetch failures with no recipient ----

def test_show_keeps_four_fetches_and_drops_only_their_recipient_in_research(box):  # noqa: F811
    shown = {}
    for intent, body in (
        ("enter", 'entry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'),
        ("research", 'entry_intent = "research"\npick_delivery = "private"\n'),
    ):
        box.config.write_text("[scheduler]\n" + body)
        response = box.run("show")
        assert response.returncode == 0, response.stderr
        shown[intent] = [ln for ln in response.stdout.splitlines() if "# BTS-HETZNER" in ln]
    assert box.lines() is None                      # show writes nothing
    for intent in shown:
        fetch = [ln for ln in shown[intent] if FETCH_CMD in ln]
        assert [ln.split()[:5] for ln in fetch] == [
            "10 1 * * *".split(), "10 2 * * *".split(),
            "30 10 * * *".split(), "30 13 * * *".split(),
        ]
        assert all((RECIPIENT in ln) is (intent == "enter") for ln in fetch)
    # Nothing else differs: every other research line, the day-outcome DM included, is the enter line.
    assert [ln.replace(RECIPIENT, "") if FETCH_CMD in ln else ln
            for ln in shown["enter"] if ENTRY_CMD not in ln] == shown["research"]
    assert any("bts day-outcome" in ln and RECIPIENT in ln for ln in shown["research"])


@pytest.mark.parametrize("name,rounds,category", [
    ("empty-rounds", {}, "actionable"),
    ("rounds-network", "offline rounds", "transient"),
])
def test_fetch_rounds_failure_without_recipient_still_fails_loudly(tmp_path, monkeypatch, name, rounds, category):
    import httpx
    import bts.contest_fetch as cf
    import bts.dm

    send = Mock()
    monkeypatch.setattr(bts.dm, "send_dm", send)
    _mock_auth_success(monkeypatch)
    fetch_rounds = (Mock(return_value=rounds) if isinstance(rounds, dict)
                    else Mock(side_effect=httpx.ConnectError(rounds)))
    monkeypatch.setattr(cli_module, "_fetch_rounds", fetch_rounds)
    monkeypatch.setattr(cf, "fetch_profile", lambda *_: {
        "activeStreak": 0, "seasonBestStreak": 0, "predictions": [],
    })
    picks_dir = tmp_path / "picks"
    result = CliRunner().invoke(cli_module.cli, ["fetch-contest-streak", "--picks-dir", str(picks_dir)])
    assert result.exit_code == 2, (result.output, result.exception)
    status = json.loads((tmp_path / "health_state/contest_streak_fetch_status.json").read_text())
    assert status["last_error"] in result.output
    assert status["last_error_category"] == category
    assert status["last_alert_at"] is None
    send.assert_not_called()
