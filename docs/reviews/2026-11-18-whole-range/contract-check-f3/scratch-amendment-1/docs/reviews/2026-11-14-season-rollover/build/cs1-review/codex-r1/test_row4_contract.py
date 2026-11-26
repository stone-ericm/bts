"""The authoritative kickoff row 4 accepts any recorded_at string in CS1."""
import json
from datetime import datetime, timezone

import pytest

from bts.contest_state import load_decision_streak_state


@pytest.mark.parametrize("recorded_at", ["", "not-a-time"])
def test_row4_preserves_recorded_at_strings(tmp_path, recorded_at):
    account = tmp_path / "account_state"
    account.mkdir()
    (account / "contest_streak.json").write_text(json.dumps({
        "active_streak": 7, "best_streak": 20, "recorded_at": recorded_at,
    }))
    now = datetime(2026, 6, 17, 16, tzinfo=timezone.utc)
    optional = load_decision_streak_state(tmp_path, now=now)
    required = load_decision_streak_state(tmp_path, require_contest_state=True, now=now)
    assert (required.streak, required.status) == (7, "stale")
    assert required == optional
