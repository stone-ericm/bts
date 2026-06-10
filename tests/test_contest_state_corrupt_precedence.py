"""A corrupt contest file must not block the other in precedence (audit D2).

load_contest_streak_state parsed both files eagerly and _parse_state_file raises
on malformed JSON, so a corrupt auto file blocked a valid unexpired manual
override — exactly the emergency the override exists to rescue. The fix keeps
fail-closed when the SELECTED file is corrupt but ignores a corrupt non-selected
file.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from bts.contest_state import load_contest_streak_state, ContestStateError


def _write_manual(state_dir, expires_at):
    (state_dir / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_manual_v2",
        "active_streak": 7, "best_streak": 9, "source": "manual",
        "source_date": "2026-06-08", "override_expires_at": expires_at,
    }))


def test_corrupt_auto_does_not_block_manual_override(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text("{corrupt not json")  # corrupt auto
    future = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
    _write_manual(state_dir, future)

    state = load_contest_streak_state(tmp_path)

    assert state is not None
    assert state.streak == 7  # the override won despite the corrupt auto file


def test_corrupt_auto_without_override_still_fails_closed(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text("{corrupt")  # corrupt auto, no override

    with pytest.raises(ContestStateError):
        load_contest_streak_state(tmp_path)
