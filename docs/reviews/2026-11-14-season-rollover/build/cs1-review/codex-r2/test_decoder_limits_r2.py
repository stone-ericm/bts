"""Valid nonobject JSON must be isolated even when Python refuses integer conversion."""
import json
import sys
from datetime import date, datetime, timezone

import pytest

from bts.contest_state import (
    ContestStateError, _has_unconfirmed_miss, _resolved_pick_dates,
    load_decision_streak_state,
)

NOW = datetime(2026, 6, 18, 16, tzinfo=timezone.utc)


def write_large_scalar(path):
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("this interpreter disables the integer conversion limit")
    raw = b"9" * (limit + 1)
    assert json.loads(raw, parse_int=lambda text: len(text)) == limit + 1
    path.write_bytes(raw)


@pytest.mark.parametrize("observation", ["manual", "auto"])
@pytest.mark.parametrize("required", [False, True])
def test_row5_large_json_scalar_cannot_block_observation(tmp_path, observation, required, caplog):
    account = tmp_path / "account_state"
    account.mkdir()
    payload = {"active_streak": 7, "best_streak": 20, "source_date": "2026-06-18"}
    if observation == "manual":
        payload["override_expires_at"] = "2026-06-18T23:59:59Z"
    filename = "contest_streak.manual.json" if observation == "manual" else "contest_streak.json"
    (account / filename).write_text(json.dumps(payload))
    write_large_scalar(tmp_path / "streak.json")
    state = load_decision_streak_state(tmp_path, require_contest_state=required, now=NOW)
    assert state.streak == 7
    assert state.model_streak is state.model_saver_available is None
    assert "ValueError" in state.model_state_error
    assert len([r for r in caplog.records if "model_state_corrupt" in r.message]) == 1


@pytest.mark.parametrize("required", [False, True])
def test_rows6_7_large_json_scalar_has_domain_error(tmp_path, required):
    write_large_scalar(tmp_path / "streak.json")
    message = "required but missing" if required else "model state is corrupt"
    with pytest.raises(ContestStateError, match=message):
        load_decision_streak_state(tmp_path, require_contest_state=required, now=NOW)


@pytest.mark.parametrize("scan", ["resolved", "miss"])
def test_row1_large_json_scalar_is_skipped(tmp_path, scan):
    write_large_scalar(tmp_path / "2026-06-18.json")
    if scan == "resolved":
        assert _resolved_pick_dates(tmp_path) == []
    else:
        assert not _has_unconfirmed_miss(tmp_path, date(2026, 6, 16))
    (tmp_path / "2026-06-17.json").write_text('{"result":"miss"}')
    if scan == "resolved":
        assert _resolved_pick_dates(tmp_path) == [date(2026, 6, 17)]
    else:
        assert _has_unconfirmed_miss(tmp_path, date(2026, 6, 16))
