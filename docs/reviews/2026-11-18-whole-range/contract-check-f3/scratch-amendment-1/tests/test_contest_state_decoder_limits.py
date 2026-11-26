# Adopted verbatim from the Codex review of change set 1, round 2 (marker CS1RC2-5e45e3ae, finding B2),
# then extended by the lead below the marked line (the contest state files, an object with an
# over-long number inside it, and a deeply nested body).
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
def test_row5_large_json_scalar_cannot_block_observation(tmp_path, observation, required, caplog, covered_automatic):
    account = tmp_path / "account_state"
    account.mkdir()
    payload = {"active_streak": 7, "best_streak": 20, "source_date": "2026-06-18"}
    if observation == "manual":
        payload["override_expires_at"] = "2026-06-18T23:59:59Z"
        if required:
            from tests.test_season_contract_cs4b2 import dated
            payload = dated("2026-06-18", active_streak=7, best_streak=20)
    filename = "contest_streak.manual.json" if observation == "manual" else "contest_streak.json"
    (account / filename).write_text(json.dumps(payload))
    if observation == "auto" and required:
        covered_automatic(tmp_path, NOW, payload)
    write_large_scalar(tmp_path / "streak.json")
    state = load_decision_streak_state(tmp_path, require_contest_state=required, now=NOW)
    assert state.streak == 7
    assert state.model_streak is state.model_saver_available is None
    assert "ValueError" in state.model_state_error
    assert len([r for r in caplog.records if "model_state_corrupt" in r.message]) == 1


@pytest.mark.parametrize("required", [False, True])
def test_rows6_7_large_json_scalar_has_domain_error(tmp_path, required):
    write_large_scalar(tmp_path / "streak.json")
    message = "switch is absent" if required else "model state is corrupt"
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


# --- added by the lead after round 2 ---

@pytest.mark.parametrize("filename", ["contest_streak.json", "contest_streak.manual.json"])
def test_row2_large_json_scalar_in_contest_file_is_a_domain_error(tmp_path, filename):
    account = tmp_path / "account_state"
    account.mkdir()
    write_large_scalar(account / filename)
    with pytest.raises(ContestStateError) as exc:
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert filename in str(exc.value)


def test_row1_pick_object_holding_a_large_number_is_skipped(tmp_path):
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("this interpreter disables the integer conversion limit")
    (tmp_path / "2026-06-18.json").write_bytes(b'{"result":' + b"9" * (limit + 1) + b"}")
    (tmp_path / "2026-06-17.json").write_text('{"result":"miss"}')
    assert _resolved_pick_dates(tmp_path) == [date(2026, 6, 17)]
    assert _has_unconfirmed_miss(tmp_path, date(2026, 6, 16))


DEEP = b"[" * 200_000 + b"]" * 200_000


def test_deeply_nested_body_is_isolated_everywhere(tmp_path):
    """A body nested past the interpreter's recursion limit (RecursionError) is corrupt, not fatal."""
    account = tmp_path / "account_state"
    account.mkdir()
    (account / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_manual_v3", "season": 2026,
        "effective_date": "2026-06-18", "active_streak": 7, "best_streak": 20,
        "saver_state": "used", "source": "manual_cli", "recorded_at": NOW.isoformat(),
        "expires_at": "2026-06-19T03:59:59.999999Z"}))
    (tmp_path / "streak.json").write_bytes(DEEP)
    (tmp_path / "2026-06-18.json").write_bytes(DEEP)
    (tmp_path / "2026-06-17.json").write_text('{"result":"miss"}')
    assert _resolved_pick_dates(tmp_path) == [date(2026, 6, 17)]
    assert _has_unconfirmed_miss(tmp_path, date(2026, 6, 16))
    state = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert state.streak == 7 and state.model_streak is None and state.model_state_error

    (account / "contest_streak.manual.json").write_bytes(DEEP)
    with pytest.raises(ContestStateError):
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
