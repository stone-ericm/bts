"""Adoptable regression tests from whole-range fix set B round-2 review."""
import json
from unittest.mock import Mock

import pytest

from bts import contest_state
from bts.leaderboard import auth
from tests.test_season_contract_cs4b1 import NOW, contract, fetch, files, offline, put, run, tmp_path
from tests.test_season_contract_cs4b2 import automatic, dated


@pytest.mark.parametrize("bad_saver", [b"9" * 5000, b"[" * 12000 + b"]" * 12000])
def test_status_automatic_fallback_surfaces_uncaught_saver_loader_errors(tmp_path, bad_saver):
    """Kills the mutant where omit_selected_dated=True skips load_saver_state.

    The independent saver diagnostic catches _UNREADABLE (including plain ValueError
    from int_max_str_digits and RecursionError), whereas load_saver_state only
    catches (JSONDecodeError, OSError, UnicodeDecodeError). When today's dated
    observation is in effect, live decision now succeeds on the dated saver, while
    automatic path alone must still invoke load_saver_state and refuse just as
    tomorrow's live loader raises.
    """
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic())
    manual_path = put(tmp_path, "contest_streak.manual.json", dated())
    (tmp_path / "account_state/saver_state.json").write_bytes(bad_saver)
    before = files(tmp_path)

    result = run(tmp_path, "season-contract", "status")
    assert result.exit_code == 0, result.output
    assert "live decision now: accepted; source=contest;" in result.output
    verdict, = [ln for ln in result.output.splitlines() if ln.startswith("automatic path alone:")]
    assert verdict.startswith("automatic path alone: refused:")
    assert "automatic path alone: accepted" not in result.output
    assert "expires at midnight ET" in result.output
    assert files(tmp_path) == before

    manual_path.unlink()
    with pytest.raises((ValueError, RecursionError)):
        contest_state.load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)


def test_status_automatic_fallback_reads_saver_and_model_in_second_verdict(tmp_path, monkeypatch):
    """Verify the omit_selected_dated=True call itself opens saver_state.json and streak.json."""
    from pathlib import Path

    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic())
    put(tmp_path, "contest_streak.manual.json", dated())
    put(tmp_path, "saver_state.json", {"season": NOW.year, "state": "active"})
    (tmp_path / "streak.json").write_text(json.dumps({"season": NOW.year, "streak": 4}))

    counts = {"saver_state.json": 0, "streak.json": 0}
    orig = Path.read_text

    def counting_read(path, *args, **kwargs):
        if path.name in counts:
            counts[path.name] += 1
        return orig(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)
    result = run(tmp_path, "season-contract", "status")
    assert result.exit_code == 0, result.output
    # saver_state.json: 2 reads in diagnostic section + 1 read in verdict(omit_selected_dated=True)
    assert counts["saver_state.json"] == 3
    # streak.json: 1 read in verdict() + 1 read in verdict(omit_selected_dated=True)
    assert counts["streak.json"] == 2


@pytest.mark.parametrize("layout", ["legacy", "category"])
@pytest.mark.parametrize(("stamp", "cooled"), [
    ("0001-01-01T00:00:00+01:00", False),
    ("2027-07-15T15:00:00+0000", True),
    ("2027-07-15", False),
    ("2027-07-15Z", False),
])
def test_fetch_cooldown_year_zero_compact_offset_and_date_only(tmp_path, monkeypatch, layout, stamp, cooled):
    from bts.dm import send_dm

    put(tmp_path, "season_contract.json", contract())
    status = tmp_path.parent / "health_state/contest_streak_fetch_status.json"
    status.parent.mkdir()
    prior = ({"last_alert_at": stamp} if layout == "legacy"
             else {"last_alert_at_by_category": {"actionable": stamp}})
    status.write_text(json.dumps(prior))
    send_dm.return_value = "ack-id"

    assert fetch(tmp_path).exit_code == 0
    after_success = json.loads(status.read_text())
    usable = stamp in {"0001-01-01T00:00:00+01:00", "2027-07-15T15:00:00+0000"}
    if usable:
        assert {k: after_success[k] for k in prior} == prior
    else:
        assert after_success.get("last_alert_at") is None
        assert after_success.get("last_alert_at_by_category", {}) == {}

    monkeypatch.setattr(auth, "fetch_login_session", Mock(side_effect=auth.AuthError("offline")))
    assert fetch(tmp_path).exit_code == 2
    assert send_dm.call_count == (0 if cooled else 1)
