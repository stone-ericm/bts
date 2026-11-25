"""CS1 contract: corrupt disk state cannot masquerade as an observation.

All account values are synthetic; no contest account or network is used.
"""
import json
import logging
from dataclasses import replace
from datetime import date, datetime, timezone

import pytest
from click.testing import CliRunner

from bts.contest_state import (
    ContestStateError,
    _has_unconfirmed_miss,
    _resolved_pick_dates,
    load_decision_streak_state,
)


NOW = datetime(2027, 3, 24, 16, tzinfo=timezone.utc)
CORRUPTIONS = [b"\xff", b"[]", b"{broken", None]
CORRUPTION_IDS = ["utf8", "non_object", "json", "oserror"]


def _corrupt(path, contents):
    if contents is None:
        # Reading a real directory raises OSError on every supported test host;
        # chmod-based failures disappear when the test runner is privileged.
        if path.exists():
            path.unlink()
        path.mkdir()
    else:
        path.write_bytes(contents)


def _auto(picks_dir, **extra):
    from bts.contest_fetch import build_observation, validate_fetch

    profile = {"activeStreak": 13, "seasonBestStreak": 20}
    validate_fetch(profile)
    body = build_observation(
        profile,
        source_date=date(2027, 3, 24), user_id=1, username="fixture",
        recorded_at=NOW,
    )
    body.update(extra)
    path = picks_dir / "account_state" / "contest_streak.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def _manual(picks_dir):
    from bts.cli import cli

    result = CliRunner().invoke(cli, [
        "set-contest-streak", "--streak", "0", "--best-streak", "0",
        "--date", "2027-03-24", "--saver", "not_earned",
        "--picks-dir", str(picks_dir),
    ])
    assert result.exit_code == 0, result.output


def _covered(picks, factory, **extra):
    path = _auto(picks, **extra)
    return factory(picks, NOW, json.loads(path.read_text()))


@pytest.fixture
def pinned_clock(monkeypatch):
    from tests.test_season_contract_cs4b1 import pin
    pin(monkeypatch, NOW)


@pytest.mark.parametrize("contents", CORRUPTIONS, ids=CORRUPTION_IDS)
def test_corrupt_pick_is_skipped_by_both_scans(tmp_path, contents, covered_automatic):
    """Row 1: neither scan loses the other readable pick files."""
    _covered(tmp_path, covered_automatic, source_date="2027-03-22")
    _corrupt(tmp_path / "2027-03-24.json", contents)
    assert _resolved_pick_dates(tmp_path) == []
    assert not _has_unconfirmed_miss(tmp_path, date(2027, 3, 22))

    (tmp_path / "2027-03-23.json").write_text('{"result":"miss"}')
    assert _resolved_pick_dates(tmp_path) == [date(2027, 3, 23)]
    assert _has_unconfirmed_miss(tmp_path, date(2027, 3, 22))
    state = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert (state.streak, state.status) == (13, "stale")


@pytest.mark.parametrize("result", [[], {}, ["miss"]], ids=["list", "object", "list-of-miss"])
def test_pick_with_unhashable_result_is_skipped_by_both_scans(tmp_path, result, covered_automatic):
    """Row 1 (review r1 N6): a pick object whose result is not a string is not a settled pick."""
    _covered(tmp_path, covered_automatic, source_date="2027-03-22")
    (tmp_path / "2027-03-24.json").write_text(json.dumps({"result": result}))
    (tmp_path / "2027-03-23.json").write_text('{"result":"miss"}')
    assert _resolved_pick_dates(tmp_path) == [date(2027, 3, 23)]
    assert _has_unconfirmed_miss(tmp_path, date(2027, 3, 22))
    state = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert (state.streak, state.status) == (13, "stale")


@pytest.mark.parametrize("filename", ["contest_streak.json", "contest_streak.manual.json"])
@pytest.mark.parametrize("contents", CORRUPTIONS, ids=CORRUPTION_IDS)
def test_corrupt_contest_file_raises_domain_error(tmp_path, filename, contents):
    """Row 2: bad encoding is reported like other malformed observations."""
    path = _auto(tmp_path).with_name(filename)
    _corrupt(path, contents)
    with pytest.raises(ContestStateError) as exc:
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("contents", CORRUPTIONS, ids=CORRUPTION_IDS)
def test_manual_precedes_corrupt_auto(tmp_path, contents, pinned_clock):
    """Row 2 precedence: the non-selected automatic file is never parsed."""
    _corrupt(_auto(tmp_path), contents)
    _manual(tmp_path)
    assert load_decision_streak_state(
        tmp_path, require_contest_state=True, now=NOW).streak == 0


@pytest.mark.parametrize("streak,best", [(13, 10), (401, 401), (13, 401)])
def test_u1_bad_best_reachable_after_corruption(tmp_path, streak, best, covered_automatic):
    """Row 3 / U1: mutate a valid producer artifact AFTER successful loading."""
    path = _covered(tmp_path, covered_automatic)
    assert load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW).streak == 13
    text = path.read_text()
    path.write_text(text.replace('"active_streak": 13', f'"active_streak": {streak}')
                   .replace('"best_streak": 20', f'"best_streak": {best}'))
    stored = json.loads(path.read_text())
    assert (stored["active_streak"], stored["best_streak"]) == (streak, best)
    with pytest.raises(ContestStateError) as exc:
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert all(value in str(exc.value) for value in (str(path), str(streak), str(best)))


@pytest.mark.parametrize("streak,best", [(13, 10), (401, 401), (13, 401)])
def test_manual_bounds_unchanged_in_cs1(tmp_path, streak, best):
    path = _auto(tmp_path, active_streak=streak, best_streak=best)
    path.rename(path.with_name("contest_streak.manual.json"))
    state = load_decision_streak_state(tmp_path, require_contest_state=False, now=NOW)
    assert (state.streak, state.best_streak, state.best_status) == (streak, best, "untrusted")
    with pytest.raises(ContestStateError, match="legacy"):
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)


@pytest.mark.parametrize("source_date", [None, "not-a-date"])
def test_legacy_fixture_exception_is_live(tmp_path, source_date):
    """Row 4 / F4: a bare fixture is indistinguishable from a restored live file."""
    path = _auto(tmp_path)
    path.write_text(json.dumps({"active_streak": 14, "best_streak": 18,
                                "source_date": source_date}))
    # Optional mode keeps accepting the same last-confirmed state.
    assert load_decision_streak_state(tmp_path, now=NOW).status == "stale"
    with pytest.raises(ContestStateError, match="undated") as exc:
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("recorded_at", ["", "not-a-time", 5, None])
def test_unparseable_recorded_at_does_not_date_an_observation(tmp_path, recorded_at):
    """Row 4 (lead's correction after the handover): only a timestamp that parses counts."""
    path = _auto(tmp_path)
    path.write_text(json.dumps({"schema_version": "bts_contest_streak_auto_v1",
                                "active_streak": 14, "best_streak": 18,
                                "recorded_at": recorded_at}))
    assert load_decision_streak_state(tmp_path, now=NOW).status == "stale"
    with pytest.raises(ContestStateError, match="undated"):
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)


@pytest.mark.parametrize("metadata", [
    {"source_date": "2027-03-24"}, {"as_of": "2027-03-24"},
    {"source_date": "not-a-date", "as_of": "2027-03-24"},
    {"recorded_at": NOW.isoformat()},
])
def test_dated_version_one_auto_still_accepted(tmp_path, metadata):
    path = _auto(tmp_path)
    path.write_text(json.dumps({"schema_version": "bts_contest_streak_auto_v1",
                                "active_streak": 14, "best_streak": 18, **metadata}))
    optional = load_decision_streak_state(tmp_path, now=NOW)
    assert optional.streak == 14
    with pytest.raises(ContestStateError, match="absent"):
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    if metadata.get("source_date") == "not-a-date":
        assert optional.status == "stale"  # preserve the existing primary-date parsing


@pytest.mark.parametrize("observation", ["manual", "automatic"])
@pytest.mark.parametrize("contents", CORRUPTIONS, ids=CORRUPTION_IDS)
def test_manual_not_universal_recovery(tmp_path, caplog, observation, contents, covered_automatic, pinned_clock):
    """Row 5 / F1: model diagnostics cannot block either valid observation."""
    _covered(tmp_path, covered_automatic)
    if observation == "manual":
        _manual(tmp_path)
    path = tmp_path / "streak.json"
    path.write_text('{"streak":4,"saver_available":true}')
    healthy = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    _corrupt(path, contents)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with caplog.at_level(logging.WARNING, logger="bts.contest_state"):
        state = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert state.model_streak is None and state.model_saver_available is None
    assert str(path) in state.model_state_error
    assert state == replace(healthy, model_streak=None, model_saver_available=None,
                            model_state_error=state.model_state_error)
    warnings = [r for r in caplog.records if r.name == "bts.contest_state" and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "model_state_corrupt" in warnings[0].message
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    if contents is None:
        assert path.is_dir()
    else:
        assert path.read_bytes() == contents


@pytest.mark.parametrize("contents", CORRUPTIONS, ids=CORRUPTION_IDS)
def test_corrupt_model_cannot_be_fallback(tmp_path, contents):
    """Row 6: report corruption instead of inventing a model streak."""
    path = tmp_path / "streak.json"
    _corrupt(path, contents)
    with pytest.raises(ContestStateError, match="model state.*corrupt") as exc:
        load_decision_streak_state(tmp_path, now=NOW)
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("contents", CORRUPTIONS, ids=CORRUPTION_IDS)
def test_missing_required_contest_wins_over_corrupt_model(tmp_path, contents):
    """Row 7: the required-contest refusal keeps its established meaning."""
    _corrupt(tmp_path / "streak.json", contents)
    with pytest.raises(ContestStateError, match="switch is absent"):
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)


def test_empty_model_object_keeps_existing_defaults(tmp_path):
    """Row 8: {} is a legacy object, not malformed model state."""
    from bts.picks import load_applied_dates, load_saver_available, load_streak

    (tmp_path / "streak.json").write_text("{}")
    assert (load_streak(tmp_path), load_saver_available(tmp_path), load_applied_dates(tmp_path)) == (0, True, [])
    state = load_decision_streak_state(tmp_path, now=NOW)
    assert (state.streak, state.model_streak, state.model_saver_available) == (0, 0, True)
    assert state.model_state_error is None
    _auto(tmp_path)
    assert load_decision_streak_state(tmp_path, now=NOW).model_state_error is None


def test_dashboard_handles_missing_model_diagnostics(tmp_path):
    from bts.web import _streak_subtitle

    _auto(tmp_path)
    (tmp_path / "streak.json").write_bytes(b"\xff")
    state = load_decision_streak_state(tmp_path, now=NOW)
    assert _streak_subtitle(state) == "Contest State"

