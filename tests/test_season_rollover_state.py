"""Pinned season-boundary cases for decision streaks and contest health."""

import json
from datetime import datetime, timezone

import pytest

from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.health.contest_state import check

NOW = datetime(2027, 3, 24, 16, tzinfo=timezone.utc)


def _observation(picks, *, manual=False, **fields):
    directory = picks / "account_state"
    directory.mkdir(exist_ok=True)
    name = "contest_streak.manual.json" if manual else "contest_streak.json"
    (directory / name).write_text(json.dumps({"active_streak": 18, **fields}))


def _model(picks):
    path = picks / "streak.json"
    path.write_text(json.dumps({"streak": 9, "saver_available": True}))
    return path.read_bytes()


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("fields", [
    {"source_date": "2026-09-26", "recorded_at": "2026-09-26T12:00:00Z"},
    {"source_date": "2027-03-24", "recorded_at": "2026-09-26T12:00:00Z"},
    {"recorded_at": "2026-09-26T12:00:00Z"},
    {"source_date": "2026-09-26"},
    {"source_date": "2026-09-26", "recorded_at": "invalid"},
    # UTC 2027, but still New Year's Eve 2026 in New York.
    {"source_date": "2027-03-24", "recorded_at": "2027-01-01T02:00:00Z"},
])
@pytest.mark.parametrize("required", [False, True])
def test_s1_prior_season_is_absent(tmp_path, manual, fields, required):
    before = _model(tmp_path)
    _observation(tmp_path, manual=manual, override_expires_at="2027-03-24T15:00:00Z", **fields)
    if required:
        with pytest.raises(ContestStateError, match="season 2026"):
            load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)
    else:
        state = load_decision_streak_state(tmp_path, now=NOW)
        assert (state.streak, state.model_streak, state.source, state.status) == (0, 0, "model", "model_only")
        assert state.contest_streak is None
        assert state.best_status == "missing"
    assert (tmp_path / "streak.json").read_bytes() == before


@pytest.mark.parametrize("contest", [False, True])
@pytest.mark.parametrize("kind,expected", [
    ("none", 0), ("prior", 0), ("pending", 0), ("shadow", 0), ("nested", 0),
    ("hit", 9), ("miss", 9), ("void", 9),
])
def test_s1_model_fallback_needs_current_settlement(tmp_path, contest, kind, expected):
    before = _model(tmp_path)
    (tmp_path / "2026-09-26.json").write_text('{"result":"hit"}')
    if contest:
        _observation(tmp_path, source_date="2026-09-26")
    if kind == "shadow":
        (tmp_path / "2027-03-24.shadow.json").write_text('{"result":"hit"}')
    elif kind == "nested":
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "2027-03-24.json").write_text('{"result":"hit"}')
    elif kind not in ("none", "prior"):
        (tmp_path / "2027-03-24.json").write_text(json.dumps({"result": kind}))
    state = load_decision_streak_state(tmp_path, now=NOW)
    assert (state.streak, state.model_streak, state.status) == (expected, expected, "model_only")
    assert (tmp_path / "streak.json").read_bytes() == before


@pytest.mark.parametrize("fields,status", [
    ({"source_date": "2027-03-24", "recorded_at": "2027-03-24T12:00:00Z"}, "fresh"),
    ({"source_date": "2027-03-24"}, "fresh"),
    ({"source_date": "2027-03-24", "recorded_at": "invalid"}, "fresh"),
    ({"recorded_at": "2027-03-24T12:00:00Z"}, "stale"),
    ({}, "stale"),
    ({"recorded_at": "invalid"}, "stale"),
])
def test_s1_current_or_unknown_season_unchanged(tmp_path, fields, status):
    _observation(tmp_path, **fields)
    state = load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)
    assert (state.streak, state.source, state.status) == (18, "contest", status)


@pytest.mark.parametrize("required", [False, True])
def test_s1_current_recording_with_prior_round_is_used(tmp_path, required):
    _model(tmp_path)
    _observation(tmp_path, source_date="2026-09-26", recorded_at="2027-03-24T12:00:00Z")
    state = load_decision_streak_state(tmp_path, now=NOW, require_contest_state=required)
    assert (state.streak, state.source, state.status) == (18, "contest", "fresh")
    assert state.contest_source_date.isoformat() == "2026-09-26"


@pytest.mark.parametrize("fields,status", [
    ({"source_date": "2026-09-26"}, "fresh"),
    ({"source_date": "2027-03-24", "recorded_at": "2026-09-26T00:00:00Z"}, "fresh"),
    ({"recorded_at": "2026-09-26T00:00:00Z"}, "stale"),
    ({}, "stale"),
])
def test_s1_unexpired_manual_override_preserved(tmp_path, fields, status):
    _observation(tmp_path, source_date="2027-03-24", active_streak=3)
    _observation(tmp_path, manual=True, override_expires_at="2027-03-25T00:00:00Z", **fields)
    state = load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)
    assert (state.streak, state.status, state.source) == (18, status, "contest")


def test_s1_auto_expiration_field_does_not_make_manual_override(tmp_path):
    _observation(tmp_path, source_date="2026-09-26", override_expires_at="2027-03-25T00:00:00Z")
    with pytest.raises(ContestStateError, match="season 2026"):
        load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)


def test_s1_now_year_is_new_york_year(tmp_path):
    _observation(tmp_path, source_date="2026-09-26")
    state = load_decision_streak_state(tmp_path, now=datetime(2027, 1, 1, 2, tzinfo=timezone.utc),
                                      require_contest_state=True)
    assert state.streak == 18 and state.status == "fresh"


@pytest.mark.parametrize("fields", [
    {"source_date": "2026-09-26", "recorded_at": "2026-09-26T12:00:00Z"},
    {"source_date": "2027-03-24", "recorded_at": "2026-09-26T12:00:00Z"},
    {"recorded_at": "2026-09-26T12:00:00Z"},
    {"source_date": "2026-09-26"},
    {"source_date": "2026-09-26", "recorded_at": "invalid"},
    {"source_date": "2027-03-24", "recorded_at": "2027-01-01T02:00:00Z"},
])
def test_s2_prior_season_alert_names_season_and_fetch(tmp_path, fields):
    _observation(tmp_path, **fields)
    alerts = check(tmp_path, expected=True, now=NOW)
    critical = [a.message for a in alerts if a.level == "CRITICAL"]
    assert len(critical) == 1
    assert "2026" in critical[0] and "2027" in critical[0] and "fetch" in critical[0].lower()
    assert "recorded" in critical[0].lower()


@pytest.mark.parametrize("settled,level", [(0, None), (1, "INFO"), (2, "WARN")])
def test_s2_current_recording_with_prior_round_keeps_coverage_checks(tmp_path, settled, level):
    _observation(tmp_path, active_streak=0, source_date="2026-09-26", recorded_at="2027-03-24T12:00:00Z")
    for day in range(settled):
        (tmp_path / f"2027-03-{22 + day}.json").write_text('{"result":"hit"}')
    alerts = check(tmp_path, expected=True, now=NOW)
    assert [a.level for a in alerts] == ([] if level is None else [level])


@pytest.mark.parametrize("local", ["none", "prior", "pending", "shadow", "nested"])
def test_s2_first_fetch_zero_without_settlement_is_not_critical(tmp_path, local):
    _observation(tmp_path, active_streak=0, recorded_at="2027-03-24T12:00:00Z")
    paths = {"prior": "2026-09-26.json", "pending": "2027-03-24.json",
             "shadow": "2027-03-24.shadow.json", "nested": "nested/2027-03-24.json"}
    if local in paths:
        path = tmp_path / paths[local]
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"result": "pending" if local == "pending" else "hit"}))
    assert not any(a.level == "CRITICAL" for a in check(tmp_path, expected=True, now=NOW))


@pytest.mark.parametrize("streak,recorded,result", [
    (1, "2027-03-24T12:00:00Z", None), (0, None, None), (0, "bad", None),
    (0, "2027-03-24T12:00:00Z", "hit"), (0, "2027-03-24T12:00:00Z", "miss"),
    (0, "2027-03-24T12:00:00Z", "void"),
])
def test_s2_other_missing_source_date_still_critical(tmp_path, streak, recorded, result):
    _observation(tmp_path, active_streak=streak, recorded_at=recorded)
    if result:
        (tmp_path / "2027-03-24.json").write_text(json.dumps({"result": result}))
    assert any(a.level == "CRITICAL" for a in check(tmp_path, expected=True, now=NOW))
