"""Elapsed-time and pick-wire contracts for automatic readings."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from bts import contest_state, picks
from tests.test_pick_chosen_on_cs5a import DAY, NOW, account_file, daily_pick, set_automatic, set_dated


KEYS = ("schema", "recorded_at", "age_hours", "stale", "reason")
SCHEMA = "bts_pick_automatic_reading_v1"
RECORDED = "2027-07-14T22:35:00-04:00"
OLD = dict(zip(KEYS, (SCHEMA, RECORDED, 20.0, True, "old")))
KINDS = ("automatic", "dated", "legacy_manual", "model")
ET = ZoneInfo("America/New_York")


def reading(kind, recorded):
    state = contest_state.DecisionStreakState(7, False, True, "contest", "fresh", 0, False)
    return replace(state, observation_kind=kind, contest_recorded_at=recorded)


@pytest.mark.parametrize("kind", KINDS)
def test_real_loader_propagates_recorded_at(tmp_path, kind):
    if kind == "automatic":
        set_automatic(tmp_path)
        expected = NOW.isoformat()
    elif kind == "dated":
        set_dated(tmp_path)
        expected = NOW.isoformat()
    elif kind == "legacy_manual":
        expected = "literal legacy recorded_at"
        account_file(tmp_path, "contest_streak.manual.json", {
            "active_streak": 7, "recorded_at": expected, "source_date": DAY,
        })
    else:
        expected = None
    state = contest_state.load_decision_streak_state(
        tmp_path, now=NOW, require_contest_state=kind in {"automatic", "dated"},
    )
    assert state.observation_kind == kind
    assert state.contest_recorded_at == expected


@pytest.mark.parametrize("kind", KINDS)
def test_pure_kinds_and_no_io_or_clock(kind, monkeypatch):
    state = reading(kind, RECORDED)
    forbidden = Mock(side_effect=AssertionError("reading record used I/O or wall clock"))

    class Clock(datetime):
        now = forbidden
        today = forbidden
        utcnow = forbidden

    with monkeypatch.context() as patch:
        patch.setattr(contest_state, "datetime", Clock)
        patch.setattr(Path, "open", forbidden)
        patch.setattr(Path, "stat", forbidden)
        patch.setattr("builtins.open", forbidden)
        actual = contest_state.automatic_reading_record(state, NOW)
    forbidden.assert_not_called()
    if kind != "automatic":
        assert actual is None
    else:
        assert actual == OLD
        assert tuple(actual) == KEYS
        assert [type(value) for value in actual.values()] == [str, str, float, bool, str]
        assert contest_state.AUTOMATIC_READING_STALE_HOURS == 14


@pytest.mark.parametrize(("seconds", "microseconds", "expected_age", "reason"), [
    (50400, -1, 14.0, None), (50400, 0, 14.0, None), (50400, 1, 14.0, "old"),
    (-300, -1, -0.08, "future"), (-300, 0, -0.08, None), (-300, 1, -0.08, None),
])
def test_strict_elapsed_thresholds(seconds, microseconds, expected_age, reason):
    then = NOW.astimezone(timezone.utc) - timedelta(seconds=seconds, microseconds=microseconds)
    raw = then.isoformat().replace("+00:00", "Z")
    assert contest_state.automatic_reading_record(reading("automatic", raw), NOW) == {
        "schema": SCHEMA, "recorded_at": raw, "age_hours": expected_age,
        "stale": reason is not None, "reason": reason,
    }


@pytest.mark.parametrize(("recorded", "now", "age", "reason"), [
    ("2027-03-13T13:30:00-05:00", datetime(2027, 3, 14, 3, 30, tzinfo=ET), 13.0, None),
    ("2027-11-06T13:30:00-04:00", datetime(2027, 11, 7, 3, 30, tzinfo=ET), 15.0, "old"),
    ("2027-11-07T01:30:00-04:00", datetime(2027, 11, 7, 1, 30, tzinfo=ET, fold=1), 1.0, None),
])
def test_dst_uses_utc_instants(recorded, now, age, reason):
    assert contest_state.automatic_reading_record(reading("automatic", recorded), now) == dict(
        zip(KEYS, (SCHEMA, recorded, age, reason is not None, reason)),
    )


@pytest.mark.parametrize(("recorded", "kept"), [
    (None, None), ("", ""), ("2027-07-15T18:35", "2027-07-15T18:35"),
    ("bad ISO", "bad ISO"), ("0001-01-01T00:00:00+00:01", "0001-01-01T00:00:00+00:01"),
    ("9999-12-31T23:59:59-00:01", "9999-12-31T23:59:59-00:01"),
    ("2027-07-15\n18:35Z", None), ("2027-07-15\ud80018:35Z", None),
    ("a" * 128, "a" * 128), ("a" * 129, None), ("a\x7f", None), (1, None),
])
def test_unreadable_times(recorded, kept):
    assert contest_state.automatic_reading_record(reading("automatic", recorded), NOW) == dict(
        zip(KEYS, (SCHEMA, kept, None, True, "unreadable")),
    )


VALID = [
    OLD, dict(OLD, age_hours=14.0), dict(OLD, age_hours=14.0, reason=None, stale=False),
    dict(OLD, age_hours=-0.08, reason=None, stale=False),
    dict(OLD, age_hours=-0.08, reason="future"),
    dict(OLD, recorded_at="bad", age_hours=None, reason="unreadable"),
    dict(OLD, recorded_at=None, age_hours=None, reason="unreadable"),
    dict(OLD, age_hours=87_660_000.0),
]

INVALID = [
    None, [], 1, *({key: value for key, value in OLD.items() if key != missing} for missing in KEYS),
    dict(OLD, extra=True), dict(OLD, schema={}), dict(OLD, schema="old"),
    dict(OLD, recorded_at=[]), dict(OLD, recorded_at="bad"), dict(OLD, recorded_at="x" * 129),
    dict(OLD, recorded_at="\ud800"), dict(OLD, stale=1), dict(OLD, stale=False),
    dict(OLD, reason={"nested": ["future"]}), dict(OLD, reason="unknown"),
    dict(OLD, reason="unreadable", age_hours=None), dict(OLD, reason="unreadable", recorded_at=None),
    dict(OLD, age_hours=True), dict(OLD, age_hours=20), dict(OLD, age_hours=20.001),
    dict(OLD, age_hours=float("inf")), dict(OLD, age_hours=float("nan")),
    dict(OLD, age_hours=float("-inf")), dict(OLD, age_hours=87_660_000.01),
    dict(OLD, age_hours=13.99), dict(OLD, age_hours=-0.07, reason="future"),
    dict(OLD, age_hours=-0.09, reason=None, stale=False),
    dict(OLD, age_hours=14.01, reason=None, stale=False),
]


@pytest.mark.parametrize(("value", "expected"), [(value, value) for value in VALID]
                         + [(value, None) for value in INVALID])
def test_all_pick_loaders(tmp_path, value, expected):
    body = asdict(daily_pick())
    body["automatic_reading"] = dict(reversed(list(value.items()))) if isinstance(value, dict) else value
    raw = json.dumps(body).encode()
    (tmp_path / f"{DAY}.json").write_bytes(raw)
    (tmp_path / f"{DAY}.shadow.json").write_bytes(raw)
    loaded, held = picks.load_pick_bytes(DAY, tmp_path)
    assert held == raw
    for pick in (loaded, picks.load_pick(DAY, tmp_path), picks.load_shadow_pick(DAY, tmp_path)):
        actual = getattr(pick, "automatic_reading", None)
        assert actual == expected
        if expected is not None:
            assert tuple(actual) == KEYS
    if expected is not None:
        assert json.loads(picks.save_shadow_pick(picks.load_shadow_pick(DAY, tmp_path), tmp_path)
                          .read_bytes())["automatic_reading"] == expected


def test_canonical_main_bytes_receipt_ledger_and_history(tmp_path):
    from bts.entry_receipt import EntryReceipt
    from bts.state.export import _pick_to_historical
    from bts.state.regenerate import _hist_to_pick_file
    from scripts.audit.season_ledger.sources.pick_files import parse_pick_file

    body = asdict(picks.attach_provenance(daily_pick()))
    body.pop("automatic_reading", None)
    path = tmp_path / f"{DAY}.json"
    raw_prior = json.dumps(body, indent=2).encode()
    path.write_bytes(raw_prior)
    loaded = picks.load_pick(DAY, tmp_path)
    assert getattr(loaded, "automatic_reading", None) is None
    picks.save_pick(loaded, tmp_path)
    assert path.read_bytes() == raw_prior[:-2] + b',\n  "automatic_reading": null\n}'
    loaded.automatic_reading = OLD
    picks.attach_provenance(loaded)
    assert loaded.automatic_reading == OLD
    picks.save_pick(loaded, tmp_path)
    raw_after = path.read_bytes()
    trailing = json.dumps({"automatic_reading": OLD}, indent=2).encode()
    assert raw_after == raw_prior[:-2] + b",\n" + trailing[2:]
    observations = []
    for raw in (raw_prior, raw_after):
        path.write_bytes(raw)
        daily, held = picks.load_pick_bytes(DAY, tmp_path)
        receipt = EntryReceipt(et_date=DAY, started_at=NOW, clock=lambda: NOW,
                               expected_username=None, picks_dir=tmp_path)
        receipt.selection(daily, held, None, None)
        selected = receipt.record()["selection"]
        ledger, = parse_pick_file(path.name, raw).rows
        assert selected["pick_file_sha256"] == ledger["content_sha256"] == hashlib.sha256(raw).hexdigest()
        assert json.loads(ledger["record_raw_json"]) == json.loads(raw)
        history = _pick_to_historical(json.loads(raw))
        assert "automatic_reading" not in history
        assert "automatic_reading" not in _hist_to_pick_file(history)
        observations.append((selected, ledger, history))
    first, second = observations
    assert first[0]["slots"] == second[0]["slots"]
    assert first[0]["pick_file_sha256"] != second[0]["pick_file_sha256"]
    assert first[1]["obs_id"] != second[1]["obs_id"]
    assert first[2] == second[2]


def test_both_live_forward_projections_preserve_non_null_drift(tmp_path):
    from scripts import live_forward_capture_once as capture
    from scripts import live_forward_provenance_inventory as inventory

    absent = asdict(daily_pick())
    absent.pop("automatic_reading", None)
    null = dict(absent, automatic_reading=None)
    added = dict(absent, automatic_reading=OLD)
    changed = dict(absent, automatic_reading=dict(OLD, age_hours=20.01))
    pick_path = tmp_path / f"{DAY}.json"
    manifest_path = tmp_path / "manifest.json"
    for before, after, drift in ((absent, null, False), (null, absent, False),
                                 (null, added, True), (added, absent, True), (added, changed, True)):
        for project in (capture.decision_snapshot, inventory._decision_snapshot):
            assert (project(before) != project(after)) is drift
        before_sha = hashlib.sha256(json.dumps(before).encode()).hexdigest()
        raw_after = json.dumps(after).encode()
        pick_path.write_bytes(raw_after)
        snapshot = {"present": True, "source_sha256": before_sha, "production_pick_json": before}
        manifest_path.write_text(json.dumps({"production_pick_snapshot": snapshot}))
        captured = capture.existing_snapshot_state(
            manifest_path=manifest_path, pick_path=pick_path, expected_date=DAY,
        )
        inventoried = inventory._current_pick_snapshot_summary(
            snapshot=snapshot, date_key=DAY, picks_dir=tmp_path,
        )
        assert captured["current_pick_sha256"] == inventoried["current_pick_sha256"]
        assert captured["current_pick_sha256"] == hashlib.sha256(raw_after).hexdigest()
        assert captured["artifact_pick_snapshot_sha256"] == before_sha
        assert captured["snapshot_matches_current_pick"] is inventoried["matches_current_pick"] is False
        assert captured["snapshot_decision_matches_current_pick"] is (not drift)
        assert inventoried["decision_matches_current_pick"] is (not drift)
