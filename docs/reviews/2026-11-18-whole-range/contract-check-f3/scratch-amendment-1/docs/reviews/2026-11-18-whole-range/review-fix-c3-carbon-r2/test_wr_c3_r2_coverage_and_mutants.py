"""Adopted tests killing surviving R2 mutants (M6 non-str uri/delivered_at guard, M7 unmocked
atomicity) and covering untested Rule-2 cells (non_scoreable_other + signals, absent decision +
individual signals, past tagless/foreign markers with skip/preview evidence, and today cells).
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

from bts import cli, picks
from bts.contest_state import DecisionStreakState
from bts.daily_decision import write_decision
from bts.reconcile_receipt import ReconcileReceipt


ET = ZoneInfo("America/New_York")
RUN_ET = datetime(2027, 4, 20, 2, 0, tzinfo=ET)
TODAY = "2027-04-20"
PAST = "2027-04-16"


def _plant_raw(root: Path, day: str, **overrides) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    model = DecisionStreakState(0, True, True, "model", "fresh", 0, True)
    payload = {
        "date": day,
        "run_time": f"{day}T19:00:00Z",
        "pick": {
            "batter_name": "Synthetic batter", "batter_id": 1, "team": "X", "lineup_position": 1,
            "pitcher_name": "Synthetic pitcher", "pitcher_id": 2, "p_game_hit": 0.8, "flags": [],
            "projected_lineup": False, "game_pk": 1, "game_time": f"{day}T23:00:00Z", "pitcher_team": None,
        },
        "double_down": None,
        "runner_up": None,
        "bluesky_posted": False,
        "bluesky_uri": None,
        "notification_sent": False,
        "notification_channel": None,
        "notification_id": None,
        "delivery_attempted": False,
        "delivered_at": None,
        "result": None,
        "slot_results": None,
        "chosen_on": picks.chosen_on_record(model, required=False),
    }
    payload.update(overrides)
    path = root / f"{day}.json"
    path.write_text(json.dumps(payload))
    return path


def _invoke(root: Path, monkeypatch):
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *a, **k: {"pick": "hit"})
    monkeypatch.setattr(
        "bts.reconcile_receipt.ReconcileReceipt",
        lambda **kw: ReconcileReceipt(clock=lambda: RUN_ET, lookback_days=kw["lookback_days"]),
    )
    return CliRunner().invoke(cli.cli, ["reconcile", "--picks-dir", str(root)])


@pytest.mark.parametrize(("field", "bad_value"), [
    ("bluesky_uri", 0),
    ("bluesky_uri", []),
    ("bluesky_uri", 123),
    ("delivered_at", 0),
    ("delivered_at", {}),
])
def test_non_string_uri_or_delivered_at_blocks_even_when_falsy(tmp_path, monkeypatch, field, bad_value):
    """Kills M6: falsy non-string bluesky_uri / delivered_at beside preview stamp must block."""
    _plant_raw(tmp_path, PAST, **{field: bad_value})
    picks.save_streak(6, tmp_path, saver_available=False, season=2027)
    before = (tmp_path / "streak.json").read_bytes()
    res = _invoke(tmp_path, monkeypatch)
    assert res.exit_code == 0
    assert res.stderr.splitlines() == [f"Season replay unavailable for {PAST} (unreadable or invalid pick file)"]
    assert (tmp_path / "streak.json").read_bytes() == before


def test_unmocked_atomic_rebuild_replaces_chmod000_streak_and_preserves_on_replace_error(tmp_path, monkeypatch):
    """Kills M7 without mocking atomic_write_text."""
    if os.geteuid() == 0:
        pytest.skip("chmod 000 check requires non-root")
    _plant_raw(tmp_path, "2027-04-01", result="hit", slot_results={"pick": "hit"})
    _plant_raw(tmp_path, PAST)
    sp = tmp_path / "streak.json"
    sp.write_text('{"streak": 9, "saver_available": false, "season": 2027}')
    os.chmod(sp, 0o000)
    try:
        res = _invoke(tmp_path, monkeypatch)
        assert res.exit_code == 0
        rebuilt = json.loads(sp.read_text())
        assert (rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"]) == (1, True, 2027)

        before = sp.read_bytes()
        with monkeypatch.context() as mp:
            mp.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk error")))
            res_err = _invoke(tmp_path, mp)
        assert res_err.exit_code != 0
        assert sp.read_bytes() == before
    finally:
        if sp.exists():
            os.chmod(sp, 0o600)


@pytest.mark.parametrize(("dec_kind", "sig", "marker", "day", "expected_reason"), [
    # non_scoreable_other (action='single', scoreable=False) beside each delivery/attempt signal
    ("non_scoreable_other", "bluesky_posted", "none", PAST, "skip record contradicts a delivery"),
    ("non_scoreable_other", "bluesky_uri", "none", PAST, "skip record contradicts a delivery"),
    ("non_scoreable_other", "notification_sent", "none", PAST, "skip record contradicts a delivery"),
    ("non_scoreable_other", "notification_id", "none", PAST, "skip record contradicts a delivery"),
    ("non_scoreable_other", "delivery_attempted", "none", PAST, "skip record contradicts a delivery"),
    ("non_scoreable_other", "delivered_at", "none", PAST, "skip record contradicts a delivery"),
    # absent decision beside individual signals not covered alone in the R2 table
    ("absent", "bluesky_posted", "none", PAST, "committed pick awaiting a result"),
    ("absent", "bluesky_uri", "none", PAST, "committed pick awaiting a result"),
    ("absent", "notification_sent", "none", PAST, "committed pick awaiting a result"),
    ("absent", "delivered_at", "none", PAST, "committed pick awaiting a result"),
    # past foreign-season marker does NOT block a valid preview or skip; past tagless marker DOES block
    ("absent", "none", "other_season", PAST, None),
    ("skip", "none", "other_season", PAST, None),
    ("absent", "none", "tagless", PAST, "applied-date marker without a result"),
    ("skip", "none", "tagless", PAST, "applied-date marker without a result"),
    # today cells: unreadable decision blocks even on today; locked/corrupt scheduler state & skip+delivery do not block today
    ("unreadable", "none", "none", TODAY, "unreadable or invalid decision record"),
    ("skip", "bluesky_posted", "none", TODAY, None),
])
def test_uncovered_rule2_table_cells(tmp_path, monkeypatch, dec_kind, sig, marker, day, expected_reason):
    overrides = {}
    if sig != "none":
        overrides[sig] = True if sig in ("bluesky_posted", "notification_sent", "delivery_attempted") else "syn"
    _plant_raw(tmp_path, day, **overrides)

    if dec_kind == "skip":
        write_decision(day, tmp_path, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET)
    elif dec_kind == "non_scoreable_other":
        write_decision(day, tmp_path, action="single", source="test", delivery_status="not_applicable", scoreable=False, now=RUN_ET)
    elif dec_kind == "unreadable":
        (tmp_path / day).mkdir(exist_ok=True)
        (tmp_path / day / "decision.json").write_text("{")

    if marker == "none":
        picks.save_streak(5, tmp_path, saver_available=False, season=2027)
    elif marker == "other_season":
        picks.save_streak(5, tmp_path, saver_available=False, applied_date=day, season=2026)
    elif marker == "tagless":
        picks.save_streak(5, tmp_path, saver_available=False, applied_date=day, season=None)

    sp = tmp_path / "streak.json"
    before = sp.read_bytes()
    res = _invoke(tmp_path, monkeypatch)
    assert res.exit_code == 0
    if expected_reason is None:
        assert not res.stderr
        rebuilt = json.loads(sp.read_text())
        assert (rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"]) == (0, True, 2027)
    else:
        assert res.stderr.splitlines() == [f"Season replay unavailable for {day} ({expected_reason})"]
        assert sp.read_bytes() == before
