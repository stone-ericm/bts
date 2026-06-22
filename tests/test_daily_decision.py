# tests/test_daily_decision.py
import json
from pathlib import Path
from datetime import datetime, timezone
from bts.daily_decision import write_decision, load_decision, decision_path, DECISION_SCHEMA, is_scoreable_commit
from bts.picks import Pick, DailyPick


def _pick():
    return Pick(
        batter_name="Jacob Wilson", batter_id=700363, team="ATH", lineup_position=1,
        pitcher_name="Jose Suarez", pitcher_id=660761, p_game_hit=0.83, flags=[],
        projected_lineup=False, game_pk=778899, game_time="2026-04-01T23:10:00Z",
    )


def _delivered_daily():
    return DailyPick(
        date="2026-04-01", run_time="2026-04-01T15:00:00+00:00",
        pick=_pick(), double_down=None, runner_up=None, bluesky_posted=True,
    )


def _undelivered_daily():
    return DailyPick(
        date="2026-04-01", run_time="2026-04-01T15:00:00+00:00",
        pick=_pick(), double_down=None, runner_up=None, bluesky_posted=False,
    )

def _cand(bid=1, p=0.78):
    return {"batter_id": bid, "batter_name": "X", "team": "NYM", "game_pk": 9, "p_game_hit": p}

def test_write_and_load_roundtrip(tmp_path):
    rec = write_decision("2026-06-20", tmp_path, action="skip", source="mdp",
                         primary=_cand(), streak=10, saver_available=True,
                         delivery_status="not_applicable", scoreable=False,
                         now=datetime(2026, 6, 20, tzinfo=timezone.utc))
    assert rec is not None
    assert decision_path("2026-06-20", tmp_path).exists()
    loaded = load_decision("2026-06-20", tmp_path)
    assert loaded["schema_version"] == DECISION_SCHEMA
    assert loaded["action"] == "skip" and loaded["source"] == "mdp"
    assert loaded["primary"]["batter_id"] == 1 and loaded["primary"]["p_game_hit"] == 0.78
    assert loaded["streak"] == 10 and loaded["saver_available"] is True
    assert loaded["scoreable"] is False

def test_load_missing_is_none(tmp_path):
    assert load_decision("2026-01-01", tmp_path) is None

def test_write_is_best_effort_never_raises():
    # an unwritable picks_dir must not raise (best-effort)
    assert write_decision("2026-06-20", "/proc/cannot/write/here", action="skip", source="mdp",
                          delivery_status="not_applicable", scoreable=False) is None

def test_load_malformed_json_is_none(tmp_path):
    p = decision_path("2026-06-20", tmp_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json")
    assert load_decision("2026-06-20", tmp_path) is None

def test_load_rejects_wrong_shape(tmp_path):
    p = decision_path("2026-06-20", tmp_path); p.parent.mkdir(parents=True, exist_ok=True)
    for bad in ("[]", "\"x\"", "{\"scoreable\": true}"):   # list, string, dict missing schema_version
        p.write_text(bad)
        assert load_decision("2026-06-20", tmp_path) is None

def test_double_carries_both_slots(tmp_path):
    write_decision("2026-06-20", tmp_path, action="double", source="mdp",
                   primary=_cand(1), double_down=_cand(2), delivery_status="delivered", scoreable=True)
    loaded = load_decision("2026-06-20", tmp_path)
    assert loaded["action"] == "double"
    assert loaded["double_down"]["batter_id"] == 2


def test_is_scoreable_commit(tmp_path):
    # decision record with scoreable=False -> not a commit (ignores daily delivery state)
    write_decision("2026-06-20", tmp_path, action="skip", source="mdp",
                   delivery_status="not_applicable", scoreable=False)
    assert is_scoreable_commit("2026-06-20", tmp_path, _undelivered_daily()) is False

    # decision record with scoreable=True -> commit (even if daily looks undelivered)
    write_decision("2026-06-21", tmp_path, action="single", source="mdp",
                   delivery_status="delivered", scoreable=True)
    assert is_scoreable_commit("2026-06-21", tmp_path, _undelivered_daily()) is True

    # no decision record -> fall back to pick_was_delivered
    assert is_scoreable_commit("2026-06-22", tmp_path, _delivered_daily()) is True
    assert is_scoreable_commit("2026-06-22", tmp_path, _undelivered_daily()) is False
