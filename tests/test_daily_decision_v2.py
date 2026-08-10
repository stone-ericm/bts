"""bts_daily_decision_v2: state provenance on every final record.

Motivated by the 2026-08-09 boundary-shadow census: 31/44 v1 decision records
persisted no state (only MDP skips carried streak/saver), skip records lost
the executable second candidate, and nothing recorded which state stream fed
the decision. v2 persists (streak, saver, state_source, state_status,
allow_double, contest_source_date) on EVERY final action and the
second candidate on skips — making future censuses exact by construction.
Readers must keep accepting v1 (legacy records are on disk through 8/09).
"""
import json

from bts.daily_decision import (
    DECISION_SCHEMA, decision_path, is_scoreable_commit, load_decision, write_decision,
)


def _cand(bid=1, p=0.78, gpk=9):
    return {"batter_id": bid, "batter_name": f"B{bid}", "team": "NYM",
            "game_pk": gpk, "p_game_hit": p}


def test_writer_emits_v2_with_state_and_second_candidate(tmp_path):
    rec = write_decision(
        "2026-08-10", tmp_path, action="skip", source="mdp",
        primary=_cand(1), second_candidate=_cand(2, p=0.74, gpk=11),
        streak=7, saver_available=True,
        state_source="contest", state_status="fresh",
        allow_double=True, contest_source_date="2026-08-09",
        delivery_status="not_applicable", scoreable=False,
    )
    assert rec["schema_version"] == "bts_daily_decision_v2"
    assert rec["streak"] == 7 and rec["saver_available"] is True
    assert rec["state_source"] == "contest" and rec["state_status"] == "fresh"
    assert rec["allow_double"] is True
    assert rec["contest_source_date"] == "2026-08-09"
    assert rec["second_candidate"]["batter_id"] == 2
    assert rec["second_candidate"]["game_pk"] == 11
    assert set(rec["second_candidate"]) == {
        "batter_id", "batter_name", "team", "game_pk", "p_game_hit"}


def test_v2_roundtrip_through_load_decision(tmp_path):
    write_decision(
        "2026-08-10", tmp_path, action="double", source="mdp",
        primary=_cand(1), double_down=_cand(2, gpk=11),
        streak=2, saver_available=True,
        state_source="contest", state_status="lagged", allow_double=True,
        delivery_status="delivered", scoreable=True,
    )
    rec = load_decision("2026-08-10", tmp_path)
    assert rec is not None
    assert rec["streak"] == 2 and rec["state_status"] == "lagged"
    assert is_scoreable_commit("2026-08-10", tmp_path, None) is True


def test_load_decision_accepts_legacy_v1(tmp_path):
    path = decision_path("2026-07-28", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": "bts_daily_decision_v1", "date": "2026-07-28",
        "action": "skip", "source": "mdp",
        "primary": _cand(), "double_down": None,
        "streak": 8, "saver_available": True,
        "delivery_status": "not_applicable", "scoreable": False,
        "finalized_at": "2026-07-29T01:26:39Z",
    }))
    rec = load_decision("2026-07-28", tmp_path)
    assert rec is not None and rec["streak"] == 8
    assert is_scoreable_commit("2026-07-28", tmp_path, None) is False


def test_load_decision_rejects_unknown_schema(tmp_path):
    path = decision_path("2026-08-10", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": "bts_daily_decision_v3", "date": "2026-08-10",
        "action": "single", "scoreable": True,
    }))
    assert load_decision("2026-08-10", tmp_path) is None
