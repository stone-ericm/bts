"""Unit tests for the scheduler's decision-record finalization helpers.

These target the pure decision-writing helpers (NOT the whole daemon loop) so
they stay fast and deterministic. The real control-flow wiring is exercised in
tests/test_scheduler_decision_record_integration.py.
"""
import json
from pathlib import Path
from bts.daily_decision import load_decision
# Helpers: build a SchedulerState + a SelectionResult; call the small writer helpers the
# implementation exposes (see Step 3). Tests target the pure decision-writing helpers, NOT the
# whole daemon loop, to stay fast and deterministic.
from bts.scheduler import (_write_commit_decision, _write_classification_decision,
                           _write_endofday_skip, FinalizationState)


def _cand(bid=1, p=0.78):
    return {"batter_id": bid, "batter_name": "X", "team": "NYM", "game_pk": 9, "p_game_hit": p}


def test_commit_writes_scoreable_pick(tmp_path):
    fs = FinalizationState()
    _write_commit_decision(tmp_path, "2026-06-20", action="single", source="mdp",
                           primary=_cand(), double_down=None, delivery_status="delivered", fs=fs)
    d = load_decision("2026-06-20", tmp_path)
    assert d["action"] == "single" and d["scoreable"] is True and d["delivery_status"] == "delivered"
    assert fs.committed_pick_written is True


def test_classification_writes_only_when_delivered(tmp_path):
    fs = FinalizationState()
    # delivered existing pick -> scoreable record
    _write_classification_decision(tmp_path, "2026-06-20", action="single", delivered=True, double_down=None, primary=_cand(), fs=fs)
    assert load_decision("2026-06-20", tmp_path)["scoreable"] is True and fs.committed_pick_written
    # NON-delivered (stale preview classified-locked) -> NO record, not committed
    fs2 = FinalizationState()
    _write_classification_decision(tmp_path, "2026-06-21", action="single", delivered=False, double_down=None, primary=_cand(), fs=fs2)
    assert load_decision("2026-06-21", tmp_path) is None and fs2.committed_pick_written is False


def test_endofday_skip_only_when_uncommitted_and_candidate(tmp_path):
    fs = FinalizationState()
    fs.final_skip_candidate = {"primary": _cand(), "streak": 10, "saver_available": True}
    _write_endofday_skip(tmp_path, "2026-06-20", fs)
    d = load_decision("2026-06-20", tmp_path)
    assert d["action"] == "skip" and d["source"] == "mdp" and d["scoreable"] is False and d["streak"] == 10
    # committed pick suppresses the skip
    fs2 = FinalizationState(); fs2.committed_pick_written = True
    fs2.final_skip_candidate = {"primary": _cand(), "streak": 10, "saver_available": True}
    _write_endofday_skip(tmp_path, "2026-06-22", fs2)
    assert load_decision("2026-06-22", tmp_path) is None
    # no candidate -> no record
    _write_endofday_skip(tmp_path, "2026-06-23", FinalizationState())
    assert load_decision("2026-06-23", tmp_path) is None
