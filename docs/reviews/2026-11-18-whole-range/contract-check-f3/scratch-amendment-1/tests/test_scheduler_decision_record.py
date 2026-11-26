"""Unit tests for the scheduler's decision-record finalization helpers.

These target the pure decision-writing helpers (NOT the whole daemon loop) so
they stay fast and deterministic. The real control-flow wiring is exercised in
tests/test_scheduler_decision_record_integration.py.
"""
import json
from pathlib import Path
from bts.daily_decision import load_decision
# Helpers: build a SchedulerState (the single finalization object — FinalizationState
# was dropped); call the small writer helpers the implementation exposes (see Step 3).
# Tests target the pure decision-writing helpers, NOT the whole daemon loop.
from bts.scheduler import (_write_commit_decision, _write_classification_decision,
                           _write_endofday_skip, SchedulerState)


def _cand(bid=1, p=0.78):
    return {"batter_id": bid, "batter_name": "X", "team": "NYM", "game_pk": 9, "p_game_hit": p}


def _state(date="2026-06-20", **kw):
    base = dict(
        date=date, schedule_fetched_at="t", games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None,
        result_status=None, next_wakeup=None,
    )
    base.update(kw)
    return SchedulerState(**base)


def test_commit_writes_scoreable_pick(tmp_path):
    st = _state()
    _write_commit_decision(tmp_path, "2026-06-20", action="single", source="mdp",
                           primary=_cand(), double_down=None, delivery_status="delivered", state=st)
    d = load_decision("2026-06-20", tmp_path)
    assert d["action"] == "single" and d["scoreable"] is True and d["delivery_status"] == "delivered"
    assert st.committed_pick_written is True


def test_classification_writes_only_when_delivered(tmp_path):
    st = _state(date="2026-06-20")
    # delivered existing pick -> scoreable record
    _write_classification_decision(tmp_path, "2026-06-20", action="single", delivered=True, double_down=None, primary=_cand(), state=st)
    assert load_decision("2026-06-20", tmp_path)["scoreable"] is True and st.committed_pick_written
    # NON-delivered (stale preview classified-locked) -> NO record, not committed
    st2 = _state(date="2026-06-21")
    _write_classification_decision(tmp_path, "2026-06-21", action="single", delivered=False, double_down=None, primary=_cand(), state=st2)
    assert load_decision("2026-06-21", tmp_path) is None and st2.committed_pick_written is False


def test_commit_flag_not_set_when_decision_write_fails(tmp_path, monkeypatch):
    """Codex r3 #2: write_decision is best-effort and returns None on failure.
    committed_pick_written must reflect the on-disk truth — setting it on a
    failed write suppresses the E3 missed-pick alert with no record backing it."""
    monkeypatch.setattr("bts.daily_decision.write_decision", lambda *a, **k: None)
    st = _state()
    _write_commit_decision(tmp_path, "2026-06-20", action="single", source="mdp",
                           primary=_cand(), double_down=None, delivery_status="delivered", state=st)
    assert st.committed_pick_written is False


def test_classification_attempted_writes_locked_unconfirmed(tmp_path):
    """Codex r3 #1: an undelivered pick with delivery_attempted=True classified
    as locked must finalize as a scoreable locked_unconfirmed commit."""
    st = _state(date="2026-06-22")
    _write_classification_decision(tmp_path, "2026-06-22", action="single",
                                   delivered=False, attempted=True,
                                   double_down=None, primary=_cand(), state=st)
    d = load_decision("2026-06-22", tmp_path)
    assert d is not None
    assert d["delivery_status"] == "locked_unconfirmed"
    assert d["scoreable"] is True
    assert st.committed_pick_written is True


def test_endofday_skip_only_when_uncommitted_and_candidate(tmp_path):
    st = _state(date="2026-06-20")
    st.final_skip_candidate = {"primary": _cand(), "streak": 10, "saver_available": True}
    _write_endofday_skip(tmp_path, "2026-06-20", st)
    d = load_decision("2026-06-20", tmp_path)
    assert d["action"] == "skip" and d["source"] == "mdp" and d["scoreable"] is False and d["streak"] == 10
    # committed pick suppresses the skip
    st2 = _state(date="2026-06-22", committed_pick_written=True)
    st2.final_skip_candidate = {"primary": _cand(), "streak": 10, "saver_available": True}
    _write_endofday_skip(tmp_path, "2026-06-22", st2)
    assert load_decision("2026-06-22", tmp_path) is None
    # no candidate -> no record
    _write_endofday_skip(tmp_path, "2026-06-23", _state(date="2026-06-23"))
    assert load_decision("2026-06-23", tmp_path) is None


def test_endofday_skip_does_not_clobber_scoreable_commit_on_disk(tmp_path):
    """Overwrite-guard (#2b): if a real scoreable commit already exists on disk
    (e.g. a crash lost committed_pick_written before the state save), the EOD
    skip must NOT overwrite it — even with a candidate present and the flag False."""
    st = _state(date="2026-06-24")
    # A genuine committed pick recorded earlier in the day.
    _write_commit_decision(tmp_path, "2026-06-24", action="single", source="mdp",
                           primary=_cand(), double_down=None, delivery_status="delivered", state=st)
    # Simulate the lost flag (rebuilt state after a crash before the state save).
    st.committed_pick_written = False
    st.final_skip_candidate = {"primary": _cand(), "streak": 10, "saver_available": True}
    _write_endofday_skip(tmp_path, "2026-06-24", st)
    d = load_decision("2026-06-24", tmp_path)
    assert d["action"] == "single" and d["scoreable"] is True  # commit preserved, not clobbered


# --- bts_daily_decision_v2: state provenance threading (2026-08-09) ---

def _cand2(bid=2, p=0.74):
    return {"batter_id": bid, "batter_name": "Y", "team": "PIT", "game_pk": 11, "p_game_hit": p}


def test_commit_decision_persists_selection_state(tmp_path):
    st = _state()
    _write_commit_decision(
        tmp_path, "2026-06-20", action="single", source="mdp",
        primary=_cand(), double_down=None, delivery_status="delivered", state=st,
        streak=6, saver_available=True,
        state_source="contest", state_status="fresh", allow_double=False,
        contest_source_date="2026-06-19",
    )
    d = load_decision("2026-06-20", tmp_path)
    assert d["streak"] == 6 and d["saver_available"] is True
    assert d["state_source"] == "contest" and d["state_status"] == "fresh"
    assert d["allow_double"] is False
    assert d["contest_source_date"] == "2026-06-19"


def test_endofday_skip_persists_second_candidate_and_state_meta(tmp_path):
    st = _state(date="2026-06-20")
    st.final_skip_candidate = {
        "primary": _cand(), "double": _cand2(),
        "streak": 10, "saver_available": True,
        "state_source": "contest", "state_status": "lagged",
        "allow_double": True, "contest_source_date": "2026-06-19",
    }
    _write_endofday_skip(tmp_path, "2026-06-20", st)
    d = load_decision("2026-06-20", tmp_path)
    assert d["action"] == "skip" and d["streak"] == 10
    assert d["second_candidate"]["batter_id"] == 2
    assert d["state_source"] == "contest" and d["state_status"] == "lagged"
    assert d["allow_double"] is True


def test_capture_fallback_skip_includes_double_and_state_meta(tmp_path):
    from types import SimpleNamespace
    from bts.scheduler import _capture_fallback_skip
    st = _state(date="2026-06-20")
    sel = SimpleNamespace(
        primary_candidate=_cand(), double_candidate=_cand2(),
        streak=9, saver_available=True,
        state_source="contest", state_status="stale",
        allow_double=True, contest_source_date="2026-06-18",
    )
    _capture_fallback_skip(st, SimpleNamespace(selection=sel))
    c = st.final_skip_candidate
    assert c["double"]["batter_id"] == 2
    assert c["state_source"] == "contest" and c["state_status"] == "stale"
    assert c["allow_double"] is True and c["contest_source_date"] == "2026-06-18"
