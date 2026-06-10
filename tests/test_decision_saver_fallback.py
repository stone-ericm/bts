"""Fresh contest state must not silently disable the streak saver (audit D3).

The MLB profile API can't observe the mulligan, so the auto contest file always
has saver_available=None. Pinning that to False makes the saver-aware (more
aggressive, higher-EV) MDP policy line at streak 10-15 unreachable on the normal
production path. Fall back to the locally-tracked model saver instead.
"""
import json

from bts.picks import save_streak
from bts.contest_state import load_decision_streak_state


def _write_fresh_contest(picks_dir):
    state_dir = picks_dir / "account_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 12, "best_streak": 12,
        "source": "mlb_bts_profile", "source_date": "2026-06-08",
    }))  # no saver_available -> None


def test_fresh_contest_saver_falls_back_to_model(tmp_path):
    save_streak(12, tmp_path, saver_available=True)   # model: saver available
    _write_fresh_contest(tmp_path)                    # contest: saver unobservable
    # no resolved pick files -> contest is fresh

    state = load_decision_streak_state(tmp_path)

    assert state.status == "fresh"
    assert state.saver_available is True, "saver pinned False; saver-aware MDP line unreachable"


def test_fresh_contest_saver_false_when_model_consumed(tmp_path):
    save_streak(12, tmp_path, saver_available=False)  # model: saver already used
    _write_fresh_contest(tmp_path)

    state = load_decision_streak_state(tmp_path)

    assert state.status == "fresh"
    assert state.saver_available is False
