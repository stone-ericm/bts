"""Saver inference for the decision streak state (Phase 2c).

The MLB profile API can't observe the one-time, SEASON-SCOPED 10-15 mulligan, so the auto
contest file always has saver_available=None. Phase 1 fell back to the locally-tracked model
saver when the model and contest streaks agreed; Phase 2 retires that proxy and infers the
saver as: best_streak < 10 -> provably available (the account never reached the 10-15 zone,
so the saver was never consumable); else a STABLE ledger consumption at 10-15 -> consumed
(unavailable); else 'unknown' -> conservatively unavailable (confirming a saver survived
after reaching the zone needs complete season coverage, which is Phase 2b's job).
"""
import json

from bts.picks import save_streak
from bts.contest_state import load_decision_streak_state


def _write_fresh_contest(picks_dir, best_streak=9):
    state_dir = picks_dir / "account_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 8, "best_streak": best_streak,
        "source": "mlb_bts_profile", "source_date": "2026-06-08",
    }))  # no saver_available -> None, so the saver is inferred


def _write_ledger(picks_dir, predictions):
    """Write two identical fetch snapshots so every round is 'stable' (two-read)."""
    state_dir = picks_dir / "account_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"recorded_at": "2026-06-16T17:00:00Z", "active_streak": 12, "predictions": predictions}),
        json.dumps({"recorded_at": "2026-06-17T17:00:00Z", "active_streak": 12, "predictions": predictions}),
    ]
    (state_dir / "contest_ledger.jsonl").write_text("\n".join(lines))


def test_saver_available_when_best_streak_below_saver_zone(tmp_path):
    # never reached streak 10 this season -> saver provably available (no ledger needed)
    save_streak(8, tmp_path, saver_available=False)   # model saver is no longer consulted
    _write_fresh_contest(tmp_path, best_streak=9)
    state = load_decision_streak_state(tmp_path)
    assert state.saver_available is True


def test_saver_unavailable_when_reached_zone_and_ledger_shows_consumed(tmp_path):
    save_streak(8, tmp_path, saver_available=True)    # model saver is no longer consulted
    _write_fresh_contest(tmp_path, best_streak=12)
    _write_ledger(tmp_path, [
        {"roundId": 1, "result": "hit", "streak": 11, "roundPredictions": [{"playerId": 1, "result": "hit"}]},
        {"roundId": 2, "result": "not_hit", "streak": 11, "roundPredictions": [{"playerId": 2, "result": "not_hit"}]},
    ])  # not_hit at pre-streak 11, post 11 (no reset) -> the mulligan absorbed it -> consumed
    state = load_decision_streak_state(tmp_path)
    assert state.saver_available is False


def test_saver_unavailable_when_ledger_consumption_overrides_low_best_streak(tmp_path):
    # safety net: best_streak under-reports the peak (9) but the ledger holds a stable consuming
    # miss at 10-15 -> the evidence wins -> unavailable (guards a stale/wrong best_streak)
    save_streak(8, tmp_path, saver_available=True)
    _write_fresh_contest(tmp_path, best_streak=9)
    _write_ledger(tmp_path, [
        {"roundId": 1, "result": "hit", "streak": 11, "roundPredictions": [{"playerId": 1, "result": "hit"}]},
        {"roundId": 2, "result": "not_hit", "streak": 11, "roundPredictions": [{"playerId": 2, "result": "not_hit"}]},
    ])
    state = load_decision_streak_state(tmp_path)
    assert state.saver_available is False


def test_saver_unavailable_when_reached_zone_but_unconfirmed(tmp_path):
    # reached the zone (best_streak 12) with no ledger evidence -> can't confirm survival ->
    # conservatively unavailable. (Proxy gone: a model saver=True does NOT make it available.)
    save_streak(8, tmp_path, saver_available=True)
    _write_fresh_contest(tmp_path, best_streak=12)    # no contest_ledger.jsonl written
    state = load_decision_streak_state(tmp_path)
    assert state.saver_available is False
