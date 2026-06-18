"""The live saver comes ONLY from saver_state.json (replaces the unsound infer_saver)."""
import json

from bts.picks import save_streak
from bts.saver_state import transition_saver_state
from bts.contest_state import load_decision_streak_state


def _fresh_contest(picks_dir, best_streak=10):
    d = picks_dir / "account_state"; d.mkdir(parents=True, exist_ok=True)
    (d / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": 10,
        "best_streak": best_streak, "source": "mlb_bts_profile", "source_date": "2026-06-18"}))


def test_active_flag_makes_saver_available(tmp_path):
    save_streak(10, tmp_path, saver_available=False)   # model saver is irrelevant now
    _fresh_contest(tmp_path)
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active",
                           season=2026, source="t")
    assert load_decision_streak_state(tmp_path).saver_available is True


def test_used_or_uninitialized_flag_means_unavailable(tmp_path):
    save_streak(10, tmp_path, saver_available=True)
    _fresh_contest(tmp_path)
    # no saver_state.json -> uninitialized -> unavailable
    assert load_decision_streak_state(tmp_path).saver_available is False
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="used",
                           season=2026, source="t")
    assert load_decision_streak_state(tmp_path).saver_available is False


def test_model_only_fallback_reads_saver_state_not_streak_json(tmp_path):
    # no contest observation at all -> model-only path; the saver still comes from saver_state.json
    save_streak(10, tmp_path, saver_available=True)    # streak.json says saver True...
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="used",
                           season=2026, source="t")     # ...but the flag says used
    assert load_decision_streak_state(tmp_path).saver_available is False
