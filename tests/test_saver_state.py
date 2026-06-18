"""Tests for the Streak Saver manual flag (bts.saver_state)."""
import json
from datetime import date

from bts.saver_state import load_saver_state, SaverState, season_for


def _write(picks_dir, obj):
    d = picks_dir / "account_state"
    d.mkdir(parents=True, exist_ok=True)
    (d / "saver_state.json").write_text(json.dumps(obj))


def test_missing_file_is_uninitialized(tmp_path):
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "uninitialized" and s.is_available is False


def test_valid_active_for_matching_season(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "active", "source": "manual_init"})
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "active" and s.is_available is True


def test_wrong_season_is_uninitialized_not_not_earned(tmp_path):
    _write(tmp_path, {"season": 2025, "state": "active"})
    s = load_saver_state(tmp_path, season=2026)
    assert s.state == "uninitialized"   # stale -> fail-closed, NOT not_earned
    assert s.season == 2025             # but the stale season is preserved (health distinguishes)


def test_invalid_state_or_bad_json_is_uninitialized(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "bogus"})
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"
    (tmp_path / "account_state" / "saver_state.json").write_text("{not json")
    assert load_saver_state(tmp_path, season=2026).state == "uninitialized"


def test_not_earned_and_used_not_available(tmp_path):
    _write(tmp_path, {"season": 2026, "state": "not_earned"})
    assert load_saver_state(tmp_path, season=2026).is_available is False
    _write(tmp_path, {"season": 2026, "state": "used"})
    assert load_saver_state(tmp_path, season=2026).is_available is False


def test_season_for_uses_source_date_year_else_now(tmp_path):
    assert season_for(date(2026, 6, 18), now_year=2027) == 2026
    assert season_for(None, now_year=2027) == 2027
