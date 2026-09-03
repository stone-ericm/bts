"""best_streak trust contract on DecisionStreakState (2026-09-03, Codex r2 P0).

The tail policy's terminal stop ("season best can't be beaten") may only be
authorised by a TRUSTED best: an integer from the auto profile fetch or an
unexpired manual override, in the current season, with streak <= best <= 57.
Everything else is "untrusted"/"missing" and degrades to best = streak, which
keeps picking. A plausible-but-inflated best (e.g. 57 typed into a manual file)
would otherwise stop the account for the rest of the season.
"""
import json
from datetime import datetime, timedelta, timezone

from bts.contest_state import load_decision_streak_state

NOW = datetime(2026, 9, 3, 18, 0, tzinfo=timezone.utc)


def _auto(picks_dir, streak=0, best=18, source_date="2026-09-02", **extra):
    d = picks_dir / "account_state"; d.mkdir(exist_ok=True)
    body = {"schema_version": "bts_contest_streak_auto_v1", "active_streak": streak,
            "source": "mlb_bts_profile", "source_date": source_date,
            "recorded_at": "2026-09-03T17:30:04Z"}
    if best is not None:
        body["best_streak"] = best
    body.update(extra)
    (d / "contest_streak.json").write_text(json.dumps(body))


def _manual(picks_dir, streak=0, best=18, expires_at=None, source_date="2026-09-02"):
    d = picks_dir / "account_state"; d.mkdir(exist_ok=True)
    body = {"schema_version": "bts_contest_streak_manual_v2", "active_streak": streak,
            "source": "manual_cli", "source_date": source_date}
    if best is not None:
        body["best_streak"] = best
    if expires_at is not None:
        body["override_expires_at"] = expires_at
    (d / "contest_streak.manual.json").write_text(json.dumps(body))


def test_fresh_auto_best_is_trusted(tmp_path):
    _auto(tmp_path)
    st = load_decision_streak_state(tmp_path, now=NOW)
    assert (st.best_streak, st.best_status) == (18, "trusted")


def test_missing_best_is_missing(tmp_path):
    _auto(tmp_path, best=None)
    st = load_decision_streak_state(tmp_path, now=NOW)
    assert (st.best_streak, st.best_status) == (None, "missing")


def test_best_below_streak_is_untrusted(tmp_path):
    _auto(tmp_path, streak=5, best=3)
    st = load_decision_streak_state(tmp_path, now=NOW)
    assert (st.best_streak, st.best_status) == (3, "untrusted")


def test_best_above_target_is_untrusted(tmp_path):
    _auto(tmp_path, best=99)
    assert load_decision_streak_state(tmp_path, now=NOW).best_status == "untrusted"


def test_previous_season_observation_is_untrusted(tmp_path):
    _auto(tmp_path, source_date="2025-09-02")
    assert load_decision_streak_state(tmp_path, now=NOW).best_status == "untrusted"


def test_missing_source_date_is_untrusted(tmp_path):
    _auto(tmp_path, source_date=None)
    assert load_decision_streak_state(tmp_path, now=NOW).best_status == "untrusted"


def test_unexpired_manual_override_is_trusted(tmp_path):
    _manual(tmp_path, best=20, expires_at=(NOW + timedelta(hours=6)).isoformat())
    st = load_decision_streak_state(tmp_path, now=NOW)
    assert (st.best_streak, st.best_status) == (20, "trusted")


def test_expired_manual_used_as_last_resort_is_untrusted(tmp_path):
    _manual(tmp_path, best=20, expires_at=(NOW - timedelta(hours=6)).isoformat())
    st = load_decision_streak_state(tmp_path, now=NOW)
    assert st.source == "contest" and st.best_status == "untrusted"


def test_model_only_fallback_has_no_best(tmp_path):
    st = load_decision_streak_state(tmp_path, now=NOW)
    assert st.source == "model" and (st.best_streak, st.best_status) == (None, "missing")
