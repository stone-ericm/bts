"""Scheduler plumbing for the tail objective (2026-09-03).

The structured PolicyDecision produced once in strategy must reach: both
final-skip capture paths (normal cycle + fallback), the end-of-day skip writer,
the commit writer, the skip summary/log/DM/dashboard. Codex r2: a fallback
tail skip that lost its objective would be labelled a reach-57 skip and admitted
into the skip-policy shadow's pre-registered checkpoints.
"""
from types import SimpleNamespace

import pandas as pd

from bts.daily_decision import load_decision
from bts.scheduler import (
    SchedulerState, _capture_fallback_skip, _selection_decision_meta, _write_commit_decision,
    _write_endofday_skip, build_skip_summary, format_skip_dm,
)
from bts.strategy import PolicyDecision, SelectionResult


def _cand(bid=1, p=0.72, gpk=9):
    return {"batter_id": bid, "batter_name": f"B{bid}", "team": "NYM", "game_pk": gpk, "p_game_hit": p}


def _state(date="2026-09-19", **kw):
    base = dict(date=date, schedule_fetched_at="t", games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None,
                result_status=None, next_wakeup=None)
    base.update(kw)
    return SchedulerState(**base)


def _tail_skip_decision():
    return PolicyDecision(
        action="skip", policy_action="skip", source="mdp", objective="emax_season_best",
        streak=0, days_effective=9, best_supplied=18, best_status="trusted", effective_best=18,
        tail_sha256="t" * 64, degraded_reason=None,
        reason="season-best 18 can't be beaten with 9 days left (max reachable 18); no pick",
        pick_bar=None)


def _sel(decision, action="skip", pick_result=None):
    return SelectionResult(pick_result, action, "mdp", _cand(1), _cand(2, gpk=11), None,
                           streak=0, saver_available=False, state_source="contest",
                           state_status="fresh", allow_double=True,
                           contest_source_date="2026-09-18", decision=decision)


def test_decision_meta_extracts_objective_fields_and_tolerates_absence():
    meta = _selection_decision_meta(_sel(_tail_skip_decision()))
    assert meta == {"source": "mdp", "objective": "emax_season_best", "best_streak": 18,
                    "best_status": "trusted", "effective_best": 18,
                    "tail_policy_sha256": "t" * 64, "degraded_reason": None}
    assert _selection_decision_meta(_sel(None)) == {"source": "mdp"}
    assert _selection_decision_meta(None) == {}


def test_fallback_capture_carries_objective_fields():
    st = _state()
    _capture_fallback_skip(st, SimpleNamespace(selection=_sel(_tail_skip_decision())))
    c = st.final_skip_candidate
    assert c["objective"] == "emax_season_best" and c["effective_best"] == 18
    assert c["source"] == "mdp" and c["tail_policy_sha256"] == "t" * 64


def test_endofday_skip_persists_objective_fields(tmp_path):
    st = _state()
    st.final_skip_candidate = {
        "primary": _cand(), "double": _cand(2, gpk=11), "streak": 0, "saver_available": False,
        "state_source": "contest", "state_status": "fresh", "allow_double": True,
        "contest_source_date": "2026-09-18",
        "source": "mdp", "objective": "emax_season_best", "best_streak": 18,
        "best_status": "trusted", "effective_best": 18, "tail_policy_sha256": "t" * 64,
        "degraded_reason": None,
    }
    _write_endofday_skip(tmp_path, "2026-09-19", st)
    d = load_decision("2026-09-19", tmp_path)
    assert d["action"] == "skip" and d["source"] == "mdp"
    assert d["objective"] == "emax_season_best" and d["effective_best"] == 18
    assert d["best_status"] == "trusted" and d["tail_policy_sha256"] == "t" * 64


def test_endofday_skip_legacy_candidate_is_reach57_mdp(tmp_path):
    """A candidate captured before this change (same-day restart across the deploy)
    has no objective keys: it stays a reach-57 MDP skip, exactly as before."""
    st = _state(date="2026-09-03")
    st.final_skip_candidate = {"primary": _cand(), "double": None, "streak": 0,
                               "saver_available": False}
    _write_endofday_skip(tmp_path, "2026-09-03", st)
    d = load_decision("2026-09-03", tmp_path)
    assert d["source"] == "mdp" and d["objective"] == "reach57"   # explicit, never null (Codex r3)


def test_commit_decision_persists_objective_fields(tmp_path):
    st = _state(date="2026-09-04")
    _write_commit_decision(
        tmp_path, "2026-09-04", action="double", source="mdp", primary=_cand(),
        double_down=_cand(2, gpk=11), delivery_status="delivered", state=st,
        streak=0, saver_available=False, state_source="contest", state_status="fresh",
        allow_double=True, contest_source_date="2026-09-03",
        objective="emax_season_best", best_streak=18, best_status="trusted",
        effective_best=18, tail_policy_sha256="t" * 64, degraded_reason=None,
    )
    d = load_decision("2026-09-04", tmp_path)
    assert d["objective"] == "emax_season_best" and d["effective_best"] == 18
    assert d["tail_policy_sha256"] == "t" * 64 and d["scoreable"] is True


def test_skip_summary_and_dm_use_the_decision_reason():
    preds = pd.DataFrame([{"batter_name": "Steven Kwan", "team": "CLE", "p_game_hit": 0.738}])
    s = build_skip_summary(preds, 0, pick_bar=None, decision=_tail_skip_decision())
    assert s["reason"].startswith("season-best 18") and s["objective"] == "emax_season_best"
    dm = format_skip_dm("2026-09-19", s)
    assert "season-best 18" in dm and "9 days" in dm and "80%" not in dm


def test_skip_summary_without_decision_is_unchanged():
    preds = pd.DataFrame([{"batter_name": "Steven Kwan", "team": "CLE", "p_game_hit": 0.738}])
    s = build_skip_summary(preds, 8, pick_bar=0.796)
    assert "reason" not in s and s["bar"] == 0.796
    assert "streak-8 bar" in format_skip_dm("2026-08-01", s)


def test_run_single_check_skip_summary_comes_from_the_selection(tmp_path, monkeypatch):
    """No second state read for display: the summary's reason/bar come from the
    SelectionResult that made the decision (Codex r2)."""
    from contextlib import nullcontext
    import bts.scheduler as sch
    import bts.orchestrator as orch
    preds = pd.DataFrame([
        {"batter_name": "Steven Kwan", "team": "CLE", "p_game_hit": 0.738, "game_pk": 1},
        {"batter_name": "Luis Arraez", "team": "PHI", "p_game_hit": 0.73, "game_pk": 2},
    ])
    sel = _sel(_tail_skip_decision())
    monkeypatch.setattr(orch, "run_and_pick", lambda config, date, **k: (preds, sel, "local"))
    monkeypatch.setattr(sch, "count_new_confirmations", lambda *a, **k: 0)
    monkeypatch.setattr(sch, "heartbeat_watchdog", lambda *a, **k: nullcontext())
    def no_state_read(*a, **k):
        raise AssertionError("display must not re-read decision state")
    monkeypatch.setattr("bts.contest_state.load_decision_streak_state", no_state_read)
    config = {"orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")},
              "scheduler": {"heartbeat_stall_after_sec": 900}, "tiers": []}
    result = sch.run_single_check(date="2026-09-19", all_game_pks=[1, 2], confirmed_sides=set(),
                                  config=config, early_lock_gap=0.03)
    s = result["skip_summary"]
    assert s["best_batter"] == "Steven Kwan" and s["streak"] == 0
    assert s["reason"].startswith("season-best 18") and s.get("bar") is None


def test_render_skip_banner_shows_the_reason():
    from bts.web import render_skip_banner
    html = render_skip_banner({"best_batter": "Steven Kwan", "best_team": "CLE", "best_p": 0.738,
                               "streak": 0, "reason": "season-best 18 can't be beaten with 9 days left"})
    assert "season-best 18" in html and "below the pick bar" not in html
