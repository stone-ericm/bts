"""Codex r3 code-review fixes (2026-09-03).

1. The PolicyDecision that chose a pick is persisted WITH the pick, so the
   cached-fallback and restart-recovery commit paths (selection=None) still write
   a v3 decision.json with objective / effective best / tail sha.
2. decision v3 objective is strict: the writer coerces a missing objective to an
   explicit "reach57" and refuses anything outside the enum; a v3 record whose
   objective is null/invalid is "unknown", never silently reach57 (v1/v2 keep the
   legacy default).
3. best-streak trust is decided by CONTENTS (schema + source + unexpired
   override), not by the filename, and a future source_date is untrusted.
4. A base policy that cannot be hashed makes the tail unverifiable: the pick path
   takes the forced fallback (degraded), the loader/health report it.
5. The tail loader reads the bytes once (sha == the bytes it parsed), bounds the
   size, requires the v1 horizon exactly, re-solves from the embedded rates and
   rejects any table that is not the exact policy, and validates the manifest.
6. The boundary census builds its quintile samples from reach-57 decisions only.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bts.daily_decision import decision_objective, is_reach57_mdp_skip, load_decision, write_decision
from bts.picks import load_pick, save_pick
from bts.simulate.tail_policy import (
    MAX_TAIL_DAYS, OBJECTIVE_TAIL, TARGET, TailPolicy, TailPolicyError, load_tail_policy,
    save_tail_policy, sha256_file, solve_emax_season_best, tail_manifest,
)
from bts.strategy import DecisionContext, PolicyDecision, SelectionResult, resolve_policy_decision, select_pick
import bts.scheduler as sch

P_HIT, P_BOTH = 2641 / 3600, 1984 / 3600
D24 = "2026-09-04"


def _tail_obj(sha="t" * 64, base_sha="a" * 64) -> TailPolicy:
    sol = solve_emax_season_best(np.array([1.0]), np.array([P_HIT]), np.array([P_BOTH]))
    return TailPolicy(objective=OBJECTIVE_TAIL, policy_table=sol.policy, boundaries=[],
                      bin_freq=[1.0], bin_p_hit=[P_HIT], bin_p_both=[P_BOTH], target=TARGET,
                      max_days=MAX_TAIL_DAYS, base_policy_sha256=base_sha,
                      manifest=tail_manifest(n_bins=1, hits=2641, both=1984, late_seed_days=3600),
                      built_at="2026-09-03T00:00:00Z", solver="solve_emax_season_best", sha256=sha)


def _mdp(tail=None, base=True):
    return {"policy_table": np.zeros((58, 181, 2, 5), np.int8) if base else None,
            "boundaries": [0.796, 0.811, 0.825, 0.841] if base else None, "season_length": 180,
            "base_sha256": "a" * 64 if base else None,
            "tail": tail if tail is not None else _tail_obj(), "tail_error": None, "base_error": None}


def _preds(rows):
    defaults = {"batter_id": 100001, "team": "NYM", "lineup": 1, "pitcher_name": "P",
                "pitcher_id": 200001, "game_time": "2026-09-04T23:10:00Z", "p_hit_pa": 0.30, "flags": ""}
    return pd.DataFrame([{**defaults, "batter_name": f"B{i}", **r} for i, r in enumerate(rows)])


_AVAIL = {"abstract": "P", "detailed": "Pre-Game"}
_TWO = [{"p_game_hit": 0.72, "game_pk": 111}, {"p_game_hit": 0.70, "game_pk": 222}]


# --- 1. decision persisted with the pick ---------------------------------------

def test_select_pick_persists_the_decision_on_the_daily_pick(tmp_path, monkeypatch):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: _mdp())
    res = select_pick(_preds(_TWO), D24, tmp_path, streak=0, saver_available=False,
                      best_streak=18, best_status="trusted",
                      game_statuses_detailed={111: _AVAIL, 222: _AVAIL})
    daily = res.pick_result.daily
    assert daily.policy_decision == asdict(res.decision)
    assert daily.policy_decision["objective"] == OBJECTIVE_TAIL
    assert daily.policy_decision["effective_best"] == 18 and daily.policy_decision["tail_sha256"] == "t" * 64
    save_pick(daily, tmp_path)
    assert load_pick(D24, tmp_path).policy_decision == daily.policy_decision


def test_decision_meta_falls_back_to_the_daily_pick_when_selection_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: _mdp())
    daily = select_pick(_preds(_TWO), D24, tmp_path, streak=0, saver_available=False,
                        best_streak=18, best_status="trusted",
                        game_statuses_detailed={111: _AVAIL, 222: _AVAIL}).pick_result.daily
    meta = sch._selection_decision_meta(None, daily)
    assert meta["source"] == "mdp" and meta["objective"] == OBJECTIVE_TAIL
    assert meta["effective_best"] == 18 and meta["tail_policy_sha256"] == "t" * 64
    assert sch._selection_decision_meta(None, None) == {}


def _state(date=D24):
    return sch.SchedulerState(date=date, schedule_fetched_at="t", games=[], confirmed_game_pks=[],
                              runs_completed=[], pick_locked=False, pick_locked_at=None,
                              result_status=None, next_wakeup=None)


def _tail_daily(tmp_path, monkeypatch, delivered=False):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: _mdp())
    daily = select_pick(_preds(_TWO), D24, tmp_path, streak=0, saver_available=False,
                        best_streak=18, best_status="trusted",
                        game_statuses_detailed={111: _AVAIL, 222: _AVAIL}).pick_result.daily
    if delivered:
        daily.bluesky_posted = True
    return daily


def test_cached_fallback_commit_keeps_the_objective(tmp_path, monkeypatch):
    """The safety-net route: refresh failed (selection=None), cached tail double is
    delivered. Codex r3 P1: this wrote source=unknown, objective=None."""
    daily = _tail_daily(tmp_path, monkeypatch)
    st = _state()
    sch._commit_decision_for_pick(tmp_path, D24, daily, selection=None,
                                  delivery_status="private_locked", state=st)
    d = load_decision(D24, tmp_path)
    assert d["action"] == "double" and d["source"] == "mdp"
    assert d["objective"] == OBJECTIVE_TAIL and d["effective_best"] == 18
    assert d["best_status"] == "trusted" and d["tail_policy_sha256"] == "t" * 64


def test_classification_recovery_keeps_the_objective(tmp_path, monkeypatch):
    """Restart after delivery but before the decision write: the classification
    recovery must read the decision persisted on the pick, not write 'unknown'."""
    daily = _tail_daily(tmp_path, monkeypatch, delivered=True)
    st = _state()
    sch._write_classification_decision(tmp_path, D24, action="double", delivered=True,
                                       primary=sch._row_from_daily(daily.pick),
                                       double_down=sch._row_from_daily(daily.double_down),
                                       state=st, daily=daily)
    d = load_decision(D24, tmp_path)
    assert d["scoreable"] is True and d["source"] == "mdp"
    assert d["objective"] == OBJECTIVE_TAIL and d["effective_best"] == 18


# --- 2. strict v3 objective ---------------------------------------------------

def _cand(bid=1, p=0.72, gpk=9):
    return {"batter_id": bid, "batter_name": f"B{bid}", "team": "NYM", "game_pk": gpk, "p_game_hit": p}


def test_v3_writer_coerces_missing_objective_to_explicit_reach57(tmp_path):
    rec = write_decision("2026-09-03", tmp_path, action="skip", source="mdp", primary=_cand(),
                         delivery_status="not_applicable", scoreable=False)
    assert rec["objective"] == "reach57"
    assert is_reach57_mdp_skip(load_decision("2026-09-03", tmp_path))


def test_v3_writer_refuses_an_unknown_objective(tmp_path):
    assert write_decision("2026-09-03", tmp_path, action="skip", source="mdp", primary=_cand(),
                          delivery_status="not_applicable", scoreable=False, objective="bogus") is None
    assert load_decision("2026-09-03", tmp_path) is None


def test_v3_null_objective_is_unknown_not_reach57(tmp_path):
    from bts.daily_decision import decision_path
    p = decision_path("2026-09-19", tmp_path); p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"schema_version": "bts_daily_decision_v3", "date": "2026-09-19",
                             "action": "skip", "source": "mdp", "scoreable": False, "objective": None}))
    rec = load_decision("2026-09-19", tmp_path)
    assert decision_objective(rec) == "unknown" and is_reach57_mdp_skip(rec) is False
    assert decision_objective({"schema_version": "bts_daily_decision_v2", "action": "skip"}) == "reach57"


# --- 3. trust by contents -----------------------------------------------------

NOW = datetime(2026, 9, 19, 18, 0, tzinfo=timezone.utc)


def _write(picks_dir, name, body):
    d = picks_dir / "account_state"; d.mkdir(exist_ok=True)
    (d / name).write_text(json.dumps(body))


def _auto_body(**kw):
    body = {"schema_version": "bts_contest_streak_auto_v1", "active_streak": 0, "best_streak": 18,
            "source": "mlb_bts_profile", "source_date": "2026-09-18", "recorded_at": "2026-09-19T17:30:04Z"}
    body.update(kw); return body


def test_future_source_date_is_untrusted(tmp_path):
    from bts.contest_state import load_decision_streak_state
    _write(tmp_path, "contest_streak.json", _auto_body(source_date="2026-12-31"))
    assert load_decision_streak_state(tmp_path, now=NOW).best_status == "untrusted"


def test_manual_contents_under_the_auto_filename_are_not_auto_trusted(tmp_path):
    from bts.contest_state import load_decision_streak_state
    _write(tmp_path, "contest_streak.json", {
        "schema_version": "bts_contest_streak_manual_v2", "active_streak": 0, "best_streak": 18,
        "source": "manual_cli", "source_date": "2026-09-18",
        "override_expires_at": (NOW - timedelta(days=18)).isoformat()})
    assert load_decision_streak_state(tmp_path, now=NOW).best_status == "untrusted"


def test_unknown_schema_or_source_is_untrusted(tmp_path):
    from bts.contest_state import load_decision_streak_state
    _write(tmp_path, "contest_streak.json", _auto_body(schema_version="something_else"))
    assert load_decision_streak_state(tmp_path, now=NOW).best_status == "untrusted"
    _write(tmp_path, "contest_streak.json", _auto_body(source="copied_by_hand"))
    assert load_decision_streak_state(tmp_path, now=NOW).best_status == "untrusted"


def test_genuine_auto_observation_is_trusted_at_the_stop(tmp_path):
    from bts.contest_state import load_decision_streak_state
    _write(tmp_path, "contest_streak.json", _auto_body())
    st = load_decision_streak_state(tmp_path, now=NOW)
    assert (st.best_streak, st.best_status) == (18, "trusted")


# --- 4/5. unverifiable base + strict loader --------------------------------------

def _write_base(path):
    np.savez_compressed(path, policy_table=np.zeros((58, 181, 2, 5), np.int8),
                        boundaries=np.array([0.796, 0.811, 0.825, 0.841]),
                        season_length=np.array(180), optimal_p57=np.array(0.08))


@pytest.fixture
def fresh_loader(monkeypatch, tmp_path):
    import bts.strategy as strat
    monkeypatch.setattr(strat, "_mdp_cache", None)
    base = tmp_path / "mdp_policy.npz"; tail = tmp_path / "mdp_tail_policy.npz"
    monkeypatch.setattr("bts.simulate.mdp.DEFAULT_POLICY_PATH", base)
    monkeypatch.setattr("bts.simulate.tail_policy.DEFAULT_TAIL_POLICY_PATH", tail)
    return strat, base, tail


def test_corrupt_base_makes_the_tail_unverifiable(fresh_loader):
    strat, base, tail = fresh_loader
    base.write_bytes(b"not an npz"); save_tail_policy(_tail_obj(base_sha="0" * 64), tail)
    mdp = strat._load_mdp()
    assert mdp["policy_table"] is None and mdp["base_error"]
    assert mdp["tail"] is None and "unverifiable" in mdp["tail_error"]
    dec = resolve_policy_decision(DecisionContext(primary_p=0.72, second_p=0.70, has_diff_game=True,
                                                  date=D24, allow_double=True, mdp=mdp,
                                                  best_streak=18, best_status="trusted"), 0, False)
    assert dec.action == "single" and "unverifiable" in dec.degraded_reason


def test_loader_hashes_exactly_the_bytes_it_parsed(fresh_loader):
    strat, base, tail = fresh_loader
    _write_base(base); save_tail_policy(_tail_obj(base_sha=sha256_file(base)), tail)
    mdp = strat._load_mdp()
    assert mdp["tail"].sha256 == sha256_file(tail) and mdp["base_sha256"] == sha256_file(base)


def test_loader_rejects_a_table_that_is_not_the_exact_policy(tmp_path):
    t = _tail_obj(); table = t.policy_table.copy()
    table[0, 18, 24, 0, 0] = 1            # double -> single: still non-skip, so the partition passes
    t.policy_table = table
    path = tmp_path / "tail.npz"; save_tail_policy(t, path, validate=False)
    with pytest.raises(TailPolicyError, match="exact"):
        load_tail_policy(path)


def test_loader_requires_the_v1_horizon_exactly(tmp_path):
    t = _tail_obj(); t.max_days = MAX_TAIL_DAYS + 1
    sol = solve_emax_season_best(np.array([1.0]), np.array([P_HIT]), np.array([P_BOTH]), max_days=MAX_TAIL_DAYS + 1)
    t.policy_table = sol.policy
    path = tmp_path / "tail.npz"; save_tail_policy(t, path, validate=False)
    with pytest.raises(TailPolicyError, match="max_days"):
        load_tail_policy(path)


def test_loader_validates_the_manifest(tmp_path):
    t = _tail_obj(); t.manifest = {}
    path = tmp_path / "tail.npz"; save_tail_policy(t, path, validate=False)
    with pytest.raises(TailPolicyError, match="manifest"):
        load_tail_policy(path)
    t = _tail_obj(); t.manifest["hits"] = 2600      # inconsistent with p_hit
    save_tail_policy(t, path, validate=False)
    with pytest.raises(TailPolicyError, match="manifest"):
        load_tail_policy(path)


def test_loader_bounds_the_artifact_size(tmp_path):
    path = tmp_path / "tail.npz"; path.write_bytes(b"0" * (3 * 1024 * 1024))
    with pytest.raises(TailPolicyError, match="too large"):
        load_tail_policy(path)


def test_health_reports_unverifiable_tail_when_base_is_missing(tmp_path):
    from bts.health.tail_policy import check
    from datetime import date
    tail = tmp_path / "mdp_tail_policy.npz"; save_tail_policy(_tail_obj(base_sha="0" * 64), tail)
    alerts = check(base_path=tmp_path / "nope.npz", tail_path=tail, today=date(2026, 9, 4))
    assert any("unverifiable" in a.message for a in alerts)
    assert all(a.level == "CRITICAL" for a in alerts)


# --- 6. census samples ----------------------------------------------------------

def test_census_boundary_samples_exclude_tail_decisions():
    from scripts.audit.boundary_shadow_census import _boundary_sample_decisions
    reach = [("p1", {"schema_version": "bts_daily_decision_v2", "date": "2026-08-01", "primary": {"p_game_hit": 0.70}}),
             ("p2", {"schema_version": "bts_daily_decision_v3", "date": "2026-08-02", "objective": "reach57", "primary": {"p_game_hit": 0.71}})]
    tail = [("p3", {"schema_version": "bts_daily_decision_v3", "date": "2026-09-04", "objective": "emax_season_best", "primary": {"p_game_hit": 0.90}}),
            ("p4", {"schema_version": "bts_daily_decision_v3", "date": "2026-09-05", "objective": None, "primary": {"p_game_hit": 0.91}})]
    kept = _boundary_sample_decisions(reach + tail)
    assert [p for p, _ in kept] == ["p1", "p2"]
