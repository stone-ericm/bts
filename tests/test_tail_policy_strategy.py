"""strategy.py integration of the tail policy (2026-09-03).

Codex r2 P0: the regime (reach-57 vs tail) must be resolved from (streak, days)
BEFORE any artifact is consulted, so no artifact failure can fall through to the
0.80 heuristic and recreate skip-forever. These tests pin that, the best-streak
trust contract, the operational clamps, the pick bar, and loader isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.simulate.tail_policy import (
    MAX_TAIL_DAYS, OBJECTIVE_REACH57, OBJECTIVE_TAIL, TARGET, TailPolicy, TailPolicyError,
    save_tail_policy, solve_emax_season_best, tail_manifest,
)
from bts.strategy import (
    SKIP_THRESHOLD, DecisionContext, PolicyDecision, SelectionResult, decide_action,
    effective_pick_bar, resolve_policy_decision, select_pick,
)

P_HIT, P_BOTH = 2641 / 3600, 1984 / 3600
D24, D9, D103 = "2026-09-04", "2026-09-19", "2026-06-17"   # SEASON_END_DATE 2026-09-28


def _tail_obj(sha="t" * 64) -> TailPolicy:
    sol = solve_emax_season_best(np.array([1.0]), np.array([P_HIT]), np.array([P_BOTH]))
    return TailPolicy(objective=OBJECTIVE_TAIL, policy_table=sol.policy, boundaries=[],
                      bin_freq=[1.0], bin_p_hit=[P_HIT], bin_p_both=[P_BOTH], target=TARGET,
                      max_days=MAX_TAIL_DAYS, base_policy_sha256="a" * 64,
                      manifest=tail_manifest(n_bins=1, hits=2641, both=1984, late_seed_days=3600),
                      built_at="2026-09-03T00:00:00Z", solver="solve_emax_season_best", sha256=sha)


def _base_table(action=0):
    return np.full((58, 181, 2, 5), action, dtype=np.int8)


def _mdp(tail="default", base=True, tail_error=None):
    d = {"policy_table": _base_table() if base else None,
         "boundaries": [0.796, 0.811, 0.825, 0.841] if base else None, "season_length": 180,
         # the decision path re-verifies the pairing: an injected dict must carry the base sha
         "base_sha256": "a" * 64 if base else None}
    d["tail"] = _tail_obj() if tail == "default" else tail
    if tail_error:
        d["tail_error"] = tail_error
    return d


def _ctx(p=0.72, date=D24, second_p=0.70, has_diff=True, allow_double=True, mdp="default",
         best_streak=18, best_status="trusted"):
    return DecisionContext(primary_p=p, second_p=second_p, has_diff_game=has_diff, date=date,
                           allow_double=allow_double, mdp=(_mdp() if mdp == "default" else mdp),
                           best_streak=best_streak, best_status=best_status)


# --- regime + table use --------------------------------------------------------

def test_tail_regime_uses_tail_table_and_records_provenance():
    dec = resolve_policy_decision(_ctx(), streak=0, saver=False)
    assert isinstance(dec, PolicyDecision)
    assert (dec.action, dec.source, dec.objective) == ("double", "mdp", OBJECTIVE_TAIL)
    assert dec.days_effective == 24 and dec.effective_best == 18
    assert dec.best_supplied == 18 and dec.best_status == "trusted"
    assert dec.tail_sha256 == "t" * 64 and dec.degraded_reason is None
    # The base table above says SKIP everywhere; the tail must not consult it.
    assert decide_action(_ctx(), 0, False) == ("double", "mdp")


def test_reach57_regime_is_untouched():
    mdp = _mdp(); mdp["policy_table"] = _base_table(action=1)   # base says single
    dec = resolve_policy_decision(_ctx(p=0.85, date=D103, mdp=mdp), streak=8, saver=False)
    assert (dec.action, dec.source, dec.objective) == ("single", "mdp", OBJECTIVE_REACH57)
    assert dec.effective_best is None and dec.tail_sha256 is None


def test_reach57_without_base_still_falls_to_heuristic():
    dec = resolve_policy_decision(_ctx(p=0.50, date=D103, mdp=None, best_streak=None,
                                       best_status=None), streak=10, saver=True)
    assert (dec.action, dec.source, dec.objective) == ("skip", "heuristic", OBJECTIVE_REACH57)


def test_stop_rule_skip_has_a_human_reason():
    dec = resolve_policy_decision(_ctx(date=D9), streak=0, saver=False)
    assert dec.action == "skip" and dec.objective == OBJECTIVE_TAIL
    assert "season-best 18" in dec.reason and "9 days" in dec.reason


# --- best-streak trust contract ----------------------------------------------

def test_untrusted_best_cannot_authorise_the_stop():
    dec = resolve_policy_decision(_ctx(date=D9, best_status="untrusted"), streak=0, saver=False)
    assert dec.action != "skip"
    assert dec.best_status == "untrusted" and dec.effective_best == 0 and dec.best_supplied == 18


def test_missing_best_means_best_equals_streak():
    dec = resolve_policy_decision(_ctx(date=D9, best_streak=None, best_status=None), streak=3, saver=False)
    assert dec.best_status == "missing" and dec.effective_best == 3 and dec.action != "skip"


def test_stale_low_best_is_clamped_up_to_streak():
    dec = resolve_policy_decision(_ctx(best_streak=2), streak=5, saver=False)
    assert dec.effective_best == 5


# --- forced fallback (Codex r2 P0) -------------------------------------------

def test_whole_mdp_absent_in_tail_never_reaches_the_heuristic():
    # p=0.72 < SKIP_THRESHOLD: the heuristic would skip; the tail contract forbids that.
    dec = resolve_policy_decision(_ctx(mdp=None), streak=0, saver=False)
    assert (dec.action, dec.source, dec.objective) == ("single", "mdp", OBJECTIVE_TAIL)
    assert dec.degraded_reason and "unavailable" in dec.degraded_reason


def test_whole_mdp_absent_still_honours_the_stop_rule():
    dec = resolve_policy_decision(_ctx(mdp=None, date=D9), streak=0, saver=False)
    assert dec.action == "skip" and dec.degraded_reason


def test_base_missing_makes_the_tail_unverifiable_and_forces_the_fallback():
    # Codex r3 P1: without the base sha the tail's pairing cannot be verified.
    dec = resolve_policy_decision(_ctx(mdp=_mdp(base=False)), streak=0, saver=False)
    assert dec.action == "single" and "unverifiable" in dec.degraded_reason


def test_tail_missing_carries_the_loader_error():
    dec = resolve_policy_decision(_ctx(mdp=_mdp(tail=None, tail_error="stop rule violated")), 0, False)
    assert dec.action == "single" and "stop rule violated" in dec.degraded_reason


def test_tail_lookup_exception_degrades_to_forced_action(monkeypatch):
    def boom(*a, **k):
        raise TailPolicyError("horizon")
    monkeypatch.setattr("bts.strategy.lookup_tail_action", boom)
    dec = resolve_policy_decision(_ctx(), streak=0, saver=False)
    assert dec.action == "single" and "horizon" in dec.degraded_reason


# --- operational clamps still apply --------------------------------------------

def test_tail_double_downgrades_without_a_different_game():
    dec = resolve_policy_decision(_ctx(has_diff=False, second_p=None), streak=0, saver=False)
    assert dec.action == "single" and dec.policy_action == "double"


def test_tail_double_downgrades_when_allow_double_false():
    dec = resolve_policy_decision(_ctx(allow_double=False), streak=0, saver=False)
    assert dec.action == "single" and dec.policy_action == "double"


# --- pick bar ------------------------------------------------------------------

def test_pick_bar_in_tail_is_zero_for_play_and_none_at_the_stop():
    assert effective_pick_bar(0, D24, False, mdp=_mdp(), best_streak=18, best_status="trusted") == 0.0
    assert effective_pick_bar(0, D9, False, mdp=_mdp(), best_streak=18, best_status="trusted") is None
    dec = resolve_policy_decision(_ctx(date=D9), streak=0, saver=False)
    assert dec.pick_bar is None


def test_pick_bar_reach57_without_base_is_the_heuristic_threshold():
    assert effective_pick_bar(8, D103, False, mdp=_mdp(base=False)) == SKIP_THRESHOLD


# --- select_pick end-to-end ----------------------------------------------------

def _preds(rows):
    defaults = {"batter_id": 100001, "team": "NYM", "lineup": 1, "pitcher_name": "P",
                "pitcher_id": 200001, "game_time": "2026-09-04T23:10:00Z", "p_hit_pa": 0.30, "flags": ""}
    return pd.DataFrame([{**defaults, "batter_name": f"B{i}", **r} for i, r in enumerate(rows)])


_AVAIL = {"abstract": "P", "detailed": "Pre-Game"}
_TWO = [{"p_game_hit": 0.72, "game_pk": 111}, {"p_game_hit": 0.70, "game_pk": 222}]


def test_select_pick_first_tail_day_is_a_double(tmp_path, monkeypatch):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: _mdp())
    res = select_pick(_preds(_TWO), D24, tmp_path, streak=0, saver_available=False,
                      best_streak=18, best_status="trusted",
                      game_statuses_detailed={111: _AVAIL, 222: _AVAIL})
    assert res.pick_result.daily.double_down is not None
    assert res.action == "double" and res.source == "mdp"
    assert res.decision.objective == OBJECTIVE_TAIL and res.decision.effective_best == 18


def test_select_pick_with_no_artifacts_in_tail_picks_a_single(tmp_path, monkeypatch):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: None)
    res = select_pick(_preds(_TWO), D24, tmp_path, streak=0, saver_available=False,
                      best_streak=18, best_status="trusted",
                      game_statuses_detailed={111: _AVAIL, 222: _AVAIL})
    assert res.pick_result is not None and res.pick_result.daily.double_down is None
    assert res.action == "single" and res.source == "mdp" and res.decision.degraded_reason


def test_select_pick_stop_rule_returns_skip_with_decision(tmp_path, monkeypatch):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: _mdp())
    res = select_pick(_preds(_TWO), D9, tmp_path, streak=0, saver_available=False,
                      best_streak=18, best_status="trusted",
                      game_statuses_detailed={111: _AVAIL, 222: _AVAIL})
    assert res.pick_result is None and res.action == "skip" and res.source == "mdp"
    assert res.decision.reason and res.decision.pick_bar is None


def test_selection_result_default_decision_is_none():
    assert SelectionResult(None, None, None, None, None, None).decision is None


# --- loader isolation ----------------------------------------------------------

def _write_base(path):
    np.savez_compressed(path, policy_table=_base_table(), boundaries=np.array([0.796, 0.811, 0.825, 0.841]),
                        season_length=np.array(180), optimal_p57=np.array(0.08))


def _write_tail(path, base_sha):
    t = _tail_obj(); t.base_policy_sha256 = base_sha
    save_tail_policy(t, path)


@pytest.fixture
def fresh_loader(monkeypatch, tmp_path):
    import bts.strategy as strat
    monkeypatch.setattr(strat, "_mdp_cache", None)
    base = tmp_path / "mdp_policy.npz"; tail = tmp_path / "mdp_tail_policy.npz"
    monkeypatch.setattr("bts.simulate.mdp.DEFAULT_POLICY_PATH", base)
    monkeypatch.setattr("bts.simulate.tail_policy.DEFAULT_TAIL_POLICY_PATH", tail)
    return strat, base, tail


def test_loader_binds_tail_to_base_sha(fresh_loader):
    from bts.simulate.tail_policy import sha256_file
    strat, base, tail = fresh_loader
    _write_base(base); _write_tail(tail, sha256_file(base))
    mdp = strat._load_mdp()
    assert mdp["policy_table"] is not None and mdp["tail"] is not None and mdp.get("tail_error") is None


def test_loader_rejects_tail_paired_with_another_base(fresh_loader):
    strat, base, tail = fresh_loader
    _write_base(base); _write_tail(tail, "0" * 64)
    mdp = strat._load_mdp()
    assert mdp["policy_table"] is not None and mdp["tail"] is None
    assert "mismatch" in mdp["tail_error"]


def test_loader_survives_a_corrupt_base_but_marks_the_tail_unverifiable(fresh_loader):
    strat, base, tail = fresh_loader
    base.write_bytes(b"not an npz"); _write_tail(tail, "0" * 64)
    mdp = strat._load_mdp()
    assert mdp is not None and mdp["policy_table"] is None and mdp["base_error"]
    assert mdp["tail"] is None and "unverifiable" in mdp["tail_error"]


def test_loader_returns_none_when_both_absent(fresh_loader):
    strat, base, tail = fresh_loader
    assert strat._load_mdp() is None
    assert strat._load_mdp() is None   # cached negative


def test_loader_base_only_keeps_legacy_shape(fresh_loader):
    strat, base, tail = fresh_loader
    _write_base(base)
    mdp = strat._load_mdp()
    assert mdp["tail"] is None and "missing" in mdp["tail_error"]
    assert list(mdp["boundaries"]) == pytest.approx([0.796, 0.811, 0.825, 0.841])
