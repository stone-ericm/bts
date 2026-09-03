"""Tail policy: exact E[season-best] once 57 is unreachable (2026-09-03).

The reach-57 solver values every action at exactly 0 once ``streak + 2*days < 57``,
so argmax fell to index 0 = skip and production idled for the rest of the season.
The tail policy replaces that regime with the exact expected-season-best objective
on the augmented state (streak, best, days, saver) and an EXPLICIT stop rule: skip
iff no outcome can raise the season best. Everything below is pure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bts.simulate.tail_policy import (
    DEFAULT_TAIL_POLICY_PATH,
    MAX_TAIL_DAYS,
    OBJECTIVE_REACH57,
    OBJECTIVE_TAIL,
    TAIL_SCHEMA,
    TARGET,
    TailPolicy,
    TailPolicyError,
    can_beat_best,
    effective_days,
    forced_tail_action,
    load_tail_policy,
    lookup_tail_action,
    mdp_objective,
    save_tail_policy,
    solve_emax_season_best,
    tail_manifest,
)

# Production-shaped late-season one-bin rates (24-seed estimated-PA profiles,
# first different-game candidate as the double): 2641/3600 and 1984/3600.
P_HIT = 2641 / 3600
P_BOTH = 1984 / 3600


# --- objective switch ---------------------------------------------------------

class TestObjectiveSwitch:
    def test_effective_days_clamps_to_season_length_and_zero(self):
        assert effective_days(25) == 25
        assert effective_days(500, season_length=180) == 180
        assert effective_days(-3) == 0

    @pytest.mark.parametrize("streak,days,expected", [
        (0, 29, OBJECTIVE_REACH57),   # 0 + 58 >= 57: still reachable
        (0, 28, OBJECTIVE_TAIL),      # 0 + 56 < 57: the 9/03 state class
        (7, 25, OBJECTIVE_REACH57),   # 7 + 50 == 57: equality is reachable
        (6, 25, OBJECTIVE_TAIL),      # the live 9/02->9/03 flip
        (56, 1, OBJECTIVE_REACH57),
    ])
    def test_reachability_boundary(self, streak, days, expected):
        assert mdp_objective(streak, days) == expected

    def test_no_days_or_finished_streak_stay_with_base_policy(self):
        # lookup_action already returns skip for these; the base table owns them.
        assert mdp_objective(0, 0) == OBJECTIVE_REACH57
        assert mdp_objective(57, 10) == OBJECTIVE_REACH57

    def test_days_are_clamped_before_the_predicate(self):
        # 500 raw days clamps to the 180-day horizon, which is reachable.
        assert mdp_objective(0, 500, season_length=180) == OBJECTIVE_REACH57
        # A short synthetic season: 0 + 2*10 < 57 -> tail.
        assert mdp_objective(0, 500, season_length=10) == OBJECTIVE_TAIL


# --- stop rule + forced fallback ---------------------------------------------

class TestStopRule:
    def test_can_beat_best_is_capped_at_target(self):
        assert can_beat_best(0, 18, 10)          # reach 19 needs 10 doubles: possible
        assert not can_beat_best(0, 18, 9)       # 0 + 18 <= 18
        assert not can_beat_best(56, 57, 1)      # min(57, 58) <= 57: cap matters
        assert can_beat_best(0, 0, 1)

    def test_forced_action_is_skip_only_at_the_stop_rule(self):
        assert forced_tail_action(0, 18, 9) == "skip"
        assert forced_tail_action(0, 18, 10) == "single"
        assert forced_tail_action(56, 57, 1) == "skip"
        assert forced_tail_action(0, 0, 1) == "single"
        assert forced_tail_action(0, 18, 0) == "skip"


# --- solver -------------------------------------------------------------------

def _solve_one_bin(p_hit=P_HIT, p_both=P_BOTH, target=TARGET, max_days=MAX_TAIL_DAYS):
    return solve_emax_season_best(
        np.array([1.0]), np.array([p_hit]), np.array([p_both]),
        target=target, max_days=max_days,
    )


class TestSolver:
    def test_hand_computed_values_on_tiny_instance(self):
        """target=5, p=0.6, pb=0.4, one bin, saver off; V[s, m, d] = E[season best].

        d=1: V(0,0,1)=max(0, .6*1, .4*2)=0.8 (double)
             V(1,1,1)=max(1, .6*2+.4*1=1.6, .4*3+.6*1=1.8)=1.8 (double)
             V(2,2,1)=max(2, 2.6, 2.8)=2.8 (double)
        d=2: V(0,0,2)=max(skip=0.8, single=.6*1.8+.4*.8=1.40, double=.4*2.8+.6*.8=1.60)
             =1.60 (double)
        stop: (s=0, m=4, d=2): 0+4 <= 4 -> skip, value 4.
        """
        sol = _solve_one_bin(0.6, 0.4, target=5, max_days=3)
        V, pol = sol.value, sol.policy
        assert V[0, 0, 1, 0] == pytest.approx(0.8, abs=1e-12)
        assert V[1, 1, 1, 0] == pytest.approx(1.8, abs=1e-12)
        assert V[2, 2, 1, 0] == pytest.approx(2.8, abs=1e-12)
        assert V[0, 0, 2, 0] == pytest.approx(1.6, abs=1e-12)
        assert pol[0, 0, 2, 0, 0] == 2 and pol[2, 2, 1, 0, 0] == 2
        assert V[0, 4, 2, 0] == pytest.approx(4.0)
        assert pol[0, 4, 2, 0, 0] == 0

    def test_values_match_validated_audit_solver(self):
        """The port must reproduce scripts/audit/skip_threshold_resolve.solve_emax
        (validated 2026-06-29) value-for-value. Policies may differ only on exact
        ties (the audit solver's argmax is skip-first; ours is play-first)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "skip_threshold_resolve", Path("scripts/audit/skip_threshold_resolve.py"))
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        freq = np.array([0.4, 0.6]); ph = np.array([0.70, 0.80]); pb = np.array([0.50, 0.62])
        ref_value, _ = ref.solve_emax(freq, ph, pb, target=57, D=12)
        sol = solve_emax_season_best(freq, ph, pb, target=57, max_days=12)
        # ref returns the value ENTERING day D+1 (i.e. after D days), marginalised over bins.
        np.testing.assert_allclose(sol.value[:, :, 12, :], ref_value, atol=1e-10)

    def test_stop_rule_partition_is_exact_over_every_consulted_cell(self):
        """Over every cell the runtime can consult (s <= m <= 57, 1 <= d <= 28,
        s + 2d < 57, both saver states): skip iff min(57, s + 2d) <= m. Codex r2
        confirmed independently that no strategic skip exists in this region for
        the production rates (smallest play-over-skip margin 5.7e-8)."""
        sol = _solve_one_bin()
        pol = sol.policy[:, :, :, :, 0]
        shape = (TARGET + 1, TARGET + 1, MAX_TAIL_DAYS + 1)
        s = np.broadcast_to(np.arange(TARGET + 1)[:, None, None], shape)
        m = np.broadcast_to(np.arange(TARGET + 1)[None, :, None], shape)
        d = np.broadcast_to(np.arange(MAX_TAIL_DAYS + 1)[None, None, :], shape)
        valid = (s <= m) & (d >= 1) & (s + 2 * d < TARGET)
        stop = np.minimum(TARGET, s + 2 * d) <= m
        assert valid.sum() == 31_234   # per saver state; Codex r2 counted 62,468 over both
        for sv in (0, 1):
            is_skip = pol[:, :, :, sv] == 0
            assert np.array_equal(is_skip[valid], stop[valid])

    def test_never_skips_at_streak_zero_while_best_is_beatable(self):
        sol = _solve_one_bin()
        for d in range(1, MAX_TAIL_DAYS + 1):
            for best in range(0, 2 * d):           # best < 2d  <=> beatable from 0
                assert sol.policy[0, best, d, 0, 0] in (1, 2), (d, best)
            assert sol.policy[0, 2 * d, d, 0, 0] == 0  # exactly unbeatable

    def test_first_live_tail_state_is_double(self):
        """9/04: streak 0, best 18, 24 days left."""
        sol = _solve_one_bin()
        assert sol.policy[0, 18, 24, 0, 0] == 2

    def test_exact_single_double_tie_prefers_single(self):
        # p=0.5, pb=0.25 at (s=0, m=0, d=1): single 0.5*1 == double 0.25*2.
        sol = _solve_one_bin(0.5, 0.25, target=5, max_days=1)
        assert sol.policy[0, 0, 1, 0, 0] == 1

    def test_multi_bin_smoke(self):
        freq = np.array([0.8, 0.2]); ph = np.array([0.7146, 0.8097]); pb = np.array([0.5403, 0.5944])
        sol = solve_emax_season_best(freq, ph, pb)
        assert sol.policy.shape == (TARGET + 1, TARGET + 1, MAX_TAIL_DAYS + 1, 2, 2)
        assert set(np.unique(sol.policy)) <= {0, 1, 2}
        assert np.all(sol.policy[:, :, 0] == 0)

    def test_rejects_bad_rates(self):
        with pytest.raises(ValueError):
            solve_emax_season_best(np.array([1.0]), np.array([0.5]), np.array([0.6]))  # p_both > p_hit
        with pytest.raises(ValueError):
            solve_emax_season_best(np.array([0.5]), np.array([0.7]), np.array([0.5]))  # freq != 1


# --- artifact contract --------------------------------------------------------

def _tail(tmp_path, **overrides) -> TailPolicy:
    sol = _solve_one_bin()
    fields = dict(
        objective=OBJECTIVE_TAIL, policy_table=sol.policy, boundaries=[],
        bin_freq=[1.0], bin_p_hit=[P_HIT], bin_p_both=[P_BOTH],
        target=TARGET, max_days=MAX_TAIL_DAYS,
        base_policy_sha256="a" * 64,
        manifest=tail_manifest(n_bins=1, hits=2641, both=1984, late_seed_days=3600, seed_dirs=24),
        built_at="2026-09-03T20:00:00Z", solver="solve_emax_season_best",
    )
    fields.update(overrides)
    return TailPolicy(**fields)


class TestArtifact:
    def test_round_trip_and_sha(self, tmp_path):
        path = tmp_path / "tail.npz"
        save_tail_policy(_tail(tmp_path), path)
        loaded = load_tail_policy(path, expected_base_sha="a" * 64)
        assert loaded.schema_version == TAIL_SCHEMA
        assert loaded.objective == OBJECTIVE_TAIL
        assert loaded.policy_table.dtype == np.int8
        assert loaded.manifest["seed_dirs"] == 24
        assert len(loaded.sha256) == 64
        assert loaded.bin_p_hit == pytest.approx([P_HIT])

    def test_base_sha_mismatch_is_invalid(self, tmp_path):
        path = tmp_path / "tail.npz"
        save_tail_policy(_tail(tmp_path), path)
        with pytest.raises(TailPolicyError, match="base policy"):
            load_tail_policy(path, expected_base_sha="b" * 64)

    def test_missing_file_is_invalid(self, tmp_path):
        with pytest.raises(TailPolicyError):
            load_tail_policy(tmp_path / "nope.npz")

    @pytest.mark.parametrize("mutate,match", [
        (lambda t: setattr(t, "objective", "reach57"), "objective"),
        (lambda t: setattr(t, "boundaries", [0.9, 0.8]), "boundaries"),
        (lambda t: setattr(t, "bin_p_both", [P_HIT + 0.01]), "p_both"),
        (lambda t: setattr(t, "bin_freq", [0.5]), "frequency"),
        (lambda t: setattr(t, "base_policy_sha256", "zz"), "sha256"),
    ])
    def test_field_validation(self, tmp_path, mutate, match):
        t = _tail(tmp_path); mutate(t)
        path = tmp_path / "tail.npz"
        save_tail_policy(t, path, validate=False)
        with pytest.raises(TailPolicyError, match=match):
            load_tail_policy(path)

    def test_stop_rule_violation_in_table_is_invalid(self, tmp_path):
        t = _tail(tmp_path)
        table = t.policy_table.copy()
        table[0, 18, 24, 0, 0] = 0          # a consulted play cell turned into a skip
        t.policy_table = table
        path = tmp_path / "tail.npz"
        save_tail_policy(t, path, validate=False)
        with pytest.raises(TailPolicyError, match="stop rule"):
            load_tail_policy(path)

    def test_wrong_shape_dtype_or_action_range_is_invalid(self, tmp_path):
        for bad in (
            _tail(tmp_path).policy_table[:, :, :10],                 # shape
            _tail(tmp_path).policy_table.astype(np.int32),           # dtype
        ):
            t = _tail(tmp_path); t.policy_table = bad
            path = tmp_path / "tail.npz"
            save_tail_policy(t, path, validate=False)
            with pytest.raises(TailPolicyError):
                load_tail_policy(path)
        t = _tail(tmp_path); table = t.policy_table.copy(); table[1, 1, 1, 0, 0] = 3
        t.policy_table = table
        path = tmp_path / "tail.npz"
        save_tail_policy(t, path, validate=False)
        with pytest.raises(TailPolicyError, match="action"):
            load_tail_policy(path)

    def test_nonzero_day_zero_slice_is_invalid(self, tmp_path):
        t = _tail(tmp_path); table = t.policy_table.copy(); table[0, 0, 0, 0, 0] = 1
        t.policy_table = table
        path = tmp_path / "tail.npz"
        save_tail_policy(t, path, validate=False)
        with pytest.raises(TailPolicyError, match="day 0"):
            load_tail_policy(path)

    def test_missing_key_is_invalid(self, tmp_path):
        path = tmp_path / "tail.npz"
        save_tail_policy(_tail(tmp_path), path)
        data = dict(np.load(path, allow_pickle=False))
        del data["bin_p_both"]
        np.savez_compressed(path, **data)
        with pytest.raises(TailPolicyError, match="bin_p_both"):
            load_tail_policy(path)

    def test_loader_never_uses_pickle(self, tmp_path, monkeypatch):
        path = tmp_path / "tail.npz"
        save_tail_policy(_tail(tmp_path), path)
        seen = {}
        real_load = np.load
        def spy(*a, **k):
            seen["allow_pickle"] = k.get("allow_pickle", "unset")
            return real_load(*a, **k)
        monkeypatch.setattr("bts.simulate.tail_policy.np.load", spy)
        load_tail_policy(path)
        assert seen["allow_pickle"] is False


# --- lookup -------------------------------------------------------------------

class TestLookup:
    @pytest.fixture
    def tail(self, tmp_path):
        path = tmp_path / "tail.npz"
        save_tail_policy(_tail(tmp_path), path)
        return load_tail_policy(path)

    def test_first_live_day(self, tail):
        assert lookup_tail_action(tail, streak=0, best=18, days=24, saver=False,
                                  p_game_hit=0.72) == "double"

    def test_stop_rule_reached(self, tail):
        assert lookup_tail_action(tail, 0, 18, 9, False, 0.72) == "skip"
        assert lookup_tail_action(tail, 0, 18, 10, False, 0.72) != "skip"

    def test_best_below_streak_is_clamped_up(self, tail):
        # A stale-low best can never make the streak look below itself.
        assert lookup_tail_action(tail, 5, 2, 3, False, 0.72) == \
            lookup_tail_action(tail, 5, 5, 3, False, 0.72)

    def test_best_above_target_is_capped(self, tail):
        assert lookup_tail_action(tail, 0, 99, 28, False, 0.72) == "skip"

    def test_no_days_or_finished_streak_skip(self, tail):
        assert lookup_tail_action(tail, 0, 18, 0, False, 0.72) == "skip"
        assert lookup_tail_action(tail, 57, 18, 5, False, 0.72) == "skip"

    def test_days_beyond_table_horizon_is_a_contract_error(self, tail):
        with pytest.raises(TailPolicyError, match="horizon"):
            lookup_tail_action(tail, 0, 18, MAX_TAIL_DAYS + 1, False, 0.72)

    def test_default_path_is_under_models(self):
        assert DEFAULT_TAIL_POLICY_PATH == Path("data/models/mdp_tail_policy.npz")
