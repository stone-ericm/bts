"""Tests for scripts/audit/dd_p_policy_value_sensitivity.py core machinery.

The load-bearing pieces — the generalized reach-K solver, the fixed-policy
evaluator, the different-game pairing, DD-leg shading, and the vectorized
thinned replay — are pinned against the existing bts.simulate.mdp /
pooled_policy implementations and against scalar reference ports of the
7/06 comparator (scripts/audit/confirm_mdp_policy_replay.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import solve_mdp
from bts.simulate.pooled_policy import evaluate_mdp_policy
from bts.simulate.quality_bins import QualityBin, QualityBins

from scripts.audit.dd_p_policy_value_sensitivity import (
    Env,
    aggregate_env_by_env_bin,
    build_env,
    env_from_quality_bins,
    freq_at,
    leg_breakeven,
    no_dd_low_action_fn,
    pair_diffgame,
    quintile_boundaries,
    replay_provider_const,
    replay_provider_no_dd_low,
    replay_provider_table,
    replay_vectorized,
    shade_env,
    solve_reach,
    start_value,
    table_action_fn,
    thinned_top2,
)


def _mk_bins(p_hits, p_boths, freqs, boundaries) -> QualityBins:
    bins = [
        QualityBin(index=i, p_range=(0.0, 1.0), p_hit=ph, p_both=pb, frequency=fr)
        for i, (ph, pb, fr) in enumerate(zip(p_hits, p_boths, freqs))
    ]
    return QualityBins(bins=bins, boundaries=boundaries)


EARLY = _mk_bins(
    [0.703, 0.741, 0.762, 0.781, 0.812],
    [0.501, 0.533, 0.552, 0.571, 0.603],
    [0.19, 0.21, 0.20, 0.22, 0.18],
    [0.72, 0.75, 0.77, 0.79],
)
LATE = _mk_bins(
    [0.691, 0.729, 0.751, 0.769, 0.801],
    [0.483, 0.517, 0.541, 0.557, 0.588],
    [0.22, 0.18, 0.21, 0.19, 0.20],
    [0.71, 0.74, 0.76, 0.78],
)


# ---------------------------------------------------------------- solver oracle


def test_solve_reach_matches_solve_mdp_no_late():
    sol = solve_mdp(EARLY, season_length=15)
    env = env_from_quality_bins(EARLY)
    V, pol = solve_reach(env, None, K=57, season_length=15)
    assert V.shape == sol.value_table.shape
    np.testing.assert_allclose(V, sol.value_table, atol=1e-12)
    np.testing.assert_array_equal(pol, sol.policy_table[:57])


def test_solve_reach_matches_solve_mdp_with_late_bins():
    sol = solve_mdp(EARLY, season_length=12, late_bins=LATE, late_phase_days=5)
    env_e = env_from_quality_bins(EARLY)
    env_l = env_from_quality_bins(LATE)
    V, pol = solve_reach(env_e, env_l, K=57, season_length=12, late_phase_days=5)
    np.testing.assert_allclose(V, sol.value_table, atol=1e-12)
    np.testing.assert_array_equal(pol, sol.policy_table[:57])


def test_fixed_eval_matches_evaluate_mdp_policy():
    # Solve a policy on one bin set, evaluate it on another (their holdout
    # quintile-position convention == pol_bin = arange(T)).
    donor = _mk_bins(
        [0.71, 0.73, 0.77, 0.79, 0.83],
        [0.52, 0.54, 0.56, 0.58, 0.62],
        [0.2] * 5,
        [0.72, 0.75, 0.78, 0.81],
    )
    sol = solve_mdp(donor, season_length=12, late_bins=LATE, late_phase_days=5)
    theirs = evaluate_mdp_policy(
        sol.policy_table, EARLY, season_length=12, late_bins=LATE, late_phase_days=5
    )
    env_e = env_from_quality_bins(EARLY)
    env_l = env_from_quality_bins(LATE)
    fn = table_action_fn(sol.policy_table, np.arange(5), streak_cap=56)
    V, pol = solve_reach(
        env_e, env_l, K=57, season_length=12, late_phase_days=5, action_fn=fn
    )
    assert pol is None
    mine = start_value(V, env_e, env_l, s=0, d=12, saver=1, late_phase_days=5)
    assert mine == pytest.approx(theirs, abs=1e-12)


def test_saver_semantics_and_target_cap():
    env = Env(
        freq=np.array([1.0]),
        p_hit=np.array([0.6]),
        p_both=np.array([0.35]),
        env_bin=np.array([0]),
        pol_bin=np.array([0]),
    )
    K, L = 14, 6

    def force(action):
        def fn(s, d, saver):
            return np.array([action], dtype=np.int8)

        return fn

    Vs, _ = solve_reach(env, None, K=K, season_length=L, action_fn=force(1))
    for d in range(2, L + 1):
        # saver catches a single miss at streak 12: hold streak, consume saver
        assert Vs[12, d, 1, 0] == pytest.approx(
            0.6 * Vs[13, d - 1, 1, 0] + 0.4 * Vs[12, d - 1, 0, 0], abs=1e-14
        )
        # no catch at streak 9: miss resets, saver kept
        assert Vs[9, d, 1, 0] == pytest.approx(
            0.6 * Vs[10, d - 1, 1, 0] + 0.4 * Vs[0, d - 1, 1, 0], abs=1e-14
        )

    Vd, _ = solve_reach(env, None, K=K, season_length=L, action_fn=force(2))
    for d in range(1, L + 1):
        # double at streak 12 with saver: 12+2 caps at K=14 (absorbing, V=1)
        assert Vd[12, d, 1, 0] == pytest.approx(
            0.35 * 1.0 + 0.65 * Vd[12, d - 1, 0, 0], abs=1e-14
        )
        # double at streak 13: 13+2 caps at 14 as well
        assert Vd[13, d, 1, 0] == pytest.approx(
            0.35 * 1.0 + 0.65 * Vd[13, d - 1, 0, 0], abs=1e-14
        )


# ---------------------------------------------------------------- environment


def test_pair_diffgame_prefers_best_rank_in_other_game():
    df = pd.DataFrame(
        [
            # date A: rank2 shares rank1's game -> leg comes from rank3
            {"date": "2021-04-01", "rank": 1, "game_pk": 10, "p_game_hit": 0.80, "actual_hit": 1},
            {"date": "2021-04-01", "rank": 2, "game_pk": 10, "p_game_hit": 0.79, "actual_hit": 0},
            {"date": "2021-04-01", "rank": 3, "game_pk": 20, "p_game_hit": 0.77, "actual_hit": 1},
            {"date": "2021-04-01", "rank": 4, "game_pk": 30, "p_game_hit": 0.76, "actual_hit": 0},
            # date B: every candidate in rank1's game -> same-game fallback
            {"date": "2021-04-02", "rank": 1, "game_pk": 99, "p_game_hit": 0.81, "actual_hit": 0},
            {"date": "2021-04-02", "rank": 2, "game_pk": 99, "p_game_hit": 0.78, "actual_hit": 1},
        ]
    )
    paired = pair_diffgame(df)
    assert list(paired["date"]) == ["2021-04-01", "2021-04-02"]
    a = paired.iloc[0]
    assert a["top1"] == 1 and a["top2"] == 1
    assert a["p1"] == pytest.approx(0.80) and a["p2"] == pytest.approx(0.77)
    assert not a["same_game_fallback"]
    b = paired.iloc[1]
    # fallback mirrors the 7/06 comparator: leg outcome = rank-1's own outcome
    assert b["top1"] == 0 and b["top2"] == 0
    assert b["p2"] == pytest.approx(0.81)
    assert b["same_game_fallback"]


def test_build_env_joint_cells_and_boundary_semantics():
    paired = pd.DataFrame(
        {
            "p1": [0.70, 0.70, 0.76, 0.75, 0.79, 0.79],
            "top1": [1, 0, 1, 1, 1, 0],
            "top2": [1, 1, 0, 1, 1, 1],
        }
    )
    env = build_env(paired, env_boundaries=[0.75], pol_boundaries=[0.78])
    # full (env_bin x pol_bin) grid in lexicographic order, empty cells kept
    # at freq 0 (compute_bins_with_boundaries precedent). 0.75 == boundary ->
    # upper bin, matching lookup_action's >= semantics.
    np.testing.assert_array_equal(env.env_bin, [0, 0, 1, 1])
    np.testing.assert_array_equal(env.pol_bin, [0, 1, 0, 1])
    np.testing.assert_allclose(env.freq, [2 / 6, 0.0, 2 / 6, 2 / 6])
    np.testing.assert_allclose(env.p_hit, [0.5, 0.0, 1.0, 0.5])
    np.testing.assert_allclose(env.p_both, [0.5, 0.0, 0.5, 0.5])
    assert env.freq.sum() == pytest.approx(1.0)


def test_shade_env_identity_floor_and_linearity():
    env = Env(
        freq=np.array([0.5, 0.3, 0.2]),
        p_hit=np.array([0.8, 0.75, 0.0]),
        p_both=np.array([0.6, 0.5, 0.0]),
        env_bin=np.array([0, 1, 2]),
        pol_bin=np.array([0, 0, 1]),
    )
    same = shade_env(env, 0.0)
    np.testing.assert_allclose(same.p_both, env.p_both)
    np.testing.assert_allclose(same.p_hit, env.p_hit)

    shaded = shade_env(env, 0.10)
    # p_both' = p_hit * (p_both/p_hit - delta) = p_both - delta * p_hit
    np.testing.assert_allclose(shaded.p_both, [0.6 - 0.08, 0.5 - 0.075, 0.0])
    np.testing.assert_allclose(shaded.p_hit, env.p_hit)  # primaries untouched

    floored = shade_env(env, 0.9)
    assert (floored.p_both >= 0).all()
    np.testing.assert_allclose(floored.p_both, [0.0, 0.0, 0.0])

    # shading commutes with env-bin aggregation (linearity, below the floor)
    agg_then_shade = shade_env(aggregate_env_by_env_bin(env), 0.10)
    shade_then_agg = aggregate_env_by_env_bin(shade_env(env, 0.10))
    np.testing.assert_allclose(agg_then_shade.p_both, shade_then_agg.p_both, atol=1e-15)
    np.testing.assert_allclose(agg_then_shade.p_hit, shade_then_agg.p_hit, atol=1e-15)


def test_env5_value_linearity_joint_vs_aggregated():
    rng = np.random.default_rng(11)
    n = 400
    p1 = rng.uniform(0.70, 0.85, n)
    paired = pd.DataFrame(
        {
            "p1": p1,
            "top1": (rng.random(n) < p1).astype(int),
            "top2": (rng.random(n) < 0.73).astype(int),
        }
    )
    env_b = quintile_boundaries(paired["p1"], n_bins=5)
    joint = build_env(paired, env_boundaries=env_b, pol_boundaries=[0.796, 0.8115, 0.8252, 0.8407])
    env5 = aggregate_env_by_env_bin(joint)
    assert len(env5.freq) == 5

    K, L = 20, 12
    V5, pol5 = solve_reach(env5, None, K=K, season_length=L)
    v_on_env5 = start_value(V5, env5, None, s=0, d=L, saver=1)

    fn = table_action_fn(pol5, joint.env_bin, streak_cap=K - 1)
    Vj, _ = solve_reach(joint, None, K=K, season_length=L, action_fn=fn)
    v_on_joint = start_value(Vj, joint, None, s=0, d=L, saver=1)
    assert v_on_joint == pytest.approx(v_on_env5, abs=1e-12)


# ---------------------------------------------------------------- breakeven


def test_leg_breakeven_self_consistency():
    env_e = env_from_quality_bins(EARLY)
    env_l = env_from_quality_bins(LATE)
    # K must be reachable from streak 0 within the remaining horizon, else
    # EV[s+1] == EV[s] == 0 and the breakeven is legitimately 0.
    K, L, lpd = 8, 15, 5
    V, _ = solve_reach(env_e, env_l, K=K, season_length=L, late_phase_days=lpd)
    for s in (0, 1, 2):
        d = 10
        r_star = leg_breakeven(V, env_e, env_l, s=s, d=d, saver=1, K=K, late_phase_days=lpd)
        assert 0.0 < r_star < 1.0
        # at leg rate r*, Q(double) == Q(single) for any p_hit > 0
        nf = freq_at(env_e, env_l, d - 1, lpd)
        ev1 = float(nf @ V[min(s + 1, K), d - 1, 1, :])
        ev2 = float(nf @ V[min(s + 2, K), d - 1, 1, :])
        ev0 = float(nf @ V[0, d - 1, 1, :])
        p_hit = 0.75
        q_single = p_hit * ev1 + (1 - p_hit) * ev0
        q_double = (p_hit * r_star) * ev2 + (1 - p_hit * r_star) * ev0
        assert q_double == pytest.approx(q_single, rel=1e-9)


def test_leg_breakeven_rejects_saver_zone():
    env = env_from_quality_bins(EARLY)
    V, _ = solve_reach(env, None, K=57, season_length=15)
    with pytest.raises(ValueError):
        leg_breakeven(V, env, None, s=12, d=10, saver=1, K=57)


# ---------------------------------------------------------------- replay


def _scalar_replay(top1, top2_rep, action_lookup, season_length):
    """Direct port of confirm_mdp_policy_replay.replay_season semantics."""
    streak = max_streak = resets = 0
    saver = 1
    for i in range(len(top1)):
        d = season_length - i
        if d <= 0:
            break
        if streak >= 57:
            a = 0
        else:
            a = action_lookup(streak, d, saver, i)
        if a == 0:
            continue
        hit = bool(top1[i]) if a == 1 else (bool(top1[i]) and bool(top2_rep[i]))
        if hit:
            streak += 1 if a == 1 else 2
        else:
            if saver and 10 <= streak <= 15:
                saver = 0
            else:
                streak = 0
                resets += 1
        max_streak = max(max_streak, streak)
        if streak >= 57:
            break
    return max_streak, resets


@pytest.mark.parametrize("season_length", [60, 40])
def test_replay_vectorized_matches_scalar_reference(season_length):
    rng = np.random.default_rng(5)
    n, R = 50, 6
    top1 = (rng.random(n) < 0.75).astype(bool)
    top2_base = (rng.random(n) < 0.73).astype(bool)
    thinned2 = np.broadcast_to(top2_base, (R, n)) & (rng.random((R, n)) >= 0.15)
    table = rng.integers(0, 3, size=(58, 61, 2, 3), dtype=np.int8)
    bins_per_day = rng.integers(0, 3, size=n)

    providers = {
        "table": replay_provider_table(table, bins_per_day, streak_cap=56),
        "single": replay_provider_const(1),
        "double": replay_provider_const(2),
        "no_dd": replay_provider_no_dd_low(
            replay_provider_table(table, bins_per_day, streak_cap=56), max_streak=2
        ),
    }

    def lookup_table(s, d, sv, i):
        return int(table[min(s, 56), min(d, 60), sv, bins_per_day[i]])

    lookups = {
        "table": lookup_table,
        "single": lambda s, d, sv, i: 1,
        "double": lambda s, d, sv, i: 2,
        "no_dd": lambda s, d, sv, i: (
            1 if (s <= 2 and lookup_table(s, d, sv, i) == 2) else lookup_table(s, d, sv, i)
        ),
    }

    for name, provider in providers.items():
        out = replay_vectorized(top1, thinned2, provider, season_length=season_length)
        for r in range(R):
            ms, rs = _scalar_replay(top1, thinned2[r], lookups[name], season_length)
            assert out["max_streak"][r] == ms, (name, r)
            assert out["resets"][r] == rs, (name, r)
            assert bool(out["reach20"][r]) == (ms >= 20)
            assert bool(out["reach30"][r]) == (ms >= 30)


def test_thinned_top2_statistics_and_determinism():
    rng = np.random.default_rng(7)
    top2 = rng.random(20000) < 0.73
    r_bar = top2.mean()
    delta = 0.10
    q = delta / r_bar

    th = thinned_top2(top2, q=q, n_reps=8, rng=np.random.default_rng(123))
    assert th.shape == (8, 20000)
    assert th.mean() == pytest.approx(r_bar - delta, abs=0.006)
    # thinning only turns hits off, never on
    assert not (th & ~top2[None, :]).any()

    same = thinned_top2(top2, q=q, n_reps=8, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(th, same)

    ident = thinned_top2(top2, q=0.0, n_reps=3, rng=np.random.default_rng(9))
    assert (ident == top2[None, :]).all()


def test_hit_runs_extraction():
    from scripts.audit.dd_p_policy_value_sensitivity import hit_runs

    assert hit_runs(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1])) == [2, 1, 3]
    assert hit_runs(np.array([0, 0])) == []
    assert hit_runs(np.array([1, 1])) == [2]  # trailing run counted


def test_run_structure_permutation_null_calibrates_on_iid_data():
    from scripts.audit.dd_p_policy_value_sensitivity import run_structure_diagnostics

    rng = np.random.default_rng(21)
    frames = []
    for fi in range(12):
        n = 250
        frames.append(
            pd.DataFrame(
                {
                    "file_idx": fi,
                    "season": 2021 + fi % 5,
                    "date": [f"d{i:03d}" for i in range(n)],
                    "top1": (rng.random(n) < 0.8).astype(int),
                    "p1": np.full(n, 0.8),
                }
            )
        )
    paired = pd.concat(frames, ignore_index=True)
    diag = run_structure_diagnostics(paired, window=8, n_perms=40, seed=5)
    aw = diag["allhit_windows_per_file"]
    # on genuinely iid data the day order carries no signal — the observed /
    # permutation ratio sits near 1 (the real profiles read ~0.1)
    assert aw["observed_over_permutation"] == pytest.approx(1.0, abs=0.2)
    assert diag["run_tail"][8]["ratio_vs_permutation"] == pytest.approx(1.0, abs=0.25)
    # deterministic under a fixed seed
    again = run_structure_diagnostics(paired, window=8, n_perms=40, seed=5)
    assert again["allhit_windows_per_file"]["observed_over_permutation"] == pytest.approx(
        aw["observed_over_permutation"], abs=1e-15
    )
    # per-season accumulator covers every season present
    assert sorted(diag["per_season_window_ratio"]) == [2021, 2022, 2023, 2024, 2025]


def test_no_dd_low_action_fn_only_rewrites_low_streak_doubles():
    base_actions = np.array([0, 1, 2, 2], dtype=np.int8)

    def base(s, d, saver):
        return base_actions.copy()

    fn = no_dd_low_action_fn(base, max_streak=2)
    np.testing.assert_array_equal(fn(0, 50, 1), [0, 1, 1, 1])
    np.testing.assert_array_equal(fn(2, 50, 1), [0, 1, 1, 1])
    np.testing.assert_array_equal(fn(3, 50, 1), [0, 1, 2, 2])
