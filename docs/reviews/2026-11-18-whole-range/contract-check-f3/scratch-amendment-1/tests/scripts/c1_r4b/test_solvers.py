"""T4: phase-aware whole-season solvers with the legal availability kernel (4b registration §§2, 3, 5)."""
import itertools

import numpy as np
import pytest

from scripts.audit.c1_r4b import solvers as S
from scripts.audit.c1_r4b import oracle as O

# Tiny worlds: target 4, saver zone {1}, horizon 5, late phase = last 2 days, 2 quality bins.
TINY = dict(target=4, zone=(1, 1), late_days=2)
ENV = {"early": [(0.35, 0, True, 0.55, 0.30), (0.25, 0, False, 0.60, 0.0), (0.30, 1, True, 0.80, 0.62),
                 (0.10, None, False, 0.0, 0.0)],
       "late": [(0.5, 0, True, 0.50, 0.24), (0.3, 1, False, 0.75, 0.0), (0.2, 1, True, 0.70, 0.45)]}


def env_obj(env=ENV, n_bins=2):
    to_t = lambda xs: tuple(S.DayType(freq=f, q=q, partner=pt, p_hit=ph, p_both=pb) for f, q, pt, ph, pb in xs)
    return S.Environment(n_bins=n_bins, early=to_t(env["early"]), late=to_t(env["late"]))


def solve(objective, horizon=5, env=ENV):
    return S.solve(env_obj(env), horizon=horizon, objective=objective, target=TINY["target"],
                   saver_zone=TINY["zone"], late_days=TINY["late_days"])


def as_policy(sol):
    return lambda s, m, d, sv, q: int(sol.policy[s, m, d, sv, q])


STATES = [(s, m, d, sv) for d in range(0, 6) for sv in (0, 1) for s in range(0, 5) for m in range(s, 5)]


@pytest.mark.parametrize("objective", ["emax", "reach"])
def test_values_equal_brute_force_optimum_and_the_policy_achieves_it(objective):
    sol = solve(objective)
    pol = as_policy(sol)
    for s, m, d, sv in STATES:
        opt = O.optimal(ENV, objective, s, m, d, sv, **TINY)
        got = O.evaluate(ENV, pol, objective, s, m, d, sv, **TINY)
        assert sol.value[s, m, d, sv] == pytest.approx(opt, abs=1e-12), (s, m, d, sv)
        assert got == pytest.approx(opt, abs=1e-12), (s, m, d, sv)


def test_emax_policy_depends_on_best_and_the_oracle_sees_it():
    """Codex r1 F6: an m-dependent policy must be evaluated with m retained across resets."""
    sol = solve("emax")
    differs = [(s, d, sv, q) for s, d, sv, q in itertools.product(range(4), range(1, 6), (0, 1), (0, 1))
               if len({int(sol.policy[s, m, d, sv, q]) for m in range(s, 5)}) > 1]
    assert differs, "the tiny world should exhibit at least one m-dependent action"


def test_emax_stop_rule_and_play_first_ties():
    sol = solve("emax")
    T = TINY["target"]
    for s, m, d, sv in STATES:
        if d >= 1 and s < T and min(T, s + 2 * d) <= m:
            assert (sol.policy[s, m, d, sv, :] == S.SKIP).all(), (s, m, d, sv)


def test_reach_breaks_ties_toward_skip_where_the_target_is_unreachable():
    sol = solve("reach")
    T = TINY["target"]
    for s, m, d, sv in STATES:
        if d >= 1 and s + 2 * d < T:
            assert (sol.policy[s, m, d, sv, :] == S.SKIP).all()
            assert sol.value[s, m, d, sv] == 0.0


def test_a_partnerless_double_is_valued_as_a_single():
    only_partnerless = {"early": [(1.0, 0, False, 0.7, 0.0)], "late": [(1.0, 0, False, 0.7, 0.0)]}
    sol = S.solve(env_obj(only_partnerless, n_bins=1), horizon=4, objective="emax", target=4, saver_zone=(1, 1),
                  late_days=2)
    played = sol.policy[sol.policy != S.SKIP]
    assert played.size and (played == S.SINGLE).all()     # play-first ties: single before an identical double


def test_hybrid_routes_by_reachability():
    reach, emax = solve("reach"), solve("emax")
    T = TINY["target"]
    hyb = S.Hybrid(reach=reach, emax=emax, target=T)
    for s, m, d, sv in STATES:
        if d == 0 or s >= T:
            continue
        for q in (0, 1):
            want = reach.policy[s, m, d, sv, q] if s + 2 * d >= T else emax.policy[s, m, d, sv, q]
            assert hyb.action(s, m, d, sv, q) == want


def test_reach_equals_solve_mdp_on_an_all_partners_kernel():
    """Registration §5: A1's reach slice must equal solve_mdp on matching rates with every partner available."""
    from bts.simulate.mdp import solve_mdp
    from bts.simulate.quality_bins import QualityBin, QualityBins
    early = [(0.5, 0.70, 0.48), (0.5, 0.80, 0.62)]
    late = [(0.5, 0.65, 0.40), (0.5, 0.78, 0.58)]
    qb = lambda xs: QualityBins(bins=[QualityBin(index=i, p_range=(0.0, 1.0), p_hit=ph, p_both=pb, frequency=f)
                                      for i, (f, ph, pb) in enumerate(xs)], boundaries=[0.75])
    D = 40
    ref = solve_mdp(qb(early), season_length=D, late_bins=qb(late), late_phase_days=30)
    env = {"early": [(f, i, True, ph, pb) for i, (f, ph, pb) in enumerate(early)],
           "late": [(f, i, True, ph, pb) for i, (f, ph, pb) in enumerate(late)]}
    mine = S.solve(env_obj(env), horizon=D, objective="reach", target=57, saver_zone=(10, 15), late_days=30)
    freq_e, freq_l = np.array([0.5, 0.5]), np.array([0.5, 0.5])
    for s in range(57):
        for d in range(1, D + 1):
            for sv in (0, 1):
                f = freq_l if d <= 30 else freq_e
                assert mine.value[s, s, d, sv] == pytest.approx(float(f @ ref.value_table[s, d, sv, :]), abs=1e-12)
                assert (mine.policy[s, s, d, sv, :] == ref.policy_table[s, d, sv, :]).all()


def test_environment_validation_refuses_incoherent_rates():
    bad = {"early": [(1.0, 0, True, 0.5, 0.6)], "late": [(1.0, 0, True, 0.5, 0.4)]}     # p_both > p_hit
    with pytest.raises(ValueError):
        S.validate(env_obj(bad, n_bins=1))
    bad2 = {"early": [(1.0, 0, False, 0.5, 0.1)], "late": [(1.0, 0, True, 0.5, 0.4)]}   # partnerless p_both != 0
    with pytest.raises(ValueError):
        S.validate(env_obj(bad2, n_bins=1))
    bad3 = {"early": [(0.6, 0, True, 0.5, 0.4)], "late": [(1.0, 0, True, 0.5, 0.4)]}    # freq sums to 0.6
    with pytest.raises(ValueError):
        S.validate(env_obj(bad3, n_bins=1))
