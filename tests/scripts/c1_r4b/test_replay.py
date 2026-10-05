"""T5: the corrected realized-sequence replay (4b registration §4) and its arm providers."""
import numpy as np
import pytest

from scripts.audit.c1_r4b import replay as R
from scripts.audit.c1_r4b import solvers as S
from scripts.audit.c1_r4b.oracle import scalar_replay

T = 57


def season(rows):
    """rows = (opp, known, d_raw, p1, hit1, partner, hit2)."""
    cols = list(zip(*rows))
    return {"opp": np.array(cols[0], bool), "known": np.array(cols[1], bool), "d_raw": np.array(cols[2]),
            "p1": np.array(cols[3], float), "hit1": np.array(cols[4], bool), "partner": np.array(cols[5], bool),
            "hit2": np.array(cols[6], bool)}


def rand_season(rng, n=120, unknown=(), no_opp=()):
    rows = []
    for i in range(n):
        opp = i not in no_opp
        rows.append((opp, opp and i not in unknown, n - i, float(rng.uniform(0.6, 0.9)), bool(rng.random() < 0.72),
                     bool(rng.random() < 0.9), bool(rng.random() < 0.7)))
    return season(rows)


def scalarize(provider):
    return lambda s, m, d, sv, p1, partner: int(provider(np.array([s]), np.array([m]), d, np.array([sv]), p1, partner)[0])


@pytest.mark.parametrize("arm", ["single", "double", "mixed"])
def test_vectorized_replay_equals_the_scalar_reference(arm):
    rng = np.random.default_rng(7)
    days = rand_season(rng, unknown=(5, 40), no_opp=(10, 11, 60))
    prov = {"single": R.const(S.SINGLE), "double": R.const(S.DOUBLE),
            "mixed": lambda s, m, d, sv, p1, partner: np.where((s >= 3) & (p1 > 0.75), S.DOUBLE,
                                                               np.where(m > s + 4, S.SKIP, S.SINGLE))}[arm]
    masks = rng.random((5, len(days["opp"]))) > 0.3
    got = R.replay(days, {"x": prov}, hit2_masks=masks & days["hit2"][None, :])["x"]
    for r in range(5):
        want = scalar_replay(days, scalarize(prov), hit2=masks[r] & days["hit2"])
        assert got["max"][r] == want["max"] and got["resets"][r] == want["resets"]
        for k in ("skip", "single", "double", "demoted"):
            assert got["actions"][k][r] == want[k], (k, r)


def test_skips_and_unknown_days_consume_calendar_time_and_a_partnerless_double_is_a_single():
    days = season([(True, True, 3, 0.8, True, False, False), (False, False, 2, np.nan, False, False, False),
                   (True, False, 1, np.nan, False, False, False)])
    seen = []
    prov = lambda s, m, d, sv, p1, partner: (seen.append(d), np.full(s.shape, S.DOUBLE))[1]
    out = R.replay(days, {"x": prov}, hit2_masks=np.zeros((1, 3), bool))["x"]
    assert seen == [3]                                      # only the known opportunity day calls the policy
    assert out["max"][0] == 1 and out["actions"]["demoted"][0] == 1


def a0_tables():
    base = np.zeros((58, 181, 2, 5), np.int8)
    base[:, :, :, 3:] = S.DOUBLE
    base[:, :, :, 1:3] = S.SINGLE
    tail = np.zeros((58, 58, 29, 2, 1), np.int8)
    tail[:, :, :, :, 0] = S.SINGLE
    tail[20, 25, :, :, 0] = S.SKIP
    return base, tail


def test_a0_routes_like_production():
    base, tail = a0_tables()
    a0 = R.A0(base=base, base_bounds=[0.7, 0.75, 0.8, 0.85], base_season_length=180, tail=tail, tail_bounds=[])
    s, m, sv = np.array([0, 0, 30, 20, 20]), np.array([0, 0, 30, 20, 25]), np.array([1, 1, 1, 1, 1])
    assert list(a0(s, m, 200, sv, 0.85, True)) == [S.DOUBLE] * 5          # 0.85 == bound -> upper bin; d clamps to 180
    assert list(a0(s, m, 100, sv, 0.70, True)) == [S.SINGLE] * 5          # equality enters bin 1
    # tail when s + 2d < 57: d = 10 -> s < 37: rows 0,1,3,4 tail; row 2 (s=30 -> 50 < 57) tail too
    got = a0(s, m, 10, sv, 0.90, True)
    assert list(got) == [S.SINGLE, S.SINGLE, S.SINGLE, S.SINGLE, S.SKIP]   # tail uses m = max(s, best)
    edge = a0(np.array([37]), np.array([37]), 10, np.array([1]), 0.90, True)  # 37 + 20 = 57: still reach-57
    assert list(edge) == [S.DOUBLE]


def test_solution_arm_and_hybrid_use_the_fold_classifier_and_horizon():
    tiny = S.Environment(n_bins=2, early=(S.DayType(1.0, 0, True, 0.7, 0.5), S.DayType(0.0, 1, True, 0.8, 0.6)),
                         late=(S.DayType(0.5, 0, True, 0.7, 0.5), S.DayType(0.5, 1, True, 0.8, 0.6)))
    with pytest.raises(ValueError):          # a zero-weight bin in a phase is an empty fitting cell
        S.validate(tiny)
    env = S.Environment(n_bins=2, early=(S.DayType(0.5, 0, True, 0.7, 0.5), S.DayType(0.5, 1, True, 0.8, 0.6)),
                        late=(S.DayType(0.5, 0, True, 0.7, 0.5), S.DayType(0.5, 1, True, 0.8, 0.6)))
    emax = S.solve(env, horizon=40, objective="emax")
    reach = S.solve(env, horizon=40, objective="reach")
    arm = R.Table(emax, cuts=np.array([0.75]))
    hyb = R.HybridArm(reach=reach, emax=emax, cuts=np.array([0.75]))
    s, m, sv = np.array([0, 5]), np.array([0, 9]), np.array([1, 0])
    assert list(arm(s, m, 40, sv, 0.75, True)) == [emax.policy[0, 0, 40, 1, 1], emax.policy[5, 9, 40, 0, 1]]
    assert list(hyb(s, m, 30, sv, 0.70, True)) == [reach.policy[0, 0, 30, 1, 0], reach.policy[5, 9, 30, 0, 0]]  # s + 2d >= 57
    assert list(hyb(s, m, 10, sv, 0.70, True)) == [emax.policy[0, 0, 10, 1, 0], emax.policy[5, 9, 10, 0, 0]]


def test_thinning_masks_are_nested_across_delta_and_stable_per_identity():
    hit2 = np.array([True] * 50 + [False] * 10)
    m05 = R.thin_masks(hit2, delta=0.05, r_bar=0.7, reps=200, identity=(2021, 11))
    m10 = R.thin_masks(hit2, delta=0.10, r_bar=0.7, reps=200, identity=(2021, 11))
    again = R.thin_masks(hit2, delta=0.10, r_bar=0.7, reps=200, identity=(2021, 11))
    assert (m10 <= m05).all() and (again == m10).all()                   # nested; reproducible
    assert not (m05 & ~hit2).any()                                       # never adds a hit
    assert R.thin_masks(hit2, delta=0.0, r_bar=0.7, reps=200, identity=(2021, 11)).shape == (1, 60)
    with pytest.raises(ValueError):
        R.thin_masks(hit2, delta=0.05, r_bar=0.0, reps=200, identity=(2021, 11))


def test_decision_consequences_compare_actions_on_common_states():
    days = season([(True, True, 5, 0.8, True, True, True)] * 3)
    single, double = R.const(S.SINGLE), R.const(S.DOUBLE)
    out = R.replay(days, {"A0": single, "A2": double}, hit2_masks=np.ones((1, 3), bool),
                   compare=("A0", "A2"))
    c = out["_consequences"]
    assert c["A0_states"]["visits"] == 3 and c["A0_states"]["differ"] == 3
    assert c["A2_states"]["visits"] == 3 and c["A2_states"]["differ"] == 3
    assert c["own_trajectory"]["differ"] == 3
