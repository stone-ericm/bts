"""T7: fixed-policy P(reach target) projection over (s, m, saver, r) on a calendar (4b registration §5)."""
import numpy as np
import pytest

from scripts.audit.c1_r4b import project as P
from scripts.audit.c1_r4b import solvers as S
from scripts.audit.c1_r4b.oracle import independent_categories, projection as oracle


def env(early, late=None, n_bins=1):
    to_t = lambda xs: tuple(S.DayType(freq=f, q=q, partner=pt, p_hit=ph, p_both=pb) for f, q, pt, ph, pb in xs)
    return S.Environment(n_bins=n_bins, early=to_t(early), late=to_t(late if late is not None else early))


def table_policy(fn, T):
    """A policy as (d, q) -> (T+1, T+1, 2) raw actions, from a scalar fn(s, m, d, sv, q)."""
    def table(d, q):
        a = np.zeros((T + 1, T + 1, 2), dtype=np.int8)
        for s in range(T + 1):
            for m in range(T + 1):
                for sv in (0, 1):
                    a[s, m, sv] = fn(s, m, d, sv, q)
        return a
    return table


def cal(n, no_opp=()):
    """n calendar days, d_raw = n..1, with the given day indexes as no-opportunity days."""
    return [(i not in no_opp, n - i) for i in range(n)]


# ---------- independent scalar oracle: explicit recursion over days, types and outcome categories ----------


def test_codex_counterexample_best_must_survive_resets():
    """Codex r1 F6: target 3, five days, saver off, p_hit = p_both = 0.5."""
    e = env([(1.0, 0, True, 0.5, 0.5)])
    days = cal(5)
    always_double = table_policy(lambda s, m, d, sv, q: S.DOUBLE, 3)
    by_best = table_policy(lambda s, m, d, sv, q: S.DOUBLE if m < 2 else S.SINGLE, 3)
    kw = dict(target=3, saver_zone=(10, 15), late_days=0, stress=P.Stress(), init_saver=0)
    assert P.project(always_double, e, days, **kw)["p_reach"] == pytest.approx(0.59375, abs=1e-12)
    assert P.project(by_best, e, days, **kw)["p_reach"] == pytest.approx(0.50, abs=1e-12)


TINY_ENV = env(early=[(0.4, 0, True, 0.6, 0.35), (0.3, 1, False, 0.75, 0.0), (0.2, 1, True, 0.8, 0.6),
                      (0.1, None, False, 0.0, 0.0)],
               late=[(0.6, 0, True, 0.55, 0.3), (0.4, 1, True, 0.7, 0.5)], n_bins=2)


@pytest.mark.parametrize("stress", [P.Stress(), P.Stress(c=0.02), P.Stress(h=0.04), P.Stress(c=0.04, h=0.02, delta=0.1)])
def test_matches_the_scalar_oracle_for_an_m_dependent_policy(stress):
    T, zone = 4, (1, 1)
    fn = lambda s, m, d, sv, q: (S.DOUBLE if (m < 2 and q == 1) else S.SINGLE if s + d >= 3 else S.SKIP)
    days = cal(6, no_opp=(2,))
    got = P.project(table_policy(fn, T), TINY_ENV, days, target=T, saver_zone=zone, late_days=2, stress=stress,
                    r_cap=2)["p_reach"]
    want = oracle(fn, TINY_ENV, days, target=T, zone=zone, late_days=2, stress=stress, r_cap=2)
    assert got == pytest.approx(want, abs=1e-12)


def test_h_zero_equals_iid_and_dependence_lowers_the_run_tail():
    T = 4
    fn = lambda s, m, d, sv, q: S.SINGLE
    days = cal(8)
    base = P.project(table_policy(fn, T), TINY_ENV, days, target=T, saver_zone=(1, 1), late_days=2,
                     stress=P.Stress(), r_cap=2)["p_reach"]
    h0 = P.project(table_policy(fn, T), TINY_ENV, days, target=T, saver_zone=(1, 1), late_days=2,
                   stress=P.Stress(h=0.0), r_cap=2)["p_reach"]
    h4 = P.project(table_policy(fn, T), TINY_ENV, days, target=T, saver_zone=(1, 1), late_days=2,
                   stress=P.Stress(h=0.04), r_cap=2)["p_reach"]
    assert base == h0 and h4 < base


def test_categories_are_coherent_after_every_transform():
    t = S.DayType(freq=1.0, q=0, partner=True, p_hit=0.05, p_both=0.04)
    for st in (P.Stress(c=0.04), P.Stress(c=0.04, delta=0.139), P.Stress(c=0.04, h=0.04, delta=0.139)):
        for at_cap in (False, True):
            pm, po, pj = P.categories(t, st, at_cap=at_cap)
            assert min(pm, po, pj) >= 0 and pm + po + pj == pytest.approx(1.0)
    zero = S.DayType(freq=1.0, q=0, partner=True, p_hit=0.0, p_both=0.0)
    assert P.categories(zero, P.Stress(h=0.04), at_cap=True) == (1.0, 0.0, 0.0)


# ---------- review r1 (interim) fixes ----------


def test_categories_match_hand_computed_values():
    t = S.DayType(freq=1.0, q=0, partner=True, p_hit=0.6, p_both=0.35)
    pm, po, pj = P.categories(t, P.Stress(c=0.04, h=0.02, delta=0.1), at_cap=True)
    # c: ph 0.56, pb 0.31; Δ: pb 0.31 - 0.056 = 0.254; h at cap: ph 0.54, pb 0.254 * 0.54 / 0.56
    assert (pm, po, pj) == pytest.approx((0.46, 0.54 - 0.254 * 0.54 / 0.56, 0.254 * 0.54 / 0.56), abs=1e-12)
    assert P.categories(t, P.Stress(c=0.04, h=0.02, delta=0.1), at_cap=False) == pytest.approx((0.44, 0.56 - 0.254, 0.254))


@pytest.mark.parametrize("stress", [P.Stress(c=0.04), P.Stress(delta=0.139), P.Stress(c=0.02, h=0.04, delta=0.1)])
def test_projection_matches_an_oracle_with_independent_stress_transforms(stress, monkeypatch):
    T, zone = 4, (1, 1)
    fn = lambda s, m, d, sv, q: S.DOUBLE if q == 1 else S.SINGLE
    days = cal(6, no_opp=(2,))
    got = P.project(table_policy(fn, T), TINY_ENV, days, target=T, saver_zone=zone, late_days=2, stress=stress,
                    r_cap=2)["p_reach"]
    monkeypatch.setattr(P, "categories", lambda t, st, at_cap: independent_categories(
        t.p_hit, t.p_both, st.c, st.h, st.delta, at_cap))
    want = oracle(fn, TINY_ENV, days, target=T, zone=zone, late_days=2, stress=stress, r_cap=2)
    assert got == pytest.approx(want, abs=1e-12)


def test_a_phase_without_types_makes_the_projection_unavailable_not_zero():
    e = S.Environment(n_bins=1, early=(S.DayType(1.0, 0, True, 0.7, 0.5),), late=())
    with pytest.raises(P.Unavailable):
        P.project(table_policy(lambda *a: S.SINGLE, 4), e, cal(5), target=4, saver_zone=(1, 1), late_days=2,
                  stress=P.Stress())
    short = S.Environment(n_bins=1, early=(S.DayType(0.6, 0, True, 0.7, 0.5),), late=(S.DayType(1.0, 0, True, 0.7, 0.5),))
    with pytest.raises(P.Unavailable):
        P.project(table_policy(lambda *a: S.SINGLE, 4), short, cal(5), target=4, saver_zone=(1, 1), late_days=2,
                  stress=P.Stress())


# ---------- code review r1 F4 ----------
@pytest.mark.parametrize("bad", [
    (1.0, 0, True, float("nan"), float("nan")),          # positive-weight type with unsupported rates
    (float("nan"), 0, True, 0.7, 0.5),                    # non-finite frequency
    (1.0, 0, True, 0.5, 0.6),                             # incoherent: p_both > p_hit
    (1.0, 3, True, 0.7, 0.5),                             # bin outside the mapping
])
def test_unsupported_environments_are_unavailable_not_zero(bad):
    e = env([bad])
    with pytest.raises(P.Unavailable):
        P.project(table_policy(lambda *a: S.SINGLE, 4), e, cal(4), target=4, saver_zone=(1, 1), late_days=1,
                  stress=P.Stress())


def test_a_zero_weight_cell_needs_no_rates():
    e = env([(1.0, 0, True, 0.7, 0.5), (0.0, 0, False, float("nan"), float("nan"))])
    r = P.project(table_policy(lambda *a: S.SINGLE, 4), e, cal(4), target=4, saver_zone=(1, 1), late_days=1,
                  stress=P.Stress())
    assert r["mass"] == pytest.approx(1.0)
