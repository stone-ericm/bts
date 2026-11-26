"""C1 4a T4/T5: the MAP intercept fit and the registered evaluation (registration §§1, 3, 4). Fixtures only."""
import math

import numpy as np
import pytest

from scripts.audit.c1_r4a import evaluate as E
from scripts.audit.c1_r4a import fit as F


def sig(x):
    return 1 / (1 + math.exp(-x))


def test_the_map_solves_the_registered_first_order_condition():
    dates = [([0.5], [1])]
    a = F.fit_intercept(dates)
    assert (1 - sig(a)) - a / 0.25 == pytest.approx(0.0, abs=1e-10)       # (y - g) - a / sigma^2 = 0


def test_dates_are_weighted_by_their_mean_not_by_row_count():
    """A date with 100 rows counts once, like a date with 1 row (date means are summed)."""
    big = ([0.7] * 100, [1] * 100)
    small = ([0.7], [0])
    a = F.fit_intercept([big, small])
    g = sig(math.log(0.7 / 0.3) + a)
    assert ((1 - g) + (0 - g)) - a / 0.25 == pytest.approx(0.0, abs=1e-10)


def test_probabilities_are_clipped_and_invalid_ones_refused():
    assert np.isfinite(F.fit_intercept([([0.0, 1.0], [0, 1])]))
    for bad in ([float("nan")], [1.5], [-0.1], [True]):
        with pytest.raises(ValueError):
            F.fit_intercept([(bad, [1])])


def test_the_fit_needs_twenty_five_dates():
    assert F.fit_or_inconclusive([([0.6], [1])] * 24) == {"status": "inconclusive", "reason": "insufficient support",
                                                          "fit_dates": 24}
    assert F.fit_or_inconclusive([([0.6], [1])] * 25)["status"] == "fitted"


def test_the_map_is_monotone_and_identity_at_zero():
    p = np.array([0.0, 0.2, 0.7, 1.0])
    assert np.allclose(F.apply_map(p, 0.0), p)
    m = F.apply_map(p, 0.3)
    assert m[0] == 0.0 and m[-1] == 1.0 and np.all(np.diff(m) >= 0)


def test_date_differences_are_map_minus_identity_mean_log_loss():
    d = E.date_differences([([0.6, 0.8], [1, 0])], a=0.2)
    ll = lambda p, y: -(y * math.log(p) + (1 - y) * math.log(1 - p))     # noqa: E731
    want = np.mean([ll(F.apply_map(np.array([q]), 0.2)[0], y) - ll(q, y) for q, y in ((0.6, 1), (0.8, 0))])
    assert d == pytest.approx([want])


def test_the_bootstrap_is_seeded_and_resamples_dates_with_multiplicity():
    v = np.array([-0.002, -0.001, 0.0, 0.003, -0.004])
    lo, hi = E.bootstrap_interval(v, seed=20270101)
    again = E.bootstrap_interval(v, seed=20270101)
    rng = np.random.default_rng(20270101)
    means = v[rng.integers(0, v.size, size=(10_000, v.size))].mean(axis=1)
    assert (lo, hi) == again == (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@pytest.mark.parametrize("diff,hi,g95,n,n1,want", [
    (-0.002, -0.0001, 0.004, 60, 60, "positive"),
    (-0.002, 0.0001, 0.004, 60, 60, "inconclusive"),        # interval reaches 0
    (-0.0005, -0.0001, 0.004, 60, 60, "inconclusive"),       # not past the practical bar
    (-0.002, -0.0001, 0.006, 60, 60, "inconclusive"),        # guardrail fails
    (0.0, 0.001, 0.004, 60, 60, "negative"),                 # difference >= 0
    (-0.002, -0.0001, 0.004, 49, 60, "inconclusive"),        # primary support floor
    (-0.002, -0.0001, 0.004, 60, 49, "inconclusive"),        # rank-1 support floor
])
def test_dispositions_follow_the_registered_table(diff, hi, g95, n, n1, want):
    out = E.disposition(diff=diff, upper=hi, guard_p95=g95, n_dates=n, n_rank1=n1)
    assert out["disposition"] == want


def test_the_bootstrap_matches_an_independent_per_draw_loop():
    """Review r1 noted the earlier test repeated the implementation's array expression; this draws one resample at a
    time from the same seeded generator and takes percentiles of the loop's means."""
    v = [-0.002, -0.001, 0.0, 0.003, -0.004, 0.001]
    rng = np.random.default_rng(20270101)
    means = sorted(sum(v[i] for i in rng.integers(0, len(v), size=len(v))) / len(v) for _ in range(10_000))
    lo, hi = E.bootstrap_interval(v)
    assert lo == pytest.approx(float(np.percentile(means, 2.5)), abs=1e-15)
    assert hi == pytest.approx(float(np.percentile(means, 97.5)), abs=1e-15)


def _log_post(dates, a):
    total = -a * a / (2 * 0.25)
    for p, y in dates:
        terms = []
        for q, yy in zip(p, y):
            g = sig(math.log(q / (1 - q)) + a)
            terms.append(yy * math.log(g) + (1 - yy) * math.log(1 - g))
        total += sum(terms) / len(terms)
    return total


def test_the_fit_window_interval_matches_an_independent_integration():
    dates = [([0.6, 0.7, 0.8][: 1 + i % 3], [1, 0, 1][: 1 + i % 3]) for i in range(30)]
    a = F.fit_intercept(dates)
    out = F.posterior_interval(dates, a)
    grid = [a - 3 + 6 * k / 6000 for k in range(6001)]
    lp = [_log_post(dates, g) for g in grid]
    m = max(lp)
    w = [math.exp(x - m) for x in lp]
    cdf, acc = [0.0], 0.0
    for k in range(1, len(grid)):
        acc += (w[k] + w[k - 1]) / 2 * (grid[k] - grid[k - 1])
        cdf.append(acc)
    lo, hi = np.interp([0.025, 0.975], np.array(cdf) / acc, grid)
    assert out["bounds"] == pytest.approx([lo, hi], abs=2e-4)
    assert out["bounds"][0] < a < out["bounds"][1] and out["level"] == 0.95
    assert "descriptive" in out["label"] and "not calibrated" in out["label"] and out["method"]


def test_secondary_metrics_weigh_dates_equally():
    """R10 (review counterexample): 100 rows p=.8 y=1 on one date, 1 row p=.8 y=0 on another: the equal-date Brier
    is (0.04 + 0.64) / 2 = 0.34, not the row-weighted 0.0459."""
    out = E.secondary([([0.8] * 100, [1] * 100, 0), ([0.8], [0], 0)], a=0.0)
    assert out["brier"]["identity"] == pytest.approx(0.34)
    assert out["brier"]["difference_map_minus_identity"] == pytest.approx(0.0)
    assert out["stated_minus_realized"]["identity"] == pytest.approx(((0.8 - 1) + (0.8 - 0)) / 2)
    assert out["rank1_stated_minus_realized"] == {"identity": pytest.approx(0.3), "map": pytest.approx(0.3), "dates": 2}
    assert out["dates"] == 2 and "equally" in out["weighting"]


def test_secondary_brier_difference_and_reliability_contract():
    dates = [([0.2, 0.5, 0.9], [0, 1, 1], 2), ([0.3, 0.7], [1, 0], None)]
    out = E.secondary(dates, a=0.4)
    bd = []
    for p, y, _ in dates:
        p, y = np.array(p), np.array(y, float)
        bd.append(np.mean((F.apply_map(p, 0.4) - y) ** 2) - np.mean((p - y) ** 2))
    assert out["brier"]["difference_map_minus_identity"] == pytest.approx(np.mean(bd))
    rel = out["reliability"]
    assert len(rel["edges"]) == 11 and rel["edges"][0] == 0.2 and rel["edges"][-1] == 0.9 and "1/n_d" in rel["rule"]
    assert sum(b["n"] for b in rel["bins"]) == 5 and sum(b["weight"] for b in rel["bins"]) == pytest.approx(2.0)
    assert out["rank1_stated_minus_realized"]["dates"] == 1
