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
