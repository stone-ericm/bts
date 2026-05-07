"""Tests for SOTA tracker item #7 P0 — BH/BY FDR baseline + Poisson-binomial.

TDD: tests written before implementation. Module under test is
`bts.validate.fdr`.

Scope per Codex bus #225/#227:
- BH (Benjamini-Hochberg 1995) and BY (Benjamini-Yekutieli 2001 with c(m)
  harmonic penalty for arbitrary-dependence) on ordinary p-values.
- Poisson-binomial p-values (heterogeneous H0) via scipy.stats.poisson_binom,
  inclusive upper-tail via sf(x - 1).
- p-value FDR baseline ONLY — NOT e-BH (1/p is not a valid p-to-e calibrator
  per Wang & Ramdas 2022; e-BH deferred until valid e-values are constructed).
- Tail direction labeling per Codex's correction: overconfidence ⇔ observed
  hits LOW vs expected ⇔ p_lower < p_upper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from bts.validate.fdr import (
    bh_qvalues,
    by_qvalues,
    cell_pvalue,
    cell_pvalues_from_artifact,
    sign_flip_permutation_pvalue,
)


# ---- BH/BY adjustment ----


def test_bh_known_fixture() -> None:
    """Hand-computed BH q-values against a 5-element textbook fixture.

    p = [0.005, 0.01, 0.04, 0.05, 0.10], m = 5.
    BH steps: m/k * p_(k) = [0.025, 0.025, 0.0667, 0.0625, 0.10].
    Cumulative-min from the right:
      q_(1) = min(.025, .025, .0667, .0625, .10) = .025
      q_(2) = min(.025, .0667, .0625, .10)       = .025
      q_(3) = min(.0667, .0625, .10)             = .0625
      q_(4) = min(.0625, .10)                    = .0625
      q_(5) = .10
    """
    p = np.array([0.005, 0.01, 0.04, 0.05, 0.10])
    expected = np.array([0.025, 0.025, 0.0625, 0.0625, 0.10])
    q = bh_qvalues(p)
    np.testing.assert_allclose(q, expected, rtol=1e-9)


def test_by_known_fixture_harmonic_penalty() -> None:
    """BY = BH * c(m) where c(m) = sum_{i=1..m} 1/i (arbitrary-dependence)."""
    p = np.array([0.005, 0.01, 0.04, 0.05, 0.10])
    bh = np.array([0.025, 0.025, 0.0625, 0.0625, 0.10])
    c_m = sum(1.0 / i for i in range(1, 6))  # 1 + 1/2 + 1/3 + 1/4 + 1/5
    expected = np.minimum(1.0, bh * c_m)
    q = by_qvalues(p)
    np.testing.assert_allclose(q, expected, rtol=1e-9)


def test_by_at_least_bh_pointwise() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(size=20)
    bh = bh_qvalues(p)
    by = by_qvalues(p)
    assert np.all(by >= bh - 1e-12), "BY must be >= BH everywhere"


def test_bh_monotone_in_sorted_order() -> None:
    rng = np.random.default_rng(123)
    p = rng.uniform(size=15)
    q = bh_qvalues(p)
    order = np.argsort(p)
    sorted_q = q[order]
    diffs = np.diff(sorted_q)
    assert np.all(diffs >= -1e-12), "Sorted BH q-values must be non-decreasing"


def test_bh_preserves_input_order() -> None:
    """Output must be aligned with input, not sorted internally."""
    p = np.array([0.10, 0.005, 0.05, 0.01, 0.04])
    # Sorted: [0.005, 0.01, 0.04, 0.05, 0.10]; BH-sorted: [.025, .025, .0625, .0625, .10]
    # Map back to original positions:
    # original[0] = 0.10 -> sorted pos 4 -> q = 0.10
    # original[1] = 0.005 -> sorted pos 0 -> q = 0.025
    # original[2] = 0.05 -> sorted pos 3 -> q = 0.0625
    # original[3] = 0.01 -> sorted pos 1 -> q = 0.025
    # original[4] = 0.04 -> sorted pos 2 -> q = 0.0625
    expected = np.array([0.10, 0.025, 0.0625, 0.025, 0.0625])
    q = bh_qvalues(p)
    np.testing.assert_allclose(q, expected, rtol=1e-9)


def test_by_preserves_input_order() -> None:
    p = np.array([0.10, 0.005])  # m=2, c_m = 1 + 1/2 = 1.5
    # Sorted: [0.005, 0.10]; BH steps: [2/1 * 0.005, 2/2 * 0.10] = [0.01, 0.10]
    # BH q (cumulative min from right): [0.01, 0.10]
    # Mapped back: original[0]=0.10 -> 0.10; original[1]=0.005 -> 0.01
    # BY: BH * 1.5 = [0.15, 0.015], capped at 1.0
    bh_expected = np.array([0.10, 0.01])
    by_expected = np.array([0.15, 0.015])
    np.testing.assert_allclose(bh_qvalues(p), bh_expected, rtol=1e-9)
    np.testing.assert_allclose(by_qvalues(p), by_expected, rtol=1e-9)


def test_no_rejections_edge_all_one() -> None:
    p = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(bh_qvalues(p), np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(by_qvalues(p), np.array([1.0, 1.0, 1.0]))


def test_all_rejections_edge_all_zero() -> None:
    p = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(bh_qvalues(p), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(by_qvalues(p), np.array([0.0, 0.0, 0.0]))


def test_empty_array_returns_empty() -> None:
    p = np.array([], dtype=float)
    assert bh_qvalues(p).shape == (0,)
    assert by_qvalues(p).shape == (0,)


def test_bh_raises_on_nan() -> None:
    p = np.array([0.01, np.nan, 0.05])
    with pytest.raises(ValueError, match="(?i)nan"):
        bh_qvalues(p)


def test_by_raises_on_nan() -> None:
    p = np.array([0.01, np.nan, 0.05])
    with pytest.raises(ValueError, match="(?i)nan"):
        by_qvalues(p)


def test_bh_raises_on_out_of_range() -> None:
    with pytest.raises(ValueError):
        bh_qvalues(np.array([0.5, -0.01]))
    with pytest.raises(ValueError):
        bh_qvalues(np.array([0.5, 1.01]))


# ---- Paired sign-flip audit p-values ----


def test_sign_flip_permutation_two_positive_deltas() -> None:
    result = sign_flip_permutation_pvalue(np.array([1.0, 2.0]))

    assert result["n"] == 2
    assert result["observed_mean_delta"] == pytest.approx(1.5)
    assert result["p_two_sided"] == pytest.approx(0.5)
    assert result["p_one_sided_positive"] == pytest.approx(0.25)


def test_sign_flip_permutation_mixed_deltas_gives_no_signal() -> None:
    result = sign_flip_permutation_pvalue(np.array([1.0, -1.0]))

    assert result["observed_mean_delta"] == pytest.approx(0.0)
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_sign_flip_permutation_validates_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        sign_flip_permutation_pvalue(np.array([]))
    with pytest.raises(ValueError, match="NaN"):
        sign_flip_permutation_pvalue(np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="<= 20"):
        sign_flip_permutation_pvalue(np.ones(21))


# ---- Poisson-binomial cell p-values ----


def test_cell_pvalue_iid_matches_binomial() -> None:
    """When all p_i equal, Poisson-binomial reduces to scipy.stats.binom."""
    n, p = 10, 0.4
    probs = np.full(n, p)
    for x in [0, 3, 5, 7, 10]:
        result = cell_pvalue(actual_hits=x, row_probabilities=probs)
        expected_lower = stats.binom.cdf(x, n, p)
        expected_upper = stats.binom.sf(x - 1, n, p)  # inclusive upper via sf(x-1)
        assert abs(result["p_lower"] - expected_lower) < 1e-9, (
            f"x={x}: p_lower={result['p_lower']} vs binom={expected_lower}"
        )
        assert abs(result["p_upper"] - expected_upper) < 1e-9, (
            f"x={x}: p_upper={result['p_upper']} vs binom sf(x-1)={expected_upper}"
        )


def test_cell_pvalue_hand_computed_pmf() -> None:
    """3 rows with p=[0.5, 0.3, 0.2]; expected PMF:
       P(X=0) = 0.5*0.7*0.8 = 0.28
       P(X=1) = 0.5*0.7*0.8 + 0.5*0.3*0.8 + 0.5*0.7*0.2 = 0.47
       P(X=2) = 0.5*0.3*0.8 + 0.5*0.7*0.2 + 0.5*0.3*0.2 = 0.22
       P(X=3) = 0.5*0.3*0.2 = 0.03
    """
    probs = np.array([0.5, 0.3, 0.2])
    # actual_hits = 0: lowest possible
    r = cell_pvalue(actual_hits=0, row_probabilities=probs)
    assert abs(r["p_lower"] - 0.28) < 1e-9
    assert abs(r["p_upper"] - 1.00) < 1e-9  # P(X >= 0)
    # actual_hits = 3: highest possible
    r = cell_pvalue(actual_hits=3, row_probabilities=probs)
    assert abs(r["p_lower"] - 1.00) < 1e-9  # P(X <= 3)
    assert abs(r["p_upper"] - 0.03) < 1e-9  # P(X >= 3) = P(X=3)
    # actual_hits = 2
    r = cell_pvalue(actual_hits=2, row_probabilities=probs)
    assert abs(r["p_lower"] - 0.97) < 1e-9  # 0.28 + 0.47 + 0.22
    assert abs(r["p_upper"] - 0.25) < 1e-9  # 0.22 + 0.03


def test_cell_pvalue_inclusive_upper_via_sf() -> None:
    """Codex #227: explicit sf(x - 1) semantics for inclusive upper tail."""
    probs = np.array([0.5, 0.3, 0.2])
    # P(X >= 2) = 0.22 + 0.03 = 0.25, NOT P(X > 2) = 0.03
    r = cell_pvalue(actual_hits=2, row_probabilities=probs)
    assert abs(r["p_upper"] - 0.25) < 1e-9, (
        "p_upper must be P(X >= x), inclusive (via sf(x - 1)); "
        f"got {r['p_upper']}, expected 0.25"
    )


def test_cell_pvalue_two_sided_convention() -> None:
    """p_two_sided = min(1, 2 * min(p_lower, p_upper)). Standard discrete cap."""
    probs = np.array([0.5, 0.3, 0.2])
    r = cell_pvalue(actual_hits=3, row_probabilities=probs)
    # min(p_lower=1.0, p_upper=0.03) = 0.03; 2*0.03 = 0.06; capped = 0.06
    assert abs(r["p_two_sided"] - 0.06) < 1e-9


def test_tail_direction_overconfidence_when_observed_low() -> None:
    """Codex #227: observed hits LOW vs expected => overconfidence => p_lower < p_upper.

    Expected hits = sum(probs) = 1.0. observed = 0 (below expected).
    p_lower = 0.28, p_upper = 1.00. p_lower < p_upper => overconfidence.
    """
    probs = np.array([0.5, 0.3, 0.2])
    r = cell_pvalue(actual_hits=0, row_probabilities=probs)
    assert r["tail_direction"] == "overconfidence"


def test_tail_direction_underconfidence_when_observed_high() -> None:
    """Observed hits HIGH vs expected => underconfidence => p_upper < p_lower."""
    probs = np.array([0.5, 0.3, 0.2])
    r = cell_pvalue(actual_hits=3, row_probabilities=probs)
    # p_lower = 1.0, p_upper = 0.03
    assert r["tail_direction"] == "underconfidence"


def test_tail_direction_balanced_when_tails_equal() -> None:
    """Symmetric case: equal tails => 'balanced'."""
    # Construct a fixture where p_lower == p_upper exactly. Using a
    # symmetric binomial: n=4, p=0.5, actual_hits=2 (mode/median).
    # P(X<=2) = 11/16 = 0.6875; P(X>=2) = 11/16 = 0.6875.
    probs = np.full(4, 0.5)
    r = cell_pvalue(actual_hits=2, row_probabilities=probs)
    assert abs(r["p_lower"] - r["p_upper"]) < 1e-9
    assert r["tail_direction"] == "balanced"


def test_cell_pvalue_validates_actual_hits_range() -> None:
    probs = np.array([0.3, 0.4, 0.5])
    with pytest.raises(ValueError):
        cell_pvalue(actual_hits=-1, row_probabilities=probs)
    with pytest.raises(ValueError):
        cell_pvalue(actual_hits=4, row_probabilities=probs)


def test_cell_pvalue_validates_row_probabilities() -> None:
    with pytest.raises(ValueError):
        cell_pvalue(actual_hits=1, row_probabilities=np.array([0.3, np.nan]))
    with pytest.raises(ValueError):
        cell_pvalue(actual_hits=1, row_probabilities=np.array([0.3, -0.01]))
    with pytest.raises(ValueError):
        cell_pvalue(actual_hits=1, row_probabilities=np.array([0.3, 1.01]))


def test_cell_pvalue_empty_row_probabilities_raises() -> None:
    with pytest.raises(ValueError):
        cell_pvalue(actual_hits=0, row_probabilities=np.array([]))


# ---- Cut-C extraction from canonical artifact ----


def _synthetic_cut_c_artifact() -> pd.DataFrame:
    """4 resolved rows in two distinct cells of Cut C; identical regime/slot/env;
    differ only by skill_quartile.

    Cell A (regime=R, slot=primary, env=False, Q=4): n=2, hits=2, p_game_hit=[0.7, 0.8]
    Cell B (regime=R, slot=primary, env=False, Q=1): n=2, hits=0, p_game_hit=[0.6, 0.7]
    """
    rows = [
        {
            "regime": "R", "slot": "primary", "is_park_driven": False,
            "batter_skill_quartile": 4, "p_game_hit": 0.7, "actual_hit": True,
            "result_status": "resolved",
        },
        {
            "regime": "R", "slot": "primary", "is_park_driven": False,
            "batter_skill_quartile": 4, "p_game_hit": 0.8, "actual_hit": True,
            "result_status": "resolved",
        },
        {
            "regime": "R", "slot": "primary", "is_park_driven": False,
            "batter_skill_quartile": 1, "p_game_hit": 0.6, "actual_hit": False,
            "result_status": "resolved",
        },
        {
            "regime": "R", "slot": "primary", "is_park_driven": False,
            "batter_skill_quartile": 1, "p_game_hit": 0.7, "actual_hit": False,
            "result_status": "resolved",
        },
    ]
    df = pd.DataFrame(rows)
    df["is_park_driven"] = df["is_park_driven"].astype("boolean")
    df["batter_skill_quartile"] = df["batter_skill_quartile"].astype("Int64")
    return df


def test_cell_pvalues_from_artifact_synthetic() -> None:
    """Build 2 known cells; verify per-cell p-values match hand-computed."""
    df = _synthetic_cut_c_artifact()
    result = cell_pvalues_from_artifact(df)
    # Should have 2 cells with the right keys
    cells = result["cells"]
    assert len(cells) == 2

    # Find cell A (Q=4)
    cell_a = next(c for c in cells if c["batter_skill_quartile"] == 4)
    # n=2, hits=2, probs=[0.7, 0.8]
    # PMF: P(X=0)=0.3*0.2=0.06, P(X=1)=0.7*0.2 + 0.3*0.8=0.38, P(X=2)=0.7*0.8=0.56
    # actual_hits=2: p_lower=1.0, p_upper=0.56
    assert cell_a["n"] == 2
    assert cell_a["hits"] == 2
    assert abs(cell_a["p_lower"] - 1.0) < 1e-9
    assert abs(cell_a["p_upper"] - 0.56) < 1e-9
    assert cell_a["tail_direction"] == "underconfidence"  # observed > expected (1.5)

    # Find cell B (Q=1)
    cell_b = next(c for c in cells if c["batter_skill_quartile"] == 1)
    # n=2, hits=0, probs=[0.6, 0.7]
    # PMF: P(X=0)=0.4*0.3=0.12, P(X=1)=0.6*0.3+0.4*0.7=0.46, P(X=2)=0.6*0.7=0.42
    # actual_hits=0: p_lower=0.12, p_upper=1.0
    assert cell_b["n"] == 2
    assert cell_b["hits"] == 0
    assert abs(cell_b["p_lower"] - 0.12) < 1e-9
    assert abs(cell_b["p_upper"] - 1.0) < 1e-9
    assert cell_b["tail_direction"] == "overconfidence"  # observed < expected (1.3)


def test_cell_pvalues_from_artifact_excludes_pending() -> None:
    """Pending rows (result_status != 'resolved') must not enter any cell."""
    df = _synthetic_cut_c_artifact()
    pending_row = pd.DataFrame([{
        "regime": "R", "slot": "primary", "is_park_driven": False,
        "batter_skill_quartile": 4, "p_game_hit": 0.9, "actual_hit": pd.NA,
        "result_status": "pending",
    }])
    pending_row["is_park_driven"] = pending_row["is_park_driven"].astype("boolean")
    pending_row["batter_skill_quartile"] = pending_row["batter_skill_quartile"].astype("Int64")
    df_with_pending = pd.concat([df, pending_row], ignore_index=True)
    result = cell_pvalues_from_artifact(df_with_pending)
    cell_a = next(c for c in result["cells"] if c["batter_skill_quartile"] == 4)
    assert cell_a["n"] == 2, "pending row must be excluded from cell"


def test_cell_pvalues_from_artifact_excludes_na_required_fields() -> None:
    """Rows where required cell-key fields are NA are excluded; counts reported."""
    df = _synthetic_cut_c_artifact()
    na_row = pd.DataFrame([{
        "regime": "R", "slot": "primary", "is_park_driven": False,
        "batter_skill_quartile": pd.NA, "p_game_hit": 0.5, "actual_hit": True,
        "result_status": "resolved",
    }])
    na_row["is_park_driven"] = na_row["is_park_driven"].astype("boolean")
    na_row["batter_skill_quartile"] = na_row["batter_skill_quartile"].astype("Int64")
    df_with_na = pd.concat([df, na_row], ignore_index=True)
    result = cell_pvalues_from_artifact(df_with_na)
    # Existing 2 cells unchanged
    assert len(result["cells"]) == 2
    # Metadata reports the exclusion
    assert result["excluded_na_rows"] == 1
