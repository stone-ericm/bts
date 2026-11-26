"""Tests for v2_5_attribution.py — nested factorial decomposition for T4.

Tests:
    1. test_parse_p57_string_with_ci  — full string with CI brackets
    2. test_parse_p57_string_without_ci  — bare float, returns (float, nan, nan)
    3. test_compute_attribution_simple_case  — only global_B_effect is nonzero
    4. test_compute_attribution_real_case_v1_v2  — actual v1/v2 endpoints + crafted midpoints
    5. test_compute_attribution_handles_missing_cell_raises  — missing '011' → ValueError
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

# v2_5_attribution.py has dots in its name; use importlib to load it.
def _load_attribution():
    spec = importlib.util.spec_from_file_location(
        "v2_5_attribution",
        str(SCRIPTS_DIR / "v2_5_attribution.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_attribution()
parse_p57_string = _mod.parse_p57_string
compute_attribution = _mod.compute_attribution
format_table = _mod.format_table


# ---------------------------------------------------------------------------
# 1. parse_p57_string with CI brackets
# ---------------------------------------------------------------------------

def test_parse_p57_string_with_ci():
    """'0.0333 [0.0000, 0.1167]' → (0.0333, 0.0, 0.1167)."""
    point, ci_lo, ci_hi = parse_p57_string("0.0333 [0.0000, 0.1167]")
    assert pytest.approx(point) == 0.0333
    assert pytest.approx(ci_lo) == 0.0000
    assert pytest.approx(ci_hi) == 0.1167


# ---------------------------------------------------------------------------
# 2. parse_p57_string without CI brackets
# ---------------------------------------------------------------------------

def test_parse_p57_string_without_ci():
    """'0.0333' (no CI) → (0.0333, nan, nan)."""
    point, ci_lo, ci_hi = parse_p57_string("0.0333")
    assert pytest.approx(point) == 0.0333
    assert math.isnan(ci_lo)
    assert math.isnan(ci_hi)


# ---------------------------------------------------------------------------
# 3. Simple case: only global_B_effect is nonzero
# ---------------------------------------------------------------------------

def test_compute_attribution_simple_case():
    """Synthetic verdicts where only global_B_effect should be nonzero.

    Construction: V010 = V000 + 0.01, all other non-baseline cells equal V000.
    V111 = V000 (so total = 0).

    Expected:
    - global_B_effect = +0.01
    - all other effects = 0 (within float tolerance)
    - total = 0
    """
    base = 0.01
    verdicts = {
        "000": base,
        "010": base + 0.01,   # B switched on under global policy
        "001": base,           # C switched on, B still scalar — same as base
        "011": base,           # B and C on — same as base (no extra benefit)
        "101": base,           # A and C on — same as base
        "111": base,           # all on — same as base
    }
    result = compute_attribution(verdicts)
    d = result["decomposition"]

    assert pytest.approx(d["total_v1_to_v2"], abs=1e-9) == 0.0
    assert pytest.approx(d["global_B_effect"], abs=1e-9) == 0.01
    assert pytest.approx(d["policy_switch_effect"], abs=1e-9) == -0.005  # avg(base,base) - avg(base, base+0.01)
    assert pytest.approx(d["A_effect_given_per_fold"], abs=1e-9) == 0.0
    assert pytest.approx(d["B_effect_given_per_fold"], abs=1e-9) == 0.0
    assert pytest.approx(d["nested_AB_interaction"], abs=1e-9) == 0.0


# ---------------------------------------------------------------------------
# 4. Real-case: actual v1=0.0083 and v2=0.0333 endpoints + crafted midpoints
# ---------------------------------------------------------------------------

def test_compute_attribution_real_case_v1_v2():
    """Synthetic verdicts using the actual v1=0.0083 and v2=0.0333 endpoints.

    Middle cell values are chosen so that the 4-term path sum equals total exactly.
    This requires: -V000/2 + V001 + V010/2 - V011 - V101/2 + V111/2 = 0

    Given V010=0.015, V011=0.020, V101=0.018:
        V001 = V000/2 - V010/2 + V011 + V101/2 - V111/2
             = 0.0083/2 - 0.015/2 + 0.020 + 0.018/2 - 0.0333/2
             ≈ 0.009

    Verify: total = global_B + policy_switch + A_given_per_fold + nested_AB (within 1e-9).
    """
    V000 = 0.0083
    V111 = 0.0333
    V010 = 0.015
    V011 = 0.020
    V101 = 0.018
    # V001 chosen to make 4-term residual = 0
    V001 = V000 / 2 - V010 / 2 + V011 + V101 / 2 - V111 / 2

    verdicts = {
        "000": V000,
        "010": V010,
        "001": V001,
        "011": V011,
        "101": V101,
        "111": V111,
    }
    result = compute_attribution(verdicts)
    d = result["decomposition"]
    sc = result["sanity_check"]

    total = V111 - V000
    assert pytest.approx(d["total_v1_to_v2"], abs=1e-9) == total

    # 4-term path sum must equal total (by construction of V001)
    four_term_sum = (
        d["global_B_effect"]
        + d["policy_switch_effect"]
        + d["A_effect_given_per_fold"]
        + d["nested_AB_interaction"]
    )
    assert pytest.approx(four_term_sum, abs=1e-9) == total, (
        f"4-term sum {four_term_sum} != total {total}; residual={sc['residual']}"
    )
    assert sc["effects_sum_to_total"] is True
    assert abs(sc["residual"]) < 1e-9


# ---------------------------------------------------------------------------
# 5. Missing cell raises ValueError
# ---------------------------------------------------------------------------

def test_compute_attribution_handles_missing_cell_raises():
    """Verdicts dict missing '011' → raises ValueError with informative message."""
    verdicts = {
        "000": 0.01,
        "010": 0.02,
        "001": 0.015,
        # "011" intentionally omitted
        "101": 0.025,
        "111": 0.03,
    }
    with pytest.raises(ValueError, match="011"):
        compute_attribution(verdicts)
