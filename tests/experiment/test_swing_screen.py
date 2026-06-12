"""Tests for the Stage-1 screen arm registry (swing campaign)."""
import numpy as np
import pandas as pd

from bts.experiment.swing_screen import (
    ARMS,
    FAMILY_OF,
    build_arm_frame,
)


def _pa_frame():
    # PA rows already carrying rolling swing features (as attach would produce)
    return pd.DataFrame({
        "batter_id": [1, 2], "pitcher_id": [9, 9],
        "date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
        "season": [2024, 2024], "is_hit": [1, 0],
        "batter_miss_dist_7g": [2.0, 3.0],
        "batter_miss_dist_60g": [2.5, 2.5],
        "batter_intercept_y_7g": [31.0, 30.0],
        "batter_intercept_y_60g": [30.0, 30.0],
        "batter_swing_len_7g": [7.1, 7.0],
        "batter_swing_len_60g": [7.0, 7.0],
        "batter_whiff_high_share_30g": [0.6, 0.4],
        "pitcher_whiff_high_share_30g": [0.7, 0.7],
    })


def test_registry_arm_names_unique_and_families_mapped():
    assert len(ARMS) == len(set(ARMS))
    assert "baseline" in ARMS
    for arm in ARMS:
        assert arm in FAMILY_OF
    assert {"P", "B", "T", "S", "M", "omnibus", "control", "baseline"} >= set(FAMILY_OF.values())


def test_derived_drift_and_interaction_features():
    pa = _pa_frame()
    frame, cols = build_arm_frame("t_intercept_drift", pa)
    assert cols == ["t_intercept_drift"]
    assert abs(frame["t_intercept_drift"].iloc[0] - 1.0) < 1e-9  # 31-30

    frame, cols = build_arm_frame("m_high_alignment", pa)
    assert abs(frame["m_high_alignment"].iloc[0] - 0.42) < 1e-9  # 0.6*0.7


def test_baseline_arm_adds_no_columns():
    pa = _pa_frame()
    frame, cols = build_arm_frame("baseline", pa)
    assert cols == []


def test_permuted_control_preserves_values_but_breaks_dates():
    pa = pd.DataFrame({
        "batter_id": [1] * 6, "pitcher_id": [9] * 6,
        "date": pd.to_datetime([f"2024-05-{d:02d}" for d in range(1, 7)]),
        "season": 2024, "is_hit": 1,
        "batter_miss_dist_30g": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    # registry permutes within entity with a fixed seed
    frame, cols = build_arm_frame("ctl_permuted", pa, permute_seed=7)
    # permuted copies carry the VARIANT name (b_miss_30g <- batter_miss_dist_30g)
    col = [c for c in cols if c == "perm_b_miss_30g"]
    assert col, "permuted control must include permuted copies of omnibus features"
    vals = sorted(frame[col[0]].tolist())
    assert vals == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]      # marginal preserved
    assert frame[col[0]].tolist() != pa["batter_miss_dist_30g"].tolist()  # order broken


def test_placebo_control_is_flags_only():
    pa = _pa_frame()
    frame, cols = build_arm_frame("ctl_placebo", pa)
    assert all(c.startswith("has_") for c in cols)
    assert all(frame[c].dtype == bool for c in cols)
