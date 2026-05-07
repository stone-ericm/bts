import json
from pathlib import Path

import pytest

from scripts.pooled_policy_gap_ci import build_report, summarize_gaps


def test_summarize_gaps_reports_exact_sign_test_and_bootstrap_ci():
    rows = [
        {"seed": 1, "v_prod": 0.01, "v_loo": 0.03},
        {"seed": 2, "v_prod": 0.02, "v_loo": 0.05},
        {"seed": 3, "v_prod": 0.03, "v_loo": 0.04},
        {"seed": 4, "v_prod": 0.04, "v_loo": 0.06},
    ]

    result = summarize_gaps(rows, variant_key="v_loo", n_bootstrap=500, seed=7)

    assert result["n"] == 4
    assert result["mean_gap"] == pytest.approx(0.02)
    assert result["n_positive"] == 4
    assert result["n_negative"] == 0
    assert result["exact_sign_p_two_sided"] == pytest.approx(0.125)
    assert result["bootstrap"]["ci_lower"] > 0
    assert result["bootstrap"]["prob_mean_gt_zero"] == 1.0


def test_build_report_labels_artifact_level_not_block_bootstrap(tmp_path: Path):
    artifact = {
        "within_pool": [
            {"seed": 1, "v_prod": 0.01, "v_pool": 0.02},
            {"seed": 2, "v_prod": 0.01, "v_pool": 0.03},
        ],
        "leave_one_out": [
            {"seed": 1, "v_prod": 0.01, "v_loo": 0.02},
            {"seed": 2, "v_prod": 0.01, "v_loo": 0.03},
        ],
        "within_pool_summary": {"n_seeds": 2},
        "leave_one_out_summary": {"n_seeds": 2},
    }
    path = tmp_path / "pooled_policy_ab.json"
    path.write_text(json.dumps(artifact))

    report = build_report(path, n_bootstrap=100, seed=42)

    assert report["schema_version"] == "pooled_policy_gap_ci_v1"
    assert report["methodology"]["is_profile_block_bootstrap"] is False
    assert "does not address day-level dependence" in report["methodology"]["limitation"]
    assert report["leave_one_out"]["screen_verdict"] == "positive_screen_unchanged"
    assert report["leave_one_out"]["deployment_ready"] is False
    assert report["leave_one_out"]["mean_gap"] == pytest.approx(0.015)
