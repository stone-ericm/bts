from __future__ import annotations

import json

import pytest

from scripts.determinism_gap_bound import (
    build_report,
    main,
    summarize_deterministic_feature_delta_screen,
    summarize_distribution_shift,
    summarize_gap_section,
)


def _gap_section() -> dict:
    return {
        "gaps_by_seed": [
            {"seed": 1, "gap": 0.03},
            {"seed": 2, "gap": 0.01},
            {"seed": 3, "gap": 0.02},
        ],
    }


def _gap_ci() -> dict:
    return {
        "leave_one_out": _gap_section(),
        "within_pool": _gap_section(),
    }


def _deterministic_baseline() -> dict:
    return {
        "corpus": "test deterministic baseline",
        "flags": {"BTS_LGBM_DETERMINISTIC": 1},
        "seed_pool_size": 100,
        "metrics": {
            "p_57_mdp": {"mean": 0.031, "std": 0.012, "n": 100},
            "p_at_1_avg": {"mean": 0.856, "std": 0.007, "n": 100},
        },
        "comparison_non_deterministic_prior": {
            "p_57_mdp": {"mean": 0.030, "std": 0.010},
            "p_at_1_avg": {"mean": 0.855, "std": 0.005},
            "verdict": "no shift detected",
        },
    }


def _deterministic_screen() -> dict:
    return {
        "corpus": "test deterministic screen",
        "flags": {"BTS_LGBM_DETERMINISTIC": 1},
        "n_seeds": 10,
        "n_experiments": 2,
        "results": {
            "feature_a": {
                "pooled": {
                    "delta_p_57_mdp": {
                        "n": 10,
                        "mean": 0.01,
                        "std": 0.02,
                        "se": 0.006324555320336759,
                        "t": 1.5811388300841895,
                    },
                },
            },
            "feature_b": {
                "pooled": {
                    "delta_p_57_mdp": {
                        "n": 10,
                        "mean": 0.0,
                        "std": 0.0,
                        "se": 0.0,
                        "t": 0.0,
                    },
                },
            },
        },
    }


def test_summarize_gap_section_reports_sign_flip_margin():
    summary = summarize_gap_section(_gap_section())

    assert summary["n"] == 3
    assert summary["mean_gap"] == 0.02
    assert summary["n_positive"] == 3
    assert summary["n_negative"] == 0
    assert summary["sign_flip_margin"] == 0.01


def test_summarize_distribution_shift_reports_z_score():
    shift = summarize_distribution_shift(_deterministic_baseline(), "p_57_mdp")

    assert shift["mean_delta"] == pytest.approx(0.001)
    assert shift["z_vs_prior_std"] == pytest.approx(0.1)


def test_summarize_deterministic_feature_delta_screen_compares_reference_std():
    summary = summarize_deterministic_feature_delta_screen(
        _deterministic_screen(),
        reference_std=0.015,
    )

    assert summary["n_experiments"] == 2
    assert summary["std_max"] == 0.02
    assert summary["n_experiments_std_ge_reference"] == 1
    assert summary["top_by_std"][0]["experiment"] == "feature_a"


def test_build_report_keeps_direct_bound_distinct_from_shift_screen():
    report = build_report(_gap_ci(), _deterministic_baseline(), _deterministic_screen())

    assert report["methodology"]["direct_paired_bound_available"] is False
    assert report["verdict"]["iid_seed_assumption_verdict"] == "not_evaluable_from_existing_artifacts"
    assert report["verdict"]["c0_determinism_caveat_resolved"] is False
    assert report["verdict"]["pooled_gap_screen_status"] == "unchanged"
    assert report["deterministic_feature_delta_proxy"]["n_experiments"] == 2
    assert "paired same-seed" in report["verdict"]["next_required_evidence"][0]


def test_cli_writes_report(tmp_path):
    gap_path = tmp_path / "gap.json"
    det_path = tmp_path / "det.json"
    screen_path = tmp_path / "screen.json"
    out_path = tmp_path / "out.json"
    gap_path.write_text(json.dumps(_gap_ci()))
    det_path.write_text(json.dumps(_deterministic_baseline()))
    screen_path.write_text(json.dumps(_deterministic_screen()))

    rc = main([
        "--gap-ci", str(gap_path),
        "--deterministic-baseline", str(det_path),
        "--deterministic-screen", str(screen_path),
        "--out", str(out_path),
    ])

    assert rc == 0
    data = json.loads(out_path.read_text())
    assert data["inputs"]["pooled_policy_gap_ci"] == str(gap_path)
    assert data["inputs"]["deterministic_feature_delta_screen"] == str(screen_path)
    assert data["verdict"]["status"] == "distribution_shift_not_detected_but_direct_bound_missing"
