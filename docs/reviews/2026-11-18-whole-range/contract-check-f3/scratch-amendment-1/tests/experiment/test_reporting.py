from bts.experiment.reporting import format_phase1_table, format_phase2_log


def test_format_phase1_table():
    results = [
        {
            "name": "eb_shrinkage",
            "passed": True,
            "reason": "P@1 improves on both seasons",
            "diff": {
                "p_at_1_by_season": {
                    "2024": {"delta": 0.003},
                    "2025": {"delta": 0.002},
                },
                "p_57_mdp": {"delta": 0.008},
            },
        },
        {
            "name": "catboost_blend",
            "passed": False,
            "reason": "P@1 drops on 2025",
            "diff": {
                "p_at_1_by_season": {
                    "2024": {"delta": 0.001},
                    "2025": {"delta": -0.005},
                },
                "p_57_mdp": {"delta": -0.001},
            },
        },
    ]
    table = format_phase1_table(results)
    assert "eb_shrinkage" in table
    assert "catboost_blend" in table
    assert "✓" in table
    assert "✗" in table
    assert "Winners: 1/2" in table


def test_format_phase2_log():
    selection_result = {
        "forward_log": [
            {"name": "eb_shrinkage", "p57_before": 0.0891, "p57_after": 0.0950, "delta": 0.0059, "kept": True},
            {"name": "kl_div", "p57_before": 0.0950, "p57_after": 0.0940, "delta": -0.0010, "kept": False},
        ],
        "backward_log": [
            {"name": "eb_shrinkage", "p57_with": 0.0950, "p57_without": 0.0891, "delta": 0.0059, "kept": True},
        ],
        "included": ["eb_shrinkage"],
        "final_scorecard": {"p_57_mdp": 0.0950},
    }
    log = format_phase2_log(selection_result)
    assert "eb_shrinkage" in log
    assert "KEPT" in log
    assert "DROP" in log
