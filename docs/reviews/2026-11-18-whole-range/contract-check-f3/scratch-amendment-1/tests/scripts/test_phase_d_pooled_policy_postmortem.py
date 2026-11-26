import numpy as np
import pandas as pd
import pytest

from scripts.phase_d_pooled_policy_postmortem import (
    action_counts,
    action_transition_counts,
    compare_bin_summaries,
    rank_pair_metrics,
)


def test_action_counts_uses_decision_states_only():
    table = np.zeros((58, 3, 2, 2), dtype=int)
    table[:57, 1:, :, :] = 1
    table[57, :, :, :] = 2
    table[:, 0, :, :] = 2

    counts = action_counts(table)

    assert counts == {"single": 57 * 2 * 2 * 2}


def test_action_transition_counts_reports_differences():
    reference = np.zeros((58, 3, 2, 2), dtype=int)
    candidate = reference.copy()
    candidate[0, 1, 0, 0] = 1
    candidate[1, 2, 1, 1] = 2

    result = action_transition_counts(reference, candidate)

    assert result["total_decision_states"] == 57 * 2 * 2 * 2
    assert result["differing_states"] == 2
    assert result["transition_counts"]["skip->single"] == 1
    assert result["transition_counts"]["skip->double"] == 1


def test_compare_bin_summaries_reports_outer_minus_selection_deltas():
    selection = {
        "boundaries": [0.5],
        "bins": [
            {"index": 0, "p_hit": 0.6, "p_both": 0.3, "frequency": 0.4},
            {"index": 1, "p_hit": 0.8, "p_both": 0.5, "frequency": 0.6},
        ],
    }
    outer = {
        "boundaries": [0.55],
        "bins": [
            {"index": 0, "p_hit": 0.7, "p_both": 0.35, "frequency": 0.3},
            {"index": 1, "p_hit": 0.9, "p_both": 0.7, "frequency": 0.7},
        ],
    }

    result = compare_bin_summaries(selection, outer)

    assert result["boundary_deltas"] == pytest.approx([0.05])
    assert result["bins"][0]["delta_p_hit"] == pytest.approx(0.1)
    assert result["bins"][1]["delta_p_both"] == pytest.approx(0.2)
    assert result["bins"][0]["delta_frequency"] == pytest.approx(-0.1)


def test_rank_pair_metrics_summarizes_seed_date_pairs():
    profiles = pd.DataFrame([
        {"seed": 1, "season": 2025, "date": "2025-04-01", "rank": 1, "p_game_hit": 0.8, "actual_hit": 1},
        {"seed": 1, "season": 2025, "date": "2025-04-01", "rank": 2, "p_game_hit": 0.7, "actual_hit": 1},
        {"seed": 1, "season": 2025, "date": "2025-04-02", "rank": 1, "p_game_hit": 0.6, "actual_hit": 0},
        {"seed": 1, "season": 2025, "date": "2025-04-02", "rank": 2, "p_game_hit": 0.5, "actual_hit": 1},
        {"seed": 2, "season": 2025, "date": "2025-04-01", "rank": 1, "p_game_hit": 0.9, "actual_hit": 1},
        {"seed": 2, "season": 2025, "date": "2025-04-01", "rank": 2, "p_game_hit": 0.4, "actual_hit": 0},
    ])

    result = rank_pair_metrics(profiles)

    assert result["n_seed_dates"] == 3
    assert result["n_seeds"] == 2
    assert result["rank1_actual_hit_rate"] == pytest.approx(2 / 3)
    assert result["rank2_actual_hit_rate"] == pytest.approx(2 / 3)
    assert result["rank1_rank2_both_hit_rate"] == pytest.approx(1 / 3)
