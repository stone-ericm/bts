import numpy as np
import pandas as pd
import pytest

from scripts.rolling_origin_policy_candidate_screen import (
    CandidateSpec,
    apply_phase_scope,
    compute_weighted_pooled_bins,
    season_decay_weights,
    summarize_gaps,
    weighted_quantile,
)


def test_season_decay_weights_halves_by_distance():
    weights = season_decay_weights([2021, 2022, 2023], half_life=1.0)

    assert weights == {
        2021: pytest.approx(0.25),
        2022: pytest.approx(0.5),
        2023: pytest.approx(1.0),
    }


def test_weighted_quantile_moves_toward_heavier_values():
    values = np.array([0.0, 10.0])
    equal = weighted_quantile(values, np.array([1.0, 1.0]), [0.5])
    heavy_high = weighted_quantile(values, np.array([1.0, 9.0]), [0.5])

    assert equal[0] == pytest.approx(0.0)
    assert heavy_high[0] > equal[0]


def test_compute_weighted_pooled_bins_preserves_seed_date_pairs():
    profiles = pd.DataFrame([
        {"seed": 1, "season": 2022, "date": "2022-04-01", "rank": 1, "p_game_hit": 0.10, "actual_hit": 1},
        {"seed": 1, "season": 2022, "date": "2022-04-01", "rank": 2, "p_game_hit": 0.09, "actual_hit": 1},
        {"seed": 2, "season": 2022, "date": "2022-04-01", "rank": 1, "p_game_hit": 0.20, "actual_hit": 0},
        {"seed": 2, "season": 2022, "date": "2022-04-01", "rank": 2, "p_game_hit": 0.19, "actual_hit": 1},
        {"seed": 1, "season": 2023, "date": "2023-04-01", "rank": 1, "p_game_hit": 0.90, "actual_hit": 1},
        {"seed": 1, "season": 2023, "date": "2023-04-01", "rank": 2, "p_game_hit": 0.89, "actual_hit": 0},
        {"seed": 2, "season": 2023, "date": "2023-04-01", "rank": 1, "p_game_hit": 0.95, "actual_hit": 1},
        {"seed": 2, "season": 2023, "date": "2023-04-01", "rank": 2, "p_game_hit": 0.94, "actual_hit": 1},
    ])

    bins = compute_weighted_pooled_bins(
        profiles,
        season_weights={2022: 1.0, 2023: 2.0},
        n_bins=2,
    )

    assert len(bins.bins) == 2
    assert sum(bin_.frequency for bin_ in bins.bins) == pytest.approx(1.0)
    assert bins.bins[1].p_hit == pytest.approx(1.0)
    assert 0.0 <= bins.bins[0].p_both <= 1.0


def test_summarize_gaps_counts_signs():
    rows = [
        {"gap": 0.1},
        {"gap": -0.2},
        {"gap": 0.0},
    ]

    result = summarize_gaps(rows)

    assert result["n"] == 3
    assert result["n_positive"] == 1
    assert result["n_negative"] == 1
    assert result["n_zero"] == 1
    assert result["mean_gap"] == pytest.approx(-1 / 30)


def test_summarize_gaps_can_emit_bootstrap_ci():
    rows = [
        {"gap": 0.1},
        {"gap": 0.2},
        {"gap": 0.3},
    ]

    result = summarize_gaps(rows, n_bootstrap=200, seed=7)

    assert result["bootstrap"]["kind"] == "iid_seed_fold_bootstrap"
    assert result["bootstrap"]["ci_lower"] > 0
    assert result["bootstrap"]["prob_mean_gt_zero"] == 1.0


def test_apply_phase_scope_builds_production_anchored_hybrids():
    prod = np.zeros((58, 6, 2, 2), dtype=int)
    candidate = np.ones((58, 6, 2, 2), dtype=int)

    late_only = apply_phase_scope(
        spec=CandidateSpec("late", "cumulative", phase_scope="late_only"),
        base_table=candidate,
        prod_table=prod,
        late_phase_days=2,
    )
    assert np.all(late_only[:, 1:3, :, :] == 1)
    assert np.all(late_only[:, 3:, :, :] == 0)
    assert np.all(late_only[:, 0, :, :] == 0)

    early_only = apply_phase_scope(
        spec=CandidateSpec("early", "cumulative", phase_scope="early_only"),
        base_table=candidate,
        prod_table=prod,
        late_phase_days=2,
    )
    assert np.all(early_only[:, 1:3, :, :] == 0)
    assert np.all(early_only[:, 3:, :, :] == 1)
    assert np.all(early_only[:, 0, :, :] == 0)
