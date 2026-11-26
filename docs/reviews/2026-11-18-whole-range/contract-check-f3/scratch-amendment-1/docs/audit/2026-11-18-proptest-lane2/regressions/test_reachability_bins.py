from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from bts.simulate.mdp import load_policy, lookup_action, solve_mdp
from bts.simulate.pooled_policy import build_pooled_policy, compute_pooled_bins
from bts.simulate.quality_bins import compute_bins


def profiles(*probabilities):
    return pd.DataFrame([
        (f"2027-04-{i:02d}", 1, rank, p, rank == 1 or i % 2 == 1)
        for i, p in enumerate(probabilities, 1) for rank in (1, 2)
    ], columns=["date", "seed", "rank", "p_game_hit", "actual_hit"])


@pytest.mark.parametrize("build_bins", [compute_bins, compute_pooled_bins])
def test_single_profile_policy_can_be_used_after_reload(build_bins):
    bins = build_bins(profiles(0.1))
    solved = solve_mdp(bins, season_length=1)
    with TemporaryDirectory() as directory:
        table, boundaries, days = load_policy(solved.save(Path(directory) / "mdp.npz"))
        assert lookup_action(table, boundaries, 55, 1, True, 0.1, days) == "double"


def test_phase_training_retains_all_lookup_columns():
    solved = build_pooled_policy(profiles(0.1, 0.1), season_length=2, late_phase_days=1)
    assert solved.policy_table.shape[-1] == len(solved.quality_bins.boundaries) + 1


def test_late_quality_cutpoint_used_by_lookup():
    solved = build_pooled_policy(profiles(0.1, 0.2, 0.8, 0.9), season_length=2,
                                 late_phase_days=2, n_bins=2)
    assert solved.policy(55, 1, True, 0) == "double"
    assert solved.policy(55, 1, True, 1) == "skip"
    assert lookup_action(solved.policy_table, solved.quality_bins.boundaries,
                         55, 1, True, 0.8, 2) == "double"
