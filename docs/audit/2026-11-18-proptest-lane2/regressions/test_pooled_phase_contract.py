import pandas as pd
import pytest

from bts.simulate.mdp import lookup_action
from bts.simulate.pooled_policy import build_pooled_policy


def profiles(*probabilities):
    return pd.DataFrame([
        (f"2027-04-{i:02d}", 2027, 1, rank, p, rank == 1 or i % 2 == 1)
        for i, p in enumerate(probabilities, 1) for rank in (1, 2)
    ], columns=["date", "season", "seed", "rank", "p_game_hit", "actual_hit"])


@pytest.mark.parametrize(("probs", "late_days"), [
    ((0.1, 0.1), 1),
    ((0.1, 0.1, 0.2), 2),
    ((0.1, 0.2, 0.1), 1),
])
def test_phase_table_matches_lookup_width(probs, late_days):
    solved = build_pooled_policy(profiles(*probs), season_length=late_days + 1,
                                 late_phase_days=late_days)
    assert solved.policy_table.shape[-1] == len(solved.quality_bins.boundaries) + 1


def test_late_column_uses_late_cutpoints():
    solved = build_pooled_policy(profiles(0.1, 0.2, 0.8, 0.9), season_length=3,
                                 late_phase_days=2, n_bins=2)
    assert solved.policy(55, 1, True, 0) == "double"
    assert solved.policy(55, 1, True, 1) == "skip"
    assert lookup_action(solved.policy_table, solved.quality_bins.boundaries,
                         55, 1, True, 0.8, 3) == "double"
