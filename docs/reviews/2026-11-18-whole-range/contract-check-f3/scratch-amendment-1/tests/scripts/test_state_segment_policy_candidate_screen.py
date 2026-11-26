import numpy as np
import pytest

from scripts.rolling_origin_policy_candidate_screen import CandidateSpec
from scripts.state_segment_policy_candidate_screen import (
    StateSegment,
    apply_state_segment,
    candidate_name,
    default_segments,
    fdr_table,
    segment_mask,
    table_diagnostics,
)


def test_default_segments_match_pre_specified_fdr_grid():
    segments = default_segments(n_bins=5)

    assert len(segments) == 45
    assert min(segment.day_min for segment in segments) == 1
    assert max(segment.day_max for segment in segments) == 180
    assert {segment.q_bins for segment in segments} == {
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
    }
    assert any(segment.name == "mid_d31_90_s10_29_q4" for segment in segments)
    with pytest.raises(ValueError, match="5 quality bins"):
        default_segments(n_bins=4)


def test_segment_mask_selects_days_streaks_saver_states_and_quality_bins():
    segment = StateSegment(
        name="sample",
        day_min=2,
        day_max=4,
        streak_min=1,
        streak_max=3,
        q_bins=(0, 2),
    )

    mask = segment_mask(segment, (5, 6, 2, 4))

    assert mask.sum() == 3 * 3 * 2 * 2
    assert mask[1, 2, 0, 0]
    assert mask[3, 4, 1, 2]
    assert not mask[0, 2, 0, 0]
    assert not mask[1, 1, 0, 0]
    assert not mask[1, 2, 0, 1]


def test_apply_state_segment_preserves_production_outside_mask():
    prod = np.zeros((5, 6, 2, 4), dtype=int)
    base = np.ones((5, 6, 2, 4), dtype=int)
    segment = StateSegment(
        name="sample",
        day_min=2,
        day_max=4,
        streak_min=1,
        streak_max=3,
        q_bins=(0, 2),
    )

    table = apply_state_segment(base_table=base, prod_table=prod, segment=segment)
    mask = segment_mask(segment, prod.shape)

    assert np.all(table[mask] == 1)
    assert np.all(table[~mask] == 0)


def test_candidate_name_and_table_diagnostics_are_stable():
    prod = np.zeros((5, 6, 2, 4), dtype=int)
    base = np.zeros((5, 6, 2, 4), dtype=int)
    base[1:4, 2:5, :, 0] = 1
    segment = StateSegment(
        name="sample",
        day_min=2,
        day_max=4,
        streak_min=1,
        streak_max=3,
        q_bins=(0, 2),
    )
    table = apply_state_segment(base_table=base, prod_table=prod, segment=segment)

    assert candidate_name(CandidateSpec("cumulative", "cumulative"), segment) == (
        "cumulative__sample"
    )
    diagnostics = table_diagnostics(table=table, prod_table=prod, segment=segment)

    assert diagnostics["segment_states"] == 36
    assert diagnostics["n_changed_states"] == 18
    assert diagnostics["n_changed_states_in_segment"] == 18
    assert diagnostics["changed_fraction_of_segment"] == pytest.approx(0.5)


def test_fdr_table_applies_bh_by_and_end_state():
    summaries = {
        "winner": {
            "mean_gap": 0.1,
            "n_positive": 10,
            "n": 10,
            "bootstrap": {"p_one_sided_positive": 0.001},
        },
        "loser": {
            "mean_gap": -0.1,
            "n_positive": 0,
            "n": 10,
            "bootstrap": {"p_one_sided_positive": 0.9},
        },
    }

    result = fdr_table(summaries)

    winner = next(row for row in result["rows"] if row["candidate"] == "winner")
    loser = next(row for row in result["rows"] if row["candidate"] == "loser")
    assert result["m"] == 2
    assert result["n_survive_BH_0_05"] == 1
    assert result["n_survive_BY_0_05"] == 1
    assert result["end_state_by_BH"] == "E2_freeze_surviving_segments_for_fresh_lockbox"
    assert winner["q_BH"] == pytest.approx(0.002)
    assert winner["survives_BH_0_05"]
    assert not loser["survives_BH_0_05"]
