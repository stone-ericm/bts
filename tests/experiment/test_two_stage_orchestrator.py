"""Unit tests for two-stage screening decision logic."""
import pytest

from bts.experiment.two_stage import (
    decide_after_stage_one,
    StageOneResult,
)


def test_kills_experiments_with_mostly_negative_deltas():
    # 8 seeds, 1 win, 7 losses
    result = StageOneResult(name="wind_vector", seeds_run=8, wins=1, mean_delta=-0.005)
    decision = decide_after_stage_one(result)
    assert decision.action == "kill"
    assert decision.posterior_p_gt_half < 0.15


def test_fast_tracks_experiments_with_strong_positive_signal():
    result = StageOneResult(name="savant_xba_30g", seeds_run=8, wins=8, mean_delta=0.006)
    decision = decide_after_stage_one(result)
    assert decision.action == "fast_track_ship_candidate"


def test_promotes_borderline_to_stage_two():
    result = StageOneResult(name="eb_shrinkage", seeds_run=8, wins=5, mean_delta=0.0015)
    decision = decide_after_stage_one(result)
    assert decision.action == "promote"


def test_promotes_strong_signal_without_positive_mean_delta():
    """Positive win rate but near-zero effect size should promote, not fast-track."""
    result = StageOneResult(name="marginal", seeds_run=8, wins=7, mean_delta=0.0002)
    decision = decide_after_stage_one(result)
    assert decision.action == "promote"  # mean_delta too small for fast track


def test_aggregate_from_fake_seed_dirs(tmp_path):
    """Create fake diff.json files across 4 seed dirs, verify aggregation + decision."""
    import json
    from bts.experiment.two_stage import aggregate_stage_one_results

    seeds = [tmp_path / f"seed_{i}" for i in range(4)]
    for seed_dir in seeds:
        (seed_dir / "phase1" / "good_exp").mkdir(parents=True)
        (seed_dir / "phase1" / "good_exp" / "diff.json").write_text(
            json.dumps({"precision": {"1": {"delta": 0.005}}})
        )
        (seed_dir / "phase1" / "bad_exp").mkdir(parents=True)
        (seed_dir / "phase1" / "bad_exp" / "diff.json").write_text(
            json.dumps({"precision": {"1": {"delta": -0.003}}})
        )

    results = aggregate_stage_one_results(seeds, ["good_exp", "bad_exp"])
    assert results["good_exp"].wins == 4
    assert results["good_exp"].seeds_run == 4
    assert results["bad_exp"].wins == 0
