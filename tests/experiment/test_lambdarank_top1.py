"""Smoke tests for LambdaRankTop1Experiment."""

from bts.experiment.registry import load_all_experiments, get_experiment
from bts.experiment.runner_factored import _is_eligible_for_model_swap_fast_path
from bts.model.predict import BLEND_CONFIGS


def test_lambdarank_top1_registered():
    load_all_experiments()
    exp = get_experiment("lambdarank_top1")
    assert exp.category == "model"
    desc = exp.description.lower()
    assert "top-1" in desc or "ndcg@1" in desc


def test_lambdarank_top1_eligible_for_model_swap_fast_path():
    """Should append a 13th config (model-swap eligible)."""
    load_all_experiments()
    exp = get_experiment("lambdarank_top1")
    eligible, reason = _is_eligible_for_model_swap_fast_path(exp)
    assert eligible, f"lambdarank_top1 should be eligible but rejected: {reason}"


def test_lambdarank_top1_blend_config_has_top1_params():
    load_all_experiments()
    exp = get_experiment("lambdarank_top1")
    new_configs = exp.modify_blend_configs(list(BLEND_CONFIGS))
    # Should have one more config than baseline
    assert len(new_configs) == len(BLEND_CONFIGS) + 1
    # The new config's extra_params should pin top-1 ranking specifics
    name, _cols, extra = new_configs[-1]
    assert name == "lambdarank_top1"
    assert extra["objective"] == "lambdarank"
    assert extra["eval_at"] == [1]
    assert extra["ndcg_eval_at"] == [1]
    assert extra["label_gain"] == [0, 1]
