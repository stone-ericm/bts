from bts.experiment.models import (
    LambdaRankExperiment,
    CatBoostExperiment,
    XENDCGExperiment,
    VRExExperiment,
)
from bts.model.predict import BLEND_CONFIGS


def test_lambdarank_adds_blend_member():
    exp = LambdaRankExperiment()
    configs = list(BLEND_CONFIGS)
    new_configs = exp.modify_blend_configs(configs)
    names = [c[0] for c in new_configs]
    assert "lambdarank" in names
    assert len(new_configs) == len(BLEND_CONFIGS) + 1


def test_catboost_adds_blend_member():
    exp = CatBoostExperiment()
    configs = list(BLEND_CONFIGS)
    new_configs = exp.modify_blend_configs(configs)
    names = [c[0] for c in new_configs]
    assert "catboost" in names


def test_xendcg_adds_blend_member():
    exp = XENDCGExperiment()
    configs = list(BLEND_CONFIGS)
    new_configs = exp.modify_blend_configs(configs)
    names = [c[0] for c in new_configs]
    assert "xendcg" in names


def test_vrex_adds_blend_member_with_extra_params():
    """VRExExperiment must route through modify_blend_configs so that
    vrex_beta lands in the per-config extra_params dict. The training
    dispatch in backtest_blend.py checks extras, not global lgb_params."""
    exp = VRExExperiment()
    configs = list(BLEND_CONFIGS)
    new_configs = exp.modify_blend_configs(configs)
    assert len(new_configs) == len(configs) + 1
    name, cols, extras = new_configs[-1]
    assert name == "vrex"
    assert "vrex_beta" in extras
    assert extras["vrex_beta"] == 10.0
    assert extras["vrex_rounds"] == 5
