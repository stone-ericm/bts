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


def test_vrex_modifies_training_params():
    exp = VRExExperiment()
    params = {"n_estimators": 200, "max_depth": 6}
    new_params = exp.modify_training_params(params)
    assert "vrex_beta" in new_params
    assert new_params["vrex_beta"] == 10.0
    assert new_params["n_estimators"] == 200  # original preserved
