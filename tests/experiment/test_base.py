from bts.experiment.base import ExperimentDef
import pandas as pd


def test_experiment_def_defaults():
    exp = ExperimentDef(
        name="test_exp",
        phase=1,
        category="feature",
        description="A test experiment",
    )
    assert exp.name == "test_exp"
    assert exp.phase == 1
    assert exp.dependencies == []


def test_modify_features_passthrough(mini_pa_df):
    exp = ExperimentDef(
        name="noop", phase=1, category="feature", description="no-op",
    )
    result = exp.modify_features(mini_pa_df)
    assert result is mini_pa_df


def test_modify_blend_configs_passthrough():
    exp = ExperimentDef(
        name="noop", phase=1, category="model", description="no-op",
    )
    configs = [("baseline", ["col_a", "col_b"])]
    result = exp.modify_blend_configs(configs)
    assert result is configs


def test_feature_cols_default_none():
    exp = ExperimentDef(
        name="noop", phase=1, category="feature", description="no-op",
    )
    assert exp.feature_cols() is None


def test_run_diagnostic_raises():
    exp = ExperimentDef(
        name="noop", phase=0, category="diagnostic", description="no-op",
    )
    try:
        exp.run_diagnostic(pd.DataFrame(), {})
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass


def test_touches_features_false_for_base():
    exp = ExperimentDef(
        name="noop", phase=1, category="feature", description="no-op",
    )
    assert exp.touches_features() is False


def test_touches_features_true_for_override():
    class _FeatureExp(ExperimentDef):
        def modify_features(self, df):
            df["new_col"] = 1
            return df

    exp = _FeatureExp(
        name="feat", phase=1, category="feature", description="adds a feature",
    )
    assert exp.touches_features() is True
