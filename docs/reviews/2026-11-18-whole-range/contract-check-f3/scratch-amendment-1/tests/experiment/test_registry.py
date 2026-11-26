from bts.experiment.base import ExperimentDef
from bts.experiment.registry import (
    EXPERIMENTS,
    get_experiment,
    list_experiments,
    register,
)


class _DummyExperiment(ExperimentDef):
    pass


def test_register_and_retrieve():
    exp = _DummyExperiment(
        name="_test_dummy", phase=1, category="feature", description="test",
    )
    register(exp)
    assert get_experiment("_test_dummy") is exp
    # Cleanup
    EXPERIMENTS.pop("_test_dummy", None)


def test_get_experiment_missing():
    try:
        get_experiment("nonexistent_xyz")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_list_experiments_by_phase():
    phase0 = list_experiments(phase=0)
    phase1 = list_experiments(phase=1)
    assert all(e.phase == 0 for e in phase0)
    assert all(e.phase == 1 for e in phase1)


def test_list_experiments_by_category():
    features = list_experiments(category="feature")
    assert all(e.category == "feature" for e in features)


def test_list_experiments_no_filter():
    all_exps = list_experiments()
    assert isinstance(all_exps, list)


def test_stub_experiments_unregistered():
    """quantile_gated_skip is a stub (identity modify_strategy) and copula_doubles
    mutates quality_bins the runner discards. Both are intentionally unregistered
    so they don't appear as misleading 0pp "FAIL" entries in Phase 1 re-screens."""
    from bts.experiment.registry import load_all_experiments, EXPERIMENTS
    load_all_experiments()
    assert "quantile_gated_skip" not in EXPERIMENTS
    assert "copula_doubles" not in EXPERIMENTS
