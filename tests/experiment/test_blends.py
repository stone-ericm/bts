from bts.experiment.blends import (
    FWLSExperiment,
    FixedShareHedgeExperiment,
    CopulaDoublesExperiment,
)


def test_fwls_experiment_metadata():
    exp = FWLSExperiment()
    assert exp.name == "fwls"
    assert exp.category == "blend"
    assert exp.phase == 1


def test_hedge_experiment_metadata():
    exp = FixedShareHedgeExperiment()
    assert exp.name == "fixed_share_hedge"
    assert exp.category == "blend"


def test_copula_experiment_metadata():
    exp = CopulaDoublesExperiment()
    assert exp.name == "copula_doubles"
    assert exp.category == "blend"
