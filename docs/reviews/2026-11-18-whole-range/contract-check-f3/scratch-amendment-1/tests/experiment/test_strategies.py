from bts.experiment.strategies import (
    DecisionCalibrationExperiment,
    QuantileGatedSkipExperiment,
)


def test_decision_calibration_metadata():
    exp = DecisionCalibrationExperiment()
    assert exp.name == "decision_calibration"
    assert exp.category == "strategy"
    assert exp.phase == 1


def test_quantile_gated_skip_metadata():
    exp = QuantileGatedSkipExperiment()
    assert exp.name == "quantile_gated_skip"
    assert exp.category == "strategy"
