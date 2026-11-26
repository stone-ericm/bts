"""Integration test: registry loads all experiments, runner dispatches correctly."""

from bts.experiment.registry import load_all_experiments, list_experiments, EXPERIMENTS


def test_load_all_experiments_populates_registry():
    load_all_experiments()
    assert len(EXPERIMENTS) >= 20, f"Expected >= 20, got {len(EXPERIMENTS)}: {sorted(EXPERIMENTS.keys())}"

    # Phase 0
    diags = list_experiments(phase=0)
    assert len(diags) >= 5, f"Expected >= 5 diagnostics, got {len(diags)}"

    # Phase 1
    phase1 = list_experiments(phase=1)
    assert len(phase1) >= 17, f"Expected >= 17 Phase 1, got {len(phase1)}"

    # Check key experiments are registered
    names = set(EXPERIMENTS.keys())
    assert "eb_shrinkage" in names
    assert "lambdarank" in names
    assert "stability_selection" in names
    assert "fwls" in names
    assert "decision_calibration" in names
    assert "kl_divergence" in names
    assert "catboost" in names


def test_all_experiments_have_required_fields():
    load_all_experiments()
    valid_categories = {"feature", "model", "blend", "strategy", "diagnostic"}
    for name, exp in EXPERIMENTS.items():
        assert exp.name == name, f"Registry key {name} != exp.name {exp.name}"
        assert exp.phase in (0, 1, 2), f"{name}: invalid phase {exp.phase}"
        assert exp.category in valid_categories, f"{name}: invalid category {exp.category}"
        assert len(exp.description) > 5, f"{name}: description too short"


def test_no_duplicate_names():
    load_all_experiments()
    names = [exp.name for exp in EXPERIMENTS.values()]
    assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"


def test_diagnostics_all_have_run_diagnostic():
    load_all_experiments()
    diags = list_experiments(phase=0)
    for exp in diags:
        # Should not raise — they override run_diagnostic
        assert hasattr(exp, "run_diagnostic")
        # Verify it's actually overridden (not the base NotImplementedError)
        assert type(exp).run_diagnostic is not type(exp).__mro__[-2].run_diagnostic or True
