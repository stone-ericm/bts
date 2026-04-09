"""Central experiment registry.

All experiments are registered here. Import experiment modules to trigger
registration via their module-level register() calls.
"""

from __future__ import annotations

from bts.experiment.base import ExperimentDef

EXPERIMENTS: dict[str, ExperimentDef] = {}


def register(experiment: ExperimentDef) -> None:
    """Register an experiment by name. Raises if name already taken."""
    if experiment.name in EXPERIMENTS:
        raise ValueError(f"Duplicate experiment name: {experiment.name!r}")
    EXPERIMENTS[experiment.name] = experiment


def get_experiment(name: str) -> ExperimentDef:
    """Look up experiment by name. Raises KeyError if not found."""
    return EXPERIMENTS[name]


def list_experiments(
    phase: int | None = None,
    category: str | None = None,
) -> list[ExperimentDef]:
    """List experiments, optionally filtered by phase and/or category."""
    result = list(EXPERIMENTS.values())
    if phase is not None:
        result = [e for e in result if e.phase == phase]
    if category is not None:
        result = [e for e in result if e.category == category]
    return sorted(result, key=lambda e: e.name)


def load_all_experiments() -> None:
    """Import all experiment modules to trigger registration."""
    import bts.experiment.diagnostics  # noqa: F401
    import bts.experiment.features  # noqa: F401
    import bts.experiment.models  # noqa: F401
    import bts.experiment.blends  # noqa: F401
    import bts.experiment.strategies as _strategies  # noqa: F401
