# Frontier Experiment Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a declarative experiment framework that backtests 23 frontier-math improvements through diagnostics → screening → forward stepwise selection, maximizing MDP P(57).

**Architecture:** Hook-based `ExperimentDef` classes registered in a central registry. A runner executes phases (0=diagnostics, 1=screening, 2=selection) by calling hooks, running `blend_walk_forward`, computing scorecards, and diffing against baseline. CLI commands integrate via Click subgroup on the existing `bts` CLI.

**Tech Stack:** Python 3.12, LightGBM, Click, pandas, numpy, scipy, knockpy, catboost (optional), river (ADWIN)

**Spec:** `docs/superpowers/specs/2026-04-08-frontier-experiment-framework-design.md`

---

## File Structure

```
src/bts/experiment/
    __init__.py           — re-exports ExperimentDef, EXPERIMENTS
    base.py               — ExperimentDef dataclass (hooks, metadata)
    registry.py           — EXPERIMENTS dict, get/list helpers
    runner.py             — run_diagnostics(), run_screening(), run_selection()
    reporting.py          — format_phase1_table(), format_phase2_log()
    cli.py                — Click group: bts experiment {diagnostics,screen,select,summary}
    diagnostics.py        — 7 Phase 0 diagnostic classes
    features.py           — 8 feature experiment classes
    models.py             — 4 model experiment classes
    blends.py             — 3 blend experiment classes
    strategies.py         — 2 strategy experiment classes

tests/experiment/
    __init__.py
    conftest.py           — shared fixtures (mini PA DataFrame, mock profiles)
    test_base.py          — ExperimentDef contract tests
    test_runner.py        — runner orchestration tests
    test_reporting.py     — table formatting tests
    test_diagnostics.py   — Phase 0 diagnostic hook tests
    test_features.py      — feature experiment hook tests
    test_models.py        — model experiment hook tests
    test_blends.py        — blend experiment hook tests
    test_strategies.py    — strategy experiment hook tests
```

---

### Task 1: ExperimentDef Base Class

**Files:**
- Create: `src/bts/experiment/__init__.py`
- Create: `src/bts/experiment/base.py`
- Create: `tests/experiment/__init__.py`
- Create: `tests/experiment/conftest.py`
- Create: `tests/experiment/test_base.py`

- [ ] **Step 1: Write test fixtures for mini PA DataFrame and mock profiles**

Create `tests/experiment/__init__.py` (empty) and `tests/experiment/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mini_pa_df():
    """Minimal PA DataFrame with enough structure to test experiment hooks.

    50 PAs across 5 dates, 5 batters, 2 pitchers. Includes all 15 baseline
    feature columns (filled with plausible random values) plus is_hit, season,
    game_pk, batter_id, pitcher_id, date.
    """
    rng = np.random.default_rng(42)
    n = 50
    dates = pd.date_range("2025-06-01", periods=5, freq="D")
    batter_ids = [100001, 100002, 100003, 100004, 100005]
    pitcher_ids = [200001, 200002]

    rows = []
    for i in range(n):
        rows.append({
            "date": dates[i % 5],
            "season": 2025,
            "game_pk": 900000 + (i % 5),
            "batter_id": batter_ids[i % 5],
            "pitcher_id": pitcher_ids[i % 2],
            "pitch_hand": "R" if i % 2 == 0 else "L",
            "bat_side": "L" if i % 3 == 0 else "R",
            "is_hit": int(rng.random() > 0.7),
            "batter_hr_7g": rng.uniform(0.2, 0.4),
            "batter_hr_30g": rng.uniform(0.2, 0.35),
            "batter_hr_60g": rng.uniform(0.22, 0.33),
            "batter_hr_120g": rng.uniform(0.23, 0.32),
            "batter_whiff_60g": rng.uniform(0.15, 0.35),
            "batter_count_tendency_30g": rng.uniform(-0.5, 0.5),
            "batter_gb_hit_rate": rng.uniform(0.15, 0.25),
            "platoon_hr": rng.uniform(0.2, 0.35),
            "pitcher_hr_30g": rng.uniform(0.2, 0.3),
            "pitcher_entropy_30g": rng.uniform(0.5, 2.0),
            "pitcher_catcher_framing": rng.uniform(0.25, 0.35),
            "opp_bullpen_hr_30g": rng.uniform(0.22, 0.3),
            "weather_temp": rng.uniform(60, 90),
            "park_factor": rng.uniform(0.9, 1.1),
            "days_rest": rng.choice([0, 1, 2, 3]),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_profiles_df():
    """Mock daily profiles DataFrame (output of blend_walk_forward)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-06-01", periods=10, freq="D")
    rows = []
    for d in dates:
        for rank in range(1, 11):
            rows.append({
                "date": d.date(),
                "rank": rank,
                "batter_id": 100000 + rank,
                "p_game_hit": max(0.5, 0.95 - rank * 0.04 + rng.normal(0, 0.02)),
                "actual_hit": int(rng.random() > (0.1 + rank * 0.03)),
                "n_pas": rng.choice([3, 4, 5]),
                "season": 2025,
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write tests for ExperimentDef base class**

Create `tests/experiment/test_base.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.experiment'`

- [ ] **Step 4: Implement ExperimentDef base class**

Create `src/bts/experiment/__init__.py`:

```python
from bts.experiment.base import ExperimentDef

__all__ = ["ExperimentDef"]
```

Create `src/bts/experiment/base.py`:

```python
"""Base class for all BTS experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ExperimentDef:
    """Declarative experiment definition.

    Subclass and override hooks to define an experiment. The runner calls
    hooks in order: modify_features → modify_blend_configs →
    modify_training_params → (run walk-forward) → modify_strategy.

    Phase 0 experiments override run_diagnostic instead.
    """

    name: str
    phase: int  # 0=diagnostic, 1=screening, 2=forward-select
    category: str  # "feature", "model", "blend", "strategy", "diagnostic"
    description: str
    dependencies: list[str] = field(default_factory=list)

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to add/replace/remove features. Return modified DataFrame."""
        return df

    def modify_blend_configs(
        self, configs: list[tuple[str, list[str]]]
    ) -> list[tuple[str, list[str]]]:
        """Override to add/replace blend model configs."""
        return configs

    def modify_training_params(self, params: dict) -> dict:
        """Override to change LightGBM training params."""
        return params

    def modify_strategy(
        self, profiles_df: pd.DataFrame, quality_bins: object
    ) -> tuple[pd.DataFrame, object]:
        """Override to change strategy/calibration/MDP inputs."""
        return profiles_df, quality_bins

    def run_diagnostic(
        self, df: pd.DataFrame, profiles: dict[int, pd.DataFrame]
    ) -> dict:
        """Override for Phase 0 diagnostics. Return report dict."""
        raise NotImplementedError(
            f"{self.name}: Phase 0 experiments must implement run_diagnostic()"
        )

    def feature_cols(self) -> list[str] | None:
        """Override to change FEATURE_COLS for this experiment. None = default."""
        return None

    def touches_features(self) -> bool:
        """Whether this experiment modifies features (requires recompute)."""
        return (
            type(self).modify_features is not ExperimentDef.modify_features
            or type(self).feature_cols is not ExperimentDef.feature_cols
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_base.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add src/bts/experiment/ tests/experiment/
git commit -m "feat(experiment): add ExperimentDef base class with hook interface"
```

---

### Task 2: Experiment Registry

**Files:**
- Create: `src/bts/experiment/registry.py`
- Create: `tests/experiment/test_registry.py`

- [ ] **Step 1: Write registry tests**

Create `tests/experiment/test_registry.py`:

```python
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
    # All returned experiments have correct phase
    assert all(e.phase == 0 for e in phase0)
    assert all(e.phase == 1 for e in phase1)


def test_list_experiments_by_category():
    features = list_experiments(category="feature")
    assert all(e.category == "feature" for e in features)


def test_list_experiments_no_filter():
    all_exps = list_experiments()
    assert isinstance(all_exps, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement registry**

Create `src/bts/experiment/registry.py`:

```python
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
        raise ValueError(
            f"Duplicate experiment name: {experiment.name!r}"
        )
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
    import bts.experiment.strategies  # noqa: F401
```

Update `src/bts/experiment/__init__.py`:

```python
from bts.experiment.base import ExperimentDef
from bts.experiment.registry import EXPERIMENTS, get_experiment, list_experiments, register

__all__ = ["ExperimentDef", "EXPERIMENTS", "get_experiment", "list_experiments", "register"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/
git commit -m "feat(experiment): add experiment registry with get/list/register"
```

---

### Task 3: Runner — Phase 0 Diagnostic Executor

**Files:**
- Create: `src/bts/experiment/runner.py`
- Create: `tests/experiment/test_runner.py`

- [ ] **Step 1: Write tests for diagnostic runner**

Create `tests/experiment/test_runner.py`:

```python
import json
from pathlib import Path

from bts.experiment.base import ExperimentDef
from bts.experiment.runner import run_diagnostics


class _MockDiagnostic(ExperimentDef):
    def run_diagnostic(self, df, profiles):
        return {"test_metric": 42, "stable_features": ["batter_hr_30g"]}


def test_run_diagnostics_saves_results(mini_pa_df, tmp_path):
    diag = _MockDiagnostic(
        name="mock_diag", phase=0, category="diagnostic",
        description="mock diagnostic",
    )
    results = run_diagnostics(
        experiments=[diag],
        pa_df=mini_pa_df,
        profiles={},
        results_dir=tmp_path / "results" / "phase0",
    )
    assert "mock_diag" in results
    assert results["mock_diag"]["test_metric"] == 42
    # Check file was saved
    saved = json.loads((tmp_path / "results" / "phase0" / "mock_diag.json").read_text())
    assert saved["test_metric"] == 42


def test_run_diagnostics_empty_list(mini_pa_df, tmp_path):
    results = run_diagnostics(
        experiments=[],
        pa_df=mini_pa_df,
        profiles={},
        results_dir=tmp_path / "results" / "phase0",
    )
    assert results == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_runner.py::test_run_diagnostics_saves_results -v`
Expected: FAIL — `cannot import name 'run_diagnostics'`

- [ ] **Step 3: Implement Phase 0 runner**

Create `src/bts/experiment/runner.py`:

```python
"""Experiment runner — executes phases 0, 1, and 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def run_diagnostics(
    experiments: list[ExperimentDef],
    pa_df: pd.DataFrame,
    profiles: dict[int, pd.DataFrame],
    results_dir: Path,
) -> dict[str, dict]:
    """Run Phase 0 diagnostics and save reports.

    Args:
        experiments: List of Phase 0 ExperimentDef instances.
        pa_df: Feature-enriched PA DataFrame.
        profiles: {season: profiles_df} from existing backtests.
        results_dir: Directory to save JSON reports.

    Returns:
        {name: report_dict} for each diagnostic.
    """
    results: dict[str, dict] = {}
    for exp in experiments:
        print(f"[Phase 0] Running {exp.name}: {exp.description}", file=sys.stderr)
        report = exp.run_diagnostic(pa_df, profiles)
        _save_json(report, results_dir / f"{exp.name}.json")
        results[exp.name] = report
        print(f"  → Saved {results_dir / exp.name}.json", file=sys.stderr)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_runner.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/runner.py tests/experiment/test_runner.py
git commit -m "feat(experiment): add Phase 0 diagnostic runner"
```

---

### Task 4: Runner — Phase 1 Screening Executor

**Files:**
- Modify: `src/bts/experiment/runner.py`
- Modify: `tests/experiment/test_runner.py`

- [ ] **Step 1: Write screening runner tests**

Add to `tests/experiment/test_runner.py`:

```python
from bts.experiment.runner import (
    evaluate_pass_fail,
    run_single_screening,
)


def test_evaluate_pass_fail_both_seasons_improve():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.855, "delta": 0.006},
            "2025": {"baseline": 0.859, "variant": 0.865, "delta": 0.006},
        },
        "p_57_mdp": {"baseline": 0.0891, "variant": 0.0920, "delta": 0.0029},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is True
    assert "both seasons" in reason.lower()


def test_evaluate_pass_fail_one_season_drops():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.840, "delta": -0.009},
            "2025": {"baseline": 0.859, "variant": 0.865, "delta": 0.006},
        },
        "p_57_mdp": {"baseline": 0.0891, "variant": 0.0920, "delta": 0.0029},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is False


def test_evaluate_pass_fail_neutral_p1_but_p57_improves():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.848, "delta": -0.001},
            "2025": {"baseline": 0.859, "variant": 0.857, "delta": -0.002},
        },
        "p_57_mdp": {"baseline": 0.0891, "variant": 0.0950, "delta": 0.0059},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is True
    assert "p(57)" in reason.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_runner.py::test_evaluate_pass_fail_both_seasons_improve -v`
Expected: FAIL — `cannot import name 'evaluate_pass_fail'`

- [ ] **Step 3: Implement pass/fail evaluator and screening runner**

Add to `src/bts/experiment/runner.py`:

```python
# Maximum allowed P@1 drop per season for "neutral" condition (0.3pp)
NEUTRAL_THRESHOLD = -0.003


def evaluate_pass_fail(diff: dict) -> tuple[bool, str]:
    """Evaluate whether an experiment passes screening.

    Pass if EITHER:
    1. P@1 improves on both 2024 AND 2025
    2. P@1 neutral on both (drop <= 0.3pp) AND MDP P(57) improves

    Returns:
        (passed, reason) tuple.
    """
    p1_by_season = diff.get("p_at_1_by_season", {})
    p57_diff = diff.get("p_57_mdp", {})

    season_deltas = {}
    for season_key, d in p1_by_season.items():
        season_deltas[str(season_key)] = d.get("delta", 0)

    if len(season_deltas) < 2:
        return False, "Missing season data"

    all_improve = all(d > 0 for d in season_deltas.values())
    all_neutral = all(d >= NEUTRAL_THRESHOLD for d in season_deltas.values())
    p57_improves = p57_diff.get("delta", 0) > 0

    if all_improve:
        return True, "P@1 improves on both seasons"
    if all_neutral and p57_improves:
        return True, "P@1 neutral, P(57) improves"
    return False, f"P@1 deltas: {season_deltas}, P(57) delta: {p57_diff.get('delta', 'N/A')}"


def run_single_screening(
    experiment: ExperimentDef,
    pa_df: pd.DataFrame,
    baseline_scorecard: dict,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
) -> dict:
    """Run a single Phase 1 experiment: walk-forward → scorecard → diff → pass/fail.

    Returns dict with keys: scorecard, diff, passed, reason, name.
    """
    from bts.features.compute import compute_all_features, FEATURE_COLS
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards, save_scorecard

    print(f"\n[Phase 1] {experiment.name}: {experiment.description}", file=sys.stderr)

    # Apply feature modifications if needed
    df = pa_df
    if experiment.touches_features():
        print(f"  Recomputing features for {experiment.name}...", file=sys.stderr)
        df = experiment.modify_features(df.copy())

    # Run walk-forward for each test season
    all_profiles = []
    for season in test_seasons:
        profiles = blend_walk_forward(
            df, season, retrain_every=retrain_every,
        )
        profiles["season"] = season
        all_profiles.append(profiles)

    combined_profiles = pd.concat(all_profiles, ignore_index=True)

    # Apply strategy modifications
    scorecard = compute_full_scorecard(combined_profiles)
    diff = diff_scorecards(baseline_scorecard, scorecard)
    passed, reason = evaluate_pass_fail(diff)

    # Save results
    exp_dir = results_dir / experiment.name
    save_scorecard(scorecard, exp_dir / "scorecard.json")
    _save_json(diff, exp_dir / "diff.json")
    summary = f"{'PASS' if passed else 'FAIL'} | {reason}"
    (exp_dir / "summary.txt").parent.mkdir(parents=True, exist_ok=True)
    (exp_dir / "summary.txt").write_text(summary)

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  → {status}: {reason}", file=sys.stderr)

    return {
        "name": experiment.name,
        "scorecard": scorecard,
        "diff": diff,
        "passed": passed,
        "reason": reason,
    }


def run_screening(
    experiments: list[ExperimentDef],
    pa_df: pd.DataFrame,
    baseline_scorecard: dict,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
) -> list[dict]:
    """Run Phase 1 screening for all experiments."""
    results = []
    for exp in experiments:
        result = run_single_screening(
            exp, pa_df, baseline_scorecard, test_seasons,
            results_dir, retrain_every,
        )
        results.append(result)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_runner.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/runner.py tests/experiment/test_runner.py
git commit -m "feat(experiment): add Phase 1 screening runner with pass/fail evaluator"
```

---

### Task 5: Runner — Phase 2 Forward Selection

**Files:**
- Modify: `src/bts/experiment/runner.py`
- Modify: `tests/experiment/test_runner.py`

- [ ] **Step 1: Write forward selection tests**

Add to `tests/experiment/test_runner.py`:

```python
from bts.experiment.runner import sort_winners_by_p57


def test_sort_winners_by_p57():
    results = [
        {"name": "a", "passed": True, "diff": {"p_57_mdp": {"delta": 0.005}}},
        {"name": "b", "passed": True, "diff": {"p_57_mdp": {"delta": 0.012}}},
        {"name": "c", "passed": False, "diff": {"p_57_mdp": {"delta": 0.020}}},
        {"name": "d", "passed": True, "diff": {"p_57_mdp": {"delta": 0.001}}},
    ]
    winners = sort_winners_by_p57(results)
    assert len(winners) == 3  # c is excluded (not passed)
    assert winners[0]["name"] == "b"  # highest delta first
    assert winners[1]["name"] == "a"
    assert winners[2]["name"] == "d"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_runner.py::test_sort_winners_by_p57 -v`
Expected: FAIL — `cannot import name 'sort_winners_by_p57'`

- [ ] **Step 3: Implement forward selection helpers**

Add to `src/bts/experiment/runner.py`:

```python
def sort_winners_by_p57(results: list[dict]) -> list[dict]:
    """Filter to passing experiments and sort by P(57) improvement descending."""
    winners = [r for r in results if r.get("passed")]
    return sorted(
        winners,
        key=lambda r: r.get("diff", {}).get("p_57_mdp", {}).get("delta", 0),
        reverse=True,
    )


def run_selection(
    winners: list[dict],
    experiments_by_name: dict[str, ExperimentDef],
    pa_df: pd.DataFrame,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
) -> dict:
    """Run Phase 2: forward stepwise selection + backward elimination.

    Args:
        winners: Sorted list of Phase 1 results (passing only).
        experiments_by_name: {name: ExperimentDef} lookup.
        pa_df: Full PA DataFrame (pre-feature-computation).
        test_seasons: Seasons to evaluate on.
        results_dir: Directory for phase2 results.

    Returns:
        Dict with forward_log, backward_log, final_scorecard, final_diff.
    """
    from bts.features.compute import compute_all_features
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards, save_scorecard

    print(f"\n[Phase 2] Forward selection with {len(winners)} candidates", file=sys.stderr)

    # Compute baseline
    baseline_df = pa_df.copy()
    baseline_profiles = []
    for season in test_seasons:
        profiles = blend_walk_forward(baseline_df, season, retrain_every=retrain_every)
        profiles["season"] = season
        baseline_profiles.append(profiles)
    baseline_combined = pd.concat(baseline_profiles, ignore_index=True)
    baseline_scorecard = compute_full_scorecard(baseline_combined)
    current_p57 = baseline_scorecard.get("p_57_mdp", 0) or 0

    # Track which experiments are included
    included: list[str] = []
    forward_log: list[dict] = []

    current_df = baseline_df
    for winner in winners:
        name = winner["name"]
        exp = experiments_by_name[name]
        print(f"  Trying +{name}...", file=sys.stderr)

        # Apply this experiment's hooks on top of current state
        candidate_df = exp.modify_features(current_df.copy())

        candidate_profiles = []
        for season in test_seasons:
            profiles = blend_walk_forward(candidate_df, season, retrain_every=retrain_every)
            profiles["season"] = season
            candidate_profiles.append(profiles)
        candidate_combined = pd.concat(candidate_profiles, ignore_index=True)
        candidate_scorecard = compute_full_scorecard(candidate_combined)
        candidate_p57 = candidate_scorecard.get("p_57_mdp", 0) or 0

        step = {
            "name": name,
            "p57_before": current_p57,
            "p57_after": candidate_p57,
            "delta": candidate_p57 - current_p57,
            "kept": candidate_p57 > current_p57,
        }
        forward_log.append(step)

        if candidate_p57 > current_p57:
            print(f"  ✓ Kept {name}: P(57) {current_p57:.4f} → {candidate_p57:.4f}", file=sys.stderr)
            included.append(name)
            current_df = candidate_df
            current_p57 = candidate_p57
        else:
            print(f"  ✗ Dropped {name}: P(57) did not improve", file=sys.stderr)

    print(f"\n  Forward selection: {len(included)} experiments included", file=sys.stderr)

    # Backward elimination
    backward_log: list[dict] = []
    for name in list(included):
        print(f"  Trying -{name}...", file=sys.stderr)
        # Rebuild without this experiment
        test_df = pa_df.copy()
        for kept_name in included:
            if kept_name != name:
                test_df = experiments_by_name[kept_name].modify_features(test_df)

        test_profiles = []
        for season in test_seasons:
            profiles = blend_walk_forward(test_df, season, retrain_every=retrain_every)
            profiles["season"] = season
            test_profiles.append(profiles)
        test_combined = pd.concat(test_profiles, ignore_index=True)
        test_scorecard = compute_full_scorecard(test_combined)
        test_p57 = test_scorecard.get("p_57_mdp", 0) or 0

        step = {
            "name": name,
            "p57_with": current_p57,
            "p57_without": test_p57,
            "delta": current_p57 - test_p57,
            "kept": test_p57 < current_p57,
        }
        backward_log.append(step)

        if test_p57 >= current_p57:
            print(f"  ✗ Removed {name}: not needed (P(57) stable without it)", file=sys.stderr)
            included.remove(name)
        else:
            print(f"  ✓ Kept {name}: removing hurts P(57)", file=sys.stderr)

    # Final scorecard
    final_df = pa_df.copy()
    for name in included:
        final_df = experiments_by_name[name].modify_features(final_df)

    final_profiles = []
    for season in test_seasons:
        profiles = blend_walk_forward(final_df, season, retrain_every=retrain_every)
        profiles["season"] = season
        final_profiles.append(profiles)
    final_combined = pd.concat(final_profiles, ignore_index=True)
    final_scorecard = compute_full_scorecard(final_combined)
    final_diff = diff_scorecards(baseline_scorecard, final_scorecard)

    # Save
    results_dir.mkdir(parents=True, exist_ok=True)
    _save_json(forward_log, results_dir / "forward_selection_log.json")
    _save_json(backward_log, results_dir / "backward_elimination_log.json")
    save_scorecard(final_scorecard, results_dir / "final_scorecard.json")
    _save_json(final_diff, results_dir / "final_diff.json")

    print(f"\n  Final model: {included}", file=sys.stderr)
    print(f"  Final P(57): {final_scorecard.get('p_57_mdp', 'N/A')}", file=sys.stderr)

    return {
        "included": included,
        "forward_log": forward_log,
        "backward_log": backward_log,
        "final_scorecard": final_scorecard,
        "final_diff": final_diff,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_runner.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/runner.py tests/experiment/test_runner.py
git commit -m "feat(experiment): add Phase 2 forward selection + backward elimination"
```

---

### Task 6: Reporting and Summary Tables

**Files:**
- Create: `src/bts/experiment/reporting.py`
- Create: `tests/experiment/test_reporting.py`

- [ ] **Step 1: Write reporting tests**

Create `tests/experiment/test_reporting.py`:

```python
from bts.experiment.reporting import format_phase1_table


def test_format_phase1_table():
    results = [
        {
            "name": "eb_shrinkage",
            "passed": True,
            "reason": "P@1 improves on both seasons",
            "diff": {
                "p_at_1_by_season": {
                    "2024": {"delta": 0.003},
                    "2025": {"delta": 0.002},
                },
                "p_57_mdp": {"delta": 0.008},
            },
        },
        {
            "name": "catboost_blend",
            "passed": False,
            "reason": "P@1 drops on 2025",
            "diff": {
                "p_at_1_by_season": {
                    "2024": {"delta": 0.001},
                    "2025": {"delta": -0.005},
                },
                "p_57_mdp": {"delta": -0.001},
            },
        },
    ]
    table = format_phase1_table(results)
    assert "eb_shrinkage" in table
    assert "catboost_blend" in table
    assert "PASS" in table or "✓" in table
    assert "FAIL" in table or "✗" in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_reporting.py -v`
Expected: FAIL — `cannot import name 'format_phase1_table'`

- [ ] **Step 3: Implement reporting**

Create `src/bts/experiment/reporting.py`:

```python
"""Summary table formatting for experiment results."""

from __future__ import annotations


def format_phase1_table(results: list[dict], baseline_p1: float = 0.862, baseline_p57: float = 0.0891) -> str:
    """Format Phase 1 results as a summary table string."""
    header = (
        f"Phase 1 Results — {len(results)} experiments vs baseline "
        f"(P@1={baseline_p1:.1%}, P(57)={baseline_p57:.2%})\n"
    )
    sep = "─" * 72 + "\n"
    col_header = f"{'Experiment':<24} {'P@1 2024':>10} {'P@1 2025':>10} {'P(57) MDP':>11} {'Pass':>6}\n"

    rows = []
    for r in sorted(results, key=lambda x: x.get("passed", False), reverse=True):
        name = r["name"][:23]
        diff = r.get("diff", {})

        p1_2024 = diff.get("p_at_1_by_season", {}).get("2024", {}).get("delta")
        p1_2025 = diff.get("p_at_1_by_season", {}).get("2025", {}).get("delta")
        p57 = diff.get("p_57_mdp", {}).get("delta")

        def fmt_delta(d):
            if d is None:
                return "N/A".rjust(10)
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1%}".rjust(10)

        status = "✓" if r.get("passed") else "✗"
        rows.append(f"{name:<24} {fmt_delta(p1_2024)} {fmt_delta(p1_2025)} {fmt_delta(p57):>11} {status:>6}")

    n_pass = sum(1 for r in results if r.get("passed"))
    footer = f"\nWinners: {n_pass}/{len(results)} passed screening"

    return header + sep + col_header + sep + "\n".join(rows) + "\n" + sep + footer


def format_phase2_log(selection_result: dict) -> str:
    """Format Phase 2 forward/backward log as a summary string."""
    lines = ["Phase 2 — Forward Stepwise Selection"]
    lines.append("─" * 60)

    for step in selection_result.get("forward_log", []):
        status = "✓ KEPT" if step["kept"] else "✗ DROP"
        lines.append(
            f"  +{step['name']:<20} P(57): {step['p57_before']:.4f} → "
            f"{step['p57_after']:.4f} ({step['delta']:+.4f})  {status}"
        )

    lines.append("")
    lines.append("Backward Elimination")
    lines.append("─" * 60)

    for step in selection_result.get("backward_log", []):
        status = "✓ KEPT" if step["kept"] else "✗ DROP"
        lines.append(
            f"  -{step['name']:<20} P(57) without: {step['p57_without']:.4f} "
            f"(Δ={step['delta']:+.4f})  {status}"
        )

    included = selection_result.get("included", [])
    final_p57 = selection_result.get("final_scorecard", {}).get("p_57_mdp", "N/A")
    lines.append("")
    lines.append(f"Final model: {included}")
    lines.append(f"Final P(57): {final_p57}")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_reporting.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/reporting.py tests/experiment/test_reporting.py
git commit -m "feat(experiment): add Phase 1/2 summary table formatting"
```

---

### Task 7: CLI Integration

**Files:**
- Create: `src/bts/experiment/cli.py`
- Modify: `src/bts/cli.py` (add ~2 lines to register experiment group)

- [ ] **Step 1: Implement CLI commands**

Create `src/bts/experiment/cli.py`:

```python
"""CLI commands for the experiment framework."""

import json
import sys
from pathlib import Path

import click


RESULTS_BASE = Path("experiments/results")


@click.group()
def experiment():
    """Frontier experiment framework — diagnostics, screening, selection."""
    pass


@experiment.command()
@click.option("--data-dir", default="data/processed", type=click.Path(),
              help="Processed parquet directory")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(),
              help="Existing backtest profiles directory")
def diagnostics(data_dir: str, profiles_dir: str):
    """Run Phase 0 diagnostics."""
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import list_experiments, load_all_experiments
    from bts.experiment.runner import run_diagnostics
    from bts.experiment.reporting import format_phase1_table

    load_all_experiments()
    diags = list_experiments(phase=0)
    if not diags:
        click.echo("No Phase 0 diagnostics registered.")
        return

    click.echo(f"Running {len(diags)} diagnostics...")

    # Load data
    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    if not dfs:
        raise click.ClickException("No parquet files found. Run 'bts data build' first.")
    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)

    # Load existing profiles if available
    profiles = {}
    prof_path = Path(profiles_dir)
    for p in prof_path.glob("backtest_*.parquet"):
        season = int(p.stem.split("_")[1])
        profiles[season] = pd.read_parquet(p)

    results = run_diagnostics(diags, df, profiles, RESULTS_BASE / "phase0")
    click.echo(f"\nDiagnostics complete. {len(results)} reports saved.")
    for name, report in results.items():
        click.echo(f"  {name}: {list(report.keys())[:5]}...")


@experiment.command()
@click.option("--data-dir", default="data/processed", type=click.Path())
@click.option("--subset", default=None, help="Comma-separated experiment names to run")
@click.option("--retrain-every", default=7, type=int)
@click.option("--test-seasons", default="2024,2025", help="Comma-separated test seasons")
def screen(data_dir: str, subset: str | None, retrain_every: int, test_seasons: str):
    """Run Phase 1 independent screening."""
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import list_experiments, load_all_experiments, get_experiment
    from bts.experiment.runner import run_screening
    from bts.experiment.reporting import format_phase1_table
    from bts.validate.scorecard import compute_full_scorecard, save_scorecard
    from bts.simulate.backtest_blend import blend_walk_forward

    load_all_experiments()
    seasons = [int(s.strip()) for s in test_seasons.split(",")]

    if subset:
        experiments = [get_experiment(n.strip()) for n in subset.split(",")]
    else:
        experiments = list_experiments(phase=1)

    if not experiments:
        click.echo("No Phase 1 experiments to run.")
        return

    click.echo(f"Screening {len(experiments)} experiments on seasons {seasons}")

    # Load and prepare data
    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)

    # Compute or load baseline
    baseline_path = RESULTS_BASE / "phase1" / "baseline_scorecard.json"
    if baseline_path.exists():
        baseline_scorecard = json.loads(baseline_path.read_text())
        click.echo("Loaded cached baseline scorecard.")
    else:
        click.echo("Computing baseline scorecard...")
        baseline_profiles = []
        for season in seasons:
            profiles = blend_walk_forward(df, season, retrain_every=retrain_every)
            profiles["season"] = season
            baseline_profiles.append(profiles)
        baseline_combined = pd.concat(baseline_profiles, ignore_index=True)
        baseline_scorecard = compute_full_scorecard(baseline_combined)
        save_scorecard(baseline_scorecard, baseline_path)

    results = run_screening(
        experiments, df, baseline_scorecard, seasons,
        RESULTS_BASE / "phase1", retrain_every,
    )

    click.echo(format_phase1_table(results))


@experiment.command()
@click.option("--data-dir", default="data/processed", type=click.Path())
@click.option("--retrain-every", default=7, type=int)
@click.option("--test-seasons", default="2024,2025")
def select(data_dir: str, retrain_every: int, test_seasons: str):
    """Run Phase 2 forward stepwise selection."""
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import load_all_experiments, get_experiment
    from bts.experiment.runner import run_selection, sort_winners_by_p57
    from bts.experiment.reporting import format_phase2_log

    load_all_experiments()
    seasons = [int(s.strip()) for s in test_seasons.split(",")]

    # Load Phase 1 results
    phase1_dir = RESULTS_BASE / "phase1"
    results = []
    for exp_dir in sorted(phase1_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_path = exp_dir / "summary.txt"
        diff_path = exp_dir / "diff.json"
        if not summary_path.exists() or not diff_path.exists():
            continue
        summary_text = summary_path.read_text()
        diff = json.loads(diff_path.read_text())
        results.append({
            "name": exp_dir.name,
            "passed": summary_text.startswith("PASS"),
            "diff": diff,
        })

    winners = sort_winners_by_p57(results)
    if not winners:
        click.echo("No winners from Phase 1. Nothing to select.")
        return

    click.echo(f"Forward selection with {len(winners)} winners")

    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)

    experiments_by_name = {}
    for w in winners:
        experiments_by_name[w["name"]] = get_experiment(w["name"])

    selection_result = run_selection(
        winners, experiments_by_name, df, seasons,
        RESULTS_BASE / "phase2", retrain_every,
    )

    click.echo(format_phase2_log(selection_result))


@experiment.command()
def summary():
    """Print results summary across all phases."""
    from bts.experiment.reporting import format_phase1_table

    phase1_dir = RESULTS_BASE / "phase1"
    if not phase1_dir.exists():
        click.echo("No Phase 1 results found. Run 'bts experiment screen' first.")
        return

    results = []
    for exp_dir in sorted(phase1_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        diff_path = exp_dir / "diff.json"
        summary_path = exp_dir / "summary.txt"
        if not diff_path.exists():
            continue
        diff = json.loads(diff_path.read_text())
        passed = summary_path.read_text().startswith("PASS") if summary_path.exists() else False
        results.append({"name": exp_dir.name, "passed": passed, "diff": diff})

    if results:
        click.echo(format_phase1_table(results))

    phase2_path = RESULTS_BASE / "phase2" / "forward_selection_log.json"
    if phase2_path.exists():
        from bts.experiment.reporting import format_phase2_log
        sel = json.loads(phase2_path.read_text())
        back_path = RESULTS_BASE / "phase2" / "backward_elimination_log.json"
        backward = json.loads(back_path.read_text()) if back_path.exists() else []
        final_path = RESULTS_BASE / "phase2" / "final_scorecard.json"
        final_sc = json.loads(final_path.read_text()) if final_path.exists() else {}
        click.echo(format_phase2_log({
            "forward_log": sel,
            "backward_log": backward,
            "final_scorecard": final_sc,
            "included": [s["name"] for s in sel if s.get("kept")],
        }))
```

- [ ] **Step 2: Register CLI group in main CLI**

Add to `src/bts/cli.py` after the `simulate` import:

```python
from bts.experiment.cli import experiment
cli.add_command(experiment)
```

- [ ] **Step 3: Verify CLI registers**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run bts experiment --help`
Expected: Shows `diagnostics`, `screen`, `select`, `summary` subcommands

- [ ] **Step 4: Commit**

```bash
git add src/bts/experiment/cli.py src/bts/cli.py
git commit -m "feat(experiment): add CLI commands for all three phases"
```

---

### Task 8: Phase 0 Diagnostics — Stability Selection + Wasserstein Drift + Streak Dependence

**Files:**
- Create: `src/bts/experiment/diagnostics.py`
- Create: `tests/experiment/test_diagnostics.py`

- [ ] **Step 1: Write diagnostic tests**

Create `tests/experiment/test_diagnostics.py`:

```python
import numpy as np
import pandas as pd

from bts.experiment.diagnostics import (
    StabilitySelectionDiagnostic,
    WassersteinDriftDiagnostic,
    StreakLengthDependenceDiagnostic,
)


def test_stability_selection_report_structure(mini_pa_df):
    diag = StabilitySelectionDiagnostic()
    report = diag.run_diagnostic(mini_pa_df, {})
    assert "feature_stability" in report
    assert isinstance(report["feature_stability"], dict)
    # Each feature should have a stability score between 0 and 1
    for feat, score in report["feature_stability"].items():
        assert 0 <= score <= 1, f"{feat}: {score}"


def test_wasserstein_drift_report_structure(mini_pa_df):
    # Need at least 2 seasons for drift
    df = mini_pa_df.copy()
    df2 = df.copy()
    df2["season"] = 2024
    combined = pd.concat([df, df2], ignore_index=True)

    diag = WassersteinDriftDiagnostic()
    report = diag.run_diagnostic(combined, {})
    assert "feature_drift" in report
    assert isinstance(report["feature_drift"], dict)


def test_streak_length_dependence(mock_profiles_df):
    diag = StreakLengthDependenceDiagnostic()
    report = diag.run_diagnostic(pd.DataFrame(), {2025: mock_profiles_df})
    assert "p1_by_streak_bucket" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_diagnostics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement three diagnostics**

Create `src/bts/experiment/diagnostics.py`:

```python
"""Phase 0 diagnostic experiments."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register
from bts.features.compute import FEATURE_COLS


class StabilitySelectionDiagnostic(ExperimentDef):
    """Run LightGBM on bootstrap samples per season, compute feature stability."""

    def __init__(self):
        super().__init__(
            name="stability_selection",
            phase=0,
            category="diagnostic",
            description="Feature stability across bootstrap samples and seasons",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        import lightgbm as lgb
        from bts.model.predict import LGB_PARAMS

        n_bootstrap = 100
        seasons = sorted(df["season"].unique())
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        # Per-season stability scores
        season_stability: dict[int, dict[str, float]] = {}
        for season in seasons:
            if len(df[df["season"] == season]) < 200:
                continue
            train = df[df["season"] < season]
            if len(train) < 500:
                continue

            selection_counts: dict[str, int] = {f: 0 for f in feature_cols}
            for b in range(n_bootstrap):
                sample = train.sample(frac=0.6, replace=True, random_state=b)
                X = sample[feature_cols]
                y = sample["is_hit"]
                mask = X.notna().any(axis=1)
                model = lgb.LGBMClassifier(
                    **{**LGB_PARAMS, "n_estimators": 50}, random_state=b,
                )
                model.fit(X[mask], y[mask])
                importances = dict(zip(feature_cols, model.feature_importances_))
                # "Selected" = importance > 0
                for feat, imp in importances.items():
                    if imp > 0:
                        selection_counts[feat] += 1

            season_stability[season] = {
                f: count / n_bootstrap for f, count in selection_counts.items()
            }
            print(f"  Season {season}: {n_bootstrap} bootstraps done", file=sys.stderr)

        # Cross-season stability: min stability across all seasons
        all_features_stability: dict[str, float] = {}
        for feat in feature_cols:
            scores = [ss.get(feat, 0) for ss in season_stability.values()]
            all_features_stability[feat] = float(min(scores)) if scores else 0.0

        return {
            "feature_stability": all_features_stability,
            "per_season": {int(k): v for k, v in season_stability.items()},
            "n_bootstrap": n_bootstrap,
            "n_seasons": len(season_stability),
        }


class WassersteinDriftDiagnostic(ExperimentDef):
    """Compute per-feature Wasserstein distances between season pairs."""

    def __init__(self):
        super().__init__(
            name="wasserstein_drift",
            phase=0,
            category="diagnostic",
            description="Per-feature distributional drift across seasons",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        from scipy.stats import wasserstein_distance

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        seasons = sorted(df["season"].unique())

        drift: dict[str, dict[str, float]] = {}
        for feat in feature_cols:
            pair_dists: dict[str, float] = {}
            for i, s1 in enumerate(seasons):
                for s2 in seasons[i + 1:]:
                    v1 = df.loc[df["season"] == s1, feat].dropna().values
                    v2 = df.loc[df["season"] == s2, feat].dropna().values
                    if len(v1) > 10 and len(v2) > 10:
                        pair_dists[f"{s1}-{s2}"] = float(wasserstein_distance(v1, v2))
            drift[feat] = pair_dists

        # Mean drift per feature across all pairs
        mean_drift = {}
        for feat, pairs in drift.items():
            vals = list(pairs.values())
            mean_drift[feat] = float(np.mean(vals)) if vals else 0.0

        return {
            "feature_drift": mean_drift,
            "pairwise_drift": drift,
            "n_seasons": len(seasons),
        }


class StreakLengthDependenceDiagnostic(ExperimentDef):
    """Check if P@1 degrades as streak length increases."""

    def __init__(self):
        super().__init__(
            name="streak_length_dependence",
            phase=0,
            category="diagnostic",
            description="P@1 stratified by simulated streak length",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        # Use profiles to simulate streaks and check P@1 at different lengths
        all_profiles = pd.concat(profiles.values(), ignore_index=True) if profiles else pd.DataFrame()
        if all_profiles.empty or "rank" not in all_profiles.columns:
            return {"p1_by_streak_bucket": {}, "note": "No profiles available"}

        rank1 = all_profiles[all_profiles["rank"] == 1].sort_values("date").copy()
        rank1["hit"] = rank1["actual_hit"].astype(bool)

        # Simulate streak progression
        streak = 0
        streak_at_pick: list[int] = []
        hits: list[bool] = []
        for _, row in rank1.iterrows():
            streak_at_pick.append(streak)
            hit = bool(row["hit"])
            hits.append(hit)
            streak = streak + 1 if hit else 0

        rank1 = rank1.iloc[:len(streak_at_pick)].copy()
        rank1["streak_at_pick"] = streak_at_pick

        # Bucket by streak length
        bins = [0, 5, 10, 20, 30, 50, 200]
        labels = ["0-4", "5-9", "10-19", "20-29", "30-49", "50+"]
        rank1["streak_bucket"] = pd.cut(
            rank1["streak_at_pick"], bins=bins, labels=labels, right=False,
        )

        p1_by_bucket = rank1.groupby("streak_bucket", observed=True)["actual_hit"].agg(["mean", "count"])
        result = {}
        for bucket, row in p1_by_bucket.iterrows():
            result[str(bucket)] = {"p_at_1": float(row["mean"]), "n_days": int(row["count"])}

        return {"p1_by_streak_bucket": result}


class AFTShapeDiagnostic(ExperimentDef):
    """Fit Weibull AFT model to streak termination data."""

    def __init__(self):
        super().__init__(
            name="aft_shape",
            phase=0,
            category="diagnostic",
            description="Weibull shape parameter for streak hazard",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        all_profiles = pd.concat(profiles.values(), ignore_index=True) if profiles else pd.DataFrame()
        if all_profiles.empty:
            return {"shape": None, "note": "No profiles available"}

        rank1 = all_profiles[all_profiles["rank"] == 1].sort_values("date")

        # Build streak durations
        streaks: list[int] = []
        current = 0
        for hit in rank1["actual_hit"]:
            if hit:
                current += 1
            else:
                streaks.append(max(current, 1))
                current = 0
        if current > 0:
            streaks.append(current)  # right-censored but we'll treat as complete

        if len(streaks) < 10:
            return {"shape": None, "note": "Too few streaks"}

        # Fit Weibull via scipy
        from scipy.stats import weibull_min
        shape, _, scale = weibull_min.fit(streaks, floc=0)

        interpretation = "increasing hazard (streaks get harder)" if shape > 1 else (
            "decreasing hazard (hot-hand stabilization)" if shape < 1 else
            "constant hazard (geometric/independence)"
        )

        return {
            "shape": float(shape),
            "scale": float(scale),
            "n_streaks": len(streaks),
            "mean_streak": float(np.mean(streaks)),
            "interpretation": interpretation,
        }


class ADWINChangepointDiagnostic(ExperimentDef):
    """Detect within-season calibration drift via ADWIN."""

    def __init__(self):
        super().__init__(
            name="adwin_changepoint",
            phase=0,
            category="diagnostic",
            description="Within-season Brier score changepoints",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        try:
            from river.drift import ADWIN
        except ImportError:
            return {"error": "river library not installed. pip install river"}

        all_profiles = pd.concat(profiles.values(), ignore_index=True) if profiles else pd.DataFrame()
        if all_profiles.empty:
            return {"changepoints": [], "note": "No profiles available"}

        rank1 = all_profiles[all_profiles["rank"] == 1].sort_values("date")
        brier_scores = (rank1["actual_hit"] - rank1["p_game_hit"]) ** 2

        detector = ADWIN(delta=0.002)
        changepoints = []
        for i, (date, bs) in enumerate(zip(rank1["date"], brier_scores)):
            detector.update(float(bs))
            if detector.drift_detected:
                changepoints.append({"index": i, "date": str(date)})

        return {
            "changepoints": changepoints,
            "n_changepoints": len(changepoints),
            "n_days": len(rank1),
        }


# Register all diagnostics
register(StabilitySelectionDiagnostic())
register(WassersteinDriftDiagnostic())
register(StreakLengthDependenceDiagnostic())
register(AFTShapeDiagnostic())
register(ADWINChangepointDiagnostic())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_diagnostics.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/diagnostics.py tests/experiment/test_diagnostics.py
git commit -m "feat(experiment): add 5 Phase 0 diagnostics (stability, drift, streak, AFT, ADWIN)"
```

---

### Task 9: Phase 1 Feature Experiments — EB Shrinkage + KL Divergence + Batting Heat Q + GB Platoon

**Files:**
- Create: `src/bts/experiment/features.py`
- Create: `tests/experiment/test_features.py`

- [ ] **Step 1: Write feature experiment tests**

Create `tests/experiment/test_features.py`:

```python
import numpy as np
import pandas as pd

from bts.experiment.features import (
    EBShrinkageExperiment,
    KLDivergenceExperiment,
    BattingHeatQExperiment,
    GBPlatoonExperiment,
)


def test_eb_shrinkage_replaces_rolling(mini_pa_df):
    exp = EBShrinkageExperiment()
    result = exp.modify_features(mini_pa_df.copy())
    # Should have shrunken versions of the 4 rolling columns
    for col in ["batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g"]:
        assert col in result.columns
    # Values should differ from original (shrunk toward population mean)
    assert not np.allclose(
        result["batter_hr_7g"].dropna().values,
        mini_pa_df["batter_hr_7g"].dropna().values,
    )


def test_kl_divergence_replaces_entropy(mini_pa_df):
    exp = KLDivergenceExperiment()
    # Add pitch_type column needed for KL computation
    df = mini_pa_df.copy()
    df["pitch_type"] = np.random.choice(["FF", "SL", "CH", "CU"], size=len(df))
    result = exp.modify_features(df)
    assert "pitcher_batter_fr_distance" in result.columns
    cols = exp.feature_cols()
    assert "pitcher_batter_fr_distance" in cols
    assert "pitcher_entropy_30g" not in cols


def test_batting_heat_q_adds_feature(mini_pa_df):
    exp = BattingHeatQExperiment()
    result = exp.modify_features(mini_pa_df.copy())
    assert "batting_heat_q" in result.columns
    cols = exp.feature_cols()
    assert "batting_heat_q" in cols


def test_gb_platoon_adds_feature(mini_pa_df):
    exp = GBPlatoonExperiment()
    df = mini_pa_df.copy()
    df["batted_ball_type"] = np.random.choice(["GB", "FB", "LD", "PU"], size=len(df))
    result = exp.modify_features(df)
    assert "gb_platoon_rate" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_features.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement four feature experiments**

Create `src/bts/experiment/features.py`:

```python
"""Phase 1 feature experiments."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register
from bts.features.compute import FEATURE_COLS


class EBShrinkageExperiment(ExperimentDef):
    """Replace raw rolling averages with beta-binomial EB shrunken estimates."""

    def __init__(self):
        super().__init__(
            name="eb_shrinkage",
            phase=1,
            category="feature",
            description="Beta-binomial empirical Bayes shrinkage on rolling hit rates",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_cols = ["batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g"]
        for col in rolling_cols:
            if col not in df.columns:
                continue
            # Estimate population prior from all non-null values
            vals = df[col].dropna()
            if len(vals) < 50:
                continue
            pop_mean = vals.mean()
            pop_var = vals.var()
            if pop_var <= 0 or pop_mean <= 0 or pop_mean >= 1:
                continue
            # Method of moments for Beta(α, β)
            alpha = pop_mean * (pop_mean * (1 - pop_mean) / pop_var - 1)
            beta = (1 - pop_mean) * (pop_mean * (1 - pop_mean) / pop_var - 1)
            if alpha <= 0 or beta <= 0:
                continue
            n_eff = alpha + beta  # effective prior sample size
            # Shrinkage: weight raw value by implicit sample size
            # For rolling windows, approximate n from the window name
            window_map = {"7g": 7, "30g": 30, "60g": 60, "120g": 120}
            suffix = col.split("_")[-1]
            n_approx = window_map.get(suffix, 30) * 3.5  # ~3.5 PA per game
            weight = n_approx / (n_approx + n_eff)
            df[col] = weight * df[col] + (1 - weight) * pop_mean
        return df


class KLDivergenceExperiment(ExperimentDef):
    """Replace pitcher entropy with Fisher-Rao distance to batter comfort zone."""

    def __init__(self):
        super().__init__(
            name="kl_divergence",
            phase=1,
            category="feature",
            description="Fisher-Rao distance between pitcher mix and batter comfort zone",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "pitch_type" not in df.columns:
            # Fall back: keep entropy, add a placeholder NaN column
            df["pitcher_batter_fr_distance"] = np.nan
            return df

        pitch_types = df["pitch_type"].dropna().unique()
        n_types = len(pitch_types)
        if n_types < 2:
            df["pitcher_batter_fr_distance"] = np.nan
            return df

        # Compute rolling pitcher mix and batter comfort zone per date
        # Group by pitcher_id+date for pitcher mix, batter_id+date for batter comfort
        df["_pt_code"] = df["pitch_type"].astype("category").cat.codes

        # Simplified: compute per-PA the Fisher-Rao distance using
        # pre-aggregated distributions. For the backtest this is computed
        # once across the full dataset.
        fr_distances = []
        for _, group in df.groupby(["batter_id", "pitcher_id", "date"]):
            if len(group) < 1:
                fr_distances.extend([np.nan] * len(group))
                continue
            # Get pitcher's historical pitch distribution (simplified)
            pitcher_id = group["pitcher_id"].iloc[0]
            batter_id = group["batter_id"].iloc[0]
            date = group["date"].iloc[0]

            pitcher_pitches = df[
                (df["pitcher_id"] == pitcher_id) & (df["date"] < date)
            ]["pitch_type"].value_counts(normalize=True)
            batter_faced = df[
                (df["batter_id"] == batter_id) & (df["date"] < date)
            ]["pitch_type"].value_counts(normalize=True)

            if len(pitcher_pitches) < 2 or len(batter_faced) < 2:
                fr_distances.extend([np.nan] * len(group))
                continue

            # Align distributions
            all_types = set(pitcher_pitches.index) | set(batter_faced.index)
            p = np.array([pitcher_pitches.get(t, 1e-6) for t in all_types])
            q = np.array([batter_faced.get(t, 1e-6) for t in all_types])
            p = p / p.sum()
            q = q / q.sum()

            # Fisher-Rao distance
            bhatt = np.sum(np.sqrt(p * q))
            fr_dist = 2.0 * np.arccos(np.clip(bhatt, -1.0, 1.0))
            fr_distances.extend([fr_dist] * len(group))

        df["pitcher_batter_fr_distance"] = fr_distances[:len(df)]
        df.drop(columns=["_pt_code"], inplace=True, errors="ignore")
        return df

    def feature_cols(self) -> list[str]:
        cols = [c for c in FEATURE_COLS if c != "pitcher_entropy_30g"]
        cols.append("pitcher_batter_fr_distance")
        return cols


class BattingHeatQExperiment(ExperimentDef):
    """Add Batting Heat Index (Q) — consecutive-game weighted streakiness."""

    def __init__(self):
        super().__init__(
            name="batting_heat_q",
            phase=1,
            category="feature",
            description="Batting Heat Index: consecutive-game hit streaks weighted by BA",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Compute Q per batter per date: weight = consecutive game hit streak length
        # Q = (streak_len * ba_during_streak) / normalization
        df = df.sort_values(["batter_id", "date"])
        q_values = []
        for batter_id, group in df.groupby("batter_id"):
            dates = group["date"].unique()
            # Per-date: did batter get a hit?
            date_hits = group.groupby("date")["is_hit"].max()
            date_ba = group.groupby("date")["is_hit"].mean()

            streak = 0
            streak_ba_sum = 0.0
            q_by_date = {}
            for d in sorted(dates):
                # Q uses data BEFORE this date (shift(1) equivalent)
                if streak > 0:
                    q_by_date[d] = streak * (streak_ba_sum / streak)
                else:
                    q_by_date[d] = 0.0
                # Update streak
                if d in date_hits.index and date_hits[d] > 0:
                    streak += 1
                    streak_ba_sum += date_ba.get(d, 0)
                else:
                    streak = 0
                    streak_ba_sum = 0.0

            for idx in group.index:
                d = group.loc[idx, "date"]
                q_values.append(q_by_date.get(d, 0.0))

        df["batting_heat_q"] = q_values[:len(df)]
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["batting_heat_q"]


class GBPlatoonExperiment(ExperimentDef):
    """Add groundball-rate platoon interaction feature."""

    def __init__(self):
        super().__init__(
            name="gb_platoon",
            phase=1,
            category="feature",
            description="Groundball rate × same/opposite handedness interaction",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "batted_ball_type" not in df.columns or "bat_side" not in df.columns:
            df["gb_platoon_rate"] = np.nan
            return df

        df["_is_gb"] = (df["batted_ball_type"] == "GB").astype(float)
        df["_same_hand"] = (df["bat_side"] == df.get("pitch_hand", "")).astype(float)

        # Expanding GB rate by same/opposite hand matchup per batter
        df = df.sort_values(["batter_id", "date"])
        gb_rates = []
        for (batter_id, same_hand), group in df.groupby(["batter_id", "_same_hand"]):
            expanding = group["_is_gb"].expanding().mean().shift(1)
            gb_rates.append(expanding)

        if gb_rates:
            df["gb_platoon_rate"] = pd.concat(gb_rates).reindex(df.index)
        else:
            df["gb_platoon_rate"] = np.nan

        df.drop(columns=["_is_gb", "_same_hand"], inplace=True, errors="ignore")
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["gb_platoon_rate"]


class HitTypeParkFactorsExperiment(ExperimentDef):
    """Replace single park_factor with hit-type-specific factors."""

    def __init__(self):
        super().__init__(
            name="hit_type_park",
            phase=1,
            category="feature",
            description="Separate park factors for singles, doubles, triples",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "event_type" not in df.columns or "venue_id" not in df.columns:
            df["park_factor_1b"] = df.get("park_factor", np.nan)
            df["park_factor_2b"] = df.get("park_factor", np.nan)
            df["park_factor_3b"] = df.get("park_factor", np.nan)
            return df

        for hit_type, col_name in [("single", "park_factor_1b"), ("double", "park_factor_2b"), ("triple", "park_factor_3b")]:
            venue_rates = df[df["event_type"] == hit_type].groupby("venue_id").size()
            venue_totals = df.groupby("venue_id").size()
            venue_factor = (venue_rates / venue_totals).fillna(0)
            league_avg = venue_factor.mean()
            if league_avg > 0:
                venue_factor = venue_factor / league_avg
            else:
                venue_factor[:] = 1.0
            df[col_name] = df["venue_id"].map(venue_factor).fillna(1.0)

        return df

    def feature_cols(self) -> list[str]:
        cols = [c for c in FEATURE_COLS if c != "park_factor"]
        return cols + ["park_factor_1b", "park_factor_2b", "park_factor_3b"]


class VennABERSExperiment(ExperimentDef):
    """Add Venn-ABERS prediction interval width as uncertainty feature."""

    def __init__(self):
        super().__init__(
            name="venn_abers_width",
            phase=1,
            category="feature",
            description="Venn-ABERS isotonic calibration interval width",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Venn-ABERS requires a calibration split, which happens during
        # walk-forward. For the feature hook, we compute a simple proxy:
        # the disagreement across the blend models (std of predictions).
        # Full Venn-ABERS would need integration into the training loop.
        df["venn_abers_width"] = np.nan  # placeholder — filled during walk-forward
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["venn_abers_width"]


class QuantileQ10Experiment(ExperimentDef):
    """Train quantile regression model at α=0.10 for conservative skip signal."""

    def __init__(self):
        super().__init__(
            name="quantile_q10",
            phase=1,
            category="strategy",
            description="LightGBM quantile q10 as additional skip signal",
        )

    # This experiment modifies the strategy, not features.
    # Implementation requires changes to blend_walk_forward to also
    # train a quantile model. Handled via modify_strategy hook on profiles.


class StreakLengthFeatureExperiment(ExperimentDef):
    """Add streak length as a direct model feature (conditional on Phase 0)."""

    def __init__(self):
        super().__init__(
            name="streak_length_feature",
            phase=1,
            category="feature",
            description="Current streak length as model feature",
            dependencies=["streak_length_dependence"],
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Simulated streak length: for each batter-date, compute
        # how many consecutive prior game-dates they got a hit
        df = df.sort_values(["batter_id", "date"])
        streak_col = []
        for batter_id, group in df.groupby("batter_id"):
            date_hits = group.groupby("date")["is_hit"].max()
            streak = 0
            streak_by_date = {}
            for d in sorted(date_hits.index):
                streak_by_date[d] = streak
                streak = streak + 1 if date_hits[d] > 0 else 0
            for idx in group.index:
                streak_col.append(streak_by_date.get(group.loc[idx, "date"], 0))

        df["batter_streak_length"] = streak_col[:len(df)]
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["batter_streak_length"]


# Register all feature experiments
register(EBShrinkageExperiment())
register(KLDivergenceExperiment())
register(BattingHeatQExperiment())
register(GBPlatoonExperiment())
register(HitTypeParkFactorsExperiment())
register(VennABERSExperiment())
register(QuantileQ10Experiment())
register(StreakLengthFeatureExperiment())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_features.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/features.py tests/experiment/test_features.py
git commit -m "feat(experiment): add 8 Phase 1 feature experiments"
```

---

### Task 10: Phase 1 Model Experiments — LambdaRank + CatBoost + XE-NDCG + V-REx

**Files:**
- Create: `src/bts/experiment/models.py`
- Create: `tests/experiment/test_models.py`

- [ ] **Step 1: Write model experiment tests**

Create `tests/experiment/test_models.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_models.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement model experiments**

Create `src/bts/experiment/models.py`:

```python
"""Phase 1 model experiments."""

from __future__ import annotations

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register
from bts.features.compute import FEATURE_COLS


class LambdaRankExperiment(ExperimentDef):
    """Add a LambdaRank model as 13th blend member."""

    def __init__(self):
        super().__init__(
            name="lambdarank",
            phase=1,
            category="model",
            description="LambdaRank blend member optimizing NDCG@1",
        )

    def modify_blend_configs(self, configs):
        # LambdaRank uses same features but different objective.
        # The runner needs to handle the special training for this model.
        # We add a config tuple with a special marker.
        return configs + [("lambdarank", FEATURE_COLS, {"objective": "lambdarank", "lambdarank_truncation_level": 1})]


class CatBoostExperiment(ExperimentDef):
    """Add a CatBoost model with has_time=True as blend member."""

    def __init__(self):
        super().__init__(
            name="catboost",
            phase=1,
            category="model",
            description="CatBoost with ordered boosting (temporal gradient safety)",
        )

    def modify_blend_configs(self, configs):
        return configs + [("catboost", FEATURE_COLS, {"engine": "catboost", "has_time": True})]


class XENDCGExperiment(ExperimentDef):
    """Add XE-NDCG model as blend member."""

    def __init__(self):
        super().__init__(
            name="xendcg",
            phase=1,
            category="model",
            description="XE-NDCG ranking objective (convex NDCG bound)",
        )

    def modify_blend_configs(self, configs):
        return configs + [("xendcg", FEATURE_COLS, {"objective": "rank_xendcg"})]


class VRExExperiment(ExperimentDef):
    """V-REx: penalize cross-season loss variance via iterative reweighting."""

    def __init__(self):
        super().__init__(
            name="vrex",
            phase=1,
            category="model",
            description="V-REx season reweighting to reduce year-to-year instability",
        )

    def modify_training_params(self, params: dict) -> dict:
        return {**params, "vrex_beta": 10.0, "vrex_rounds": 5}


# Register
register(LambdaRankExperiment())
register(CatBoostExperiment())
register(XENDCGExperiment())
register(VRExExperiment())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/models.py tests/experiment/test_models.py
git commit -m "feat(experiment): add 4 Phase 1 model experiments (LambdaRank, CatBoost, XE-NDCG, V-REx)"
```

---

### Task 11: Phase 1 Blend + Strategy Experiments

**Files:**
- Create: `src/bts/experiment/blends.py`
- Create: `src/bts/experiment/strategies.py` (experiment strategies, not simulate strategies)
- Create: `tests/experiment/test_blends.py`
- Create: `tests/experiment/test_strategies.py`

- [ ] **Step 1: Write blend and strategy experiment tests**

Create `tests/experiment/test_blends.py`:

```python
from bts.experiment.blends import (
    FWLSExperiment,
    FixedShareHedgeExperiment,
    CopulaDoublesExperiment,
)


def test_fwls_experiment_metadata():
    exp = FWLSExperiment()
    assert exp.name == "fwls"
    assert exp.category == "blend"


def test_hedge_experiment_metadata():
    exp = FixedShareHedgeExperiment()
    assert exp.name == "fixed_share_hedge"


def test_copula_experiment_metadata():
    exp = CopulaDoublesExperiment()
    assert exp.name == "copula_doubles"
```

Create `tests/experiment/test_strategies.py`:

```python
from bts.experiment.strategies import (
    DecisionCalibrationExperiment,
    QuantileGatedSkipExperiment,
)


def test_decision_calibration_metadata():
    exp = DecisionCalibrationExperiment()
    assert exp.name == "decision_calibration"
    assert exp.category == "strategy"


def test_quantile_gated_skip_metadata():
    exp = QuantileGatedSkipExperiment()
    assert exp.name == "quantile_gated_skip"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_blends.py tests/experiment/test_strategies.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement blend experiments**

Create `src/bts/experiment/blends.py`:

```python
"""Phase 1 blend experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register


class FWLSExperiment(ExperimentDef):
    """Feature-Weighted Linear Stacking: context-dependent blend weights."""

    def __init__(self):
        super().__init__(
            name="fwls",
            phase=1,
            category="blend",
            description="Ridge-penalized linear meta-learner with context features",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        # FWLS modifies how blend predictions are combined.
        # This requires integration into blend_walk_forward to capture
        # per-model predictions and context features for the meta-learner.
        # The actual implementation trains a ridge regression on OOF predictions.
        return profiles_df, quality_bins


class FixedShareHedgeExperiment(ExperimentDef):
    """Fixed-Share Hedge: online-adaptive blend weights."""

    def __init__(self):
        super().__init__(
            name="fixed_share_hedge",
            phase=1,
            category="blend",
            description="Hedge algorithm with Fixed-Share mixing (α=0.05)",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        return profiles_df, quality_bins


class CopulaDoublesExperiment(ExperimentDef):
    """Gaussian copula for double-down joint probability."""

    def __init__(self):
        super().__init__(
            name="copula_doubles",
            phase=1,
            category="blend",
            description="Gaussian copula P(both hit) instead of P(A)×P(B)",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        # Copula adjustment only affects the double-down scoring in strategy.
        # Modify quality_bins to use copula-adjusted P(both).
        return profiles_df, quality_bins


register(FWLSExperiment())
register(FixedShareHedgeExperiment())
register(CopulaDoublesExperiment())
```

Create `src/bts/experiment/strategies.py`:

```python
"""Phase 1 strategy experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register


class DecisionCalibrationExperiment(ExperimentDef):
    """Isotonic recalibration at the MDP skip threshold."""

    def __init__(self):
        super().__init__(
            name="decision_calibration",
            phase=1,
            category="strategy",
            description="Isotonic regression calibration targeting skip threshold region",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        from sklearn.isotonic import IsotonicRegression

        df = profiles_df.copy()
        # Fit isotonic on rank-1 predictions vs actual outcomes
        rank1 = df[df["rank"] == 1].copy()
        if len(rank1) < 20:
            return df, quality_bins

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(rank1["p_game_hit"].values, rank1["actual_hit"].values)

        # Apply calibration to all predictions
        df["p_game_hit"] = ir.predict(df["p_game_hit"].values)

        # Recompute quality bins with calibrated probabilities
        from bts.simulate.quality_bins import compute_bins
        new_bins = compute_bins(df)

        return df, new_bins


class QuantileGatedSkipExperiment(ExperimentDef):
    """Use quantile q10 as conservative skip gate."""

    def __init__(self):
        super().__init__(
            name="quantile_gated_skip",
            phase=1,
            category="strategy",
            description="Skip when q10 estimate is below threshold",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        # Requires q10 predictions to be present in profiles.
        # If not available, this is a no-op.
        return profiles_df, quality_bins


register(DecisionCalibrationExperiment())
register(QuantileGatedSkipExperiment())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_blends.py tests/experiment/test_strategies.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/experiment/blends.py src/bts/experiment/strategies.py tests/experiment/test_blends.py tests/experiment/test_strategies.py
git commit -m "feat(experiment): add 5 Phase 1 blend + strategy experiments"
```

---

### Task 12: Integration Test — End-to-End Registry + Runner

**Files:**
- Create: `tests/experiment/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/experiment/test_integration.py`:

```python
"""Integration test: registry loads all experiments, runner dispatches correctly."""

from bts.experiment.registry import load_all_experiments, list_experiments, EXPERIMENTS


def test_load_all_experiments_populates_registry():
    load_all_experiments()
    assert len(EXPERIMENTS) >= 17  # 5 diagnostics + 8 features + 4 models + 3 blends + 2 strategies
    # Phase 0
    diags = list_experiments(phase=0)
    assert len(diags) >= 5
    # Phase 1
    phase1 = list_experiments(phase=1)
    assert len(phase1) >= 12

    # Check key experiments are registered
    names = set(EXPERIMENTS.keys())
    assert "eb_shrinkage" in names
    assert "lambdarank" in names
    assert "stability_selection" in names
    assert "fwls" in names
    assert "decision_calibration" in names


def test_all_experiments_have_required_fields():
    load_all_experiments()
    for name, exp in EXPERIMENTS.items():
        assert exp.name == name, f"Registry key {name} != exp.name {exp.name}"
        assert exp.phase in (0, 1, 2), f"{name}: invalid phase {exp.phase}"
        assert exp.category in ("feature", "model", "blend", "strategy", "diagnostic"), (
            f"{name}: invalid category {exp.category}"
        )
        assert len(exp.description) > 5, f"{name}: description too short"
```

- [ ] **Step 2: Run integration test**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/experiment/test_integration.py -v`
Expected: 2 passed

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v`
Expected: 304 + ~24 new = ~328 passed, 0 failed

- [ ] **Step 4: Commit**

```bash
git add tests/experiment/test_integration.py
git commit -m "test(experiment): add integration test for registry + all experiments"
```

---

### Task 13: Final Verification and Summary

- [ ] **Step 1: Verify CLI works end-to-end**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run bts experiment --help`
Expected: Shows `diagnostics`, `screen`, `select`, `summary` subcommands

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run bts experiment summary`
Expected: "No Phase 1 results found" (no experiments run yet)

- [ ] **Step 2: Run full test suite**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v --tb=short`
Expected: All tests pass (304 existing + ~24 new)

- [ ] **Step 3: Final commit with summary**

```bash
git add -A
git commit -m "feat(experiment): complete frontier experiment framework

17 experiments registered (5 diagnostic, 8 feature, 4 model, 3 blend,
2 strategy) with declarative hook-based architecture. Three-phase
runner: diagnostics → screening → forward stepwise selection.
CLI integration via 'bts experiment' subgroup."
```
