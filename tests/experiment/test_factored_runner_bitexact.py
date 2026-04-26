"""Bit-exact validation: factored runner produces identical output to current runner
on strategy experiments (first factored path we enable).

Note on `quantile_q10` and `kl_divergence` from the plan: empirically these
experiments are NOT strategy-only — `quantile_q10` overrides
`modify_blend_configs` AND `requires_per_model_capture`, and `kl_divergence`
overrides `modify_features` + `feature_cols`. The eligibility detector
correctly rejects both. The truly eligible experiments are
`decision_calibration` and `venn_abers_width`; those are what the
parametrized bit-exact path validates. The other two are checked via the
"must-refuse" assertion.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.experiment.registry import load_all_experiments, get_experiment
from bts.features.compute import compute_all_features


def _strip_timestamp(scorecard: dict) -> dict:
    """Remove non-deterministic fields before bit-exact comparison.

    `compute_full_scorecard` stamps `datetime.now(timezone.utc).isoformat()`
    at every call, so two identical computations produce different JSON.
    Stripping `timestamp` reduces the comparison to the actual numeric
    content of the scorecard.
    """
    out = dict(scorecard)
    out.pop("timestamp", None)
    return out


# ---------------------------------------------------------------------------
# Refuse-eligibility tests — fast: use a tiny synthetic profiles DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_profiles() -> pd.DataFrame:
    """Tiny synthetic profiles for refuse-tests; never reaches the actual
    fast-path body because the eligibility check raises first.

    Mirrors the schema of blend_walk_forward output: date, rank, batter_id,
    p_game_hit, actual_hit, n_pas, season.
    """
    rng = np.random.default_rng(42)
    rows = []
    for season in (2024, 2025):
        for date in pd.date_range(f"{season}-04-01", periods=10, freq="D"):
            for rank in range(1, 11):
                rows.append({
                    "date": date.date(),
                    "rank": rank,
                    "batter_id": 100000 + rank,
                    "p_game_hit": 0.95 - rank * 0.04 + rng.normal(0, 0.01),
                    "actual_hit": int(rng.random() > 0.3),
                    "n_pas": 4,
                    "season": season,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_scorecard(synthetic_profiles) -> dict:
    """Synthetic baseline scorecard for refuse-tests.

    A minimal dict — refuse path raises before consuming it.
    """
    return {
        "p_at_1_by_season": {2024: 0.85, 2025: 0.87},
        "streak_metrics": {"mean_max_streak": 18.0},
        "p_57_exact": 0.08,
        "p_57_mdp": 0.082,
    }


def test_fast_path_refuses_feature_experiment(synthetic_profiles, synthetic_scorecard, tmp_path):
    """Fast-path function must reject experiments that modify features."""
    from bts.experiment.runner_factored import run_strategy_experiment_fast

    load_all_experiments()
    exp = get_experiment("wind_vector")  # modifies features

    with pytest.raises(ValueError, match="not eligible for fast strategy path"):
        run_strategy_experiment_fast(exp, synthetic_profiles, synthetic_scorecard, tmp_path)


@pytest.mark.parametrize(
    "exp_name",
    ["quantile_q10", "kl_divergence"],
)
def test_fast_path_refuses_ineligible_named_experiments(
    exp_name, synthetic_profiles, synthetic_scorecard, tmp_path
):
    """The plan's quantile_q10/kl_divergence are NOT strategy-only — they
    override blend configs / per-model capture / features respectively, so
    the eligibility check must reject them."""
    from bts.experiment.runner_factored import run_strategy_experiment_fast

    load_all_experiments()
    exp = get_experiment(exp_name)

    with pytest.raises(ValueError, match="not eligible for fast strategy path"):
        run_strategy_experiment_fast(exp, synthetic_profiles, synthetic_scorecard, tmp_path)


# ---------------------------------------------------------------------------
# Bit-exact comparison tests — slow: real walk-forward via fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_pa_df() -> pd.DataFrame:
    """Load + feature-enrich a small PA frame (2 seasons for speed)."""
    proc = Path("data/processed")
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    return compute_all_features(df)


@pytest.fixture(scope="module")
def baseline_profiles(test_pa_df) -> pd.DataFrame:
    """Produce baseline profiles via standard walk-forward — one-time for the test module."""
    from bts.simulate.backtest_blend import blend_walk_forward
    profiles = []
    for season in [2024, 2025]:
        p = blend_walk_forward(test_pa_df, season, retrain_every=7)
        p["season"] = season
        profiles.append(p)
    return pd.concat(profiles, ignore_index=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "exp_name",
    ["decision_calibration", "venn_abers_width"],
)
def test_strategy_path_matches_full_walkforward(
    exp_name, tmp_path, test_pa_df, baseline_profiles
):
    """For each truly strategy-only experiment, the factored path should
    produce a scorecard/diff matching the current full-walk-forward path
    bit-exactly (modulo the non-deterministic timestamp field)."""
    from bts.experiment.runner import run_single_screening
    from bts.experiment.runner_factored import run_strategy_experiment_fast
    from bts.validate.scorecard import compute_full_scorecard

    load_all_experiments()
    exp = get_experiment(exp_name)
    baseline_scorecard = compute_full_scorecard(baseline_profiles)

    # Current path — full walk-forward, then modify_strategy
    dir_current = tmp_path / "current"
    dir_current.mkdir()
    run_single_screening(
        exp, test_pa_df, baseline_scorecard, [2024, 2025],
        dir_current, retrain_every=7,
    )

    # Factored path — reuses baseline_profiles, no walk-forward
    dir_factored = tmp_path / "factored"
    dir_factored.mkdir()
    run_strategy_experiment_fast(
        exp, baseline_profiles, baseline_scorecard, dir_factored,
    )

    sc_current = _strip_timestamp(json.loads((dir_current / exp.name / "scorecard.json").read_text()))
    sc_factored = _strip_timestamp(json.loads((dir_factored / exp.name / "scorecard.json").read_text()))
    assert sc_current == sc_factored, (
        f"scorecard mismatch for {exp_name}:\n"
        f"  current: {sc_current}\n"
        f"  factored: {sc_factored}"
    )

    df_current = json.loads((dir_current / exp.name / "diff.json").read_text())
    df_factored = json.loads((dir_factored / exp.name / "diff.json").read_text())
    assert df_current == df_factored, (
        f"diff mismatch for {exp_name}:\n"
        f"  current: {df_current}\n"
        f"  factored: {df_factored}"
    )
