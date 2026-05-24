from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import ACTIONS
from scripts.measure_pa_basis_rebin_gate import (
    aggregate_pa_probability,
    bootstrap_gap_interval,
    implied_pa_probability,
    pair_frame_from_profiles,
    run_measurement,
    transform_profiles_to_target_pas,
)


def _write_policy(path, *, n_days=12):
    policy = np.full((58, n_days + 1, 2, 5), ACTIONS.index("single"), dtype=np.int8)
    policy[:, :, :, 0] = ACTIONS.index("skip")
    np.savez_compressed(
        path,
        policy_table=policy,
        boundaries=np.array([0.80, 0.82, 0.84, 0.86]),
        season_length=np.array(n_days),
        optimal_p57=np.array(0.0),
    )


def _write_profiles(profiles_dir, *, n_days=8):
    profiles_dir.mkdir()
    rows = []
    for i in range(n_days):
        d = pd.Timestamp("2025-04-01") + pd.Timedelta(days=i)
        rows.append({
            "date": d.date(),
            "rank": 1,
            "p_game_hit": 0.80 + i * 0.01,
            "actual_hit": 1 if i % 3 else 0,
            "n_pas": 6,
        })
        rows.append({
            "date": d.date(),
            "rank": 2,
            "p_game_hit": 0.78 + i * 0.01,
            "actual_hit": 1 if i % 4 else 0,
            "n_pas": 5,
        })
    pd.DataFrame(rows).to_parquet(profiles_dir / "backtest_2025.parquet", index=False)


def test_product_aggregation_inversion_round_trips():
    p_game = 0.81
    n_pas = 6

    p_pa = implied_pa_probability(p_game, n_pas)
    recovered = aggregate_pa_probability(p_pa, n_pas)

    assert recovered == pytest.approx(p_game)


def test_transform_profiles_uses_rank_specific_target_pas():
    profiles = pd.DataFrame([
        {"date": "2025-04-01", "rank": 1, "p_game_hit": 0.80, "actual_hit": 1, "n_pas": 6},
        {"date": "2025-04-01", "rank": 2, "p_game_hit": 0.78, "actual_hit": 1, "n_pas": 5},
    ])

    transformed = transform_profiles_to_target_pas(
        profiles,
        rank1_target_pas=4.0,
        rank2_target_pas=3.5,
    )

    rank1 = transformed[transformed["rank"] == 1].iloc[0]
    rank2 = transformed[transformed["rank"] == 2].iloc[0]
    assert rank1["source_p_game_hit"] == pytest.approx(0.80)
    assert rank1["p_game_hit"] < rank1["source_p_game_hit"]
    assert rank1["pa_basis_target_pas"] == pytest.approx(4.0)
    assert rank2["pa_basis_target_pas"] == pytest.approx(3.5)


def test_run_measurement_reports_exploratory_gate(tmp_path):
    profiles_dir = tmp_path / "simulation"
    policy_path = tmp_path / "mdp_policy.npz"
    _write_profiles(profiles_dir, n_days=8)
    _write_policy(policy_path, n_days=12)

    result = run_measurement(
        profiles_dir=profiles_dir,
        policy_path=policy_path,
        rank1_target_pas=4.0,
        rank2_target_pas=3.8,
        n_bins_values=(2,),
        min_per_bin=2,
        season_length=12,
        bootstrap_reps=5,
        bootstrap_seed=7,
        today=date(2026, 5, 24),
    )

    assert result["production_deploy_claim"] is False
    assert result["heavy_compute"] is False
    assert result["raw_distribution"]["1"]["n"] == 8
    assert result["pa_basis_distribution"]["1"]["mean"] < result["raw_distribution"]["1"]["mean"]
    assert result["gate_b_screen"]["evaluations"][0]["n_bins"] == 2
    assert result["gate_b_screen"]["evaluations"][0]["bootstrap_gap"]["n_bootstrap"] == 5
    assert len(result["gate_b_screen"]["evaluations"][0]["bootstrap_gap"]["ci95"]) == 2
    assert result["methodology"]["not_full_gate"].startswith("this is an exploratory")
    assert "in-sample/optimistic" in result["methodology"]["not_full_gate"]


def test_pair_bootstrap_can_be_skipped(tmp_path):
    profiles_dir = tmp_path / "simulation"
    _write_profiles(profiles_dir, n_days=8)
    profiles = transform_profiles_to_target_pas(
        pd.read_parquet(profiles_dir / "backtest_2025.parquet"),
        rank1_target_pas=4.0,
        rank2_target_pas=3.8,
    )
    pairs = pair_frame_from_profiles(profiles)

    assert bootstrap_gap_interval(
        pairs,
        n_bins=2,
        policy_table=np.zeros((58, 13, 2, 2), dtype=np.int8),
        policy_boundaries=[0.82],
        season_length=12,
        n_bootstrap=0,
        seed=42,
    ) is None
