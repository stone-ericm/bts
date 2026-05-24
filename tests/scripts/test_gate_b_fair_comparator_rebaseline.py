from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import ACTIONS
from scripts.gate_b_fair_comparator_rebaseline import run_rebaseline


def _write_policy(path, *, season_length=60, n_bins=1, action="skip"):
    policy = np.full(
        (57, season_length + 1, 2, n_bins),
        ACTIONS.index(action),
        dtype=np.int8,
    )
    np.savez_compressed(
        path,
        policy_table=policy,
        boundaries=np.array([] if n_bins == 1 else [0.75]),
        season_length=np.array(season_length),
        optimal_p57=np.array(0.0),
    )


def _write_profiles(profiles_dir, seasons=(2021, 2022, 2023), *, marked=True):
    profiles_dir.mkdir()
    for season in seasons:
        rows = []
        for day in range(35):
            d = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day)
            for rank, p in [(1, 0.80), (2, 0.78)]:
                row = {
                    "date": d.date(),
                    "season": season,
                    "rank": rank,
                    "batter_id": season * 1000 + day * 10 + rank,
                    "game_pk": season * 100 + day,
                    "p_game_hit": p,
                    "actual_hit": 1,
                    "n_pas": 4,
                }
                if marked:
                    row["p_game_hit_basis"] = "estimated_pa"
                    row["total_batter_games"] = 12
                    row["starter_matchup_batter_games"] = 11
                    row["dropped_no_starter_matchup"] = 1
                rows.append(row)
        pd.DataFrame(rows).to_parquet(profiles_dir / f"backtest_{season}.parquet", index=False)


def test_run_rebaseline_holds_bins_fixed_and_varies_action_table(tmp_path):
    profiles_dir = tmp_path / "profiles"
    policy_path = tmp_path / "deployed_policy.npz"
    _write_profiles(profiles_dir)
    _write_policy(policy_path, season_length=60, n_bins=1, action="skip")

    result = run_rebaseline(
        profiles_dir=profiles_dir,
        prod_policy_path=policy_path,
        seasons=[2021, 2022, 2023],
        n_bins=1,
        season_length=60,
    )

    assert result["production_deploy_claim"] is False
    assert result["writes_policy_artifact"] is False
    assert result["decision"] == "RE_SOLVE_ACTION_TABLE_SIGNAL_POSITIVE_REQUIRES_FULL_GATE"
    assert [fold["holdout_season"] for fold in result["folds"]] == [2022, 2023]
    assert all(
        fold["re_solved_candidate_p57"] > fold["deployed_action_structure_p57"]
        for fold in result["folds"]
    )
    assert all(
        fold["shared_train_bins"]["boundaries"] == fold["shared_holdout_bins"]["boundaries"]
        for fold in result["folds"]
    )
    assert all(
        fold["action_table_comparison"]["same_action_fraction"] < 1.0
        for fold in result["folds"]
    )
    assert result["starter_matchup_drop_summary"]["available"] is True


def test_run_rebaseline_rejects_deployed_policy_bin_mismatch(tmp_path):
    profiles_dir = tmp_path / "profiles"
    policy_path = tmp_path / "deployed_policy.npz"
    _write_profiles(profiles_dir)
    _write_policy(policy_path, season_length=60, n_bins=2, action="skip")

    with pytest.raises(ValueError, match="deployed policy has 2 bins"):
        run_rebaseline(
            profiles_dir=profiles_dir,
            prod_policy_path=policy_path,
            seasons=[2021, 2022],
            n_bins=1,
            season_length=60,
        )


def test_run_rebaseline_requires_estimated_pa_marker(tmp_path):
    profiles_dir = tmp_path / "profiles"
    policy_path = tmp_path / "deployed_policy.npz"
    _write_profiles(profiles_dir, marked=False)
    _write_policy(policy_path, season_length=60, n_bins=1, action="skip")

    with pytest.raises(ValueError, match="missing p_game_hit_basis"):
        run_rebaseline(
            profiles_dir=profiles_dir,
            prod_policy_path=policy_path,
            seasons=[2021, 2022],
            n_bins=1,
            season_length=60,
        )
