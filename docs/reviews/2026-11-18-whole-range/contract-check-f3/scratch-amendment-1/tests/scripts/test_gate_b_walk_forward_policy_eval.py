from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import ACTIONS
from scripts.gate_b_walk_forward_policy_eval import (
    load_profiles,
    parse_seasons,
    run_evaluation,
)


def _write_policy(path, *, season_length=60, n_bins=1, action="skip"):
    policy = np.full(
        (58, season_length + 1, 2, n_bins),
        ACTIONS.index(action),
        dtype=np.int8,
    )
    np.savez_compressed(
        path,
        policy_table=policy,
        boundaries=np.array([]),
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


def test_parse_seasons_requires_at_least_two():
    assert parse_seasons("2021,2022") == [2021, 2022]
    with pytest.raises(ValueError, match="at least two"):
        parse_seasons("2021")


def test_load_profiles_requires_estimated_basis_marker(tmp_path):
    profiles_dir = tmp_path / "profiles"
    _write_profiles(profiles_dir, seasons=(2021,), marked=False)

    with pytest.raises(ValueError, match="missing p_game_hit_basis"):
        load_profiles(profiles_dir, [2021])

    loaded = load_profiles(profiles_dir, [2021], require_estimated_basis=False)
    assert len(loaded) == 70


def test_run_evaluation_uses_expanding_prior_season_folds(tmp_path):
    profiles_dir = tmp_path / "profiles"
    policy_path = tmp_path / "deployed_policy.npz"
    _write_profiles(profiles_dir)
    _write_policy(policy_path, season_length=60, n_bins=1, action="skip")

    result = run_evaluation(
        profiles_dir=profiles_dir,
        prod_policy_path=policy_path,
        seasons=[2021, 2022, 2023],
        n_bins=1,
        season_length=60,
    )

    assert result["production_deploy_claim"] is False
    assert result["writes_policy_artifact"] is False
    assert result["decision"] == "WALK_FORWARD_SIGNAL_POSITIVE_REQUIRES_REBASELINE"
    assert [fold["holdout_season"] for fold in result["folds"]] == [2022, 2023]
    assert result["folds"][0]["train_seasons"] == [2021]
    assert result["folds"][1]["train_seasons"] == [2021, 2022]
    assert all(fold["gap"] > 0 for fold in result["folds"])
    assert result["starter_matchup_drop_summary"]["available"] is True
    assert result["starter_matchup_drop_summary"]["overall"]["dropped_fraction"] == pytest.approx(1 / 12)
