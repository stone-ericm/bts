from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import ACTIONS
from scripts.evaluate_mdp_dd_guardrail import (
    build_row_days,
    evaluate_reach_probability_row_stream,
    run_evaluation,
)


def _write_policy(path, *, season_length=6):
    policy = np.full(
        (57, season_length + 1, 2, 2),
        ACTIONS.index("single"),
        dtype=np.int8,
    )
    policy[:, :, :, 0] = ACTIONS.index("double")
    policy[:, :, :, 1] = ACTIONS.index("single")
    np.savez_compressed(
        path,
        policy_table=policy,
        boundaries=np.array([0.80]),
        season_length=np.array(season_length),
        optimal_p57=np.array(0.0),
    )


def _profile_frame(*, include_game_pk=True, include_season=True, season=2021):
    rows = [
        {
            "date": "2021-04-01",
            "season": season,
            "rank": 1,
            "batter_id": 1,
            "p_game_hit": 0.70,
            "actual_hit": 1,
            "game_pk": 100,
        },
        {
            "date": "2021-04-01",
            "season": season,
            "rank": 2,
            "batter_id": 2,
            "p_game_hit": 0.75,
            "actual_hit": 1,
            "game_pk": 100,
        },
        {
            "date": "2021-04-01",
            "season": season,
            "rank": 3,
            "batter_id": 3,
            "p_game_hit": 0.60,
            "actual_hit": 0,
            "game_pk": 101,
        },
    ]
    frame = pd.DataFrame(rows)
    if not include_game_pk:
        frame = frame.drop(columns=["game_pk"])
    if not include_season:
        frame = frame.drop(columns=["season"])
    return frame


def test_build_row_days_selects_first_different_game_candidate():
    days = build_row_days(_profile_frame(), 2021)

    assert len(days) == 1
    assert days[0].double_rank == 3
    assert days[0].double_p == pytest.approx(0.60)
    assert days[0].p_both == pytest.approx(0.42)


def test_build_row_days_rank2_proxy_uses_rank2_without_game_pk():
    days = build_row_days(_profile_frame(include_game_pk=False), 2021, allow_rank2_proxy=True)

    assert len(days) == 1
    assert days[0].double_rank == 2
    assert days[0].double_p == pytest.approx(0.75)


def test_first_passage_counts_double_jump_crossing_threshold(tmp_path):
    policy_path = tmp_path / "policy.npz"
    _write_policy(policy_path, season_length=1)
    policy = np.load(policy_path)["policy_table"]
    days = build_row_days(_profile_frame(), 2021)

    result = evaluate_reach_probability_row_stream(
        days,
        policy,
        [0.80],
        1,
        target=9,
        floor=None,
        initial_streak=8,
        initial_saver_available=False,
    )

    assert result.probability == pytest.approx(0.42)


def test_guardrail_downgrades_double_to_single_when_pair_floor_fires(tmp_path):
    policy_path = tmp_path / "policy.npz"
    _write_policy(policy_path, season_length=1)
    policy = np.load(policy_path)["policy_table"]
    days = build_row_days(_profile_frame(), 2021)

    result = evaluate_reach_probability_row_stream(
        days,
        policy,
        [0.80],
        1,
        target=9,
        floor=0.50,
        initial_streak=8,
        initial_saver_available=False,
    )

    assert result.probability == pytest.approx(0.70)
    assert result.changed_date_count == 1
    assert result.changed_state_count == 1


def test_run_evaluation_marks_missing_game_pk_invalid_primary_surface(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _profile_frame(include_game_pk=False).to_parquet(profiles_dir / "backtest_2021.parquet")
    policy_path = tmp_path / "policy.npz"
    _write_policy(policy_path)

    result = run_evaluation(
        profiles_dir=profiles_dir,
        policy_path=policy_path,
        seasons=[2021],
        floors=[0.50],
    )

    assert result["production_deploy_claim"] is False
    assert result["writes_policy_artifact"] is False
    assert result["floors"]["0.50"]["label"] == "INVALID_PRIMARY_SURFACE"
    assert result["profile_schema"]["missing_columns_by_season"] == {"2021": ["game_pk"]}
    json.dumps(result)


def test_rank2_proxy_run_is_diagnostic_not_primary_valid(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _profile_frame(include_game_pk=False).to_parquet(profiles_dir / "backtest_2021.parquet")
    policy_path = tmp_path / "policy.npz"
    _write_policy(policy_path)

    result = run_evaluation(
        profiles_dir=profiles_dir,
        policy_path=policy_path,
        seasons=[2021],
        floors=[0.50],
        allow_rank2_proxy=True,
    )

    assert result["profile_schema"]["allow_rank2_proxy"] is True
    assert result["floors"]["0.50"]["label"] == "INVALID_PRIMARY_SURFACE"
    assert result["backtest_p_both_distribution"]["n"] == 1


def test_rank2_proxy_can_infer_season_from_legacy_filename(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _profile_frame(include_game_pk=False, include_season=False).to_parquet(
        profiles_dir / "backtest_2021.parquet"
    )
    policy_path = tmp_path / "policy.npz"
    _write_policy(policy_path)

    result = run_evaluation(
        profiles_dir=profiles_dir,
        policy_path=policy_path,
        seasons=[2021],
        floors=[0.50],
        allow_rank2_proxy=True,
    )

    assert result["profile_schema"]["valid"] is True
    assert result["profile_schema"]["inferred_season_by_season"] == {"2021": True}
    assert result["floors"]["0.50"]["label"] == "INVALID_PRIMARY_SURFACE"
    assert result["backtest_p_both_distribution"]["n"] == 1


def test_run_evaluation_accepts_sourced_production_p_both_summary(tmp_path):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    _profile_frame().to_parquet(profiles_dir / "backtest_2021.parquet")
    policy_path = tmp_path / "policy.npz"
    _write_policy(policy_path)
    summary_path = tmp_path / "prod_summary.json"
    summary_path.write_text(json.dumps({"n": 58, "mean": 0.5447, "source": "test"}))

    result = run_evaluation(
        profiles_dir=profiles_dir,
        policy_path=policy_path,
        seasons=[2021],
        floors=[0.50],
        production_p_both_summary_path=summary_path,
    )

    assert result["production_p_both_distribution"] == {
        "n": 58,
        "mean": 0.5447,
        "source": "test",
    }
