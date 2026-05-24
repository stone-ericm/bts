from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bts.simulate.mdp import ACTIONS
from bts.simulate.quality_bins import QualityBin, QualityBins
from scripts.gate_b_production_metric_rebaseline import (
    evaluate_policy_reach_probability,
    exact_policy_metrics,
    run_measurement,
)


def _write_boundary_policy(path, *, season_length=20):
    policy = np.full(
        (57, season_length + 1, 2, 2),
        ACTIONS.index("skip"),
        dtype=np.int8,
    )
    policy[:, :, :, 1] = ACTIONS.index("single")
    np.savez_compressed(
        path,
        policy_table=policy,
        boundaries=np.array([0.75]),
        season_length=np.array(season_length),
        optimal_p57=np.array(0.0),
    )


def _write_profiles(profiles_dir, seasons=(2021, 2022, 2023)):
    profiles_dir.mkdir()
    for season in seasons:
        rows = []
        for day in range(20):
            d = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day)
            p1 = 0.70 if day % 2 == 0 else 0.72
            for rank, p in [(1, p1), (2, 0.69)]:
                rows.append({
                    "date": d.date(),
                    "season": season,
                    "rank": rank,
                    "batter_id": season * 1000 + day * 10 + rank,
                    "game_pk": season * 100 + day,
                    "p_game_hit": p,
                    "actual_hit": 1,
                    "n_pas": 4,
                    "p_game_hit_basis": "estimated_pa",
                })
        pd.DataFrame(rows).to_parquet(profiles_dir / f"backtest_{season}.parquet", index=False)


def _write_picks(picks_dir, *, n=15):
    picks_dir.mkdir()
    for day in range(n):
        date = pd.Timestamp("2026-04-01") + pd.Timedelta(days=day)
        body = {
            "date": date.date().isoformat(),
            "run_time": f"{date.date().isoformat()}T15:00:00+00:00",
            "pick": {
                "batter_name": f"Batter {day}",
                "batter_id": 1000 + day,
                "team": "AAA",
                "lineup_position": 1,
                "pitcher_name": "Pitcher",
                "pitcher_id": 2000 + day,
                "p_game_hit": 0.70 if day % 2 == 0 else 0.72,
                "flags": [],
                "projected_lineup": False,
                "game_pk": 9000 + day,
                "game_time": f"{date.date().isoformat()}T20:00:00Z",
                "pitcher_team": "BBB",
            },
            "double_down": None,
            "runner_up": None,
            "bluesky_posted": False,
            "bluesky_uri": None,
            "notification_sent": True,
            "notification_channel": "bluesky_dm",
            "notification_id": f"notif-{day}",
            "result": "hit",
            "slot_results": {"pick": "hit"},
        }
        (picks_dir / f"{date.date().isoformat()}.json").write_text(json.dumps(body))


def test_exact_reach_and_expected_max_are_deterministic_for_all_hit_singles():
    policy = np.full((57, 4, 2, 1), ACTIONS.index("single"), dtype=np.int8)
    bins = QualityBins(
        bins=[QualityBin(index=0, p_range=(0.7, 0.7), p_hit=1.0, p_both=1.0, frequency=1.0)],
        boundaries=[],
    )

    assert evaluate_policy_reach_probability(policy, bins, target=3, season_length=3) == 1.0
    assert evaluate_policy_reach_probability(policy, bins, target=4, season_length=3) == 0.0
    metrics = exact_policy_metrics(policy, bins, season_length=3, ladder_targets=[1, 2, 3])
    assert metrics["expected_max_streak"] == 3.0
    assert metrics["reach_probabilities"] == {"1": 1.0, "2": 1.0, "3": 1.0}


def test_run_measurement_keeps_action_table_fixed_and_tests_boundary_only(tmp_path):
    profiles_dir = tmp_path / "profiles"
    picks_dir = tmp_path / "picks"
    policy_path = tmp_path / "mdp_policy.npz"
    _write_profiles(profiles_dir)
    _write_picks(picks_dir)
    _write_boundary_policy(policy_path)

    result = run_measurement(
        profiles_dir=profiles_dir,
        picks_dir=picks_dir,
        prod_policy_path=policy_path,
        seasons=[2021, 2022, 2023],
        n_bins=2,
        season_length=20,
        today=pd.Timestamp("2026-04-20").date(),
    )

    assert result["production_deploy_claim"] is False
    assert result["writes_policy_artifact"] is False
    assert result["methodology"]["comparator"] == "boundary_only_deployed_action_table_fixed"
    assert result["mechanism"]["decision"] == "MECHANISM_PASSES_NOT_SWAP_JUSTIFYING"
    assert result["mechanism"]["actions"]["changed_decision_count"] > 0
    assert result["mechanism"]["bin_occupancy"]["current_primary_alerts"] is True
    assert result["mechanism"]["bin_occupancy"]["candidate_primary_alerts"] is False

    outcome = result["outcome_quality"]
    assert outcome["decision"] == "OUTCOME_POSITIVE_REQUIRES_FULL_GATE"
    assert outcome["metric_summaries"]["expected_max_streak"]["mean_gap"] > 0
    assert outcome["metric_summaries"]["p_reach_10"]["active_after_floor_guard"] is False
    assert all(
        fold["candidate_metrics"]["expected_max_streak"]
        > fold["current_metrics"]["expected_max_streak"]
        for fold in outcome["folds"]
    )
