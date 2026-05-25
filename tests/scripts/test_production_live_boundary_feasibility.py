from __future__ import annotations

import json

import pandas as pd

from scripts.production_live_boundary_feasibility import build_inventory


def _pick_body(day: str, p1: float, p2: float, *, git=None, policy=None, feature_env=None):
    return {
        "date": day,
        "run_time": f"{day}T15:00:00+00:00",
        "pick": {
            "batter_name": f"Primary {day}",
            "batter_id": int(day[-2:]),
            "team": "AAA",
            "lineup_position": 1,
            "pitcher_name": "Pitcher",
            "pitcher_id": 100,
            "p_game_hit": p1,
            "flags": [],
            "projected_lineup": False,
            "game_pk": 9000 + int(day[-2:]),
            "game_time": f"{day}T20:00:00Z",
            "pitcher_team": "BBB",
        },
        "double_down": {
            "batter_name": f"Double {day}",
            "batter_id": 1000 + int(day[-2:]),
            "team": "CCC",
            "lineup_position": 2,
            "pitcher_name": "Pitcher 2",
            "pitcher_id": 200,
            "p_game_hit": p2,
            "flags": [],
            "projected_lineup": False,
            "game_pk": 8000 + int(day[-2:]),
            "game_time": f"{day}T21:00:00Z",
            "pitcher_team": "DDD",
        },
        "runner_up": None,
        "result": None,
        "slot_results": None,
        "model_git_sha": git,
        "model_pickle_sha256": f"pickle-{feature_env or day}",
        "policy_npz_sha256": policy,
        "feature_env_schema_version": (
            "bts_feature_env_v1" if feature_env is not None else None
        ),
        "feature_env_hash": feature_env,
    }


def _write_picks(picks_dir):
    picks_dir.mkdir()
    rows = [
        ("2026-04-01", 0.70, 0.69, None, None, None),
        ("2026-04-02", 0.71, 0.70, None, None, None),
        ("2026-04-03", 0.72, 0.71, "git-a", "policy-a", "env-a"),
        ("2026-04-04", 0.73, 0.72, "git-a", "policy-a", "env-a"),
        ("2026-04-05", 0.74, 0.73, "git-b", "policy-a", "env-a"),
        ("2026-04-06", 0.75, 0.74, "git-b", "policy-a", "env-b"),
        ("2026-04-07", 0.76, 0.75, "git-c", "policy-b", "env-b"),
    ]
    for day, p1, p2, git, policy, feature_env in rows:
        (picks_dir / f"{day}.json").write_text(
            json.dumps(
                _pick_body(
                    day,
                    p1,
                    p2,
                    git=git,
                    policy=policy,
                    feature_env=feature_env,
                )
            )
        )
    (picks_dir / "lineup_evolution_2026-04-06.jsonl").write_text(
        json.dumps({
            "date": "2026-04-06",
            "captured_at": "2026-04-06T14:00:00+00:00",
            "run_time": "2026-04-06T14:00:00+00:00",
            "primary": {"batter_id": 6, "batter_name": "P", "p_game_hit": 0.755},
            "double_down": {"batter_id": 7, "batter_name": "D", "p_game_hit": 0.745},
        }) + "\n"
    )


def _write_profiles(profiles_dir):
    profiles_dir.mkdir()
    rows = []
    for i in range(20):
        rows.append({
            "date": pd.Timestamp("2021-04-01") + pd.Timedelta(days=i),
            "rank": 1,
            "p_game_hit": 0.80 + i * 0.001,
        })
        rows.append({
            "date": pd.Timestamp("2021-04-01") + pd.Timedelta(days=i),
            "rank": 2,
            "p_game_hit": 0.78 + i * 0.001,
        })
    pd.DataFrame(rows).to_parquet(profiles_dir / "backtest_2021.parquet", index=False)


def test_build_inventory_segments_production_windows_and_keeps_scope_evidence_only(tmp_path):
    picks_dir = tmp_path / "picks"
    profiles_dir = tmp_path / "profiles"
    _write_picks(picks_dir)
    _write_profiles(profiles_dir)

    result = build_inventory(
        picks_dir=picks_dir,
        historical_profiles_dir=profiles_dir,
        direct_min_rank1=5,
        direct_holdout_rank1=3,
        reconcile_min_rank1=4,
    )

    assert result["production_deploy_claim"] is False
    assert result["writes_policy_artifact"] is False
    assert result["derives_boundaries"] is False
    assert result["builds_reconciliation_map"] is False
    assert result["surface_inventory"]["pick_json"]["rank1_n"] == 7
    assert result["surface_inventory"]["pick_json"]["rank2_n"] == 7
    assert result["surface_inventory"]["lineup_evolution"]["rank1_n"] == 1
    assert (
        result["surface_inventory"]["pick_json"]["feature_env_hash_coverage"]["rank1_present"]
        == 5
    )
    assert result["windows"]["best_policy_hash_window"]["rank1_n"] == 4
    assert result["windows"]["best_strict_model_git_policy_window"]["rank1_n"] == 2
    assert result["windows"]["best_non_null_complete_scale_window"]["rank1_n"] == 3
    assert result["scale_parity"]["all_pick_json_rank1_vs_historical_estimated_pa"]["available"] is True
    assert result["feasibility"]["decision"] == (
        "DIRECT_NOT_FEASIBLE_RECONCILIATION_CANDIDATE_REQUIRES_PREREG"
    )


def test_build_inventory_without_historical_profiles_keeps_scale_parity_unavailable(tmp_path):
    picks_dir = tmp_path / "picks"
    _write_picks(picks_dir)

    result = build_inventory(picks_dir=picks_dir)

    parity = result["scale_parity"]["all_pick_json_rank1_vs_historical_estimated_pa"]
    assert parity["available"] is False
    assert result["feasibility"]["decision"] == (
        "NOT_FEASIBLE_DIRECT_OR_RECONCILIATION_NEEDS_MORE_LIVE_N"
    )
