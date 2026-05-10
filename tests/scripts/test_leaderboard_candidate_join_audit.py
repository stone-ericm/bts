from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.leaderboard_candidate_join_audit import build_audit
from scripts.leaderboard_candidate_join_audit import load_user_picks


def _write_user_picks(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _pick(
    *,
    captured_at: str,
    pick_number: int,
    batter_id: int,
    batter_name: str,
    result: str,
):
    return {
        "captured_at": pd.Timestamp(captured_at),
        "round_id": 1,
        "pick_date": pd.Timestamp("2026-05-10").date(),
        "pick_number": pick_number,
        "unit_id": 1,
        "bts_player_id": batter_id,
        "result": result,
        "at_bats": 4,
        "hits": 1 if result == "hit" else 0,
        "streak_after": 1,
        "batter_id": batter_id,
        "batter_name": batter_name,
        "batter_team": "BOS",
        "opponent_team": "NYY",
        "home_or_away": "home",
    }


def _write_snapshot(path, users):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "captured_at": pd.Timestamp("2026-05-10T09:00:00"),
            "tab": "active_streak",
            "rank": i + 1,
            "username": user,
            "streak": 10 - i,
            "hits_today": 0,
        }
        for i, user in enumerate(users)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_candidate_artifact(root):
    prod = pd.DataFrame([
        {
            "artifact_schema_version": "bts_candidate_ranked_slate_pair_v1",
            "run_kind": "live_forward_resolved",
            "variant": "production",
            "model_name": "production_lgbm_v0",
            "generated_at": "2026-05-10T10:00:00+00:00",
            "git_commit": "abc",
            "date": "2026-05-10",
            "season": 2026,
            "rank": 1,
            "batter_id": 11,
            "game_pk": 1001,
            "p_game_hit": 0.70,
            "actual_hit": 0,
            "n_pas": 4,
        },
        {
            "artifact_schema_version": "bts_candidate_ranked_slate_pair_v1",
            "run_kind": "live_forward_resolved",
            "variant": "production",
            "model_name": "production_lgbm_v0",
            "generated_at": "2026-05-10T10:00:00+00:00",
            "git_commit": "abc",
            "date": "2026-05-10",
            "season": 2026,
            "rank": 2,
            "batter_id": 20,
            "game_pk": 1002,
            "p_game_hit": 0.69,
            "actual_hit": 1,
            "n_pas": 4,
        },
    ])
    cand = prod.copy()
    cand["variant"] = "candidate"
    cand["model_name"] = "candidate_v0"
    cand.loc[0, "batter_id"] = 10
    cand.loc[0, "p_game_hit"] = 0.75

    prod_path = root / "profiles" / "production" / "live_2026-05-10.parquet"
    cand_path = root / "profiles" / "candidate" / "live_2026-05-10.parquet"
    prod_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    prod.to_parquet(prod_path, index=False)
    cand.to_parquet(cand_path, index=False)
    manifest = {
        "schema_version": "bts_candidate_ranked_slate_pair_v1",
        "run_kind": "live_forward_resolved",
        "candidate_name": "candidate_v0",
        "baseline_name": "production_lgbm_v0",
        "dates": ["2026-05-10"],
        "top_n": 2,
        "generated_at": "2026-05-10T10:00:00+00:00",
        "profile_paths": {
            "production": {"2026-05-10": "profiles/production/live_2026-05-10.parquet"},
            "candidate": {"2026-05-10": "profiles/candidate/live_2026-05-10.parquet"},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def _fixture_tree(tmp_path):
    leaderboard = tmp_path / "leaderboard"
    _write_user_picks(
        leaderboard / "user_picks" / "alice.parquet",
        [
            _pick(
                captured_at="2026-05-10T10:00:00",
                pick_number=1,
                batter_id=10,
                batter_name="Consensus One",
                result="hit",
            ),
            _pick(
                captured_at="2026-05-10T10:00:00",
                pick_number=2,
                batter_id=20,
                batter_name="Tie Low Id",
                result="not_hit",
            ),
        ],
    )
    _write_user_picks(
        leaderboard / "user_picks" / "bob.parquet",
        [
            _pick(
                captured_at="2026-05-10T10:05:00",
                pick_number=1,
                batter_id=10,
                batter_name="Consensus One",
                result="hit",
            ),
            _pick(
                captured_at="2026-05-10T10:05:00",
                pick_number=2,
                batter_id=30,
                batter_name="Tie High Id",
                result="hit",
            ),
        ],
    )
    _write_user_picks(
        leaderboard / "user_picks" / "eve.parquet",
        [
            _pick(
                captured_at="2026-05-10T12:30:00",
                pick_number=1,
                batter_id=99,
                batter_name="Late Leak",
                result="hit",
            ),
        ],
    )
    _write_snapshot(leaderboard / "leaderboard_snapshots" / "2026-05-10.parquet", ["alice", "bob"])

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _write_candidate_artifact(artifact)
    return leaderboard, artifact


def test_build_audit_filters_to_pre_lock_cutoff_and_joins_candidate_features(tmp_path):
    leaderboard, artifact = _fixture_tree(tmp_path)
    output = tmp_path / "leaderboard_clue_audit_2026-05-10.json"

    report = build_audit(
        leaderboard_dir=leaderboard,
        artifact_dir=artifact,
        output_path=output,
        joined_output_path=None,
        decision_cutoff_iso="2026-05-10T11:00:00Z",
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates={"2026-05-10"},
        n_bootstrap=0,
        expected_block_length=7,
        seed=7,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    assert output.exists()
    assert report["pre_lock_visibility_claim"] is True
    assert report["inventory"]["leaderboard"]["raw_rows"] == 5
    assert report["inventory"]["leaderboard"]["cutoff_rows"] == 4
    assert report["cohorts"]["fixed_cohort"]["n_users"] == 2
    assert report["comparison"]["fixed_cohort"]["n_units"] == 2
    assert report["comparison"]["fixed_cohort"]["n_disagreements"] == 1
    assert report["comparison"]["fixed_cohort"]["disagreement_mean_delta"] == pytest.approx(1.0)

    joined = pd.read_parquet(report["joined_profiles_path"])
    candidate_consensus = joined[
        (joined["variant"] == "candidate")
        & (joined["batter_id"] == 10)
    ].iloc[0]
    assert candidate_consensus["lb_fixed_cohort_slot1_public_pick_share"] == pytest.approx(1.0)
    assert candidate_consensus["lb_fixed_cohort_slot1_is_public_consensus"] == 1
    assert pd.isna(candidate_consensus["lb_fixed_cohort_slot2_public_pick_share"])
    assert 99 not in set(joined["batter_id"])


def test_build_audit_can_use_explicit_fixed_cohort_json(tmp_path):
    leaderboard, artifact = _fixture_tree(tmp_path)
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps({"users": ["alice"]}))

    report = build_audit(
        leaderboard_dir=leaderboard,
        artifact_dir=artifact,
        output_path=tmp_path / "audit.json",
        joined_output_path=tmp_path / "joined.parquet",
        decision_cutoff_iso="2026-05-10T11:00:00Z",
        cohort_as_of_iso=None,
        cohort_users_json=cohort_path,
        dates={"2026-05-10"},
        n_bootstrap=0,
        expected_block_length=7,
        seed=7,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    assert report["cohorts"]["fixed_cohort"]["source"] == "cohort_users_json"
    assert report["cohorts"]["fixed_cohort"]["n_users"] == 1
    joined = pd.read_parquet(report["joined_profiles_path"])
    prod_rank2 = joined[
        (joined["variant"] == "production")
        & (joined["rank"] == 2)
    ].iloc[0]
    assert prod_rank2["lb_fixed_cohort_slot2_public_pick_share"] == pytest.approx(1.0)
    assert prod_rank2["lb_fixed_cohort_slot2_is_public_consensus"] == 1


def test_load_user_picks_fails_loud_on_missing_required_columns(tmp_path):
    leaderboard = tmp_path / "leaderboard"
    broken = leaderboard / "user_picks" / "broken.parquet"
    broken.parent.mkdir(parents=True)
    pd.DataFrame([
        {
            "captured_at": pd.Timestamp("2026-05-10T10:00:00"),
            "pick_date": pd.Timestamp("2026-05-10").date(),
            "pick_number": 1,
            "result": "hit",
        }
    ]).to_parquet(broken, index=False)

    with pytest.raises(ValueError, match="missing leaderboard user-pick columns"):
        load_user_picks(leaderboard, decision_cutoff=None)
