from __future__ import annotations

import json

import pandas as pd
import pytest

from bts.experiment.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PRODUCTION_PICK_SNAPSHOT_VERSION,
    PROFILE_SCHEMA_COLUMNS,
)
from scripts.leaderboard_backfilled_model_audit import load_ranked_surfaces
from scripts.live_forward_surface_export import build_surface


def _profile_row(*, variant: str, rank: int, actual_hit=None, n_pas=None):
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_kind": "live_forward_preoutcome",
        "variant": variant,
        "model_name": "decision_weighted_lgbm_v0" if variant == "candidate" else "production",
        "generated_at": "2026-05-10T14:00:00+00:00",
        "git_commit": "5004b1c8b093da0f8acb11bd728430ebacbf92d3",
        "date": "2026-05-10",
        "season": 2026,
        "rank": rank,
        "batter_id": 1000 + rank + (100 if variant == "candidate" else 0),
        "game_pk": 2000 + rank,
        "p_game_hit": 0.80 - rank * 0.01,
        "actual_hit": actual_hit,
        "n_pas": n_pas,
    }


def _write_profiles(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PROFILE_SCHEMA_COLUMNS).to_parquet(path, index=False)


def _snapshot():
    return {
        "snapshot_version": PRODUCTION_PICK_SNAPSHOT_VERSION,
        "date": "2026-05-10",
        "source_sha256": "abc123",
        "production_pick_json": {"pick": {"batter_id": 1001, "game_pk": 2001}},
        "slots": {
            "pick": {"batter_id": 1001, "game_pk": 2001},
            "double_down": {"batter_id": 1002, "game_pk": 2002},
        },
    }


def _write_artifact(root, *, with_snapshot: bool = True):
    date_dir = root / "2026-05-10"
    prod = date_dir / "profiles" / "production" / "live_2026-05-10.parquet"
    cand = date_dir / "profiles" / "candidate" / "live_2026-05-10.parquet"
    _write_profiles(
        prod,
        [
            _profile_row(variant="production", rank=1),
            _profile_row(variant="production", rank=2),
        ],
    )
    _write_profiles(
        cand,
        [
            _profile_row(variant="candidate", rank=1),
            _profile_row(variant="candidate", rank=2),
        ],
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": "2026-05-10T14:00:00+00:00",
        "git_commit": "5004b1c8b093da0f8acb11bd728430ebacbf92d3",
        "run_kind": "live_forward_preoutcome",
        "production_deploy_claim": False,
        "fresh_target_claim": True,
        "candidate_name": "decision_weighted_lgbm_v0",
        "baseline_name": "production",
        "date": "2026-05-10",
        "dates": ["2026-05-10"],
        "top_n": 2,
        "production_pick_snapshot": _snapshot() if with_snapshot else None,
        "profile_paths": {
            "production": {"2026-05-10": "profiles/production/live_2026-05-10.parquet"},
            "candidate": {"2026-05-10": "profiles/candidate/live_2026-05-10.parquet"},
        },
        "row_counts": {
            "production": {"2026-05-10": 2},
            "candidate": {"2026-05-10": 2},
        },
    }
    (date_dir / "manifest.json").write_text(json.dumps(manifest))
    (date_dir / "verification.json").write_text(json.dumps({
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "ok": True,
        "failure_count": 0,
    }))
    return date_dir


def test_exports_production_surface_in_mechanism_contract(tmp_path):
    artifact_root = tmp_path / "live"
    _write_artifact(artifact_root, with_snapshot=True)

    report = build_surface(
        artifact_root=artifact_root,
        variant="production",
        require_official_ready=True,
        output_path=tmp_path / "surface.parquet",
        manifest_output_path=tmp_path / "surface.json",
        resolved_root=None,
        expected_candidate="decision_weighted_lgbm_v0",
        expected_top_n=2,
        require_production_pick_snapshot=True,
        dates=None,
        min_date=None,
        max_date=None,
        generated_at="2026-05-10T15:00:00+00:00",
    )

    assert report["schema_version"] == "live_forward_ranked_surface_export_v1"
    assert report["rows"] == 2
    assert report["dates"] == 1
    assert report["require_official_ready"] is True
    assert report["contract_check_inventory"]["rows"] == 2

    surface = pd.read_parquet(tmp_path / "surface.parquet")
    assert ["date", "rank", "batter_id", "p_game_hit", "actual_hit"] == [
        col for col in ["date", "rank", "batter_id", "p_game_hit", "actual_hit"]
        if col in surface.columns
    ]
    assert surface["surface_variant"].unique().tolist() == ["production"]
    assert surface["official_fresh_target_ready"].unique().tolist() == [True]

    ranked, joinable, inventory = load_ranked_surfaces(
        {"live_forward_production": tmp_path / "surface.parquet"}
    )
    assert len(ranked) == 2
    assert len(joinable) == 2
    assert inventory["live_forward_production"]["max_rank"] == 2


def test_exports_at_lock_joinable_surface_without_official_snapshot(tmp_path):
    artifact_root = tmp_path / "live"
    _write_artifact(artifact_root, with_snapshot=False)

    report = build_surface(
        artifact_root=artifact_root,
        variant="production",
        require_official_ready=False,
        output_path=tmp_path / "surface.parquet",
        manifest_output_path=None,
        resolved_root=None,
        expected_candidate="decision_weighted_lgbm_v0",
        expected_top_n=2,
        require_production_pick_snapshot=True,
        dates=None,
        min_date=None,
        max_date=None,
        generated_at="2026-05-10T15:00:00+00:00",
    )

    assert report["rows"] == 2
    surface = pd.read_parquet(tmp_path / "surface.parquet")
    assert surface["at_lock_ranked_surface_joinable"].unique().tolist() == [True]
    assert surface["official_fresh_target_ready"].unique().tolist() == [False]


def test_require_official_ready_filters_missing_snapshot(tmp_path):
    artifact_root = tmp_path / "live"
    _write_artifact(artifact_root, with_snapshot=False)

    with pytest.raises(ValueError, match="no eligible live-forward profile rows"):
        build_surface(
            artifact_root=artifact_root,
            variant="production",
            require_official_ready=True,
            output_path=tmp_path / "surface.parquet",
            manifest_output_path=None,
            resolved_root=None,
            expected_candidate="decision_weighted_lgbm_v0",
            expected_top_n=2,
            require_production_pick_snapshot=True,
            dates=None,
            min_date=None,
            max_date=None,
            generated_at="2026-05-10T15:00:00+00:00",
        )


def test_require_official_ready_filters_snapshot_mismatch(tmp_path):
    artifact_root = tmp_path / "live"
    picks_dir = tmp_path / "data" / "picks"
    _write_artifact(artifact_root, with_snapshot=True)
    picks_dir.mkdir(parents=True)
    (picks_dir / "2026-05-10.json").write_text(json.dumps({
        "date": "2026-05-10",
        "result": "miss",
        "pick": {"batter_id": 9999},
    }))

    with pytest.raises(ValueError, match="no eligible live-forward profile rows"):
        build_surface(
            artifact_root=artifact_root,
            variant="production",
            require_official_ready=True,
            output_path=tmp_path / "surface.parquet",
            manifest_output_path=None,
            resolved_root=None,
            picks_dir=picks_dir,
            expected_candidate="decision_weighted_lgbm_v0",
            expected_top_n=2,
            require_production_pick_snapshot=True,
            dates=None,
            min_date=None,
            max_date=None,
            generated_at="2026-05-10T15:00:00+00:00",
        )


def test_exports_candidate_variant(tmp_path):
    artifact_root = tmp_path / "live"
    _write_artifact(artifact_root, with_snapshot=True)

    build_surface(
        artifact_root=artifact_root,
        variant="candidate",
        require_official_ready=True,
        output_path=tmp_path / "candidate.parquet",
        manifest_output_path=None,
        resolved_root=None,
        expected_candidate="decision_weighted_lgbm_v0",
        expected_top_n=2,
        require_production_pick_snapshot=True,
        dates=None,
        min_date=None,
        max_date=None,
        generated_at="2026-05-10T15:00:00+00:00",
    )

    surface = pd.read_parquet(tmp_path / "candidate.parquet")
    assert surface["surface_variant"].unique().tolist() == ["candidate"]
    assert surface.sort_values("rank")["batter_id"].tolist() == [1101, 1102]
