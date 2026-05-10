from __future__ import annotations

import json

import pandas as pd

from bts.experiment.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PRODUCTION_PICK_SNAPSHOT_VERSION,
    PROFILE_SCHEMA_COLUMNS,
)
from scripts.live_forward_provenance_inventory import build_inventory


def _profile_row(*, variant: str, run_kind: str, rank: int, actual_hit=None, n_pas=None):
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_kind": run_kind,
        "variant": variant,
        "model_name": "decision_weighted_lgbm_v0" if variant == "candidate" else "production",
        "generated_at": "2026-05-10T14:00:00+00:00",
        "git_commit": "5004b1c8b093da0f8acb11bd728430ebacbf92d3",
        "date": "2026-05-10",
        "season": 2026,
        "rank": rank,
        "batter_id": 1000 + rank,
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
            _profile_row(variant="production", run_kind="live_forward_preoutcome", rank=1),
            _profile_row(variant="production", run_kind="live_forward_preoutcome", rank=2),
        ],
    )
    _write_profiles(
        cand,
        [
            _profile_row(variant="candidate", run_kind="live_forward_preoutcome", rank=1),
            _profile_row(variant="candidate", run_kind="live_forward_preoutcome", rank=2),
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
        "environment": {
            "BTS_LGBM_RANDOM_STATE": "42",
            "BTS_LGBM_DETERMINISTIC": "1",
        },
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


def _write_resolved_artifact(root):
    date_dir = root / "2026-05-10"
    prod = date_dir / "profiles" / "production" / "live_2026-05-10.parquet"
    cand = date_dir / "profiles" / "candidate" / "live_2026-05-10.parquet"
    _write_profiles(
        prod,
        [
            _profile_row(
                variant="production",
                run_kind="live_forward_resolved",
                rank=1,
                actual_hit=True,
                n_pas=4,
            ),
            _profile_row(
                variant="production",
                run_kind="live_forward_resolved",
                rank=2,
                actual_hit=False,
                n_pas=4,
            ),
        ],
    )
    _write_profiles(
        cand,
        [
            _profile_row(
                variant="candidate",
                run_kind="live_forward_resolved",
                rank=1,
                actual_hit=True,
                n_pas=4,
            ),
            _profile_row(
                variant="candidate",
                run_kind="live_forward_resolved",
                rank=2,
                actual_hit=True,
                n_pas=4,
            ),
        ],
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": "2026-05-11T03:00:00+00:00",
        "git_commit": "5004b1c8b093da0f8acb11bd728430ebacbf92d3",
        "run_kind": "live_forward_resolved",
        "source_run_kind": "live_forward_preoutcome",
        "production_deploy_claim": False,
        "fresh_target_claim": True,
        "candidate_name": "decision_weighted_lgbm_v0",
        "baseline_name": "production",
        "date": "2026-05-10",
        "dates": ["2026-05-10"],
        "top_n": 2,
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
    (date_dir / "resolution.json").write_text(json.dumps({"complete": True}))


def test_inventory_marks_verified_snapshot_artifact_ready_and_joinable(tmp_path):
    artifact_root = tmp_path / "live"
    resolved_root = tmp_path / "resolved"
    _write_artifact(artifact_root)
    _write_resolved_artifact(resolved_root)

    report = build_inventory(
        artifact_root=artifact_root,
        output_path=tmp_path / "inventory.json",
        rows_output_path=tmp_path / "inventory.rows.parquet",
        resolved_root=resolved_root,
        expected_candidate="decision_weighted_lgbm_v0",
        expected_top_n=2,
        require_production_pick_snapshot=True,
        generated_at="2026-05-10T15:00:00+00:00",
    )

    assert report["schema_version"] == "live_forward_provenance_inventory_v1"
    assert report["research_only"] is True
    assert report["production_deploy_claim"] is False
    assert report["summary"]["artifact_count"] == 1
    assert report["summary"]["official_fresh_target_ready_count"] == 1
    assert report["summary"]["at_lock_ranked_surface_joinable_count"] == 1
    assert report["summary"]["resolved_outcome_joinable_count"] == 1

    row = report["rows"][0]
    assert row["official_fresh_target_ready"] is True
    assert row["at_lock_ranked_surface_joinable"] is True
    assert row["resolved_outcome_joinable"] is True
    assert row["production_pick_snapshot"]["version_ok"] is True
    assert row["variants"]["production"]["actual_hit_null_rows"] == 2
    assert row["resolved"]["variants"]["production"]["actual_hit_null_rows"] == 0
    assert (tmp_path / "inventory.rows.parquet").exists()


def test_inventory_requires_snapshot_for_official_ready_flag(tmp_path):
    artifact_root = tmp_path / "live"
    _write_artifact(artifact_root, with_snapshot=False)

    report = build_inventory(
        artifact_root=artifact_root,
        output_path=tmp_path / "inventory.json",
        rows_output_path=None,
        resolved_root=None,
        expected_candidate="decision_weighted_lgbm_v0",
        expected_top_n=2,
        require_production_pick_snapshot=True,
        generated_at="2026-05-10T15:00:00+00:00",
    )

    row = report["rows"][0]
    assert row["at_lock_ranked_surface_joinable"] is True
    assert row["official_fresh_target_ready"] is False
    assert report["summary"]["missing_production_pick_snapshot_count"] == 1


def test_inventory_requires_passing_verification_for_official_ready_flag(tmp_path):
    artifact_root = tmp_path / "live"
    date_dir = _write_artifact(artifact_root, with_snapshot=True)
    (date_dir / "verification.json").unlink()

    report = build_inventory(
        artifact_root=artifact_root,
        output_path=tmp_path / "inventory.json",
        rows_output_path=None,
        resolved_root=None,
        expected_candidate="decision_weighted_lgbm_v0",
        expected_top_n=2,
        require_production_pick_snapshot=True,
        generated_at="2026-05-10T15:00:00+00:00",
    )

    row = report["rows"][0]
    assert row["at_lock_ranked_surface_joinable"] is True
    assert row["production_pick_snapshot"]["present"] is True
    assert row["verification"]["present"] is False
    assert row["official_fresh_target_ready"] is False
    assert row["requires_verifier_before_official_use"] is True
    assert report["summary"]["missing_verification_count"] == 1


def test_inventory_handles_missing_root(tmp_path):
    report = build_inventory(
        artifact_root=tmp_path / "missing",
        output_path=tmp_path / "inventory.json",
        rows_output_path=None,
        resolved_root=None,
        expected_candidate="decision_weighted_lgbm_v0",
        expected_top_n=10,
        require_production_pick_snapshot=True,
        generated_at="2026-05-10T15:00:00+00:00",
    )

    assert report["summary"]["artifact_count"] == 0
    assert report["rows"] == []
