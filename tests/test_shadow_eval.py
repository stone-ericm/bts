"""Tests for shadow backfill manifest and quality helpers."""

from __future__ import annotations

import json

from click.testing import CliRunner
from unittest.mock import patch

from bts.cli import cli
from bts.picks import DailyPick, Pick, load_shadow_pick, save_pick, save_shadow_pick
from bts.shadow_eval import (
    apply_shadow_backfill_manifest,
    build_shadow_backfill_manifest,
    build_shadow_cycle_status,
)


def _pick(name: str, batter_id: int, *, game_pk: int | None = None) -> Pick:
    return Pick(
        batter_name=name,
        batter_id=batter_id,
        team="BOS",
        lineup_position=1,
        pitcher_name="Pitcher",
        pitcher_id=200,
        p_game_hit=0.75,
        flags=[],
        projected_lineup=False,
        game_pk=game_pk or (800000 + batter_id),
        game_time="2026-04-01T23:00:00Z",
        pitcher_team="NYY",
    )


def _daily(date: str, pick: Pick, *, double_down: Pick | None = None,
           result: str | None = None) -> DailyPick:
    return DailyPick(
        date=date,
        run_time=f"{date}T15:00:00+00:00",
        pick=pick,
        double_down=double_down,
        runner_up=None,
        bluesky_posted=False,
        bluesky_uri=None,
        result=result,
    )


def _hit_checker(results: dict[int, bool | str | None]):
    def check(game_pk, batter_id, batter_name, date, team):
        return results[batter_id]
    return check


def test_build_shadow_backfill_manifest_recomputes_all_results(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    save_pick(_daily("2026-04-01", _pick("Prod Hit", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily(
        "2026-04-01",
        _pick("Shadow Hit", 2),
        double_down=_pick("Shadow Miss", 3),
        result=None,
    ), picks_dir)

    save_pick(_daily("2026-04-02", _pick("Prod Miss", 4), result="miss"), picks_dir)
    save_shadow_pick(_daily("2026-04-02", _pick("Legacy Hit", 5), result="hit"), picks_dir)

    manifest = build_shadow_backfill_manifest(
        picks_dir,
        n_bootstrap=0,
        hit_checker=_hit_checker({1: True, 2: True, 3: False, 4: False, 5: False}),
    )

    assert manifest["counts"]["shadow_files"] == 2
    assert manifest["counts"]["resolved"] == 2
    assert manifest["counts"]["would_change"] == 2
    assert manifest["counts"]["change_class"]["new"] == 1
    assert manifest["counts"]["change_class"]["changed"] == 1
    rows = {row["date"]: row for row in manifest["rows"]}
    assert rows["2026-04-01"]["new_shadow_result"] == "miss"
    assert rows["2026-04-01"]["change_class"] == "new"
    assert rows["2026-04-02"]["old_shadow_result"] == "hit"
    assert rows["2026-04-02"]["new_shadow_result"] == "miss"
    assert rows["2026-04-02"]["change_class"] == "changed"
    assert manifest["quality_if_applied"]["production_day_hit_rate"]["hits"] == 1
    assert manifest["quality_if_applied"]["shadow_day_hit_rate"]["hits"] == 0
    assert manifest["quality_if_applied"]["paired_outcomes"]["production_only_hit"] == 1


def test_shadow_backfill_voids_primary_and_scores_double_only(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    save_pick(_daily("2026-04-01", _pick("Prod Hit", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily(
        "2026-04-01",
        _pick("Shadow Void", 2),
        double_down=_pick("Shadow Active", 3),
        result=None,
    ), picks_dir)

    manifest = build_shadow_backfill_manifest(
        picks_dir,
        n_bootstrap=0,
        hit_checker=_hit_checker({1: True, 2: "void", 3: True}),
    )

    row = manifest["rows"][0]
    assert row["new_shadow_result"] == "hit"
    assert row["shadow"]["slot_results"] == {"pick": "void", "double_down": "hit"}
    assert row["shadow"]["slots"][0]["slot_result"] == "void"
    assert row["shadow"]["slots"][1]["slot_result"] == "hit"
    assert manifest["counts"]["void"] == 0


def test_shadow_backfill_both_void_is_resolved_void(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    save_pick(_daily("2026-04-01", _pick("Prod Void", 1), result="void"), picks_dir)
    save_shadow_pick(_daily(
        "2026-04-01",
        _pick("Shadow Void", 2),
        double_down=_pick("Shadow Void 2", 3),
        result=None,
    ), picks_dir)

    manifest = build_shadow_backfill_manifest(
        picks_dir,
        n_bootstrap=0,
        hit_checker=_hit_checker({1: "void", 2: "void", 3: "void"}),
    )

    row = manifest["rows"][0]
    assert row["shadow"]["status"] == "resolved"
    assert row["new_shadow_result"] == "void"
    assert row["change_class"] == "new"
    assert manifest["counts"]["void"] == 1
    assert manifest["quality_if_applied"]["n_evaluable_days"] == 0
    assert manifest["quality_if_applied"]["outcome_counts"]["shadow"]["void"] == 1


def test_apply_shadow_backfill_manifest_preserves_backup(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod Hit", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Shadow Miss", 2), result="hit"), picks_dir)

    manifest = build_shadow_backfill_manifest(
        picks_dir,
        n_bootstrap=0,
        hit_checker=_hit_checker({1: True, 2: False}),
    )
    result = apply_shadow_backfill_manifest(manifest, backup_dir=tmp_path / "backups")

    assert result["applied"][0]["date"] == "2026-04-01"
    assert (tmp_path / "backups" / "2026-04-01.shadow.json").exists()
    assert load_shadow_pick("2026-04-01", picks_dir).result == "miss"


def test_apply_shadow_backfill_manifest_writes_slot_results(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod Hit", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily(
        "2026-04-01",
        _pick("Shadow Void", 2),
        double_down=_pick("Shadow Active", 3),
        result=None,
    ), picks_dir)

    manifest = build_shadow_backfill_manifest(
        picks_dir,
        n_bootstrap=0,
        hit_checker=_hit_checker({1: True, 2: "void", 3: True}),
    )
    apply_shadow_backfill_manifest(manifest, backup_dir=tmp_path / "backups")

    loaded = load_shadow_pick("2026-04-01", picks_dir)
    assert loaded.result == "hit"
    assert loaded.slot_results == {"pick": "void", "double_down": "hit"}



def test_build_shadow_backfill_manifest_is_idempotent(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod Hit", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Shadow Hit", 2), result=None), picks_dir)

    kwargs = dict(
        n_bootstrap=0,
        hit_checker=_hit_checker({1: True, 2: True}),
    )
    first = build_shadow_backfill_manifest(picks_dir, **kwargs)
    second = build_shadow_backfill_manifest(picks_dir, **kwargs)
    first["generated_at"] = None
    second["generated_at"] = None

    assert first == second


def test_build_shadow_backfill_manifest_prefers_cached_game_json(tmp_path):
    picks_dir = tmp_path / "picks"
    raw_dir = tmp_path / "raw"
    game_pk = 800002
    (raw_dir / "2026").mkdir(parents=True)
    (raw_dir / "2026" / f"{game_pk}.json").write_text(json.dumps({
        "gameData": {"status": {"abstractGameCode": "F"}},
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {"ID2": {"stats": {"batting": {"hits": 1}}}}},
                    "home": {"players": {}},
                }
            }
        },
    }))
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod Hit", 2, game_pk=game_pk), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Shadow Hit", 2, game_pk=game_pk), result=None), picks_dir)

    manifest = build_shadow_backfill_manifest(picks_dir, raw_dir=raw_dir, n_bootstrap=0)
    row = manifest["rows"][0]

    assert row["shadow"]["slots"][0]["data_source"] == "cached_game_json"
    assert row["shadow"]["slots"][0]["hit"] is True
    assert row["api_calls"] == []


def test_shadow_backfill_results_cli_is_dry_run_by_default(tmp_path):
    picks_dir = tmp_path / "picks"
    output = tmp_path / "manifest.json"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod Hit", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Shadow Miss", 2), result="hit"), picks_dir)

    with patch(
        "bts.shadow_eval.resolve_pick_slot_result",
        side_effect=lambda pick, date: "hit" if pick.batter_id == 1 else "miss",
    ):
        result = CliRunner().invoke(cli, [
            "shadow-backfill-results",
            "--picks-dir", str(picks_dir),
            "--output", str(output),
            "--bootstrap", "0",
        ])

    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert "Change classes: new=0, changed=1" in result.output
    assert "Changed dates: 2026-04-01" in result.output
    assert "No files changed" in result.output
    assert output.exists()
    assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"


def test_apply_shadow_backfill_manifest_skips_sha_mismatch(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    shadow_file = picks_dir / "2026-04-01.shadow.json"
    save_pick(_daily("2026-04-01", _pick("Prod Hit", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Shadow Miss", 2), result="hit"), picks_dir)

    manifest = build_shadow_backfill_manifest(
        picks_dir,
        n_bootstrap=0,
        hit_checker=_hit_checker({1: True, 2: False}),
    )
    shadow_data = json.loads(shadow_file.read_text())
    shadow_data["manual_note"] = "changed after manifest"
    shadow_file.write_text(json.dumps(shadow_data, indent=2))

    result = apply_shadow_backfill_manifest(manifest, backup_dir=tmp_path / "backups")

    assert result["applied"] == []
    assert result["skipped"] == [{"date": "2026-04-01", "reason": "sha_changed"}]
    current_data = json.loads(shadow_file.read_text())
    assert current_data["result"] == "hit"
    assert current_data["manual_note"] == "changed after manifest"


def test_build_shadow_backfill_manifest_reports_production_mismatch(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod Miss", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Shadow Miss", 2), result=None), picks_dir)

    manifest = build_shadow_backfill_manifest(
        picks_dir,
        n_bootstrap=0,
        hit_checker=_hit_checker({1: False, 2: False}),
    )

    assert manifest["quality_if_applied"]["production_recorded_mismatches"] == [{
        "date": "2026-04-01",
        "recorded_result": "hit",
        "evaluated_result": "miss",
    }]


def test_build_shadow_cycle_status_tracks_recorded_monitoring_state(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Same", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Same", 1), result="hit"), picks_dir)
    save_pick(_daily("2026-04-02", _pick("Prod", 2), result="miss"), picks_dir)
    save_shadow_pick(_daily("2026-04-02", _pick("Shadow", 3), result=None), picks_dir)

    status = build_shadow_cycle_status(
        picks_dir,
        min_days=2,
        generated_at="2026-05-09T00:00:00+00:00",
        git_commit="test-sha",
    )

    assert status["schema_version"] == "bts_shadow_cycle_status_v1"
    assert status["model"]["name"] == "context_stack_shadow_v1"
    assert status["model"]["production_deploy_claim"] is False
    assert status["cycle_state"] == "needs_result_reconciliation"
    assert status["counts"]["shadow_files"] == 2
    assert status["counts"]["resolved_paired_days"] == 1
    assert status["counts"]["unresolved_shadow_results"] == 1
    assert status["coverage"]["unresolved_shadow_dates"] == ["2026-04-02"]
    assert status["quality_recorded"]["production_day_hit_rate"]["hits"] == 1
    assert status["quality_recorded"]["shadow_day_hit_rate"]["hits"] == 1
    assert "separate pre-registration" in status["methodology_note"]
    assert "Semantic local version" in status["model"]["versioning_policy"]
    assert "single latest-state artifact" in status["history_policy"]


def test_build_shadow_cycle_status_counts_void_as_resolved_but_not_evaluable(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod", 1), result="hit"), picks_dir)
    shadow = _daily("2026-04-01", _pick("Shadow", 2), result="void")
    shadow.slot_results = {"pick": "void"}
    save_shadow_pick(shadow, picks_dir)

    status = build_shadow_cycle_status(
        picks_dir,
        min_days=1,
        generated_at="2026-05-09T00:00:00+00:00",
        git_commit="test-sha",
    )

    assert status["cycle_state"] == "collecting_live_forward"
    assert status["counts"]["resolved_shadow_results"] == 1
    assert status["counts"]["unresolved_shadow_results"] == 0
    assert status["counts"]["void_shadow_results"] == 1
    assert status["counts"]["resolved_paired_days"] == 0
    assert status["counts"]["resolved_or_void_paired_days"] == 1
    assert status["coverage"]["unresolved_shadow_dates"] == []
    assert status["quality_recorded"]["outcome_counts"]["shadow"]["void"] == 1


def test_build_shadow_cycle_status_no_shadow_files(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    status = build_shadow_cycle_status(
        picks_dir,
        generated_at="2026-05-09T00:00:00+00:00",
        git_commit="test-sha",
    )

    assert status["cycle_state"] == "no_shadow_files"
    assert status["counts"]["shadow_files"] == 0
    assert status["counts"]["resolved_paired_days"] == 0
    assert "Verify scheduler.shadow_model=true" in status["action_items"][0]


def test_shadow_status_cli_writes_status_artifact(tmp_path):
    picks_dir = tmp_path / "picks"
    output = tmp_path / "status.json"
    picks_dir.mkdir()
    save_pick(_daily("2026-04-01", _pick("Prod", 1), result="hit"), picks_dir)
    save_shadow_pick(_daily("2026-04-01", _pick("Shadow", 2), result="miss"), picks_dir)

    result = CliRunner().invoke(cli, [
        "shadow-status",
        "--picks-dir", str(picks_dir),
        "--output", str(output),
        "--min-days", "1",
    ])

    assert result.exit_code == 0
    assert "Shadow cycle status: ready_for_manual_review" in result.output
    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "bts_shadow_cycle_status_v1"
    assert payload["counts"]["resolved_paired_days"] == 1


def test_shadow_backfill_results_apply_requires_backup_dir(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    result = CliRunner().invoke(cli, [
        "shadow-backfill-results",
        "--picks-dir", str(picks_dir),
        "--apply",
    ])
    assert result.exit_code != 0
    assert "--backup-dir is required with --apply" in result.output
