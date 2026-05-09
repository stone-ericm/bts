"""Tests for shadow backfill manifest and quality helpers."""

from __future__ import annotations

import json

from click.testing import CliRunner
from unittest.mock import patch

from bts.cli import cli
from bts.picks import DailyPick, Pick, load_shadow_pick, save_pick, save_shadow_pick
from bts.shadow_eval import build_shadow_backfill_manifest, apply_shadow_backfill_manifest


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


def _hit_checker(results: dict[int, bool | None]):
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

    with patch("bts.shadow_eval.check_hit", side_effect=lambda *args, **kwargs: args[1] == 1):
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
