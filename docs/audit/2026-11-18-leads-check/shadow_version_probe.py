import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner

from bts import picks, shadow_eval
from bts.cli import cli


DATE = "2027-06-12"
VERSION = shadow_eval.SHADOW_MODEL_NAME


def sample(date, result=None):
    time = date + "T19:00:00Z"
    pick = picks.Pick("Example Batter", 1, "NYM", 1, "Example Pitcher", 2, .8, [], False, 101, time)
    return picks.DailyPick(date, time, pick, None, None, result=result)


def grade(root):
    prod_path = picks.save_pick(sample(DATE, "hit"), root)
    prod_bytes = prod_path.read_bytes()
    daily = sample(DATE)
    shadow_eval.stamp_shadow_version(daily)
    path = picks.save_shadow_pick(daily, root)
    assert picks.load_pick(DATE, root).shadow_model_version is None
    assert picks.parse_daily_pick(path.read_text()).shadow_model_version is None
    assert picks.load_shadow_pick(DATE, root).shadow_model_version == VERSION

    with patch.object(picks, "resolve_daily_slot_results", return_value={"pick": "hit"}) as resolve:
        result = CliRunner().invoke(cli, ["check-results", "--picks-dir", str(root), "--date", DATE],
                                    catch_exceptions=False)
        resolve.assert_called_once()
    assert result.exit_code == 0, result.output
    graded = json.loads(path.read_text())
    assert graded["shadow_model_version"] == VERSION and graded["result"] == "hit"
    assert prod_path.read_bytes() == prod_bytes
    status = shadow_eval.build_shadow_cycle_status(root)
    assert status["counts"]["resolved_paired_days"] == 1

    # Deliberate unsupported round trip shows the consequence if a future
    # caller passes shadow content to the generic production parser.
    picks.save_shadow_pick(picks.parse_daily_pick(path.read_text()), root)
    assert shadow_eval.build_shadow_cycle_status(root)["counts"]["shadow_files"] == 0
    print("check-results keeps the tag and v2 pair; forced production-parser round trip loses both")


def backfill(root):
    picks.save_pick(sample(DATE, "hit"), root)
    current = sample(DATE)
    shadow_eval.stamp_shadow_version(current)
    path = picks.save_shadow_pick(current, root)
    legacy = "2027-06-13"
    legacy_path = picks.save_shadow_pick(sample(legacy), root)
    legacy_bytes = legacy_path.read_bytes()
    manifest = shadow_eval.build_shadow_backfill_manifest(
        root, raw_dir=root / "absent", hit_checker=lambda *_: True, n_bootstrap=0)
    assert [row["date"] for row in manifest["rows"]] == [DATE]
    applied = shadow_eval.apply_shadow_backfill_manifest(manifest, backup_dir=root / "backup")
    assert [row["date"] for row in applied["applied"]] == [DATE]
    assert picks.load_shadow_pick(DATE, root).shadow_model_version == VERSION
    assert picks.load_shadow_pick(DATE, root).result == "hit"
    assert legacy_path.read_bytes() == legacy_bytes
    print("backfill independently filters out legacy and applies current rows while preserving the tag")


if __name__ == "__main__":
    with TemporaryDirectory() as tmp, patch("socket.socket.connect", side_effect=AssertionError("network")):
        root = Path(tmp)
        grade(root / "grade")
        backfill(root / "backfill")
