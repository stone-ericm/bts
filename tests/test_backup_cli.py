"""CLI wiring tests for `bts backup` (audit F5)."""
import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from bts.cli import cli


def test_backup_run_invokes_module_and_exits_zero_on_success(tmp_path):
    ok_entry = {"set": "ops", "ok": True, "snapshot_id": "abc"}
    with patch("bts.data.backup.run_backup", return_value=ok_entry) as mock_run:
        result = CliRunner().invoke(
            cli, ["backup", "run", "--set", "ops", "--repo-root", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == "ops"
    assert "abc" in result.output


def test_backup_run_exits_nonzero_on_failure(tmp_path):
    bad = {"set": "ops", "ok": False, "error": "Fatal: locked"}
    with patch("bts.data.backup.run_backup", return_value=bad):
        result = CliRunner().invoke(
            cli, ["backup", "run", "--set", "ops", "--repo-root", str(tmp_path)],
        )
    assert result.exit_code != 0
    assert "locked" in result.output


def test_backup_status_prints_entries(tmp_path):
    hs = tmp_path / "data" / "health_state"
    hs.mkdir(parents=True)
    (hs / "backup_status.json").write_text(json.dumps({
        "ops": {"set": "ops", "ok": True, "last_success_at": "2026-07-10T12:00:00+00:00"},
    }))
    result = CliRunner().invoke(
        cli, ["backup", "status", "--repo-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "ops" in result.output and "2026-07-10" in result.output


def test_backup_prune_invokes_module(tmp_path):
    with patch(
        "bts.data.backup.run_prune", return_value={"ok": True, "rc": 0},
    ) as mock_prune:
        result = CliRunner().invoke(
            cli, ["backup", "prune", "--repo-root", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    mock_prune.assert_called_once()

    with patch(
        "bts.data.backup.run_prune",
        return_value={"ok": False, "rc": 1, "error": "repo locked"},
    ):
        bad = CliRunner().invoke(
            cli, ["backup", "prune", "--repo-root", str(tmp_path)],
        )
    assert bad.exit_code != 0


def test_backup_restore_drill_exit_codes(tmp_path):
    with patch(
        "bts.data.backup.restore_drill",
        return_value={"ok": True, "problems": [], "target": "/x"},
    ):
        ok = CliRunner().invoke(
            cli, ["backup", "restore-drill", "--repo-root", str(tmp_path),
                  "--target", str(tmp_path / "t")],
        )
    assert ok.exit_code == 0, ok.output

    with patch(
        "bts.data.backup.restore_drill",
        return_value={"ok": False, "problems": ["saver_state.json not found in restore"]},
    ):
        bad = CliRunner().invoke(
            cli, ["backup", "restore-drill", "--repo-root", str(tmp_path),
                  "--target", str(tmp_path / "t")],
        )
    assert bad.exit_code != 0
    assert "saver_state" in bad.output
