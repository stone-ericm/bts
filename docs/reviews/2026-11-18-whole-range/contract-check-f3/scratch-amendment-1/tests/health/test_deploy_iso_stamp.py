"""since_deploy_iso (2)+(3): prefer a real deploy-time stamp over git commit-time.

The stamp is wall-clock and monotonic, so it fixes both commit-time≠deploy-time
and the canary-rollback-moves-HEAD's-date-backward case. Falls back to git %cI
when absent (older boxes / local runs) — no regression.
"""
from unittest.mock import patch

from bts.health.realized_calibration import _current_deploy_iso


def test_uses_deploy_stamp_when_present(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / ".last_deploy_iso").write_text("2026-06-11T03:30:00Z\n")
    assert _current_deploy_iso(tmp_path) == "2026-06-11T03:30:00Z"


@patch("subprocess.check_output", return_value="2026-06-10T12:00:00-04:00\n")
def test_falls_back_to_git_commit_time_when_no_stamp(mock_co, tmp_path):
    assert _current_deploy_iso(tmp_path) == "2026-06-10T12:00:00-04:00"
    mock_co.assert_called_once()  # git was consulted only because the stamp was absent


@patch("subprocess.check_output", return_value="2026-06-10T12:00:00-04:00\n")
def test_empty_stamp_falls_back_to_git(mock_co, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / ".last_deploy_iso").write_text("   \n")
    assert _current_deploy_iso(tmp_path) == "2026-06-10T12:00:00-04:00"
