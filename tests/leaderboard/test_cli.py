"""CLI smoke tests for `bts leaderboard`."""
from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from bts.leaderboard.cli import leaderboard


class TestLeaderboardCLI:
    def test_scrape_invokes_run_with_cookies_and_xsid(self, tmp_path):
        runner = CliRunner()
        with patch("bts.leaderboard.cli.scraper_run") as mock_run, \
             patch("bts.leaderboard.cli.load_session_cookies", return_value={"oktaid": "00u"}), \
             patch("bts.leaderboard.cli.extract_uid", return_value="00u"), \
             patch("bts.leaderboard.cli.fetch_xsid", return_value="x_123"):
            result = runner.invoke(leaderboard, ["scrape", "--output-dir", str(tmp_path), "--top-n", "10"])
        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        assert kwargs["xsid"] == "x_123"
        assert kwargs["top_n"] == 10

    def test_status_when_no_data(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(leaderboard, ["status", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "no successful scrape yet" in result.output.lower()

    def test_scrape_handles_auth_error(self, tmp_path):
        from bts.leaderboard.auth import AuthError
        runner = CliRunner()
        with patch("bts.leaderboard.cli.load_session_cookies", side_effect=AuthError("expired")):
            result = runner.invoke(leaderboard, ["scrape", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "auth" in result.output.lower()
