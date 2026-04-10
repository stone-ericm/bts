"""Tests for bts shadow-report CLI command."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bts.cli import cli


def _write_pick_pair(picks_dir: Path, date: str,
                     prod_name: str, prod_p: float, prod_result: str | None,
                     shadow_name: str, shadow_p: float, shadow_result: str | None):
    """Write matching production + shadow pick files."""
    for suffix, name, p, result in [
        (".json", prod_name, prod_p, prod_result),
        (".shadow.json", shadow_name, shadow_p, shadow_result),
    ]:
        data = {
            "date": date,
            "run_time": f"{date}T22:00:00Z",
            "pick": {
                "batter_name": name, "batter_id": 100, "team": "SF",
                "lineup_position": 1, "pitcher_name": "Pitcher", "pitcher_id": 200,
                "p_game_hit": p, "flags": [], "projected_lineup": False,
                "game_pk": 800000, "game_time": f"{date}T23:00:00Z", "pitcher_team": "BAL",
            },
            "double_down": None,
            "runner_up": None,
            "bluesky_posted": False,
            "bluesky_uri": None,
            "result": result,
        }
        (picks_dir / f"{date}{suffix}").write_text(json.dumps(data))


class TestShadowReport:
    def test_no_shadow_picks(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["shadow-report", "--picks-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No shadow pick pairs found" in result.output

    def test_agreement_rate(self, tmp_path):
        _write_pick_pair(tmp_path, "2026-04-01", "Arraez", 0.77, "hit", "Arraez", 0.76, None)
        _write_pick_pair(tmp_path, "2026-04-02", "Kwan", 0.72, "hit", "Marte", 0.73, None)
        _write_pick_pair(tmp_path, "2026-04-03", "Arraez", 0.75, "miss", "Arraez", 0.74, None)
        runner = CliRunner()
        result = runner.invoke(cli, ["shadow-report", "--picks-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Agreement" in result.output
        assert "66.7%" in result.output  # 2 of 3 agree

    def test_shows_disagreement_detail(self, tmp_path):
        _write_pick_pair(tmp_path, "2026-04-01", "Arraez", 0.77, "hit", "Marte", 0.76, None)
        runner = CliRunner()
        result = runner.invoke(cli, ["shadow-report", "--picks-dir", str(tmp_path)])
        assert "Arraez" in result.output
        assert "Marte" in result.output
