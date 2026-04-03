"""Tests for Pi5 orchestrator cascade logic."""

import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from bts.orchestrator import ssh_predict, run_cascade, load_config


SAMPLE_PREDICTIONS = json.dumps([
    {
        "batter_name": "Jacob Wilson",
        "batter_id": 700363,
        "team": "ATH",
        "lineup": 1,
        "pitcher_name": "Jose Suarez",
        "pitcher_id": 660761,
        "game_pk": 778899,
        "game_time": "2026-04-01T23:10:00Z",
        "p_hit_pa": 0.312,
        "p_game_hit": 0.763,
        "flags": "",
    },
])


class TestSshPredict:
    @patch("subprocess.run")
    def test_success_returns_dataframe(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=SAMPLE_PREDICTIONS, stderr="Running..."
        )
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)

        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["batter_name"] == "Jacob Wilson"

    @patch("subprocess.run")
    def test_ssh_failure_returns_none(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=300)
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)
        assert df is None

    @patch("subprocess.run")
    def test_nonzero_exit_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ERROR: No data"
        )
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)
        assert df is None

    @patch("subprocess.run")
    def test_invalid_json_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not json", stderr=""
        )
        df = ssh_predict("mac", "/path/to/bts", "2026-04-01", timeout_sec=300)
        assert df is None


class TestRunCascade:
    @patch("bts.orchestrator.ssh_predict")
    def test_first_tier_succeeds(self, mock_ssh):
        import pandas as pd
        mock_ssh.return_value = pd.DataFrame(json.loads(SAMPLE_PREDICTIONS))

        tiers = [
            {"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5},
        ]
        df, tier_name = run_cascade(tiers, "2026-04-01")

        assert tier_name == "mac"
        assert len(df) == 1

    @patch("bts.orchestrator.ssh_predict")
    def test_falls_through_to_second_tier(self, mock_ssh):
        import pandas as pd

        def side_effect(host, bts_dir, date, timeout_sec, platform="unix"):
            if host == "mac":
                return None  # Mac failed
            return pd.DataFrame(json.loads(SAMPLE_PREDICTIONS))

        mock_ssh.side_effect = side_effect

        tiers = [
            {"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5},
            {"name": "alienware", "ssh_host": "alienware", "bts_dir": "/bts", "timeout_min": 10},
        ]
        df, tier_name = run_cascade(tiers, "2026-04-01")

        assert tier_name == "alienware"
        assert len(df) == 1

    @patch("bts.orchestrator.ssh_predict")
    def test_all_tiers_fail_returns_none(self, mock_ssh):
        mock_ssh.return_value = None

        tiers = [
            {"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5},
            {"name": "alienware", "ssh_host": "alienware", "bts_dir": "/bts", "timeout_min": 10},
        ]
        df, tier_name = run_cascade(tiers, "2026-04-01")

        assert df is None
        assert tier_name is None


class TestLoadConfig:
    def test_loads_toml(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[orchestrator]
picks_dir = "/home/bts/data/picks"

[bluesky]
dm_recipient = "stonehengee.bsky.social"

[[tiers]]
name = "mac"
ssh_host = "macbook-pro.local"
bts_dir = "/Users/stone/projects/bts"
timeout_min = 5

[[tiers]]
name = "alienware"
ssh_host = "alienware"
bts_dir = "/c/Users/stone/projects/bts"
timeout_min = 10
""")
        config = load_config(config_path)

        assert len(config["tiers"]) == 2
        assert config["tiers"][0]["name"] == "mac"
        assert config["orchestrator"]["picks_dir"] == "/home/bts/data/picks"
        assert config["bluesky"]["dm_recipient"] == "stonehengee.bsky.social"


class TestRunAndPick:
    @patch("bts.orchestrator.run_cascade")
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    @patch("bts.strategy._mdp_action", return_value="single")
    def test_returns_predictions_and_result(self, _mdp, _statuses, mock_cascade, tmp_path):
        import pandas as pd
        from bts.orchestrator import run_and_pick

        mock_cascade.return_value = (
            pd.DataFrame(json.loads(SAMPLE_PREDICTIONS)),
            "mac",
        )
        config = {
            "orchestrator": {"picks_dir": str(tmp_path)},
            "tiers": [{"name": "mac", "ssh_host": "mac", "bts_dir": "/bts", "timeout_min": 5}],
        }
        predictions, pick_result, tier = run_and_pick(config, "2026-04-01")

        assert predictions is not None
        assert len(predictions) == 1
        assert tier == "mac"
        assert pick_result is not None
        assert pick_result.daily.pick.batter_name == "Jacob Wilson"
