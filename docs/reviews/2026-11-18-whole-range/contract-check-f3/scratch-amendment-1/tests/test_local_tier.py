"""Tests for the local execution tier (no SSH)."""
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from bts.orchestrator import predict_local, run_cascade


def test_predict_local_calls_run_pipeline_directly():
    fake_predictions = pd.DataFrame([
        {"batter_name": "A", "batter_id": 1, "p_game_hit": 0.85},
    ])
    with patch("bts.model.predict.run_pipeline", return_value=fake_predictions) as mock_run, \
         patch("bts.model.predict.load_blend"):
        result = predict_local(date="2026-04-10")

    assert result is not None
    assert len(result) == 1
    mock_run.assert_called_once()


def test_predict_local_returns_none_on_exception():
    with patch("bts.model.predict.run_pipeline", side_effect=RuntimeError("disk full")), \
         patch("bts.model.predict.load_blend"):
        result = predict_local(date="2026-04-10")
    assert result is None


def test_cascade_supports_local_tier_type():
    tiers = [
        {"name": "local", "type": "local"},
    ]
    fake_df = pd.DataFrame([{"batter_name": "X", "p_game_hit": 0.80}])
    with patch("bts.orchestrator.predict_local", return_value=fake_df):
        df, tier = run_cascade(tiers=tiers, date="2026-04-10")
    assert df is not None
    assert tier == "local"


def test_cascade_default_ssh_type_when_unspecified():
    """Existing [[tiers]] entries without type field still work (backward compat)."""
    tiers = [
        {"name": "mac", "ssh_host": "mac", "bts_dir": "/path", "timeout_min": 5},
    ]
    with patch("bts.orchestrator.ssh_predict", return_value=pd.DataFrame([{"p_game_hit": 0.9}])) as mock_ssh:
        df, tier = run_cascade(tiers=tiers, date="2026-04-10")
    assert df is not None
    assert tier == "mac"
    mock_ssh.assert_called_once()
