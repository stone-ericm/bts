"""Tests for `bts experiment select` CLI flags added 2026-04-29 (item #4 from
2026-04-28 retro):

  --keep-t-threshold FLOAT  : pass-through to run_selection
  --min-effect-size FLOAT   : pass-through to run_selection
  --seed-set NAME           : load seeds from data/seed_sets/<NAME>.json
                               (mutually exclusive with --seeds)

The pooled-mode capability already exists via `--seeds 1,2,3,...`; these flags
make it easier to invoke pooled selection with the new keep-rule knobs from
item #1 and the canonical seed manifest, without copy/pasting seed lists.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    """Wire RESULTS_BASE + a one-passing-experiment phase1 directory."""
    from bts.experiment import cli as cli_mod
    monkeypatch.setattr(cli_mod, "RESULTS_BASE", tmp_path)
    p1 = tmp_path / "phase1" / "heat_dome"
    p1.mkdir(parents=True)
    (p1 / "summary.txt").write_text("PASS | reason")
    (p1 / "diff.json").write_text(json.dumps({
        "p_at_1_by_season": {"2024": {"delta": 0.005}, "2025": {"delta": 0.005}},
        "streak_metrics": {"mean_max_streak": {"delta": 1.0}},
        "p_57_exact": {"delta": 0.001},
        "p_57_mdp": {"delta": 0.001},
    }))
    return tmp_path


@pytest.fixture
def fake_pa_data(tmp_path):
    """Empty pa_2024.parquet so cli's parquet glob succeeds."""
    proc = tmp_path / "data_processed"
    proc.mkdir()
    pd.DataFrame({"batter_id": [1, 2], "is_hit": [0, 1]}).to_parquet(proc / "pa_2024.parquet")
    return proc


@pytest.fixture
def stub_run_selection():
    """Patch run_selection in its module — the cli imports it lazily."""
    with patch("bts.experiment.runner.run_selection") as mock_run:
        mock_run.return_value = {
            "included": [], "forward_log": [], "backward_log": [],
            "final_scorecard": {}, "final_diff": {},
        }
        yield mock_run


@pytest.fixture
def stub_features():
    """compute_all_features pass-through (avoids the 1.5M-row pipeline)."""
    with patch("bts.features.compute.compute_all_features", side_effect=lambda x: x):
        yield


class TestKeepRuleFlags:
    def test_keep_t_threshold_passes_through(self, fake_results, fake_pa_data,
                                              stub_run_selection, stub_features):
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "select",
            "--data-dir", str(fake_pa_data),
            "--keep-t-threshold", "2.0",
            "--seeds", "1,2,3",
        ])
        assert result.exit_code == 0, result.output
        kwargs = stub_run_selection.call_args.kwargs
        assert kwargs.get("keep_t_threshold") == 2.0

    def test_min_effect_size_passes_through(self, fake_results, fake_pa_data,
                                             stub_run_selection, stub_features):
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "select",
            "--data-dir", str(fake_pa_data),
            "--min-effect-size", "0.01",
            "--seeds", "1,2,3",
        ])
        assert result.exit_code == 0, result.output
        kwargs = stub_run_selection.call_args.kwargs
        assert kwargs.get("min_effect_size") == 0.01

    def test_keep_t_threshold_default_is_1_5(self, fake_results, fake_pa_data,
                                              stub_run_selection, stub_features):
        """When --keep-t-threshold is not passed, default 1.5 should propagate."""
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "select",
            "--data-dir", str(fake_pa_data),
            "--seeds", "1,2,3",
        ])
        assert result.exit_code == 0, result.output
        kwargs = stub_run_selection.call_args.kwargs
        # Either explicitly 1.5 or absent (so run_selection's default applies).
        assert kwargs.get("keep_t_threshold", 1.5) == 1.5


class TestSeedSetFlag:
    def test_seed_set_canonical_n10_loads_10_seeds(self, fake_results, fake_pa_data,
                                                     stub_run_selection, stub_features):
        """--seed-set canonical-n10 should load the 10 seeds from the manifest."""
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "select",
            "--data-dir", str(fake_pa_data),
            "--seed-set", "canonical-n10",
        ])
        assert result.exit_code == 0, result.output
        kwargs = stub_run_selection.call_args.kwargs
        seeds = kwargs.get("seeds")
        assert seeds is not None
        assert len(seeds) == 10
        # Sanity: all the seeds match the manifest content.
        manifest = json.loads(Path("data/seed_sets/canonical-n10.json").read_text())
        assert sorted(seeds) == sorted(int(s) for s in manifest["seeds"])

    def test_seed_set_unknown_errors(self, fake_results, fake_pa_data,
                                      stub_run_selection, stub_features):
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "select",
            "--data-dir", str(fake_pa_data),
            "--seed-set", "nope-not-a-real-set",
        ])
        assert result.exit_code != 0
        assert "nope-not-a-real-set" in result.output.lower() or \
               "not found" in result.output.lower()
        # Should NOT have called run_selection — fail fast on the bad arg.
        assert stub_run_selection.call_count == 0

    def test_seed_set_and_seeds_mutually_exclusive(self, fake_results, fake_pa_data,
                                                    stub_run_selection, stub_features):
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "select",
            "--data-dir", str(fake_pa_data),
            "--seeds", "1,2,3",
            "--seed-set", "canonical-n10",
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or \
               "cannot use both" in result.output.lower() or \
               "--seeds" in result.output.lower() and "--seed-set" in result.output.lower()
        assert stub_run_selection.call_count == 0
