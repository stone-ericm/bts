"""Tests for `bts experiment screen` multi-seed flags (item #3 from
2026-04-28 retro):

  --seeds COMMA       : pool across explicit seed list
  --seed-set NAME     : pool using canonical-n10 manifest (mutually exclusive
                        with --seeds)
  --keep-t-threshold  : ignored at Phase 1 (kept for symmetry with select);
                        not exposed yet on screen — we'll add later if Phase 1
                        adopts the same keep rule.

When --seeds (or --seed-set) is provided, the CLI loops over seeds setting
BTS_LGBM_RANDOM_STATE per iteration, calls run_screening once per seed, and
writes results into seed-specific subdirs under phase1/seed_<seed>/. Default
behavior (no flags) is unchanged — single-seed via env var, preserving the
audit_driver-as-orchestrator contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    """Wire RESULTS_BASE to a tmp dir so per-seed subdirs land in tmp."""
    from bts.experiment import cli as cli_mod
    monkeypatch.setattr(cli_mod, "RESULTS_BASE", tmp_path)
    return tmp_path


@pytest.fixture
def fake_pa_data(tmp_path):
    """Empty pa_2024.parquet so cli's parquet glob succeeds."""
    proc = tmp_path / "data_processed"
    proc.mkdir()
    pd.DataFrame({"batter_id": [1, 2], "is_hit": [0, 1]}).to_parquet(proc / "pa_2024.parquet")
    return proc


@pytest.fixture
def stub_run_screening():
    """Patch run_screening at its module — cli imports lazily."""
    with patch("bts.experiment.runner.run_screening") as mock_run:
        mock_run.return_value = []  # empty results — format_phase1_table handles []
        yield mock_run


@pytest.fixture
def stub_baseline():
    """Skip the actual baseline walk-forward path."""
    with patch("bts.simulate.backtest_blend.blend_walk_forward") as mock_wf, \
         patch("bts.validate.scorecard.compute_full_scorecard", return_value={"p_57_mdp": 0.5}), \
         patch("bts.validate.scorecard.save_scorecard"):
        # Return a tiny synthetic profiles frame so concat / column-set ops don't blow up
        mock_wf.return_value = pd.DataFrame({
            "date": pd.date_range("2024-04-01", periods=3).date,
            "rank": [1] * 3,
            "batter_id": [100, 101, 102],
            "p_game_hit": [0.7] * 3,
            "actual_hit": [1, 0, 1],
            "n_pas": [4] * 3,
        })
        yield mock_wf


@pytest.fixture
def stub_features():
    with patch("bts.features.compute.compute_all_features", side_effect=lambda x: x):
        yield


class TestSeedsFlag:
    def test_screen_accepts_seeds_flag_and_loops(
        self, fake_results, fake_pa_data, stub_run_screening, stub_baseline, stub_features,
    ):
        """`screen --seeds 1,2,3` should call run_screening 3 times."""
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "screen",
            "--data-dir", str(fake_pa_data),
            "--subset", "heat_dome",
            "--seeds", "1,2,3",
            "--no-use-factored",
        ])
        assert result.exit_code == 0, result.output
        assert stub_run_screening.call_count == 3

    def test_screen_seeds_sets_env_var_per_iteration(
        self, fake_results, fake_pa_data, stub_baseline, stub_features,
    ):
        """For each call, BTS_LGBM_RANDOM_STATE should be set to the iteration's seed."""
        from bts.cli import cli
        seeds_seen = []

        def capture_env(*args, **kwargs):
            seeds_seen.append(int(os.environ.get("BTS_LGBM_RANDOM_STATE", "-1")))
            return []

        with patch("bts.experiment.runner.run_screening", side_effect=capture_env):
            runner = CliRunner()
            result = runner.invoke(cli, [
                "experiment", "screen",
                "--data-dir", str(fake_pa_data),
                "--subset", "heat_dome",
                "--seeds", "11,22,33",
                "--no-use-factored",
            ])
        assert result.exit_code == 0, result.output
        assert seeds_seen == [11, 22, 33]

    def test_screen_writes_per_seed_subdirs(
        self, fake_results, fake_pa_data, stub_run_screening, stub_baseline, stub_features,
    ):
        """When seeds provided, results_dir passed to run_screening should be
        seed-specific (phase1/seed_<seed>/)."""
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "screen",
            "--data-dir", str(fake_pa_data),
            "--subset", "heat_dome",
            "--seeds", "11,22",
            "--no-use-factored",
        ])
        assert result.exit_code == 0, result.output
        # Each call's results_dir arg is seed-specific
        results_dirs = [c.args[4] for c in stub_run_screening.call_args_list]
        assert any("seed_11" in str(d) for d in results_dirs)
        assert any("seed_22" in str(d) for d in results_dirs)


class TestSeedSetFlag:
    def test_screen_seed_set_canonical_n10(
        self, fake_results, fake_pa_data, stub_run_screening, stub_baseline, stub_features,
    ):
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "screen",
            "--data-dir", str(fake_pa_data),
            "--subset", "heat_dome",
            "--seed-set", "canonical-n10",
            "--no-use-factored",
        ])
        assert result.exit_code == 0, result.output
        assert stub_run_screening.call_count == 10

    def test_screen_seed_set_unknown_errors(
        self, fake_results, fake_pa_data, stub_run_screening, stub_baseline, stub_features,
    ):
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "screen",
            "--data-dir", str(fake_pa_data),
            "--subset", "heat_dome",
            "--seed-set", "nope-not-a-real-set",
            "--no-use-factored",
        ])
        assert result.exit_code != 0
        assert "nope-not-a-real-set" in result.output.lower() or "not found" in result.output.lower()
        assert stub_run_screening.call_count == 0

    def test_screen_seeds_and_seed_set_mutually_exclusive(
        self, fake_results, fake_pa_data, stub_run_screening, stub_baseline, stub_features,
    ):
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "screen",
            "--data-dir", str(fake_pa_data),
            "--subset", "heat_dome",
            "--seeds", "1,2",
            "--seed-set", "canonical-n10",
            "--no-use-factored",
        ])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()
        assert stub_run_screening.call_count == 0


class TestDefaultBehaviorUnchanged:
    def test_screen_no_flags_calls_run_screening_once(
        self, fake_results, fake_pa_data, stub_run_screening, stub_baseline, stub_features,
    ):
        """Default behavior preserved: no --seeds/--seed-set → 1 call (single-seed)."""
        from bts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "screen",
            "--data-dir", str(fake_pa_data),
            "--subset", "heat_dome",
            "--no-use-factored",
        ])
        assert result.exit_code == 0, result.output
        assert stub_run_screening.call_count == 1
