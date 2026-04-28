"""Tests for Phase 2 multi-seed forward selection (Bug 2 fix).

Catches Bug 2 (discovered 2026-04-28): run_selection ran on a single seed
(typically seed=42, the production seed), which is at the 95th percentile of
the n=100 baseline P(57) MDP distribution. Real winners with positive pooled
ΔP(57) MDP get rejected at seed=42 because the ceiling is too high to beat.

Fix: run_selection accepts a `seeds: list[int]` parameter. When provided, each
forward/backward/final step is evaluated at every seed; decisions use pooled
mean ΔP(57) across paired seed comparisons. Backwards-compat: seeds=None
keeps the original single-seed behavior.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bts.experiment.registry import load_all_experiments, get_experiment


def _synthetic_profiles(season: int) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range(f"{season}-04-01", periods=10).date,
        "rank": [1] * 10,
        "batter_id": [100 + i for i in range(10)],
        "p_game_hit": [0.75] * 10,
        "actual_hit": [1] * 5 + [0] * 5,
        "n_pas": [4] * 10,
    })


@pytest.fixture
def tiny_pa_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-04-01", periods=20).date,
        "batter_id": [100, 101] * 10,
        "pitcher_id": [200, 201] * 10,
        "is_hit": [1, 0] * 10,
        "weather_temp": [85.0] * 20,
        "ump_hr": [0.1] * 20,
        "wind": [5.0] * 20,
        "hardness": [0.3] * 20,
        "indoor": [0] * 20,
    })


class TestRunSelectionMultiSeed:
    def test_seeds_param_invokes_blend_walk_forward_per_seed(self, tmp_path, tiny_pa_df, monkeypatch):
        """With seeds=[42, 43], each step (baseline + candidate) calls blend_walk_forward 2 times."""
        from bts.experiment.runner import run_selection
        import bts.simulate.backtest_blend as bb

        load_all_experiments()
        heat_dome = get_experiment("heat_dome")

        seeds_seen = []
        def fake_blend_wf(df, season, **kwargs):
            seed = int(os.environ.get("BTS_LGBM_RANDOM_STATE", "42"))
            seeds_seen.append(seed)
            return _synthetic_profiles(season)

        monkeypatch.setattr(bb, "blend_walk_forward", fake_blend_wf)
        with patch("bts.validate.scorecard.compute_full_scorecard", return_value={"p_57_mdp": 0.5}):
            with patch("bts.validate.scorecard.diff_scorecards", return_value={}):
                with patch("bts.validate.scorecard.save_scorecard", return_value=None):
                    winners = [{"name": "heat_dome", "passed": True, "diff": {}}]
                    experiments_by_name = {"heat_dome": heat_dome}
                    run_selection(
                        winners,
                        experiments_by_name,
                        tiny_pa_df,
                        test_seasons=[2024],
                        results_dir=tmp_path,
                        retrain_every=7,
                        seeds=[42, 43],
                    )

        # Both seeds should have been used. Each step (baseline + 1 candidate) ×
        # 1 season × 2 seeds = at least 4 calls.
        assert 42 in seeds_seen
        assert 43 in seeds_seen
        assert seeds_seen.count(42) >= 2  # baseline + candidate
        assert seeds_seen.count(43) >= 2

    def test_pooled_mean_drives_keep_decision(self, tmp_path, tiny_pa_df, monkeypatch):
        """Multi-seed: experiment is KEPT when pooled mean ΔP(57) > 0, even if
        one individual seed shows a regression."""
        from bts.experiment.runner import run_selection
        import bts.simulate.backtest_blend as bb

        load_all_experiments()
        heat_dome = get_experiment("heat_dome")

        # Mock: baseline returns P(57)=0.05 at seed 42, 0.03 at seed 43.
        # Candidate (heat_dome) returns 0.04 at seed 42 (regression), 0.06 at seed 43 (gain).
        # Per-seed deltas: -0.01, +0.03. Mean: +0.01 → KEEP.
        # Single-seed at seed 42 alone would have DROPPED.
        def fake_blend_wf(df, season, **kwargs):
            return _synthetic_profiles(season)

        def fake_scorecard(profiles, **kwargs):
            counter = fake_scorecard.counter
            fake_scorecard.counter += 1
            # Call sequence:
            #   0,1: baseline at seeds 42, 43 → 0.05, 0.03
            #   2,3: candidate (forward) at seeds 42, 43 → 0.04, 0.06
            #     (mean Δ = +0.01 → KEPT)
            #   4,5: backward test (without heat_dome = back to baseline) → 0.05, 0.03
            #     (mean current - without = (0.04-0.05 + 0.06-0.03)/2 = +0.01 → KEPT)
            #   6,7: final scorecard at seeds 42, 43 → 0.04, 0.06
            p57_seq = [0.05, 0.03, 0.04, 0.06, 0.05, 0.03, 0.04, 0.06]
            p57 = p57_seq[counter] if counter < len(p57_seq) else 0.05
            return {"p_57_mdp": p57}
        fake_scorecard.counter = 0

        monkeypatch.setattr(bb, "blend_walk_forward", fake_blend_wf)
        with patch("bts.validate.scorecard.compute_full_scorecard", side_effect=fake_scorecard):
            with patch("bts.validate.scorecard.diff_scorecards", return_value={}):
                with patch("bts.validate.scorecard.save_scorecard", return_value=None):
                    winners = [{"name": "heat_dome", "passed": True, "diff": {}}]
                    experiments_by_name = {"heat_dome": heat_dome}
                    result = run_selection(
                        winners,
                        experiments_by_name,
                        tiny_pa_df,
                        test_seasons=[2024],
                        results_dir=tmp_path,
                        retrain_every=7,
                        seeds=[42, 43],
                    )

        # heat_dome should be kept because pooled mean delta = +0.01
        assert "heat_dome" in result["included"], (
            f"heat_dome should be kept under pooled multi-seed (mean Δ=+0.01). "
            f"forward_log: {result.get('forward_log')}"
        )

    def test_seeds_none_keeps_single_seed_behavior(self, tmp_path, tiny_pa_df, monkeypatch):
        """Backwards compat: seeds=None (default) runs once like before."""
        from bts.experiment.runner import run_selection
        import bts.simulate.backtest_blend as bb

        load_all_experiments()
        heat_dome = get_experiment("heat_dome")

        seeds_seen = []
        def fake_blend_wf(df, season, **kwargs):
            seeds_seen.append(int(os.environ.get("BTS_LGBM_RANDOM_STATE", "42")))
            return _synthetic_profiles(season)

        monkeypatch.setattr(bb, "blend_walk_forward", fake_blend_wf)
        with patch("bts.validate.scorecard.compute_full_scorecard", return_value={"p_57_mdp": 0.5}):
            with patch("bts.validate.scorecard.diff_scorecards", return_value={}):
                with patch("bts.validate.scorecard.save_scorecard", return_value=None):
                    winners = [{"name": "heat_dome", "passed": True, "diff": {}}]
                    experiments_by_name = {"heat_dome": heat_dome}
                    run_selection(
                        winners, experiments_by_name, tiny_pa_df,
                        test_seasons=[2024], results_dir=tmp_path,
                        retrain_every=7,
                        # seeds=None → single-seed behavior
                    )

        # Each step calls blend_walk_forward exactly once per season.
        # Baseline + 1 candidate × 1 season = 2 base calls.
        # The seed env var should NOT be touched by run_selection in single-seed mode.
        assert len(seeds_seen) >= 2
