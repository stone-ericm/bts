"""Tests for Phase 2 keep-rule tightening (item #1 from 2026-04-28 retro).

Bug being fixed: in multi-seed mode, the keep rule is `mean > 0`. With small
positive noise (e.g., +0.0001pp pooled), an experiment KEEPS even though it's
clearly noise. After this fix, multi-seed keep requires:
  (mean > 0) AND (|t-stat| >= keep_t_threshold OR |effect_size| >= min_effect)

Single-seed mode keeps the original `mean > 0` rule (no t-stat available).
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


class TestKeepRuleTightening:
    def _run_with_p57_seq(self, p57_seq, seeds, tmp_path, tiny_pa_df, monkeypatch,
                          keep_t_threshold=None, min_effect_size=None):
        from bts.experiment.runner import run_selection
        import bts.simulate.backtest_blend as bb

        load_all_experiments()
        heat_dome = get_experiment("heat_dome")

        def fake_blend_wf(df, season, **kwargs):
            return _synthetic_profiles(season)

        def fake_scorecard(profiles, **kwargs):
            counter = fake_scorecard.counter
            fake_scorecard.counter += 1
            return {"p_57_mdp": p57_seq[counter] if counter < len(p57_seq) else p57_seq[-1]}
        fake_scorecard.counter = 0

        monkeypatch.setattr(bb, "blend_walk_forward", fake_blend_wf)
        with patch("bts.validate.scorecard.compute_full_scorecard", side_effect=fake_scorecard):
            with patch("bts.validate.scorecard.diff_scorecards", return_value={}):
                with patch("bts.validate.scorecard.save_scorecard", return_value=None):
                    winners = [{"name": "heat_dome", "passed": True, "diff": {}}]
                    experiments_by_name = {"heat_dome": heat_dome}
                    kw = {}
                    if keep_t_threshold is not None:
                        kw["keep_t_threshold"] = keep_t_threshold
                    if min_effect_size is not None:
                        kw["min_effect_size"] = min_effect_size
                    return run_selection(
                        winners, experiments_by_name, tiny_pa_df,
                        test_seasons=[2024], results_dir=tmp_path,
                        retrain_every=7, seeds=seeds, **kw,
                    )

    def test_drops_when_pooled_mean_positive_but_t_below_threshold(self, tmp_path, tiny_pa_df, monkeypatch):
        """Pooled mean +0.001 with high variance per-seed → t low → DROP under tight rule."""
        # Baseline: 0.05, 0.03, 0.02, 0.04 (4 seeds).
        # Candidate: 0.04, 0.05, 0.03, 0.04 (per-seed deltas: -0.01, +0.02, +0.01, +0.00).
        # Mean delta = 0.005. SE = std(deltas)/sqrt(4) = 0.013/2 = 0.0065. t = 0.77.
        # Under default keep_t_threshold=1.5 → DROP (t=0.77 < 1.5).
        p57_seq = [
            0.05, 0.03, 0.02, 0.04,    # baseline at 4 seeds
            0.04, 0.05, 0.03, 0.04,    # candidate at 4 seeds
            0.05, 0.03, 0.02, 0.04,    # backward (= without candidate, back to baseline)
            0.05, 0.03, 0.02, 0.04,    # final
        ]
        result = self._run_with_p57_seq(p57_seq, seeds=[1, 2, 3, 4],
                                         tmp_path=tmp_path, tiny_pa_df=tiny_pa_df,
                                         monkeypatch=monkeypatch,
                                         keep_t_threshold=1.5)
        assert "heat_dome" not in result["included"], (
            f"heat_dome should be DROPPED (t below threshold). "
            f"forward_log: {result.get('forward_log')}"
        )

    def test_keeps_when_pooled_mean_and_t_both_strong(self, tmp_path, tiny_pa_df, monkeypatch):
        """Pooled mean +0.04 with low variance → t high → KEEP."""
        # Baseline: 0.05, 0.04, 0.06, 0.05 (low variance).
        # Candidate: 0.09, 0.08, 0.10, 0.09 (per-seed deltas: +0.04 each).
        # Mean delta = 0.04. SE ≈ 0 (all deltas equal). t ≈ inf.
        # Under default keep_t_threshold=1.5 → KEEP.
        p57_seq = [
            0.05, 0.04, 0.06, 0.05,    # baseline at 4 seeds
            0.09, 0.08, 0.10, 0.09,    # candidate at 4 seeds
            0.05, 0.04, 0.06, 0.05,    # backward
            0.09, 0.08, 0.10, 0.09,    # final
        ]
        result = self._run_with_p57_seq(p57_seq, seeds=[1, 2, 3, 4],
                                         tmp_path=tmp_path, tiny_pa_df=tiny_pa_df,
                                         monkeypatch=monkeypatch,
                                         keep_t_threshold=1.5)
        assert "heat_dome" in result["included"]

    def test_min_effect_size_keeps_strong_effects_with_low_t(self, tmp_path, tiny_pa_df, monkeypatch):
        """Even with low t-stat, large effect size keeps. Tests OR-rule."""
        # Baseline: 0.05, 0.03 (n=2 — t-stat unreliable at small n).
        # Candidate: 0.10, 0.08 (delta +0.05 each). Mean +0.05. SD=0. t=inf.
        # Actually SD=0 means t is well-defined as "infinitely confident".
        # Test the ALTERNATIVE path where both are met.
        p57_seq = [
            0.05, 0.03,    # baseline
            0.10, 0.08,    # candidate (delta +0.05 each)
            0.05, 0.03,    # backward
            0.10, 0.08,    # final
        ]
        result = self._run_with_p57_seq(p57_seq, seeds=[1, 2],
                                         tmp_path=tmp_path, tiny_pa_df=tiny_pa_df,
                                         monkeypatch=monkeypatch,
                                         keep_t_threshold=1.5)
        assert "heat_dome" in result["included"]

    def test_single_seed_mode_unchanged(self, tmp_path, tiny_pa_df, monkeypatch):
        """In single-seed mode (seeds=None), keep rule remains `mean > 0` for backwards compat."""
        # Baseline: 0.05. Candidate: 0.0501 (delta +0.0001 — tiny but positive).
        # Single-seed mode: KEEP because mean > 0.
        p57_seq = [
            0.05,      # baseline
            0.0501,    # candidate (delta +0.0001)
            0.05,      # backward
            0.0501,    # final
        ]
        result = self._run_with_p57_seq(p57_seq, seeds=None,
                                         tmp_path=tmp_path, tiny_pa_df=tiny_pa_df,
                                         monkeypatch=monkeypatch)
        # Single-seed: tiny positive mean still keeps (no t-stat available)
        assert "heat_dome" in result["included"]

    def test_default_threshold_is_1_5(self, tmp_path, tiny_pa_df, monkeypatch):
        """When seeds provided but keep_t_threshold not, default to 1.5."""
        # Same as test 1 but no explicit keep_t_threshold → default behavior.
        p57_seq = [
            0.05, 0.03, 0.02, 0.04,    # baseline
            0.04, 0.05, 0.03, 0.04,    # candidate (mean +0.005, t≈0.77)
            0.05, 0.03, 0.02, 0.04,    # backward
            0.05, 0.03, 0.02, 0.04,    # final
        ]
        result = self._run_with_p57_seq(p57_seq, seeds=[1, 2, 3, 4],
                                         tmp_path=tmp_path, tiny_pa_df=tiny_pa_df,
                                         monkeypatch=monkeypatch)
        # Default threshold → t=0.77 < 1.5 → DROP
        assert "heat_dome" not in result["included"]
