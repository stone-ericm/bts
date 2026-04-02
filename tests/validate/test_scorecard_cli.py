"""Tests for scorecard CLI functions: compute_full_scorecard, save_scorecard, diff_scorecards.

TDD: tests written before implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.validate.scorecard import (
    compute_full_scorecard,
    diff_scorecards,
    save_scorecard,
)


# ---------------------------------------------------------------------------
# Helpers (shared with test_scorecard.py — duplicated to keep tests independent)
# ---------------------------------------------------------------------------

def _make_profiles(days: int = 30, top_n: int = 10) -> pd.DataFrame:
    """Generate synthetic backtest profile DataFrame.

    Rank 1: always hits.
    Rank 2: hits with 80% probability.
    Ranks 3+: hit with 50% probability.
    """
    rng = np.random.default_rng(42)
    rows = []
    for day_idx in range(days):
        date = pd.Timestamp("2025-04-01") + pd.Timedelta(days=day_idx)
        for rank in range(1, top_n + 1):
            p_hit = 0.90 - (rank - 1) * 0.02
            if rank == 1:
                hit = 1
            elif rank == 2:
                hit = int(rng.random() < 0.80)
            else:
                hit = int(rng.random() < 0.50)
            rows.append({
                "date": date,
                "rank": rank,
                "batter_id": 1000 + rank,
                "p_game_hit": p_hit,
                "actual_hit": hit,
                "n_pas": 4,
                "season": 2025,
            })
    return pd.DataFrame(rows)


def _make_multi_season_profiles() -> pd.DataFrame:
    """Profiles spanning two seasons."""
    rng = np.random.default_rng(99)
    rows = []
    for season in [2024, 2025]:
        for day_idx in range(20):
            date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day_idx)
            for rank in range(1, 6):
                p_hit = 0.88 - (rank - 1) * 0.03
                hit = int(rng.random() < p_hit)
                rows.append({
                    "date": date,
                    "rank": rank,
                    "batter_id": 1000 + rank,
                    "p_game_hit": p_hit,
                    "actual_hit": hit,
                    "n_pas": 4,
                    "season": season,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_full_scorecard
# ---------------------------------------------------------------------------

class TestComputeFullScorecard:
    def test_returns_dict(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert isinstance(result, dict)

    def test_has_metadata_keys(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "timestamp" in result
        assert "n_days" in result
        assert "n_rows" in result
        assert "mc_trials" in result
        assert "season_length" in result

    def test_metadata_values(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert result["n_days"] == 30
        assert result["n_rows"] == len(df)
        assert result["mc_trials"] == 50
        assert result["season_length"] == 50

    def test_has_precision_key(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "precision" in result
        assert isinstance(result["precision"], dict)
        assert 1 in result["precision"]

    def test_has_precision_by_season(self):
        df = _make_multi_season_profiles()
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "precision_by_season" in result
        # With season column, should have per-season entries
        assert len(result["precision_by_season"]) >= 1

    def test_has_p_at_1_by_season(self):
        df = _make_multi_season_profiles()
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "p_at_1_by_season" in result
        # Each entry is season → float
        for season_key, val in result["p_at_1_by_season"].items():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    def test_has_miss_analysis(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "miss_analysis" in result
        ma = result["miss_analysis"]
        assert "n_miss_days" in ma
        assert "rank_2_hit_rate_on_miss" in ma

    def test_has_calibration(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "calibration" in result
        assert isinstance(result["calibration"], list)

    def test_has_streak_metrics(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "streak_metrics" in result
        sm = result["streak_metrics"]
        assert "mean_max_streak" in sm
        assert "p_57_monte_carlo" in sm

    def test_has_p57_exact(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "p_57_exact" in result
        # Should be a float or None on error
        if result["p_57_exact"] is not None:
            assert isinstance(result["p_57_exact"], float)
            assert 0.0 <= result["p_57_exact"] <= 1.0

    def test_has_p57_mdp(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        assert "p_57_mdp" in result
        # May be None if unavailable
        if result["p_57_mdp"] is not None:
            assert isinstance(result["p_57_mdp"], float)
            assert 0.0 <= result["p_57_mdp"] <= 1.0

    def test_infers_season_from_date_if_missing(self):
        """If no season column, should still work (infer from date year)."""
        df = _make_profiles(days=30, top_n=5)
        df = df.drop(columns=["season"])
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        # Should not raise; precision_by_season should have year-keyed entries
        assert "precision_by_season" in result

    def test_json_serializable(self):
        """Full scorecard should be JSON-serializable via save_scorecard."""
        df = _make_profiles(days=30, top_n=5)
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        # Should not raise when serialized with our custom handler
        from bts.validate.scorecard import save_scorecard
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            save_scorecard(result, tmp_path)
            loaded = json.loads(tmp_path.read_text())
            assert "precision" in loaded
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_season_keys_are_int_type(self):
        """p_at_1_by_season should have integer keys, not strings."""
        df = _make_multi_season_profiles()
        result = compute_full_scorecard(df, mc_trials=50, season_length=50)
        for k in result["p_at_1_by_season"]:
            assert isinstance(k, int), f"Expected int key, got {type(k)}: {k}"


# ---------------------------------------------------------------------------
# save_scorecard
# ---------------------------------------------------------------------------

class TestSaveScorecard:
    def test_creates_file(self, tmp_path):
        df = _make_profiles(days=30, top_n=5)
        scorecard = compute_full_scorecard(df, mc_trials=50, season_length=50)
        out_path = tmp_path / "scorecard.json"
        result = save_scorecard(scorecard, out_path)
        assert out_path.exists()
        assert result == out_path

    def test_creates_parent_dirs(self, tmp_path):
        df = _make_profiles(days=30, top_n=5)
        scorecard = compute_full_scorecard(df, mc_trials=50, season_length=50)
        out_path = tmp_path / "deep" / "nested" / "scorecard.json"
        save_scorecard(scorecard, out_path)
        assert out_path.exists()

    def test_valid_json(self, tmp_path):
        df = _make_profiles(days=30, top_n=5)
        scorecard = compute_full_scorecard(df, mc_trials=50, season_length=50)
        out_path = tmp_path / "scorecard.json"
        save_scorecard(scorecard, out_path)
        content = out_path.read_text()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_handles_numpy_integers(self, tmp_path):
        """Numpy integers in the scorecard should be serialized cleanly."""
        scorecard = {
            "n_days": np.int64(42),
            "p_hit": np.float64(0.75),
            "counts": np.array([1, 2, 3]),
        }
        out_path = tmp_path / "scorecard.json"
        save_scorecard(scorecard, out_path)
        loaded = json.loads(out_path.read_text())
        assert loaded["n_days"] == 42
        assert abs(loaded["p_hit"] - 0.75) < 1e-9
        assert loaded["counts"] == [1, 2, 3]

    def test_returns_path_object(self, tmp_path):
        scorecard = {"test": 1}
        out_path = tmp_path / "sc.json"
        result = save_scorecard(scorecard, out_path)
        assert isinstance(result, Path)

    def test_accepts_string_path(self, tmp_path):
        """save_scorecard should accept a string path as well as Path."""
        scorecard = {"test": 2}
        out_str = str(tmp_path / "sc2.json")
        result = save_scorecard(scorecard, out_str)
        assert Path(out_str).exists()

    def test_indented_output(self, tmp_path):
        """Output should be pretty-printed (indent=2)."""
        scorecard = {"key": "value"}
        out_path = tmp_path / "sc.json"
        save_scorecard(scorecard, out_path)
        content = out_path.read_text()
        # Pretty-printed JSON has newlines
        assert "\n" in content

    def test_season_int_keys_survive_roundtrip(self, tmp_path):
        """Integer season keys become strings in JSON but should be noted."""
        # JSON spec: all object keys are strings. This test documents the behavior.
        df = _make_multi_season_profiles()
        scorecard = compute_full_scorecard(df, mc_trials=50, season_length=50)
        out_path = tmp_path / "sc.json"
        save_scorecard(scorecard, out_path)
        loaded = json.loads(out_path.read_text())
        # JSON keys are strings — verify the structure is preserved
        p_at_1 = loaded.get("p_at_1_by_season", {})
        assert len(p_at_1) >= 1
        # All values should be floats
        for v in p_at_1.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# diff_scorecards
# ---------------------------------------------------------------------------

class TestDiffScorecards:
    def _make_scorecard(self, p1_val=0.80, miss_days=5) -> dict:
        """Build a minimal scorecard dict with known values."""
        return {
            "precision": {1: p1_val, 5: 0.70, 10: 0.65},
            "p_at_1_by_season": {2024: 0.78, 2025: p1_val},
            "miss_analysis": {
                "n_miss_days": miss_days,
                "rank_2_hit_rate_on_miss": 0.60,
                "mean_p_hit_on_miss": 0.75,
                "mean_p_hit_on_hit": 0.80,
            },
            "streak_metrics": {
                "mean_max_streak": 12.5,
                "median_max_streak": 11,
                "p90_max_streak": 20,
                "p99_max_streak": 35,
                "p_57_monte_carlo": 0.05,
                "longest_replay_streak": 18,
            },
            "p_57_exact": 0.06,
            "p_57_mdp": 0.065,
        }

    def test_returns_dict(self):
        baseline = self._make_scorecard(p1_val=0.80)
        variant = self._make_scorecard(p1_val=0.82)
        result = diff_scorecards(baseline, variant)
        assert isinstance(result, dict)

    def test_precision_diff_has_delta(self):
        baseline = self._make_scorecard(p1_val=0.80)
        variant = self._make_scorecard(p1_val=0.82)
        result = diff_scorecards(baseline, variant)
        assert "precision" in result
        p1_diff = result["precision"][1]
        assert "baseline" in p1_diff
        assert "variant" in p1_diff
        assert "delta" in p1_diff
        assert abs(p1_diff["delta"] - 0.02) < 1e-9
        assert abs(p1_diff["baseline"] - 0.80) < 1e-9
        assert abs(p1_diff["variant"] - 0.82) < 1e-9

    def test_miss_analysis_diff(self):
        baseline = self._make_scorecard(miss_days=5)
        variant = self._make_scorecard(miss_days=3)
        result = diff_scorecards(baseline, variant)
        assert "miss_analysis" in result
        miss_diff = result["miss_analysis"]["n_miss_days"]
        assert miss_diff["delta"] == -2  # variant has fewer misses
        assert miss_diff["baseline"] == 5
        assert miss_diff["variant"] == 3

    def test_streak_metrics_diff(self):
        baseline = self._make_scorecard()
        variant = self._make_scorecard()
        variant["streak_metrics"]["p_57_monte_carlo"] = 0.07
        result = diff_scorecards(baseline, variant)
        assert "streak_metrics" in result
        p57_diff = result["streak_metrics"]["p_57_monte_carlo"]
        assert abs(p57_diff["delta"] - 0.02) < 1e-9

    def test_p57_exact_diff(self):
        baseline = self._make_scorecard()
        variant = self._make_scorecard()
        variant["p_57_exact"] = 0.07
        result = diff_scorecards(baseline, variant)
        assert "p_57_exact" in result
        p57_diff = result["p_57_exact"]
        assert "delta" in p57_diff
        assert abs(p57_diff["delta"] - 0.01) < 1e-9

    def test_p57_mdp_diff(self):
        baseline = self._make_scorecard()
        variant = self._make_scorecard()
        variant["p_57_mdp"] = 0.08
        result = diff_scorecards(baseline, variant)
        assert "p_57_mdp" in result
        assert abs(result["p_57_mdp"]["delta"] - 0.015) < 1e-9

    def test_p_at_1_by_season_diff(self):
        baseline = self._make_scorecard()
        variant = self._make_scorecard()
        variant["p_at_1_by_season"][2025] = 0.85
        result = diff_scorecards(baseline, variant)
        assert "p_at_1_by_season" in result
        s2025_diff = result["p_at_1_by_season"][2025]
        assert abs(s2025_diff["delta"] - 0.05) < 1e-9

    def test_none_values_skipped(self):
        """Fields that are None in baseline or variant should be skipped gracefully."""
        baseline = self._make_scorecard()
        baseline["p_57_mdp"] = None
        variant = self._make_scorecard()
        result = diff_scorecards(baseline, variant)
        # Should not raise; p_57_mdp diff should be absent or handled
        # (either skipped or shows None → float)
        assert isinstance(result, dict)

    def test_no_mutation(self):
        """diff_scorecards should not mutate its inputs."""
        baseline = self._make_scorecard(p1_val=0.80)
        variant = self._make_scorecard(p1_val=0.82)
        import copy
        baseline_copy = copy.deepcopy(baseline)
        variant_copy = copy.deepcopy(variant)
        diff_scorecards(baseline, variant)
        assert baseline == baseline_copy
        assert variant == variant_copy
