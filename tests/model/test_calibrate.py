"""Tests for post-hoc isotonic calibration of p_game_hit.

Background: 2026-05-01 diagnostic showed +29.7pp overconfidence in 75-80%
bucket on 2026 production. This module's job is to learn the (predicted_p,
realized_hit) mapping from recent picks and remap predict-time output.

Test strategy:
- Synthetic picks_dir + pa_df with KNOWN overconfidence pattern
- Verify calibrator learns to map down
- Verify failsafe (None) when data is sparse
- Verify identity when calibrator is None
"""
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from bts.model.calibrate import (
    fit_calibrator_from_picks,
    apply_calibrator,
    apply_calibrator_series,
    _resolve_pick_outcomes,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MIN_N,
)


def _write_pick(picks_dir: Path, d: date, primary_p: float, primary_bid: int, dd_p: float | None = None, dd_bid: int | None = None, result: str = "hit"):
    body = {
        "date": d.isoformat(),
        "pick": {"batter_name": "X", "batter_id": primary_bid, "p_game_hit": primary_p, "team": "T1"},
        "result": result,
    }
    if dd_p is not None and dd_bid is not None:
        body["double_down"] = {"batter_name": "Y", "batter_id": dd_bid, "p_game_hit": dd_p, "team": "T2"}
    (picks_dir / f"{d.isoformat()}.json").write_text(json.dumps(body))


def _build_pa_frame(rows: list[dict]) -> pd.DataFrame:
    """Helper to build a minimal pa_df with the columns the resolver expects."""
    return pd.DataFrame(rows)


class TestResolvePickOutcomes:
    def test_empty_picks_dir_returns_empty(self, tmp_path):
        out = _resolve_pick_outcomes(tmp_path, pd.DataFrame(columns=["batter_id", "date", "is_hit"]), date(2026, 5, 1), 30)
        assert out == []

    def test_picks_outside_window_excluded(self, tmp_path):
        # 100 days ago — beyond 30d lookback
        _write_pick(tmp_path, date(2026, 1, 1), 0.75, 100, result="hit")
        pa = _build_pa_frame([{"batter_id": 100, "date": pd.Timestamp("2026-01-01"), "is_hit": 1}])
        out = _resolve_pick_outcomes(tmp_path, pa, date(2026, 5, 1), 30)
        assert out == []

    def test_unresolved_picks_excluded(self, tmp_path):
        _write_pick(tmp_path, date(2026, 4, 30), 0.75, 100, result=None)
        pa = _build_pa_frame([{"batter_id": 100, "date": pd.Timestamp("2026-04-30"), "is_hit": 1}])
        out = _resolve_pick_outcomes(tmp_path, pa, date(2026, 5, 1), 30)
        assert out == []

    def test_pick_joined_to_pa(self, tmp_path):
        _write_pick(tmp_path, date(2026, 4, 30), 0.75, 100, dd_p=0.70, dd_bid=200, result="hit")
        pa = _build_pa_frame([
            {"batter_id": 100, "date": pd.Timestamp("2026-04-30"), "is_hit": 1},
            {"batter_id": 200, "date": pd.Timestamp("2026-04-30"), "is_hit": 0},
        ])
        out = _resolve_pick_outcomes(tmp_path, pa, date(2026, 5, 1), 30)
        # Both picks resolved — primary p=0.75, hit=1; dd p=0.70, hit=0
        assert sorted(out) == [(0.70, 0), (0.75, 1)]

    def test_pick_missing_pa_skipped(self, tmp_path):
        # Batter 100 is in picks but NOT in pa_df (e.g., late data)
        _write_pick(tmp_path, date(2026, 4, 30), 0.75, 100, result="hit")
        pa = _build_pa_frame([{"batter_id": 999, "date": pd.Timestamp("2026-04-30"), "is_hit": 1}])
        out = _resolve_pick_outcomes(tmp_path, pa, date(2026, 5, 1), 30)
        assert out == []

    def test_multiple_pas_one_hit_counts_as_hit(self, tmp_path):
        # Batter had 4 PAs, one was a hit → day_hit = 1
        _write_pick(tmp_path, date(2026, 4, 30), 0.75, 100, result="hit")
        pa = _build_pa_frame([
            {"batter_id": 100, "date": pd.Timestamp("2026-04-30"), "is_hit": 0},
            {"batter_id": 100, "date": pd.Timestamp("2026-04-30"), "is_hit": 0},
            {"batter_id": 100, "date": pd.Timestamp("2026-04-30"), "is_hit": 1},
            {"batter_id": 100, "date": pd.Timestamp("2026-04-30"), "is_hit": 0},
        ])
        out = _resolve_pick_outcomes(tmp_path, pa, date(2026, 5, 1), 30)
        assert out == [(0.75, 1)]


class TestFitCalibrator:
    def test_returns_none_when_insufficient_data(self, tmp_path):
        # Only 5 picks — below default min_n=30
        for i in range(5):
            d = date(2026, 4, 26 - i)
            _write_pick(tmp_path, d, 0.75, 100 + i, result="hit")
        pa = _build_pa_frame([
            {"batter_id": 100 + i, "date": pd.Timestamp(f"2026-04-{26-i:02d}"), "is_hit": 1}
            for i in range(5)
        ])
        cal = fit_calibrator_from_picks(tmp_path, pa, date(2026, 5, 1))
        assert cal is None

    def test_learns_overconfidence_correction(self, tmp_path):
        """With 60 synthetic samples where high P miss often, calibrator should map p=0.78 down."""
        # 30 picks at p=0.78, only 12 actually hit → realized 40%, predicted 78% (gap +38pp)
        # 30 picks at p=0.65, 18 actually hit → realized 60%, predicted 65% (gap +5pp)
        rows = []
        pa_rows = []
        for i in range(30):
            d = date(2026, 4, 30) - timedelta(days=i % 25)  # within 30d window
            bid = 1000 + i
            hit = 1 if i < 12 else 0
            _write_pick(tmp_path, d, 0.78, bid, result="hit" if hit else "miss")
            pa_rows.append({"batter_id": bid, "date": pd.Timestamp(d), "is_hit": hit})
        # Use distinct dates for the second batch (and distinct bids) — overwrite picks dir entry per date
        # Actually each pick file is one date — so we need 30 distinct dates total spread across two p-values.
        # Restructure: 60 picks across 30 distinct dates, 2 picks per date (primary + dd).
        for f in tmp_path.glob("*.json"):
            f.unlink()
        rows = []
        pa_rows = []
        d0 = date(2026, 4, 6)  # 25 days before today=2026-05-01, within 30d window
        for i in range(30):
            d = d0 + timedelta(days=i)
            primary_bid = 1000 + i
            dd_bid = 2000 + i
            primary_hit = 1 if i < 12 else 0  # 40% hit rate at p=0.78
            dd_hit = 1 if i < 18 else 0       # 60% hit rate at p=0.65
            _write_pick(tmp_path, d, 0.78, primary_bid, dd_p=0.65, dd_bid=dd_bid,
                        result="hit" if primary_hit else "miss")
            pa_rows.append({"batter_id": primary_bid, "date": pd.Timestamp(d), "is_hit": primary_hit})
            pa_rows.append({"batter_id": dd_bid, "date": pd.Timestamp(d), "is_hit": dd_hit})
        pa = pd.DataFrame(pa_rows)
        cal = fit_calibrator_from_picks(tmp_path, pa, date(2026, 5, 1), lookback_days=30, min_n=30)
        assert cal is not None
        # Mapping should pull p=0.78 down toward ~0.40 and p=0.65 down toward ~0.60
        out_high = apply_calibrator(0.78, cal)
        out_mid = apply_calibrator(0.65, cal)
        assert out_high < 0.78, f"calibrator should pull 0.78 down, got {out_high}"
        # Direction: high should be more aggressively pulled down than mid
        assert (0.78 - out_high) >= (0.65 - out_mid) - 0.02, "high-P correction should be ≥ mid-P correction"

    def test_handles_corrupt_pick_file_gracefully(self, tmp_path):
        for i in range(35):
            d = date(2026, 4, 30) - timedelta(days=i)
            _write_pick(tmp_path, d, 0.70, 1000 + i, result="hit")
        # Add a corrupt file
        (tmp_path / "2026-04-15.json").write_text("not json{{{")
        pa = pd.DataFrame([
            {"batter_id": 1000 + i, "date": pd.Timestamp(date(2026, 4, 30) - timedelta(days=i)), "is_hit": 1}
            for i in range(35)
        ])
        # Should not crash, just skip the corrupt file
        cal = fit_calibrator_from_picks(tmp_path, pa, date(2026, 5, 1), lookback_days=35)
        # 35 valid picks - 1 overwritten by corrupt = 34 still valid → above min_n=30
        assert cal is not None


class TestApplyCalibrator:
    def test_none_calibrator_is_identity(self):
        assert apply_calibrator(0.75, None) == 0.75
        assert apply_calibrator(0.0, None) == 0.0
        assert apply_calibrator(1.0, None) == 1.0

    def test_apply_with_real_calibrator(self):
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        # Toy mapping: predicted=[0.5, 0.7, 0.9], realized=[0.4, 0.5, 0.55]
        cal.fit([0.5, 0.7, 0.9], [0.4, 0.5, 0.55])
        # 0.7 should map close to 0.5
        assert 0.4 <= apply_calibrator(0.7, cal) <= 0.6

    def test_series_apply_handles_nan(self):
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit([0.5, 0.9], [0.4, 0.6])
        s = pd.Series([0.6, float("nan"), 0.8])
        out = apply_calibrator_series(s, cal)
        assert pd.isna(out.iloc[1])
        # Non-nan elements should be in calibrated range
        assert out.iloc[0] >= 0
        assert out.iloc[2] >= 0

    def test_series_apply_with_none_calibrator_is_identity(self):
        s = pd.Series([0.5, 0.7, float("nan")])
        out = apply_calibrator_series(s, None)
        # Should be exactly the same Series (identity)
        pd.testing.assert_series_equal(out, s)
