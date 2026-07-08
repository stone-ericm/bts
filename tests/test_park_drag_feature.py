"""Tests for the park_drag_delta context feature (shadow stack).

Spec: docs/superpowers/specs/2026-07-07-park-drag-delta-context-feature.md
The external table is produced by ~/projects/juiced-ball-analysis (one row per
venue_id x calendar date, computed from strictly-prior games). BTS-side
requirements under test here:
  - loader validates schema/uniqueness and NEVER raises into the pick path
  - training merge and serving lookup read the IDENTICAL (venue_id, date) value
    (Codex #1: no off-by-one), doubleheader rows share one value (Codex #5)
  - stale table suppresses SERVING values but not historical training values
    (Codex #6)
  - CONTEXT_COLS gains park_drag_delta; shadow cache path carries a feature-set
    hash so a same-day cached 4-col model can't serve 5-col features (Codex #4)
  - predict() populates row["park_drag_delta"] (Codex #17 populator coverage)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bts.features import park_drag

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_table(tmp_path, rows, manifest_max_date=None):
    p = tmp_path / "park_drag_export.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    if manifest_max_date is not None:
        (tmp_path / "park_drag_manifest.json").write_text(
            json.dumps({"max_source_game_date": manifest_max_date})
        )
    return p


GOOD_ROWS = [
    {"venue_id": 1, "date": "2026-06-01", "park_drag_delta": -0.010,
     "park_drag_delta_expanding": -0.008, "park_drag_n_window": 15},
    {"venue_id": 1, "date": "2026-06-02", "park_drag_delta": -0.011,
     "park_drag_delta_expanding": -0.009, "park_drag_n_window": 15},
    {"venue_id": 2, "date": "2026-06-01", "park_drag_delta": 0.001,
     "park_drag_delta_expanding": 0.002, "park_drag_n_window": 12},
]


class TestLoader:
    def test_missing_file_returns_none(self, tmp_path):
        assert park_drag.load_table(tmp_path / "nope.csv") is None

    def test_good_table_loads(self, tmp_path):
        p = _write_table(tmp_path, GOOD_ROWS)
        t = park_drag.load_table(p)
        assert t is not None and len(t) == 3
        assert pd.api.types.is_datetime64_any_dtype(t["date"])

    def test_schema_drift_returns_none(self, tmp_path):
        rows = [{"venue_id": 1, "date": "2026-06-01", "wrong_col": 0.1}]
        p = _write_table(tmp_path, rows)
        assert park_drag.load_table(p) is None

    def test_duplicate_key_returns_none(self, tmp_path):
        p = _write_table(tmp_path, GOOD_ROWS + [GOOD_ROWS[0]])
        assert park_drag.load_table(p) is None

    def test_env_var_override(self, tmp_path, monkeypatch):
        p = _write_table(tmp_path, GOOD_ROWS)
        monkeypatch.setenv(park_drag.ENV_VAR, str(p))
        t = park_drag.load_table()
        assert t is not None and len(t) == 3


class TestAttach:
    def _df(self):
        return pd.DataFrame([
            {"venue_id": 1, "date": pd.Timestamp("2026-06-01"), "batter_id": 10},
            {"venue_id": 1, "date": pd.Timestamp("2026-06-01"), "batter_id": 11},  # DH game 2 row
            {"venue_id": 2, "date": pd.Timestamp("2026-06-01"), "batter_id": 12},
            {"venue_id": 9, "date": pd.Timestamp("2026-06-01"), "batter_id": 13},  # unknown venue
            {"venue_id": 1, "date": pd.Timestamp("2026-06-03"), "batter_id": 14},  # unknown date
        ])

    def test_merges_by_venue_and_date(self, tmp_path):
        table = park_drag.load_table(_write_table(tmp_path, GOOD_ROWS))
        out = park_drag.attach_park_drag(self._df(), table=table)
        assert out.loc[0, "park_drag_delta"] == pytest.approx(-0.010)
        assert out.loc[2, "park_drag_delta"] == pytest.approx(0.001)
        assert np.isnan(out.loc[3, "park_drag_delta"])
        assert np.isnan(out.loc[4, "park_drag_delta"])

    def test_doubleheader_rows_share_one_value(self, tmp_path):
        table = park_drag.load_table(_write_table(tmp_path, GOOD_ROWS))
        out = park_drag.attach_park_drag(self._df(), table=table)
        assert out.loc[0, "park_drag_delta"] == out.loc[1, "park_drag_delta"]

    def test_none_table_gives_all_nan_column(self):
        out = park_drag.attach_park_drag(self._df(), table=None)
        assert "park_drag_delta" in out.columns
        assert out["park_drag_delta"].isna().all()

    def test_attach_never_raises(self, monkeypatch):
        monkeypatch.setattr(park_drag, "load_table",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        out = park_drag.attach_park_drag(self._df())  # loads internally -> explodes -> NaN
        assert out["park_drag_delta"].isna().all()

    def test_row_count_and_order_preserved(self, tmp_path):
        table = park_drag.load_table(_write_table(tmp_path, GOOD_ROWS))
        df = self._df()
        out = park_drag.attach_park_drag(df, table=table)
        assert len(out) == len(df)
        assert (out["batter_id"].values == df["batter_id"].values).all()


class TestServingParity:
    def test_serving_equals_training_merge(self, tmp_path):
        """Codex #1: the serving lookup must read the identical value the
        training-time merge would attach for the same (venue_id, date)."""
        p = _write_table(tmp_path, GOOD_ROWS, manifest_max_date="2026-06-01")
        table = park_drag.load_table(p)
        manifest = park_drag.load_manifest(p)
        train_df = pd.DataFrame(
            [{"venue_id": 1, "date": pd.Timestamp("2026-06-02")}])
        merged = park_drag.attach_park_drag(train_df, table=table)
        served = park_drag.serving_value(
            table, manifest, venue_id=1, on_date=pd.Timestamp("2026-06-02"))
        assert served == pytest.approx(merged.loc[0, "park_drag_delta"])

    def test_serving_missing_row_returns_none(self, tmp_path):
        p = _write_table(tmp_path, GOOD_ROWS, manifest_max_date="2026-06-01")
        table = park_drag.load_table(p)
        manifest = park_drag.load_manifest(p)
        assert park_drag.serving_value(
            table, manifest, venue_id=99, on_date=pd.Timestamp("2026-06-01")) is None

    def test_stale_table_suppresses_serving(self, tmp_path):
        """Codex #6: serving far past the table's source data must yield None
        (never a silently-stale number); training-history merges unaffected."""
        p = _write_table(tmp_path, GOOD_ROWS, manifest_max_date="2026-06-01")
        table = park_drag.load_table(p)
        manifest = park_drag.load_manifest(p)
        late = pd.Timestamp("2026-06-02") + pd.Timedelta(days=park_drag.STALE_AFTER_DAYS)
        assert park_drag.serving_value(table, manifest, 1, late) is None
        # training merge for a historical date still works
        hist = park_drag.attach_park_drag(
            pd.DataFrame([{"venue_id": 1, "date": pd.Timestamp("2026-06-01")}]),
            table=table)
        assert hist.loc[0, "park_drag_delta"] == pytest.approx(-0.010)

    def test_none_table_serving_returns_none(self):
        assert park_drag.serving_value(None, None, 1, pd.Timestamp("2026-06-01")) is None


class TestContextColsIntegration:
    def test_context_cols_gained_park_drag_delta(self):
        from bts.features.compute import CONTEXT_COLS
        assert "park_drag_delta" in CONTEXT_COLS
        assert len(CONTEXT_COLS) == 5

    def test_compute_all_features_has_column_without_table(self, monkeypatch):
        """No external table configured -> column exists, all-NaN, no crash
        (Codex #3: the production pick path must survive a missing artifact)."""
        monkeypatch.setenv(park_drag.ENV_VAR, "/nonexistent/park_drag.csv")
        park_drag._reset_cache()
        from tests.test_context_features import _make_pa_df
        from bts.features.compute import compute_all_features
        out = compute_all_features(_make_pa_df(20))
        assert "park_drag_delta" in out.columns
        assert out["park_drag_delta"].isna().all()

    def test_compute_all_features_merges_configured_table(self, tmp_path, monkeypatch):
        from tests.test_context_features import _make_pa_df
        from bts.features.compute import compute_all_features
        pa = _make_pa_df(20)  # venue_id=1, dates 2024-06-01..05
        rows = [{"venue_id": 1, "date": d.strftime("%Y-%m-%d"),
                 "park_drag_delta": -0.007, "park_drag_delta_expanding": -0.006,
                 "park_drag_n_window": 15}
                for d in pd.date_range("2024-06-01", periods=5)]
        p = _write_table(tmp_path, rows)
        monkeypatch.setenv(park_drag.ENV_VAR, str(p))
        park_drag._reset_cache()
        out = compute_all_features(pa)
        assert out["park_drag_delta"].notna().all()
        assert out["park_drag_delta"].unique() == pytest.approx([-0.007])
        park_drag._reset_cache()


class TestShadowCacheIdentity:
    def test_cache_path_includes_feature_hash(self, tmp_path):
        from bts.orchestrator import shadow_cache_path
        p = shadow_cache_path(tmp_path, "2026-07-07")
        assert p.name.startswith("blend_2026-07-07_shadow_")
        assert p.suffix == ".pkl"
        assert p.name != "blend_2026-07-07_shadow_.pkl"

    def test_cache_path_changes_with_feature_set(self, tmp_path, monkeypatch):
        """Codex #4: a cached shadow model trained on a different context set
        must not be picked up — the hash in the filename is the guard."""
        from bts import orchestrator
        import bts.features.compute as compute
        before = orchestrator.shadow_cache_path(tmp_path, "2026-07-07")
        monkeypatch.setattr(compute, "CONTEXT_COLS", compute.CONTEXT_COLS[:-1])
        after = orchestrator.shadow_cache_path(tmp_path, "2026-07-07")
        assert before != after

    def test_shadow_model_name_bumped(self):
        from bts.shadow_eval import SHADOW_MODEL_NAME
        assert SHADOW_MODEL_NAME == "context_stack_shadow_v2"


class TestPredictPopulatesContextCols:
    def test_predict_source_populates_every_context_col(self):
        """Codex #17: mirror of the FEATURE_COLS populator scan for CONTEXT_COLS."""
        from bts.features.compute import CONTEXT_COLS
        predict_src = (_REPO_ROOT / "src/bts/model/predict.py").read_text()
        pred_start = predict_src.find("def predict(")
        assert pred_start > 0
        pred_end = predict_src.find("\ndef ", pred_start + 1)
        if pred_end == -1:
            pred_end = len(predict_src)
        body = predict_src[pred_start:pred_end]
        missing = [c for c in CONTEXT_COLS
                   if f'row["{c}"]' not in body and f"row['{c}']" not in body
                   and c not in body]
        assert not missing, f"CONTEXT_COLS {missing} not populated in predict()"
