"""Tests for blend walk-forward backtest output."""

import pandas as pd
import pytest
import numpy as np


class TestBlendBacktestOutput:
    def test_output_schema(self):
        """Verify the output parquet has the expected columns."""
        from bts.simulate.backtest_blend import PROFILE_COLUMNS
        assert PROFILE_COLUMNS == [
            "date", "rank", "batter_id", "game_pk",
            "p_game_hit", "actual_hit", "n_pas",
        ]

    def test_load_saved_profiles(self, tmp_path):
        """Round-trip: save profiles, load them back."""
        from bts.simulate.backtest_blend import save_profiles
        from bts.simulate.monte_carlo import load_all_profiles

        df = pd.DataFrame({
            "date": ["2024-04-01"] * 10,
            "rank": list(range(1, 11)),
            "batter_id": [i * 1000 for i in range(1, 11)],
            "game_pk": [1000 + i for i in range(10)],
            "p_game_hit": [0.90 - i * 0.02 for i in range(10)],
            "actual_hit": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
            "n_pas": [4] * 10,
        })
        save_profiles(df, 2024, tmp_path)
        loaded = load_all_profiles(tmp_path)
        assert len(loaded) == 1  # 1 day
        assert loaded[0].top1_p == df.iloc[0]["p_game_hit"]


class TestDecisionSensitivityWeights:
    def test_weights_prioritize_top_daily_candidates_and_normalize(self):
        from bts.simulate.backtest_blend import _decision_sensitivity_sample_weights

        available = pd.DataFrame({
            "date": ["2024-04-01"] * 6,
            "batter_id": [1, 1, 2, 2, 3, 3],
            "game_pk": [10, 10, 10, 10, 10, 10],
        })
        pa_probs = np.array([0.30, 0.30, 0.45, 0.45, 0.10, 0.10])

        weights = _decision_sensitivity_sample_weights(
            available,
            pa_probs,
            top_n=2,
            alpha=2.0,
            rank_scale=3.0,
        )

        assert len(weights) == len(available)
        assert np.isclose(weights.mean(), 1.0)
        assert np.all(np.isfinite(weights))
        # Batter 2 has the highest estimated game-hit probability and should
        # receive more weight than the low-probability non-top candidate.
        assert weights[2] > weights[4]
        assert weights[3] > weights[5]

    def test_pop_decision_weight_params_strips_non_lgbm_keys(self):
        from bts.simulate.backtest_blend import _pop_decision_weight_params

        lgb_params, weight_params = _pop_decision_weight_params({
            "n_estimators": 200,
            "decision_weight_mode": "top_slate_v0",
            "decision_weight_top_n": 7,
            "engine": "ignored",
        })

        assert lgb_params == {"n_estimators": 200}
        assert weight_params["top_n"] == 7

    def test_train_lgbm_classifier_uses_decision_sample_weight(self, monkeypatch):
        import lightgbm as lgb
        from bts.simulate.backtest_blend import _train_lgbm_classifier

        fit_sample_weights = []

        class FakeClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, X, y, sample_weight=None):
                fit_sample_weights.append(sample_weight)
                return self

            def predict_proba(self, X):
                probs = np.linspace(0.15, 0.85, len(X))
                return np.column_stack([1.0 - probs, probs])

        monkeypatch.setattr(lgb, "LGBMClassifier", FakeClassifier)
        available = pd.DataFrame({
            "date": ["2024-04-01"] * 6,
            "batter_id": [1, 1, 2, 2, 3, 3],
            "game_pk": [10, 10, 10, 10, 10, 10],
            "feature": [0.1, 0.2, 0.5, 0.6, 0.9, 1.0],
            "is_hit": [0, 1, 1, 1, 0, 0],
        })

        _train_lgbm_classifier(
            available,
            ["feature"],
            {
                "n_estimators": 1,
                "decision_weight_mode": "top_slate_v0",
            },
        )

        assert len(fit_sample_weights) == 2
        assert fit_sample_weights[0] is None  # probe model
        assert fit_sample_weights[1] is not None  # final weighted model
        assert np.isclose(fit_sample_weights[1].mean(), 1.0)

    def test_weighted_config_does_not_reuse_plain_cached_model(self, monkeypatch):
        import bts.simulate.backtest_blend as bb

        trained = object()

        def fake_train(available, cols, merged_params):
            return trained

        monkeypatch.setattr(bb, "_train_lgbm_classifier", fake_train)
        available = pd.DataFrame({
            "feature": [1.0],
            "is_hit": [1],
        })
        cached_model = ("cached", ["feature"], bb._predict_lgbm_classifier)

        blend, _ = bb._train_blend_for_day(
            available,
            [("baseline", ["feature"], {"decision_weight_mode": "top_slate_v0"})],
            {},
            cached_models={"baseline": cached_model},
        )

        assert blend["baseline"][0] is trained

    def test_plain_config_reuses_cached_model(self, monkeypatch):
        import bts.simulate.backtest_blend as bb

        def fail_train(*args, **kwargs):
            raise AssertionError("plain cached config should not retrain")

        monkeypatch.setattr(bb, "_train_lgbm_classifier", fail_train)
        available = pd.DataFrame({
            "feature": [1.0],
            "is_hit": [1],
        })
        cached_model = ("cached", ["feature"], bb._predict_lgbm_classifier)

        blend, _ = bb._train_blend_for_day(
            available,
            [("baseline", ["feature"])],
            {},
            cached_models={"baseline": cached_model},
        )

        assert blend["baseline"] == cached_model
