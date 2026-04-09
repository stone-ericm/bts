import numpy as np
import pandas as pd

from bts.experiment.features import (
    EBShrinkageExperiment,
    KLDivergenceExperiment,
    BattingHeatQExperiment,
    GBPlatoonExperiment,
    HitTypeParkFactorsExperiment,
    StreakLengthFeatureExperiment,
)
from bts.features.compute import FEATURE_COLS


def test_eb_shrinkage_replaces_rolling(mini_pa_df):
    exp = EBShrinkageExperiment()
    original = mini_pa_df["batter_hr_7g"].copy()
    result = exp.modify_features(mini_pa_df.copy())
    for col in ["batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g"]:
        assert col in result.columns
    # Values should differ from original (shrunk toward population mean)
    assert not np.allclose(
        result["batter_hr_7g"].dropna().values,
        original.dropna().values,
    )


def test_kl_divergence_adds_column(mini_pa_df):
    exp = KLDivergenceExperiment()
    df = mini_pa_df.copy()
    df["pitch_type"] = np.random.choice(["FF", "SL", "CH", "CU"], size=len(df))
    result = exp.modify_features(df)
    assert "pitcher_batter_fr_distance" in result.columns
    cols = exp.feature_cols()
    assert "pitcher_batter_fr_distance" in cols
    assert "pitcher_entropy_30g" not in cols


def test_kl_divergence_without_pitch_type(mini_pa_df):
    exp = KLDivergenceExperiment()
    result = exp.modify_features(mini_pa_df.copy())
    assert "pitcher_batter_fr_distance" in result.columns
    assert result["pitcher_batter_fr_distance"].isna().all()


def test_batting_heat_q_adds_feature(mini_pa_df):
    exp = BattingHeatQExperiment()
    result = exp.modify_features(mini_pa_df.copy())
    assert "batting_heat_q" in result.columns
    assert "batting_heat_q" in exp.feature_cols()
    assert not result["batting_heat_q"].isna().all()


def test_gb_platoon_adds_feature(mini_pa_df):
    exp = GBPlatoonExperiment()
    df = mini_pa_df.copy()
    df["batted_ball_type"] = np.random.choice(["GB", "FB", "LD", "PU"], size=len(df))
    result = exp.modify_features(df)
    assert "gb_platoon_rate" in result.columns


def test_hit_type_park_factors(mini_pa_df):
    exp = HitTypeParkFactorsExperiment()
    result = exp.modify_features(mini_pa_df.copy())
    for col in ["park_factor_1b", "park_factor_2b", "park_factor_3b"]:
        assert col in result.columns
    cols = exp.feature_cols()
    assert "park_factor" not in cols
    assert "park_factor_1b" in cols


def test_streak_length_feature(mini_pa_df):
    exp = StreakLengthFeatureExperiment()
    result = exp.modify_features(mini_pa_df.copy())
    assert "batter_streak_length" in result.columns
    assert "batter_streak_length" in exp.feature_cols()
    # All streak values should be non-negative integers
    assert (result["batter_streak_length"] >= 0).all()
