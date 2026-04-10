"""Tests for shadow model pipeline (feature_cols_override)."""

import pytest
from unittest.mock import patch, MagicMock

from bts.features.compute import FEATURE_COLS, CONTEXT_COLS
from bts.model.predict import _build_blend_configs


class TestBuildBlendConfigs:
    def test_default_uses_feature_cols(self):
        configs = _build_blend_configs()
        base_name, base_cols = configs[0]
        assert base_name == "baseline"
        assert base_cols == FEATURE_COLS

    def test_override_replaces_base(self):
        override = FEATURE_COLS + CONTEXT_COLS
        configs = _build_blend_configs(base_feature_cols=override)
        base_name, base_cols = configs[0]
        assert base_name == "baseline"
        assert base_cols == override

    def test_override_preserves_statcast_boltons(self):
        override = FEATURE_COLS + CONTEXT_COLS
        configs = _build_blend_configs(base_feature_cols=override)
        barrel_name, barrel_cols = configs[1]
        assert barrel_name == "barrel"
        assert "batter_barrel_rate_30g" in barrel_cols
        for col in CONTEXT_COLS:
            assert col in barrel_cols

    def test_override_keeps_12_configs(self):
        override = FEATURE_COLS + CONTEXT_COLS
        configs = _build_blend_configs(base_feature_cols=override)
        assert len(configs) == 12
