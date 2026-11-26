"""Tests for build_season runtime schema assertion."""
import json
from pathlib import Path

import pandas as pd
import pytest

from bts.data.schema import PA_COLUMNS
from bts.data.build import assert_columns_match_schema


def test_assert_passes_when_columns_match():
    df = pd.DataFrame({col: [] for col in PA_COLUMNS})
    # Should not raise
    assert_columns_match_schema(df)


def test_assert_fails_when_column_missing():
    cols = [c for c in PA_COLUMNS if c != "is_hit"]
    df = pd.DataFrame({col: [] for col in cols})
    with pytest.raises(RuntimeError, match=r"missing columns.*is_hit"):
        assert_columns_match_schema(df)


def test_assert_fails_when_extra_column():
    cols = PA_COLUMNS + ["unexpected_col"]
    df = pd.DataFrame({col: [] for col in cols})
    with pytest.raises(RuntimeError, match=r"extra columns.*unexpected_col"):
        assert_columns_match_schema(df)


def test_assert_error_message_includes_pa_columns_hint():
    df = pd.DataFrame({col: [] for col in PA_COLUMNS[:-2]})
    with pytest.raises(RuntimeError, match=r"Update PA_COLUMNS"):
        assert_columns_match_schema(df)
