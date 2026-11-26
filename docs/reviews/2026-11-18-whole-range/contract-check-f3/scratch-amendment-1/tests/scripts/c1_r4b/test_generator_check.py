"""The one-block generator reproduction check (register row C1-4b-generator-commit, Eric's ruling (a) of 2026-10-05).
The script runs standalone inside the generator worktree; here only its pure helpers are exercised, loaded by path."""
import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[3] / "scripts" / "audit" / "c1_r4b" / "generator_check.py"
_spec = importlib.util.spec_from_file_location("generator_check", _PATH)
GC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GC)


def frame():
    return pd.DataFrame({"date": [date(2023, 9, 25)] * 2 + [date(2023, 9, 26)],
                         "rank": [1, 2, 1], "batter_id": [10, 11, 12], "game_pk": [1, 2, 3],
                         "p_game_hit": [0.812345678901, 0.79, np.nan], "actual_hit": [1, 0, 1]})


def test_identical_blocks_reproduce():
    out = GC.compare(frame(), frame())
    assert out["reproduced"] is True and out["rows"] == 3 and out["differing_cells"] == {}


def test_a_one_ulp_difference_fails_and_reports_no_values():
    mine = frame()
    mine.loc[0, "p_game_hit"] = np.nextafter(mine.loc[0, "p_game_hit"], 1.0)
    out = GC.compare(mine, frame())
    assert out["reproduced"] is False and out["differing_cells"] == {"p_game_hit": 1}
    text = json.dumps(out)
    assert "0.8123" not in text and "max_abs_diff" in out          # magnitude only, never a value


def test_a_label_mismatch_affects_only_the_overall_equality_verdict():
    """Code review r2 N1: no count, position, value or magnitude of the outcome label enters the result."""
    mine = frame()
    mine.loc[1, "actual_hit"] = 1
    out = GC.compare(mine, frame())
    assert out["reproduced"] is False and out["differing_cells"] == {}
    assert "actual_hit" not in out["max_abs_diff"] and "actual_hit" not in json.dumps(out)


def test_dtype_row_count_and_column_mismatches_fail():
    mine = frame()
    mine["rank"] = mine["rank"].astype("int32")
    assert GC.compare(mine, frame())["reproduced"] is False
    assert GC.compare(frame().iloc[:2], frame())["reproduced"] is False
    assert GC.compare(frame()[["date", "rank"]], frame())["reproduced"] is False


def test_nan_in_the_same_place_is_equal():
    assert GC.compare(frame(), frame())["reproduced"] is True


def test_prepare_env_clears_bts_vars_and_pins_determinism_only():
    """r2 N2: OMP_NUM_THREADS does not set LightGBM's sklearn thread count, so the check no longer claims to."""
    env = {"BTS_LGBM_RANDOM_STATE": "7", "BTS_ROOKIE_GATE_K": "0", "PATH": "/bin", "OMP_NUM_THREADS": "2"}
    removed = GC.prepare_env(env)
    assert removed == ["BTS_LGBM_RANDOM_STATE", "BTS_ROOKIE_GATE_K"]
    assert env == {"PATH": "/bin", "OMP_NUM_THREADS": "2", "BTS_LGBM_DETERMINISTIC": "1"}


def test_the_2026_file_is_truncated_before_the_run_and_cut_to_the_2023_schema():
    df = pd.DataFrame({"date": ["2026-06-09", "2026-06-10"], "a": [1, 2], "b": [3, 4], "is_resumed_portion": [False, False]})
    out = GC.cut_2026(df, ["b", "a"])
    assert list(out.columns) == ["b", "a"] and list(out["a"]) == [1]
    with pytest.raises(ValueError):
        GC.cut_2026(df, ["a", "missing"])


def test_importing_the_module_does_not_touch_the_environment(monkeypatch):
    monkeypatch.setenv("BTS_LGBM_RANDOM_STATE", "5")
    _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
    import os
    assert os.environ["BTS_LGBM_RANDOM_STATE"] == "5"


def block(dates=None, ranks=10):
    dates = dates or GC.BLOCK_DATES
    return pd.DataFrame({"date": [date.fromisoformat(d) for d in dates for _ in range(ranks)],
                         "rank": [r for _ in dates for r in range(1, ranks + 1)]})


def test_the_block_must_hold_the_seven_dates_with_ranks_one_to_ten():
    assert GC.block_coverage(block()) == []
    assert GC.block_coverage(block().iloc[:0]) != []                       # an empty block never passes
    assert GC.block_coverage(block(dates=GC.BLOCK_DATES[:6])) != []
    assert GC.block_coverage(block(ranks=9)) != []
    two = block()
    two.loc[3, "rank"] = 2
    assert GC.block_coverage(two) != []


def test_two_empty_frames_are_not_a_reproduction():
    """r2 N3: equality of empty selections must not read as success (status needs coverage on both sides)."""
    assert GC.status(GC.compare(block().iloc[:0], block().iloc[:0]), block().iloc[:0], block().iloc[:0])[0] == "refused"
    full = block()
    assert GC.status(GC.compare(full, full), full, full) == ("reproduced", [])
    other = full.copy()
    other.loc[0, "rank"] = 11
    assert GC.status(GC.compare(other, full), other, full)[0] == "refused"   # coverage fails before equality


def test_lock_versions_are_read_from_uv_lock():
    lock = """version = 1
[[package]]
name = "lightgbm"
version = "4.6.0"

[[package]]
name = "numpy"
version = "2.4.3"
"""
    assert GC.lock_versions(lock, ("lightgbm", "numpy", "pandas")) == {"lightgbm": "4.6.0", "numpy": "2.4.3", "pandas": None}


def test_version_mismatches_are_listed():
    assert GC.version_mismatches({"lightgbm": "4.6.0"}, {"lightgbm": "4.6.0"}) == []
    assert GC.version_mismatches({"lightgbm": "4.5.0"}, {"lightgbm": "4.6.0"}) == ["lightgbm: installed 4.5.0, lock 4.6.0"]
    assert GC.version_mismatches({"pandas": "3.0.1"}, {"pandas": None}) == ["pandas: installed 3.0.1, lock None"]
