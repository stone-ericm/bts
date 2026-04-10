"""Tests for bts state verify."""
import json
from pathlib import Path

import pytest

from bts.state.verify import diff_pick_files, DriftReport


def test_diff_identical_dirs_returns_no_drift(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    pick = {"date": "2026-04-01", "pick": {"batter_name": "X"}, "result": "hit"}
    (a / "2026-04-01.json").write_text(json.dumps(pick))
    (b / "2026-04-01.json").write_text(json.dumps(pick))
    (a / "streak.json").write_text('{"streak": 1, "saver_available": true}')
    (b / "streak.json").write_text('{"streak": 1, "saver_available": true}')

    report = diff_pick_files(a, b)
    assert report.is_clean


def test_diff_streak_mismatch_reported(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "streak.json").write_text('{"streak": 5, "saver_available": true}')
    (b / "streak.json").write_text('{"streak": 3, "saver_available": true}')

    report = diff_pick_files(a, b)
    assert not report.is_clean
    assert any("streak" in issue for issue in report.issues)


def test_diff_saver_available_mismatch_reported(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "streak.json").write_text('{"streak": 5, "saver_available": true}')
    (b / "streak.json").write_text('{"streak": 5, "saver_available": false}')

    report = diff_pick_files(a, b)
    assert not report.is_clean
    assert any("saver_available" in issue for issue in report.issues)


def test_diff_missing_pick_file_reported(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "2026-04-01.json").write_text('{"date": "2026-04-01"}')
    (a / "streak.json").write_text('{"streak": 1, "saver_available": true}')
    (b / "streak.json").write_text('{"streak": 1, "saver_available": true}')

    report = diff_pick_files(a, b)
    assert not report.is_clean
    assert any("2026-04-01" in issue for issue in report.issues)
