"""_refresh_season_data runs the pull+rebuild once per (season, yesterday) — later
intraday cascades skip the ~12-minute no-op re-pull (2026-08-30)."""
from unittest.mock import patch

import pandas as pd


def _run(date, tmp_path):
    from bts.model.predict import _refresh_season_data
    _refresh_season_data(date, raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "proc"))


def _fake_build(raw_dir, output_path, season):
    """Stand-in for build_season that, like the real one, writes the parquet.

    The memo skip is gated on the parquet existing (a marker alone must never
    hide a missing season file), so the fake has to leave the file behind.
    """
    output_path.write_bytes(b"parquet")
    return pd.DataFrame({"a": [1]})


def test_second_call_same_day_skips(tmp_path, monkeypatch):
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    (tmp_path / "raw" / "2026").mkdir(parents=True)
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", side_effect=_fake_build) as build:
        (tmp_path / "proc").mkdir()
        _run("2026-08-30", tmp_path)
        _run("2026-08-30", tmp_path)
    assert pull.call_count == 1 and build.call_count == 1
    assert (tmp_path / "proc" / ".refreshed_2026_through_2026-08-29").exists()


def test_next_day_refreshes_again(tmp_path, monkeypatch):
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    (tmp_path / "raw" / "2026").mkdir(parents=True); (tmp_path / "proc").mkdir()
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", side_effect=_fake_build):
        _run("2026-08-30", tmp_path)
        _run("2026-08-31", tmp_path)
    assert pull.call_count == 2


def test_force_env_refreshes(tmp_path, monkeypatch):
    monkeypatch.setenv("BTS_REFRESH_ALWAYS", "1")
    (tmp_path / "raw" / "2026").mkdir(parents=True); (tmp_path / "proc").mkdir()
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", side_effect=_fake_build):
        _run("2026-08-30", tmp_path); _run("2026-08-30", tmp_path)
    assert pull.call_count == 2


def test_failed_build_does_not_write_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    (tmp_path / "raw" / "2026").mkdir(parents=True); (tmp_path / "proc").mkdir()
    with patch("bts.data.pull.pull_feeds", return_value=[]), \
         patch("bts.data.build.build_season", side_effect=RuntimeError("boom")):
        try:
            _run("2026-08-30", tmp_path)
        except RuntimeError:
            pass
    assert not (tmp_path / "proc" / ".refreshed_2026_through_2026-08-29").exists()


def test_marker_without_parquet_still_refreshes(tmp_path, monkeypatch):
    """A stale marker must not mask a missing season parquet (e.g. deleted
    between cascades): run_pipeline loads pa_*.parquet right after this."""
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    (tmp_path / "raw" / "2026").mkdir(parents=True); (tmp_path / "proc").mkdir()
    (tmp_path / "proc" / ".refreshed_2026_through_2026-08-29").write_text("stale")
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", side_effect=_fake_build):
        _run("2026-08-30", tmp_path)
    assert pull.call_count == 1
