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


def test_no_regular_games_yet_returns_without_marker_and_retries(tmp_path, monkeypatch, sample_game_feed, capsys):
    import json
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    directory = tmp_path / "raw" / "2027"
    directory.mkdir(parents=True)
    sample_game_feed["gameData"]["game"]["type"] = "S"
    (directory / "spring.json").write_text(json.dumps(sample_game_feed))
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull:
        assert _run("2027-03-24", tmp_path) is None
        assert _run("2027-03-24", tmp_path) is None
    assert pull.call_count == 2
    assert not list((tmp_path / "proc").glob(".refreshed_*"))
    assert not (tmp_path / "proc" / "pa_2027.parquet").exists()
    assert capsys.readouterr().err.splitlines().count(
        "  Season 2027 has no regular-season games yet; using historical data.") == 2


def test_none_build_return_does_not_memoize_even_with_existing_parquet(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    proc = tmp_path / "proc"
    proc.mkdir()
    existing = proc / "pa_2027.parquet"
    existing.write_bytes(b"retained parquet")
    before = (existing.read_bytes(), existing.stat().st_mtime_ns)
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull, \
         patch("bts.data.build.build_season", return_value=None):
        _run("2027-03-24", tmp_path)
        _run("2027-03-24", tmp_path)
    assert pull.call_count == 2
    assert not list(proc.glob(".refreshed_*"))
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == before
    assert "no regular-season games yet" in capsys.readouterr().err


def _cold_start_pipeline(tmp_path, monkeypatch, sample_game_feed):
    """Real feed build, historical parquet reads and features; no live transport or model training."""
    import json
    import bts.model.predict as model
    from bts.data.build import build_season
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    raw = tmp_path / "data" / "raw"
    proc = tmp_path / "data" / "processed"
    for year in (2025, 2026):
        directory = raw / str(year)
        directory.mkdir(parents=True)
        sample_game_feed["gameData"]["game"]["season"] = str(year)
        sample_game_feed["gameData"]["datetime"]["officialDate"] = f"{year}-06-15"
        (directory / "999999.json").write_text(json.dumps(sample_game_feed))
        build_season(raw, proc / f"pa_{year}.parquet", year)
    spring = raw / "2027"
    spring.mkdir()
    sample_game_feed["gameData"]["game"]["type"] = "S"
    (spring / "888888.json").write_text(json.dumps(sample_game_feed))
    seen = []
    compute = model.compute_all_features

    def features(frame):
        seen.append(frame.copy())
        assert set(frame["season"]) == {2025, 2026}
        assert len(frame) == 4
        return compute(frame)

    monkeypatch.setattr("bts.data.pull.pull_feeds", lambda *a, **k: [])
    monkeypatch.setattr(model, "compute_all_features", features)
    monkeypatch.setattr(model, "train_model", lambda *a, **k: object())
    monkeypatch.setattr(model, "train_blend", lambda *a, **k: {})
    monkeypatch.setattr(model, "_build_feature_lookups", lambda *a, **k: {})
    monkeypatch.setattr(model, "predict", lambda *a, **k: pd.DataFrame())
    return proc, seen


def test_pipeline_loads_historical_parquets_and_computes_features_without_current_season(
        tmp_path, monkeypatch, sample_game_feed):
    from bts.model.predict import run_pipeline
    proc, seen = _cold_start_pipeline(tmp_path, monkeypatch, sample_game_feed)
    predictions = run_pipeline("2027-03-24", data_dir=str(proc))
    assert predictions.empty and len(seen) == 1
    assert not list(proc.glob(".refreshed_*"))
    assert not (proc / "pa_2027.parquet").exists()


def test_preview_cli_reaches_historical_features_without_failed_line(tmp_path, monkeypatch, sample_game_feed):
    from click.testing import CliRunner
    from bts.cli import cli
    proc, seen = _cold_start_pipeline(tmp_path, monkeypatch, sample_game_feed)
    res = CliRunner().invoke(cli, ["preview", "--date", "2027-03-24", "--data-dir", str(proc)])
    assert res.exit_code == 0, res.output
    assert "Failed:" not in res.output, res.output
    assert "[preview] No games found for 2027-03-24." in res.output
    assert len(seen) == 1
    assert not list(proc.glob(".refreshed_*"))


def test_regular_season_with_one_bad_cached_feed_refreshes_and_writes_marker(
        tmp_path, monkeypatch, sample_game_feed, capsys):
    import json
    monkeypatch.delenv("BTS_REFRESH_ALWAYS", raising=False)
    directory = tmp_path / "raw" / "2027"
    directory.mkdir(parents=True)
    sample_game_feed["gameData"]["game"]["season"] = "2027"
    sample_game_feed["gameData"]["datetime"]["officialDate"] = "2027-06-15"
    (directory / "999999.json").write_text(json.dumps(sample_game_feed))
    bad = directory / "000000.json"
    bad.write_text("{}")
    with patch("bts.data.pull.pull_feeds", return_value=[]) as pull:
        _run("2027-06-16", tmp_path)
    assert pull.call_count == 1
    assert len(pd.read_parquet(tmp_path / "proc" / "pa_2027.parquet")) == 2
    assert (tmp_path / "proc" / ".refreshed_2027_through_2027-06-15").exists()
    assert bad.read_text() == "{}"  # The bad cached body remains, but cannot poison subsequent predictions.
    assert capsys.readouterr().err.splitlines().count(
        f"Skipping {bad}: missing or unreadable gameData.game.type.") == 1
