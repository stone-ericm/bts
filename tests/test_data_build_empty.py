"""The real nightly CLI must pass an unavailable pre-season without publishing data."""
import json

import pandas as pd
import pytest
from click.testing import CliRunner

from bts.cli import cli


def _run(tmp_path, seasons="2027"):
    return CliRunner().invoke(cli, ["data", "build", "--seasons", seasons,
        "--raw-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "processed")])


def _feed(tmp_path, feed, season=2027):
    directory = tmp_path / "raw" / str(season)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "999999.json").write_text(json.dumps(feed))


@pytest.mark.parametrize("kind", ["missing", "empty", "non_regular", "excluded_doubleheader"])
def test_cli_empty_season_is_success_without_writes(tmp_path, sample_game_feed, kind):
    if kind == "empty":
        (tmp_path / "raw" / "2027").mkdir(parents=True)
    elif kind in ("non_regular", "excluded_doubleheader"):
        if kind == "non_regular":
            sample_game_feed["gameData"]["game"]["type"] = "S"
        else:
            sample_game_feed["gameData"]["game"]["doubleHeader"] = "Y"
        _feed(tmp_path, sample_game_feed)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    res = _run(tmp_path)
    assert res.exit_code == 0, res.output
    assert res.stderr.splitlines() == ["No regular-season games available for season 2027; nothing written."]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_cli_empty_season_preserves_existing_parquet(tmp_path):
    target = tmp_path / "processed" / "pa_2027.parquet"
    target.parent.mkdir()
    pd.DataFrame({"sentinel": [1]}).to_parquet(target)
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    res = _run(tmp_path)
    assert res.exit_code == 0, res.output
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


@pytest.mark.parametrize("seasons", ["2026", "2027,2026", "2026,2027"])
def test_cli_builds_available_seasons_even_beside_an_empty_one(tmp_path, sample_game_feed, seasons):
    _feed(tmp_path, sample_game_feed, season=2026)
    res = _run(tmp_path, seasons)
    assert res.exit_code == 0, res.output
    assert len(pd.read_parquet(tmp_path / "processed" / "pa_2026.parquet")) == 2
    assert not (tmp_path / "processed" / "pa_2027.parquet").exists()


def test_cli_regular_season_build_error_still_fails(tmp_path, sample_game_feed):
    sample_game_feed["liveData"]["plays"]["allPlays"] = []
    _feed(tmp_path, sample_game_feed)
    res = _run(tmp_path)
    assert res.exit_code != 0
    assert isinstance(res.exception, RuntimeError) and "Schema drift" in str(res.exception)
    assert not (tmp_path / "processed").exists()
