"""The real nightly CLI must pass an unavailable pre-season without publishing data."""
import importlib.util
import json
from pathlib import Path
import subprocess

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


@pytest.mark.parametrize("kind", ["missing", "empty", "non_regular", "weather_only"])
def test_cli_empty_season_is_success_without_writes(tmp_path, sample_game_feed, kind):
    if kind == "empty":
        (tmp_path / "raw" / "2027").mkdir(parents=True)
    elif kind == "non_regular":
        sample_game_feed["gameData"]["game"]["type"] = "S"
        _feed(tmp_path, sample_game_feed)
        (tmp_path / "raw" / "2027" / "another.json").write_text(json.dumps(sample_game_feed))
    elif kind == "weather_only":
        directory = tmp_path / "raw" / "2027"
        directory.mkdir(parents=True)
        (directory / "999999_weather.json").write_text("not a game feed")
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


UNREADABLE_FEEDS = [
    {}, {"gameData": {"game": {"pk": 999999}}}, {"gameData": None},
    {"gameData": {"game": None}}, [],
    *[{"gameData": {"game": {"type": value}}} for value in (None, 7, {}, [], True, "", "   ")],
]


@pytest.mark.parametrize("feed", UNREADABLE_FEEDS)
@pytest.mark.parametrize("companion_type", [None, "S", "R"])
def test_cli_skips_unreadable_type_but_requires_a_regular_feed(tmp_path, sample_game_feed, feed, companion_type):
    _feed(tmp_path, feed)
    if companion_type is not None:
        sample_game_feed["gameData"]["game"]["type"] = companion_type
        (tmp_path / "raw" / "2027" / "companion.json").write_text(json.dumps(sample_game_feed))
    res = _run(tmp_path)
    assert res.stderr.splitlines() == [
        f"Skipping {tmp_path / 'raw' / '2027' / '999999.json'}: missing or unreadable gameData.game.type."]
    if companion_type == "R":
        assert res.exit_code == 0, res.output
        assert len(pd.read_parquet(tmp_path / "processed" / "pa_2027.parquet")) == 2
    else:
        assert res.exit_code != 0
        assert isinstance(res.exception, ValueError)
        assert "gameData.game.type" in str(res.exception) and "999999.json" in str(res.exception)
        assert not (tmp_path / "processed").exists()


@pytest.fixture(scope="module")
def parent_build(tmp_path_factory):
    """Import the actual main builder into scratch; do not maintain a copy of its behavior."""
    source = subprocess.run(["git", "show", "main:src/bts/data/build.py"],
                            cwd=Path(__file__).resolve().parents[1],
                            check=True, capture_output=True, text=True).stdout
    path = tmp_path_factory.mktemp("g2-parent-build") / "parent_build.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("g2_parent_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_season


@pytest.mark.parametrize("bad_feed", [{}, {"gameData": {"game": {"pk": 123}}}])
@pytest.mark.parametrize("companion_type", ["S", "R"])
@pytest.mark.parametrize("bad_name", ["000000.json", "zzzzzz.json"])
def test_cli_typeless_feed_preserves_parent_rows_with_regular_feed(
        tmp_path, sample_game_feed, parent_build, companion_type, bad_name, bad_feed):
    sample_game_feed["gameData"]["game"]["type"] = companion_type
    _feed(tmp_path, sample_game_feed)
    (tmp_path / "raw" / "2027" / bad_name).write_text(json.dumps(bad_feed))
    target = tmp_path / "processed" / "pa_2027.parquet"
    target.parent.mkdir()
    target.write_bytes(b"existing parquet")
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    res = _run(tmp_path)
    assert bad_name in res.stderr and res.stderr.count("Skipping ") == 1
    if companion_type == "R":
        assert res.exit_code == 0, res.output
        parent_output = tmp_path / "parent.parquet"
        parent_build(tmp_path / "raw", parent_output, 2027)
        pd.testing.assert_frame_equal(pd.read_parquet(target), pd.read_parquet(parent_output))
    else:
        assert res.exit_code != 0
        assert isinstance(res.exception, ValueError) and bad_name in str(res.exception)
        assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_cli_without_regular_feeds_names_every_unreadable_file(tmp_path):
    _feed(tmp_path, {})
    (tmp_path / "raw" / "2027" / "second.json").write_text('{"gameData": {"game": {}}}')
    res = _run(tmp_path)
    assert res.exit_code != 0 and isinstance(res.exception, ValueError)
    assert all(name in str(res.exception) for name in ("999999.json", "second.json"))
    assert res.stderr.count("Skipping ") == 2
    assert not (tmp_path / "processed").exists()


def test_cli_seven_inning_regular_game_still_takes_schema_error_path(tmp_path, sample_game_feed):
    sample_game_feed["gameData"]["game"]["doubleHeader"] = "S"
    for play in sample_game_feed["liveData"]["plays"]["allPlays"]:
        play["about"]["inning"] = 7
    _feed(tmp_path, sample_game_feed)
    res = _run(tmp_path)
    assert res.exit_code != 0
    assert isinstance(res.exception, RuntimeError) and "Schema drift" in str(res.exception)
    assert not (tmp_path / "processed").exists()


def test_cli_truncated_feed_still_raises(tmp_path):
    directory = tmp_path / "raw" / "2027"
    directory.mkdir(parents=True)
    (directory / "broken.json").write_text('{"gameData":')
    res = _run(tmp_path)
    assert res.exit_code != 0 and isinstance(res.exception, json.JSONDecodeError)
    assert not (tmp_path / "processed").exists()
