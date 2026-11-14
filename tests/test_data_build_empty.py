"""The real nightly CLI must pass an unavailable pre-season without publishing data."""
import json
import shutil

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


@pytest.mark.parametrize("kind", ["missing", "empty", "non_regular", "exhibition", "unknown_type", "weather_only"])
@pytest.mark.parametrize("existing", [False, True])
def test_cli_empty_season_is_success_without_writes(tmp_path, sample_game_feed, kind, existing):
    if kind == "empty":
        (tmp_path / "raw" / "2027").mkdir(parents=True)
    elif kind in ("non_regular", "exhibition", "unknown_type"):
        sample_game_feed["gameData"]["game"]["type"] = {"non_regular": "S", "exhibition": "E", "unknown_type": "X"}[kind]
        _feed(tmp_path, sample_game_feed)
        (tmp_path / "raw" / "2027" / "another.json").write_text(json.dumps(sample_game_feed))
    elif kind == "weather_only":
        directory = tmp_path / "raw" / "2027"
        directory.mkdir(parents=True)
        (directory / "999999_weather.json").write_text("not a game feed")
    if existing:
        target = tmp_path / "processed" / "pa_2027.parquet"
        target.parent.mkdir()
        pd.DataFrame({"sentinel": [1]}).to_parquet(target)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None for p in tmp_path.rglob("*")}
    res = _run(tmp_path)
    assert res.exit_code == 0, res.output
    assert res.stderr.splitlines() == ["No regular-season games available for season 2027; nothing written."]
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
            for p in tmp_path.rglob("*")} == before


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


UNTYPED_FEEDS = [
    {}, {"gameData": {}}, {"gameData": {"game": {}}}, {"gameData": {"game": {"pk": 1}}},
    *[{"gameData": {"game": {"type": value}}} for value in (None, 7, True, "", "   ")],
]


@pytest.mark.parametrize("feed", UNTYPED_FEEDS)
@pytest.mark.parametrize("companion_type", [None, "S", "R"])
@pytest.mark.parametrize("existing", [False, True])
def test_cli_untyped_feed_is_skipped_only_beside_requested_feed(
        tmp_path, sample_game_feed, feed, companion_type, existing):
    _feed(tmp_path, feed)
    if companion_type is not None:
        sample_game_feed["gameData"]["game"]["type"] = companion_type
        (tmp_path / "raw" / "2027" / "companion.json").write_text(json.dumps(sample_game_feed))
    target = tmp_path / "processed" / "pa_2027.parquet"
    if existing:
        target.parent.mkdir()
        pd.DataFrame({"sentinel": [1]}).to_parquet(target)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None for p in tmp_path.rglob("*")}
    res = _run(tmp_path)
    assert res.stderr.splitlines() == [
        f"Skipping {tmp_path / 'raw' / '2027' / '999999.json'}: missing or unreadable gameData.game.type."]
    if companion_type == "R":
        assert res.exit_code == 0, res.output
        actual = pd.read_parquet(target)
        assert len(actual) == 2
        clean = tmp_path / "without_untyped_feed"
        shutil.copytree(tmp_path / "raw", clean / "raw", ignore=shutil.ignore_patterns("999999.json"))
        control = _run(clean)
        assert control.exit_code == 0, control.output
        expected = pd.read_parquet(clean / "processed" / "pa_2027.parquet")
        assert len(expected) == 2
        pd.testing.assert_frame_equal(actual, expected)
    else:
        assert res.exit_code != 0
        assert isinstance(res.exception, ValueError)
        assert "gameData.game.type" in str(res.exception) and "999999.json" in str(res.exception)
        assert {p: (p.read_bytes(), p.stat().st_mtime_ns) if p.is_file() else None
                for p in tmp_path.rglob("*")} == before


def test_cli_without_regular_feeds_names_every_untyped_file(tmp_path):
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


FATAL_FEEDS = [
    pytest.param([], AttributeError, id="list_body"),
    pytest.param(None, AttributeError, id="null_body"),
    pytest.param({"gameData": None}, AttributeError, id="null_gameData"),
    pytest.param({"gameData": {"game": None}}, AttributeError, id="null_game"),
    pytest.param({"gameData": {"game": {"type": []}}}, TypeError, id="list_type"),
    pytest.param({"gameData": {"game": {"type": {}}}}, TypeError, id="dict_type"),
]


@pytest.mark.parametrize("feed,error", FATAL_FEEDS)
@pytest.mark.parametrize("companion_type", [None, "S", "R"])
@pytest.mark.parametrize("existing", [False, True])
def test_cli_fatal_shapes_raise_without_writes(tmp_path, sample_game_feed, feed, error, companion_type, existing):
    _feed(tmp_path, feed)
    if companion_type is not None:
        sample_game_feed["gameData"]["game"]["type"] = companion_type
        (tmp_path / "raw" / "2027" / "companion.json").write_text(json.dumps(sample_game_feed))
    target = tmp_path / "processed" / "pa_2027.parquet"
    if existing:
        target.parent.mkdir()
        pd.DataFrame({"sentinel": [1]}).to_parquet(target)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    res = _run(tmp_path)
    assert res.exit_code != 0, res.output
    assert isinstance(res.exception, error), repr(res.exception)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()} == before
    assert "Skipping " not in res.stderr
    if not existing:
        assert not target.parent.exists()
