"""Weather scan, feed publication and pre-season build regression cases."""

import io
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

import bts.data.pull as pull
import bts.features.compute as compute
from bts.data.build import build_season


@pytest.mark.parametrize("stem", ["9001_weather", "notes", "-1", "123x"])
def test_s3_sidecars_do_not_change_numeric_cache(tmp_path, monkeypatch, stem):
    raw = tmp_path / "raw"
    season = raw / "2027"
    season.mkdir(parents=True)
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", cache)
    feed = {"gameData": {"teams": {"away": {"id": 1}, "home": {"id": 2}},
                         "probablePitchers": {"away": {"id": 11}, "home": {"id": 22}}}}
    (season / "9001.json").write_text(json.dumps(feed))
    expected = {9001: {"away": 11, "home": 22, "away_tid": 1, "home_tid": 2}}
    assert compute._build_probable_pitcher_lookup(str(raw)) == expected
    before = cache.read_bytes()
    cache.unlink()  # force the numeric feed to be rebuilt alongside the sidecar
    (season / f"{stem}.json").write_text(json.dumps(feed) if stem == "-1" else "not even JSON")
    assert compute._build_probable_pitcher_lookup(str(raw)) == expected
    assert cache.read_bytes() == before
    assert compute._build_probable_pitcher_lookup(str(raw)) == expected
    assert cache.read_bytes() == before


def test_s3_weather_only_creates_no_cache(tmp_path, monkeypatch):
    (tmp_path / "2027").mkdir()
    (tmp_path / "2027" / "9001_weather.json").write_text("{}")
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", cache)
    assert compute._build_probable_pitcher_lookup(str(tmp_path)) == {}
    assert not cache.exists()


@pytest.mark.parametrize("payload", [b'{"gameData":', b'gateway error', b'[]', b'null', b'12', b'"ok"', b'\xff'])
def test_s4_invalid_body_retries_without_files(tmp_path, monkeypatch, payload):
    opened = Mock(side_effect=lambda *a, **k: io.BytesIO(payload))
    slept = Mock()
    monkeypatch.setattr(pull, "urlopen", opened)
    monkeypatch.setattr(pull.time, "sleep", slept)
    with pytest.raises((ValueError, UnicodeError)):
        pull.download_game_feed(9001, tmp_path)
    assert opened.call_count == 3
    assert [c.args for c in slept.call_args_list] == [(1,), (2,)]
    assert list(tmp_path.iterdir()) == []


def test_s4_retries_invalid_then_publishes_identical_bytes_atomically(tmp_path, monkeypatch):
    payload = b' { "gameData": {}, "liveData": {} }\n'
    opened = Mock(side_effect=[io.BytesIO(b'{"truncated":'), io.BytesIO(payload)])
    monkeypatch.setattr(pull, "urlopen", opened)
    monkeypatch.setattr(pull.time, "sleep", Mock())
    target = tmp_path / "9001.json"
    real_replace = os.replace
    replaced = []

    def replace(source, destination):
        source, destination = Path(source), Path(destination)
        assert source.parent == target.parent and destination == target
        assert not source.match("*.json")
        assert not target.exists()
        assert source.read_bytes() == payload
        real_replace(source, destination)
        replaced.append(destination)

    monkeypatch.setattr(os, "replace", replace)
    assert pull.download_game_feed(9001, tmp_path) == target
    assert opened.call_count == 2 and replaced == [target]
    assert target.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [target]


def test_s4_replace_failure_cleans_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pull, "urlopen", Mock(side_effect=lambda *a, **k: io.BytesIO(b'{}')))
    monkeypatch.setattr(pull.time, "sleep", Mock())
    replace = Mock(side_effect=OSError("rename failed"))
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError, match="rename failed"):
        pull.download_game_feed(9001, tmp_path)
    assert replace.call_count == 3
    assert list(tmp_path.iterdir()) == []


def test_s4_write_failure_cleans_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pull, "urlopen", Mock(side_effect=lambda *a, **k: io.BytesIO(b'{}')))
    monkeypatch.setattr(pull.time, "sleep", Mock())
    # Exhausting a regular file's size limit would be platform dependent; fail
    # the actual file object's write while retaining its close/unlink lifecycle.
    import tempfile
    real_temp = tempfile.NamedTemporaryFile

    def temporary(*args, **kwargs):
        result = real_temp(*args, **kwargs)
        result.write = Mock(side_effect=OSError("disk full"))
        return result

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", temporary)
    with pytest.raises(OSError, match="disk full"):
        pull.download_game_feed(9001, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_s4_existing_even_invalid_file_is_unchanged(tmp_path, monkeypatch):
    target = tmp_path / "9001.json"
    target.write_bytes(b'{"old truncated":')
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    opened = Mock(side_effect=AssertionError("cached feed must not be requested"))
    monkeypatch.setattr(pull, "urlopen", opened)
    assert pull.download_game_feed(9001, tmp_path) == target
    opened.assert_not_called()
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_s4_pull_feeds_skips_invalid_and_continues(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pull, "discover_games", lambda *a: [
        {"gamePk": 1, "date": "2027-03-24"}, {"gamePk": 2, "date": "2027-03-24"}])
    monkeypatch.setattr(pull, "urlopen", Mock(side_effect=[
        io.BytesIO(b'no'), io.BytesIO(b'no'), io.BytesIO(b'no'), io.BytesIO(b'{}')]))
    monkeypatch.setattr(pull.time, "sleep", Mock())
    paths = pull.pull_feeds("2027-03-24", "2027-03-24", tmp_path, delay=0)
    assert paths == [tmp_path / "2027" / "2.json"]
    assert list((tmp_path / "2027").iterdir()) == paths
    assert "SKIP 1" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "empty", "spring", "weather"])
@pytest.mark.parametrize("existing", [False, True])
def test_s5_no_requested_feed_noop_only_before_output_exists(tmp_path, kind, existing):
    raw = tmp_path / "raw"
    season = raw / "2027"
    target = tmp_path / "processed" / "pa_2027.parquet"
    if kind != "missing":
        season.mkdir(parents=True)
    if kind == "spring":
        (season / "1.json").write_text('{"gameData":{"game":{"type":"S"}}}')
    elif kind == "weather":
        (season / "1_weather.json").write_text("not a feed")
    if existing:
        target.parent.mkdir()
        target.write_bytes(b"existing season data")
        before = (target.read_bytes(), target.stat().st_mtime_ns)
        with pytest.raises(ValueError) as caught:
            build_season(raw, target, 2027)
        assert "2027" in str(caught.value) and str(season) in str(caught.value)
        assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    else:
        assert build_season(raw, target, 2027) is None
        assert not target.parent.exists()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("kind", ["file", "unreadable"])
def test_s5_invalid_season_directory_raises(tmp_path, monkeypatch, existing, kind):
    raw = tmp_path / "raw"
    raw.mkdir()
    season = raw / "2027"
    target = tmp_path / "pa_2027.parquet"
    if existing:
        target.write_bytes(b"existing season data")
    if kind == "file":
        season.write_text("not a directory")
        error = NotADirectoryError
    else:
        season.mkdir()
        (season / "1.json").write_text('{"gameData":{"game":{"type":"R"}}}')
        # Inject the real OS failure at directory enumeration, independently of
        # the test runner's privileges and platform's glob error suppression.
        real_iterdir = Path.iterdir

        def iterdir(path):
            if path == season:
                raise PermissionError("cannot list season")
            return real_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", iterdir)
        real_scandir = os.scandir

        def scandir(path):
            if Path(path) == season:
                raise PermissionError("cannot list season")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", scandir)
        error = PermissionError
    with pytest.raises(error):
        build_season(raw, target, 2027)
    assert target.read_bytes() == b"existing season data" if existing else not target.exists()
