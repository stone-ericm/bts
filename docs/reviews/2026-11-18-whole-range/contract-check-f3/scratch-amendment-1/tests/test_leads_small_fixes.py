"""The probable-pitcher lookup cache was written in place: an interrupted write left a
truncated file, which the next read silently treated as an empty lookup
(2026-11-18 leads check, docs/audit/2026-11-18-leads-check/, lead 3).
"""
import json
import os
from pathlib import Path

import pytest

from bts.features import compute


def _feed(away, home):
    return json.dumps({"gameData": {
        "teams": {"away": {"id": 51}, "home": {"id": 52}},
        "probablePitchers": {"away": {"id": away}, "home": {"id": home}},
    }})


@pytest.fixture
def cache(tmp_path, monkeypatch):
    path = tmp_path / "models" / "probable_pitcher_lookup.json"
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", path)
    raw = tmp_path / "raw" / "2025"
    raw.mkdir(parents=True)
    return path, tmp_path / "raw"


def test_interrupted_cache_write_leaves_the_previous_cache_intact(cache, monkeypatch):
    path, raw = cache
    (raw / "2025" / "100.json").write_text(_feed(21, 22))
    first = compute._build_probable_pitcher_lookup(str(raw))
    assert first[100]["away"] == 21
    before = path.read_bytes()

    (raw / "2025" / "101.json").write_text(_feed(23, 24))
    real_write_text = Path.write_text

    def torn_write_text(self, data, *args, **kwargs):
        # The process dies half-way through writing whichever file the writer chose.
        real_write_text(self, data[: len(data) // 2], *args, **kwargs)
        raise OSError("interrupted")

    monkeypatch.setattr(Path, "write_text", torn_write_text)
    with pytest.raises(OSError, match="interrupted"):
        compute._build_probable_pitcher_lookup(str(raw))
    monkeypatch.setattr(Path, "write_text", real_write_text)

    assert path.read_bytes() == before
    assert not [p for p in path.parent.iterdir() if p.name != path.name], "no temporary file left behind"
    again = compute._build_probable_pitcher_lookup(str(raw))
    assert again[100]["away"] == 21 and again[101]["away"] == 23


def test_cache_write_replaces_the_file_atomically(cache, monkeypatch):
    path, raw = cache
    (raw / "2025" / "100.json").write_text(_feed(21, 22))
    replaced = []
    real_replace = os.replace

    def spy(src, dst, **kwargs):
        replaced.append((Path(src).parent, Path(dst)))
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    compute._build_probable_pitcher_lookup(str(raw))
    assert replaced == [(path.parent, path)]
    assert json.loads(path.read_text()) == {"100": {"away": 21, "home": 22, "away_tid": 51, "home_tid": 52}}


def test_unreadable_cache_is_reported_and_rebuilt(cache, capsys):
    path, raw = cache
    (raw / "2025" / "100.json").write_text(_feed(21, 22))
    path.parent.mkdir(parents=True)
    path.write_text('{"100": {"away": 2')
    lookup = compute._build_probable_pitcher_lookup(str(raw))
    assert lookup[100]["away"] == 21
    assert "probable_pitcher_lookup.json" in capsys.readouterr().err
    assert json.loads(path.read_text())["100"]["home"] == 22


def test_readable_cache_is_used_silently_and_not_rewritten(cache, capsys):
    path, raw = cache
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"7": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}))
    stamp = path.stat().st_mtime_ns
    lookup = compute._build_probable_pitcher_lookup(str(raw))
    assert lookup == {7: {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}
    assert capsys.readouterr().err == ""
    assert path.stat().st_mtime_ns == stamp
