"""Adversarial regression tests for probable-pitcher cache atomic replacement and warning."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bts.features import compute


def _feed(away: int | None, home: int | None, away_tid: int = 51, home_tid: int = 52) -> str:
    pp: dict[str, dict[str, int]] = {}
    if away is not None:
        pp["away"] = {"id": away}
    if home is not None:
        pp["home"] = {"id": home}
    return json.dumps({
        "gameData": {
            "teams": {"away": {"id": away_tid}, "home": {"id": home_tid}},
            "probablePitchers": pp,
        }
    })


@pytest.fixture
def cache_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cache_path = tmp_path / "models" / "probable_pitcher_lookup.json"
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", cache_path)
    raw_dir = tmp_path / "raw"
    (raw_dir / "2025").mkdir(parents=True)
    return cache_path, raw_dir


@pytest.mark.parametrize("exc_factory", [
    lambda: KeyboardInterrupt("sigint mid-write"),
    lambda: SystemExit(2),
])
def test_base_exception_during_temp_write_cleans_up_temp_and_keeps_cache(
    cache_env, monkeypatch: pytest.MonkeyPatch, exc_factory
):
    path, raw = cache_env
    (raw / "2025" / "100.json").write_text(_feed(11, 12))
    compute._build_probable_pitcher_lookup(str(raw))
    before = path.read_bytes()

    (raw / "2025" / "101.json").write_text(_feed(21, 22))
    real_write_text = Path.write_text

    def interrupted_write(self: Path, data: str, *args, **kwargs):
        real_write_text(self, data[: max(1, len(data) // 2)], *args, **kwargs)
        raise exc_factory()

    monkeypatch.setattr(Path, "write_text", interrupted_write)
    with pytest.raises(BaseException) as exc_info:
        compute._build_probable_pitcher_lookup(str(raw))
    assert isinstance(exc_info.value, (KeyboardInterrupt, SystemExit))
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_replace_failure_cleans_up_temp_and_preserves_cache_only_entries(
    cache_env, monkeypatch: pytest.MonkeyPatch
):
    path, raw = cache_env
    path.parent.mkdir(parents=True)
    seed = {"999": {"away": 91, "home": 92, "away_tid": 1, "home_tid": 2}}
    path.write_text(json.dumps(seed))
    before = path.read_bytes()

    (raw / "2025" / "101.json").write_text(_feed(21, 22))

    def fail_replace(src, dst, *args, **kwargs):
        assert Path(src).exists()
        raise OSError("rename failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="rename failed"):
        compute._build_probable_pitcher_lookup(str(raw))
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir()] == [path.name]


@pytest.mark.parametrize("payload,exc_name", [
    ('{"50": {"away": 1', "JSONDecodeError"),
    ("[1, 2, 3]", "AttributeError"),
    ('{"50": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}, "bad": {}}', "ValueError"),
])
def test_unreadable_or_malformed_cache_warns_to_stderr_only_and_discards_partial_state(
    cache_env, capsys: pytest.CaptureFixture[str], payload: str, exc_name: str
):
    path, raw = cache_env
    (raw / "2025" / "100.json").write_text(_feed(21, 22))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)

    lookup = compute._build_probable_pitcher_lookup(str(raw))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"WARNING: unreadable probable-pitcher cache {path} ({exc_name});" in captured.err
    assert lookup == {100: {"away": 21, "home": 22, "away_tid": 51, "home_tid": 52}}


def test_symlinked_cache_is_replaced_without_mutating_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "frozen.json"
    init = json.dumps({"50": {"away": 1, "home": 2, "away_tid": 3, "home_tid": 4}}).encode()
    target.write_bytes(init)

    link = tmp_path / "models" / "probable_pitcher_lookup.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", link)

    raw = tmp_path / "raw"
    (raw / "2025").mkdir(parents=True)
    (raw / "2025" / "100.json").write_text(_feed(21, 22))

    lookup = compute._build_probable_pitcher_lookup(str(raw))
    assert 50 in lookup and 100 in lookup
    assert not link.is_symlink()
    assert target.read_bytes() == init
