"""Incident, recovery, and filesystem ordering through the real cache builder."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import pandas as pd

from tests.test_wr_w15_corpus import RECOVERY, SEED, corpus_env, feature, feed_payload, pa
from bts.features import compute, park_drag


AT = "2027-01-01T04:59:00+00:00"
REPO_ROOT = str(Path(__file__).resolve().parents[1])


def child_env():
    paths = [REPO_ROOT]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}


@pytest.fixture
def clock(monkeypatch):
    freeze_clock(monkeypatch, AT)


def freeze_clock(monkeypatch, timestamp, module=None):
    at = datetime.fromisoformat(timestamp)
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz) if tz is not None else at.replace(tzinfo=None)

    # The producer module is introduced by W15. Exercise the old builder on RED.
    if module is None:
        try:
            from bts.features import probable_pitcher_cache as module
        except ImportError:
            return
    monkeypatch.setattr(module, "datetime", Frozen)


@pytest.fixture
def incident(corpus_env, clock):
    return corpus_env


def records(cache):
    folder = cache.parent.parent / "health_state/probable_pitcher_cache"
    return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]


def build():
    return compute._build_probable_pitcher_lookup("data/raw")


def build_child(cache):
    command = (
        "import json, sys, pytest; from pathlib import Path; from bts.features import compute; "
        "from tests.test_wr_w15_incidents import freeze_clock, AT; freeze_clock(pytest.MonkeyPatch(), AT); "
        "compute._probable_pitcher_cache_path = Path(sys.argv[1]); "
        "print(json.dumps(compute._build_probable_pitcher_lookup('data/raw')))"
    )
    return subprocess.run(
        [sys.executable, "-c", command, str(cache)],
        env=child_env(),
        capture_output=True, text=True, timeout=10,
    )


def seed(cache, values):
    cache.write_text(json.dumps(values))


def test_clean_exact_order_types_unchanged(incident, capsys):
    cache, _ = incident
    values = {"0100": SEED, "99": dict(reversed(list(SEED.items()))), "100": RECOVERY}
    seed(cache, values)
    stamp, original = cache.stat().st_mtime_ns, cache.read_bytes()
    result = build()
    assert list(result) == [100, 99]
    assert list(result[99].items()) == list(values["99"].items())
    assert list(result[100].items()) == list(RECOVERY.items())
    assert all(type(key) is int and all(type(v) is int for v in value.values())
               for key, value in result.items())
    assert cache.stat().st_mtime_ns == stamp and cache.read_bytes() == original
    assert list(cache.parent.iterdir()) == [cache]
    assert not records(cache)
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("feed", ["usable", "absent", "invalid"])
def test_nonobject_features_and_status(incident, capsys, feed):
    cache, raw = incident
    seed(cache, {"100": None, "999": SEED})
    if feed == "usable":
        raw.write_text(feed_payload(RECOVERY))
    if feed == "invalid":
        raw.write_text(feed_payload(dict(RECOVERY, away=True)))
    observed, error, _ = feature()
    assert error is None
    expected = [(2, True, 61), (23, True, 62)] if feed == "usable" else [(None, False, None)] * 2
    assert observed == expected
    k, u, m = tuple(int(feed == name) for name in ("usable", "invalid", "absent"))
    assert capsys.readouterr().err == (
        f"WARNING: probable-pitcher cache {cache}: dropped 1 invalid entry "
        f"(0 recovered from valid cache entries, {k} rescanned from raw feeds, "
        f"{u} with unusable feeds, {m} without feeds); first game ids: 100\n"
    )
    row, = records(cache)
    assert row["recorded_date_et"] == "2026-12-31"
    assert row["game_ids"] == [100] and row["dropped"] == 1
    assert (row["rescanned"], row["unusable_feed"], row["without_feed"]) == (k, u, m)


def test_first_usable_feed_in_sorted_discovery_order(incident):
    cache, raw = incident
    seed(cache, {"100": None, "-100": None})
    raw.write_text(feed_payload(SEED))
    raw.with_name("00100.json").write_text(feed_payload(RECOVERY))
    raw.with_name("-100.json").write_text(feed_payload(SEED))
    earlier = raw.parent.with_name("2024")
    earlier.mkdir()
    (earlier / "100.json").write_text(feed_payload(dict(SEED, away=True)))
    assert build() == {100: RECOVERY}
    row, = records(cache)
    assert row["rescanned"] == 1 and row["without_feed"] == 1


def test_feed_extraction_type_error_is_unusable(incident, monkeypatch):
    cache, raw = incident
    seed(cache, {"100": None})
    raw.write_text(feed_payload(RECOVERY))
    raw.with_name("101.json").write_text(feed_payload(SEED))
    original_loads = json.loads

    class MalformedGameData:
        def get(self, *_args):
            raise TypeError("invalid game data")

    def loads(payload, *args, **kwargs):
        decoded = original_loads(payload, *args, **kwargs)
        if decoded.get("gameData", {}).get("probablePitchers", {}).get("away", {}).get("id") == 2:
            decoded["gameData"] = MalformedGameData()
        return decoded

    monkeypatch.setattr(json, "loads", loads)
    assert build() == {101: SEED}
    row, = records(cache)
    assert row["unusable_feed"] == 1 and row["game_ids"] == [100]


def test_unreadable_evidence_survives_failed_active_replace(incident, monkeypatch):
    cache, raw = incident
    cache.write_text("{")
    raw.write_text(feed_payload(RECOVERY))
    replace = os.replace

    def deny(src, dst, **kwargs):
        if Path(dst) == cache:
            raise OSError("active replace failed")
        return replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", deny)
    with pytest.raises(OSError, match="active replace failed"):
        build()
    row, = records(cache)
    assert cache.read_bytes() == b"{"
    assert (cache.parent / "rejected" / row["preserved"]).read_bytes() == b"{"


@pytest.mark.parametrize("bad_first", [True, False])
def test_last_valid_alias_retains_first_position(incident, bad_first):
    cache, _ = incident
    values = {"00100": None, "99": SEED, "100": RECOVERY}
    if not bad_first:
        values = {"100": RECOVERY, "99": SEED, "00100": None}
    seed(cache, values)
    lookup = build()
    assert list(lookup.items()) == [(100, RECOVERY), (99, SEED)]
    row, = records(cache)
    assert row["recovered_from_cache"] == row["dropped"] == 1


@pytest.mark.parametrize("payload", [b"{", b"[]", b'{"100":null,"bad":{}}', b"\xff"])
@pytest.mark.parametrize("raw_state", ["missing", "empty", "usable"])
def test_unreadable_cache_records_and_preserves_original(incident, capsys, payload, raw_state):
    cache, raw = incident
    cache.write_bytes(payload)
    if raw_state == "missing":
        raw.parent.rmdir()
        raw.parent.parent.rmdir()
    elif raw_state == "usable":
        raw.write_text(feed_payload(RECOVERY))
    assert build() == ({100: RECOVERY} if raw_state == "usable" else {})
    row, = records(cache)
    assert row["reason"] == "unreadable_cache" and row["game_ids"] == []
    assert row["rescanned"] == (raw_state == "usable")
    assert all(row[key] is None for key in ("dropped", "recovered_from_cache", "unusable_feed", "without_feed"))
    assert row["original_sha256"] == hashlib.sha256(payload).hexdigest()
    if raw_state == "usable":
        assert (cache.parent / "rejected" / row["preserved"]).read_bytes() == payload
    else:
        assert row["preserved"] is None and cache.read_bytes() == payload
    assert len(capsys.readouterr().err.splitlines()) == 1


@pytest.mark.parametrize("source", ["evidence", "status", "cache"])
@pytest.mark.parametrize("after", [False, True])
def test_interrupt_at_each_rename(incident, monkeypatch, source, after):
    cache, raw = incident
    seed(cache, {"100": None})
    original = cache.read_bytes()
    raw.write_text(feed_payload(RECOVERY))
    replace = os.replace

    def stop(src, dst, **kwargs):
        dst = Path(dst)
        label = "cache" if dst == cache else "evidence" if dst.parent.name == "rejected" else "status"
        if label != source:
            return replace(src, dst, **kwargs)
        if after:
            replace(src, dst, **kwargs)
        raise KeyboardInterrupt(label)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", stop)
        with pytest.raises(KeyboardInterrupt, match=source):
            build()
    if source == "cache" and after:
        assert json.loads(cache.read_bytes()) == {"100": RECOVERY}
    else:
        assert cache.read_bytes() == original
    saved = list((cache.parent / "rejected").glob("*.json"))
    assert [path.read_bytes() for path in saved] == ([] if source == "evidence" and not after else [original])
    assert len(records(cache)) == int(source == "cache" or (source == "status" and after))


@pytest.mark.parametrize("destinations", ["evidence", "status", "both"])
def test_independent_publication_failures(incident, monkeypatch, capsys, destinations):
    cache, raw = incident
    cache.write_text("{")
    raw.write_text(feed_payload(RECOVERY))
    replace = os.replace

    def deny(src, dst, **kwargs):
        dst = Path(dst)
        label = "evidence" if dst.parent.name == "rejected" else "status"
        if dst != cache and destinations in (label, "both"):
            raise PermissionError(dst)
        return replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", deny)
    assert build() == {100: RECOVERY}
    assert cache.read_bytes() == b"{"
    evidence = list((cache.parent / "rejected").glob("*.json"))
    assert [path.read_bytes() for path in evidence] == ([b"{"] if destinations == "status" else [])
    assert [row["preserved"] for row in records(cache)] == ([None] if destinations == "evidence" else [])
    assert len(capsys.readouterr().err.splitlines()) == (3 if destinations == "both" else 2)


@pytest.mark.parametrize("where", ["evidence", "status"])
@pytest.mark.parametrize("kind", ["directory", "fifo", "symlink", "fake"])
def test_occupied_evidence_and_status_destinations(incident, where, kind):
    cache, raw = incident
    seed(cache, {"100": None})
    original = cache.read_bytes()
    raw.write_text(feed_payload(RECOVERY))
    assert build() == {100: RECOVERY}
    status = next((cache.parent.parent / "health_state/probable_pitcher_cache").glob("*.json"))
    evidence = next((cache.parent / "rejected").glob("*.json"))
    status.unlink()
    evidence.unlink()
    cache.write_bytes(original)
    occupied = evidence if where == "evidence" else status
    if where == "status":
        evidence.write_bytes(original)
    if kind == "directory":
        occupied.mkdir()
    elif kind == "fifo":
        os.mkfifo(occupied)
    elif kind == "symlink":
        occupied.symlink_to(cache)
    else:
        occupied.write_text("fake")
    result = build_child(cache)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"100": RECOVERY}
    assert cache.read_bytes() == original
    assert len(result.stderr.splitlines()) == 2


@pytest.mark.parametrize("where", ["evidence", "status"])
def test_unwritable_artifact_directory(incident, capsys, where):
    cache, raw = incident
    seed(cache, {"100": None})
    original = cache.read_bytes()
    raw.write_text(feed_payload(RECOVERY))
    folder = (cache.parent / "rejected" if where == "evidence"
              else cache.parent.parent / "health_state/probable_pitcher_cache")
    folder.mkdir(parents=True)
    folder.chmod(0o500)
    try:
        if os.access(folder, os.W_OK):
            pytest.skip("permissions do not restrict this test process")
        assert build() == {100: RECOVERY}
        assert cache.read_bytes() == original
        assert len(capsys.readouterr().err.splitlines()) == 2
    finally:
        folder.chmod(0o700)


def test_broken_stderr(incident, monkeypatch):
    cache, _ = incident
    cache.write_text("{")

    class Closed:
        def write(self, _message):
            raise BrokenPipeError

    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", Closed())
        assert build() == {}


@pytest.mark.parametrize("kind", ["cache_fifo", "feed_fifo", "unicode_name", "file_root"])
def test_discovery_edge_in_bounded_child(incident, kind):
    cache, raw = incident
    if kind == "cache_fifo":
        os.mkfifo(cache)
        raw.write_text(feed_payload(RECOVERY))
    else:
        seed(cache, {"100": None})
    if kind == "feed_fifo":
        os.mkfifo(raw)
    elif kind == "unicode_name":
        raw.with_name("².json").write_text(feed_payload(RECOVERY))
    elif kind == "file_root":
        raw.parent.rmdir()
        raw.parent.parent.rmdir()
        raw.parent.parent.write_text("file")
    result = build_child(cache)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ({"100": RECOVERY} if kind == "cache_fifo" else {})
    row, = records(cache)
    if kind == "cache_fifo":
        assert row["reason"] == "unreadable_cache" and cache.is_fifo()
        assert row["preserved"] is None and not (cache.parent / "rejected").exists()
    else:
        assert row["unusable_feed"] == int(kind == "feed_fifo")


def test_unlistable_season_is_unusable(incident, monkeypatch):
    cache, raw = incident
    seed(cache, {"100": None})
    iterdir, scandir = Path.iterdir, os.scandir

    def deny(path):
        if Path(path).absolute() == raw.parent:
            raise PermissionError(path)

    def list_path(path):
        deny(path)
        return iterdir(path)

    def scan_path(path):
        deny(path)
        return scandir(path)

    monkeypatch.setattr(Path, "iterdir", list_path)
    monkeypatch.setattr(os, "scandir", scan_path)
    assert build() == {}
    row, = records(cache)
    assert row["unusable_feed"] == 1


def test_exact_repeat_and_changed_per_game_outcomes(incident):
    cache, raw = incident
    seed(cache, {"100": None, "101": None})
    raw.write_text("{")
    assert build() == {}
    assert build() == {}
    assert len(records(cache)) == 1
    raw.rename(raw.with_name("101.json"))
    assert build() == {}
    first, second = records(cache)
    assert first["identity"] != second["identity"]
    assert first["game_ids"] == second["game_ids"] == [100, 101]
    assert first["unusable_feed"] == second["unusable_feed"] == 1


def test_identity_includes_full_sha_and_all_per_game_outcomes(incident):
    cache, _ = incident
    seed(cache, {str(game): None for game in range(112, 99, -1)})
    assert build() == {}
    row, = records(cache)
    identity_fields = {key: row[key] for key in (
        "reason", "cache_path", "dropped", "recovered_from_cache", "rescanned",
        "unusable_feed", "without_feed", "original_sha256", "preserved",
    )}
    identity_fields["outcomes"] = [[game, 0, 0, 0, 1, None] for game in range(100, 113)]
    canonical = json.dumps(identity_fields, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    assert row["identity"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert row["original_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert row["game_ids"] == list(range(100, 110))


def test_repeat_keeps_earlier_timestamp(incident, monkeypatch):
    cache, _ = incident
    seed(cache, {"100": None})
    build()
    before, = records(cache)
    freeze_clock(monkeypatch, "2027-01-01T04:59:30+00:00")
    build()
    assert records(cache) == [before]
    assert before["recorded_at"] == AT


def bullpen_fixture():
    dates = pd.date_range("2025-04-01", periods=11)
    return [pa(100 + i, day, 3 if i < 10 else 1, True, i % 2) for i, day in enumerate(dates)]


def test_directory_feed_candidates_keep_descriptors_and_bullpen_value(incident):
    cache, raw = incident
    seed(cache, {str(game): SEED for game in range(100, 111) if game != 109})
    for game in range(1, 81):
        raw.with_name(f"{game:05d}.json").mkdir()
    raw.with_name("109.json").write_text(feed_payload(SEED))
    script = """
import json, os, resource, sys
from pathlib import Path
from tests.test_wr_w15_incidents import bullpen_fixture, compute, feature, park_drag

compute._probable_pitcher_cache_path = Path(sys.argv[1])
park_drag.get_table = lambda: None
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
before = len(os.listdir("/dev/fd"))
_, error, frame = feature(bullpen_fixture())
assert error is None, error
after = len(os.listdir("/dev/fd"))
assert "109" in json.loads(compute._probable_pitcher_cache_path.read_bytes())
assert before == after, (before, after)
print(frame.iloc[-1]["opp_bullpen_hr_30g"])
"""
    child = subprocess.run([sys.executable, "-c", script, str(cache)],
                           env=child_env(), capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr
    assert float(child.stdout) == 0.5


def test_feature_child_exits_zero_with_closed_stderr_reader(incident):
    cache, raw = incident
    seed(cache, {"100": None})
    raw.write_text(feed_payload(RECOVERY))
    script = """
import sys
from pathlib import Path
from pytest import MonkeyPatch
from tests.test_wr_w15_incidents import AT, compute, feature, freeze_clock, park_drag

compute._probable_pitcher_cache_path = Path(sys.argv[1])
park_drag.get_table = lambda: None
with MonkeyPatch.context() as monkeypatch:
    freeze_clock(monkeypatch, AT)
    tuples, error, _ = feature()
assert error is None and tuples == [(2, True, 61), (23, True, 62)]
print("FEATURES_OK", flush=True)
"""
    reader, writer = os.pipe()
    os.close(reader)
    try:
        child = subprocess.run([sys.executable, "-c", script, str(cache)], env=child_env(),
                               stdout=subprocess.PIPE, stderr=writer, text=True, timeout=15)
    finally:
        os.close(writer)
    assert child.stdout == "FEATURES_OK\n"
    assert child.returncode == 0


def test_missing_home_team_member_loses_prior_away_bullpen_bucket(incident):
    cache, _ = incident
    values = {str(game): dict(SEED) for game in range(100, 111)}
    del values["100"]["home_tid"]
    seed(cache, values)
    _, error, frame = feature(bullpen_fixture())
    assert error is None
    assert pd.isna(frame.iloc[-1]["opp_bullpen_hr_30g"])
    row, = records(cache)
    assert row["game_ids"] == [100] and row["without_feed"] == 1


def test_inaccessible_raw_root_marks_unverified_feeds_unusable(incident, monkeypatch):
    cache, raw = incident
    seed(cache, {"100": None})
    iterdir = Path.iterdir

    def deny_root(path):
        if path.absolute() == raw.parent.parent:
            raise PermissionError(path)
        return iterdir(path)

    monkeypatch.setattr(Path, "iterdir", deny_root)
    assert build() == {}
    row, = records(cache)
    assert (row["unusable_feed"], row["without_feed"]) == (1, 0)


def test_publish_rejects_valid_conflicting_record_at_same_destination(incident, capsys):
    cache, raw = incident
    seed(cache, {"100": None})
    original = cache.read_bytes()
    raw.write_text(feed_payload(RECOVERY))
    assert build() == {100: RECOVERY}
    folder = cache.parent.parent / "health_state/probable_pitcher_cache"
    path, = folder.glob("*.json")
    conflicting = json.loads(path.read_bytes())
    conflicting["identity"] = conflicting["identity"][:12] + (
        "1" if conflicting["identity"][12] == "0" else "0"
    ) + conflicting["identity"][13:]
    path.write_text(json.dumps(conflicting))
    cache.write_bytes(original)
    capsys.readouterr()
    assert build() == {100: RECOVERY}
    assert cache.read_bytes() == original
    assert json.loads(path.read_bytes()) == conflicting
    assert "could not write probable-pitcher cache status" in capsys.readouterr().err


def test_publish_rejects_oversize_serialization_before_writing(incident):
    from bts.features.probable_pitcher_cache import publish
    from bts.probable_pitcher_status import MAX_BYTES

    cache, _ = incident
    path = cache.parent.parent / "health_state/probable_pitcher_cache/status.json"
    # The payload is valid JSON, and the destination is fresh.
    row = {"cache_path": "x" * (MAX_BYTES + 1)}
    with pytest.raises(ValueError, match="status exceeds size limit"):
        publish(path, row)
    assert not path.exists()


def test_unrelated_bad_member_does_not_lose_new_game_after_raw_removed(incident):
    cache, raw = incident
    seed(cache, {**{str(game): SEED for game in range(100, 111) if game != 109}, "999": None})
    new_feed = raw.with_name("109.json")
    new_feed.write_text(feed_payload(SEED))
    _, error, result = feature(bullpen_fixture())
    assert error is None
    assert result.iloc[-1]["opp_bullpen_hr_30g"] == 0.5
    assert json.loads(cache.read_bytes())["109"] == SEED
    new_feed.unlink()
    child_code = (
        "import runpy,sys; from pathlib import Path; "
        "p=runpy.run_path(sys.argv[1]); p['compute']._probable_pitcher_cache_path=Path(sys.argv[2]); "
        "p['park_drag'].get_table=lambda:None; "
        "_,error,result=p['feature'](p['bullpen_fixture']()); "
        "assert error is None, error; print(result.iloc[-1]['opp_bullpen_hr_30g'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", child_code, str(Path(__file__).resolve()), str(cache)],
        env=child_env(),
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.strip()) == 0.5


def test_live_rewrite_retains_original_bad_bytes_and_incident(incident):
    cache, raw = incident
    seed(cache, {"999": None})
    original = cache.read_bytes()
    raw.write_text(feed_payload(RECOVERY))
    assert build().get(100) == RECOVERY
    row, = records(cache)
    assert row["without_feed"] == 1 and row["game_ids"] == [999]
    assert (cache.parent / "rejected" / row["preserved"]).read_bytes() == original
    assert json.loads(cache.read_bytes()) == {"100": RECOVERY}


@pytest.mark.parametrize("source", [
    Path.home() / "projects/bts/data/models/probable_pitcher_lookup.json",
    Path.home() / "bts-runs/inputs/probable_pitcher_lookup.2017-2025.json",
], ids=["main-copy", "frozen-copy"])
def test_real_cache_parity(incident, capsys, source):
    if os.environ.get("BTS_W15_REAL_ARTIFACTS") != "1":
        pytest.skip("studio artifact check")
    cache, _ = incident
    shutil.copy2(source, cache)
    before, stamp = cache.read_bytes(), cache.stat().st_mtime_ns
    expected = {int(key): entry for key, entry in json.loads(before).items()}
    actual = build()
    assert actual == expected and list(actual) == list(expected)
    for key, entry in actual.items():
        assert type(key) is int
        assert list(entry) == list(expected[key])
        assert [type(val) for val in entry.values()] == [type(val) for val in expected[key].values()]
    assert cache.read_bytes() == before and cache.stat().st_mtime_ns == stamp
    assert list(cache.parent.iterdir()) == [cache]
    assert not records(cache)
    assert capsys.readouterr().err == ""
    print(f"{source.name}: {len(actual)} entries; exact equality, key orders and types; bytes and mtime unchanged; no stderr, status or evidence")
