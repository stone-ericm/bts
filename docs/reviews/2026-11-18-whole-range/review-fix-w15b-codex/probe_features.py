import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import bts
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_wr_w15_corpus import RECOVERY, SEED, feature, feed_payload, park_drag
from tests.test_wr_w15_incidents import bullpen_fixture
from bts.features import compute

CACHE = Path("data/models/probable_pitcher_lookup.json")
RAW = Path("data/raw")
FEED = RAW / "2025/109.json"
STATUS = Path("data/health_state/probable_pitcher_cache")
ORIGINALS = [
    Path("/Users/eric/projects/bts/data/models/probable_pitcher_lookup.json"),
    Path("/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json"),
]


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 12, 31, 23, 0, tzinfo=timezone.utc).astimezone(tz)


def generated():
    rng = random.Random(1729)
    keys = ("0", "0000", "+0", "-0", " 1 ", "01", "+1", "٠١", "²", "-999", "−1", "123_456")
    keys = [key for key in keys if key not in {"²", "−1"}]
    for index in range(40):
        entries = {}
        generated_keys = [str(rng.randrange(-1000, 1000)) for _ in range(index)]
        for key in rng.sample(keys, k=index % len(keys)) + generated_keys:
            fields = rng.sample(list(SEED), k=4)
            entries[key] = {field: rng.choice((None, -1, 0, 2 ** 80)) for field in fields}
        yield json.dumps(entries, ensure_ascii=False).encode()


def fingerprint(actual):
    serial = [(type(key).__name__, key,
               [(field, type(value).__name__, value) for field, value in entry.items()])
              for key, entry in actual.items()]
    return hashlib.sha256(json.dumps(serial).encode()).hexdigest()


def parity(src):
    seen = []
    for index, original in enumerate([*ORIGINALS, *generated()]):
        if isinstance(original, Path):
            shutil.copy2(original, CACHE)
        else:
            CACHE.write_bytes(original)
            stamp = 1798761540123456789
            os.utime(CACHE, ns=(stamp, stamp))
        original_bytes, original_mtime = CACHE.read_bytes(), CACHE.stat().st_mtime_ns
        expected = {int(key): value for key, value in json.loads(original_bytes).items()}
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            actual = compute._build_probable_pitcher_lookup(RAW)
        assert actual == expected and fingerprint(actual) == fingerprint(expected)
        assert CACHE.read_bytes() == original_bytes and CACHE.stat().st_mtime_ns == original_mtime
        assert output.getvalue() == "" and not STATUS.exists() and not (CACHE.parent / "rejected").exists()
        seen.append((len(actual), fingerprint(actual), hashlib.sha256(original_bytes).hexdigest(), original_mtime))
        if index < len(ORIGINALS):
            print("Parity", original.name, seen[-1])
    print("Both orders, types, bytes, mtime, stderr and artifacts", len(seen), "caches",
          "cross-check", hashlib.sha256(json.dumps(seen).encode()).hexdigest())


def base_crash(is_base):
    for with_feed in (False, True):
        CACHE.write_bytes(b'{"100":null}')
        recovery_feed = FEED.with_name("100.json")
        if with_feed:
            recovery_feed.write_text(feed_payload(RECOVERY))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            observed, error, frame = feature()
        if is_base:
            assert isinstance(error, AttributeError) and str(error) == "'NoneType' object has no attribute 'get'"
            assert not stderr.getvalue()
        else:
            assert error is None and len(frame) == 2
            assert observed == ([(2, True, 61), (23, True, 62)] if with_feed else [])
            assert len(stderr.getvalue().splitlines()) == 1
        print("real compute_all_features", "feed", with_feed, "result", str(error) if error else observed,
              "stderr", stderr.getvalue().strip())


def feature_tail():
    _, error, frame = feature(bullpen_fixture())
    assert error is None
    return frame.iloc[-1]["opp_bullpen_hr_30g"]


def nan_reverse(is_base):
    existing = {str(game): dict(SEED) for game in range(100, 111)}
    existing["100"]["home"] = float("nan")
    CACHE.write_text(json.dumps(existing))
    score = feature_tail()
    assert score == 0.5 if is_base else pd.isna(score)
    print("Opposite side missing pitcher encoded as NaN, valid away fields, no feed:", score)


def dangling_link(is_base):
    CACHE.symlink_to("removed-old-cache.json")
    for game in range(100, 111):
        FEED.with_name(f"{game}.json").write_text(feed_payload(SEED))
    assert feature_tail() == 0.5
    assert CACHE.is_symlink() != is_base
    for path in RAW.glob("*/*.json"):
        path.unlink()
    score = feature_tail()
    assert score == 0.5 if is_base else pd.isna(score)
    print("Dangling cache symlink first run 0.5; after the raw feeds leave:", score, "symlink remains", CACHE.is_symlink())


def status_obstruction(is_base):
    CACHE.write_text(json.dumps({"999": 1, **{str(game): SEED for game in range(100, 111) if game != 109}}))
    FEED.write_text(feed_payload(SEED))
    STATUS.parent.mkdir(parents=True)
    STATUS.write_text("existing regular file where directory is expected")
    assert feature_tail() == 0.5
    FEED.unlink()
    STATUS.unlink()
    score = feature_tail()
    assert score == 0.5 if is_base else pd.isna(score)
    print("Unrelated incident plus status obstruction, then feed leaves before retry:", score)


def main():
    source = Path(sys.argv[1]).resolve()
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    is_base = source == HERE / "base/src"
    if not is_base:
        from bts.features import probable_pitcher_cache
        probable_pitcher_cache.datetime = Clock
    print("Imported", bts.__file__, flush=True)
    start = Path.cwd()
    with TemporaryDirectory(dir=HERE, prefix="features-") as work:
        for probe in (parity, base_crash, nan_reverse, dangling_link, status_obstruction):
            case = Path(work) / probe.__name__
            case.mkdir()
            os.chdir(case)
            try:
                CACHE.parent.mkdir(parents=True)
                FEED.parent.mkdir(parents=True)
                compute._probable_pitcher_cache_path = CACHE
                park_drag.get_table = lambda: None
                probe(is_base)
            finally:
                os.chdir(start)


if __name__ == "__main__":
    main()
