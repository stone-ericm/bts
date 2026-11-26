"""Compare the legacy diagnostic bytes and exercise the descriptor proxy fallback."""
import contextlib
import io
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts.features import compute


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY_PATH = Path("oom") / ("café" + chr(96) + "\nnext") / "models/cache.json"


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 12, 31, 23, 0, tzinfo=timezone.utc).astimezone(tz)


class FdProxy:
    encoding = "utf-8"

    def __init__(self, stream):
        self.stream = stream

    def fileno(self):
        return self.stream.fileno()

    def flush(self):
        self.stream.flush()


def child(source, case):
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    cache = LEGACY_PATH if case == "legacy" else Path("models/cache.json")
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"{" if case == "legacy" else b'{"100":null}')
    with patch.object(compute, "_probable_pitcher_cache_path", cache):
        if case == "legacy":
            compute._build_probable_pitcher_lookup("empty")
        else:
            from bts.features import probable_pitcher_cache as producer
            with (
                io.FileIO(os.dup(2), "wb") as stream,
                contextlib.redirect_stderr(FdProxy(stream)),
                patch.object(producer, "datetime", Clock),
            ):
                compute._build_probable_pitcher_lookup("empty")


def invoke(source, case, encoding, dest):
    dest.mkdir()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(source), case], cwd=dest,
        env=dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING=encoding),
        capture_output=True, timeout=8,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def main():
    previous, head = HERE / "previous/src", ROOT / "src"
    sources = [HERE.parent / "base/src", previous, head]
    with TemporaryDirectory(dir=HERE, prefix="stream-") as tmp:
        directory = Path(tmp)
        for encoding in ("utf-8", "ascii"):
            results = [invoke(src, "legacy", encoding, directory / f"{encoding}-{i}")
                       for i, src in enumerate(sources)]
            assert results[0] == results[1] == results[2]
            prefix = "WARNING: unreadable probable-pitcher cache"
            assert results[0].startswith(prefix.encode())
            assert b"oom/" in results[0] and b"\nnext/" in results[0] and b"`" in results[0]
            print("Legacy P2", encoding, repr(results[0]))
        earlier = invoke(previous, "proxy", "utf-8", directory / "proxy-previous")
        now = invoke(head, "proxy", "utf-8", directory / "proxy-head")
        assert not earlier and now.startswith(b"WARNING: probable-pitcher cache: dropped 1 invalid entry")
        assert now.count(b"\n") == 1
        print("FD proxy lacks errors attribute:", repr(earlier), "->", repr(now))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        child(Path(sys.argv[1]).resolve(), sys.argv[2])
