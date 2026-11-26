import contextlib
import io
import json
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
QUOTED = "c" + chr(96) + "af" + chr(92) + "é"


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 12, 31, 23, 0, tzinfo=timezone.utc).astimezone(tz)


def child(source, action):
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    cache = Path(QUOTED) / "data/models/probable_pitcher_lookup.json"
    compute._probable_pitcher_cache_path = cache
    if action == "strict-utf8" or action == "surrogate-utf8":
        from bts.features import probable_pitcher_cache as producer
        errors = "strict" if action == "strict-utf8" else "backslashreplace"
        with io.TextIOWrapper(io.FileIO(os.dup(2), "wb"), encoding="utf-8", errors=errors) as stream:
            with contextlib.redirect_stderr(stream):
                producer.warn("sample path x" + chr(0xdcff) + ".json")
        return
    if action == "bad-json":
        compute._build_probable_pitcher_lookup()
        return
    from bts.features import probable_pitcher_cache as producer
    with patch.object(producer, "datetime", Clock):
        lookup = compute._build_probable_pitcher_lookup()
    assert list(lookup) == [100]


def invoke(source, work, action, encoding):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(source), action],
        cwd=work,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(source), PYTHONIOENCODING=encoding),
        timeout=6, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def main():
    root = HERE.parents[1]
    prior = HERE / "r2/src"
    head = root / "src"
    base = HERE / "base/src"
    with TemporaryDirectory(dir=HERE, prefix="stderr-") as work:
        work = Path(work)
        cache = work / QUOTED / "data/models/probable_pitcher_lookup.json"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"{")
        tails = b" (JSONDecodeError); rebuilding it from raw feeds\n"
        prefix = f"WARNING: unreadable probable-pitcher cache {cache.relative_to(work)}".encode()
        for encoding in ("ascii", "utf-8"):
            outputs = [invoke(source, work, "bad-json", encoding) for source in (base, prior, head)]
            assert outputs[0] == outputs[2]
            if encoding == "utf-8":
                assert outputs[0] == outputs[1] == prefix + tails
            else:
                assert b"\\xe9" in outputs[0] and b"caf\\?.json" not in outputs[0] and b"?" in outputs[1]
            print("Unreadable existing stderr bytes", encoding, [repr(out) for out in outputs])
        for action in ("strict-utf8", "surrogate-utf8"):
            outputs = [invoke(source, work, action, "utf-8") for source in (prior, head)]
            assert b"x?.json" in outputs[0]
            assert outputs[1] == b"" if action == "strict-utf8" else b"x\\udcff.json" in outputs[1]
            print(action, [repr(out) for out in outputs])
        cache.write_bytes(b'{"100":null}')
        (cache.parent / "rejected").write_bytes(b"non-directory")
        status = cache.parent.parent / "health_state/probable_pitcher_cache"
        status.parent.mkdir(exist_ok=True)
        status.write_bytes(b"non-directory")
        feed = work / "data/raw/2026/100.json"
        feed.parent.mkdir(parents=True)
        feed.write_text("{}")
        result = invoke(head, work, "failed-evidence-and-status", "utf-8")
        lines = result.decode().splitlines()
        assert len(lines) == 3
        shown = QUOTED.replace(chr(92), chr(92) * 2).replace(chr(96), chr(92) + "x60")
        for line in lines:
            assert shown in line and chr(96) not in line
        assert "could not preserve" in lines[1] and "could not write" in lines[2]
        assert cache.read_bytes() == b'{"100":null}'
        print("All three displayed producer paths escaped on real stderr", json.dumps(lines, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(Path(sys.argv[1]).resolve(), sys.argv[2])
    else:
        main()
