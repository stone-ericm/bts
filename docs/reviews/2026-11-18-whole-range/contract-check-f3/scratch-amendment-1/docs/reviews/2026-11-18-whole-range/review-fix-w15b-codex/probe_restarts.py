import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts.features import probable_pitcher_cache as cache


HERE = Path(__file__).resolve().parent
CACHE = Path("data/models/probable_pitcher_lookup.json")
RAW = Path("data/raw")
ORIGINAL = b'{"100":null, "900":{"away":23,"away_tid":51,"home":null,"home_tid":52}}\n'


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 12, 31, 23, 0, tzinfo=timezone.utc).astimezone(tz)


def child(source, crash_at, side):
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    rename = os.replace
    renames = 0

    def exit_at(src, target):
        nonlocal renames
        renames += 1
        if renames == crash_at and side == "before":
            os._exit(97)
        rename(src, target)
        if renames == crash_at and side == "after":
            os._exit(97)

    with patch.object(cache, "datetime", Clock), patch.object(cache.os, "replace", exit_at):
        assert set(cache.build(CACHE, RAW)) == {100, 900}
    print("restart OK")


def run_child(source, work, crash_at, side):
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(source), str(crash_at), side],
        cwd=work, env=dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1"),
        timeout=7, capture_output=True,
    )


def snapshot(folder):
    copies = list((folder / CACHE.parent / "rejected").glob("probable_pitcher_lookup.*.json"))
    status = list((folder / "data/health_state/probable_pitcher_cache").glob("*.json"))
    assert len(copies) <= 1 and len(status) <= 1
    active = (folder / CACHE).read_bytes()
    if copies:
        assert copies[0].read_bytes() == ORIGINAL
        assert hashlib.sha256(ORIGINAL).hexdigest()[:12] in copies[0].name
    if status:
        row = json.loads(status[0].read_bytes())
        assert copies and row["preserved"] == copies[0].name
        assert row["rescanned"] == 1 and row["game_ids"] == [100]
    return (bool(copies), bool(status), active != ORIGINAL)


def main(source):
    with TemporaryDirectory(dir=HERE, prefix="restart-") as work:
        for rename in range(1, 4):
            for side in ("before", "after"):
                folder = Path(work) / f"{rename}-{side}"
                (folder / CACHE.parent).mkdir(parents=True)
                (folder / CACHE).write_bytes(ORIGINAL)
                feed = folder / RAW / "2026/100.json"
                feed.parent.mkdir(parents=True)
                feed.write_text('{"gameData":{"teams":{"away":{"id":51},"home":{"id":52}},"probablePitchers":{"away":{"id":1},"home":{"id":22}}}}')
                initial = run_child(source, folder, rename, side)
                assert initial.returncode == 97, (initial.returncode, initial.stderr)
                observed = snapshot(folder)
                completed = rename - (side == "before")
                assert observed == tuple(index < completed for index in range(3)), observed
                restart = run_child(source, folder, 0, side)
                assert restart.returncode == 0, restart.stderr
                assert snapshot(folder) == (True, True, True)
                feed.unlink()
                final = run_child(source, folder, 0, side)
                assert final.returncode == 0 and final.stderr == b"", final.stderr
                print("Hard exit", rename, side, "found", observed, "restarted, retained game without feed")


if __name__ == "__main__":
    source = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 4:
        child(source, int(sys.argv[2]), sys.argv[3])
    else:
        main(source)
