"""Replay monitor behavior and atomic replacement using stock Python only."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from heartbeat_probe import HERE, check


def replay(root):
    heartbeat = root / ".heartbeat"
    pending = {"state": "deploy_failed", "timestamp": "2027-01-01T00:00:00+00:00",
               "step": "sync", "target_sha": "fef05b5"}
    sleeping = {"state": "sleeping", "timestamp": "2026-12-31T20:00:00+00:00",
                "sleeping_until": "2027-01-01T15:00:00+00:00"}
    result = {"version": sys.version.split()[0],
              "failure": check(heartbeat, pending),
              "next_tick": check(heartbeat),
              "canary_before_commit": check(heartbeat, sleeping),
              "canary_later_failed": check(heartbeat, pending),
              "old_heartbeat_if_marker_unwritten": check(heartbeat, sleeping),
              "iso_z": check(heartbeat, {**sleeping, "timestamp": "2026-12-31T20:00:00Z"})}
    assert result["failure"] == result["next_tick"] and result["failure"]["exit"] == 1
    assert result["canary_before_commit"]["exit"] == 0 and result["canary_later_failed"]["exit"] == 1
    outside = root / "original"
    outside.write_text("retained")
    result["atomic_replace"] = {}
    for kind in ("file", "symlink", "fifo", "directory"):
        if heartbeat.exists():
            heartbeat.unlink()
        if kind == "file":
            heartbeat.write_text("previous")
        elif kind == "symlink":
            heartbeat.symlink_to(outside)
        elif kind == "fifo":
            os.mkfifo(heartbeat)
        else:
            heartbeat.mkdir()
        temporary = root / "temporary"
        temporary.write_text("replacement")
        try:
            os.replace(temporary, heartbeat)
        except OSError as error:
            assert kind == "directory" and list(heartbeat.iterdir()) == []
            result["atomic_replace"][kind] = type(error).__name__
        else:
            assert heartbeat.is_file() and not heartbeat.is_symlink() and outside.read_text() == "retained"
            result["atomic_replace"][kind] = heartbeat.read_text()
    return result


if __name__ == "__main__":
    with TemporaryDirectory(prefix="heartbeat-r2-", dir=HERE) as temp:
        print(json.dumps(replay(Path(temp)), indent=2))
