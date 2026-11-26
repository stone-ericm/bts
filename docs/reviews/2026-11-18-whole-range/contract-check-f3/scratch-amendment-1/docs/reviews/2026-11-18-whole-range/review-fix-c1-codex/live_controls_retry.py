"""Use the scheduler's supported configuration key in the lock refusal fixture."""

import json
import logging
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from live_controls import HERE, encode, run_tree


def compare():
    with TemporaryDirectory(dir=HERE) as scratch:
        root = Path(scratch) / "shared"
        outputs = []
        for revision in ("base", "head"):
            child = subprocess.run([sys.executable, __file__, revision, str(root)], check=True,
                                   stdout=subprocess.PIPE, timeout=22)
            (HERE / f"live-{revision}.json").write_bytes(child.stdout)
            outputs.append(child.stdout)
            shutil.rmtree(root)
    if outputs[0] != outputs[1]:
        base, head = map(json.loads, outputs)
        print("Differences:", [name for name in base if base[name] != head[name]])
    assert outputs[0] == outputs[1]
    print(f"{len(json.loads(outputs[0]))} controls (4 runner routes, 11 file fixtures); identical serialized bytes")


def run(revision, root):
    source = HERE / "base/src" if revision == "base" else HERE.parents[1] / "src"
    sys.path.insert(0, str(source))
    from bts import scheduler

    deliver = scheduler._deliver_and_lock_pick

    def configure_required(daily, config, *args, **kwargs):
        required = config["health_checks"]["contest_state_required"]
        return deliver(daily, {**config, "health_checks": {"contest_state_expected": required}}, *args, **kwargs)

    with (patch.object(scheduler, "_deliver_and_lock_pick", configure_required),
          patch.object(socket, "socket", side_effect=AssertionError("offline")),
          patch.object(socket, "create_connection", side_effect=AssertionError("offline"))):
        print(json.dumps(run_tree(revision, root), default=encode, sort_keys=True))


if __name__ == "__main__":
    signal.alarm(55)
    logging.disable(logging.CRITICAL)
    if sys.argv[1] == "compare":
        compare()
    else:
        run(sys.argv[1], Path(sys.argv[2]))
