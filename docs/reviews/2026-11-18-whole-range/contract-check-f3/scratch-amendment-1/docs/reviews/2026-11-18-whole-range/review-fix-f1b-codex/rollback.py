"""Replay real F1 markers against the archived pre-F1 watchdog."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import bts
from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def check(root):
    print(f"bts.__file__ = {bts.__file__}")
    for name, confirm, dm in (("alerted", True, True), ("dm_failed", True, False),
                               ("present_unverified-initial", True, True),
                               ("present_unverified-wrong-tier", True, False), ("terminal", False, False)):
        world = World(root / name)
        before = world.marker.read_bytes()
        got = world.check("19:00", MATCH)
        assert got["exit"] == 0 and got["marker"]["status"] == "confirmed"
        assert bool(got["fetches"]) == confirm and bool(got["attempted_dm"]) == dm
        if not confirm:
            assert world.marker.read_bytes() == before
        if dm:
            assert got["attempted_dm"] == [f"✅ BTS pick entry confirmed for {DAY} (match)."]
        print(name, bool(got["fetches"]), bool(got["attempted_dm"]), json.loads(before)["status"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(Path(sys.argv[1]))
    else:
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            archive = subprocess.check_output(["git", "archive", "1e69d0f0", "src"], cwd=ROOT, timeout=4)
            subprocess.run(["tar", "xf", "-", "-C", tmp], input=archive, check=True, timeout=4)
            for name in ("alerted", "dm_failed", "present_unverified-initial", "present_unverified-wrong-tier", "terminal"):
                world = World(root / name)
                world.commit(pitch=PITCH)
                world.capture([WRONG])
                if name == "present_unverified-initial":
                    world.check("18:25")
                first = world.check("18:30", MATCH, dm="failure" if name == "dm_failed" else "success")
                assert first["marker"]["game_unresolved"]
                if name.startswith("present_unverified"):
                    bridge = world.check("18:45", MATCH, crosswalk={})
                    assert bridge["marker"]["status"] == "present_unverified"
                if name == "terminal":
                    world.capture([RIGHT])
                    assert world.check("18:45", MATCH)["marker"]["status"] == "confirmed"
            subprocess.run([sys.executable, __file__, tmp], cwd=ROOT, check=True, timeout=12,
                           env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (root / "src", ROOT)))})
