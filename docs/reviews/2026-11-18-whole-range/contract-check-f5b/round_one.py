"""Replay the late manual fix with the first F5 code actually running."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from replay_support import DAY, RIGHT, WRONG, HERE, ROOT, SCRATCH, World, compact, row


if sys.argv[1:] == ["replay"]:
    import bts
    assert "/round1/" in bts.__file__, bts.__file__
    with TemporaryDirectory() as tmp:
        w = World(Path(tmp))
        w.commit_at("10:05", at="09:40")
        w.capture([WRONG], f"{DAY}T134000Z")
        early = w.cron("09:45", pending=[row()])
        w.capture([{**RIGHT, "id": 78}], f"{DAY}T135000Z")
        manual = w.invoke("09:50", manual=True, pending=[row(78)])
        assert early["sent"] and manual["sent"] and manual["sent"][0].startswith("✅")
        assert w.cron("10:00")["receipt"]["outcome"] == "outside_window"
        print("round 1 late manual fix", bts.__file__, compact(early), compact(manual))
else:
    with TemporaryDirectory(dir=SCRATCH) as tmp:
        dst = Path(tmp) / "round1"
        dst.mkdir()
        shutil.copytree(SCRATCH / "base/src", dst / "src")
        diff = subprocess.run(["git", "show", "--format=", "5a80f0cc", "--", "src/bts/cli.py"],
                              cwd=ROOT, capture_output=True, check=True, timeout=3).stdout
        subprocess.run(["patch", "-p1", "-f"], cwd=dst, input=diff, check=True, timeout=3)
        result = subprocess.run([sys.executable, __file__, "replay"], cwd=dst,
                                env=os.environ | {"PYTHONPATH": f"{dst}/src:{SCRATCH}/base"},
                                check=True, timeout=4)
