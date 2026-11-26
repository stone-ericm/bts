"""Compare day-outcome with the same sealed receipt and an unreadable saved pick."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from support import HERE, ROOT, DAY, Replay, source
from marker_health import ReadBlocked, deadline


def read(root, version):
    source(version)
    world = Replay(root)
    if version == "base":
        with deadline():
            code, result, dms = world.morning()
        assert code == 1 and result["state"] == "not_entered" and len(dms) == 1, (code, result, dms)
        print("morning on FIFO:", code, result["state"], dms)
        return
    try:
        with deadline():
            world.morning()
    except AssertionError as exc:
        assert isinstance(exc.args[0][1], ReadBlocked), exc
        print("morning on FIFO: blocked before output or the NOT ENTERED DM")
    else:
        raise AssertionError("expected the early-receipt path to open the FIFO")


def pair():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        world = Replay(root)
        world.commit_at("10:05", at="09:40")
        early = world.invoke("09:45")
        assert early["receipt"]["early_cron_run"] is True and early["receipt"]["verifier"]["reason"] == "no_pick"
        saved_pick = world.picks / f"{DAY}.json"
        saved_pick.unlink()
        os.mkfifo(saved_pick)
        for version, path in (("base", HERE / "base/src"), ("f5", ROOT / "src")):
            subprocess.run([sys.executable, __file__, str(root), version], check=True, timeout=5,
                           env={**os.environ, "PYTHONPATH": str(path)})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        read(Path(sys.argv[1]), sys.argv[2])
    else:
        pair()
