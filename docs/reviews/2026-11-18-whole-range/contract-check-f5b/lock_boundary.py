"""Project a delayed last early launch onto two adjacent lock attempts."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from replay_support import World, compact, installed
from tests.test_wr_f1b_lock import _await, _records, _runner, HANG_BOUND
from tests.test_entry_intent import box


with TemporaryDirectory() as temp:
    root = Path(temp)
    installer_root = root / "installer"
    base, candidate = (
        [line for line in installed(root / v, v) if " bts check-pick-entered " in line][0]
        for v in ("base", "f5"))
    installer_root.mkdir()
    installer = box.__wrapped__(installer_root)
    for name, line in (("base", base), ("f5", candidate)):
        directory = root / ("wrap-" + name)
        directory.mkdir()
        # The F1b harness expects this fixture's HOME. The line and wrapper tokens are otherwise real.
        generated = line.replace(str(root / name / "home"), str(installer_root / "home"))
        with _runner(generated, installer, directory) as (launch, _):
            if name == "f5":
                release = directory / "release"
                holder, trace = launch("hold", str(release), duration="7")
                assert _await(lambda: dict(_records(trace)).get("ready"))
            ten, trace = launch("quick")
            code = ten.wait(timeout=HANG_BOUND)
            assert code == (1 if name == "f5" else 0)
            print(name, "10:00 lock exit", code, _records(trace))
            if name == "f5":
                release.touch()
                assert holder.wait(timeout=HANG_BOUND) == 0
    w = World(root / "ten", "base")
    w.commit_at("10:06", at="09:59")
    ten = w.cron("10:00")
    assert ten["sent"] and w.invoke("10:15")["receipt"]["outcome"] == "outside_window"
    print("if allowed through, the base's sole 10:00 run alerts:", compact(ten))
