"""Deadline-bound child probes for every contest path and the manual diagnostic reread."""

from datetime import datetime, timedelta
from itertools import product
import json
import logging
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SOURCES = {"contest_state", "saver_state"}
KINDS = ("manual", "automatic", "switch", "saver", "coverage")
SHAPES = ("fifo", "directory", "malformed", "non-UTF8", "regular-link", "dangling-link", "fifo-link")
LEGACY = "A legacy or expired manual override file is present."
COVERAGE_INFO = "Coverage trails by one settled pick, the expected ledger lag."
COVERAGE_WARN = "Coverage trails by more than one settled pick. The predictions array can trail the live counter."
SAVER = "The saver flag is uninitialized or unreadable. The saver counted as spent."
CRASH = "The contest state health check failed before it could finish."


def configure(shape, target, seed):
    target.unlink(missing_ok=True)
    if shape == "fifo":
        os.mkfifo(target)
    elif shape == "directory":
        target.mkdir()
    elif shape == "malformed":
        target.write_text("{")
    elif shape == "non-UTF8":
        target.write_bytes(b"\xc0")
    elif shape.endswith("-link"):
        destination = target.parent / "alias-target"
        if shape == "regular-link":
            destination.write_bytes(seed)
        elif shape == "fifo-link":
            os.mkfifo(destination)
        target.symlink_to(destination.name)
    else:
        raise ValueError(shape)


def snapshot(root):
    result = {}
    for path in root.iterdir():
        mode = path.lstat().st_mode
        content = (path.read_bytes() if stat.S_ISREG(mode) else os.readlink(path) if stat.S_ISLNK(mode) else None)
        result[path.name] = (stat.S_IFMT(mode), content)
    return result


def child(root, kind, shape, tree, full):
    source = {"head": HERE.parents[1] / "src", "base": HERE / "base/src"}.get(tree, Path(tree))
    sys.path[:0] = [str(source), str(HERE / "contract-check-c1")]
    import bts
    from bts.health import alert, contest_state, runner
    from bts import season_contract
    import probe

    assert Path(bts.__file__).resolve().is_relative_to(source)
    picks = root / "picks"
    day = probe.D
    wall = probe.at(day + timedelta(days=1), 2)
    manual = picks / "account_state/contest_streak.manual.json"
    targets = {
        "manual": manual,
        "automatic": probe.auto(picks, day),
        "switch": probe.switch(picks, day),
        "saver": probe.saver_flag(picks, day),
        "coverage": picks / f"{day}.json",
        "reread": manual,
    }
    if kind in ("manual", "reread"):
        probe.put(picks, manual.name, {"streak": 4})
    if kind == "coverage":
        probe.settled(picks, day)
        probe.settled(picks, day - timedelta(days=1))
    if shape == "masked":
        probe.dated(picks, day)
    target = targets[kind]
    seed = target.read_bytes()
    if kind != "reread":
        configure("fifo" if shape in ("masked", "interrupt") else shape, target, seed)
    before = snapshot(picks / "account_state")
    original_open = os.open
    original_read = Path.read_text
    safe_run = runner._safe_run
    opens = []

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz)

    def log_attempt(method, path):
        if path == target:
            opens.append(method)
            print(f"attempt {method}", flush=True)
            if shape == "interrupt":
                raise KeyboardInterrupt

    def os_open(path, *args, **kwargs):
        log_attempt("os.open", path)
        return original_open(path, *args, **kwargs)

    def read_text(path, *args, **kwargs):
        log_attempt("Path.read_text", path)
        return original_read(path, *args, **kwargs)

    if kind == "reread":
        select = contest_state._select_contest_streak_state

        def swap_after_selection(*args, **kwargs):
            state = select(*args, **kwargs)
            configure(shape, target, seed)
            return state
    else:
        swap_after_selection = None

    def isolate(name, fn):
        return safe_run(name, fn) if full or name == "contest_state" else []

    with (patch.object(runner, "datetime", Clock, create=True),
          patch.object(runner, "_safe_run", isolate),
          patch.object(alert, "send_dm", return_value="offline-ack"),
          patch.object(os, "open", os_open), patch.object(Path, "read_text", read_text),
          patch.object(contest_state, "_select_contest_streak_state", swap_after_selection, create=True)
          if kind == "reread" else patch.object(season_contract, "ET", probe.ET)):
        try:
            items = runner.run_all_checks(picks, root / "models", "offline.test", today=day,
                                          contest_state_expected=True)
        except KeyboardInterrupt:
            assert shape == "interrupt"
            assert snapshot(picks / "account_state") == before
            print(json.dumps({"interrupt": True, "attempts": opens}))
            return
    if kind != "reread":
        assert snapshot(picks / "account_state") == before
    assert shape != "interrupt"
    items = [a for a in items if a.source in SOURCES or a.incident_key == "health_runner:contest_state"]
    reasons = [a.message.split(" ET. ", 1)[1].split(" This check used files", 1)[0] for a in items]
    print(json.dumps({"attempts": opens, "reasons": reasons, "alerts": [(a.level, a.source) for a in items]}))


def run_child(root, kind, shape, *, tree="head", full=False, timeout=3):
    args = [sys.executable, __file__, "child", str(root), kind, shape, tree, "full" if full else "isolated"]
    proc = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
    return json.loads(proc.stdout.splitlines()[-1])


def expected(kind, shape):
    if shape == "regular-link":
        return {"manual": [LEGACY], "coverage": [COVERAGE_WARN]}.get(kind, [])
    if shape == "dangling-link":
        return {
            "manual": [], "automatic": ["Contest streak observation is missing from account_state.",
                                        "There is no automatic observation for the switch activation."],
            "switch": ["The season switch file is malformed or unreadable."],
            "saver": [SAVER], "coverage": [COVERAGE_INFO],
        }[kind]
    return {
        "manual": ["The manual contest streak file is malformed or unreadable."],
        "automatic": ["The automatic contest streak file is malformed or unreadable."],
        "switch": ["The season switch file is malformed or unreadable."],
        "saver": [SAVER], "coverage": [COVERAGE_INFO],
    }[kind]


def main():
    with TemporaryDirectory(dir=HERE) as scratch:
        root = Path(scratch)
        if sys.argv[1] == "reread":
            tree = sys.argv[2] if len(sys.argv) > 2 else "head"
            for shape in ("fifo", "directory", "malformed", "non-UTF8", "fifo-link"):
                outcome = run_child(root / shape, "reread", shape, tree=tree)
                assert outcome["reasons"] == ([CRASH] if shape == "non-UTF8" else [LEGACY]), outcome
                assert outcome["attempts"] == ["os.open", "os.open"], outcome
                print("reread", shape, outcome)
            return
        for kind, shape in product(KINDS, SHAPES):
            outcome = run_child(root / f"{kind}-{shape}", kind, shape)
            assert outcome["reasons"] == expected(kind, shape), (kind, shape, outcome)
            count = {"manual": int(shape != "dangling-link") + int(shape == "regular-link"),
                     "automatic": int(shape != "dangling-link"), "switch": 1,
                     "saver": int(shape != "dangling-link"), "coverage": 2}[kind]
            assert outcome["attempts"] == ["os.open"] * count, (kind, shape, outcome)
            print(kind, shape, outcome)
        for kind in ("automatic", "switch", "saver"):
            assert run_child(root / f"{kind}-masked", kind, "masked") == {
                "attempts": [], "reasons": [], "alerts": [],
            }
        assert run_child(root / "interrupt", "automatic", "interrupt") == {
            "interrupt": True, "attempts": ["os.open"],
        }
        for tree in ("base", "head"):
            try:
                run_child(root / f"{tree}-full", "coverage", "fifo", tree=tree, full=True, timeout=2)
            except subprocess.TimeoutExpired as error:
                assert (error.stdout or b"").splitlines() == [b"attempt Path.read_text"], error.stdout
                print(f"{tree}: unchanged full-run calibration reached coverage FIFO before the contest slot and blocked")
            else:
                raise AssertionError(f"Expected calibration to block on the FIFO on {tree}")


if __name__ == "__main__":
    signal.alarm(55)
    logging.disable(logging.CRITICAL)
    if sys.argv[1] == "child":
        with patch.object(socket, "socket", side_effect=AssertionError("offline")), \
             patch.object(socket, "create_connection", side_effect=AssertionError("offline")):
            child(Path(sys.argv[2]), *sys.argv[3:6], full=sys.argv[6] == "full")
    else:
        main()
