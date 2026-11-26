"""Drive a fake crontab, the installed shell line, BSD timeout and the CLI."""

from contextlib import ExitStack
from datetime import date
from errno import EISDIR
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from click.testing import CliRunner

from bts import entry_receipt, receipt_io
from bts.cli import _atomic_write_json, cli
from tests.test_entry_intent import LIVE_DM, RESEARCH, _exe, box
from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World, XW, row


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INSTALLER = ROOT / "scripts/cron-setup-hetzner.sh"
WRAPPER = "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840"
COMMAND = "bts check-pick-entered"
MARK = " # BTS-HETZNER"
ALL_CLEAR = "✅ BTS pick entry confirmed"


def run_installer(fixture, action, script=INSTALLER):
    from tests import test_entry_intent

    with patch.object(test_entry_intent, "SCRIPT", script):
        result = fixture.run(action)
    assert result.returncode == 0, (action, result.stdout, result.stderr)
    return result.stdout


def install_parity(root):
    fixture = box.__wrapped__(root)
    old = root / "base-cron.sh"
    old.write_bytes(subprocess.check_output(["git", "show", "1e69d0f0:scripts/cron-setup-hetzner.sh"],
                                            cwd=ROOT, timeout=3))
    outputs = {}
    for intent, config in (("research", RESEARCH), ("enter", LIVE_DM + "contest_state_required = true\n")):
        fixture.config.write_text(config)
        before, after = (run_installer(fixture, "show", path) for path in (old, INSTALLER))
        assert before == after.replace(WRAPPER + " ", "")
        if intent == "research":
            assert before == after and COMMAND not in after.replace("check-pick-entered left out", "")
        run_installer(fixture, "install", old)
        before = fixture.cron.read_bytes()
        run_installer(fixture, "install")
        after = fixture.cron.read_bytes()
        assert before == after.replace((WRAPPER + " ").encode(), b"")
        outputs[intent] = len(after.splitlines())
    old_job, = (ln for ln in before.decode().splitlines() if COMMAND in ln)
    installed, = (ln for ln in after.decode().splitlines() if COMMAND in ln)
    unmarked = old_job.removesuffix(MARK)
    preserved = "17 1 * * * echo other\n" + unmarked + "\n# " + unmarked + "\n"
    fixture.cron.write_text(preserved + old_job + "\n# " + old_job + "\n")
    for _ in range(2):
        run_installer(fixture, "install")
        assert fixture.cron.read_text() == preserved + after.decode()
    run_installer(fixture, "remove")
    assert fixture.cron.read_text() == preserved
    assert installed.split()[:5] == ["*/15", "10-23", "*", "*", "*"]
    assert 30 + 840 < 60 * int(installed.split()[0].split("/")[1])
    (HERE / "installed-cron.txt").write_text(installed.replace(str(root), "<fixture>") + "\n")
    print("byte comparison, show and install", outputs, "legacy active/comment retained; marked both replaced", flush=True)
    return fixture, installed, unmarked


def record(label, **detail):
    with open(os.environ["REVIEW_TRACE"], "a") as stream:
        stream.write(json.dumps({"label": label, "pid": os.getpid(), **detail}) + "\n")


def flock(path, *command):
    # Existing files/directories only need read access with util-linux flock.
    try:
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDONLY, 0o600)
        except OSError as exc:
            if exc.errno != EISDIR:
                raise
            fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 1
    except OSError as exc:
        print(f"flock fixture: {exc}", file=sys.stderr)
        return 73
    os.set_inheritable(fd, True)
    os.environ["REVIEW_LOCK_FD"] = str(fd)
    record("lock")
    os.execv(command[0], command)


def broker(args):
    record("broker")
    mode = json.loads(os.environ["REVIEW_CASE"])["signal_mode"]
    if mode == "kill":
        signal.signal(signal.SIGTERM, lambda *_: record("broker-term"))
    child = subprocess.Popen([sys.executable, __file__, "worker", *args],
                             pass_fds=(int(os.environ["REVIEW_LOCK_FD"]),))
    return child.wait(timeout=8)


def worker(args):
    options = json.loads(os.environ["REVIEW_CASE"])
    stage = options.get("stage")
    release = options.get("release")
    lock_fd = os.environ.get("REVIEW_LOCK_FD")
    inherited = False
    if lock_fd is not None:
        try:
            os.fstat(int(lock_fd))
            inherited = True
        except OSError:
            pass
    record("cli", ppid=os.getppid(), pgid=os.getpgrp(), inherited=inherited)

    if options.get("signal_mode"):
        mode = options["signal_mode"]
        if mode == "escape":
            os.setsid()

        def terminated(*_):
            record("term")
            if mode == "term":
                raise SystemExit(143)

        signal.signal(signal.SIGTERM, terminated)

    def pause(label):
        if label != stage:
            return
        record("paused", stage=label)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if release and Path(release).exists():
                return
            time.sleep(0.015)
        raise RuntimeError(f"fixture wait ended at {label}")

    pending = {"match": MATCH, "other": [row(player=3)]}.get(options.get("pending"), [])
    day, at = options.get("day", DAY), options.get("at", "18:30")
    crosswalk = {} if options.get("gap") else XW

    def load_cookies():
        pause("before-fetch")
        return {}

    def fetch(*_):
        pause("fetch")
        delay = options.get("delay", 0)
        if delay:
            stop = time.monotonic() + delay
            while time.monotonic() < stop:
                time.sleep(0.03)
        return pending

    def send(_, message):
        record("attempt", message=message)
        pause("send")
        if options.get("fail"):
            raise RuntimeError("fixture DM failed")
        record("sent", message=message)

    real_fdopen, real_replace = os.fdopen, os.replace

    class PartialWrite:
        def __init__(self, stream, label):
            self.stream = stream
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:len(data) // 2])
            self.stream.flush()
            pause(self.label)

    def marker(path, value):
        pause("before-marker")
        if stage == "marker-temp":
            with patch("os.fdopen", lambda *a, **k: PartialWrite(real_fdopen(*a, **k), stage)):
                return _atomic_write_json(path, value)
        return _atomic_write_json(path, value)

    def replace(src, dest):
        real_replace(src, dest)
        path = Path(dest)
        if "pick_entry_receipts" in path.parts:
            pause("sealed" if path.suffix == ".sealed" else "receipt-renamed")

    with ExitStack() as stack:
        for target, fn in {
            "bts.leaderboard.auth.load_session_cookies": load_cookies,
            "bts.leaderboard.auth.extract_uid": lambda _: "fixture",
            "bts.leaderboard.auth.fetch_login_session": lambda **_: SimpleNamespace(
                user_id="fixture", username="stonehengee", xsid="fixture"),
            "bts.contest_fetch.fetch_profile": lambda *_: {"predictions": []},
            "bts.contest_fetch.fetch_pending_predictions": fetch,
            "bts.cli._fetch_rounds": lambda: {1: date.fromisoformat(day)},
            "bts.cli._fetch_bts_to_mlb": lambda: crosswalk,
            "bts.dm.send_dm": send,
            "bts.cli._atomic_write_json": marker,
            "os.replace": replace,
        }.items():
            stack.enter_context(patch(target, fn))
        if stage == "receipt-temp":
            stack.enter_context(patch.object(receipt_io, "open", lambda *a, **k: PartialWrite(open(*a, **k), stage),
                                             create=True))
        result = CliRunner().invoke(cli, args + ["--now-et", f"{day}T{at}:00"])
    record("finished", code=result.exit_code, out=result.stdout, err=result.stderr,
           exception=repr(result.exception))
    print(result.output, end="", flush=True)
    return result.exit_code


class Runs:
    def __init__(self, root):
        self.root = root
        self.fixture, self.line, self.legacy = install_parity(root)
        self.lock = root / "pick-entry.lock"
        self.uv = self.fixture.config.parent / ".local/bin/uv"
        self.data = self.fixture.config.parent / "projects/bts/data"
        self.log = self.fixture.config.parent / "logs/cron.log"
        self.processes = []
        self.env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT)))}

    def world(self, day=DAY, *, pitch=PITCH):
        world = World(self.data, day=day)
        world.commit(pitch=pitch)
        return world

    def launch(self, *, legacy=False, real_uv=False, use_broker=False, lock=None, duration=3, grace=0.25, **case):
        script = shlex.join([sys.executable, __file__])
        cmd = (self.legacy if legacy else self.line).split(maxsplit=5)[5]
        if not legacy:
            wrapper = f"{script} flock {shlex.quote(str(lock or self.lock))} /usr/bin/timeout --verbose -k {grace} {duration}"
            assert WRAPPER in cmd
            cmd = cmd.replace(WRAPPER, wrapper, 1)
        command = f"{script} {'broker' if use_broker else 'worker'}"
        if real_uv:
            uv = shutil.which("uv")
            assert uv
            command = f"{shlex.quote(uv)} run --offline --no-project --no-sync {command}"
        cmd = cmd.replace(f"{self.uv} run bts", command, 1)
        trace = self.root / f"events-{len(self.processes):03d}"
        started = time.monotonic()
        proc = subprocess.Popen(["/bin/sh", "-c", cmd], start_new_session=True,
                                env={**self.env, "REVIEW_CASE": json.dumps(case), "REVIEW_TRACE": str(trace)})
        self.processes.append((proc, trace))
        print(f"shell pid={proc.pid} trace={trace.name} {case}", flush=True)
        return proc, trace, started

    def close(self):
        for proc, trace in reversed(self.processes):
            if proc.poll() is None:
                for pid in {e["pid"] for e in events(trace)} | {proc.pid}:
                    kill(pid)
                proc.wait(timeout=3)
        (HERE / f"trace-{sys.argv[1]}.json").write_text(json.dumps(
            {trace.name: {"shell_pid": proc.pid, "status": proc.returncode, "events": events(trace)}
             for proc, trace in self.processes}, ensure_ascii=False, indent=2))


def events(trace):
    if not trace.exists():
        return []
    return [json.loads(line) for line in trace.read_text().splitlines() if line.endswith("}")]


def reached(launch, label):
    process, trace, start = launch
    while time.monotonic() < start + 6:
        if found := next((e for e in events(trace) if e["label"] == label), None):
            return found
        if process.poll() is not None:
            break
        time.sleep(0.01)
    raise AssertionError((label, process.poll(), events(trace)))


def result(launch, expected):
    process, trace, started = launch
    assert process.wait(timeout=8) == expected, (process.returncode, expected, events(trace))
    print(f"{trace.name} rc={expected} elapsed={time.monotonic()-started:.3f}s", flush=True)
    return events(trace)


def kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def messages(steps):
    return [s["message"] for s in steps if s["label"] == "sent"]


def serial(run):
    world = run.world()
    for index, bridge in enumerate(({"pending": "match", "gap": True}, {}, {"pending": "other"})):
        world.marker.unlink(missing_ok=True)
        conflict = world.capture([WRONG], f"{DAY}T230000Z")
        conflict.unlink()
        world.capture([WRONG])
        if index != 1:
            result(run.launch(at="18:25"), 1)
        result(run.launch(at="18:35", pending="match", fail=True), 1)
        assert world.read_marker()["game_unresolved"]
        result(run.launch(at="18:45", **bridge), 0 if bridge.get("gap") else 1)
        assert world.read_marker()["game_unresolved"]
        world.capture([RIGHT], f"{DAY}T230000Z")
        final = result(run.launch(at="19:00", pending="match"), 1)
        assert world.read_marker()["game_unresolved"]
        assert len(messages(final)) == 1 and "BTS entry game needs verification" in messages(final)[0]
        assert ALL_CLEAR not in messages(final)[0]
        conflict.unlink()
    world.capture([RIGHT])
    final = result(run.launch(at="19:01", pending="match"), 0)
    assert world.read_marker()["status"] == "confirmed" and ALL_CLEAR in messages(final)[0]


def overlap(run):
    world = run.world()
    world.capture([WRONG])
    result(run.launch(at="18:25"), 1)
    release = run.root / "release"
    older = run.launch(stage="fetch", release=str(release), duration=6)
    reached(older, "paused")
    output = run.log.read_bytes()
    contender = result(run.launch(at="18:45", pending="match", fail=True), 1)
    assert not contender and run.log.read_bytes() == output
    assert older[0].poll() is None
    release.touch()
    result(older, 1)
    result(run.launch(at="18:45", pending="match", fail=True), 1)
    assert world.read_marker()["game_unresolved"]
    print("installed contention skipped without starting or writing; next check observed wrong game", flush=True)

    # Keep the same generated legacy shell alive across a fresh install.
    world.marker.unlink()
    result(run.launch(at="18:25"), 1)
    release.unlink()
    older = run.launch(legacy=True, stage="fetch", release=str(release))
    reached(older, "paused")
    run_installer(run.fixture, "install")
    result(run.launch(at="18:45", pending="match", fail=True), 1)
    assert world.read_marker()["game_unresolved"]
    release.touch()
    result(older, 1)
    assert "game_unresolved" not in world.read_marker()
    world.capture([RIGHT], f"{DAY}T230000Z")
    final = result(run.launch(at="19:00", pending="match"), 0)
    assert ALL_CLEAR in messages(final)[0]
    print("already-running unwrapped line crossed install and recreated false all-clear", flush=True)


def interrupts(run):
    world = run.world()
    stages = ("before-fetch", "fetch", "send", "before-marker", "marker-temp", "receipt-temp", "receipt-renamed", "sealed")
    for stage in stages:
        world.marker.unlink(missing_ok=True)
        world.capture([WRONG])
        world.capture([RIGHT], f"{DAY}T230000Z").unlink()
        result(run.launch(at="18:25"), 1)
        prior = world.marker.read_bytes()
        sealed = len(entry_receipt.discover(world.picks, DAY))
        job = run.launch(at="18:45", pending="match", stage=stage, duration=1.4)
        reached(job, "paused")
        result(job, 124)
        replaced = stage in {"receipt-temp", "receipt-renamed", "sealed"}
        assert (world.marker.read_bytes() != prior) == replaced
        assert len(entry_receipt.discover(world.picks, DAY)) == sealed + (stage == "sealed")
        world.capture([RIGHT], f"{DAY}T230000Z")
        next_slot = result(run.launch(at="19:00", pending="match"), 1 if replaced else 0)
        assert any(ALL_CLEAR in msg for msg in messages(next_slot)) != replaced
        print(stage, "new marker" if replaced else "prior marker", flush=True)


def limits(run):
    world = run.world()
    world.capture([WRONG])
    for _ in range(2):
        job = run.launch(pending="match", stage="fetch", duration=1.4, real_uv=True, signal_mode="term")
        reached(job, "paused")
        signal_events = result(job, 124)
        assert any(e["label"] == "term" for e in signal_events)
        assert world.read_marker() is None
    assert run.lock.is_file()
    result(run.launch(pending="match"), 1)
    assert world.read_marker()["game_unresolved"]

    # A trickling response can finish with its first warning only before the submission cutoff.
    world.marker.unlink()
    result(run.launch(pending="match", at="19:34", delay=1.5, duration=0.8), 124)
    assert world.read_marker() is None
    result(run.launch(pending="match", at="19:45"), 0)
    assert world.read_marker() is None
    completed = result(run.launch(pending="match", at="19:34", delay=1.5, legacy=True), 1)
    assert len(messages(completed)) == 1
    assert world.check("19:34", MATCH, elapsed=61)["dm"] == []

    # Lock path shape and stale inode. Compare a control after each failure.
    missing = run.root / "absent-parent/lock"
    bad = result(run.launch(lock=missing, pending="match"), 73)
    assert not bad and "flock fixture" in run.log.read_text()
    lock_dir = run.root / "lock-dir"
    lock_dir.mkdir()
    result(run.launch(lock=lock_dir, pending="match"), 1)
    readonly = run.root / "readonly"
    readonly.touch(mode=0o400)
    result(run.launch(lock=readonly, pending="match"), 1)

    # At 23:45 ET the launch-date still selects the old year's inventory.
    old_day, new_day = "2027-12-31", "2028-01-01"
    old = run.world(old_day, pitch="2028-01-01T00:20:00-05:00")
    old.capture([WRONG], "2027-12-31T230000Z")
    old.capture([RIGHT], "2028-01-01T020000Z")
    result(run.launch(day=old_day, at="23:45", pending="match", stage="send", duration=1.4), 124)
    assert old.read_marker()["date"] == DAY
    first = run.world(new_day, pitch="2028-01-01T11:10:00-05:00")
    final = result(run.launch(day=new_day, at="10:15", pending="match"), 0)
    assert first.read_marker()["date"] == new_day and not messages(final)

    first.marker.unlink(missing_ok=True)
    job = run.launch(day=new_day, at="10:30", stage="before-fetch", signal_mode="kill", duration=1.4, use_broker=True)
    reached(job, "paused")
    killed = result(job, 137)
    assert {"term", "broker-term"} <= {e["label"] for e in killed}
    assert first.read_marker() is None
    job = run.launch(day=new_day, at="10:30", stage="before-fetch", signal_mode="escape", duration=1.4, use_broker=True)
    child = reached(job, "paused")
    try:
        result(job, 124)
        assert not result(run.launch(day=new_day, at="10:45"), 1)
    finally:
        kill(child["pid"])
    deadline = time.monotonic() + 2
    with run.lock.open() as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
    result(run.launch(day=new_day, at="11:00"), 1)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "flock":
        raise SystemExit(flock(*args))
    if mode == "worker":
        raise SystemExit(worker(args))
    if mode == "broker":
        raise SystemExit(broker(args))
    with tempfile.TemporaryDirectory(dir=HERE) as scratch:
        run = Runs(Path(scratch))
        try:
            {"serial": serial, "overlap": overlap, "interrupts": interrupts, "limits": limits}[mode](run)
        finally:
            run.close()
