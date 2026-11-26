"""Exercise the watchdog command emitted by the Hetzner installer."""

from contextlib import contextmanager
from pathlib import Path
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest

from tests.test_entry_intent import LIVE_DM, RESEARCH, _exe, box
from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World

REPO = Path(__file__).resolve().parents[1]
FLOCK = "/usr/bin/flock"
TIMEOUT = "/usr/bin/timeout"
LOCK = "/home/bts/.check-pick-entered.lock"
WRAPPERS = f"{FLOCK} -n {LOCK} {TIMEOUT} --verbose -k 30 840"
ENTER = LIVE_DM + "contest_state_required = true\n"
ENTRY = "run bts check-pick-entered"
MARKED_LEGACY = f"*/15 10-23 * * * uv {ENTRY} # BTS-HETZNER"
HANG_BOUND = 15
TOOLS_REQUIRED = "requires /usr/bin/flock and GNU coreutils /usr/bin/timeout"

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX process and lock semantics")


@pytest.fixture
def installer(box, monkeypatch):
    # The shared installer fixture allows two minutes. Bound its process here.
    from tests import test_entry_intent

    class BoundedSubprocess:
        @staticmethod
        def run(*args, **kwargs):
            kwargs["timeout"] = HANG_BOUND
            return subprocess.run(*args, **kwargs)

    monkeypatch.setattr(test_entry_intent, "subprocess", BoundedSubprocess)
    box.config.write_text(ENTER)
    return box


@pytest.fixture
def generated(installer):
    result = installer.run("install")
    assert result.returncode == 0, result.stderr
    line, = [line for line in installer.lines() if ENTRY in line]
    return line


def test_generated_wrapper_order_and_interval(generated):
    schedule = generated.split()[:5]
    assert schedule == ["*/15", "10-23", "*", "*", "*"]
    durations = re.search(
        rf"{FLOCK} -n {re.escape(LOCK)} {TIMEOUT} --verbose -k (\d+) (\d+) \S+/uv {ENTRY}",
        generated,
    )
    assert durations, generated
    assert durations.groups() == ("30", "840")
    assert sum(map(int, durations.groups())) < 60 * int(schedule[0].split("/")[1])


def test_install_replaces_marked_jobs_and_keeps_legacy_job(installer):
    legacy = MARKED_LEGACY.removesuffix(" # BTS-HETZNER")
    preserved = ["0 9 * * * echo foreign", legacy, "# " + legacy]
    installer.cron.write_text("\n".join([*preserved, MARKED_LEGACY, "# " + MARKED_LEGACY]) + "\n")
    shown = installer.run("show")
    assert shown.returncode == 0, shown.stderr
    proposed = shown.stdout.split("Would install:\n")[1].splitlines()
    new_entry, = [line for line in proposed if ENTRY in line]
    assert WRAPPERS in new_entry
    for _ in range(2):
        result = installer.run("install")
        assert result.returncode == 0, result.stderr
        assert installer.lines() == [*preserved, *proposed]
    assert installer.run("remove").returncode == 0
    assert installer.lines() == preserved


def test_no_crontab_and_research_generated_line_parity(installer, generated):
    enter = installer.cron.read_text()
    installer.config.write_text(RESEARCH)
    result = installer.run("install")
    assert result.returncode == 0, result.stderr
    research = installer.cron.read_text()
    assert ENTRY not in research
    expected = []
    for line in enter.splitlines(keepends=True):
        if line.rstrip("\n") == generated:
            continue
        if "bts fetch-contest-streak" in line:
            line = line.replace(" --dm-recipient stonehengee.bsky.social", "")
        expected.append(line)
    assert "".join(expected) == research


@pytest.mark.parametrize("config", ["", RESEARCH.replace('"private"', '"dm"'), LIVE_DM])
def test_intent_refusal_leaves_crontab_untouched(installer, config):
    before = MARKED_LEGACY.encode() + b"\n"
    installer.cron.write_bytes(before)
    installer.config.write_text(config)
    assert installer.run("install").returncode != 0
    assert installer.cron.read_bytes() == before
    assert not Path(f"{installer.cron}.calls").exists()


@pytest.mark.parametrize("failure", ["unreadable", "filter"])
def test_install_and_remove_refuse_crontab_errors(installer, tmp_path, failure):
    before = b"0 9 * * * echo foreign\n"
    installer.cron.write_bytes(before)
    if failure == "unreadable":
        Path(f"{installer.cron}.unreadable").touch()
        shown = installer.run("show")
        assert shown.returncode == 0 and "(none)" in shown.stdout
    else:
        _exe(tmp_path / "shims/grep", '#!/bin/sh\n[ "$1" != "-v" ] || exit 2\nexec /usr/bin/grep "$@"\n')
    for action in ("install", "remove"):
        result = installer.run(action)
        assert result.returncode != 0
        assert installer.cron.read_bytes() == before
    assert "-" not in Path(f"{installer.cron}.calls").read_text().splitlines()


def _kill(pid, sig=signal.SIGKILL):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _record(label, pid=None):
    with open(os.environ["BTS_TEST_TRACE"], "a") as stream:
        stream.write(f"{label} {os.getpid() if pid is None else pid}\n")


def _flock(option, path, *command):
    import fcntl

    assert option == "-n"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 1
    os.set_inheritable(fd, True)
    os.environ["BTS_TEST_LOCK_FD"] = str(fd)
    os.execvp(command[0], command)


def _timeout(verbose, kill_after, grace, duration, *command):
    assert (verbose, kill_after) == ("--verbose", "-k")
    fd = int(os.environ["BTS_TEST_LOCK_FD"])
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(fd,))
    try:
        _record("timeout-child", child.pid)
        try:
            code = child.wait(timeout=float(duration))
            return code if code >= 0 else 128 - code
        except subprocess.TimeoutExpired:
            print("timeout: sending signal TERM", file=sys.stderr, flush=True)
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=float(grace))
        except subprocess.TimeoutExpired:
            print("timeout: sending signal KILL", file=sys.stderr, flush=True)
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=HANG_BOUND)
            return 137
        return 124
    finally:
        if child.poll() is None:
            _kill(child.pid)
            child.wait(timeout=HANG_BOUND)


def _uv(mode, *args):
    # Model uv's wrapper and child even on hosts where only the shims run.
    if mode == "kill":
        signal.signal(signal.SIGTERM, lambda *_: _record("uv-term"))
    fd = os.environ.get("BTS_TEST_LOCK_FD")
    child = subprocess.Popen(_harness("child", mode, *args), pass_fds=(int(fd),) if fd else ())
    try:
        _record("uv-child", child.pid)
        code = child.wait(timeout=HANG_BOUND)
        return code if code >= 0 else 128 - code
    finally:
        if child.poll() is None:
            _kill(child.pid)
            child.wait(timeout=HANG_BOUND)


def _child(mode, *args):
    if mode == "interrupt":
        _interrupt_at(args[0], Path(args[1]))
        return
    if mode == "term":
        def terminated(*_):
            _record("python-term")
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, terminated)
    if mode == "kill":
        signal.signal(signal.SIGTERM, lambda *_: _record("python-term"))
    if mode == "escape":
        os.setsid()
        fd = int(os.environ["BTS_TEST_LOCK_FD"])
        assert os.fstat(fd).st_ino == Path(os.environ["BTS_TEST_LOCK"]).stat().st_ino
    _record("ready")
    print(f"watchdog {mode}", flush=True)
    if mode == "hold":
        _await(lambda: Path(args[0]).exists())
    elif mode != "quick":
        time.sleep(HANG_BOUND)


def _interrupt_at(stage, root):
    # Adapt the archived killed.py probe to the shared Click fixture.
    from unittest.mock import patch

    from bts import receipt_io
    from bts.cli import _atomic_write_json

    def pause(label):
        if stage == label:
            (root / "ready").write_text(label)
            time.sleep(HANG_BOUND)
            raise AssertionError("parent did not interrupt the watchdog")

    def send(_, message):
        pause("all-clear")
        if stage == "failed-dm-before-marker":
            raise RuntimeError("synthetic send failure")
        (root / "accepted-dm").write_text(message)

    real_fdopen, real_replace = os.fdopen, os.replace

    class PartialWrite:
        def __init__(self, stream, label):
            self.stream, self.label = stream, label

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:len(data) // 2])
            self.stream.flush()
            pause(self.label)

    def write_marker(path, value):
        pause("dm-before-marker")
        pause("failed-dm-before-marker")
        if stage == "marker-temp":
            with patch("os.fdopen", lambda *a, **kw: PartialWrite(real_fdopen(*a, **kw), "marker-temp")):
                return _atomic_write_json(path, value)
        return _atomic_write_json(path, value)

    def replace(src, dest):
        real_replace(src, dest)
        path = Path(dest)
        if "pick_entry_receipts" in path.parts:
            pause("receipt-sealed" if path.suffix == ".sealed" else "receipt-renamed")

    opens = open
    if stage == "receipt-temp":
        def opens(*args, **kwargs):
            return PartialWrite(open(*args, **kwargs), "receipt-temp")

    with patch.object(receipt_io, "open", opens, create=True):
        result = World(root).check("18:30", MATCH, more_patches=[
            ("bts.leaderboard.auth.load_session_cookies", lambda: pause("before-fetch") or {}),
            ("bts.contest_fetch.fetch_pending_predictions", lambda *_: pause("during-fetch") or MATCH),
            ("bts.cli._atomic_write_json", write_marker),
            ("bts.dm.send_dm", send),
            ("os.replace", replace),
        ])
    raise AssertionError(f"watchdog returned before the interruption at {stage}: {result}")


def _harness(role, *args):
    return [sys.executable, "-m", "tests.test_wr_f1b_lock", role, *args]


def _await(predicate):
    deadline = time.monotonic() + HANG_BOUND
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.01)
    raise AssertionError("watchdog did not reach the expected state before the deadline")


def _records(trace):
    if not trace.exists():
        return []
    return [(label, int(pid)) for label, pid in (line.split() for line in trace.read_text().splitlines())]


def _wait_for_unlock(lock):
    import fcntl

    def lock_released():
        with lock.open("r+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True

    _await(lock_released)


def _require_real_tools():
    if not Path(FLOCK).is_file() or not Path(TIMEOUT).is_file():
        pytest.skip(TOOLS_REQUIRED)
    version = subprocess.run([TIMEOUT, "--version"], capture_output=True, timeout=HANG_BOUND)
    if version.returncode or b"GNU coreutils" not in version.stdout:
        pytest.skip(TOOLS_REQUIRED)


@contextmanager
def _runner(generated, installer, tmp_path, *, real=False):
    if real:
        _require_real_tools()
    home = installer.config.parent
    log = home / "logs/cron.log"
    log.parent.mkdir(exist_ok=True)
    log.touch()
    lock = tmp_path / "watchdog.lock"
    runs = []
    # The shell changes directory to the fake checkout before running the harness.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (str(REPO), os.environ.get("PYTHONPATH")))),
               BTS_TEST_LOCK=str(lock))

    def launch(mode, *args, duration="4", actual_uv=False):
        trace = tmp_path / f"processes-{len(runs)}"
        shell = generated.split(maxsplit=5)[5]
        flock = _harness("exec", FLOCK) if real else _harness("flock")
        timer = _harness("exec", TIMEOUT) if real else _harness("timeout")
        shell = shell.replace(FLOCK, shlex.join(flock), 1).replace(LOCK, shlex.quote(str(lock)), 1)
        shell = shell.replace(f"{TIMEOUT} --verbose -k 30 840", f"{shlex.join(timer)} --verbose -k 0.3 {duration}", 1)
        command = _harness("uv", mode, *args)
        if actual_uv:
            uv = shutil.which("uv")
            assert uv, "the Linux gate must provide uv"
            command = _harness("exec", uv, "run", "--offline", "--no-project", "--no-sync",
                               *_harness("child", mode, *args))
        shell = re.sub(r"\S+/uv run bts check-pick-entered\b", lambda _: shlex.join(command), shell, count=1)
        process = subprocess.Popen(["/bin/sh", "-c", shell], env={**env, "BTS_TEST_TRACE": str(trace)},
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        runs.append((process, trace))
        return process, trace

    try:
        yield launch, log
    finally:
        for process, trace in reversed(runs):
            if process.poll() is None:
                _kill(process.pid)
            for pid in {pid for _, pid in _records(trace)}:
                _kill(pid)
            process.wait(timeout=HANG_BOUND)


@pytest.mark.parametrize("real", [False, True], ids=["shim", "real"])
def test_k2_generated_job_contends_then_runs(generated, installer, tmp_path, real):
    release = tmp_path / "release"
    with _runner(generated, installer, tmp_path, real=real) as (launch, log):
        holder, trace = launch("hold", str(release), duration="7")
        _await(lambda: dict(_records(trace)).get("ready"))
        _await(lambda: "watchdog hold" in log.read_text())
        before = log.read_bytes()

        contender, trace = launch("quick")
        assert contender.wait(timeout=HANG_BOUND) == 1
        assert not {"uv", "child", "ready"} & dict(_records(trace)).keys()
        assert log.read_bytes() == before
        assert holder.poll() is None

        release.touch()
        assert holder.wait(timeout=HANG_BOUND) == 0
        successor, trace = launch("quick")
        assert successor.wait(timeout=HANG_BOUND) == 0
        assert dict(_records(trace)).get("ready")


@pytest.mark.parametrize("real", [False, True], ids=["shim", "real"])
@pytest.mark.parametrize(("mode", "code"), [("term", 124), ("kill", 137)])
def test_k3_generated_job_times_out_and_releases_lock(generated, installer, tmp_path, real, mode, code):
    with _runner(generated, installer, tmp_path, real=real) as (launch, log):
        assert WRAPPERS in generated
        actual_uv = real and mode == "term"
        holder, trace = launch(mode, duration="3" if actual_uv else "2", actual_uv=actual_uv)
        assert holder.wait(timeout=HANG_BOUND) == code
        _await(lambda: {"ready", "python-term"} <= dict(_records(trace)).keys())
        assert "sending signal TERM" in log.read_text()
        if mode == "kill":
            assert "uv-term" in dict(_records(trace))
            assert "sending signal KILL" in log.read_text()
        _wait_for_unlock(tmp_path / "watchdog.lock")
        successor, trace = launch("quick")
        assert successor.wait(timeout=HANG_BOUND) == 0
        assert dict(_records(trace)).get("ready")


def test_k3_escaped_descendant_can_keep_the_lock(generated, installer, tmp_path):
    assert WRAPPERS in generated
    with _runner(generated, installer, tmp_path) as (launch, _):
        holder, trace = launch("escape", duration="2")
        pid = _await(lambda: dict(_records(trace)).get("ready"))
        assert holder.wait(timeout=HANG_BOUND) == 124
        blocked, _ = launch("quick")
        assert blocked.wait(timeout=HANG_BOUND) == 1
        _kill(pid)
        _wait_for_unlock(tmp_path / "watchdog.lock")
        successor, trace = launch("quick")
        assert successor.wait(timeout=HANG_BOUND) == 0
        assert dict(_records(trace)).get("ready")


@pytest.mark.parametrize("stage", [
    "before-fetch", "during-fetch", "dm-before-marker", "failed-dm-before-marker",
    "marker-temp", "receipt-temp", "receipt-renamed", "receipt-sealed", "all-clear",
])
def test_interrupted_cli_keeps_only_replaced_marker_and_sealed_receipt(generated, installer, tmp_path, stage):
    from bts import entry_receipt

    assert WRAPPERS in generated
    world = World(tmp_path / "world")
    world.commit(pitch=PITCH)
    world.capture([RIGHT if stage == "all-clear" else WRONG])
    missing = stage == "failed-dm-before-marker"
    if not missing:
        assert world.check("18:25")["marker"]["escalations"] == ["initial"]
    prior = world.marker.read_bytes() if not missing else None
    receipts = entry_receipt.discover(world.picks, DAY)
    with _runner(generated, installer, tmp_path) as (launch, _):
        holder, trace = launch("interrupt", stage, str(world.root), duration="7")
        _await(lambda: (world.root / "ready").exists() or holder.poll() is not None)
        assert (world.root / "ready").exists(), "watchdog exited before the requested interruption"
        pid = dict(_records(trace))["child"]
        sig = signal.SIGKILL if stage in {"marker-temp", "receipt-renamed"} else signal.SIGTERM
        _kill(pid, sig)
        assert holder.wait(timeout=HANG_BOUND) == 128 + sig
        successor, _ = launch("quick")
        assert successor.wait(timeout=HANG_BOUND) == 0

    replaced = stage in {"receipt-temp", "receipt-renamed", "receipt-sealed", "all-clear"}
    marker = world.read_marker()
    if replaced:
        assert marker["status"] == ("confirmed" if stage == "all-clear" else "alerted")
        if stage != "all-clear":
            assert marker["game_unresolved"] is True
            assert marker["escalations"] == ["initial", "wrong_game"]
    else:
        assert (world.marker.read_bytes() if world.marker.exists() else None) == prior
    after = entry_receipt.discover(world.picks, DAY)
    assert len(after) == len(receipts) + (stage == "receipt-sealed")
    marker_temps = list(world.marker.parent.glob("*.tmp"))
    receipt_dir = entry_receipt.receipts_root(world.picks) / DAY
    assert bool(marker_temps) == (stage == "marker-temp")
    assert bool(list(receipt_dir.glob("*.tmp"))) == (stage == "receipt-temp")
    assert bool([path for path in receipt_dir.glob("*.json")
                 if not path.with_suffix(".json.sealed").exists()]) == (stage == "receipt-renamed")
    next_check = world.check("18:45", MATCH)
    if stage == "all-clear":
        assert next_check["fetches"] == next_check["dm"] == []
    elif stage == "dm-before-marker":
        assert (world.root / "accepted-dm").read_text() == next_check["dm"][0]
    else:
        assert bool(next_check["dm"]) == (not replaced)


@pytest.mark.skipif(os.environ.get("BTS_F1B_OVERLAP_DIAGNOSTIC") != "1",
                    reason="set BTS_F1B_OVERLAP_DIAGNOSTIC=1 to run the archived K4 diagnostic")
def test_k4_unwrapped_overlapping_invocations_preserve_newer_observation(tmp_path):
    # Adapt the archived interleaving.py probe, keeping its safety assertion.
    from click.testing import CliRunner

    world = World(tmp_path)
    world.commit(pitch=PITCH)
    assert world.check("18:25")["marker"]["escalations"] == ["initial"]
    world.capture([WRONG])
    original_runner = world.runner

    def older_pending(*args, **kwargs):
        world.runner = CliRunner()
        try:
            second = world.check("18:45", MATCH, dm="failure")
        finally:
            world.runner = original_runner
        assert second["marker"]["game_unresolved"] is True
        assert second["marker"]["escalations"] == ["initial"]
        return []

    world.check("18:30", more_patches=[("bts.contest_fetch.fetch_pending_predictions", older_pending)])
    world.capture([RIGHT], f"{DAY}T230000Z")
    final = world.check("19:00", MATCH)
    receipt = next(item for item in final["receipts"] if item["attempt_id"] == final["marker"]["receipt"])
    assert receipt["qualification"]["slots"][0]["state"] == "unit_unverified"
    all_clear = f"✅ BTS pick entry confirmed for {DAY} (match)."
    assert not (final["marker"]["status"] == "confirmed" or all_clear in final["attempted_dm"]), (
        "An overlapping no-entry run erased the wrong-game observation, then a batter match gave an all-clear."
    )


if __name__ == "__main__":
    role, *args = sys.argv[1:]
    _record(role)
    if role == "exec":
        os.execv(args[0], args)
    roles = {"flock": _flock, "timeout": _timeout, "uv": _uv, "child": _child}
    raise SystemExit(roles[role](*args))
