"""Run the bounded pytest families serially under the review wrapper."""

import os
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = Path("src/bts/health/alert.py")
LEAD = "tests/health/test_alert_status_cs6b2_review.py"
CS6B2 = [
    "tests/health/test_alert_status_cs6b2.py", "tests/test_alert_status_callers_cs6b2.py",
    LEAD, "tests/test_alert_status_cs6b2_carbon.py",
]
NEIGHBORS = [
    "tests/test_lightgbm_optional.py", "tests/test_web_saver.py",
    "tests/test_morning_state_cs6b1.py", "tests/test_saved_pick_paths_cs5a.py",
    "tests/health/test_alert.py",
]
RACES = [f"{LEAD}::test_fifo_swapped_in_before_the_{side}_cannot_block" for side in ("read", "write")]
GUARD = '        if not stat.S_ISREG(os.fstat(fd).st_mode):\n            raise OSError("not a regular file")\n'
TRUNCATE = "        os.ftruncate(fd, 0)\n"
WRITE_OPEN = "os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK"
READ_OPEN = "os.O_RDONLY | os.O_NONBLOCK"


def archive(commit, folder):
    folder.mkdir()
    with subprocess.Popen(["git", "archive", commit, "src"], cwd=ROOT, stdout=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as bundle:
            bundle.extractall(folder, filter="data")
        assert proc.wait() == 0


def run(name, files, basetemp, *, overlay=None, year=None, extra=()):
    paths = [HERE, ROOT]
    if overlay:
        paths.insert(1, overlay / "src")
    env = dict(os.environ, CS6B2_OFFLINE="1", TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(map(str, paths)))
    if year:
        env["DYLD_INSERT_LIBRARIES"] = str(HERE / "realtime.dylib")
        instant = datetime(year, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))
        env["CS6B2_WALL_EPOCH"] = str(int(instant.timestamp()))
    check = (
        "from datetime import datetime; import time, socket;"
        "from bts.health import alert;"
        "print(alert.__file__, datetime.now(), time.time(), time.monotonic(),"
        " socket.getaddrinfo.__module__)"
    )
    log_path = HERE / f"{name}.log"
    with log_path.open("w") as log:
        info = subprocess.check_output([sys.executable, "-c", check], cwd=ROOT, env=env, text=True, timeout=20)
        assert str((overlay or ROOT) / SOURCE) in info and "sitecustomize" in info, info
        if year:
            assert f"{year}-07-15 12:00:" in info, info
        log.write(info)
        log.flush()
        command = [
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "review_state",
            "-q", "--tb=short", "--basetemp", str(basetemp / name), *extra, *files,
        ]
        status = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=200).returncode
    data = log_path.read_text()
    summary = next((line for line in reversed(data.splitlines()) if re.search(r" in \d+\.\d+s", line)), "no summary")
    print(f"{name}: rc={status}; {summary}", flush=True)
    return status, data


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def mutations(source):
    writer, reader = source.split("def _read_regular_file_text", 1)
    yield "read-blocking", writer + "def _read_regular_file_text" + replace_once(reader, READ_OPEN, "os.O_RDONLY")
    yield "write-blocking", replace_once(writer, WRITE_OPEN, "os.O_WRONLY | os.O_CREAT") + "def _read_regular_file_text" + reader
    yield "no-read-fstat", writer + "def _read_regular_file_text" + replace_once(reader, GUARD, "")
    yield "no-write-fstat", replace_once(writer, GUARD, "") + "def _read_regular_file_text" + reader
    yield "truncate-before-fstat", replace_once(writer, GUARD + TRUNCATE, TRUNCATE + GUARD) + "def _read_regular_file_text" + reader
    yield "no-truncate", replace_once(writer, TRUNCATE, "") + "def _read_regular_file_text" + reader
    yield "leak-read-fd", writer + "def _read_regular_file_text" + replace_once(reader, "        os.close(fd)", "        pass")
    yield "leak-write-fd", replace_once(writer, "        os.close(fd)", "        pass") + "def _read_regular_file_text" + reader


def main():
    failed = []
    with tempfile.TemporaryDirectory(prefix="pytest-", dir=HERE) as local:
        folder = Path(local)
        for year in (None, 2027, 2028):
            for order, files in (("6b2-first", CS6B2 + NEIGHBORS), ("6b2-last", NEIGHBORS + CS6B2)):
                case = f"combined-{year or 'native'}-{order}"
                if run(case, files, folder, year=year)[0] != 0:
                    failed.append(case)
        prior = folder / "round1"
        archive("38d707bd", prior)
        status, output = run("lead-round1", RACES, folder, overlay=prior)
        if not (status == 1 and "7 failed" in output and output.count(
                "Failed: dispatcher blocked on a replaced status path") == 5):
            failed.append("lead-round1 expected five alarm failures and two warning-assertion failures")
        if run("lead-head", RACES, folder)[0] != 0:
            failed.append("lead-head")
        scratch = folder / "scratch"
        archive("af15e511", scratch)
        source = (scratch / SOURCE).read_text()
        for name, mutant in mutations(source):
            (scratch / SOURCE).write_text(mutant)
            status, _ = run(f"mutant-{name}", CS6B2, folder, overlay=scratch, extra=("-x",))
            if status not in (0, 1):
                failed.append(f"mutant-{name} unexpected exit {status}")
    print(f"unexpected results: {failed}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
