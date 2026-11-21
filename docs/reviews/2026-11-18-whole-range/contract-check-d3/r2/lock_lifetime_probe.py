"""Model recovery contending while a mutator survives its Bash owner."""

import fcntl
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


HERE = Path(__file__).resolve().parent
SCRIPT = r'''
exec 9>>lock
"$PY" "$PROBE" lock || exit 1
if [[ $MODE == supervised ]]; then
    (
        (exec 9>&-; exec "$PY" "$PROBE" mutate) &
        wait "$!"
    ) &
else
    (exec 9>&-; exec "$PY" "$PROBE" mutate) &
fi
wait "$!"
'''


def wait_for(path):
    deadline = time.monotonic() + 1.0
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(path)
        time.sleep(0.005)


def attempt(path):
    with path.open("a") as contender:
        try:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def probe(root, mode):
    root.mkdir()
    process = subprocess.Popen(["/bin/bash", "-c", SCRIPT], cwd=root,
                               env=dict(os.environ, PY=sys.executable, PROBE=__file__, MODE=mode),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(root / "ready")
        process.kill()
        process.wait(timeout=1)
        acquired = attempt(root / "lock")
        assert acquired == (mode == "closed-early")
        assert not (root / "mutation").exists()
        print(f"{mode}: recovery acquired before surviving mutation = {acquired}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        (root / "proceed").touch()
    wait_for(root / "mutation")
    deadline = time.monotonic() + 1.0
    while not attempt(root / "lock"):
        if time.monotonic() > deadline:
            raise TimeoutError("supervisor did not release lock")
        time.sleep(0.005)


if __name__ == "__main__":
    if sys.argv[1:] == ["lock"]:
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(0)
    if sys.argv[1:] == ["mutate"]:
        Path("ready").touch()
        wait_for(Path("proceed"))
        Path("mutation").touch()
        raise SystemExit(0)
    with TemporaryDirectory(dir=HERE) as directory:
        box = Path(directory)
        probe(box / "closed", "closed-early")
        probe(box / "supervised", "supervised")
