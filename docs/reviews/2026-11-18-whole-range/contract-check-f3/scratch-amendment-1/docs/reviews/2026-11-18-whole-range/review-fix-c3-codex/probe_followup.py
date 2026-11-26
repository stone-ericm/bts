"""Run the remaining bounded caller and special-path comparisons."""

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import callers
from probe_support import BASE, HEAD, HERE, REPO, RUN, archive, offline, plant, python, python_env, run_cli, scratch
from reconcile_probe import menu


def callers_worker(target):
    from bts import scorecard, web

    with patch.object(web, "fetch_live_scorecard", return_value=None, create=True), \
         patch.object(scorecard, "fetch_live_scorecard", return_value=None):
        print(json.dumps(callers.invoke_checks(target), default=str, sort_keys=True))


def compare(work):
    observations = []
    for revision in (BASE, HEAD):
        source = archive(revision, work / revision, "src") / "src"
        try:
            stdout = python(source, __file__, "callers-worker", work / "data", timeout=25).stdout
        except subprocess.CalledProcessError as error:
            print(error.stdout, error.stderr, file=sys.stderr)
            raise
        observations.append(json.loads(stdout))
        (work / "data").rename(work / f"data-{revision}")
    base, head = observations
    for case, outcome in head.items():
        print(json.dumps({"caller": case, "equal": base[case] == outcome, "base": base[case], "head": outcome}))
    assert base == head


def receipt_worker(root, which):
    from bts.reconcile_receipt import receipts_root

    root.mkdir()
    path = receipts_root(root)
    if which == "date":
        path /= RUN.date().isoformat()
    path.parent.mkdir(parents=True)
    os.mkfifo(path)
    with menu():
        result = run_cli(root)
    print(json.dumps({"path": str(path), "exit": result.exit_code, "stderr": result.stderr,
                      "stdout": result.stdout, "exception": repr(result.exception), "fifo_survived": path.is_fifo()}))


def edge(work):
    cases = [
        ("old pick FIFO repeat", HERE / "reconcile_probe.py", "special-worker", ["old_pick", "fifo"]),
        ("lineup journal FIFO", HERE / "reconcile_probe.py", "special-worker", ["lineup_evolution", "fifo"]),
        ("lineup journal directory", HERE / "reconcile_probe.py", "special-worker", ["lineup_evolution", "directory"]),
        ("receipt root FIFO", __file__, "receipt-worker", ["root"]),
        ("receipt date FIFO", __file__, "receipt-worker", ["date"]),
    ]
    for name, script, entry, arguments in cases:
        directory = work / name
        directory.mkdir()
        try:
            result = python(REPO / "src", script, entry, directory / "picks", *arguments, timeout=6)
            info = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            info = {"timeout_seconds": 6}
        except subprocess.CalledProcessError as error:
            info = {"error": error.stderr, "exit": error.returncode}
        print(json.dumps({"case": name} | info), flush=True)


def dry(work):
    path = plant(work, "2026-05-01", "hit")
    before = {file.name: file.read_bytes() for file in work.iterdir()}
    result = python(REPO / "src", HERE / "dry_compare.py", work, "--today", "2026-12-31", timeout=20)
    assert before == {file.name: file.read_bytes() for file in work.iterdir()}
    print(result.stdout, end="")
    assert path.is_file()


if __name__ == "__main__":
    mode = sys.argv[1]
    with offline():
        if mode == "callers-worker":
            callers_worker(Path(sys.argv[2]))
        elif mode == "receipt-worker":
            receipt_worker(Path(sys.argv[2]), sys.argv[3])
        else:
            with scratch(mode) as directory:
                globals()[mode](Path(directory))
