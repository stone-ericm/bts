"""Exercise complete heartbeat CLI calls with stock Python and no network."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import importlib.util
from io import BytesIO, StringIO
from itertools import product
import json
import os
from pathlib import Path
from subprocess import run
import sys
from tarfile import open as tar_open
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = "scripts/check_heartbeat.py"
NOW = datetime(2027, 7, 15, 12, 0, tzinfo=timezone.utc)
URL = "https://checks.invalid/cron"


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def load(path):
    spec = importlib.util.spec_from_file_location("heartbeat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.datetime = FixedDatetime
    return module


def get_result(mod, heartbeat, churnfile, no_churn=False):
    history = []
    queries = []
    argv = [SCRIPT, "--heartbeat-path", str(heartbeat), "--ping-url", URL, "--churn-state", str(churnfile)]
    if no_churn:
        argv.append("--no-churn")

    def ping(request, timeout):
        history.append((request.full_url, request.get_method(), timeout))
        return BytesIO(b"OK")

    with (
        patch.object(sys, "argv", argv),
        patch.object(mod.urllib.request, "urlopen", ping),
        patch.object(mod, "read_nrestarts", lambda unit: queries.append(unit) or 0),
        redirect_stdout(StringIO()) as out,
        redirect_stderr(StringIO()) as err,
    ):
        try:
            mod.main()
        except SystemExit as exc:
            rc = exc.code
    return rc, out.getvalue(), err.getvalue(), history, queries


print("Python", sys.version.split()[0], flush=True)
with TemporaryDirectory(prefix="heartbeat-review-", dir=HERE) as folder:
    temp = Path(folder)
    tree = run(["git", "archive", "24019a9a", SCRIPT], cwd=ROOT, timeout=15, capture_output=True, check=True).stdout
    with tar_open(fileobj=BytesIO(tree)) as archive:
        base = temp / "base.py"
        base.write_bytes(archive.extractfile(SCRIPT).read())
    mods = [load(base), load(ROOT / SCRIPT)]
    heartbeat = temp / ".heartbeat"
    state_file = temp / "churn.json"
    ordinary = {"missing": None, "bad-json": b"{", "windows-newline": b"{\r\n}\r\n", "bad-ts": {"timestamp": "x"}}
    for state, age in product(("running", "waiting_for_games", "idle_end_of_day", "stalled", "unknown"),
                              (-1, 0, 299, 300, 301, 600, 601, 5400, 5401)):
        ordinary[f"{state}/{age}"] = {"state": state, "timestamp": (NOW - timedelta(seconds=age)).isoformat()}
    for age in (None, -1, 0, 599, 600, 601):
        ordinary[f"sleeping/{age}"] = {"state": "sleeping", "timestamp": NOW.isoformat(),
                                     "sleeping_until": None if age is None else (NOW - timedelta(seconds=age)).isoformat()}
    for case, payload in ordinary.items():
        heartbeat.unlink(missing_ok=True)
        if payload is not None:
            heartbeat.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        pair = []
        for mod in mods:
            state_file.unlink(missing_ok=True)
            pair.append((get_result(mod, heartbeat, state_file), state_file.read_bytes() if state_file.exists() else None))
        assert pair[0] == pair[1], (case, pair)
    print(f"{len(ordinary)} ordinary CLI outputs, ping argv and churn files identical to base")

    for case, payload in {
        "deploy marker": b'{"state":"deploy_failed", "step":"stamp", "target_sha":"abcdef0", "timestamp":"invalid"}',
        "spoofed step": b'{"state":"deploy_failed", "step":"x\\nCANARY PASSED", "target_sha":"ABCDE00"}',
        "array": b"[]",
        "oversized": b" " * 65537,
        "encoding": b"\xff",
        "naive": b'{"state":"running","timestamp":"2027-07-15T12:00:00"}',
        "invalid sleep type": b'{"state":"sleeping","timestamp":"2027-07-15T12:00:00Z","sleeping_until":3}',
        "fifo": None,
        "directory": None,
    }.items():
        heartbeat.unlink(missing_ok=True)
        if case == "fifo":
            os.mkfifo(heartbeat)
        elif case == "directory":
            heartbeat.mkdir()
        else:
            heartbeat.write_bytes(payload)
        output = get_result(mods[1], heartbeat, state_file)
        print(case, output)
        assert output[0] == 1 and output[3] == [(URL + "/fail", "GET", 10)]
        assert "CANARY PASSED" not in output[1] and not output[4]
