"""Compare the unchanged base and candidate heartbeat CLI under an ET fixture."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NOW = datetime(2026, 11, 21, 15, 30, tzinfo=timezone.utc)
URL = "https://heartbeat.invalid/monitor"


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.datetime = FixedDateTime
    return module


def invoke(mod, heartbeat, history, counters=8):
    requests = []
    churn = []
    std_out, std_err = StringIO(), StringIO()
    orig_open, orig_args = mod.urllib.request.urlopen, sys.argv
    mod.read_nrestarts = lambda unit: churn.append(unit) or counters

    class HTTP:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return b"OK"

    def request_stub(request, *, timeout):
        requests.append([request.method, request.full_url, timeout])
        return HTTP()

    mod.urllib.request.urlopen = request_stub
    sys.argv = ["check_heartbeat.py", "--heartbeat-path", str(heartbeat), "--ping-url", URL,
                "--churn-state", str(history)]
    try:
        with redirect_stdout(std_out), redirect_stderr(std_err):
            try:
                mod.main()
            except SystemExit as exc:
                code = exc.code
    finally:
        mod.urllib.request.urlopen = orig_open
        sys.argv = orig_args
    return dict(code=code, stdout=std_out.getvalue(), stderr=std_err.getvalue(),
                requests=requests, churn=churn, history=history.read_bytes().hex() if history.is_file() else None)


def running(age):
    return {"state": "running", "timestamp": (NOW - timedelta(seconds=age)).isoformat()}


with TemporaryDirectory(prefix="wrd3-hb-", dir="/tmp") as tmp:
    directory = Path(tmp)
    heartbeat = directory / ".heartbeat"
    base = load("base_heartbeat", HERE / "base/scripts/check_heartbeat.py")
    candidate = load("candidate_heartbeat", REPO / "scripts/check_heartbeat.py")
    ordinary = {"missing": None, "invalid-json": b"{", "crlf": b"{\r\n}\r\n"}
    for state, ages in {
        "running": (-1, 0, 299, 300, 301, 7200),
        "waiting_for_games": (0, 600, 601),
        "idle_end_of_day": (0, 5400, 5401),
        "unknown": (0, 7200),
        "stalled": (0, 7200),
    }.items():
        for age in ages:
            ordinary[f"{state}-{age}"] = running(age) | {"state": state}
    for age in (-1, 0, 600, 601):
        ordinary[f"sleeping-{age}"] = running(7200) | {
            "state": "sleeping", "sleeping_until": (NOW - timedelta(seconds=age)).isoformat(),
        }
    for index, wake in enumerate((None, "", "bad", False)):
        ordinary[f"sleeping-wake-{index}"] = running(7200) | {"state": "sleeping", "sleeping_until": wake}
    ordinary["timestamp-numeric"] = {"state": "running", "timestamp": 17}
    ordinary["timestamp-invalid"] = {"state": "running", "timestamp": "bad"}
    ordinary["timestamp-none"] = {"state": "running"}
    ordinary["crash-loop"] = running(0)
    ordinary["symlink"] = running(0)
    for name, payload in ordinary.items():
        heartbeat.unlink(missing_ok=True)
        if payload is not None:
            if name == "symlink":
                target = directory / "target"
                target.write_text(json.dumps(payload))
                heartbeat.symlink_to(target)
            else:
                heartbeat.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        histories = [directory / "base.history", directory / "candidate.history"]
        for history in histories:
            history.unlink(missing_ok=True)
            if name == "crash-loop":
                history.write_text(json.dumps({"samples": [{"ts": (NOW - timedelta(minutes=5)).isoformat(), "n": 1}]}))
        former = invoke(base, heartbeat, histories[0])
        latter = invoke(candidate, heartbeat, histories[1])
        assert former == latter, (name, former, latter)
        print(name, json.dumps(latter, ensure_ascii=False), flush=True)

    history = directory / "candidate.history"
    history.unlink(missing_ok=True)
    for index, (step, target, timestamp, reason) in enumerate([
        ("restart", "abc0123", "bad", "Deploy failed at restart (requested target abc0123)."),
        ([], "abc0123", None, "Deploy failed at unknown (requested target abc0123)."),
        ("interrupted", {}, NOW.isoformat(), "Deploy failed at interrupted (requested target unknown)."),
        ("wait\nFORGED", "ABC0123", 0, "Deploy failed at unknown (requested target unknown)."),
    ]):
        heartbeat.unlink(missing_ok=True)
        heartbeat.write_text(json.dumps(dict(state="deploy_failed", step=step, target_sha=target, timestamp=timestamp)))
        result = invoke(candidate, heartbeat, history)
        assert result["code"] == 1 and reason + " See INCIDENT.md §4." in result["stdout"]
        assert result["requests"] == [["GET", URL + "/fail", 10]] and not result["churn"]
        print(f"deploy-failed-{index}", json.dumps(result, ensure_ascii=False), flush=True)

    for name, payload in {
        "fifo": None, "socket": None, "directory": None, "array": b"[]", "bad-utf8": b"\xff",
        "large": b" " * 65537, "naive-ts": b'{"timestamp":"2026-11-21T15:30:00"}',
        "naive-wake": json.dumps(running(0) | {"state": "sleeping", "sleeping_until": "2026-11-21T18:00:00"}).encode(),
    }.items():
        heartbeat.unlink(missing_ok=True)
        if name == "fifo":
            os.mkfifo(heartbeat)
        elif name == "directory":
            heartbeat.mkdir()
        elif name != "socket":
            heartbeat.write_bytes(payload)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stub:
            if name == "socket":
                stub.bind(str(heartbeat))
            result = invoke(candidate, heartbeat, history)
        if name == "directory":
            heartbeat.rmdir()
        assert result["code"] == 1 and "heartbeat unreadable or malformed" in result["stdout"], (name, result)
        assert result["requests"] == [["GET", URL + "/fail", 10]] and not result["churn"]
        print(name, json.dumps(result, ensure_ascii=False), flush=True)

print(sys.version, flush=True)
