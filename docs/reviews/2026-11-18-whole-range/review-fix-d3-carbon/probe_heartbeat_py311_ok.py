#!/usr/bin/env python3
"""Item 6 clean verification on both Python 3.11.18 and Python 3.12.16."""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import yaml

REPO = Path(__file__).resolve().parents[2]
PY311 = Path("/Users/eric/.local/share/uv/python/cpython-3.11.18-macos-aarch64-none/bin/python3.11")


def main() -> None:
    assert PY311.is_file()
    base_hb = REPO / ".codex-review/wr-d3/base/scripts/check_heartbeat.py"
    cand_hb = REPO / "scripts/check_heartbeat.py"

    wf = yaml.safe_load((REPO / ".github/workflows/deploy.yml").read_text())
    for step_idx, label in [(3, "deploy"), (4, "recovery")]:
        script = wf["jobs"]["deploy"]["steps"][step_idx]["with"]["script"]
        m = re.search(r"fs_code='\n(.*?)\n'", script, re.DOTALL)
        assert m
        r = subprocess.run([str(PY311), "-c", m.group(1), "clock"], capture_output=True, text=True, timeout=5)
        assert r.returncode == 0 and r.stdout.strip().isdigit()
    print("Inline Python on Python 3.11.18: OK")

    driver = r'''
import io, json, os, sys, tempfile
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

base_src = Path(sys.argv[1]).read_text()
cand_path = Path(sys.argv[2])
cand_ns = {"__name__": "cand", "__file__": str(cand_path)}
base_ns = {"__name__": "base", "__file__": sys.argv[1]}
exec(compile(base_src, sys.argv[1], "exec"), base_ns)
exec(compile(cand_path.read_text(), str(cand_path), "exec"), cand_ns)

now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)
class PinnedDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return now.astimezone(tz) if tz else now.replace(tzinfo=None)

base_ns["datetime"] = PinnedDT
cand_ns["datetime"] = PinnedDT

ordinary_payloads = [
    None,
    b"{invalid json",
    *(json.dumps(d).encode() for ts in ("2026-11-21T11:59:00Z", "2026-11-20T10:00:00Z") for d in [
        {"state": "running", "timestamp": ts},
        {"state": "waiting_for_games", "timestamp": ts},
        {"state": "sleeping", "timestamp": ts, "sleeping_until": "2026-11-21T13:00:00Z"},
        {"state": "sleeping", "timestamp": ts, "sleeping_until": "2026-11-21T11:00:00Z"},
        {"state": "sleeping", "timestamp": ts},
        {"state": "sleeping", "timestamp": ts, "sleeping_until": "bad-wake"},
        {"state": "idle_end_of_day", "timestamp": ts},
        {"state": "stalled", "timestamp": ts, "stage": "predict", "stalled_for_s": 45},
        {"state": "weird", "timestamp": ts},
        {"state": "running"},
        {"state": "running", "timestamp": "bad-ts"},
    ]),
]

def run_ns(ns, hb_path):
    pings = []
    ns["ping"] = lambda *a: pings.append(a)
    sys.argv = ["check_heartbeat.py", "--heartbeat-path", str(hb_path), "--no-churn", "--ping-url", "https://hc.invalid/u"]
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            ns["main"]()
            rc = 0
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue().encode(), err.getvalue().encode(), pings

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    for i, payload in enumerate(ordinary_payloads):
        hb = root / f"hb_{i}.json"
        if payload is not None:
            hb.write_bytes(payload)
        assert run_ns(base_ns, hb) == run_ns(cand_ns, hb)
        if payload is not None:
            link = root / f"link_{i}.json"
            link.symlink_to(hb)
            assert run_ns(base_ns, link) == run_ns(cand_ns, link)

    unreadable = root / "unreadable.json"
    unreadable.write_text('{"state":"running"}')
    unreadable.chmod(0)
    try:
        assert run_ns(base_ns, unreadable) == run_ns(cand_ns, unreadable)
    finally:
        unreadable.chmod(0o600)

    for step, target, exp_s, exp_t in [
        ("sync", "0123abc", "sync", "0123abc"),
        ("cd", "0123abc", "unknown", "0123abc"),
        ("attempt-record", "0123abc", "unknown", "0123abc"),
        ("deploy-step-incomplete", "fedcba9", "deploy-step-incomplete", "fedcba9"),
        ("sync", "0123ab", "sync", "unknown"),
        ("sync", "0123abcd", "sync", "unknown"),
        ("sync", "0123abc\n", "sync", "unknown"),
        ("sync", "0123ABC", "sync", "unknown"),
        (True, 1234567, "unknown", "unknown"),
        (None, None, "unknown", "unknown"),
        (["sync"], {"sha": "0123abc"}, "unknown", "unknown"),
    ]:
        hb = root / "df.json"
        hb.write_text(json.dumps({
            "state": "deploy_failed", "step": step, "target_sha": target,
            "timestamp": "naive-ts", "sleeping_until": 123,
        }))
        rc, out, err, pings = run_ns(cand_ns, hb)
        assert rc == 1 and pings == [("https://hc.invalid/u", "/fail")]
        expected = f"Deploy failed at {exp_s} (requested target {exp_t}). See INCIDENT.md §4."
        assert expected.encode() in out

    if sys.version_info[:2] == (3, 11):
        deep = b'{"a":' * 1500 + b'1' + b'}' * 1500
        hb_deep = root / "deep.json"
        hb_deep.write_bytes(deep)
        assert cand_ns["is_stale"](hb_deep, now=now) == (True, "heartbeat unreadable or malformed")

    for label, bad in [
        ("list", b"[]"),
        ("int", b"123"),
        ("str", b'"str"'),
        ("null", b"null"),
        ("invalid_utf8", b"\xff"),
        ("oversized", b" " * 65537),
        ("naive_ts", b'{"state":"running","timestamp":"2026-11-21T12:00:00"}'),
        ("naive_wake", b'{"state":"sleeping","timestamp":"2026-11-21T12:00:00Z","sleeping_until":"2026-11-21T13:00:00"}'),
        ("int_wake", b'{"state":"sleeping","timestamp":"2026-11-21T12:00:00Z","sleeping_until":123}'),
        ("bool_wake", b'{"state":"sleeping","timestamp":"2026-11-21T12:00:00Z","sleeping_until":true}'),
        ("huge_int", b'{"a":' + b"1" * 5000 + b"}"),
    ]:
        hb = root / "bad.json"
        hb.write_bytes(bad)
        got = cand_ns["is_stale"](hb, now=now)
        assert got == (True, "heartbeat unreadable or malformed"), (label, got)

    d = root / "dir_hb"
    d.mkdir()
    assert cand_ns["is_stale"](d, now=now) == (True, "heartbeat unreadable or malformed")
    fifo = root / "fifo_hb"
    os.mkfifo(fifo)
    assert cand_ns["is_stale"](fifo, now=now) == (True, "heartbeat unreadable or malformed")
    s_fifo = root / "sym_fifo"
    s_fifo.symlink_to(fifo)
    assert cand_ns["is_stale"](s_fifo, now=now) == (True, "heartbeat unreadable or malformed")

print("Python", sys.version.split()[0], "all ordinary + chmod0 + deploy_failed + special/malformed checks OK")
'''
    for interp in (Path(sys.executable), PY311):
        proc = subprocess.run(
            [str(interp), "-c", driver, str(base_hb), str(cand_hb)],
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, (interp, proc.stdout, proc.stderr)
        print(f"{interp.name}: {proc.stdout.strip()}")


if __name__ == "__main__":
    main()
