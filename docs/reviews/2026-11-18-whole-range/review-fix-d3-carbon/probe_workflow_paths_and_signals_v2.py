#!/usr/bin/env python3
"""Probes for Review Items 1 and 4 (v2 with 60s subprocess timeout for loaded host)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".codex-review/wr-d3"))

import probe_workflow_paths_and_signals as orig


def run_script_60(box: Path, script: str, *, scenario: str = "active", fault: str = "",
                  run_id: str = "101", attempt: str = "1", target: str = orig.TARGET,
                  closed_output: bool = False):
    for absolute, rel in {
        "/home/bts/.local/bin/uv": "shims/uv",
        "/home/bts/projects/bts/.venv/bin/bts": "shims/bts",
        "/home/bts": "home",
        "/run/bts-deploy": "run/bts-deploy",
        "/run/user": "run/user",
        "/sys/fs/cgroup": "sys/fs/cgroup",
        "/etc/systemd/system": "etc/systemd/system",
        "/usr/bin/python3": "shims/python3",
        "/tmp/uv-cache": "uv-cache",
        "/dev/null": "null",
    }.items():
        script = script.replace(absolute, str(box / rel))
    argv = [
        "/usr/bin/env", "-i", f"PATH={box / 'shims'}", f"BOX={box}", f"FAULT={fault}",
        f"SCENARIO={scenario}", f"TARGET_SHA={target}", f"RUN_ID={run_id}",
        f"RUN_ATTEMPT={attempt}", "LC_ALL=C", "/bin/bash", "-c", script,
    ]
    launcher = (
        "import os, signal, sys\n"
        "for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGPIPE):\n"
        "    signal.signal(sig, signal.SIG_DFL)\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n"
    )
    writer = subprocess.PIPE
    if closed_output:
        r_fd, w_fd = os.pipe()
        os.close(r_fd)
        writer = w_fd
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", launcher, *argv],
            cwd=box, stdout=writer, stderr=subprocess.PIPE, text=True,
        )
    finally:
        if closed_output:
            os.close(writer)
    with proc:
        (box / "owner.pid").write_text(str(proc.pid))
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=2)
            raise
    calls = [json.loads(l) for l in (box / "calls").read_text().splitlines()] if (box / "calls").exists() else []
    return subprocess.CompletedProcess(argv[:2], proc.returncode, out or "", err), calls


orig.run_script = run_script_60

if __name__ == "__main__":
    orig.main()
