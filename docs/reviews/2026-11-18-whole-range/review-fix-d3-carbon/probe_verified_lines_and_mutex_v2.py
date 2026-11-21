#!/usr/bin/env python3
"""Probes for Review Items 2 and 3 (v2 with 60s subprocess timeout for loaded host)."""
from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".codex-review/wr-d3"))

import probe_verified_lines_and_mutex as orig
import subprocess

orig_run_script = orig.run_script


def run_script_60(box: Path, script: str, *, scenario: str = "active", fail_sync: bool = False,
                  run_id: str = "123", attempt: str = "1", target: str = orig.TARGET,
                  post_code: str = "200", post_rc: str = "0"):
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
        "/usr/bin/env", "-i", f"PATH={box / 'shims'}", f"BOX={box}",
        f"SCENARIO={scenario}", f"FAIL_SYNC={'1' if fail_sync else '0'}",
        f"POST_CODE={post_code}", f"POST_RC={post_rc}",
        f"TARGET_SHA={target}", f"RUN_ID={run_id}", f"RUN_ATTEMPT={attempt}",
        "LC_ALL=C", "/bin/bash", "-c", script,
    ]
    proc = subprocess.Popen(argv, cwd=box, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with proc:
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=2)
            raise
    import json
    calls = [json.loads(l) for l in (box / "calls").read_text().splitlines()] if (box / "calls").exists() else []
    return subprocess.CompletedProcess(argv[:2], proc.returncode, out, err), calls


orig.run_script = run_script_60

if __name__ == "__main__":
    orig.main()
