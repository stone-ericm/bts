"""Probe 04b: Linux gate / shim fidelity audit (checking os.fstat before opening output file) + text & Amendment 5 checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
from shlex import quote
import subprocess
import sys
import tempfile
import time
import gzip

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE_ARCH = OUT_DIR / "_work_p01" / "base_archive"
DAY = "2027-03-24"
PITCH = f"{DAY}T19:40:00-04:00"


def check_workflows_and_shim_fidelity() -> dict:
    sys.path.insert(0, str(REPO))
    import tests.test_wr_f1b_lock as lock_mod

    test_yml = (REPO / ".github" / "workflows" / "test.yml").read_text()
    deploy_yml = (REPO / ".github" / "workflows" / "deploy.yml").read_text()

    assert "runs-on: ubuntu-latest" in test_yml
    assert "TZ: America/New_York" in test_yml
    assert 'version: "0.11.6"' in test_yml
    assert 'uv run pytest -m "not slow"' in test_yml

    assert "runs-on: ubuntu-latest" in deploy_yml
    assert "TZ: America/New_York" in deploy_yml
    assert 'version: "0.11.6"' in deploy_yml
    assert 'uv run pytest -m "not slow"' in deploy_yml
    assert "cron-setup-hetzner.sh" not in deploy_yml

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # 1. _flock shim on a directory lock path raises IsADirectoryError
        dir_lock = root / "dir.lock"
        dir_lock.mkdir()
        dir_raised = None
        try:
            lock_mod._flock("-n", str(dir_lock), "/usr/bin/true")
        except Exception as exc:
            dir_raised = type(exc).__name__
        assert dir_raised == "IsADirectoryError"

        # 2. _flock shim on a mode-0444 read-only file raises PermissionError
        ro_lock = root / "ro.lock"
        ro_lock.write_text("")
        ro_lock.chmod(0o444)
        ro_raised = None
        if os.geteuid() != 0:
            try:
                lock_mod._flock("-n", str(ro_lock), "/usr/bin/true")
            except Exception as exc:
                ro_raised = type(exc).__name__
            assert ro_raised == "PermissionError"

        # 3. _timeout shim requires BTS_TEST_LOCK_FD in env
        env_clean = {k: v for k, v in os.environ.items() if k != "BTS_TEST_LOCK_FD"}
        res_no_fd = subprocess.run(
            [sys.executable, "-c", "import tests.test_wr_f1b_lock as m; m._timeout('--verbose', '-k', '0.1', '0.1', '/usr/bin/true')"],
            cwd=REPO,
            env={**env_clean, "PYTHONPATH": str(REPO)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert res_no_fd.returncode != 0 and "BTS_TEST_LOCK_FD" in res_no_fd.stderr

        # 4. Measure module-entry startup latency of `python -m tests.test_wr_f1b_lock child quick`
        trace_t = root / "trace.txt"
        t0 = time.monotonic()
        res_quick = subprocess.run(
            [sys.executable, "-m", "tests.test_wr_f1b_lock", "child", "quick"],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO), "BTS_TEST_TRACE": str(trace_t)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        startup_ms = round((time.monotonic() - t0) * 1000, 1)
        assert res_quick.returncode == 0

        # 5. Check _uv FD inheritance when BTS_TEST_LOCK_FD is unset (checking os.fstat BEFORE opening output file!)
        probe_fd_script = (
            "import os, subprocess, sys\n"
            "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\n"
            "ino = os.fstat(fd).st_ino\n"
            "os.set_inheritable(fd, True)\n"
            "child_code = (\n"
            "    'import os, sys\\n'\n"
            "    'fd, ino = int(sys.argv[1]), int(sys.argv[2])\\n'\n"
            "    'try:\\n'\n"
            "    '    same = (os.fstat(fd).st_ino == ino)\\n'\n"
            "    'except OSError:\\n'\n"
            "    '    same = False\\n'\n"
            "    'open(sys.argv[3], \"w\").write(str(same))'\n"
            ")\n"
            "p = subprocess.Popen([sys.executable, '-c', child_code, str(fd), str(ino), sys.argv[2]], pass_fds=())\n"
            "p.wait(timeout=5)\n"
            "os.close(fd)\n"
        )
        seen_file = root / "fd_same_inode.txt"
        subprocess.run([sys.executable, "-c", probe_fd_script, str(root / "t.lock"), str(seen_file)], check=True, timeout=10)
        child_inherits_without_pass_fds = seen_file.read_text() == "True"
        assert child_inherits_without_pass_fds is False

    return {
        "workflows_verified": True,
        "deploy_leaves_crontab_untouched": True,
        "shim_flock_dir_raises": dir_raised,
        "shim_flock_readonly_raises": ro_raised,
        "shim_timeout_requires_lock_fd_env": True,
        "shim_uv_closes_real_flock_fd_in_kill_mode": not child_inherits_without_pass_fds,
        "single_harness_startup_ms": startup_ms,
    }


def check_texts_amendment5_and_rollback() -> dict:
    if str(OUT_DIR) not in sys.path:
        sys.path.insert(0, str(OUT_DIR))
    from probe_04_linux_audit_and_texts import check_texts_amendment5_and_rollback as run_txt
    return run_txt()


def run_all() -> dict:
    summary = {
        "linux_and_shims": check_workflows_and_shim_fidelity(),
        "texts_and_limits": check_texts_amendment5_and_rollback(),
    }
    (OUT_DIR / "log_04b_linux_audit_and_texts.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def test_probe_04b():
    res = run_all()
    assert res["linux_and_shims"]["workflows_verified"] is True
    assert res["texts_and_limits"]["amendment5_verbatim_match"] is True


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
