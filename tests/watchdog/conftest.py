"""Gate 2, write confinement, enforced by the kernel (registration §4.2; W0 review r1 B2).

The watched code runs in a child Python process under macOS `sandbox-exec`, with a profile that denies every
`file-write*` except beneath the admitted root (plus `/dev`, for the stdio devices).
- **Kernel enforcement:** the sandbox mediates the operation's actual target at operation time. That includes
  symlink leaves, descriptor-relative opens, threads and child processes, which inherit the sandbox.
- **Descriptors:** the child starts with no inherited descriptors besides its stdin, stdout and stderr pipes
  (`close_fds`), so a descriptor-only write would first need an open, and the sandbox mediates that open.
- **Outcome:** any attempted outside write fails with EPERM, which the child reports. The gate passes only if the
  child completes cleanly and its own red controls show EPERM.

Where `sandbox-exec` is unavailable (not macOS), the gate tests are skipped with a visible reason, never silently
accepted.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SANDBOX = shutil.which("sandbox-exec")
REPO = Path(__file__).resolve().parents[2]
needs_sandbox = pytest.mark.skipif(SANDBOX is None, reason="gate 2 needs macOS sandbox-exec (kernel write enforcement)")


def profile(allow: Path) -> str:
    real = os.path.realpath(allow)
    return ('(version 1)(allow default)(deny file-write*)'
            f'(allow file-write* (subpath "{real}") (subpath "/dev"))')


def run_confined(code: str, allow: Path, *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run `code` in a fresh `python -B` child that may write only beneath `allow`."""
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "TZ": "America/New_York"}
    return subprocess.run([SANDBOX, "-p", profile(allow), sys.executable, "-B", "-c", code], env=env, cwd=str(allow),
                          capture_output=True, text=True, timeout=timeout, close_fds=True)
