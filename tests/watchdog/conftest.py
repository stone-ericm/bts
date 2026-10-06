"""Gate 2, write confinement, enforced by the kernel (registration §4.2; W0 review r1 B2).

The watched code runs in a child Python process under macOS `sandbox-exec`, with a profile that denies every
`file-write*` except beneath the admitted root (plus exactly `/dev/null` and `/dev/dtracehelper`). In the gate a
denial **kills** the child (SIGKILL), so a caught refusal cannot leave the gate green.
- **Kernel enforcement:** the sandbox mediates the operation's actual target at operation time. That includes
  symlink leaves, descriptor-relative opens, threads and child processes, which inherit the sandbox.
- **Descriptors:** the child starts with no inherited descriptors besides its stdin, stdout and stderr pipes
  (`close_fds`), so a descriptor-only write would first need an open, and the sandbox mediates that open.
- **Outcome:** in the gate, any attempted outside write kills the child, so the gate fails on its exit status. It also
  asserts the child's exact statuses, notices and confirmed fake deliveries. Red controls use plain EPERM and must
  witness both the child's start and the specific refused operation.

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


# Python's own startup opens /dev/dtracehelper for writing (dyld), and subprocess stdio may use /dev/null. Nothing
# else in /dev is allowed (W0 review r2 G2).
DEV_ALLOW = '(literal "/dev/null") (literal "/dev/dtracehelper")'


def profile(allow: Path, *, kill: bool = True) -> str:
    """kill=True: any denied write kills the child with SIGKILL, so application code cannot catch and swallow a
    refusal; the gate then fails on the exit status (W0 review r2 G1). kill=False: plain EPERM, for red controls
    that must witness the specific refused operation (G3)."""
    real = os.path.realpath(allow)
    deny = "(deny file-write* (with send-signal SIGKILL))" if kill else "(deny file-write*)"
    return f'(version 1)(allow default){deny}(allow file-write* (subpath "{real}") {DEV_ALLOW})'


def run_confined(code: str, allow: Path, *, kill: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run `code` in a fresh `python -B` child that may write only beneath `allow`."""
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "TZ": "America/New_York"}
    return subprocess.run([SANDBOX, "-p", profile(allow, kill=kill), sys.executable, "-B", "-c", code], env=env,
                          cwd=str(allow), capture_output=True, text=True, timeout=timeout, close_fds=True)
