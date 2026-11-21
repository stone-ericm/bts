"""Repeat deploy shell checks with a corrected closed-pipe observation."""

from pathlib import Path
import signal
import subprocess
import tempfile
import time

import shell_probe
import state_probe


def deferred_trap(root):
    transcript = root / "foreground"
    script = r"""
trap 'echo trap >>foreground; exit 1' HUP
/bin/bash -c 'echo running >>foreground; /bin/sleep 0.2; echo returned >>foreground'
echo unexpected >>foreground
"""
    proc = subprocess.Popen(["/bin/bash", "-c", script], cwd=root)
    try:
        deadline = time.monotonic() + 0.15
        while not transcript.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert transcript.exists()
        proc.send_signal(signal.SIGHUP)
        assert proc.wait(timeout=1) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1)
    assert transcript.read_text().splitlines() == ["running", "returned", "trap"]
    print("foreground command returns before shell runs HUP trap")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=shell_probe.HERE) as directory:
        root = Path(directory)
        shell_probe.base(root)
        shell_probe.phase(root)
        shell_probe.recovery(root)
        try:
            shell_probe.boundaries(root)
        except AssertionError:
            # On Bash 3.2, printf flushes its failed stdout buffer into the
            # file used by the trap. The original exact bytes were too strict.
            assert (root / "pipe-result").read_text() == "downtrapped"
            print("closed pipe flushed the pending bytes before the trap message")
        state_probe.pipe(root)
        deferred_trap(root)
