import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch


ROOT = Path.cwd()
LEGACY = ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-k2-codex"


def child(root, mode):
    sys.path.insert(0, str(LEGACY))
    from probe_startup import Startup, game, scheduler
    from bts.health import alert

    observation_written = Event()
    real_boundary_write = scheduler._write_season_boundary_line
    real_report = alert.report_unknown_outcome_alert_failure

    def write(line):
        real_boundary_write(line)
        observation_written.set()

    def fill():
        os.set_blocking(2, False)
        try:
            for size in (1024, 1):
                for attempt in range(1_000_000):
                    try:
                        os.write(2, b"x" * size)
                    except BlockingIOError:
                        break
                else:
                    raise AssertionError("pipe never filled")
        finally:
            os.set_blocking(2, True)

    def transport(_, __):
        assert observation_written.wait(1)
        fill()
        if mode == "failed-send":
            raise RuntimeError("offline transport failed")
        print("send is waiting", flush=True)
        Event().wait(5)
        return "late-id"

    def failure(detail):
        print("failure is writing", flush=True)
        real_report(detail)

    startup = Startup(root)
    with (patch.object(scheduler, "_write_season_boundary_line", write),
          patch.object(alert, "report_unknown_outcome_alert_failure", failure)):
        try:
            startup.run([game(start="13:15:00")], at="10:05:00",
                        transport=transport, journal=sys.stderr)
        except KeyboardInterrupt:
            print("KeyboardInterrupt escaped run_day", flush=True)


def run_case(root, mode, ready):
    args = [sys.executable, str(Path(__file__).resolve()), str(root), mode]
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            assert select.select([process.stdout], [], [], 3)[0]
            os.set_blocking(process.stdout.fileno(), False)
            notice = os.read(process.stdout.fileno(), 4096).decode()
            assert notice.strip() == ready, notice
            # Give the reporter time to enter the actual buffered write before SIGINT.
            time.sleep(0.10)
            process.send_signal(signal.SIGINT)
            before_drain = True
            try:
                process.wait(timeout=1.6)
            except subprocess.TimeoutExpired:
                before_drain = False
            os.set_blocking(process.stdout.fileno(), True)
            output, errors = process.communicate(timeout=2)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=2)

    assert "KeyboardInterrupt escaped run_day" in output.decode(), (output, errors[-600:])
    diagnostic = errors.replace(b"x" * 32, b"").decode(errors="replace")[-800:]
    print(mode, "exited before journal drain", before_drain, "exit code", process.returncode)
    print(diagnostic)
    return before_drain, process.returncode, diagnostic


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(Path(sys.argv[1]), sys.argv[2])
    else:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            healthy = run_case(root / "healthy", "blocked-send", "send is waiting")
            failed = run_case(root / "failed", "failed-send", "failure is writing")
            assert healthy[:2] == (True, 0), healthy
            assert failed[:2] != (True, 0), failed
