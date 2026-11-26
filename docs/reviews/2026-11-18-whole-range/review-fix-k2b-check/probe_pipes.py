import errno
import os
import select
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread, current_thread, main_thread
from unittest.mock import patch


DOCS = Path.cwd() / "docs/reviews/2026-11-18-whole-range"
sys.path.insert(0, str(DOCS / "review-fix-k2-codex"))
from probe_journal_pipe import PipeJournal
from probe_startup import DAY, Startup, game, scheduler


PREFIX = "SEASON BOUNDARY BEHIND SCHEDULE"
CHECK = "Running lineup check..."


def fill(fd):
    for size in (1024, 1):
        for _ in range(131072):
            try:
                os.write(fd, b"x" * size)
            except BlockingIOError as error:
                assert error.errno == errno.EAGAIN
                break
        else:
            raise AssertionError("pipe never filled")


def drain(fd):
    for _ in range(1024):
        try:
            if not os.read(fd, 65536):
                return
        except BlockingIOError:
            return
    raise AssertionError("pipe never drained")


def verify_line(line, day=DAY, count=1):
    amendment = (DOCS / "kickoff-fix-k2.md").read_text().split("## Amendment 5", 1)[1]
    template = amendment.split("The line: `", 1)[1].split("`", 1)[0]
    source, end = ("The configured exclusive boundary", DAY) if day == DAY else (
        "The fallback exclusive boundary for the unconfigured year 2028", "2028-09-28")
    assert line == template.format(D=day, n=count, boundary_source=source, E=end), line


def record(workers):
    class RecordedThread(Thread):
        def start(self):
            assert self.daemon
            workers.append(self)
            super().start()

    return patch.object(scheduler, "Thread", RecordedThread)


def finish(workers):
    for worker in workers:
        worker.join(1)
        assert not worker.is_alive()


def isolated_pipe(root):
    for mode, at, pitch, expected in (
        ("private", "10:05:00", "13:15:00", "10:15:00"),
        ("dm", "10:10:00", "13:15:00", "10:15:00"),
        ("public", "18:59:59", "19:05:00", "18:59:59"),
    ):
        startup = Startup(root / mode, mode=mode)
        attempted, resume = Event(), Event()
        workers, lines = [], []
        reader, writer = os.pipe()
        real_write = os.write
        os.set_blocking(reader, False)
        os.set_blocking(writer, False)
        fill(writer)

        def write(fd, data):
            if fd != 2:
                return real_write(fd, data)
            assert current_thread() is not main_thread()
            try:
                real_write(writer, data)
            except BlockingIOError:
                attempted.set()
            else:
                raise AssertionError("warning met no backpressure")
            assert resume.wait(2)
            lines.append(data.decode().removesuffix("\n"))
            return real_write(writer, data)

        try:
            with patch.object(os, "write", write), record(workers):
                try:
                    checks, _ = startup.run([game(start=pitch)], at=at)
                    assert checks == [(f"{DAY}T{expected}-04:00", [])], checks
                    assert attempted.wait(1) and not lines
                    assert not startup.bodies and not startup.receipt().exists()
                    assert not any("season_boundary" in str(p) for p in startup.root.rglob("*"))
                    assert len(workers) == 1 and workers[0].is_alive()
                finally:
                    drain(reader)
                    resume.set()
                    finish(workers)
            assert len(lines) == 1 and os.read(reader, 65536).decode() == lines[0] + "\n"
            verify_line(lines[0])
            print(mode, checks, "exact line after drain, no send or receipt")
        finally:
            os.close(reader)
            os.close(writer)


def shared_pipe(root, budget):
    at = "18:59:59"
    for warning in (False, True):
        startup = Startup(root / f"{budget}-{warning}")
        wrote = Event()
        workers, lines = [], []
        real_write = os.write

        def advance():
            startup.now += timedelta(seconds=2)

        class SharedJournal(PipeJournal):
            def write(self, value):
                if warning and CHECK in value:
                    assert wrote.wait(1)
                return super().write(value)

        sink = SharedJournal(budget, advance)

        def write(fd, data):
            if fd != 2:
                return real_write(fd, data)
            lines.append(data.decode().removesuffix("\n"))
            try:
                # Bypass the scheduler-only ordering check for the writer.
                PipeJournal.write(sink, data.decode())
                return len(data)
            finally:
                wrote.set()

        extra = [patch.object(os, "write", write), record(workers)]
        if not warning:
            extra.append(patch.object(scheduler, "_season_boundary_check", return_value=None))
        try:
            checks, _ = startup.run([game()], at=at, journal=sink, extra=extra)
            finish(workers)
            if warning:
                assert len(lines) == 1 and lines[0].startswith(PREFIX), lines
            else:
                assert checks == [(f"{DAY}T{at}-04:00", [])] and sink.blocked_write is None
            print("shared pipe", budget, "warning", warning, checks,
                  "warning bytes", len((lines[0] + "\n").encode()) if lines else 0,
                  "blocked", repr(sink.blocked_write))
        finally:
            sink.finish()


def interrupt_child(root):
    filled, forever = Event(), Event()
    startup = Startup(root)
    real_write = scheduler._write_season_boundary_line

    def write(line):
        assert current_thread() is not main_thread()
        assert filled.wait(2)
        print("boundary writer entering full pipe", flush=True)
        real_write(line)

    def wait(_):
        os.set_blocking(2, False)
        try:
            fill(2)
        finally:
            os.set_blocking(2, True)
        filled.set()
        assert forever.wait(4), "SIGINT did not arrive"

    def unexpected_send(*args):
        print("boundary transport called", flush=True)
        raise RuntimeError("offline transport failed")

    try:
        startup.run([game(start="13:15:00")], at="10:05:00", journal=sys.stderr,
                    transport=unexpected_send, extra=[
                        patch.object(scheduler, "_write_season_boundary_line", write),
                        patch.object(scheduler, "_watchdog_ping_sleep", wait),
                    ])
    except KeyboardInterrupt:
        print("KeyboardInterrupt escaped run_day", flush=True)
        assert not startup.bodies and not startup.receipt().exists()


def interrupt_parent(root):
    args = [sys.executable, str(Path(__file__).resolve()), "interrupt-child", str(root)]
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        try:
            assert select.select([child.stdout], [], [], 3)[0]
            os.set_blocking(child.stdout.fileno(), False)
            notice = os.read(child.stdout.fileno(), 4096)
            assert notice == b"boundary writer entering full pipe\n", notice
            time.sleep(.1)
            start = time.monotonic()
            child.send_signal(signal.SIGINT)
            assert child.wait(timeout=1.6) == 0
            elapsed = time.monotonic() - start
            out, err = child.communicate(timeout=2)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=2)
    assert out == b"KeyboardInterrupt escaped run_day\n", out
    assert PREFIX.encode() not in err and b"x" * 128 in err
    print(f"SIGINT exited before journal drain in {elapsed:.3f}s")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "interrupt-child":
        interrupt_child(Path(sys.argv[2]))
    else:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            if mode == "interrupt":
                interrupt_parent(root)
            elif mode == "isolated":
                isolated_pipe(root)
            elif mode == "shared":
                for budget in (510, 650):
                    shared_pipe(root, budget)
            else:
                raise ValueError(mode)
