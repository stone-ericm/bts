import os
import sys
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest.mock import patch


sys.path.insert(0, str(Path.cwd() / "docs/reviews/2026-11-18-whole-range/review-fix-k2-codex"))
from probe_startup import DAY, Startup, game, scheduler


def timed_out_starts(root):
    workers = []
    lines = []
    send_entered = Queue()
    release = Event()

    class TimedJoin(Thread):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.role = kwargs["target"].__name__
            assert self.daemon
            workers.append(self)

        def join(self, timeout=None):
            assert self.role == "dispatch" and 59 < timeout <= 60
            send_entered.get(timeout=1)

    def transport(_, __):
        send_entered.put_nowait(True)
        assert release.wait(3)
        return "late-ack"

    with patch.object(scheduler, "_write_season_boundary_line", lines.append):
        try:
            for number in range(2):
                startup = Startup(root / str(number))
                checks, _ = startup.run([game(start="13:15:00")], at="10:05:00", transport=transport,
                                        extra=[patch.object(scheduler, "Thread", TimedJoin)])
                assert checks == [(f"{DAY}T10:15:00-04:00", [])]
            alive = sorted(worker.role for worker in workers if worker.is_alive())
            assert alive == ["dispatch", "dispatch", "write_journal", "write_journal"], alive
            print("two deadline expirations in one process retain", alive)
        finally:
            release.set()
            for worker in workers:
                Thread.join(worker, 2)
                assert not worker.is_alive()
    assert sum("still sending at its deadline" in line for line in lines) == 2, lines
    assert sum(line.startswith("SEASON BOUNDARY") for line in lines) == 2, lines
    print("all four workers stop when their dispatches return")


def nonblocking_write_loss(root):
    worker = []
    attempts = []
    reader, writer = os.pipe()
    real_write = os.write
    os.set_blocking(reader, False)
    os.set_blocking(writer, False)

    class RecordThread(Thread):
        def start(self):
            worker.append(self)
            super().start()

    def write(fd, data):
        if fd != 2:
            return real_write(fd, data)
        try:
            return real_write(writer, data)
        except BlockingIOError as error:
            attempts.append((data, error.errno))
            raise

    try:
        for size in (1024, 1):
            for attempt in range(1_000_000):
                try:
                    real_write(writer, b"x" * size)
                except BlockingIOError:
                    break
            else:
                raise AssertionError("pipe never filled")
        startup = Startup(root, recipient=None)
        with patch.object(os, "write", write):
            checks, _ = startup.run([game()], at="18:59:59", extra=[
                patch.object(scheduler, "Thread", RecordThread),
            ])
            assert checks == [(f"{DAY}T18:59:59-04:00", [])]
            assert len(worker) == 1
            worker[0].join(1)
            assert not worker[0].is_alive()
        assert len(attempts) == 1 and attempts[0][0].startswith(b"SEASON BOUNDARY")
        drained = bytearray()
        while True:
            try:
                drained.extend(os.read(reader, 65536))
            except BlockingIOError:
                break
        assert drained and bytes(drained).strip(b"x") == b""
        print("full nonblocking journal", "errno", attempts[0][1], "only observation dropped before drain", checks)
    finally:
        os.close(reader)
        os.close(writer)


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        timed_out_starts(root / "timed-out")
        nonblocking_write_loss(root / "nonblocking")
