import io
import os
import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest.mock import patch

from probe_startup import DAY, HAS_WARNING, Startup, game, scheduler


class PipeJournal(io.StringIO):
    def __init__(self, capacity_for_run, on_drain):
        super().__init__()
        self.on_drain = on_drain
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)
        capacity = 0
        while True:
            try:
                capacity += os.write(self.write_fd, b"x" * 1024)
            except BlockingIOError:
                break
        while True:
            try:
                capacity += os.write(self.write_fd, b"x")
            except BlockingIOError:
                break
        assert capacity >= capacity_for_run
        assert len(os.read(self.read_fd, capacity_for_run)) == capacity_for_run
        self.blocked_write = None
        self.full = Event()
        self.done = Event()
        self.start = time.perf_counter()
        self.consumer = Thread(target=self.drain_after_stall, daemon=True)
        self.consumer.start()

    def drain_after_stall(self):
        if self.full.wait(2) and not self.done.wait(.1):
            self.on_drain()
            while True:
                try:
                    os.read(self.read_fd, 65536)
                except BlockingIOError:
                    break

    def write(self, value):
        data = value.encode()
        if not data:
            return 0
        try:
            sent = os.write(self.write_fd, data)
        except BlockingIOError:
            self.blocked_write = value
            self.full.set()
            # Use a bounded readiness wait to model the blocking journal write
            # while keeping the review probe itself bounded.
            from select import select
            assert select([], [self.write_fd], [], 1)[1]
            sent = os.write(self.write_fd, data)
        assert sent == len(data)
        return super().write(value)

    def finish(self):
        self.done.set()
        self.full.set()
        self.consumer.join(1)
        assert not self.consumer.is_alive()
        os.close(self.read_fd)
        os.close(self.write_fd)


def check():
    with TemporaryDirectory() as directory:
        run = Startup(Path(directory))

        def on_drain():
            run.now += timedelta(minutes=12)

        journal = PipeJournal(350, on_drain)
        try:
            checks, output = run.run([game(start="13:15:00")], at="10:05:00", journal=journal,
                                     extra=[patch.object(scheduler.time, "monotonic", lambda: run.now.timestamp())])
            assert bool(journal.blocked_write) is HAS_WARNING, output
            if HAS_WARNING:
                assert "SEASON BOUNDARY BEHIND SCHEDULE" in journal.blocked_write, journal.blocked_write
            assert not run.bodies
            assert checks == [(f"{DAY}T{'10:17' if HAS_WARNING else '10:15'}:00-04:00", [])], output
            print(f"first check {checks}; physical pipe backpressure at {journal.blocked_write!r}; "
                  f"probe elapsed {time.perf_counter() - journal.start:.3f}s")
        finally:
            journal.finish()


if __name__ == "__main__":
    check()
