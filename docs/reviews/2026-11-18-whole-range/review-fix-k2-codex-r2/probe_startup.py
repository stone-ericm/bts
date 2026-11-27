import errno
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread, current_thread, main_thread
from unittest.mock import patch


ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-k2-codex"))
from probe_startup import DAY, Startup, game, scheduler
from bts import strategy
from bts.health import alert


OBSERVATION = "SEASON BOUNDARY BEHIND SCHEDULE"
DEADLINE = "Season boundary warning still sending at its deadline; continuing."


class RecordedThread(Thread):
    def __init__(self, workers, **kwargs):
        super().__init__(**kwargs)
        self.role = kwargs["target"].__name__
        assert self.daemon
        workers.append(self)


def join_workers(workers):
    for worker in workers:
        Thread.join(worker, 2)
        assert not worker.is_alive(), worker.role


class BackpressuredWarning:
    def __init__(self):
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self.write_fd, False)
        self.real_write = os.write
        self.attempted = Event()
        self.release = Event()
        self.lines = []
        for size in (1024, 1):
            for attempt in range(1_000_000):
                try:
                    self.real_write(self.write_fd, b"x" * size)
                except BlockingIOError:
                    break
            else:
                raise AssertionError("pipe never filled")

    def write(self, fd, data):
        if fd != 2:
            return self.real_write(fd, data)
        assert current_thread() is not main_thread()
        if not self.release.is_set():
            try:
                self.real_write(self.write_fd, data)
            except BlockingIOError as error:
                assert error.errno == errno.EAGAIN
            else:
                raise AssertionError("expected physical pipe backpressure")
            self.attempted.set()
            assert self.release.wait(2)
        written = self.real_write(self.write_fd, data)
        assert written == len(data)
        self.lines.append(data.decode().rstrip("\n"))
        return written

    def drain(self):
        while True:
            try:
                os.read(self.read_fd, 65536)
            except BlockingIOError:
                return

    def finish(self, workers):
        self.drain()
        self.release.set()
        join_workers(workers)
        os.close(self.read_fd)
        os.close(self.write_fd)


def while_warning_is_blocked(root):
    cases = [
        ("eligible", "10:05:00", "13:15:00", "10:15:00", "recipient", 1, "raw count of zero."),
        ("inside-lead", "10:05:01", "13:15:00", "10:15:00", "recipient", 0, "within 10 minutes."),
        ("cutoff", "18:59:59", "19:05:00", "18:59:59", "recipient", 0, "already passed."),
        ("no-recipient", "10:05:00", "13:15:00", "10:15:00", None, 0, "recipient is configured."),
    ]
    for label, at, pitch, wanted, recipient, sends, ending in cases:
        probe = Startup(root / label, recipient=recipient)
        sink = BackpressuredWarning()
        workers = []
        with patch.object(os, "write", sink.write):
            try:
                checks, _ = probe.run([game(start=pitch)], at=at, extra=[patch.object(
                    scheduler, "Thread", lambda **kwargs: RecordedThread(workers, **kwargs))])
                assert checks == [(f"{DAY}T{wanted}-04:00", [])], (label, checks)
                assert sink.attempted.wait(1) and not sink.lines
                assert len(probe.bodies) == sends
            finally:
                sink.finish(workers)
        assert len(sink.lines) == 1 and sink.lines[0].endswith(ending), sink.lines
        print(label, checks, "WARNs", sends, "writer stopped after drain")


def failed_send_and_retry(root):
    probe = Startup(root)
    sink = BackpressuredWarning()
    workers = []

    def fail(_, __):
        raise RuntimeError("captured failed send")

    at = "10:05:00"
    games = [game(start="13:15:00")]
    factory = [patch.object(scheduler, "Thread", lambda **kwargs: RecordedThread(workers, **kwargs))]
    with patch.object(os, "write", sink.write):
        try:
            checks, report = probe.run(games, at=at, transport=fail, extra=factory)
            assert checks == [(f"{DAY}T10:15:00-04:00", [])]
            assert "Unknown-outcome alert failed: transport: captured failed send" in report
            assert json.loads(probe.receipt().read_text())["status"] == "failed"
            assert sink.attempted.wait(1) and not sink.lines
        finally:
            sink.finish(workers)
    assert sink.lines[0].startswith(OBSERVATION)
    for _ in range(2):
        probe.run(games, at=at, extra=factory)
    join_workers(workers)
    assert len(probe.bodies) == 2
    assert json.loads(probe.receipt().read_text())["status"] == "sent"
    print("failed transport", checks, "failure printed ahead of blocked observation; one retry acknowledged")


def after_deadline(root):
    probe = Startup(root)
    workers = []
    lines = []
    sending, resume, deadline_written = Event(), Event(), Event()
    real_write = os.write

    class SimulatedDeadline(RecordedThread):
        def join(self, timeout=None):
            assert self.role == "dispatch" and 59 < timeout <= 60
            assert sending.wait(1)
            probe.now += timedelta(seconds=timeout)

    def journal(fd, data):
        if fd != 2:
            return real_write(fd, data)
        lines.append(data.decode().strip())
        if lines[-1] == DEADLINE:
            deadline_written.set()
        return len(data)

    def blocked_send(_, __):
        sending.set()
        assert resume.wait(2)
        return "late-id"

    with patch.object(os, "write", journal):
        try:
            checks, _ = probe.run([game(start="13:15:00")], at="10:05:00", transport=blocked_send,
                                  extra=[patch.object(scheduler, "Thread", lambda **kwargs:
                                                      SimulatedDeadline(workers, **kwargs))])
            assert checks == [(f"{DAY}T10:15:00-04:00", [])]
            assert deadline_written.wait(1)
            assert all(worker.is_alive() for worker in workers)
            assert not probe.receipt().exists()
        finally:
            resume.set()
            join_workers(workers)
    assert len(lines) == 2 and lines[0].startswith(OBSERVATION) and lines[1] == DEADLINE, lines
    assert json.loads(probe.receipt().read_text())["message_id"] == "late-id"
    print("deadline", checks, "one observation, one deadline notice, both workers stop after late acknowledgment")


def expired_before_dispatch(root):
    probe = Startup(root)
    workers, lines = [], []
    clock = time.monotonic

    def monotonic():
        thread = current_thread()
        return clock() + (90 if getattr(thread, "role", None) == "dispatch" else 0)

    with patch.object(scheduler, "_write_season_boundary_line", lines.append):
        checks, _ = probe.run([game(start="13:15:00")], at="10:05:00", extra=[
            patch.object(scheduler.time, "monotonic", monotonic),
            patch.object(scheduler, "Thread", lambda **kwargs: RecordedThread(workers, **kwargs)),
        ])
        join_workers(workers)
    assert checks == [(f"{DAY}T10:15:00-04:00", [])]
    assert not probe.bodies and not probe.receipt().exists()
    assert lines[0].startswith(OBSERVATION)
    assert lines[1:] == ["Season boundary warning not sent: insufficient send budget."], lines
    print("expired before dispatch", checks, "no send")


def repeat_after_boundary_update(root):
    day = "2028-09-29"
    probe = Startup(root)
    workers = []
    lines = []
    factory = [patch.object(scheduler, "Thread", lambda **kwargs: RecordedThread(workers, **kwargs))]
    fixture = game(day=day, status={"abstractGameCode": "F", "detailedState": "Final"})
    with patch.object(scheduler, "_write_season_boundary_line", lines.append):
        probe.run([fixture, game(day=day, number=302, status={"detailedState": "Postponed"})],
                  day=day, extra=factory)
        with patch.dict(strategy.SEASON_END_DATES, {2028: "2028-09-28"}):
            probe.run([fixture], day=day, extra=factory)
            probe.run([fixture], day=day, extra=factory)
        join_workers(workers)
    sentence = "At scheduler startup MLB listed at least one regular-season game with that official date and a reported status other than postponed or canceled."
    assert len(probe.bodies) == 2 and all(sentence in body for body in probe.bodies)
    assert "fallback exclusive boundary" in probe.bodies[0] and "configured exclusive boundary" in probe.bodies[1]
    assert len(lines) == 3 and all("1 regular-season game(s)" in line for line in lines), lines
    print("boundary update", "two WARNs across three starts; final counted and postponed omitted")


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        base = Path(directory)
        while_warning_is_blocked(base / "blocked")
        failed_send_and_retry(base / "failed")
        after_deadline(base / "late")
        expired_before_dispatch(base / "expired")
        repeat_after_boundary_update(base / "boundary-update")
