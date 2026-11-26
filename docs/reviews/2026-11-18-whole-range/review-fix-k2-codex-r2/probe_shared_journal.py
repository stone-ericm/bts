import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest.mock import patch


sys.path.insert(0, str(Path.cwd() / "docs/reviews/2026-11-18-whole-range/review-fix-k2-codex"))
from probe_journal_pipe import PipeJournal
from probe_startup import DAY, Startup, game, scheduler


AT = "18:59:59"
CHECK = "Running lineup check..."


def first_check_shares_pipe(root):
    baseline = Startup(root / "baseline")
    with patch.object(scheduler, "_season_boundary_check", return_value=None):
        baseline_checks, trace = baseline.run([game()], at=AT)
    assert baseline_checks == [(f"{DAY}T{AT}-04:00", [])]
    prefix = trace[:trace.rfind("\n[18:59 ET] " + CHECK)]
    assert CHECK not in prefix

    lines, workers = [], []

    class RecordingThread(Thread):
        def start(self):
            workers.append(self)
            super().start()

    with (patch.object(scheduler, "_write_season_boundary_line", lines.append),
          patch.object(scheduler, "Thread", RecordingThread)):
        Startup(root / "measure").run([game()], at=AT)
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
    assert len(lines) == 1
    budget = len((prefix + lines[0] + "\n").encode()) + 1
    assert budget > len(trace.encode())

    for with_warning in (False, True):
        startup = Startup(root / str(with_warning))
        writer_finished = Event()
        real_write = os.write
        workers.clear()

        def advance_before_drain():
            startup.now += timedelta(seconds=2)

        class SharedJournal(PipeJournal):
            def write(self, text):
                if with_warning and CHECK in text:
                    assert writer_finished.wait(1)
                return super().write(text)

        sink = SharedJournal(budget, advance_before_drain)

        def write(fd, data):
            if fd != 2:
                return real_write(fd, data)
            try:
                sink.write(data.decode())
            finally:
                writer_finished.set()
            return len(data)

        extra = [patch.object(os, "write", write), patch.object(scheduler, "Thread", RecordingThread)]
        if not with_warning:
            extra.append(patch.object(scheduler, "_season_boundary_check", return_value=None))
        try:
            checks, _ = startup.run([game()], at=AT, journal=sink, extra=extra)
            expected = [(f"{DAY}T19:00:01-04:00", [301])] if with_warning else baseline_checks
            assert checks == expected, (checks, sink.blocked_write, budget)
            if with_warning:
                assert CHECK in sink.blocked_write, sink.blocked_write
            else:
                assert sink.blocked_write is None
            print("shared pipe", budget, "warning", with_warning, checks, "blocked line", repr(sink.blocked_write))
        finally:
            for worker in workers:
                worker.join(1)
                assert not worker.is_alive()
            sink.finish()


def no_lineup_plan(root):
    startup = Startup(root)
    lines, workers = [], []

    class RecordingThread(Thread):
        def start(self):
            assert self.daemon
            workers.append(self)
            super().start()

    now = datetime.fromisoformat(f"{DAY}T10:05:00-04:00")
    with (patch.object(scheduler, "Thread", RecordingThread),
          patch.object(scheduler, "_now_et", return_value=now),
          patch.object(scheduler, "_write_season_boundary_line", lines.append)):
        scheduler._season_boundary_check(DAY, [game()], [], startup.config, root / "picks")
        assert len(workers) == 1
        workers[0].join(1)
        assert not workers[0].is_alive()
    assert len(lines) == 1 and lines[0].endswith("no lineup check is planned."), lines
    assert not startup.receipt().exists()
    print("empty plan", "one journal-only line")


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        first_check_shares_pipe(root / "shared")
        no_lineup_plan(root / "no-plan")
