import io
import json
from contextlib import ExitStack, redirect_stderr
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest.mock import patch

from offline import install

install()

from bts import scheduler
from bts.health import alert


DAY = "2027-09-27"
HAS_WARNING = hasattr(scheduler, "_season_boundary_check")


class FirstCheck(BaseException):
    pass


def game(day=DAY, start="19:05:00", number=301, **changes):
    return {"gamePk": number, "gameType": "R", "officialDate": day,
            "gameDate": f"{day}T{start}-04:00",
            "teams": {"away": {"team": {"name": "Away"}},
                      "home": {"team": {"name": "Home"}}},
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"}} | changes


class Startup:
    def __init__(self, root, *, mode="private", required=False, recipient="offline.test"):
        self.root = root
        self.bodies = []
        self.config = {"orchestrator": {"picks_dir": str(root / "picks"),
                                        "heartbeat_path": str(root / "heartbeat")},
                       "scheduler": {"pick_delivery": mode, "contest_state_required": required},
                       "health_checks": {"enabled": False},
                       "bluesky": {"dm_recipient": recipient}}

    def receipt(self, date=DAY):
        return self.root / "health_state" / "season_boundary" / f"{date}.json"

    def run(self, games, *, at="09:00:00", day=DAY, dry=False, transport=None, journal=None,
            extra=()):
        self.now = datetime.fromisoformat(at if "T" in at else f"{day}T{at}").replace(tzinfo=scheduler.ET)
        self.stream = journal if journal is not None else io.StringIO()
        checks = []

        def advance(seconds):
            self.now += timedelta(seconds=seconds)

        def fetch(*args, **kwargs):
            if isinstance(games, Exception):
                raise games
            return io.BytesIO(json.dumps({"dates": [{"date": day, "games": games}]}).encode())

        def send(recipient, body):
            self.bodies.append(body)
            return transport(recipient, body) if transport else "dm-message-id"

        def check(**kwargs):
            checks.append((self.now.isoformat(), sorted(kwargs["unavailable_game_pks"])))
            raise FirstCheck

        with ExitStack() as stack:
            stack.enter_context(redirect_stderr(self.stream))
            stack.enter_context(patch.object(scheduler, "_now_et", lambda: self.now))
            stack.enter_context(patch.object(scheduler, "_watchdog_ping_sleep", advance))
            stack.enter_context(patch.object(scheduler, "retry_urlopen", fetch))
            stack.enter_context(patch.object(scheduler, "run_single_check", check))
            stack.enter_context(patch.object(alert, "send_dm", send))
            for name in ("notify_ready", "notify_watchdog", "_idle_until_next_wakeup"):
                stack.enter_context(patch.object(scheduler, name, lambda *args: None))
            for context in extra:
                stack.enter_context(context)
            try:
                scheduler.run_day(day, self.config, dry_run=dry)
            except FirstCheck:
                pass
        return checks, self.stream.getvalue()


def morning_parity(root):
    rows = []
    for mode in ("private", "dm", "public"):
        for lag in (0, 11):
            for at in ("15:54:55", "15:55:01"):
                start = Startup(root / f"{mode}-{at}-{lag}", mode=mode, required=True)

                def transport(_, body):
                    if "Morning check for" in body:
                        start.now += timedelta(seconds=lag)
                    return "confirmed"

                checks, journal = start.run([game()], at=at, transport=transport)
                mornings = [body for body in start.bodies if "Morning check for" in body]
                warnings = [body for body in start.bodies if "[season_boundary]" in body]
                assert len(mornings) == (mode != "private" and at == "15:54:55")
                expect_warning = at == "15:54:55" and not (mornings and lag)
                assert len(warnings) == (HAS_WARNING and expect_warning)
                if mornings:
                    assert start.bodies[0] in mornings
                if HAS_WARNING and not expect_warning:
                    assert "first lineup check is due within 10 minutes" in journal
                assert checks == [(f"{DAY}T16:05:00-04:00", [])]
                rows.append((mode, at, lag, mornings, checks))
    print("morning+lineup", json.dumps(rows))


def transport_deadline(root):
    games = [game(start="13:15:00")]
    for initial, expected in (("10:05:00", 1), ("10:05:01", 0), ("10:14:59", 0),
                              ("10:15:01", 1)):
        start = Startup(root / initial)
        checks, journal = start.run(games, at=initial)
        assert len(start.bodies) == expected * HAS_WARNING
        wanted = "11:15:00" if initial == "10:15:01" else "10:15:00"
        assert checks == [(f"{DAY}T{wanted}-04:00", [])]

    if not HAS_WARNING:
        return
    sending, resume = Event(), Event()
    workers = []
    budgets = []
    start = Startup(root / "blocked-transport")

    class SimulatedTimedJoin(Thread):
        def join(self, timeout=None):
            assert timeout is not None and 59 < timeout <= 60
            assert self.daemon
            assert sending.wait(1)
            workers.append(self)
            budgets.append(timeout)
            start.now += timedelta(seconds=timeout)

    def stalled_send(_, __):
        sending.set()
        assert resume.wait(2)
        return "late-id"

    try:
        checks, journal = start.run(games, at="10:05:00", transport=stalled_send,
                                    extra=[patch.object(scheduler, "Thread", SimulatedTimedJoin)])
        assert checks == [(f"{DAY}T10:15:00-04:00", [])]
        assert workers[0].is_alive() and not start.receipt().exists()
        assert "still sending at its deadline; continuing" in journal
        resume.set()
        Thread.join(workers[0], 1)
        assert json.loads(start.receipt().read_text())["message_id"] == "late-id"
        start.run(games, at="10:05:00")
        assert len(start.bodies) == 1
    finally:
        resume.set()
        for worker in workers:
            Thread.join(worker, 1)
    print("stall: real worker beside first check, later acknowledgement, next start suppressed", budgets)


def blocking_journal(root):
    class BackpressuredJournal(io.StringIO):
        def __init__(self, byte_budget, advance_clock):
            super().__init__()
            self.byte_budget = byte_budget
            self.advance_clock = advance_clock
            self.total = 0
            self.stalled = False

        def write(self, value):
            self.total += len(value.encode())
            if self.total > self.byte_budget and not self.stalled:
                self.stalled = True
                self.advance_clock()
            return super().write(value)

    start = Startup(root / "valid")
    def unstall_after_check_time():
        start.now += timedelta(minutes=12)

    # The available journal capacity admits baseline startup and first-check
    # output. Extra output at startup consumes it while the consumer is stalled.
    journal = BackpressuredJournal(350, unstall_after_check_time)
    checks, output = start.run([game(start="13:15:00")], at="10:05:00", journal=journal,
                               extra=[patch.object(scheduler.time, "monotonic", lambda: start.now.timestamp())])
    assert journal.stalled is HAS_WARNING, (journal.total, output)
    assert checks == [(f"{DAY}T{'10:17' if HAS_WARNING else '10:15'}:00-04:00", [])], output
    assert not start.bodies
    print("journal backpressure", json.dumps(dict(stalled=journal.stalled, checks=checks,
                                                   output=output)))


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        morning_parity(root)
        transport_deadline(root)
        blocking_journal(root)
