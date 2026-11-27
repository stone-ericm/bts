import json
import os
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_startup import DAY, HAS_WARNING, Startup, game, scheduler
from bts import strategy
from bts.health import alert


def rows(root):
    mixed = [game(), game(number=302, officialDate="2027-09-26"),
             game(number=303, officialDate="2027-09-28"), game(number=304, officialDate=None),
             game(number=305, status=None), game(number=306, status=[]),
             game(number=307, status={}), game(number=308, status={"detailedState": " \t"}),
             game(number=309, status={"detailedState": 1}),
             game(number=310, status={"detailedState": " CANCELLED "}),
             game(number=311, status={"detailedState": "Postponed"}),
             game(number=312, status={"detailedState": "Canceled"}),
             game(number=313, gameType="F"), game(number=314, gameType="S")]
    no_type = game(number=315)
    no_type.pop("gameType")
    mixed.append(no_type)
    start = Startup(root / "mixed")
    _, journal = start.run(mixed)
    assert ("1 eligible regular-season game(s)" in journal) is HAS_WARNING
    assert len(start.bodies) == int(HAS_WARNING)
    if HAS_WARNING:
        payload = json.loads(start.receipt().read_text())
        assert payload["sent_day"] == DAY
        assert payload["sent_sources"] == ["WARN_ATTENTION:season_boundary:behind_schedule"]
        assert payload["message_id"] == "dm-message-id"

    cases = [
        (DAY, [], False), (DAY, mixed[1:], False), (DAY, [game()], True),
        ("2027-09-26", [game("2027-09-26")], False), ("2027-12-31", [], False),
        ("2028-01-01", [], False), ("2028-09-28", [game("2028-09-28")], False),
        ("2027-12-31", [game("2027-12-31")], False),
    ]
    for i, (day, games, dry) in enumerate(cases):
        start = Startup(root / f"row-{i}")
        _, journal = start.run(games, day=day, dry=dry)
        warn = HAS_WARNING and i in (6, 7)
        assert ("SEASON BOUNDARY BEHIND SCHEDULE" in journal) is warn, journal
        assert len(start.bodies) == int(warn)
    start = Startup(root / "after-midnight")
    _, journal = start.run([game()], at="2027-09-28T00:01:00")
    assert not start.bodies
    if HAS_WARNING:
        assert f"{DAY} ET" in journal and "first lineup check is due within 10 minutes" in journal
    start = Startup(root / "malformed-date")
    checks, journal = start.run([game()], day="2027-bad-date", at=f"{DAY}T10:00:00")
    assert len(checks) == 1 and not start.bodies
    assert ("Season boundary check failed" in journal) is HAS_WARNING
    start = Startup(root / "failed-fetch")
    offline_failure = OSError("synthetic schedule read failure")
    try:
        start.run(offline_failure)
    except OSError as error:
        assert error is offline_failure
    else:
        raise AssertionError("bad fetch should raise")
    print("row rules, zero count, invalid date, fetch failure, dry run, midnight and year boundary passed")


def restart_matrix(root):
    if not HAS_WARNING:
        return
    next_day = "2027-09-28"
    start = Startup(root / "repeat")
    for day, count in ((DAY, 1), (DAY, 2), (next_day, 1), (DAY, 1)):
        start.run([game(day, number=301 + n) for n in range(count)], day=day)
    assert len(start.bodies) == 2
    assert all(start.receipt(day).exists() for day in (DAY, next_day))

    for name in ("malformed", "schema-less", "directory", "fifo", "symlink", "hardlink", "parent-symlink"):
        start = Startup(root / name)
        receipt = start.receipt()
        receipt.parent.parent.mkdir(parents=True)
        general = receipt.parent.parent / "health_dm_delivery_status.json"
        general.write_text('{"alert":"keep"}')
        if name == "parent-symlink":
            other_dir = start.root / "unrelated"
            other_dir.mkdir()
            receipt.parent.symlink_to(other_dir, target_is_directory=True)
        else:
            receipt.parent.mkdir()
        if name == "malformed":
            receipt.write_text("{")
        elif name == "schema-less":
            receipt.write_text('{"status":"sent"}')
        elif name == "directory":
            receipt.mkdir()
        elif name == "fifo":
            os.mkfifo(receipt)
        elif name == "symlink":
            receipt.symlink_to(general)
        elif name == "hardlink":
            os.link(general, receipt)
        for _ in range(2):
            start.run([game()])
        assert len(start.bodies) == (1 if name in {"malformed", "schema-less"} else 2), name
        assert general.read_text() == '{"alert":"keep"}'
        if name == "parent-symlink":
            assert list(other_dir.iterdir()) == []
        print("receipt", name, "sends", len(start.bodies))

    for result in (None, " ", OSError("transport unavailable")):
        start = Startup(root / f"failed-{repr(result)}")

        def failing(_, __):
            if isinstance(result, Exception):
                raise result
            return result

        start.run([game()], transport=failing)
        assert json.loads(start.receipt().read_text())["status"] == "failed"
        for _ in range(2):
            start.run([game()])
        assert len(start.bodies) == 2

    start = Startup(root / "operator-interrupt")
    class InterruptedJoin(scheduler.Thread):
        def join(self, timeout=None):
            if self in workers:
                return super().join(timeout)
            workers.append(self)
            assert in_send.wait(1)
            raise KeyboardInterrupt

    from threading import Event
    workers = []
    in_send, quit_send = Event(), Event()

    def interrupted_transport(_, __):
        in_send.set()
        assert quit_send.wait(2)
        raise KeyboardInterrupt

    try:
        with patch("threading.excepthook", lambda args: None):
            try:
                start.run([game()], transport=interrupted_transport,
                          extra=[patch.object(scheduler, "Thread", InterruptedJoin)])
            except KeyboardInterrupt:
                assert not start.receipt().exists()
            else:
                raise AssertionError("the operator interrupt should leave run_day")
            finally:
                quit_send.set()
                for worker in workers:
                    worker.join(1)
        start.run([game()])
        start.run([game()])
        assert len(start.bodies) == 2
    finally:
        quit_send.set()
    print("D1/D2/D1, changing count, failed/blank sends and restart after interrupt passed")


def changed_boundary(root):
    if not HAS_WARNING:
        return
    start = Startup(root / "boundary-changed")
    day = "2028-09-29"
    start.run([game(day)], day=day)
    first = json.loads(start.receipt(day).read_text())
    with patch.dict(strategy.SEASON_END_DATES, {2028: "2028-09-28"}):
        start.run([game(day)], day=day)
        second = json.loads(start.receipt(day).read_text())
    assert len(start.bodies) == 2 and first["body_sha256"] != second["body_sha256"]
    assert "fallback exclusive boundary" in start.bodies[0]
    assert "configured exclusive boundary" in start.bodies[1]
    assert first["sent_day"] == second["sent_day"] == day
    print("duplicate acknowledged WARN after adding fallback date to configured table", day, first, second)


def no_send_controls(root):
    if not HAS_WARNING:
        return
    start = Startup(root / "no-recipient", recipient=None)
    _, journal = start.run([game()])
    assert "no DM recipient is configured" in journal and not start.bodies
    start = Startup(root / "no-plan")
    _, journal = start.run([game()], extra=[patch.object(scheduler, "plan_run_times", return_value=[])])
    assert "no lineup check is planned" in journal and not start.bodies

    start = Startup(root / "write-failure")
    with patch.object(alert, "_replace_receipt", side_effect=OSError("volume temporarily read-only")):
        start.run([game()])
    assert not start.receipt().exists()
    start.run([game()])
    start.run([game()])
    assert len(start.bodies) == 2

    import io
    class FailedJournal(io.StringIO):
        def write(self, value):
            if "SEASON BOUNDARY BEHIND SCHEDULE" in value:
                raise OSError("journal temporarily unavailable")
            return super().write(value)

    start = Startup(root / "journal-failure")
    _, journal = start.run([game()], journal=FailedJournal())
    assert "Season boundary check failed" in journal and len(start.bodies) == 1
    print("no plan/recipient, receipt write failure and journal write exception passed")


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        root = Path(directory)
        rows(root)
        restart_matrix(root)
        changed_boundary(root)
        no_send_controls(root)
