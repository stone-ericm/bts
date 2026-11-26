"""Exercise displaced marker publication, restart and the health reader against both sources."""
from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory

from bts.health import pick_entry, runner
from support import Replay, source
from tests.test_wr_f1_wrong_game import row

PREVIOUS = "2027-12-31"
CURRENT = "2028-01-01"
NEXT = "2028-01-02"
NOW = datetime(2028, 1, 3, tzinfo=timezone.utc)
RAW = b'{ "status":"dm_failed", "date":"2027-12-31", "reason":"no_pick" }\r\n'


class ReadBlocked(RuntimeError):
    pass


@contextmanager
def deadline():
    def expire(*_):
        raise ReadBlocked("read exceeded 0.2 s")

    old = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def dated(world, day=PREVIOUS):
    return world.marker.with_name(f"pick_entry_check.{day}.json")


def health(world, day=PREVIOUS):
    with deadline():
        return [(alert.level, alert.message) for alert in pick_entry.check(world.picks, date.fromisoformat(day), NOW)]


def checker_failure(root, version):
    world = Replay(root, version, CURRENT)
    world.set_marker({})
    world.marker.write_bytes(b'{"reason":"\xff"}')
    alerts = runner._safe_run("pick_entry", lambda: pick_entry.check(world.picks, date.fromisoformat(CURRENT), NOW))
    assert [alert.level for alert in alerts] == (["CRITICAL"] if version == "base" else [])
    print("invalid UTF-8, health runner:", [(a.level, a.message) for a in alerts])


def rollover(root):
    world = Replay(root, day=CURRENT)
    world.commit_at("06:10", at="04:55")
    world.set_marker({})
    world.marker.write_bytes(RAW)
    assert health(world)[0][0] == "CRITICAL"
    first = world.invoke("05:00", auth="profile")
    assert not dated(world).exists() and world.marker.read_bytes() == RAW and first["calls"]

    def interrupt(path, value):
        assert path == world.marker and value["date"] == CURRENT
        assert dated(world).read_bytes() == RAW and path.read_bytes() == RAW
        raise KeyboardInterrupt

    result = world.invoke("05:15", pending=[row()], patches={"bts.cli._atomic_write_json": interrupt})
    assert result["exception"] == "KeyboardInterrupt" and dated(world).read_bytes() == world.marker.read_bytes() == RAW
    inode = dated(world).stat().st_ino
    restart = Replay(root, day=CURRENT)
    result = restart.invoke("05:30", dm="failed")
    assert result["marker"]["status"] == "dm_failed" and result["marker"]["date"] == CURRENT
    assert dated(world).stat().st_ino == inode and dated(world).read_bytes() == RAW
    second_raw = world.marker.read_bytes()
    assert health(world)[0][0] == health(world, CURRENT)[0][0] == "CRITICAL"
    third = Replay(root, day=NEXT)
    third.commit_at("06:10", at="04:55")
    result = third.invoke("05:00", pending=[row()])
    assert result["marker"]["date"] == NEXT and result["marker"]["status"] == "confirmed"
    assert dated(world, CURRENT).read_bytes() == second_raw and dated(world).read_bytes() == RAW
    assert list(map(lambda day: health(world, day)[0][0], (PREVIOUS, CURRENT))) == ["CRITICAL", "CRITICAL"]
    print("rollover: exact bytes; auth failure unchanged; interrupt before marker; existing copy inode kept; two CRITICALs")


def path_kind(path, kind, day):
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "malformed":
        path.write_text('{"date":')
    elif kind == "list":
        path.write_text("[]")
    elif kind == "valid":
        path.write_text(json.dumps({"date": day, "status": "dm_failed"}))


def health_inputs(root):
    for live in ("absent", "malformed", "list", "fifo", "directory", "same_confirmed", "other"):
        world = Replay(root / live, day=CURRENT)
        world.marker.parent.mkdir()
        path_kind(dated(world), "valid", PREVIOUS)
        if live == "same_confirmed":
            world.set_marker({"date": PREVIOUS, "status": "confirmed"})
        else:
            path_kind(world.marker, "valid" if live == "other" else live, CURRENT)
        try:
            levels = [level for level, _ in health(world)]
        except AttributeError:
            levels = ["AttributeError"]
        assert levels == {"list": ["AttributeError"], "other": ["CRITICAL"]}.get(live, []), (live, levels)
        print("health live", live, "+ valid copy:", levels)
    for copy in ("absent", "malformed", "list", "fifo", "directory", "wrong_day", "invalid_encoding"):
        world = Replay(root / ("copy-" + copy), day=CURRENT)
        world.set_marker({"date": CURRENT, "status": "confirmed"})
        path_kind(dated(world), "valid" if copy == "wrong_day" else copy, CURRENT)
        if copy == "invalid_encoding":
            dated(world).write_bytes(b"\xff")
        assert health(world) == [], copy
        print("health copy", copy, "+ other date live: []")


def write_inputs(root):
    for target in ("live", "copy"):
        for kind in ("malformed", "list", "fifo", "directory"):
            world = Replay(root / f"{target}-{kind}", day=CURRENT)
            world.commit_at("06:10", at="04:55")
            world.set_marker(json.loads(RAW))
            if target == "live":
                world.marker.unlink()
            path = world.marker if target == "live" else dated(world)
            path_kind(path, kind, PREVIOUS)
            identity = path.stat().st_ino
            with deadline():
                result = world.invoke("05:00", pending=[row()])
            if target == "copy" or kind in ("malformed", "fifo"):
                assert result["marker"]["status"] == "confirmed", result
                assert (dated(world).exists()) == (target == "copy")
            else:
                assert result["exception"] == ("AttributeError" if kind == "list" else "IsADirectoryError"), result
            if target == "copy":
                assert path.stat().st_ino == identity
            print("write", target, kind, result["exception"], bool(result["calls"]))


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        checker_failure(root / "encoding", version)
        if version != "base":
            rollover(root / "rollover")
            health_inputs(root / "health")
            write_inputs(root / "inputs")
