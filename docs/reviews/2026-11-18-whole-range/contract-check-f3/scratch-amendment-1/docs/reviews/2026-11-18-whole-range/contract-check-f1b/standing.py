"""Check standing rows, retention across days, and ordinary loss scenarios."""
from datetime import date, datetime, timedelta, timezone
import fcntl
import os
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World, row


def overlap(root):
    w = World(root)
    w.commit(pitch=PITCH)
    assert w.check("18:25")["marker"]["escalations"] == ["initial"]
    w.capture([WRONG])

    def newer(*_):
        from click.testing import CliRunner
        original = w.runner
        try:
            w.runner = CliRunner()
            result = w.check("18:45", MATCH, dm="failure")
        finally:
            w.runner = original
        assert result["marker"]["game_unresolved"] is True
        return []

    older = w.check("18:30", more_patches=[("bts.contest_fetch.fetch_pending_predictions", newer)])
    assert "game_unresolved" not in older["marker"]
    w.capture([RIGHT], f"{DAY}T230000Z")
    final = w.check("19:00", MATCH)
    assert final["marker"]["status"] == "confirmed" and final["dm"][0].startswith("✅")
    print("K4 archived sequence, outside cron lock:", older["marker"], final["marker"], final["dm"])


def morning(root):
    w = World(root)
    w.commit(pitch=PITCH)
    assert w.morning()["outcome"]["state"] == "entry_unverifiable"
    assert w.health() == []
    result = w.check("18:30", MATCH)
    assert result["receipts"][-1]["qualification"]["slots"][0]["state"] == "unit_unverified"
    clean = w.morning()
    assert clean["outcome"]["state"] == "entered_game_unverified" and not clean["dm"]
    print("Absent watchdog: entry_unverifiable with no bound evidence; missing subsequent slot leaves", clean)

    u = World(root / "changed")
    u.commit(pitch=PITCH)
    u.capture([WRONG])
    first = u.check("18:30", MATCH)
    assert first["marker"]["game_unresolved"] and u.morning()["outcome"]["state"] == "entry_unverifiable"
    u.capture([RIGHT], f"{DAY}T230000Z")
    second = u.check("18:45", MATCH)
    assert second["marker"]["game_unresolved"] and u.morning()["outcome"]["state"] == "entered_game_unverified"
    print("F1 changes morning receipts: wrong-game-only would alert; later unresolved unbound match is clean")


def last_slot(root):
    full = World(root / "no-hard-limit")
    capped = World(root / "killed")
    for w in (full, capped):
        w.commit(pitch=f"{DAY}T19:05:00-04:00")
        assert w.check("18:30", fetch_failure="pending")["marker"] is None
    result = full.check("18:45", elapsed=850)
    assert result["dm"] and "NOT entered" in result["dm"][0]
    after = capped.check("19:00")
    assert "outside window" in after["stdout"] and not after["dm"]
    assert capped.morning()["outcome"]["state"] == "entry_unverifiable"
    print("Last actionable slot: 850-second response with no hard limit sends initial DM; 840-second cutoff and 19:00 retry lose it")


def calendar(root):
    w = World(root, "2027-12-31")
    w.commit(pitch="2028-01-01T01:00:00-05:00")
    w.capture([WRONG], "20271231T235900Z")
    start = w.check("23:45", MATCH, elapsed=870)
    assert start["marker"]["date"] == w.day and start["marker"]["game_unresolved"]
    assert start["receipts"][-1]["season"] == 2027
    tomorrow = World(root, "2028-01-01")
    tomorrow.commit(pitch="2028-01-01T11:00:00-05:00")
    result = tomorrow.check("10:00", [row(rnd=2)], rounds={2: date.fromisoformat(tomorrow.day)})
    assert result["marker"]["date"] == tomorrow.day and result["marker"]["status"] == "confirmed"
    et = ZoneInfo("America/New_York")
    for md in ((3, 14), (11, 7)):
        utc = datetime(2027, *md, tzinfo=timezone.utc)
        local = [(utc + timedelta(minutes=15 * i)).astimezone(et) for i in range(120)]
        day = date(2027, *md)
        slots = [t for t in local if t.date() == day and 10 <= t.hour <= 23]
        assert len(slots) == 56 and all(t.fold == 0 for t in slots)
        assert all((b.astimezone(timezone.utc) - a.astimezone(timezone.utc)).seconds == 900
                   for a, b in zip(slots, slots[1:]))
        print(day, "10:00–23:45 slots", len(slots), "local 01:00 occurrences",
              [(t.isoformat(), t.fold) for t in local if t.date() == day and t.hour == 1 and t.minute == 0])
    print("ET year boundary", start["marker"]["date"], "then", result["marker"]["date"])


def lock_paths(root):
    regular = root / "read-only"
    regular.touch()
    regular.chmod(0o444)
    for path, flags in ((root, os.O_RDONLY), (regular, os.O_RDONLY | os.O_CREAT)):
        fd = os.open(path, flags)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        print("local advisory lock on", "directory" if path.is_dir() else "read-only regular file", "succeeded")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        root = Path(tmp)
        for check in (overlap, morning, last_slot, calendar, lock_paths):
            check(root / check.__name__) if check is not lock_paths else check(root)
