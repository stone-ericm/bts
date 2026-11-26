"""Read the real EOD source against every marker class and replay date crossings."""
from pathlib import Path
import json
import os
import sys
from tempfile import TemporaryDirectory

from replay_support import DAY, RIGHT, ReadBlocked, World, compact, health, row, source
import fallback_recovery


OTHER = "2027-03-23"
D = {"date": DAY, "reason": "no_pick"}
INPUTS = {
    "absent": None, "directory": None, "fifo": None, "bad-json": b"{", "invalid-utf8": b'"\xff"',
    "list": b"[]", "null": b"null", "scalar": b"17", "empty": {}, "oversize": b" " * 65537,
    "partial": D, "foreign": {"date": OTHER, "status": "dm_failed"},
    **{s: {**D, "status": s} for s in ("confirmed", "present_unverified", "dm_failed", "alerted", "other")},
    "bad-tiers": {**D, "status": "dm_failed", "escalations": 42},
}


def create(path, kind):
    value = INPUTS[kind]
    path.parent.mkdir(exist_ok=True)
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    elif isinstance(value, bytes):
        path.write_bytes(value)
    elif value is not None:
        path.write_text(json.dumps(value))


def read(world, **kw):
    try:
        return health(world, **kw)
    except (AttributeError, UnicodeDecodeError, ReadBlocked) as exc:
        return type(exc).__name__


def matrix(root, v):
    for live in INPUTS:
        actual = []
        choices = ["absent", "dm_failed"] if live == "fifo" else INPUTS
        for e in choices:
            w = World(root / (live + "-" + e), v)
            create(w.marker, live)
            create(w.early, e)
            result = read(w)
            exceptions = {"list": "AttributeError", "null": "AttributeError", "scalar": "AttributeError",
                          "invalid-utf8": "UnicodeDecodeError", "fifo": "ReadBlocked"}
            if live in exceptions:
                expected = exceptions[live]
            elif live in ("dm_failed", "bad-tiers"):
                expected = ["CRITICAL"]
            elif live == "alerted":
                expected = ["WARN"]
            elif v == "base" or live in ("confirmed", "present_unverified", "other", "partial"):
                expected = []
            else:
                expected = {"dm_failed": ["CRITICAL"], "alerted": ["WARN"]}.get(e, [])
            assert result == expected, (live, e, result, expected)
            actual.append((e, result))
        print(live, actual)


def sequences(root):
    d = World(root / "rollover", day="2027-12-31")
    d.commit_at("06:10", at="04:55")
    assert d.cron("05:00", dm="failed")["receipt"]["marker_status"] == "dm_failed"
    assert health(d) == ["CRITICAL"] and health(d, clock="05:30") == []
    j = World(d.root, day="2028-01-01")
    assert not j.cron("00:00")["calls"] and health(d) == ["CRITICAL"]
    j.commit_at("06:10", at="04:55")
    j.capture([RIGHT], "2028-01-01T095500Z")
    before = d.early.read_bytes()
    j.cron("05:00", pending=[row()])
    assert j.early_state()["date"] == j.day and health(d) == []
    j.early.write_bytes(before)  # restore the saved snapshot instead
    assert health(j) == [] and health(d) == ["CRITICAL"]
    result = j.cron("05:15", pending=[row()])
    assert result["calls"] and j.early_state()["date"] == j.day
    print("year rollover loses D CRITICAL; an old snapshot cannot be terminal for Jan 1")

    w = World(root / "resurrect")
    w.commit_at("10:05", at="09:35")
    w.cron("09:45", dm="failed")
    w.capture([RIGHT], f"{DAY}T135000Z")
    manual = w.invoke("09:50", manual=True, pending=[row()])
    assert manual["marker"]["status"] == "confirmed" and health(w) == []
    n = World(w.root, day="2027-03-25")
    n.commit_at("19:10", at="18:00")
    n.cron("18:00", pending=[row()])
    assert health(w) == ["CRITICAL"]
    print("historical false CRITICAL as the live recovery is overwritten", compact(manual))

    for kind in ("directory", "fifo", "invalid-utf8", "list", "bad-tiers", "oversize"):
        w = World(root / ("cli-" + kind))
        w.commit_at("10:05", at="09:40")
        create(w.early, kind)
        r = w.cron("09:45")
        assert r["sent"] and r["exception"] == ("IsADirectoryError" if kind == "directory" else "SystemExit"), r
        print("early CLI marker", kind, compact(r))

    for kind in ("fifo", "list", "invalid-utf8", "directory"):
        w = World(root / ("cutoff-" + kind))
        w.commit_at("10:05", at="09:40")
        w.cron("09:45", dm="failed")
        path = w.picks / f"{DAY}.json"
        path.unlink()
        create(path, kind)
        observed = read(w)
        expected = {"fifo": "ReadBlocked", "list": "AttributeError", "invalid-utf8": "UnicodeDecodeError",
                    "directory": ["CRITICAL"]}[kind]
        assert observed == expected, (kind, observed)
        print("new early health cutoff read", kind, observed)

    w = World(root / "R11", day=fallback_recovery.CURRENT)
    create(w.marker, "invalid-utf8")
    levels, messages = fallback_recovery.dispatch(w, w.day)
    assert levels == ["CRITICAL"] and len(messages) == 1 and "UnicodeDecodeError" in messages[0]
    print("R11 full dispatch", levels, messages)


if __name__ == "__main__":
    v = sys.argv[1]
    source(v)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        matrix(root, v)
        if v == "f5":
            sequences(root)
