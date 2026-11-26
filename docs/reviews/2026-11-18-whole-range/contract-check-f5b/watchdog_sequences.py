"""Compare the installed window, account traffic and independent alert ledgers."""
from datetime import datetime
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from replay_support import DAY, ET, RIGHT, WRONG, World, compact, health, load_line, probe_support, row, source
from bts import entry_receipt


def schedule(root, v):
    for strike, expected_normal, expected_spring, expected_fall in (("f5", 96, 92, 100), ("base", 56, 56, 56)):
        if strike != v:
            continue
        for day, expected in ((DAY, expected_normal), ("2027-03-14", expected_spring),
                              ("2027-11-07", expected_fall)):
            w = World(root / day, v, day)
            slots = list(probe_support.clock_slots(load_line(v), day))
            assert len(slots) == expected
            for t in slots:
                r = w.invoke(t.strftime("%H:%M"), persist=day == DAY)
                assert not (r["calls"] or r["attempts"] or r["marker_changed"])
            print("slots, no account:", day, len(slots))
            if day == DAY:
                assert len(entry_receipt.discover(w.picks, day)) == expected

    for pitch, nominal in (("06:10", ["05:15", "05:30", "05:45", "06:00"]),
                           ("10:05", ["09:00", "09:15", "09:30", "09:45"]),
                           ("19:10", ["18:00", "18:15", "18:30", "18:45", "19:00"])):
        for entered in (True, False):
            w = World(root / f"{pitch}-{entered}", v)
            w.capture([RIGHT], f"{DAY}T095500Z")
            w.commit_at(pitch, at="05:10" if pitch == "06:10" else "07:00")
            active = nominal if v == "f5" or pitch == "19:10" else []
            if v == "f5" and pitch == "06:10":
                # The pre-commit invocation needs the pre-commit filesystem.
                z = World(root / "05:00")
                assert z.cron("05:00")["receipt"]["outcome"] == "no_pick_file"
            runs = [w.cron(at, pending=[row()] if entered else []) for at in active]
            assert [at for at, r in zip(active, runs) if r["calls"]] == (active[:1] if entered else active)
            print("window", pitch, entered, list(zip(active, map(compact, runs))), w.morning())


def separate(root, v):
    w = World(root / "manual-fixes-last-slot", v)
    w.commit_at("10:05", at="09:40")
    w.capture([WRONG], f"{DAY}T134000Z")
    early = w.cron("09:45", pending=[row()])
    w.capture([{**RIGHT, "id": 78}], f"{DAY}T135000Z")
    manual = w.invoke("09:50", manual=True, pending=[row(78)])
    assert manual["marker"]["status"] == "confirmed" and not manual["sent"]
    print("last-slot manual fix", compact(early), compact(manual), w.morning(), health(w))
    if v == "base":
        return
    assert early["sent"] and w.early_state()["status"] == "alerted"
    assert w.cron("10:00")["receipt"]["outcome"] == "outside_window"

    w = World(root / "duplicate", v)
    w.commit_at("10:05", at="09:00")
    a = w.cron("09:15")
    b = w.invoke("09:15:01", manual=True)
    assert len(a["sent"]) == len(b["sent"]) == 1 and a["sent"] == b["sent"]
    assert a["calls"] == b["calls"] and w.early_state()["status"] == w.read_marker()["status"]
    w.capture([RIGHT], f"{DAY}T132000Z")
    a = w.cron("09:30", pending=[row()])
    b = w.invoke("09:31", manual=True, pending=[row()])
    assert a["sent"] == b["sent"] and a["sent"][0].startswith("✅")
    print("duplicate warning and all-clear across ledgers", compact(a), compact(b))

    w = World(root / "replaced-early", v)
    w.commit_at("06:10", at="04:55")
    w.capture([RIGHT], f"{DAY}T095500Z")
    w.cron("05:00", pending=[row()])
    w.commit_at("06:10", at="05:10", game=102)
    result = w.cron("05:15", pending=[row()])
    assert result["receipt"]["outcome"] == "already_confirmed" and not result["calls"]
    proof = w.invoke("05:16", pending=[row()], manual=True)
    assert proof["sent"] and "recorded in game 101" in proof["sent"][0]
    print("same-date replacement skipped by early terminal marker", compact(result), compact(proof))

    w = World(root / "late-slot", v)
    saved = w.commit_at("10:05:01", at="09:40")
    before = w.cron("09:45")
    after = w.invoke("10:00:02")
    assert before["receipt"]["outcome"] == "base_slot_available" and after["receipt"]["outcome"] == "outside_window"
    print("nominal 10:00 slot unavailable at actual 10:00:02", compact(before), compact(after))

    w = World(root / "alert-retry", v)
    w.commit_at("06:10", at="04:55")
    failed = w.cron("05:00", dm="failed")
    snap = w.early.read_bytes()
    d = w.cron("05:15", dm="interrupt")
    assert w.early.read_bytes() == snap and not w.marker.exists()
    retry = w.cron("05:30")
    assert retry["sent"] and w.early_state()["status"] == "alerted" and not w.marker.exists()
    print("failed send, interruption and retry", compact(failed), compact(d), compact(retry))


if __name__ == "__main__":
    v = sys.argv[1]
    source(v)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        schedule(root, v)
        separate(root, v)
