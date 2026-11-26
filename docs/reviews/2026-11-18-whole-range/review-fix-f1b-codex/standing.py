"""Exercise malformed state, late responses, receipt wording and calendar rows."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo
import json

from bts import entry_receipt
from bts.cli import cli
from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World


HERE = Path(__file__).resolve().parent


def world(root, name):
    data = World(root / name)
    data.commit(pitch=PITCH)
    return data


def malformed(root):
    for name, data, expected in (("directory", None, "match_game_unchecked"),
                                 ("truncated", b"\x1f\x8b\x08", "match_game_unchecked"),
                                 ("silent", None, "match")):
        fixture = world(root, name)
        fixture.capture([WRONG])
        assert fixture.check("18:30", MATCH)["marker"]["game_unresolved"]
        path = fixture.capture([])
        if name == "directory":
            path.unlink()
            path.mkdir()
        elif name == "truncated":
            path.write_bytes(data)
        else:
            fixture.capture({"by-id": [WRONG]})
        got = fixture.check("18:45", MATCH)
        if name == "silent":
            assert got["marker"]["game_unresolved"] and not got["stderr"]
        else:
            assert got["marker"]["reason"] == expected
            assert got["dm"] == [f"✅ BTS pick entry: the batter is entered for {DAY}; the game could not be checked."]
            assert fixture.check("19:00", MATCH)["fetches"] == []
        print(name, got["marker"]["reason"], got["dm"])

    for name, data, expected in (("bad-json", b"{", None), ("bad-shape", b"[]", "AttributeError"),
                                 ("bad-encoding", b"\xff", "UnicodeDecodeError"),
                                 ("bad-tiers", json.dumps({"date": DAY, "status": "alerted",
                                                          "escalations": 5}).encode(), "TypeError")):
        fixture = world(root, name)
        fixture.marker.parent.mkdir(exist_ok=True)
        fixture.marker.write_bytes(data)
        got = fixture.check("18:45", MATCH)
        assert got["exception"] == expected
        print(name, got["exception"], got["stdout"].strip())

    fixture = world(root, "marker-is-directory")
    fixture.marker.mkdir(parents=True)
    failure = fixture.check("18:30")
    assert failure["exception"] in ("IsADirectoryError", "OSError") and failure["dm"]
    print("marker-is-directory", failure["exception"], "after the base-form DM")

    fixture = world(root, "bad-saved-id")
    fixture.commit(game=[101], pitch=PITCH)
    fixture.capture([RIGHT])
    fixture.check("18:25")
    got = fixture.check("18:45", MATCH)
    print("saved game ID [101]", got["marker"]["reason"], got["stderr"].strip(), got["dm"])


def outcomes(root):
    fixture = world(root, "morning")
    fixture.capture([WRONG])
    missing = fixture.morning()
    assert missing["outcome"]["state"] == "entry_unverifiable" and len(missing["dm"]) == 1
    sent = []
    with patch("bts.scheduler.fetch_schedule", lambda _: [
            {"gamePk": 101, "status": {"detailedState": "Postponed"}}]), \
         patch("bts.dm.send_dm", lambda _, text: sent.append(text)):
        void = fixture.runner.invoke(cli, ["day-outcome", "--date", DAY, "--config", str(fixture.config),
                                            "--picks-dir", str(fixture.picks), "--expected-username", "operator",
                                            "--dm-recipient", "fixture.invalid"])
    assert void.exit_code == 0 and json.loads(void.stdout)["state"] == "no_games" and not sent
    print("same scoreable decision, no receipts", missing["outcome"]["state"], "vs all postponed", void.stdout.strip())

    warned = fixture.check("18:30", MATCH)
    assert warned["marker"]["game_unresolved"]
    fixture.capture([RIGHT], f"{DAY}T230000Z")
    later = fixture.check("18:45", MATCH)
    assert later["exit"] == 1
    got = fixture.morning()
    assert got["outcome"]["state"] == "entered_game_unverified" and not got["dm"]
    assert fixture.read_marker()["game_unresolved"] and fixture.health()[0]["level"] == "WARN"
    print("A6.5 later clean receipt", got["outcome"]["state"], "marker remains unresolved")

    fixture = world(root, "interrupt")
    fixture.capture([WRONG])
    for exc in [KeyboardInterrupt(), SystemExit(7)]:
        def raises(*_):
            raise exc

        got = fixture.check("18:30", MATCH, more_patches=[("bts.entry_receipt.qualify_run", raises)])
        assert got["marker"] is None and not got["dm"]
        assert entry_receipt.discover(fixture.picks, DAY)
        print(type(exc).__name__, got["exit"], got["receipts"][-1].get("detail"))

    fixture = world(root, "unconfirmed-send")
    fixture.capture([WRONG])
    accepted = []

    def failed(_, text):
        accepted.append(text)
        raise RuntimeError("connection lost after synthetic acceptance")

    first = fixture.check("18:30", MATCH, more_patches=[("bts.dm.send_dm", failed)])
    assert first["marker"]["status"] == "dm_failed" and first["marker"]["game_unresolved"]
    assert fixture.health()[0]["level"] == "CRITICAL"
    assert fixture.check("18:45", MATCH)["dm"] == accepted
    print("ambiguous DM transport acceptance repeats the warning")


def calendar():
    et = ZoneInfo("America/New_York")
    for day in ("2027-03-14", "2027-11-07"):
        start = datetime.fromisoformat(day + "T00:00:00+00:00")
        slots = []
        for hours in range(36):
            for minute in (0, 15, 30, 45):
                wall = (start + timedelta(hours=hours, minutes=minute)).astimezone(et)
                if wall.date().isoformat() == day and wall.hour in range(10, 24):
                    slots.append(wall)
        assert len(slots) == len(set(slots)) == 56
        assert all((b.timestamp() - a.timestamp()) == 900 for a, b in zip(slots, slots[1:]))
        print(day, len(slots), "separate slots, 900 real seconds apart")
    launch = datetime.fromisoformat("2027-12-31T23:45:00").replace(tzinfo=et)
    assert (launch + timedelta(seconds=840)).strftime("%H:%M:%S") == "23:59:00"
    assert (launch + timedelta(seconds=870)).strftime("%H:%M:%S") == "23:59:30"
    assert (launch + timedelta(seconds=931)).date() != launch.date()
    # The cron expression uses wall time, while timeout uses elapsed time.
    # A 45-second forward correction repeats neither the nominal DST case nor its timing.
    assert 900 - 45 < 840 + 30
    print("23:45 ET ends 23:59/23:59:30 nominally; a 61-second delay crosses midnight")
    print("45-second forward step can put the next wall slot 855 elapsed seconds after the previous launch")


if __name__ == "__main__":
    with TemporaryDirectory(dir=HERE) as tmp:
        root = Path(tmp)
        malformed(root)
        outcomes(root)
        calendar()
