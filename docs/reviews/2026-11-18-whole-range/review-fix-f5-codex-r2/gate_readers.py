"""Pin gate edges and exercise no-attempt paths, receipt discovery and backup scope."""
from collections import Counter
from datetime import timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from support import DAY, Replay, clock_slots, legacy, load_line, outcome, source
from bts import day_outcome, entry_receipt


def check(world, at, expected, *, reads=False, dm=False, write=False, **kwargs):
    got = world.invoke(at, **kwargs)
    assert outcome(got) == expected, got
    assert bool(got["calls"]) == reads and bool(got["attempts"]) == dm and got["marker_changed"] == write, got
    if reads:
        assert got["calls"] == ["cookies", "login", "profile", "pending", "rounds", "players"], got
    print(world.root.name, at, expected, f"calls={reads} dm={dm} write={write}")
    return got


def gate(root):
    for pitch, has_slot, attempts in [
        ("10:05", False, (("08:49", "outside_window"), ("08:50", "observed"), ("10:00", "outside_window"))),
        ("10:06", True, (("09:45", "base_slot_available"), ("10:00", "observed"))),
        ("11:15", True, (("09:45", "outside_window"), ("10:00", "observed"))),
        ("11:16", True, (("10:00", "outside_window"), ("10:15", "observed"))),
    ]:
        world = Replay(root / pitch.replace(":", ""))
        world.commit_at(pitch, at="08:40", dd=(1020, 202, "23:00"))
        assert day_outcome._has_original_slot(DAY, world.picks) is has_slot
        for at, expected in attempts:
            observes = expected == "observed"
            got = check(world, at, expected, reads=observes, dm=observes, write=observes)
            assert ("early_cron_run" in got["receipt"]) == (observes and at < "10:00")
    world = Replay(root / "late-primary")
    world.commit_at("21:30", dd=(1020, 202, "10:05"), at="08:40")
    assert not day_outcome._has_original_slot(DAY, world.picks)
    check(world, "09:45", "outside_window", window=19)
    result = check(world, "09:46", "observed", window=19, reads=True, dm=True, write=True)
    assert result["receipt"]["early_cron_run"] is True


def traffic(root):
    for state in ("absent", "uncommitted", "evening", "terminal", "shared"):
        world = Replay(root / state / "data")
        if state != "absent":
            pitch = {"uncommitted": "06:10", "evening": "19:10", "terminal": "06:10", "shared": "10:50"}[state]
            world.commit_at(pitch, delivery=None if state == "uncommitted" else "delivered")
        if state == "terminal":
            world.set_marker({"date": DAY, "status": "confirmed"})
        outcomes = Counter()
        for slot in clock_slots(load_line("f5")):
            if slot.hour >= 10 and state != "absent":
                break
            got = world.invoke(slot.strftime("%H:%M"))
            assert not (got["calls"] or got["attempts"] or got["marker_changed"]), got
            if outcome(got) != "already_confirmed":
                assert "early_cron_run" not in got["receipt"], got
            outcomes[outcome(got)] += 1
        files = list(entry_receipt.receipts_root(world.picks).joinpath(DAY).glob("*.json.sealed"))
        assert len(files) == sum(outcomes.values()) == (96 if state == "absent" else 40)
        print(state, dict(outcomes), "zero calls/DMs/marker writes")
        if state == "absent":
            # A preserved marker and a true-only field live under the same ops backup path.
            world.commit_at("06:10")
            dated = world.marker.with_name("pick_entry_check.2027-03-23.json")
            dated.parent.mkdir(exist_ok=True)
            dated.write_bytes(b'{"date":"2027-03-23","status":"dm_failed","early_cron_run":true}\n')
            legacy("traffic_probe").verify_backup(world.root.parent)
            assert dated.is_file() and json.loads(dated.read_bytes())["early_cron_run"] is True


def clocks(root):
    for day, expected in (("2027-03-14", 92), (DAY, 96), ("2027-11-07", 100)):
        instants = list(clock_slots(load_line("f5"), day))
        assert len(instants) == expected
        assert len(list(clock_slots(load_line("base"), day))) == 56
        utc = [slot.astimezone(timezone.utc) for slot in instants]
        assert all(second - first == timedelta(minutes=15) for first, second in zip(utc, utc[1:]))
        world = Replay(root / day, day=day)
        if expected != 96:
            for slot in instants:
                if 0 <= slot.hour <= 3:
                    got = world.invoke(slot.strftime("%H:%M"))
                    assert got["calls"] == got["attempts"] == [] and outcome(got) == "no_pick_file"
        print(day, expected, "projected slots with 900-second intervals", Counter(slot.hour for slot in instants if slot.hour in (1, 2)))


if __name__ == "__main__":
    source("f5")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        gate(root / "gate")
        traffic(root / "traffic")
        clocks(root / "clocks")
