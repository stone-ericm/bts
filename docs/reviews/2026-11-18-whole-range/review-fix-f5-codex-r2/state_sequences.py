"""Replay corrected picks, fallible manual checks and early confirmations."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from support import DAY, Replay, clock_slots, load_line, outcome, source
from bts.picks import save_pick
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row


def edit_pitch(world, daily, pitch, leg="pick"):
    getattr(daily, leg).game_time = f"{world.day}T{pitch}:00-04:00"
    save_pick(daily, world.picks)


def cron(world, version, at, **kwargs):
    slots = {slot.strftime("%H:%M") for slot in clock_slots(load_line(version), world.day)}
    if at in slots:
        return world.invoke(at, **kwargs)
    return None


def display(name, run):
    if run is None:
        print(name, "no installed slot")
        return
    receipt = run["receipt"]
    marker = {key: value for key, value in (run["marker"] or {}).items() if key != "receipt"}
    print(name, outcome(run), receipt.get("early_cron_run"), run["calls"], marker, run["sent"])
    assert run["exception"] is None or run["exception"] == "SystemExit", run


def manual_carries_tiers(root, version):
    world = Replay(root, version)
    daily = world.commit_at("10:05", at="09:40")
    display("manual ledger 09:45", cron(world, version, "09:45"))
    edit_pitch(world, daily, "10:06")
    manual = world.invoke("09:55", manual=True, pending=[row()], crosswalk={})
    display("manual ledger 09:55", manual)
    assert manual["marker"]["status"] == "present_unverified" and not manual["sent"]
    assert "early_cron_run" not in manual["marker"]
    result = cron(world, version, "10:00")
    display("manual ledger 10:00", result)
    assert bool(result["sent"]) == (version == "base")
    return result["sent"]


def duplicate_clear(root, version):
    world = Replay(root, version)
    daily = world.commit_at("10:05", at="09:10")
    world.capture([WRONG], f"{DAY}T131000Z")
    warning = cron(world, version, "09:15", pending=[row()])
    display("all-clear 09:15", warning)
    # A later capture sees the correction in MLB, ahead of a separate start-time correction.
    world.capture([RIGHT], f"{DAY}T132900Z")
    first = cron(world, version, "09:30", pending=[row()])
    display("all-clear 09:30", first)
    display("all-clear 09:45", cron(world, version, "09:45", pending=[row()]))
    edit_pitch(world, daily, "10:50")
    second = cron(world, version, "10:00", pending=[row()])
    display("all-clear 10:00", second)
    clears = [message for run in (first, second) if run for message in run["sent"]]
    assert clears == ([f"✅ BTS pick entry confirmed for {DAY} (match)."] * 2 if version == "f5" else [])
    return clears


def slot_gained_then_lost(root, version):
    world = Replay(root, version)
    daily = world.commit_at("10:05", at="09:40")
    world.capture([RIGHT], f"{DAY}T134000Z")
    early = cron(world, version, "09:45", pending=[row()])
    display("slot toggles 09:45", early)
    edit_pitch(world, daily, "10:20")
    ten = cron(world, version, "10:00")
    display("slot toggles 10:00", ten)
    assert "BTS pick NOT entered" in ten["sent"][0]
    edit_pitch(world, daily, "10:05")
    display("slot toggles 10:15", cron(world, version, "10:15"))
    morning = world.morning()
    print("slot toggles morning", morning)
    assert morning[1]["state"] == ("entered" if version == "f5" else "not_entered")
    assert len(morning[2]) == (version == "base")
    return morning


def moving_double(root, version):
    world = Replay(root, version)
    daily = world.commit_at("10:55", at="09:40", dd=(1020, 202, "10:05"))
    pending = [row(), row(78, 2, 2)]
    display("double 09:45", cron(world, version, "09:45", pending=pending))
    edit_pitch(world, daily, "11:15", leg="double_down")
    world.capture([RIGHT, {"id": 78, "roundId": 1, "feedId": 203}], f"{DAY}T140000Z")
    result = cron(world, version, "10:00", pending=pending)
    display("double 10:00", result)
    assert "entry is recorded in game 203" in result["sent"][0]
    result = world.morning()
    assert result[1]["state"] == "entry_unverifiable" and result[2]
    print("double morning", result)
    return result[1]["state"]


def early_after_original(root, version):
    world = Replay(root, version)
    daily = world.commit_at("10:50", at="09:40")
    world.capture([RIGHT], f"{DAY}T140000Z")
    first = cron(world, version, "10:00", pending=[row()])
    display("historical order 10:00", first)
    edit_pitch(world, daily, "10:05")
    historical = cron(world, version, "09:45", pending=[row()])
    display("historical order 09:45", historical)
    if version == "f5":
        assert outcome(historical) == "already_confirmed" and not historical["marker_changed"]
    edit_pitch(world, daily, "10:50")
    last = cron(world, version, "10:15", pending=[row()])
    display("historical order 10:15", last)
    assert outcome(last) == "already_confirmed" and not last["calls"]
    return outcome(last)


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cases = (manual_carries_tiers, duplicate_clear, slot_gained_then_lost, moving_double, early_after_original)
        results = {}
        for case in cases:
            results[case.__name__] = case(root / case.__name__, version)
        print(json.dumps(results, ensure_ascii=False))
