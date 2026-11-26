"""Replay the state cases with a fresh MLB unit ID when the entry changes games."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from state_sequences import (
    DAY, RIGHT, WRONG, Replay, cron, display, early_after_original, edit_pitch,
    manual_carries_tiers, moving_double, row, slot_gained_then_lost, source,
)


def duplicate_clear(root, version):
    world = Replay(root, version)
    daily = world.commit_at("10:05", at="09:10")
    world.capture([WRONG], f"{DAY}T131000Z")
    warning = cron(world, version, "09:15", pending=[row()])
    display("all-clear 09:15", warning)
    world.capture([{**RIGHT, "id": 78}], f"{DAY}T132900Z")
    fixed = [row(78)]
    first = cron(world, version, "09:30", pending=fixed)
    display("all-clear 09:30", first)
    display("all-clear 09:45", cron(world, version, "09:45", pending=fixed))
    edit_pitch(world, daily, "10:50")
    second = cron(world, version, "10:00", pending=fixed)
    display("all-clear 10:00", second)
    clears = [message for run in (first, second) if run for message in run["sent"]]
    assert clears == ([f"✅ BTS pick entry confirmed for {DAY} (match)."] * 2 if version == "f5" else [])
    return clears


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        cases = (manual_carries_tiers, duplicate_clear, slot_gained_then_lost, moving_double, early_after_original)
        results = {case.__name__: case(root / case.__name__, version) for case in cases}
        print(json.dumps(results, ensure_ascii=False))
