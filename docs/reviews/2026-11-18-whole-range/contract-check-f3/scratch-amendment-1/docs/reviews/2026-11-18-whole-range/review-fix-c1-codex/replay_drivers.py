"""Run the supplied defect drivers with verified imports and fixed scheduler clocks."""

from contextlib import nullcontext
from datetime import datetime
import importlib
import json
from pathlib import Path
import runpy
import signal
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRIVERS = HERE / "contract-check-c1"


def main():
    revision, driver, mode = sys.argv[1:]
    source = HERE / "base/src" if revision == "base" else ROOT / "src"
    sys.path.insert(0, str(source))
    import bts

    assert Path(bts.__file__).resolve().is_relative_to(source)
    print(f"{revision} bts: {bts.__file__}", flush=True)
    sys.path.insert(0, str(DRIVERS))
    probe = importlib.import_module("probe")
    # The archived driver prepends HEAD's src. Keep the selected tree first.
    sys.path.insert(0, str(source))
    from bts.health import runner

    scheduler_context = nullcontext()
    observed = []
    if mode == "scheduler":
        sys.path.insert(0, str(probe.SLICES / "slice-1-carbon"))
        from test_slice1_probes_v3 import RehearsalWorld

        original = RehearsalWorld.run_scheduler_day

        def run_and_capture(world, day, **kwargs):
            class Clock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return world.clock.now_et().astimezone(tz)

            with patch.object(runner, "datetime", Clock, create=True):
                original(world, day, **kwargs)
            dms = [dm for dm in world.dms if dm["text"].startswith("BTS health ")]
            assert len(dms) == 1, dms
            receipt_path = world.picks_dir.parent / "health_state/health_dm_delivery_status.json"
            receipt = json.loads(receipt_path.read_text())
            assert receipt["sent_day"] == day, receipt
            observed.extend(dms)
            print(f"DM at {dms[0]['at_et']}; receipt day {day}:\n{dms[0]['text']}", flush=True)

        scheduler_context = patch.object(RehearsalWorld, "run_scheduler_day", run_and_capture)

    path = DRIVERS / driver
    sys.argv = [str(path), mode]
    try:
        with scheduler_context:
            runpy.run_path(str(path), run_name="__main__")
    except AssertionError:
        if not observed:
            raise
        body = observed[0]["text"]
        if revision == "head":
            assert "[contest_state]" not in body and "[saver_state]" not in body, body
            assert "--date 2027-03-25" not in body, body
            print("Archived defect assertion failed at the intended point; HEAD EOD has no false contest/saver item.")
        elif driver == "probe.py":
            assert "[contest_state] Live decision refused" in body and "[saver_state]" in body
            assert "--date 2027-03-25" in body
            print("Carbon's old copy assertion failed on fix B wording; the recorded base defect is present.")
        else:
            raise
    else:
        if mode == "scheduler":
            assert revision == "base", "Archived defect assertions unexpectedly passed on HEAD"


if __name__ == "__main__":
    signal.alarm(55)
    main()
