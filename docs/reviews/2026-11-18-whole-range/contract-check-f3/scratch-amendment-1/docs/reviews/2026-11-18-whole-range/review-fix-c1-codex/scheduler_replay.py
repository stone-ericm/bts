"""Capture the actual EOD contest call even when the repaired replay sends no DM."""

from datetime import datetime
import importlib
import json
import logging
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from replay_drivers import DRIVERS, HERE, ROOT


class ReplayObserved(Exception):
    """Stop a defect-expecting archived driver immediately after verified EOD."""


def evidence(driver, revision, disk_warning):
    source = HERE / "base/src" if revision == "base" else ROOT / "src"
    sys.path[:0] = [str(source), str(DRIVERS)]
    import bts
    from bts.health import runner

    assert Path(bts.__file__).resolve().is_relative_to(source)
    print(f"{revision}: {bts.__file__}")
    probe = importlib.import_module("probe")
    probe.ROOT = ROOT
    probe.SLICES = ROOT / "docs/reviews/2026-11-18-whole-range"
    sys.path.insert(0, str(probe.SLICES / "slice-1-carbon"))
    from test_slice1_probes_v3 import RehearsalWorld

    original = RehearsalWorld.run_scheduler_day

    def observe(world, day, **kwargs):
        if disk_warning:
            world.disk_used_pct = 0.92

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return world.clock.now_et().astimezone(tz)

        actual = runner.contest_state.check
        calls = []

        def trace(*args, **kwargs):
            items = actual(*args, **kwargs)
            calls.append((world.clock.now_et(), kwargs, items))
            return items

        def snapshot():
            return {path.name: path.read_bytes() for path in (world.picks_dir / "account_state").iterdir()}

        before = snapshot()
        with (patch.object(runner, "datetime", Clock, create=True),
              patch.object(runner.contest_state, "check", trace)):
            original(world, day, **kwargs)
        assert snapshot() == before
        (call_time, kwargs, items), = calls
        assert call_time.isoformat() == "2027-03-25T00:15:00-04:00", calls
        dms = [dm["text"] for dm in world.dms if dm["text"].startswith("BTS health ")]
        status_path = world.picks_dir.parent / "health_state/health_dm_delivery_status.json"
        receipt = json.loads(status_path.read_text()) if status_path.exists() else None
        if revision == "head":
            assert str(kwargs["historical_day"]) == day and items == [], calls
            assert all(a not in body for a in ("[contest_state]", "[saver_state]") for body in dms), dms
            if disk_warning:
                assert len(dms) == 1 and "[disk_fill]" in dms[0] and receipt["sent_day"] == day, (dms, receipt)
            else:
                assert dms == [] and receipt is None, (dms, receipt)
        else:
            assert kwargs == {"expected": True}
            assert [(item.level, item.source) for item in items] == [
                ("WARN", "saver_state"), ("CRITICAL", "contest_state"),
            ]
            assert len(dms) == 1 and "--date 2027-03-25" in dms[0] and receipt["sent_day"] == day
        print(f"EOD {call_time.isoformat()}, kwargs={kwargs}, contest alerts={len(items)}, "
              f"health DMs={len(dms)}, receipt day={receipt and receipt['sent_day']}")
        for body in dms:
            print(body)
        raise ReplayObserved

    with TemporaryDirectory(dir=HERE) as scratch:
        try:
            with patch.object(RehearsalWorld, "run_scheduler_day", observe):
                if driver == "operator":
                    from operator_cases import full_scheduler

                    full_scheduler(Path(scratch))
                else:
                    probe.scheduler(Path(scratch))
        except ReplayObserved:
            return
    raise AssertionError("The archived driver did not reach the verified EOD call")


if __name__ == "__main__":
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    evidence(sys.argv[1], sys.argv[2], "--disk-warning" in sys.argv[3:])
