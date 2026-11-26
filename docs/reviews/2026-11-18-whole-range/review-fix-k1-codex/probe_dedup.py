"""Run both ET boundaries, invalid dates, and EOD-before/after-old-slate order."""

import inspect
import json
import logging
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts import contest_state, scheduler
from bts.health import alert, runner
from scripts.audit.season_walk.walk import _set_clock, offline
from tests.test_wr_k1_refusal_dm import dms_at, local

CURRENT = "slate_date" in inspect.signature(scheduler._alert_contest_state_failure).parameters
SOURCES = ("contest_state", "missed_pick", "late_delivery", "postponed_pick")


def config(root):
    return {"orchestrator": {"picks_dir": str(root)}, "bluesky": {"dm_recipient": "capture.invalid"},
            "scheduler": {"contest_state_required": True}}


def run():
    print("bts:", bts.__file__)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(prefix="wrk1-date-") as temp, offline(local("2027-06-14", "20:00")):
        number = 0
        bodies = []

        def send(_, body):
            bodies.append(body)
            return "captured"

        def fresh():
            nonlocal number
            number += 1
            root = Path(temp) / str(number) / "data/picks"
            root.mkdir(parents=True)
            return root, root.parent / "health_state/health_dm_delivery_status.json"

        with patch.object(alert, "send_dm", send):
            for old, new in (("2027-06-14", "2027-06-15"), ("2027-12-31", "2028-01-01")):
                for source in SOURCES:
                    for name, sequence, base, current in (
                        ("old then new", [(old, new, "00:02"), (new, new, "08:00")], [1, 0], [1, 1]),
                        ("old starts before midnight", [(old, old, "23:30"), (old, new, "00:02"),
                                                       (old, new, "00:07")], [1, 1, 0], [1, 0, 0]),
                        ("new-old-new", [(new, new, "00:01"), (old, new, "00:02"), (new, new, "08:00")],
                         [1, 0, 0], [1, 1, 1]),
                        ("old starts after midnight", [(old, new, "00:02"), (old, new, "00:07")],
                         [1, 0], [1, 0]),
                    ):
                        root, receipt = fresh()
                        actual = []
                        for slate, wall, hm in sequence:
                            at = local(wall, hm)
                            _set_clock(at)
                            before = len(bodies)
                            dms_at(source, root, slate, at, config(root))
                            actual.append(len(bodies) - before)
                            if actual[-1]:
                                row = json.loads(receipt.read_text())
                                assert row["sent_day"] == (slate if CURRENT else wall), row
                                assert row["sent_sources"] == ["CRITICAL:" + source], row
                        assert actual == (current if CURRENT else base), (name, source, actual)
                        print(old, source, name, actual)

                for eod_first in (False, True):
                    root, receipt = fresh()
                    account = root / "account_state"
                    account.mkdir()
                    automatic = account / "contest_streak.json"
                    automatic.write_text("[]")
                    real_safe_run = runner._safe_run

                    def eod():
                        with patch.object(runner, "_safe_run", lambda name, fn:
                                          real_safe_run(name, fn) if name == "contest_state" else []):
                            found = runner.run_all_checks(root, root.parent / "models", "capture.invalid",
                                                          today=date.fromisoformat(old), contest_state_expected=True)
                        assert any(a.source == "contest_state" and "malformed" in a.message for a in found)

                    def in_day():
                        # Read a distinct real failure, leaving the historical fixture
                        # in place for the digest that follows this scheduler check.
                        with patch.object(scheduler, "check_confirmed_lineups", lambda _: {}), patch.object(
                                scheduler, "run_and_pick", side_effect=contest_state.ContestStateError(
                                    "live observation for this activation disappeared")):
                            result = scheduler.run_single_check(old, [1], set(), config(root), 0.03)
                        assert result["pick_result"] is None

                    before = len(bodies)
                    _set_clock(local(new, "00:02"))
                    (eod if eod_first else in_day)()
                    _set_clock(local(new, "00:12"))
                    (in_day if eod_first else eod)()
                    _set_clock(local(new, "00:22"))
                    eod()
                    actual = len(bodies) - before
                    assert actual == (1 if CURRENT else (3 if eod_first else 2)), (old, eod_first, actual)
                    print(old, "historical-first" if eod_first else "in-day-first", actual,
                          json.loads(receipt.read_text())["sent_day"])

            for source in SOURCES:
                for bad in ("", "bad", "2027-02-29"):
                    root, receipt = fresh()
                    at = local("2027-06-15", "00:02")
                    _set_clock(at)
                    dms_at(source, root, bad, at, config(root))
                    assert json.loads(receipt.read_text())["sent_day"] == "2027-06-15"
                    print(source, "bad date", repr(bad), "wall ET")


if __name__ == "__main__":
    run()
