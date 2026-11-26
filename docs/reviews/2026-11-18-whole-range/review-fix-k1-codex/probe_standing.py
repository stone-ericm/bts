"""Exercise shared-receipt errors and interruptions through the real refusal helper."""

import inspect
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts import contest_state, scheduler
from bts.health import alert
from scripts.audit.season_walk.walk import ET, offline


def run():
    print("bts:", bts.__file__)
    logging.disable(logging.CRITICAL)
    now = datetime(2027, 3, 24, 18, tzinfo=ET)
    day = now.date().isoformat()
    kwargs = {"slate_date": day} if "slate_date" in inspect.signature(scheduler._alert_contest_state_failure).parameters else {}
    with TemporaryDirectory(prefix="wrk1-standing-") as temp, offline(now):
        for case in ("invalid JSON", "invalid UTF-8", "array", "wrong key types", "FIFO", "directory",
                     "no recipient", "send raises", "no confirmation", "interrupt before transport",
                     "interrupt after transport", "interrupt after truncate"):
            root = Path(temp) / case / "data/picks"
            receipt = root.parent / "health_state/health_dm_delivery_status.json"
            receipt.parent.mkdir(parents=True)
            bodies = []

            def send(_, body):
                bodies.append(body)
                return "accepted"

            def call(recipient="capture.invalid"):
                scheduler._alert_contest_state_failure(
                    {"orchestrator": {"picks_dir": str(root)}, "bluesky": {"dm_recipient": recipient}},
                    contest_state.ContestStateError("short real alert"), **kwargs)

            def read():
                return json.loads(receipt.read_text()) if receipt.is_file() else None

            if case == "invalid JSON":
                receipt.write_text("{")
            elif case == "invalid UTF-8":
                receipt.write_bytes(b"\xff\xfe")
            elif case == "array":
                receipt.write_text("[]")
            elif case == "wrong key types":
                receipt.write_text(json.dumps({"status": "sent", "sent_day": day, "sent_sources": "CRITICAL:contest_state"}))
            elif case == "FIFO":
                os.mkfifo(receipt)
            elif case == "directory":
                receipt.mkdir()

            with patch.object(alert, "send_dm", send):
                if case == "no recipient":
                    call(None)
                    assert read()["status"] == "skipped_no_recipient" and not bodies
                elif case == "send raises":
                    with patch.object(alert, "send_dm", side_effect=RuntimeError("transport down")):
                        call()
                    assert read()["status"] == "failed" and not bodies
                elif case == "no confirmation":
                    with patch.object(alert, "send_dm", side_effect=lambda *args: send(*args) and None):
                        call()
                    assert read()["status"] == "sent"
                elif case.startswith("interrupt"):
                    if case == "interrupt before transport":
                        failing = patch.object(alert, "send_dm", side_effect=KeyboardInterrupt)
                    elif case == "interrupt after transport":
                        failing = patch.object(alert, "_persist_health_dm_delivery_status", side_effect=KeyboardInterrupt)
                    else:
                        original = alert.os.ftruncate

                        def truncated(fd, count):
                            original(fd, count)
                            os.write(fd, b'{"status":')
                            raise KeyboardInterrupt

                        failing = patch.object(alert.os, "ftruncate", truncated)
                    try:
                        with failing:
                            call()
                    except KeyboardInterrupt:
                        if case == "interrupt after truncate":
                            assert receipt.read_bytes() == b'{"status":'
                        else:
                            assert not receipt.exists()
                    else:
                        raise AssertionError("interrupt swallowed")

                # Call with freshly constructed configuration using only the
                # surviving on-disk record, as a restarted scheduler would.
                call()
                regular = case not in ("FIFO", "directory")
                if regular:
                    assert read()["status"] == "sent" and read()["sent_day"] == day
                call()
                expected = 2 if case in ("FIFO", "directory", "interrupt after transport", "interrupt after truncate") else 1
                assert len(bodies) == expected, (case, len(bodies))
                if case == "send raises":
                    assert bodies[0].startswith("⚠️ Previous health alert DM FAILED")
                print(case, "transport calls:", len(bodies), "regular sent receipt:", regular)


if __name__ == "__main__":
    run()
