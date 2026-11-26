"""Exercise the remaining clock and validation variants against base."""

from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import signal
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch

from final_copy import POINTER
import probe


def render(picks, day, original_text, reason):
    from bts.health.contest_state import check

    alerts = check(picks, expected=True, now=probe.end(day))
    matches = [alert for alert in alerts if original_text in alert.message]
    assert len(matches) == 1, alerts
    alert, = matches
    print(f"Base {alert.level} [{alert.source}] {alert.message.splitlines()[0]}")
    print(f"Candidate {alert.level} [{alert.source}] Prior slate {day} ET. {reason} {POINTER}")


def run(root):
    day = probe.D
    next_day = day + timedelta(days=1)
    picks = root / "02_healthy_dated" / "picks"
    probe.dated(picks)
    probe.auto(picks)
    probe.switch(picks, status="revoked")
    probe.saver_flag(picks, state="not_earned")
    wall, _ = probe.real_runner(picks, day, probe.at(next_day, 2))
    assert [(a.level, a.source) for a in wall] == [("WARN", "saver_state"), ("CRITICAL", "contest_state")]
    assert all(f"--date {next_day}" in a.message for a in wall)
    historical, _ = probe.real_runner(picks, day, probe.end(day))
    assert not historical
    print("Healthy retained D, real runner at 02:00: two false wall alerts, zero at D's endpoint")

    picks = root / "0110_incompatible" / "picks"
    probe.switch(picks)
    auto = probe.auto(picks, activation_id="b" * 32, recorded_at=probe.at(next_day, 1, 10).isoformat())
    probe.saver_flag(picks)
    render(picks, day, "activation_id is", "The automatic observation and season switch have different activation IDs.")
    actual, _ = probe.real_runner(picks, day, probe.at(next_day, 2))
    assert len(actual) == 1 and "activation_id is" in actual[0].message
    assert json.loads(auto.read_text())["recorded_at"] == probe.at(next_day, 1, 10).isoformat()

    for label, changes, original, reason in (
        ("best", {"best_streak": 1}, "active_streak=", "The automatic contest streak file has an invalid streak or best streak."),
        ("saver", {"saver_available": "invalid"}, "saver_available invalid", "The automatic contest streak file has an invalid saver value."),
        ("metadata", {"activation_id": "invalid"}, "activation_id invalid", "The automatic contest streak file failed validation for activation_id."),
    ):
        picks = root / label / "picks"
        probe.auto(picks, **changes)
        render(picks, day, original, reason)

    picks = root / "dated_validation" / "picks"
    dated = probe.dated(picks)
    data = json.loads(dated.read_text())
    data["saver_state"] = "invalid"
    dated.write_text(json.dumps(data), encoding="utf-8")
    render(picks, day, "check 7, saver_state", "The dated manual contest streak file failed validation for saver_state.")

    picks = root / "switch_check_one" / "picks"
    probe.auto(picks)
    probe.saver_flag(picks)
    probe.switch(picks).write_text("{", encoding="utf-8")
    render(picks, day, "corrupt season switch, check 1", "The season switch file is malformed or unreadable.")

    picks = root / "same_year_invalid_saver" / "picks"
    probe.auto(picks)
    probe.switch(picks)
    probe.saver_flag(picks, state="invalid")
    render(picks, day, "stale-season flag", "The saver flag is uninitialized or unreadable. The saver counted as spent.")


if __name__ == "__main__":
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=probe.HERE) as scratch:
        scratch = Path(scratch)
        os.chdir(scratch)
        with patch.object(socket, "socket", probe.deny_network), patch.object(socket, "create_connection", probe.deny_network):
            run(scratch)
    signal.alarm(0)
