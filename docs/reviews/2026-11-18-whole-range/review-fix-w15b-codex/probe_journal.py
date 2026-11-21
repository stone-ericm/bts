"""Replay an in-season EOD run and scheduler restart with a captured unit journal."""
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import bts
import probe_health as health


DAY = date(2026, 7, 22)
DISPATCHED_AT = "2026-07-22T21:10:00-04:00"
STATE_DIR = health.PICKS / str(DAY)
STATUS = health.STATUS / f"{DAY}.oom-kill.json"
ANALYTICS = health.analytics_artifacts_missing
GENUINE = "Jul 22 23:30:00 synthetic systemd[2]: bts-scheduler.service: Failed with result 'oom-kill'."


def run(journal, wall, label, receipt_path):
    def captured_journal(command, **options):
        assert command[:4] == ["journalctl", "--user", "-u", "bts-scheduler.service"]
        assert command[command.index("--since") + 1] == "2026-07-22 21:10:00"
        assert options["timeout"] == 5
        return SimpleNamespace(returncode=0, stdout=journal)

    with health.captured(sources=(health.SOURCE, ANALYTICS.SOURCE)) as (sent, rejected, log):
        health.Clock.at = datetime.fromisoformat(wall)
        with patch.object(ANALYTICS.subprocess, "run", captured_journal):
            alerts = health.run(DAY, shadow=True, receipt=receipt_path)
    receipt = json.loads(receipt_path.read_bytes())
    criticals = [item.message for item in alerts if item.level == "CRITICAL"]
    print(label, "CRITICALS", criticals, "sent", list(map(len, sent)),
          "rejected", list(map(len, rejected)), "receipt", receipt["status"])
    return criticals, sent, rejected, log.getvalue()


def main(source):
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    prior = source == health.HERE / "r2/src"
    STATUS.parent.mkdir(parents=True)
    STATUS.write_bytes(b"{")
    STATE_DIR.mkdir(parents=True)
    (STATE_DIR / "decision.json").write_text(json.dumps(dict(
        schema_version=health.DECISION_SCHEMA, date=str(DAY), action="single",
        source="mdp", delivery_status="delivered", scoreable=True,
    )))
    (STATE_DIR / "scheduler_state.json").write_text(json.dumps(dict(
        date=str(DAY), pick_locked=True, analytics_jobs={
            "shadow": {"status": "dispatched", "dispatched_at": DISPATCHED_AT},
        },
    )))
    receipt = Path("health-receipt.json")
    first, sent, rejected, log = run("", "2026-07-22T23:40:00-04:00", "first-eod", receipt)
    assert not first and len(sent) == 1 and not rejected
    detail, = [line for line in log.splitlines() if "status unreadable at" in line]
    journal = "Jul 22 23:40:00 synthetic bts[1]: " + detail + "\n"
    second, sent, rejected, _ = run(journal, "2026-07-22T23:50:00-04:00", "eod-restart", receipt)
    assert len(second) == 1 and str(STATUS) in second[0]
    assert list(map(len, sent)) == ([958] if prior else [])
    assert list(map(len, rejected)) == ([] if prior else [1009])
    genuine, _, _, _ = run(GENUINE + "\n", "2026-07-22T23:50:00-04:00", "genuine-oom", Path("genuine.json"))
    assert len(genuine) == 1 and GENUINE in genuine[0]


if __name__ == "__main__":
    initial = Path.cwd()
    with TemporaryDirectory(dir=health.HERE, prefix="journal-") as work:
        os.chdir(work)
        try:
            main(Path(sys.argv[1]).resolve())
        finally:
            os.chdir(initial)
