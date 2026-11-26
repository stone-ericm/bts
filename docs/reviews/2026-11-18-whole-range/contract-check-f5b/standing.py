"""Check further receipt and marker cases against the original and scratch CLIs."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from replay_support import DAY, ET, RIGHT, World, compact, deadline, health, row, source
from bts import daily_decision
from tests.test_wr_f3_skip_state_review import evidence


def fifo_with_live(root, v):
    w = World(root, v)
    w.commit_at("10:05", at="09:40")
    w.cron("09:45")
    assert w.invoke("09:50", manual=True)["sent"]
    final = w.picks / f"{DAY}.json"
    final.unlink()
    os.mkfifo(final)
    with deadline():
        report = w.morning()
    assert report[0] == 1 and report[1]["state"] == "not_entered" and len(report[2]) == 1
    print("R10 with a genuine non-early absence", report)


def skip_and_streak(root, v):
    w = World(root, v)
    w.commit_at("10:05", at="09:40")
    w.cron("09:45", pending=[row()])
    daily_decision.write_decision(
        DAY, w.picks, action="skip", source="mdp", delivery_status="deferred",
        scoreable=False, streak=30, saver_available=False, now=datetime.fromisoformat(DAY + "T10:15:00").replace(tzinfo=ET))
    account = w.picks / "account_state"
    account.mkdir()
    (account / "contest_streak.manual.json").write_text(json.dumps(evidence("current", day=DAY)))
    (w.picks / "streak.json").write_bytes(b"[]")
    result = w.morning()
    assert result[0] == 1 and result[1]["state"] == "skip_state_review" and (
        result[1]["streak_problem"] == "not a JSON object"), result
    print("genuine early receipt plus F3 and F4", result)


def same_date_unresolved(root):
    w = World(root / "live-suppresses")
    w.commit_at("10:05", at="09:00")
    w.cron("09:15", dm="failed")
    assert health(w) == ["CRITICAL"]
    w.invoke("09:19", manual=True, pending=[row()], crosswalk={})
    assert w.read_marker()["status"] == "present_unverified" and not health(w)
    print("live crosswalk gap suppresses early CRITICAL")

    w = World(root / "interrupted-all-clear")
    w.commit_at("10:05", at="09:00")
    w.cron("09:15")
    w.capture([RIGHT], f"{DAY}T132000Z")
    interrupted = w.cron("09:30", pending=[row()], dm="interrupt")
    assert interrupted["attempts"] and not interrupted["sent"] and w.early_state()["status"] == "confirmed"
    assert not w.cron("09:45", pending=[row()])["calls"] and not health(w)
    print("all-clear interrupted after marker replacement", compact(interrupted))

    w = World(root / "new-early-fifo")
    w.commit_at("10:05", at="09:00")
    final = w.picks / f"{DAY}.json"
    final.unlink()
    os.mkfifo(final)
    with deadline():
        blocked = w.cron("09:15")
    assert blocked["exception"] == "ReadBlocked" and not blocked["calls"]
    print("new scheduled blocking read of the pick before early gate", compact(blocked))


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as name:
        root = Path(name)
        fifo_with_live(root / "fifo-live", version)
        skip_and_streak(root / "f3-f4", version)
        if version == "f5":
            same_date_unresolved(root)
