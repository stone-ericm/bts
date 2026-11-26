"""Verify what the morning report does when the final early-only cutoff changes."""
from pathlib import Path
from tempfile import TemporaryDirectory

from replay_support import DAY, RIGHT, World, row, source


source("f5")
with TemporaryDirectory() as temp:
    w = World(Path(temp))
    daily = w.commit_at("10:05", at="09:40")
    w.capture([RIGHT], f"{DAY}T134000Z")
    early = w.cron("09:45", elapsed=600, pending=[row()])["receipt"]
    assert early["observation"]["before_cutoff"] is True
    w.edit_pitch(daily, "09:55")
    result = w.morning()
    assert result[1]["state"] == "entered", result
    print("receipt at", early["observation"]["response_completed_at"], "final cutoff 09:50 ET", result)
