"""Replay date D's resolution after it has already been preserved in a dated copy."""
from datetime import date
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from marker_health import CURRENT, NOW, PREVIOUS, Replay, dated, health, row, source
from state_sequences import cron
from support import no_network
from bts.health import pick_entry, runner


def dispatch(world, day):
    messages = []
    safe = runner._safe_run
    check = pick_entry.check
    with no_network(), patch.object(runner, "_safe_run", lambda name, run: safe(name, run) if name == "pick_entry" else []), \
            patch.object(pick_entry, "check", lambda picks, today: check(picks, today, NOW)), \
            patch.object(runner.realized_calibration, "_current_deploy_iso", lambda _: None), \
            patch("bts.health.alert.send_dm", lambda recipient, message: messages.append(message)):
        alerts = runner.run_all_checks(world.picks, world.root / "models", today=date.fromisoformat(day),
                                       dm_recipient=world.recipient)
    return [item.level for item in alerts], messages


def stale_copy(root, version):
    old = Replay(root, version, PREVIOUS)
    old.commit_at("19:10", at="18:00")
    failed = cron(old, version, "18:00", dm="failed")
    assert failed["marker"]["status"] == "dm_failed"
    saved = old.marker.read_bytes()
    today = Replay(root, version, CURRENT)
    today.commit_at("06:10", at="04:55")
    cron(today, version, "05:00", pending=[row()])
    if version == "f5":
        assert dated(old).read_bytes() == saved
    # The operator replays the previous slate during recovery and verifies the entry in the historical window.
    confirmed = old.invoke("18:15", manual=True, pending=[row()])
    assert confirmed["marker"]["date"] == PREVIOUS and confirmed["marker"]["status"] == "confirmed"
    assert health(old) == []
    cron(today, version, "05:15", pending=[row()])
    levels, messages = dispatch(old, PREVIOUS)
    assert levels == (["CRITICAL"] if version == "f5" else []) and len(levels) == len(messages)
    print("after D confirmed during recovery:", version, levels, messages)


def decode_failure(root, version):
    world = Replay(root, version, CURRENT)
    world.set_marker({})
    world.marker.write_bytes(b'{"reason":"\xff"}')
    levels, messages = dispatch(world, CURRENT)
    assert levels == (["CRITICAL"] if version == "base" else []) and len(levels) == len(messages)
    if messages:
        assert "UnicodeDecodeError" in messages[0]
    print("live marker with invalid encoding:", version, levels, messages)


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        stale_copy(root / "stale", version)
        decode_failure(root / "encoding", version)
