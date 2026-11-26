"""Run selected literal checklist and diagnostic commands in temporary state."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import ACCOUNT, RECIPIENT, World
from opening import D1, D2


def dry_runs(root):
    world = World(root, D1, health=False)
    with world.switched_on():
        watchdog = world.invoke("check-pick-entered", "--dry-run", "--picks-dir", world.picks)
        assert watchdog.exit_code == 2 and "No such option" in watchdog.output
        scheduler = world.invoke("schedule", "--dry-run", "--config", world.config_path)
        assert scheduler.exit_code == 0
        heartbeat = world.root / "data/.heartbeat"
        assert heartbeat.is_file()
        world.event("dry_run_heartbeat", body=heartbeat.read_text())
    return {"case": "A4_A6", "events": world.events}


def private_with_watchdog(root):
    world = World(root, D1, health=False, live=False, required=True, intent="enter")
    with world.switched_on():
        world.slate(D1)
        world.dated()
        intent = world.invoke("entry-intent", "--config", world.config_path)
        assert intent.exit_code == 1
        world.at("18:00", lambda: world.invoke("check-pick-entered", "--cron", "--picks-dir",
                 world.picks, "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT))
        schedule = world.invoke("schedule", "--config", world.config_path)
        assert schedule.exit_code == 0
        assert any("Fix it now!" in e.get("message", "") for e in world.events)
    return {"case": "A7_private_with_installed_watchdog", "events": world.events}


def dated_and_saver(root):
    world = World(root, D1, "23:50", health=False)
    with world.switched_on():
        world.dated()
        old = world.picks / "account_state/contest_streak.manual.json"
        old_bytes = old.read_bytes()
        world.set_day(D2, "00:20")
        repair = world.invoke("set-contest-streak", "--date", D1, "--streak", 1,
                              "--best-streak", 1, "--saver", "not_earned",
                              "--picks-dir", world.picks)
        assert repair.exit_code == 2 and "must be today" in repair.output
        assert old.read_bytes() == old_bytes
        for args in (("--init", "not_earned"), ("--use",), ("--undo",),
                     ("--init", "not_earned", "--force"), ("--show",)):
            result = world.invoke("saver-state", "--picks-dir", world.picks, *args)
            assert result.exit_code == 0, result.output
        assert "not_earned" in result.output
    return {"case": "B8a_dated_and_saver", "events": world.events}


if __name__ == "__main__":
    rows = []
    for probe in (dry_runs, private_with_watchdog, dated_and_saver):
        with TemporaryDirectory(prefix="wrblind-operator-") as temp:
            rows.append(probe(Path(temp)))
    Path(sys.argv[1]).resolve().write_text(json.dumps(rows, indent=2))
