"""Synthetic replacements for selected review tests whose fixtures are absent."""
import json
from datetime import date

from harness import World, wall
from interactions import skip_review
from opening import D1


def test_skip_with_corrected_dated_read(tmp_path):
    result = skip_review(tmp_path)
    reports = [json.loads(e["text"]) for e in result["events"]
               if e["type"] == "cli" and e["args"][0] == "day-outcome"]
    assert reports[0]["state"] == "skip_state_review"
    assert "records streak 55 and policy saver availability available" in reports[0]["detail"]
    assert reports[1]["state"] == reports[0]["state"]
    assert reports[1]["streak_problem"].startswith("cannot be read")


def test_early_window_stays_open_until_the_real_cutoff():
    from bts.entry_receipt import has_original_slot
    day = date.fromisoformat(D1)
    assert not has_original_slot(day, wall(D1, "10:05"))
    assert has_original_slot(day, wall(D1, "10:06"))
    assert not has_original_slot(day, wall(D1, "06:10"))


def test_early_run_defers_for_original_slot(tmp_path):
    from harness import ACCOUNT, RECIPIENT
    world = World(tmp_path, D1, "06:25", health=False)
    with world.switched_on():
        world.slate(D1, ("10:20",))
        world.dated()
        world.run()
        world.set_day(D1, "09:45")
        check = world.invoke("check-pick-entered", "--cron", "--picks-dir", world.picks,
                             "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)
        assert "early check deferred" in check.output
