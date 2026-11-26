"""Exercise the F4 report without its unavailable archived test fixtures."""
import json

from harness import World
from opening import D1, D2
from season import outcome


def check_report(root, bytes_value):
    world = World(root, D2, "08:30", health=False)
    with world.switched_on():
        target = world.picks / "streak.json"
        if bytes_value is None:
            target.symlink_to("absent.json")
        else:
            target.write_bytes(bytes_value)
        result = outcome(world, D1)
        return result.exit_code, json.loads(result.output), world.events


def test_boolean_model_streak(tmp_path):
    code, report, events = check_report(tmp_path, b'{"streak": true}')
    assert code == 1
    assert report["state"] == "no_games"
    assert report["streak_problem"] == "streak is not a non-negative integer"
    assert any("MODEL STREAK FILE NEEDS REVIEW" in e.get("message", "") for e in events)


def test_bounded_decoder_failure(tmp_path):
    code, report, _ = check_report(tmp_path, b'{"streak":' + b"9" * 5000 + b"}")
    assert code == 1
    assert report["streak_problem"] == "JSON exceeds decoder limits (ValueError)"


def test_report_compacts_long_advisory():
    from bts.day_outcome import DayOutcome
    outcome = DayOutcome(date=D1, state="not_entered", detail="x" * 890,
                         streak_problem="not a JSON object")
    assert len(outcome.message()) < 1000
    assert "Find the day-outcome JSON in cron.log" in outcome.message()


def test_dangling_streak_is_absent(tmp_path):
    code, report, _ = check_report(tmp_path, None)
    assert code == 0
    assert report["streak_problem"] is None
