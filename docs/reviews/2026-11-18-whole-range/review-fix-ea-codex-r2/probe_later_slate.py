"""Follow B8a from an earlier warning while an intervening entry is pending."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state, dm, orchestrator, picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick


def clock(day, time):
    return datetime.fromisoformat(f"{day}T{time}").replace(tzinfo=walk.ET)


def command(path, *args):
    code, body = walk._invoke(*args, "--picks-dir", path)
    assert code == 0, (code, body)
    print(body.strip())
    return body


def current(path):
    return contest_state.load_decision_streak_state(path, require_contest_state=True)


work = Path(__file__).resolve().parent
source = (work.parent.parent / "docs/ops/2027-season-start.md").read_text()
rule = ("For any date later than the pick's slate, including D when the warning was sent after "
        "midnight ET, wait until MLB has settled that slate's entry or confirms there was none.")
assert rule in source
print("B8a:", rule)

old = "2027-07-15"
intervening = "2027-07-16"
today = "2027-07-17"
sent_at = clock(old, "18:35")
with TemporaryDirectory(dir=work) as temp, walk.offline(sent_at):
    path = Path(temp) / "scheduler" / "picks"
    path.mkdir(parents=True)
    config = walk._config(path) | {"bluesky": {"dm_recipient": "capture.invalid"}}
    assert not walk.install_state(path, sent_at.date(), walk.State("automatic", 12, 14, "active", .99))
    walk._set_clock(sent_at)
    automatic = path / "account_state/contest_streak.json"
    automatic.write_text(json.dumps(json.loads(automatic.read_text()) |
                                    {"recorded_at": (sent_at - timedelta(hours=20)).isoformat()}))
    daily = daily_pick()
    daily.chosen_on = picks.chosen_on_record(current(path), required=True)
    state = scheduler.SchedulerState(old, "pinned", [], [], [], False, None, None, None)
    warnings = []
    with patch.object(alert, "send_dm", side_effect=lambda _, body: warnings.append(body) or "warning"), \
            patch.object(dm, "send_dm", return_value="pick"), \
            patch.object(scheduler, "_trigger_live_forward_capture_on_lock"), \
            patch.object(scheduler, "_delivery_game_statuses", return_value={}):
        assert scheduler._deliver_and_lock_pick(daily, config, path, state, old, "probe")
    assert "For a later slate date, see B8a." in warnings[0]
    print(warnings[0])

    # The account counter rose 12 -> 13 on July 15's settlement. July 16's
    # late/suspended entry is still pending in MLB's profile on July 17.
    walk._set_clock(clock(today, "09:30"))
    status = command(path, "season-contract", "status")
    assert "automatic path alone: accepted (streak 12" in status
    print("MLB snapshot on July 17: July 15 hit, July 16 entry pending, 13/14/active.")
    control = Path(temp) / "control" / "picks"
    shutil.copytree(path, control)
    command(path, "set-contest-streak", "--date", today, "--streak", 13,
            "--best-streak", 14, "--saver", "active")

    # MLB settles the intervening entry before the first check, and the
    # regular fetch succeeds. The manual row still wins for the full ET day.
    walk._set_clock(clock(today, "10:30"))
    for dest in (path, control):
        with walk._fetch_stubs(14, 14, clock(intervening, "19:00").date()):
            command(dest, "fetch-contest-streak", "--expected-username", walk.ACCOUNT)
    status = command(path, "season-contract", "status")
    assert "automatic path alone: accepted (streak 14" in status
    assert "streak=13; status=fresh; saver=active" in status
    assert (current(path).streak, current(path).observation_kind) == (13, "dated")
    assert (current(control).streak, current(control).observation_kind) == (14, "automatic")
    walk._set_clock(clock(today, "18:35"))
    game_status = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in range(70000, 70003)}
    for dest, expected in ((path, 13), (control, 14)):
        frame = walk._predictions(clock(today, "18:35").date(), .99)
        with patch.object(picks, "get_game_statuses_detailed", return_value=game_status), \
                patch.object(orchestrator, "run_cascade", return_value=(frame, "pinned")):
            _, selection, _ = orchestrator.run_and_pick(walk._config(dest), today)
        assert selection.streak == expected
        print(f"Real selection used {selection.streak}, action {selection.action}.")
