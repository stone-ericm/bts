"""Follow the same-slate stale delivery instruction with an earlier entry pending."""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state, dm, picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick


def command(path, *args):
    code, output = walk._invoke(*args, "--picks-dir", path)
    assert code == 0, output
    print(output.strip())
    return output


work = Path(__file__).resolve().parent
day = "2027-07-15"
now = datetime.fromisoformat(day + "T11:40:00-04:00")
with TemporaryDirectory(dir=work) as temp, walk.offline(now):
    path = Path(temp) / "picks"
    path.mkdir()
    earlier = (now - timedelta(days=2)).date()
    assert not walk.install_state(path, earlier, walk.State("automatic", 12, 14, "active", .99))
    walk._set_clock(now)
    config = walk._config(path) | {"bluesky": {"dm_recipient": "capture.invalid"}}
    reading = contest_state.load_decision_streak_state(path, require_contest_state=True)
    daily = daily_pick()
    daily.pick.game_time = (now + timedelta(hours=1, minutes=25)).isoformat()
    daily.chosen_on = picks.chosen_on_record(reading, required=True)
    state = scheduler.SchedulerState(day, "pinned", [], [], [], False, None, None, None)
    captured = []
    with patch.object(alert, "send_dm", side_effect=lambda _, body: captured.append(body) or "warn"), \
            patch.object(dm, "send_dm", return_value="pick"), \
            patch.object(scheduler, "_trigger_live_forward_capture_on_lock"), \
            patch.object(scheduler, "_delivery_game_statuses", return_value={}):
        assert scheduler._deliver_and_lock_pick(daily, config, path, state, day, "same-slate delivery")
    assert f"the pick's slate and today's ET date both equal {day}." in captured[0]
    print(captured[0])
    status = command(path, "season-contract", "status")
    assert "automatic path alone: accepted (streak 12" in status
    print("MLB displays 13/14/active. July 14 entry remains pending.")
    command(path, "set-contest-streak", "--date", day, "--streak", 13,
            "--best-streak", 14, "--saver", "active")
    walk._set_clock(now + timedelta(minutes=30))
    with walk._fetch_stubs(14, 14, (now - timedelta(days=1)).date()):
        command(path, "fetch-contest-streak", "--expected-username", walk.ACCOUNT)
    status = command(path, "season-contract", "status")
    assert "automatic path alone: accepted (streak 14" in status
    assert "streak=13; status=fresh; saver=active" in status
    on_disk = picks.load_pick(day, path)
    assert on_disk.notification_id == "pick" and on_disk.chosen_on["streak"] == 12
    print("July 14 settles: accepted automatic streak 14, today's dated streak stays 13.")
