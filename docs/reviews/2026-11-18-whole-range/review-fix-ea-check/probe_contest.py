"""Replay B8a's settlement, deadline and directory instructions offline."""

from contextlib import nullcontext
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state, dm, orchestrator, picks, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick


WORK = Path(__file__).resolve().parent


def at(iso):
    return datetime.fromisoformat(iso).replace(tzinfo=walk.ET)


def command(path, *args, expected=0):
    code, body = walk._invoke(*args, "--picks-dir", path)
    assert code == expected, (args, code, body)
    print("CLI", code, str(args), body.strip(), sep="\n")
    return body


def observation(path, day, streak, best=14, saver="active", expected=0):
    return command(path, "set-contest-streak", "--date", day, "--streak", streak,
                   "--best-streak", best, "--saver", saver, expected=expected)


def fetch(path, streak, best, settled_day):
    with walk._fetch_stubs(streak, best, settled_day):
        command(path, "fetch-contest-streak", "--expected-username", walk.ACCOUNT)


def state(path):
    return contest_state.load_decision_streak_state(path, require_contest_state=True)


def config(path):
    return walk._config(path) | {"bluesky": {"dm_recipient": "capture.invalid"}}


def day_state(day):
    return scheduler.SchedulerState(day, "pinned", [], [], [], False, None, None, None)


def stale(path, now, streak=12, best=14, saver="active"):
    path.mkdir(parents=True)
    assert not walk.install_state(path, now.date(), walk.State("automatic", streak, best, saver, .99))
    walk._set_clock(now)
    path = path / "account_state" / "contest_streak.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data | {"recorded_at": (now - timedelta(hours=20)).isoformat()}))


def send_stale(path, before, after=None, at_cutoff=False):
    walk._set_clock(before)
    day = before.date().isoformat()
    candidate = daily_pick()
    candidate.date = day
    candidate.pick.game_time = (before + timedelta(minutes=5 if at_cutoff else 90)).isoformat()
    candidate.chosen_on = picks.chosen_on_record(state(path), required=True)
    warnings = []

    def send(*args):
        if after is not None:
            walk._set_clock(after)
        return "sent-pick"

    with patch.object(alert, "send_dm", side_effect=lambda _, body: warnings.append(body)), \
            patch.object(dm, "send_dm", side_effect=send) as pick_transport, \
            patch.object(scheduler, "_trigger_live_forward_capture_on_lock"), \
            patch.object(scheduler, "_delivery_game_statuses", return_value={}):
        delivered = scheduler._deliver_and_lock_pick(candidate, config(path), path, day_state(day), day, "review")
    if at_cutoff:
        assert not delivered and not pick_transport.called
        assert not any("WARN attention:" in body for body in warnings)
        print("B8a opening control: expired cutoff refuses with stale automatic state")
        return
    stamp = after or before
    date = stamp.date().isoformat()
    assert delivered and len(warnings) == 1
    assert pick_transport.call_count == 1
    assert (f"Write for the {date} slate only on that ET date, with prior entries settled or absent. "
            "For a later slate date, see B8a.") in warnings[0]
    assert f"In {stamp.year} ET away from midnight," in warnings[0]
    receipt = path.parent / "health_state/contest_stale_reading_dm_delivery_status.json"
    assert json.loads(receipt.read_text())["sent_day"] == date
    assert picks.load_pick(day, path).notification_id == "sent-pick"
    print("CAPTURED", warnings[0], sep="\n")
    return warnings[0]


def lineup(path, now, frame, phase):
    walk._set_clock(now)
    day = now.date().isoformat()
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in range(70000, 70003)}
    delivered = []
    with patch.object(picks, "get_game_statuses_detailed", return_value=statuses), \
            patch.object(orchestrator, "run_cascade", return_value=(frame, "pinned")), \
            patch.object(scheduler, "count_new_confirmations", return_value=0 if frame.empty else 3), \
            patch.object(scheduler, "heartbeat_watchdog", side_effect=lambda *a, **k: nullcontext()), \
            patch.object(scheduler, "_trigger_live_forward_capture_on_lock"), \
            patch.object(scheduler, "_delivery_game_statuses", return_value={}), \
            patch.object(alert, "send_dm", return_value="warning"), \
            patch.object(dm, "send_dm", side_effect=lambda _, body: delivered.append(body) or "later-pick"):
        result = scheduler.run_single_check(day, list(statuses), set(), config(path), .03)
        if result["should_post"]:
            assert scheduler._deliver_and_lock_pick(result["pick_result"].daily, config(path), path,
                                                    day_state(day), day, phase, selection=result["selection"])
    action = result["selection"].action if result["selection"] is not None else None
    print(phase, now.isoformat(), "action:", action, "captured pick DMs:", delivered)
    return result, action, delivered


def settlements(root):
    now = at("2027-07-15T11:40")
    path = root / "same" / "picks"
    stale(path, now)
    send_stale(path, now)
    send_stale(path, now, at_cutoff=True)
    assert "automatic path alone: accepted (streak 12" in command(path, "season-contract", "status")
    print("C23 July 14 pending at 11:40, MLB 13/14/active: operator waits")
    assert state(path).observation_kind == "automatic"
    saved = (path / "2027-07-15.json").read_bytes()
    flag = (path / "account_state/saver_state.json").read_bytes()
    early = root / "obsolete-rule" / "picks"
    shutil.copytree(path, early)
    observation(early, "2027-07-15", 13)
    walk._set_clock(now + timedelta(minutes=30))
    for directory in (path, early):
        fetch(directory, 14, 14, at("2027-07-14T19:00").date())
    assert (state(path).streak, state(path).observation_kind) == (14, "automatic")
    assert (state(early).streak, state(early).observation_kind) == (13, "dated")
    assert "automatic path alone: accepted (streak 14" in command(path, "season-contract", "status")
    print("C23 settled, MLB matches status so no filing; the obsolete instruction would leave dated 13")
    # Verify the newly allowed same-slate write when MLB later differs from status.
    assert "in effect today" in observation(path, "2027-07-15", 14, 15)
    assert (state(path).streak, state(path).best_streak, state(path).observation_kind) == (14, 15, "dated")
    shown = command(path, "season-contract", "status")
    assert "streak=14 best=15 saver=active in effect=True" in shown
    assert (path / "2027-07-15.json").read_bytes() == saved
    assert (path / "account_state/saver_state.json").read_bytes() == flag

    path = root / "later" / "picks"
    sent = at("2027-07-15T18:35")
    stale(path, sent)
    send_stale(path, sent)
    walk._set_clock(at("2027-07-17T09:30"))
    assert "automatic path alone: accepted (streak 12" in command(path, "season-contract", "status")
    print("C22 July 15 hit, MLB 13/14/active, July 16 pending: operator waits")
    old = root / "old-later-rule" / "picks"
    shutil.copytree(path, old)
    observation(old, "2027-07-17", 13)
    walk._set_clock(at("2027-07-17T10:30"))
    for directory in (path, old):
        fetch(directory, 14, 14, at("2027-07-16T19:00").date())
    assert (state(path).streak, state(path).observation_kind) == (14, "automatic")
    assert (state(old).streak, state(old).observation_kind) == (13, "dated")
    assert "automatic path alone: accepted (streak 14" in command(path, "season-contract", "status")
    print("C22 settled and now matches; next decision receives 14 instead of obsolete row's 13")
    for directory, expected in ((path, 14), (old, 13)):
        now = at("2027-07-17T18:20")
        result, action, _ = lineup(directory, now, walk._predictions(now.date(), .99), "settled lineup")
        assert result["selection"].streak == expected and action == "single"


def later_check(root):
    path = root / "later-check" / "picks"
    sent = at("2027-07-14T23:59:55")
    stale(path, sent, streak=0, best=15, saver="used")
    send_stale(path, sent, after=at("2027-07-15T00:00:20"))
    today = "2027-07-15"
    control = root / "corrected" / "picks"
    shutil.copytree(path, control)
    operator_dir = root / "operator-home"
    operator_dir.mkdir()

    # Two games minutes apart give an early 16:05 check and a later 17:05 check.
    games = [{"gamePk": 70000, "gameDate": "2027-07-15T23:05:00Z"},
             {"gamePk": 70001, "gameDate": "2027-07-15T23:10:00Z"}]
    checks = scheduler.plan_run_times(games, now=at("2027-07-15T10:00"))
    assert [check["time_et"].strftime("%H:%M") for check in checks[:2]] == ["16:05", "17:05"]
    # With this warning read before the first check, B8a files TODAY at the absolute DIR.
    walk._set_clock(at("2027-07-15T15:50"))
    original = Path.cwd()
    try:
        os.chdir(operator_dir)
        wrong_date = observation(control, "2027-07-14", 15, 15, "used", expected=2)
        assert f"must be today ({today})" in wrong_date
        assert "automatic path alone: accepted (streak 0" in command(control, "season-contract", "status")
        assert "in effect today" in observation(control, today, 15, 15, "used")
    finally:
        os.chdir(original)
    assert state(control).streak == 15 and not (operator_dir / "data").exists()

    rows = walk._predictions(at("2027-07-15T16:05").date(), .80).iloc[:2].copy()
    rows["p_game_hit"] = [.80, .80]
    first, action, delivered = lineup(path, checks[0]["time_et"], rows.iloc[:0], "first check, no lineups")
    assert first["pick_result"] is None and action is None and not delivered
    walk._set_clock(at("2027-07-15T16:10"))
    assert picks.load_pick(today, path) is None and load_decision(today, path) is None
    assert "automatic path alone: accepted (streak 0" in command(path, "season-contract", "status")
    print("C25 at 16:10: all prior entries settled, MLB 15/15/used, no pick yet for Eric to verify; literal later branch files nothing")
    corrected_late = root / "corrected-between-checks" / "picks"
    shutil.copytree(path, corrected_late)
    assert "in effect today" in observation(corrected_late, today, 15, 15, "used")
    second, action, sent_bodies = lineup(path, checks[1]["time_et"], rows, "later check, literal B8a")
    assert second["selection"].streak == 0 and action == "double" and len(sent_bodies) == 1
    sent = load_decision(today, path)
    assert sent["action"] == "double" and sent["delivery_status"] == "delivered"
    for directory in (control, corrected_late):
        second, action, sent_bodies = lineup(directory, checks[1]["time_et"], rows, "later check, corrected")
        assert second["selection"].streak == 15 and action == "skip" and not sent_bodies


if __name__ == "__main__":
    with TemporaryDirectory(dir=WORK) as temp, walk.offline(at("2027-07-15T11:40")):
        root = Path(temp)
        settlements(root)
        later_check(root)
