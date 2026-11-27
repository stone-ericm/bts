import io
import os
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from probe_pick_pipe import DAY, PREFIX, TOMORROW, PipeJournal, Startup, daily_at, game, scheduler

from bts import cli, entry_intent, orchestrator, picks
from bts.daily_decision import load_decision
from bts.health import runner
from bts.health.alert import Alert
from bts.strategy import PickResult
from click.testing import CliRunner


class WatchdogExpired(BaseException):
    pass


def run_case(root, scenario, warning):
    start = Startup(root, mode="dm")
    start.config["health_checks"]["enabled"] = True
    start.config["scheduler"]["live_forward_capture_on_lock"] = False
    pick_dir = root / "picks"
    polling = scenario == "poll"
    interrupted = scenario.startswith("watchdog")
    fetch_fails = scenario == "watchdog-fetch"
    start_time = "18:59:59" if polling else "23:45:00"
    stall = 1800 if interrupted else 1200
    events, workers, notices = [], [], []
    written = Event()
    real_write = os.write
    real_boundary = scheduler._season_boundary_check
    real_lookahead = scheduler._next_day_wakeup
    real_idle = scheduler._idle_until_next_wakeup
    real_health = runner.run_all_checks
    pipe = None

    def mark(name, detail=None):
        events.append((name, start.now.isoformat(), detail))

    def on_drain():
        mark("blocked", pipe.blocked_write)
        start.now += timedelta(seconds=stall)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return start.now.astimezone(tz)

    class Journal(io.StringIO):
        def write(self, value):
            if pipe is not None:
                previous = pipe.blocked_write
                pipe.write(value)
                if interrupted and previous is None and pipe.blocked_write is not None:
                    raise WatchdogExpired
            return super().write(value)

    class JournalWorker(Thread):
        def start(self):
            workers.append(self)
            super().start()

    def write(fd, data):
        if fd == 2 and data.startswith(PREFIX):
            pipe.write(data.decode())
            notices.append(len(data))
            written.set()
            return len(data)
        return real_write(fd, data)

    def boundary(*args):
        nonlocal pipe
        # Both runs have the same space left just after the last pick work.
        pipe = PipeJournal(513, on_drain)
        mark("boundary")
        if warning:
            real_boundary(*args)

    def lookahead(*args):
        if warning:
            assert written.wait(1)
        mark("lookahead")
        return real_lookahead(*args)

    def schedule(day):
        if day == DAY:
            return [game(status={"abstractGameCode": "F", "detailedState": "Final"}) if not polling else game()]
        assert day == TOMORROW
        mark("fetch", day)
        if fetch_fails:
            raise OSError("fixture schedule outage")
        return [game(day=TOMORROW, start="13:15:00")]

    def check(**kwargs):
        return {
            "new_lineups": 0, "skipped": not polling,
            "pick_result": PickResult(daily=daily, locked=True) if polling else None,
            "should_post": False,
        }

    def poll(pk):
        mark("poll", pk)
        return "suspended"

    def sleep(seconds):
        start.now += timedelta(seconds=seconds)

    def idle(*args):
        mark("idle", args[0])
        real_idle(*args)
        mark("return")

    def health(**kwargs):
        mark("health", kwargs["today"].isoformat())
        return real_health(**kwargs)

    def fake_check(name, _fn):
        if name == "pick_entry":
            return [Alert("CRITICAL", "pick_entry", f"Check entry for {DAY}")]
        return []

    def health_send(recipient, body):
        mark("DM", (recipient, body))
        return "fixture-health-id"

    if polling:
        daily = daily_at("19:05:00")
        daily.notification_sent = True
        daily.notification_channel = "bluesky_dm"
        daily.notification_id = "fixture-pick-id"
        picks.save_pick(daily, pick_dir)
    else:
        previous = scheduler.SchedulerState(
            date=DAY, schedule_fetched_at=f"{DAY}T09:00:00-04:00", games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False, pick_locked_at=None,
            result_status=None, next_wakeup=None,
            final_skip_candidate={"primary": {"batter_id": 1, "batter_name": "Fixture Batter",
                                             "team": "NYM", "game_pk": 301, "p_game_hit": .51},
                                  "streak": 20, "source": "mdp"},
        )
        scheduler.save_state(previous, pick_dir)

    mocks = [
        patch.object(os, "write", write),
        patch.object(scheduler, "Thread", JournalWorker),
        patch.object(scheduler, "datetime", Clock),
        patch.object(scheduler, "fetch_schedule", schedule),
        patch.object(scheduler, "_season_boundary_check", boundary),
        patch.object(scheduler, "_next_day_wakeup", lookahead),
        patch.object(scheduler, "run_single_check", check),
        patch.object(scheduler, "poll_game_result", poll),
        patch.object(scheduler, "_watchdog_ping_sleep", sleep),
        patch.object(scheduler, "_idle_until_next_wakeup", idle),
        patch.object(picks, "get_game_statuses_detailed", return_value={}),
        patch.object(runner, "read_systemd_nrestarts", return_value=0),
        patch.object(runner, "get_self_pid", return_value=1),
        patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
        patch.object(runner, "_safe_run", fake_check),
        patch.object(runner, "run_all_checks", health),
    ]
    try:
        start.run([], at=start_time, journal=Journal(), extra=mocks, transport=health_send)
    except WatchdogExpired:
        assert interrupted and warning
        mark("watchdog")
        config_file = root / "fixture.toml"
        config_file.touch()
        with (
            patch.object(cli, "_today_et", lambda: start.now.date().isoformat()),
            patch.object(entry_intent, "live_gate_problem", return_value=None),
            patch.object(orchestrator, "load_config", return_value=start.config),
            patch.object(scheduler, "run_day", side_effect=lambda date, **_: mark("restart date", date)),
        ):
            result = CliRunner().invoke(cli.cli, ["schedule", "--config", str(config_file)])
            assert result.exit_code == 0, result.exception
    finally:
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
        if pipe is not None:
            pipe.finish()

    state = scheduler.load_state(DAY, pick_dir)
    decision = load_decision(DAY, pick_dir)
    mark("stored", (state.next_wakeup, state.result_status, decision["action"] if decision else None))
    assert notices == ([482] if warning else [])
    print(scenario, "warning", warning, *events, sep="\n  ", flush=True)
    return SimpleNamespace(events={name: (at, detail) for name, at, detail in events}, state=state, decision=decision)


def main():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        for scenario in ("poll", "skip", "watchdog", "watchdog-fetch"):
            control, with_line = [run_case(root / f"{scenario}-{flag}", scenario, flag) for flag in (False, True)]
            assert "blocked" not in control.events
            assert "Tomorrow's wake-up:" in with_line.events["blocked"][1] or (
                scenario == "watchdog-fetch" and "Failed to fetch/compute" in with_line.events["blocked"][1]
            )
            assert control.events["fetch"] == with_line.events["fetch"]
            if scenario == "poll":
                assert control.events["poll"][0].endswith("19:15:00-04:00")
                assert with_line.events["poll"][0].endswith("19:19:59-04:00")
                assert control.state.result_status == with_line.state.result_status == "suspended"
            if scenario == "skip":
                assert with_line.events["health"][0].startswith(TOMORROW)
                assert control.decision["action"] == with_line.decision["action"] == "skip"
            if scenario in ("poll", "skip"):
                assert control.events["DM"][1] == with_line.events["DM"][1]
                assert control.events["return"] == with_line.events["return"]
            else:
                assert control.decision["action"] == "skip" and "DM" in control.events
                assert with_line.decision is None and "health" not in with_line.events
                assert "idle" not in with_line.events and with_line.events["restart date"][1] == TOMORROW
            if scenario == "watchdog-fetch":
                assert control.state.next_wakeup and not with_line.state.next_wakeup
            else:
                assert control.state.next_wakeup == with_line.state.next_wakeup


if __name__ == "__main__":
    main()
