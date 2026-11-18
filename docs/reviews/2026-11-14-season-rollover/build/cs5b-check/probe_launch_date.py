import io
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from probe_run_day_v3 import Interrupted, MonkeyPatch, et, setup
from click.testing import CliRunner
from bts import cli, dm, orchestrator, picks, scheduler
from tests.test_pick_chosen_on_cs5a import DAY, set_dated


def exercise(root, mp):
    real_poll = scheduler.run_result_polling
    config, directory, clock, _, frame, tier = setup(root, mp)
    set_dated(directory, streak=30, best=30)
    orchestrator.run_cascade.return_value = frame.assign(p_game_hit=[0.99, 0.81]), tier
    config_path = root / "config.toml"
    config_path.touch()
    tomorrow = "2027-07-16"
    fetch_original = scheduler.fetch_schedule.side_effect
    next_games = fetch_original(DAY)[0].copy()
    next_games["gameDate"] = f"{tomorrow}T17:10:00-04:00"
    scheduler.fetch_schedule.side_effect = (
        lambda day: [next_games] if day == tomorrow else fetch_original(day)
    )

    class LaunchClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz else clock[0].replace(tzinfo=None)

    mp.setattr(cli, "datetime", LaunchClock)
    mp.setattr(scheduler, "run_result_polling", Mock(wraps=real_poll))
    mp.setattr(scheduler, "_check_hits_midgame", Mock(return_value=[False]))
    mp.setattr(scheduler, "poll_game_result", Mock(return_value="live"))
    mp.setattr(scheduler, "_poll_interval_sleep", Mock(side_effect=Interrupted))
    mp.setattr(scheduler, "_watchdog_ping_sleep", lambda seconds: clock.__setitem__(
        0, clock[0] + timedelta(seconds=seconds)))

    def launch(label, instant, *, load_ends=None, polling="interrupt", mark_final=False):
        clock[0] = datetime.fromisoformat(instant)
        scheduler._poll_interval_sleep.side_effect = (
            (lambda _, seconds: clock.__setitem__(0, clock[0] + timedelta(seconds=seconds)))
            if polling == "cap" else Interrupted
        )
        scheduler._poll_interval_sleep.reset_mock()
        scheduler.run_result_polling.reset_mock()
        scheduler.fetch_schedule.reset_mock()
        scheduler._idle_until_next_wakeup.reset_mock()
        dm.send_dm.reset_mock()
        orchestrator.run_cascade.reset_mock()
        if mark_final:
            daily = picks.load_pick(DAY, directory)
            daily.result = "hit"
            picks.save_pick(daily, directory)

        def config_loader(_):
            if load_ends is not None:
                clock[0] = datetime.fromisoformat(load_ends)
            return config

        orchestrator.load_config.side_effect = config_loader
        outcome = "returned"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                result = CliRunner().invoke(cli.cli, ["schedule", "--config", str(config_path)],
                                           catch_exceptions=False)
                assert result.exit_code == 0, result.output
            except Interrupted:
                outcome = "interrupted in poll"
        today = scheduler.fetch_schedule.call_args_list[0].args[0]
        state = scheduler.load_state(today, directory)
        polled_dates = [x.args[1] for x in scheduler.run_result_polling.call_args_list]
        prior_pick = picks.load_pick(DAY, directory)
        print(label, "launch", today, "fetches",
              [x.args[0] for x in scheduler.fetch_schedule.call_args_list],
              "poll", polled_dates, "outcome", outcome, "pick DMs", dm.send_dm.call_count,
              "cascades", orchestrator.run_cascade.call_count,
              "prior result", prior_pick.result, "next wakeup", state.next_wakeup)
        return today, polled_dates

    mp.setattr(orchestrator, "load_config", Mock())
    assert launch("first launch", et("18:31").isoformat()) == (DAY, [DAY])
    assert launch("same-day late evening polling", f"{DAY}T23:40:00-04:00") == (DAY, [DAY])
    assert launch("config load straddled midnight", f"{DAY}T23:59:59-04:00",
                  load_ends=f"{tomorrow}T00:00:01-04:00") == (DAY, [DAY])
    assert launch("previous day's poll at 00:35", f"{tomorrow}T00:35:00-04:00") == (tomorrow, [])
    assert launch("previous day's poll at 04:50", f"{tomorrow}T04:50:00-04:00") == (tomorrow, [])
    assert launch("old process continued through polling cap", f"{DAY}T23:59:59-04:00",
                  load_ends=f"{tomorrow}T04:50:00-04:00", polling="cap") == (DAY, [DAY])
    assert launch("same-day idle", f"{DAY}T23:40:00-04:00", mark_final=True) == (DAY, [])
    assert launch("idle after midnight", f"{tomorrow}T00:35:00-04:00") == (tomorrow, [])


if __name__ == "__main__":
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, MonkeyPatch.context() as mp:
        exercise(Path(tmp), mp)
