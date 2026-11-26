import datetime as dt
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import contest_state, scheduler, web
from bts.health import alert
from bts.saver_state import load_saver_state, maybe_auto_earn_saver

from probe_support import DAY, HERE, auto, command, end_of_day, flag, manual


with TemporaryDirectory(dir=HERE, prefix="base-") as directory:
    for case, prior, streak, fetch in (
        ("active", "active", 14, True), ("not_earned/fetch", "not_earned", 14, True),
        ("not_earned/reused", "not_earned", 14, False), ("not_earned/below10", "not_earned", 7, False),
    ):
        picks = HERE / directory / case / "picks"
        auto(picks, streak=streak)
        flag(picks, prior)
        manual(picks, streak=streak)
        day1 = contest_state.load_decision_streak_state(picks, now=DAY, require_contest_state=True)
        dm = Mock(return_value="sent")
        same = end_of_day(picks, DAY.date(), DAY, dm)
        rollover = end_of_day(picks, DAY.date(), DAY + dt.timedelta(hours=3), dm)
        print(case, "day 1", day1.observation_kind, day1.saver_available,
              load_saver_state(picks, season=DAY.year).state, "EOD", same, "after midnight", rollover)
        assert day1.observation_kind == "dated" and not day1.saver_available
        assert not same and not rollover and not dm.called
        if case == "active":
            status = command(picks, DAY, "season-contract", "status")
            labels = ("manual:", "saver:", "live decision now:", "automatic path alone:")
            print("status", [line for line in status.splitlines() if line.startswith(labels)])
            print("dashboard", web.saver_dashboard_context(picks, now=DAY))
        tomorrow = DAY + dt.timedelta(hours=12)
        if fetch:
            auto(picks, tomorrow, streak=streak)
            maybe_auto_earn_saver(picks, best_streak=15, season=tomorrow.year)
        day2 = contest_state.load_decision_streak_state(picks, now=tomorrow, require_contest_state=True)
        with (patch.object(alert, "send_dm", dm), patch.object(scheduler, "_print_morning_check") as morning):
            scheduler._morning_state_check({"bluesky": {"dm_recipient": "offline.test"}}, picks,
                                           tomorrow.date().isoformat(), tomorrow, tomorrow + dt.timedelta(hours=2))
        health2 = end_of_day(picks, tomorrow.date(), tomorrow + dt.timedelta(hours=3), dm)
        print(case, "day 2", day2.observation_kind, day2.saver_available, day2.saver_flag_state,
              "morning", morning.call_args.args, "EOD", health2)
        assert day2.observation_kind == "automatic"
        assert day2.saver_available is (case != "not_earned/reused")
        assert bool(health2) is (case == "not_earned/reused")
