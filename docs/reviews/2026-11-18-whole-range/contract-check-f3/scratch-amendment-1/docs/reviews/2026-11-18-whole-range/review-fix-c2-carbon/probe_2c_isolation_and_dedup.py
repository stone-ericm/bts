"""Item 2 probe (v3): receipt isolation, slate-keyed dedup across ET midnight & New Year, cutoff & EOD."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import datetime, timedelta
from io import StringIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_SRC = HERE / "base" / "src"
ET = ZoneInfo("America/New_York")


def u7_copy(day: str) -> str:
    return (
        "BTS health CRITICAL alert(s):\n"
        f"- [late_delivery] DELIVERY OUTCOME UNKNOWN for {day} ET. A send attempt for that slate "
        "was never confirmed, and the scheduler locked the slate without resending. "
        f"Check the DM or public feed and the MLB entry for {day} ET. Do not enter or change a "
        f"pick based on this alert. If {day} is still today in ET and its submission window is "
        "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
        "directory before making any entry decision for that slate."
    )


def _base_cutoff_collision_worker() -> dict:
    from bts import contest_state, daily_decision, dm, picks, scheduler, strategy
    from bts.health import alert

    day = "2027-03-25"
    now = datetime(2027, 3, 25, 19, 10, 0, tzinfo=ET)

    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    results = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for order in (("unknown", "cutoff"), ("cutoff", "unknown")):
            sub = root / "_".join(order)
            unknown_dir = sub / "picks_unknown"
            cutoff_dir = sub / "picks_cutoff"
            unknown_dir.mkdir(parents=True)
            cutoff_dir.mkdir(parents=True)
            p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{day}T23:10:00Z")
            d_unk = picks.DailyPick(day, now.isoformat(), p1, None, None, delivery_attempted=True)
            d_cut = picks.DailyPick(day, now.isoformat(), p1, None, None)
            picks.save_pick(d_unk, unknown_dir)
            picks.save_pick(d_cut, cutoff_dir)
            sent = Mock(return_value="ack")
            with ExitStack() as stack:
                for p in (
                    patch.object(alert, "datetime", PinnedClock),
                    patch.object(daily_decision, "datetime", PinnedClock),
                    patch.object(scheduler, "_now_et", lambda: now),
                    patch.object(scheduler, "_void_committed_games", lambda *_: {}),
                    patch.object(scheduler, "_contest_state_required", lambda _: False),
                    patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=0)),
                    patch.object(dm, "send_dm", Mock()),
                    patch.object(alert, "send_dm", sent),
                    patch.object(strategy, "_mdp_cache", None),
                    redirect_stderr(StringIO()),
                ):
                    stack.enter_context(p)
                for step in order:
                    pdir = unknown_dir if step == "unknown" else cutoff_dir
                    dp = d_unk if step == "unknown" else d_cut
                    cfg = {
                        "orchestrator": {"picks_dir": str(pdir)},
                        "bluesky": {"dm_recipient": "offline.test"},
                        "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False},
                    }
                    st = scheduler.SchedulerState(day, now.isoformat(), [], [], [], False, None, None, None)
                    scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, day, step)
            results["_".join(order)] = [c.args[1] for c in sent.call_args_list]
    return results


def main() -> None:
    signal.alarm(45)
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
        from bts.health import alert, runner

        env = dict(os.environ, PYTHONPATH=str(BASE_SRC), PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
        proc = subprocess.run(
            [str(ROOT / ".venv/bin/python"), __file__, "--base-worker"],
            env=env, capture_output=True, text=True, timeout=20, check=True,
        )
        base_orders = json.loads(proc.stdout)
        assert len(base_orders["unknown_cutoff"]) == 1 and "OUTCOME UNKNOWN" in base_orders["unknown_cutoff"][0]
        assert len(base_orders["cutoff_unknown"]) == 1 and "LATE DELIVERY REFUSED" in base_orders["cutoff_unknown"][0]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            for d1, d2 in (("2027-03-25", "2027-03-26"), ("2027-12-31", "2028-01-01")):
                for first_after_midnight in (False, True):
                    case_dir = root / f"rollover_{d1}_{first_after_midnight}"
                    picks_dir = case_dir / "picks"
                    picks_dir.mkdir(parents=True)
                    shared_path = case_dir / "health_state" / "health_dm_delivery_status.json"
                    shared_path.parent.mkdir(parents=True)
                    shared_seed = b'{"sentinel": "untouched"}\n'
                    shared_path.write_bytes(shared_seed)

                    now_box = [datetime.fromisoformat(f"{d1}T18:30:00").replace(tzinfo=ET)]

                    class Clock(datetime):
                        @classmethod
                        def now(cls, tz=None):
                            return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

                    games = {}
                    for slate in (d1, d2):
                        p1 = picks.Pick("Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{slate}T23:10:00Z")
                        dp = picks.DailyPick(slate, now_box[0].isoformat(), p1, None, None, delivery_attempted=True)
                        picks.save_pick(dp, picks_dir)
                        games[slate] = [{
                            "gamePk": 101, "gameDate": f"{slate}T23:10:00Z",
                            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                            "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "H"}}},
                        }]
                    cfg = {
                        "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(case_dir / "hb")},
                        "bluesky": {"dm_recipient": "offline.test"},
                        "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                                      "entry_intent": "enter", "live_forward_capture_on_lock": False},
                        "health_checks": {"enabled": False, "contest_state_expected": True},
                        "tiers": [],
                    }
                    sent = Mock(return_value="ack-roll")
                    with ExitStack() as stack:
                        for p in (
                            patch.object(contest_state, "datetime", Clock),
                            patch.object(daily_decision, "datetime", Clock),
                            patch.object(alert, "datetime", Clock),
                            patch.object(cli, "datetime", Clock),
                            patch.object(scheduler, "_now_et", lambda nb=now_box: nb[0]),
                            patch.object(scheduler, "fetch_schedule", side_effect=lambda d, g=games: g.get(d, [])),
                            patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}),
                            patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None),
                            patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None),
                            patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()),
                            patch.object(scheduler, "notify_ready", Mock()),
                            patch.object(scheduler, "notify_watchdog", Mock()),
                            patch.object(scheduler, "run_result_polling", Mock(return_value="final")),
                            patch.object(dm, "send_dm", Mock()),
                            patch.object(alert, "send_dm", sent),
                            patch.object(posting, "post_to_bluesky", Mock()),
                            patch.object(strategy, "_mdp_cache", None),
                            redirect_stderr(StringIO()),
                        ):
                            stack.enter_context(p)
                        own_d1 = case_dir / "health_state/delivery_outcome_unknown" / f"{d1}.json"
                        own_d2 = case_dir / "health_state/delivery_outcome_unknown" / f"{d2}.json"
                        if not first_after_midnight:
                            scheduler.run_day(d1, cfg)
                            assert sent.call_count == 1 and sent.call_args.args[1] == u7_copy(d1)
                            d1_bytes = own_d1.read_bytes()
                            assert json.loads(d1_bytes)["sent_day"] == d1
                            assert not own_d2.exists()
                            now_box[0] = datetime.fromisoformat(f"{d2}T00:01:00").replace(tzinfo=ET)
                            scheduler.run_day(d1, cfg)
                            assert sent.call_count == 1
                            assert own_d1.read_bytes() == d1_bytes
                            assert not own_d2.exists()
                        else:
                            now_box[0] = datetime.fromisoformat(f"{d2}T00:05:00").replace(tzinfo=ET)
                            scheduler.run_day(d1, cfg)
                            assert sent.call_count == 1 and sent.call_args.args[1] == u7_copy(d1)
                            assert own_d1.is_file() and not own_d2.exists()
                            d1_bytes = own_d1.read_bytes()
                            assert json.loads(d1_bytes)["sent_day"] == d1
                            now_box[0] = datetime.fromisoformat(f"{d2}T00:10:00").replace(tzinfo=ET)
                            scheduler.run_day(d1, cfg)
                            assert sent.call_count == 1 and own_d1.read_bytes() == d1_bytes

                        # Move to d2 18:30 so run_day(d2) reaches its 18:25 lineup check immediately
                        now_box[0] = datetime.fromisoformat(f"{d2}T18:30:00").replace(tzinfo=ET)
                        toml_path = case_dir / "cfg.toml"
                        toml_path.write_text("")
                        spy_run_day = Mock(wraps=scheduler.run_day)
                        with patch.object(orchestrator, "load_config", lambda _, c=cfg: c), \
                                patch.object(scheduler, "run_day", spy_run_day):
                            r_undated = CliRunner().invoke(cli.cli, ["schedule", "--config", str(toml_path)])
                            r_dated = CliRunner().invoke(cli.cli, ["schedule", "--config", str(toml_path), "--date", d1])
                        assert r_undated.exit_code == 0 and r_dated.exit_code == 0
                        assert [c.kwargs["date"] for c in spy_run_day.call_args_list] == [d2, d1]
                        assert [c.args[1] for c in sent.call_args_list] == [u7_copy(d1), u7_copy(d2)]
                        assert json.loads(own_d2.read_bytes())["sent_day"] == d2
                        assert shared_path.read_bytes() == shared_seed

            day = "2027-03-25"
            now_cutoff = datetime(2027, 3, 25, 19, 10, 0, tzinfo=ET)

            class CutoffClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return now_cutoff.astimezone(tz) if tz else now_cutoff.replace(tzinfo=None)

            for order in (("unknown", "cutoff"), ("cutoff", "unknown")):
                sub = root / f"head_{'_'.join(order)}"
                unknown_dir = sub / "picks_unknown"
                cutoff_dir = sub / "picks_cutoff"
                unknown_dir.mkdir(parents=True)
                cutoff_dir.mkdir(parents=True)
                shared_file = sub / "health_state" / "health_dm_delivery_status.json"
                own_file = sub / "health_state" / "delivery_outcome_unknown" / f"{day}.json"

                p1 = picks.Pick("Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{day}T23:10:00Z")
                d_unk = picks.DailyPick(day, now_cutoff.isoformat(), p1, None, None, delivery_attempted=True)
                d_cut = picks.DailyPick(day, now_cutoff.isoformat(), p1, None, None)
                picks.save_pick(d_unk, unknown_dir)
                picks.save_pick(d_cut, cutoff_dir)
                sent = Mock(return_value="ack-head")

                with ExitStack() as stack:
                    for p in (
                        patch.object(alert, "datetime", CutoffClock),
                        patch.object(daily_decision, "datetime", CutoffClock),
                        patch.object(scheduler, "_now_et", lambda: now_cutoff),
                        patch.object(scheduler, "_void_committed_games", lambda *_: {}),
                        patch.object(scheduler, "_contest_state_required", lambda _: False),
                        patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=0)),
                        patch.object(dm, "send_dm", Mock()),
                        patch.object(alert, "send_dm", sent),
                        patch.object(strategy, "_mdp_cache", None),
                        redirect_stderr(StringIO()),
                    ):
                        stack.enter_context(p)
                    for step in order:
                        pdir = unknown_dir if step == "unknown" else cutoff_dir
                        dp = d_unk if step == "unknown" else d_cut
                        cfg = {
                            "orchestrator": {"picks_dir": str(pdir)},
                            "bluesky": {"dm_recipient": "offline.test"},
                            "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False},
                        }
                        if step == "unknown":
                            before_shared = shared_file.read_bytes() if shared_file.exists() else None
                        st = scheduler.SchedulerState(day, now_cutoff.isoformat(), [], [], [], False, None, None, None)
                        assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, day, step) is False
                        if step == "unknown":
                            after_shared = shared_file.read_bytes() if shared_file.exists() else None
                            assert before_shared == after_shared

                    bodies = [c.args[1] for c in sent.call_args_list]
                    assert len(bodies) == 2
                    assert sum(b == u7_copy(day) for b in bodies) == 1
                    assert sum("LATE DELIVERY REFUSED" in b for b in bodies) == 1
                    own_snap = own_file.read_bytes()

                    unrelated = [alert.Alert("CRITICAL", "pick_delivery", "Unrelated outage")]
                    sent.side_effect = [OSError("bsky down"), "eod-prev", "eod-next"]
                    assert alert.dispatch_dm_for_health_alerts(
                        unrelated, "offline.test", status_path=shared_file, now_et_date=now_cutoff.date()
                    ) is True
                    failed_shared_bytes = shared_file.read_bytes()
                    assert json.loads(failed_shared_bytes)["status"] == "failed"
                    assert own_file.read_bytes() == own_snap

                    cfg_u = {
                        "orchestrator": {"picks_dir": str(unknown_dir)},
                        "bluesky": {"dm_recipient": "offline.test"},
                        "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False},
                    }
                    st2 = scheduler.SchedulerState(day, now_cutoff.isoformat(), [], [], [], False, None, None, None)
                    assert scheduler._deliver_and_lock_pick(d_unk, cfg_u, unknown_dir, st2, day, "re") is False
                    assert shared_file.read_bytes() == failed_shared_bytes
                    assert own_file.read_bytes() == own_snap
                    assert sent.call_count == 3

                    with patch.object(runner, "_safe_run", lambda n, _: unrelated if n == "post_failure" else []), \
                            patch.object(runner.realized_calibration, "_current_deploy_iso", lambda _: None):
                        runner.run_all_checks(unknown_dir, sub / "models", "offline.test",
                                              today=now_cutoff.date() - timedelta(days=1))
                        runner.run_all_checks(unknown_dir, sub / "models", "offline.test",
                                              today=now_cutoff.date() + timedelta(days=1))
                    assert sent.call_count == 5
                    assert "Previous health alert DM FAILED" in sent.call_args_list[3].args[1]
                    assert own_file.read_bytes() == own_snap
                    st3 = scheduler.SchedulerState(day, now_cutoff.isoformat(), [], [], [], False, None, None, None)
                    assert scheduler._deliver_and_lock_pick(d_unk, cfg_u, unknown_dir, st3, day, "re2") is False
                    assert sent.call_count == 5

        print("PROBE 2C OK: midnight/New Year (pre & post rollover), undated vs dated CLI, both cutoff orders, failed health & EOD isolation verified.")
    finally:
        if str(ROOT / "src") in sys.path:
            sys.path.remove(str(ROOT / "src"))
        signal.alarm(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--base-worker":
        print(json.dumps(_base_cutoff_collision_worker()))
    else:
        main()
