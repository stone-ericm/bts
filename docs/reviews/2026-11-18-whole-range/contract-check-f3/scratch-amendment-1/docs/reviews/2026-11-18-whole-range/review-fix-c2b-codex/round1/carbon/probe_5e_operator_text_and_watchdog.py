"""Item 5 probe (v5): U7 operator text across all worst-state rows, status inspection, and B2 watchdog agreement."""

from __future__ import annotations

from contextlib import redirect_stderr
from datetime import date
from io import StringIO
from pathlib import Path
import re
import signal
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEAD_SRC = ROOT / "src"
SLICE1_CODEX = ROOT / "docs/reviews/2026-11-18-whole-range/slice-1-codex"
ET = ZoneInfo("America/New_York")

EXPECTED_TEMPLATE = (
    "BTS health CRITICAL alert(s):\n"
    "- [late_delivery] DELIVERY OUTCOME UNKNOWN for {date} ET. A send attempt for that slate "
    "was never confirmed, and the scheduler locked the slate without resending. "
    "Check the DM or public feed and the MLB entry for {date} ET. Do not enter or change a "
    "pick based on this alert. If {date} is still today in ET and its submission window is "
    "open, run bts season-contract status with --picks-dir set to the scheduler's picks "
    "directory before making any entry decision for that slate."
)

STATE_CHANGING_PATTERNS = (
    "set-contest-streak",
    "season-contract enable",
    "season-contract revoke",
    "saver-state",
    "fetch-contest-streak",
    "bts run",
    "bts orchestrate",
    "bts reconcile",
    "Fix it now",
)


def assert_no_state_changing_command(text: str, label: str) -> None:
    for pat in STATE_CHANGING_PATTERNS:
        assert pat not in text, f"{label} unexpectedly contained state-changing pattern {pat!r}: {text}"


def main() -> None:
    signal.alarm(45)
    sys.path[:0] = [str(HEAD_SRC), str(SLICE1_CODEX)]
    try:
        import opening_edges as edges
        import opening_rehearsal as opening
        from bts import contest_fetch, dm, picks, scheduler, strategy
        from bts.health import alert

        opening.REPO = ROOT

        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            season = opening.Season(root / "corrected_rehearsal")
            seed = {
                "schema_version": "bts_contest_streak_auto_v1", "season": 2026,
                "active_streak": 17, "best_streak": 26, "saver_available": False,
                "source_date": opening.OLD_DAY, "recorded_at": "2026-10-21T14:00:00Z",
                "source": "manual_cli",
            }
            opening.write_json(season.account_dir / "contest_streak.json", seed)
            status_at_recovery = []
            orig_verify = edges.verify_unconfirmed_entry

            def inspect_and_verify(s):
                status_at_recovery.append(s.status())
                return orig_verify(s)

            with patch.object(edges, "verify_unconfirmed_entry", inspect_and_verify), \
                    patch.object(strategy, "_mdp_cache", None), \
                    redirect_stderr(StringIO()):
                with season.run_offline():
                    season.restore()
                    ev = edges.restart_at_send(season, "corrected_after_marker")

            ctrl = opening.Season(root / "corrected_control")
            opening.write_json(ctrl.account_dir / "contest_streak.json", seed)
            with patch.object(strategy, "_mdp_cache", None), redirect_stderr(StringIO()):
                with ctrl.run_offline():
                    ctrl.restore()
                    ctrl.clock.move(opening.at("2027-03-25", "09:30"))
                    ctrl.dated("2027-03-25", 46, saver="used")
                    ctrl.clock.move(opening.at("2027-03-25", "10:00"))
                    ctrl.start("2027-03-25")

            attempted_pick = ev["snapshot"]["pick"]
            fresh_pick = ctrl.snapshot("2027-03-25")["pick"]
            assert attempted_pick["chosen_on"]["streak"] == 1 and attempted_pick["double_down"] is not None
            assert fresh_pick["chosen_on"]["streak"] == 46 and fresh_pick["double_down"] is None
            assert attempted_pick["pick"]["batter_name"] == fresh_pick["pick"]["batter_name"] == "Batter 27250"

            c2_dms = [m["body"] for m in season.messages if "DELIVERY OUTCOME UNKNOWN" in m["body"]]
            watchdog_dms = [m["body"] for m in season.messages if "Delivery of this pick was NOT confirmed" in m["body"]]
            assert c2_dms == [EXPECTED_TEMPLATE.format(date="2027-03-25")]
            assert len(watchdog_dms) == 1
            assert "Batter 27250" not in c2_dms[0] and "Batter 27251" not in c2_dms[0]
            assert_no_state_changing_command(c2_dms[0], "C2 U7 DM")
            assert_no_state_changing_command(watchdog_dms[0], "B2 watchdog DM")
            assert "Check your DMs and MLB entry before entering anything" in watchdog_dms[0]
            assert "bts season-contract status --picks-dir" in watchdog_dms[0]
            assert len(status_at_recovery) == 1
            assert "manual: dated date=2027-03-25 streak=46 best=46 saver=used in effect=True" in status_at_recovery[0]
            assert "live decision now: accepted; source=contest; streak=46; status=fresh; saver=used" in status_at_recovery[0]
            assert_no_state_changing_command(status_at_recovery[0], "status output on accepted dated state")

            # For `corrected` use a 16:05 game so 15:26 is 39m before pitch (inside 60m watchdog window too!)
            u7_rows = [
                ("never_left", "2027-03-25", "19:00", opening.at("2027-03-25", "18:20"), False, "dm"),
                ("corrected", "2027-03-25", "16:05", opening.at("2027-03-25", "15:26"), False, "dm"),
                ("accepted_and_entered", "2027-03-25", "19:00", opening.at("2027-03-25", "18:20"), True, "dm"),
                ("after_cutoff", "2027-03-25", "19:00", opening.at("2027-03-25", "18:56"), False, "dm"),
                ("after_first_pitch", "2027-03-25", "19:00", opening.at("2027-03-25", "19:04"), False, "dm"),
                ("after_midnight", "2027-03-25", "19:00", opening.at("2027-03-26", "00:01"), False, "dm"),
                ("new_year", "2027-12-31", "19:00", opening.at("2028-01-01", "00:01"), False, "dm"),
                ("switched_private", "2027-03-25", "19:00", opening.at("2027-03-25", "18:20"), False, "private"),
                ("private_fresh", "2027-03-25", "19:00", opening.at("2027-03-25", "18:20"), False, "private"),
            ]

            for route in ("direct", "classification"):
                for label, slate, pitch_hhmm, when, entered, mode in u7_rows:
                    case_dir = root / f"{route}_{label}"
                    pdir = case_dir / "picks"
                    pdir.mkdir(parents=True)
                    game_utc = opening.at(slate, pitch_hhmm).astimezone(opening.timezone.utc).isoformat()
                    p1 = picks.Pick("Example Batter", 27250, "EX", 1, "Opp", 2, 0.83, [], False, 27250, game_utc)
                    dp = picks.DailyPick(slate, opening.at(slate, "15:00").isoformat(), p1, None, None)
                    if label != "private_fresh":
                        dp.delivery_attempted = True
                    picks.save_pick(dp, pdir)

                    clock = opening.Clock(when)
                    cfg = {
                        "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(case_dir / "hb")},
                        "bluesky": {"dm_recipient": "operator.example"},
                        "scheduler": {
                            "pick_delivery": mode,
                            "entry_intent": "research" if mode == "private" else "enter",
                            "contest_state_required": mode != "private",
                            "live_forward_capture_on_lock": False,
                        },
                        "health_checks": {"enabled": False},
                        "tiers": [],
                    }
                    games = [{
                        "gamePk": 27250, "gameDate": game_utc,
                        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                        "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "H"}}},
                    }]
                    sent = Mock(side_effect=[None, RuntimeError("transient"), "ack-ok"])
                    pending_rows = [{"roundId": 1, "playerId": 27250, "result": None}] if entered else []

                    with opening.sandbox(clock, case_dir) as stack, redirect_stderr(StringIO()):
                        opening.auth_patches(
                            stack,
                            lambda: {"activeStreak": 1, "seasonBestStreak": 5, "predictions": []},
                            {1: date.fromisoformat(slate)},
                        )
                        stack.enter_context(patch.object(contest_fetch, "fetch_pending_predictions",
                                                         lambda *_, pr=pending_rows: pr))
                        stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", lambda: {27250: 27250}))
                        stack.enter_context(patch.object(scheduler, "_now_et", lambda c=clock: c.now))
                        stack.enter_context(patch.object(scheduler, "fetch_schedule",
                                                         side_effect=lambda d, s=slate, g=games: g if d == s else []))
                        stack.enter_context(patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}))
                        stack.enter_context(patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None))
                        stack.enter_context(patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None))
                        stack.enter_context(patch.object(scheduler, "notify_ready", Mock()))
                        stack.enter_context(patch.object(scheduler, "notify_watchdog", Mock()))
                        stack.enter_context(patch.object(scheduler, "run_result_polling", Mock(return_value="final")))
                        stack.enter_context(patch.object(scheduler, "_void_committed_games", lambda *_: {}))
                        stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="watchdog-dm")))
                        stack.enter_context(patch.object(alert, "send_dm", sent))
                        stack.enter_context(patch.object(strategy, "_mdp_cache", None))

                        if mode != "private" and clock.now.date().isoformat() == slate:
                            opening.invoke(
                                "set-contest-streak", "--date", slate, "--streak", "46" if label == "corrected" else "1",
                                "--best-streak", "46", "--saver", "used" if label == "corrected" else "not_earned",
                                "--picks-dir", str(pdir),
                            )

                        if label == "private_fresh":
                            if route == "direct":
                                st = scheduler.SchedulerState(
                                    slate, when.isoformat(), [], [], [], False, None, None, None
                                )
                                assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, slate, label) is True
                            else:
                                with patch("bts.orchestrator.run_cascade", Mock(return_value=(None, None))):
                                    scheduler.run_day(slate, cfg)
                            assert sent.call_count == 0
                            continue

                        for _ in range(4):
                            if route == "direct":
                                st = scheduler.SchedulerState(
                                    slate, when.isoformat(), [], [], [], False, None, None, None
                                )
                                assert scheduler._deliver_and_lock_pick(dp, cfg, pdir, st, slate, label) is False
                            else:
                                scheduler.run_day(slate, cfg)

                        expected = EXPECTED_TEMPLATE.format(date=slate)
                        assert [c.args[1] for c in sent.call_args_list] == [expected] * 3
                        assert "Example Batter" not in expected
                        assert_no_state_changing_command(expected, label)

                        still_today = (clock.now.date().isoformat() == slate)
                        window_open = (clock.now < picks.submission_cutoff_et(dp))
                        should_run_status = still_today and window_open
                        assert should_run_status is (label in {"never_left", "corrected", "accepted_and_entered", "switched_private"})

                        dm.send_dm.reset_mock()
                        chk = opening.invoke(
                            "check-pick-entered", "--picks-dir", str(pdir),
                            "--expected-username", opening.ACCOUNT, "--dm-recipient", "operator.example",
                            expected=None,
                        )
                        if not still_today:
                            assert chk.exit_code == 0 and f"no pick for {clock.now.date().isoformat()}" in chk.output
                            assert dm.send_dm.call_count == 0
                        elif not window_open:
                            assert chk.exit_code == 0 and "outside window" in chk.output
                            assert dm.send_dm.call_count == 0
                        elif entered:
                            assert chk.exit_code == 0 and "pick entered (match)" in chk.output
                            assert dm.send_dm.call_count == 0
                        else:
                            assert chk.exit_code == 1, chk.output
                            assert dm.send_dm.call_count == 1
                            wd_body = dm.send_dm.call_args.args[1]
                            assert "Delivery of this pick was NOT confirmed" in wd_body
                            assert "Check your DMs and MLB entry before entering anything" in wd_body
                            assert_no_state_changing_command(wd_body, f"watchdog {label}")

                        if should_run_status and mode != "private":
                            st_res = opening.invoke("season-contract", "status", "--picks-dir", str(pdir))
                            assert "live decision now: accepted" in st_res.output
                            assert_no_state_changing_command(st_res.output, f"status {label}")
                        assert re.search(r"scheduler's picks directory", expected) is not None

        print("PROBE 5E OK: U7 exact text across retries, all 9 state rows x 2 routes, status output, and B2 watchdog agreement verified.")
    finally:
        for p in (str(HEAD_SRC), str(SLICE1_CODEX)):
            if p in sys.path:
                sys.path.remove(p)
        signal.alarm(0)


if __name__ == "__main__":
    main()
