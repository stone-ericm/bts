"""Items 3 & 5 probe: U7 copy parity, CLAUDE.md C2 ledger (incl. no-ID delivered vs watchdog), and extra edges."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import date, datetime
from io import StringIO
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

ROOT = Path("/Users/eric/projects/bts-work/wr-c2-r2-carbon")
REVIEW_DIR = ROOT / ".codex-review" / "wr-c2-r2"
R1_SCHEDULER = REVIEW_DIR / "r1_dfe774df" / "src" / "bts" / "scheduler.py"
HEAD_SCHEDULER = ROOT / "src" / "bts" / "scheduler.py"
ET = ZoneInfo("America/New_York")
DAY = "2027-03-25"
PINNED_NOW = datetime(2027, 3, 25, 18, 25, 0, tzinfo=ET)
PREFIX = "Unknown-outcome alert failed: "

CONTRACT_U7_ALERT_MESSAGE = (
    "DELIVERY OUTCOME UNKNOWN for {date} ET. A send attempt for that slate was never confirmed, "
    "and the scheduler locked the slate without resending. Check the DM or public feed and the MLB "
    "entry for {date} ET. Do not enter or change a pick based on this alert. If {date} is still "
    "today in ET and its submission window is open, run bts season-contract status with --picks-dir "
    "set to the scheduler's picks directory before making any entry decision for that slate."
)


def extract_helper(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index("def _alert_delivery_outcome_unknown(")
    end = text.index("\ndef _deliver_and_lock_pick(", start)
    return text[start:end]


def main() -> None:
    signal.alarm(45)
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from click.testing import CliRunner
        from bts import cli, contest_fetch, contest_state, daily_decision, dm, picks, posting, scheduler, strategy
        from bts.daily_decision import load_decision
        from bts.health import alert
        from bts.leaderboard import auth

        # 1. Item 3: confirm _alert_delivery_outcome_unknown (and U7 text) is byte-identical to Round 1
        r1_helper = extract_helper(R1_SCHEDULER)
        head_helper = extract_helper(HEAD_SCHEDULER)
        assert r1_helper == head_helper, "scheduler helper changed between dfe774df and e897c766!"

        # Check CLAUDE.md C2 entry content
        claude_text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        c2_lines = [ln for ln in claude_text.splitlines() if ln.strip().startswith("- **C2 (W22,")]
        assert len(c2_lines) == 1
        c2_line = c2_lines[0]
        for expected_fragment in (
            "health_state/delivery_outcome_unknown/<slate>.json",
            "requires a nonblank DM message ID",
            "The recommendation is never resent.",
            "**Ledgered beyond C2:** an end-of-day second chance for an unacknowledged alert.",
            "The entry watchdog, EOD health checks and 08:30 day-outcome remain independent backstops.",
            "An undated scheduler start after midnight selects the new slate.",
            "**Known and not fixed:** a recommendation send that returns no ID can leave a scoreable `delivered` decision with an unconfirmed pick.",
            "C2 says to avoid entering or changing a pick based on its alert, while the entry watchdog can still say \u201cFix it now!\u201d for the same slate.",
        ):
            assert expected_fragment in c2_line, f"Missing in CLAUDE.md: {expected_fragment!r}"

        # Check git diff file scope & no hardcoded season year/date in src/
        diff_names = subprocess.run(
            ["git", "diff", "--name-only", "386aa846", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
        assert set(diff_names) == {
            "CLAUDE.md",
            "src/bts/health/alert.py",
            "src/bts/scheduler.py",
            "tests/test_alert_status_callers_cs6b2.py",
            "tests/test_wr_c2_carbon_v2.py",
            "tests/test_wr_c2_recovery.py",
        }, diff_names

        src_diff = subprocess.run(
            ["git", "diff", "386aa846", "HEAD", "--", "src/"],
            cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        added_src_lines = [
            ln[1:] for ln in src_diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++") and not ln[1:].lstrip().startswith("#")
        ]
        for ln in added_src_lines:
            assert not re.search(r"\b202[6789]\b", ln), f"Hardcoded year in src/: {ln}"

        class PinnedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return PINNED_NOW.astimezone(tz) if tz else PINNED_NOW.replace(tzinfo=None)

        games = [{
            "gamePk": 101,
            "gameDate": f"{DAY}T23:10:00Z",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}},
        }]

        with TemporaryDirectory() as tmp:
            troot = Path(tmp)

            def enter_env(stack: ExitStack, health_dm: Mock, pick_dm: Mock, err_buf: StringIO):
                for p in (
                    patch.object(contest_state, "datetime", PinnedClock),
                    patch.object(daily_decision, "datetime", PinnedClock),
                    patch.object(alert, "datetime", PinnedClock),
                    patch.object(cli, "datetime", PinnedClock),
                    patch.object(scheduler, "_now_et", lambda: PINNED_NOW),
                    patch.object(scheduler, "fetch_schedule", side_effect=lambda d: games if d == DAY else []),
                    patch.object(scheduler, "check_confirmed_lineups", lambda *_: {}),
                    patch.object(scheduler, "_idle_until_next_wakeup", lambda *_: None),
                    patch.object(scheduler, "_watchdog_ping_sleep", lambda *_: None),
                    patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()),
                    patch.object(scheduler, "notify_ready", Mock()),
                    patch.object(scheduler, "notify_watchdog", Mock()),
                    patch.object(scheduler, "run_result_polling", Mock(return_value="final")),
                    patch.object(scheduler, "_void_committed_games", lambda *_: {}),
                    patch.object(scheduler, "_contest_state_required", lambda _: False),
                    patch.object(contest_state, "load_decision_streak_state", return_value=Mock(streak=3)),
                    patch.object(dm, "send_dm", pick_dm),
                    patch.object(alert, "send_dm", health_dm),
                    patch.object(posting, "post_to_bluesky", Mock(return_value="at://post")),
                    patch.object(strategy, "_mdp_cache", None),
                    redirect_stderr(err_buf),
                ):
                    stack.enter_context(p)

            # 2. Live verification of CLAUDE.md's "Known and not fixed" sentence (dm.send_dm -> None)
            n1_dir = troot / "n1_known_not_fixed" / "picks"
            n1_dir.mkdir(parents=True)
            p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.82, [], False, 101, f"{DAY}T23:10:00Z")
            dp_n1 = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p1, None, None)
            picks.save_pick(dp_n1, n1_dir)
            cfg_n1 = {
                "orchestrator": {"picks_dir": str(n1_dir), "heartbeat_path": str(n1_dir.parent / "hb")},
                "bluesky": {"dm_recipient": "offline.test"},
                "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False, "contest_state_required": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            h_dm = Mock(return_value="c2-ack")
            p_dm = Mock(return_value=None)
            with ExitStack() as stack:
                enter_env(stack, h_dm, p_dm, StringIO())
                st_init = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                # Initial pick delivery returns True, writes scoreable "delivered", but notification_id is None!
                assert scheduler._deliver_and_lock_pick(dp_n1, cfg_n1, n1_dir, st_init, DAY, "lineup") is True
                dec = load_decision(DAY, n1_dir)
                saved_pick = picks.load_pick(DAY, n1_dir)
                assert dec["scoreable"] is True and dec["delivery_status"] == "delivered"
                assert saved_pick.delivery_attempted is True and not picks.pick_was_delivered(saved_pick)
                assert h_dm.call_count == 0
                # Restart run_day -> C2 fires with "Do not enter or change a pick based on this alert."
                p_dm.reset_mock()
                scheduler.run_day(DAY, cfg_n1)
                assert p_dm.call_count == 0
                expected_full = f"BTS health CRITICAL alert(s):\n- [late_delivery] {CONTRACT_U7_ALERT_MESSAGE.format(date=DAY)}"
                h_dm.assert_called_once_with("offline.test", expected_full)
                # Entry watchdog on the same state -> generic branch says "Fix it now!"
                wd_dm = Mock(return_value="wd-ack")
                with patch.object(dm, "send_dm", wd_dm), \
                        patch.object(auth, "load_session_cookies", lambda: {}), \
                        patch.object(auth, "extract_uid", lambda _: "u"), \
                        patch.object(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x")), \
                        patch.object(contest_fetch, "fetch_profile", lambda *_: {"predictions": []}), \
                        patch.object(contest_fetch, "fetch_pending_predictions", lambda *_: []), \
                        patch.object(cli, "_fetch_rounds", lambda: {1: date.fromisoformat(DAY)}), \
                        patch.object(cli, "_fetch_bts_to_mlb", lambda: {680757: 680757}):
                    wd_res = CliRunner().invoke(cli.cli, [
                        "check-pick-entered", "--picks-dir", str(n1_dir),
                        "--now-et", PINNED_NOW.strftime("%Y-%m-%dT%H:%M:%S"),
                        "--expected-username", "offline", "--dm-recipient", "offline.test",
                    ])
                assert wd_res.exit_code == 1
                assert wd_dm.call_count == 1
                assert "Fix it now!" in wd_dm.call_args.args[1]

            # 3. Item 5 edges:
            # 3a. Cross-route retry & dedup across all 4 route pairs
            for first_route, second_route in (
                ("direct", "classification"),
                ("classification", "direct"),
                ("direct", "direct"),
                ("classification", "classification"),
            ):
                pair_dir = troot / f"pair_{first_route}_{second_route}" / "picks"
                pair_dir.mkdir(parents=True)
                dp_pair = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p1, None, None, delivery_attempted=True)
                picks.save_pick(dp_pair, pair_dir)
                cfg_pair = {
                    "orchestrator": {"picks_dir": str(pair_dir), "heartbeat_path": str(pair_dir.parent / "hb")},
                    "bluesky": {"dm_recipient": "offline.test"},
                    "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False, "contest_state_required": False},
                    "health_checks": {"enabled": False},
                    "tiers": [],
                }
                pair_dm = Mock(side_effect=[None, "pair-ack", "unexpected-third"])
                with ExitStack() as stack:
                    enter_env(stack, pair_dm, Mock(), StringIO())
                    for step_route in (first_route, second_route, "classification"):
                        if step_route == "direct":
                            st_p = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                            assert scheduler._deliver_and_lock_pick(dp_pair, cfg_pair, pair_dir, st_p, DAY, "p") is False
                        else:
                            assert scheduler.run_day(DAY, cfg_pair) is None
                    assert pair_dm.call_count == 2

            # 3b. Hostile config shapes on both routes
            for idx, bad_cfg_override in enumerate([
                {},
                {"bluesky": None},
                {"bluesky": "not-a-dict"},
                {"bluesky": {"dm_recipient": ""}},
            ]):
                for route in ("direct", "classification"):
                    hc_dir = troot / f"hostile_cfg_{idx}_{route}" / "picks"
                    hc_dir.mkdir(parents=True)
                    dp_hc = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p1, None, None, delivery_attempted=True)
                    picks.save_pick(dp_hc, hc_dir)
                    cfg_hc = {
                        "orchestrator": {"picks_dir": str(hc_dir), "heartbeat_path": str(hc_dir.parent / "hb")},
                        "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False, "contest_state_required": False},
                        "health_checks": {"enabled": False},
                        "tiers": [],
                        **bad_cfg_override,
                    }
                    err_hc = StringIO()
                    dm_hc = Mock(return_value="ack")
                    with ExitStack() as stack:
                        enter_env(stack, dm_hc, Mock(), err_hc)
                        if route == "direct":
                            st_hc = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                            assert scheduler._deliver_and_lock_pick(dp_hc, cfg_hc, hc_dir, st_hc, DAY, "hc") is False
                        else:
                            assert scheduler.run_day(DAY, cfg_hc) is None
                    assert dm_hc.call_count == 0
                    assert scheduler.load_state(DAY, hc_dir).pick_locked is True
                    assert load_decision(DAY, hc_dir)["delivery_status"] == "locked_unconfirmed"
                    f_lines = [ln for ln in err_hc.getvalue().splitlines() if ln.startswith(PREFIX)]
                    assert len(f_lines) == 1 and len(f_lines[0].removeprefix(PREFIX)) <= 200

            # 3c. Unicode line separators (\u2028, \u2029, \x85, \v, \f) & lone surrogates (\ud800)
            uni_err = StringIO()
            weird_exc = RuntimeError("line1\u2028line2\u2029line3\x85line4\vline5\fline6\ud800" + "z" * 300)
            with redirect_stderr(uni_err):
                alert.report_unknown_outcome_alert_failure("stage", weird_exc)
            uni_lines = uni_err.getvalue().splitlines()
            assert len(uni_lines) == 1 and len(uni_lines[0].removeprefix(PREFIX)) == 200

            surr_receipt = troot / "surrogate_id.json"
            surr_dm = Mock(return_value="id-\ud800-surrogate")
            with patch.object(alert, "send_dm", surr_dm), redirect_stderr(StringIO()):
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", "m")],
                    "offline.test", status_path=surr_receipt, now_et_date=PINNED_NOW.date(), require_message_id=True,
                ) is True
                assert alert.dispatch_dm_for_health_alerts(
                    [alert.Alert("CRITICAL", "late_delivery", "m")],
                    "offline.test", status_path=surr_receipt, now_et_date=PINNED_NOW.date(), require_message_id=True,
                ) is False
                assert surr_dm.call_count == 1

            # 3d. Double-down unconfirmed pick on both routes: action="double" and neither batter named
            p2 = picks.Pick("Juan Soto", 665742, "NYM", 2, "Snell", 605483, 0.81, [], False, 102, f"{DAY}T23:40:00Z")
            for route in ("direct", "classification"):
                dd_dir = troot / f"dd_{route}" / "picks"
                dd_dir.mkdir(parents=True)
                dp_dd = picks.DailyPick(DAY, PINNED_NOW.isoformat(), p1, p2, None, delivery_attempted=True)
                picks.save_pick(dp_dd, dd_dir)
                cfg_dd = {
                    "orchestrator": {"picks_dir": str(dd_dir), "heartbeat_path": str(dd_dir.parent / "hb")},
                    "bluesky": {"dm_recipient": "offline.test"},
                    "scheduler": {"pick_delivery": "dm", "live_forward_capture_on_lock": False, "contest_state_required": False},
                    "health_checks": {"enabled": False},
                    "tiers": [],
                }
                dd_dm = Mock(return_value="dd-ack")
                with ExitStack() as stack:
                    enter_env(stack, dd_dm, Mock(), StringIO())
                    if route == "direct":
                        st_dd = scheduler.SchedulerState(DAY, PINNED_NOW.isoformat(), [], [], [], False, None, None, None)
                        assert scheduler._deliver_and_lock_pick(dp_dd, cfg_dd, dd_dir, st_dd, DAY, "dd") is False
                    else:
                        assert scheduler.run_day(DAY, cfg_dd) is None
                dec_dd = load_decision(DAY, dd_dir)
                assert dec_dd["action"] == "double" and dec_dd["delivery_status"] == "locked_unconfirmed"
                body_dd = dd_dm.call_args.args[1]
                assert "Steven Kwan" not in body_dd and "Juan Soto" not in body_dd

            # 3e. Empty / INFO-only alerts with FIFO at status_path and require_message_id=True: no block, no open
            fifo_path = troot / "unused_fifo.json"
            os.mkfifo(fifo_path)
            assert alert.dispatch_dm_for_health_alerts(
                [alert.Alert("INFO", "late_delivery", "info only")],
                "offline.test", status_path=fifo_path, now_et_date=PINNED_NOW.date(), require_message_id=True,
            ) is False

        print("ITEMS 3 & 5 PROBE OK: U7 copy unchanged from round 1, CLAUDE.md C2 (incl. no-ID delivered vs watchdog) verified live, and all extra adversarial edges passed.")
    finally:
        if str(ROOT / "src") in sys.path:
            sys.path.remove(str(ROOT / "src"))
        signal.alarm(0)


if __name__ == "__main__":
    main()
