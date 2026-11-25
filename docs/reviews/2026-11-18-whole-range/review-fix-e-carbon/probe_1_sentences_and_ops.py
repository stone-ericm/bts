"""Probe 1 & 4: Character-for-character contract check, worst-state execution of every operator
sentence, real CLI commands, shimmed deploy.yml / cron-setup-hetzner.sh / systemd installers,
and reproduction of document findings (A5 on 619abda7, B4 without .env, INCIDENT §3 & §4).
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OCT_DIR = REPO / ".carbon-review/wr-e/oct_619abda7"
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner
from bts import cli as cli_mod, contest_state, day_outcome, orchestrator, picks, saver_state, scheduler, strategy, web
from bts.daily_decision import write_decision
from bts.health import alert, contest_state as health_contest_state
from scripts.audit.season_walk import walk

ET = ZoneInfo("America/New_York")


def extract_october_619abda7() -> None:
    if not (OCT_DIR / "src/bts").exists():
        OCT_DIR.mkdir(parents=True, exist_ok=True)
        p1 = subprocess.Popen(
            ["git", "archive", "619abda7", "src", "scripts"],
            cwd=REPO, stdout=subprocess.PIPE,
        )
        subprocess.run(["tar", "-x", "-C", str(OCT_DIR)], stdin=p1.stdout, check=True, timeout=15)
        p1.stdout.close()
        assert p1.wait(timeout=15) == 0


def test_part_1a_character_for_character() -> dict:
    """Verify every E1, E2, E3 message character for character against contract Version 2."""
    results = {}

    # E1: K1..K5 through real run_single_check -> build_skip_summary -> format_skip_dm, journal, render_skip_banner
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        cfg = walk._config(picks_dir)
        cfg["orchestrator"]["heartbeat_path"] = str(tmp_path / "hb")

        # K1: tail stop rule (reason ends with '; no pick' -> gains one '.')
        day_k1 = date(2027, 9, 19)
        now_k1 = datetime(2027, 9, 19, 10, 0, tzinfo=ET)
        frame_k1 = walk._predictions(day_k1, 0.76)
        frame_k1.loc[0, "p_game_hit"] = 0.999  # game 70000 Final -> removed before decision
        statuses_removed = {
            70000: {"abstract": "F", "detailed": "Final"},
            70001: {"abstract": "P", "detailed": "Scheduled"},
            70002: {"abstract": "P", "detailed": "Scheduled"},
        }
        buf = io.StringIO()
        with walk.offline(now_k1), \
             patch.object(orchestrator, "run_cascade", lambda *a: (frame_k1.copy(), "t")), \
             patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses_removed), \
             patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 1), \
             patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()), \
             redirect_stderr(buf):
            assert walk.install_state(picks_dir, day_k1, walk.State("dated", 0, 18, "used", 0.76)) == []
            r_k1 = scheduler.run_single_check(day_k1.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
        s_k1 = r_k1["skip_summary"]
        exp_k1 = (
            "Candidate used in the decision: Batter 1 (T1) 75.0%. "
            "Reason: season-best 18 can't be beaten with 8 days left (max reachable 16); no pick. "
            "Streak used at this check: 0."
        )
        assert scheduler.format_skip_dm(day_k1.isoformat(), s_k1) == f"BTS {day_k1.isoformat()}: No pick at this check. {exp_k1}"
        assert f"  SKIP — {exp_k1}" in buf.getvalue().splitlines()
        assert web.render_skip_banner(s_k1) == f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(exp_k1)}</span></div>'

        # K2: below bar (including equal-rounded 0.7956 < 0.7960!)
        day_k2 = date(2027, 7, 15)
        now_k2 = datetime(2027, 7, 15, 10, 0, tzinfo=ET)
        frame_k2 = walk._predictions(day_k2, 0.8056)  # Batter 1 gets 0.8056 - 0.01 = 0.7956 -> rounds to 79.6%!
        frame_k2.loc[0, "p_game_hit"] = 0.999
        buf = io.StringIO()
        with walk.offline(now_k2), \
             patch.object(orchestrator, "run_cascade", lambda *a: (frame_k2.copy(), "t")), \
             patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses_removed), \
             patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 1), \
             patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()), \
             redirect_stderr(buf):
            assert walk.install_state(picks_dir, day_k2, walk.State("dated", 10, 10, "used", 0.75)) == []
            r_k2 = scheduler.run_single_check(day_k2.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
        s_k2 = r_k2["skip_summary"]
        exp_k2 = (
            "Candidate used in the decision: Batter 1 (T1) 79.6%, below the streak-10 bar (79.6%) "
            "using unrounded values. Streak used at this check: 10."
        )
        assert scheduler.format_skip_dm(day_k2.isoformat(), s_k2) == f"BTS {day_k2.isoformat()}: No pick at this check. {exp_k2}"
        assert f"  SKIP — {exp_k2}" in buf.getvalue().splitlines()
        assert web.render_skip_banner(s_k2) == f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(exp_k2)}</span></div>'

        # K3: shipped top-bin skip (2027-08-30 at streak 56, p=0.99 >= bar 0.825)
        day_k3 = date(2027, 8, 30)
        now_k3 = datetime(2027, 8, 30, 10, 0, tzinfo=ET)
        frame_k3 = walk._predictions(day_k3, 0.99)
        statuses_open = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in (70000, 70001, 70002)}
        buf = io.StringIO()
        with walk.offline(now_k3), \
             patch.object(orchestrator, "run_cascade", lambda *a: (frame_k3.copy(), "t")), \
             patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses_open), \
             patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 1), \
             patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()), \
             redirect_stderr(buf):
            assert walk.install_state(picks_dir, day_k3, walk.State("dated", 56, 56, "used", 0.99)) == []
            r_k3 = scheduler.run_single_check(day_k3.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
        s_k3 = r_k3["skip_summary"]
        exp_k3 = (
            "Candidate used in the decision: Batter 0 (T0) 99.0%, at or above the streak-56 bar (82.5%). "
            "Streak used at this check: 56."
        )
        assert scheduler.format_skip_dm(day_k3.isoformat(), s_k3) == f"BTS {day_k3.isoformat()}: No pick at this check. {exp_k3}"
        assert f"  SKIP — {exp_k3}" in buf.getvalue().splitlines()
        assert web.render_skip_banner(s_k3) == f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(exp_k3)}</span></div>'

        # K4: all-skip state (2027-07-15 at streak 56, bar=None)
        frame_k4 = walk._predictions(day_k2, 0.96)
        buf = io.StringIO()
        with walk.offline(now_k2), \
             patch.object(orchestrator, "run_cascade", lambda *a: (frame_k4.copy(), "t")), \
             patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses_open), \
             patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 1), \
             patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()), \
             redirect_stderr(buf):
            assert walk.install_state(picks_dir, day_k2, walk.State("dated", 56, 56, "used", 0.96)) == []
            r_k4 = scheduler.run_single_check(day_k2.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
        s_k4 = r_k4["skip_summary"]
        exp_k4 = (
            "Candidate used in the decision: Batter 0 (T0) 96.0%. "
            "No probability comparison is available for this check. Streak used at this check: 56."
        )
        assert scheduler.format_skip_dm(day_k2.isoformat(), s_k4) == f"BTS {day_k2.isoformat()}: No pick at this check. {exp_k4}"
        assert f"  SKIP — {exp_k4}" in buf.getvalue().splitlines()
        assert web.render_skip_banner(s_k4) == f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(exp_k4)}</span></div>'

        # K5: no_eligible (all games Final)
        statuses_closed = {pk: {"abstract": "F", "detailed": "Final"} for pk in (70000, 70001, 70002)}
        buf = io.StringIO()
        with walk.offline(now_k2), \
             patch.object(orchestrator, "run_cascade", lambda *a: (frame_k4.copy(), "t")), \
             patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses_closed), \
             patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 0), \
             patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()), \
             redirect_stderr(buf):
            assert walk.install_state(picks_dir, day_k2, walk.State("dated", 10, 10, "used", 0.96)) == []
            r_k5 = scheduler.run_single_check(day_k2.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
        s_k5 = r_k5["skip_summary"]
        exp_k5 = (
            "No candidate was chosen. Top model prediction (eligibility unverified): Batter 0 (T0) 96.0%. "
            "Streak used at this check: 10."
        )
        assert scheduler.format_skip_dm(day_k2.isoformat(), s_k5) == f"BTS {day_k2.isoformat()}: No pick at this check. {exp_k5}"
        assert f"  SKIP — {exp_k5}" in buf.getvalue().splitlines()
        assert web.render_skip_banner(s_k5) == f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(exp_k5)}</span></div>'

    # Also check streak=None on K2/K3/K5
    assert scheduler.format_skip_summary({"best_batter": "A", "best_team": "T", "best_p": 0.70, "bar": 0.80, "streak": None}) == (
        "Top model prediction (eligibility unverified): A (T) 70.0%, below the bar (80.0%) using unrounded values."
    )
    assert scheduler.format_skip_summary({"best_batter": "A", "best_team": "T", "best_p": 0.80, "bar": 0.80, "streak": None}) == (
        "Top model prediction (eligibility unverified): A (T) 80.0%, at or above the bar (80.0%)."
    )
    assert scheduler.format_skip_summary({"best_batter": "A", "best_team": "T", "best_p": 0.80, "basis": "no_decision", "streak": None}) == (
        "No candidate was chosen. Top model prediction (eligibility unverified): A (T) 80.0%."
    )

    # E2: Delivery & Morning warnings across old (max 128-char stamp, plus year 9999 check), future, unreadable
    for now_test in (datetime(2027, 7, 15, 10, 0, tzinfo=ET), datetime(9999, 12, 31, 10, 0, tzinfo=ET)):
        stamp = "0001-01-01T00:00:00." + "0" * 102 + "+00:00"
        ds = contest_state.DecisionStreakState(
            10, False, False, "contest", "fresh", None, None,
            observation_kind="automatic", contest_recorded_at=stamp,
        )
        rec = contest_state.automatic_reading_record(ds, now_test)
        assert len(stamp) == 128
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            sent = []
            buf = io.StringIO()
            with patch.object(scheduler, "_now_et", lambda _n=now_test: _n), \
                 patch.object(alert, "send_dm", lambda r, b: sent.append(b) or "m1"), \
                 redirect_stderr(buf):
                scheduler._warn_stale_contest_reading(rec, {"bluesky": {"dm_recipient": "op"}}, pdir)
            d_str = now_test.date().isoformat()
            detail = f"last accepted fetch {stamp}, about {rec['age_hours']:.2f} hours ago"
            exp_deliv_body = (
                f"The pick was delivered using an automatic contest reading flagged stale ({detail}). "
                f"Check MLB. If MLB differs from status (B8a), use bts set-contest-streak --date {d_str} "
                "--streak N --best-streak B --saver S --picks-dir DIR. DIR is the absolute scheduler picks path. "
                f"The sent pick and saver flag stay as they were. Write only if the pick's slate and today's ET date both equal {d_str}. "
                f"For a later date, follow B8b(1). On {d_str} ET away from midnight, if MLB shows this season's saver used, "
                "check bts saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir DIR. "
                "For uninitialized, see B8a. Check the latest fetch status."
            )
            assert sent[0] == "BTS health WARN attention:\n- [contest_state] " + exp_deliv_body
            assert len(sent[0]) == 881 < 900
            assert f"  STALE CONTEST READING — {exp_deliv_body}" in buf.getvalue().splitlines()

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            sent = []
            buf = io.StringIO()
            with patch.object(contest_state, "load_decision_streak_state", lambda *a, **kw: ds), \
                 patch.object(alert, "send_dm", lambda r, b: sent.append(b) or "m2"), \
                 redirect_stderr(buf):
                scheduler._morning_state_check(
                    {"bluesky": {"dm_recipient": "op"}}, pdir, now_test.date().isoformat(),
                    now_test, now_test + timedelta(hours=1),
                )
            exp_morn_body = (
                f"Morning check for {d_str}: the automatic contest reading was stale at startup ({detail}). "
                "This reading is eligible if it remains in use. Check MLB. If MLB differs from status (B8a), use "
                f"bts set-contest-streak --date {d_str} --streak N --best-streak B --saver S --picks-dir DIR on {d_str} ET only. "
                "DIR is the absolute scheduler picks path. This leaves the saver flag unchanged. "
                f"On {d_str} ET away from midnight, if MLB shows this season's saver used, check bts saver-state --picks-dir DIR. "
                "For active or not_earned, run bts saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."
            )
            assert sent[0] == "BTS health WARN attention:\n- [contest_state] " + exp_morn_body
            assert len(sent[0]) == 842 < 900
            assert f"  MORNING STATE CHECK — stale contest reading: {detail}" in buf.getvalue().splitlines()

    # E3: Late and Postponed refusals (double-down and multi-game too)
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        now = datetime(2027, 7, 15, 19, 12, tzinfo=ET)
        cutoff = datetime(2027, 7, 15, 19, 0, tzinfo=ET)
        dp = picks.DailyPick(
            date="2027-07-15", run_time=now.isoformat(),
            pick=picks.BatterPick("Steven Kwan", 101, "CLE", 1, "P1", 901, 0.82, "", 70001, "2027-07-15T23:05:00Z", "BOS"),
            double_down=picks.BatterPick("Jose Ramirez", 102, "CLE", 3, "P2", 902, 0.80, "", 70002, "2027-07-15T23:10:00Z", "NYY"),
            runner_up=None, bluesky_posted=False, bluesky_uri=None,
        )
        picks.save_pick(dp, pdir)
        st = scheduler.SchedulerState("2027-07-15", now.isoformat(), [], [], [], False, None, None, None)
        sent_alerts = []
        with patch.object(scheduler, "_now_et", lambda: now), \
             patch.object(alert, "dispatch_dm_for_health_alerts", lambda al, *a, **kw: sent_alerts.extend(al)), \
             redirect_stderr(io.StringIO()):
            scheduler._refuse_late_delivery(dp, {}, pdir, st, "2027-07-15", "fallback", cutoff, now)
        tail_five = (
            "This pick was withheld. A later check may produce another recommendation. "
            "Use a later official pick message only while the entry window is open. "
            "A manual entry can differ from a later recommendation. "
            "The entry check may report a mismatch if it verifies the entry in time."
        )
        assert sent_alerts[0].message == f"LATE DELIVERY REFUSED: Steven Kwan + Jose Ramirez would have been sent 12 min after the 19:00 ET cutoff. {tail_five}"

        picks.save_pick(dp, pdir)
        sent_alerts.clear()
        with patch.object(scheduler, "_now_et", lambda: now), \
             patch.object(alert, "dispatch_dm_for_health_alerts", lambda al, *a, **kw: sent_alerts.extend(al)), \
             redirect_stderr(io.StringIO()):
            scheduler._refuse_void_game_delivery(dp, {}, pdir, st, "2027-07-15", "fallback", {70002: "Cancelled", 70001: "Postponed"})
        assert sent_alerts[0].message == f"DELIVERY REFUSED for 2027-07-15: game 70001 is Postponed, game 70002 is Cancelled. {tail_five}"

    results["part_1a"] = "PASS"
    return results


def test_part_1b_cli_and_worst_states() -> dict:
    """Exercise every CLI command and worst state from brief item 1."""
    out = {}

    # 1. After midnight ET with slate's game still live
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        # Clock at 00:10 ET on 2027-07-16 while 2027-07-15 game is still live
        after_midnight = datetime(2027, 7, 16, 0, 10, tzinfo=ET)
        with walk.offline(after_midnight):
            # Literal yesterday date from 23:50 warning is refused with exit 2
            code, msg = walk._invoke(
                "set-contest-streak", "--date", "2027-07-15",
                "--streak", "5", "--best-streak", "10", "--saver", "not_earned",
                "--picks-dir", pdir,
            )
            assert code == 2 and "must be today (2027-07-16)" in msg
            assert not (pdir / "account_state/contest_streak.manual.json").exists()

            # Historical EOD health check (C1) for 2027-07-15 evaluated at 00:10 ET names slate and prints no command
            eod_alerts = health_contest_state.check(pdir, expected=True, today=date(2027, 7, 15))
            assert eod_alerts and all("set-contest-streak" not in a.message for a in eod_alerts)
            assert any("2027-07-15" in a.message for a in eod_alerts)
    out["after_midnight_refusal_and_c1"] = "PASS"

    # 2. Saver flag matrix (active, not_earned, used, uninitialized + race + Jan 1 + wrong cwd)
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "real_picks"
        pdir.mkdir()
        wrong_cwd = Path(tmp) / "wrong_cwd"
        wrong_cwd.mkdir()
        old_cwd = os.getcwd()
        try:
            os.chdir(wrong_cwd)
            noon_jul = datetime(2027, 7, 15, 12, 0, tzinfo=ET)
            with walk.offline(noon_jul):
                # uninitialized: check prints uninitialized, --use exits 1 with --init hint, B8a --init used succeeds
                c0, t0 = walk._invoke("saver-state", "--picks-dir", pdir)
                assert c0 == 0 and "saver_state: uninitialized (season 2027)" in t0
                cu, tu = walk._invoke("saver-state", "--use", "--picks-dir", pdir)
                assert cu == 1 and "bts saver-state --init used --season 2027" in tu
                ci, ti = walk._invoke("saver-state", "--init", "used", "--season", "2027", "--picks-dir", pdir)
                assert ci == 0 and "saver_state: used (season 2027)" in ti
                # used -> --use is idempotent exit 0
                cu2, tu2 = walk._invoke("saver-state", "--use", "--picks-dir", pdir)
                assert cu2 == 0 and "Saver is already used." in tu2
                # active -> --use marks used
                walk._invoke("saver-state", "--undo", "--picks-dir", pdir)
                ca, ta = walk._invoke("saver-state", "--use", "--picks-dir", pdir)
                assert ca == 0 and "Marked saver used." in ta
                # not_earned -> --use marks used and prints --init not_earned --force --season 2027 revert
                walk._invoke("saver-state", "--init", "not_earned", "--force", "--season", "2027", "--picks-dir", pdir)
                cn, tn = walk._invoke("saver-state", "--use", "--picks-dir", pdir)
                assert cn == 0 and "bts saver-state --init not_earned --force --season 2027" in tn
                # Race: auto-earn sets not_earned before --init used -> exits 1 without --force, reread + --use succeeds
                (pdir / "account_state/saver_state.json").unlink()
                saver_state.maybe_auto_earn_saver(pdir, best_streak=5, season=2027)
                cr, tr = walk._invoke("saver-state", "--init", "used", "--season", "2027", "--picks-dir", pdir)
                assert cr == 1 and "already initialized as 'not_earned'" in tr
                cr2, _ = walk._invoke("saver-state", "--use", "--picks-dir", pdir)
                assert cr2 == 0 and saver_state.load_saver_state(pdir, season=2027).state == "used"

                # Jan 1 boundary: at 2028-01-01 00:05 ET, 2028 saver is uninitialized while 2027 is active
                walk._invoke("saver-state", "--init", "active", "--force", "--season", "2027", "--picks-dir", pdir)
                jan1 = datetime(2028, 1, 1, 0, 5, tzinfo=ET)
                walk._set_clock(jan1)
                cj, tj = walk._invoke("saver-state", "--picks-dir", pdir)
                assert cj == 0 and "saver_state: uninitialized (season 2028)" in tj

            # Verify nothing was created in wrong_cwd
            assert list(wrong_cwd.iterdir()) == []
        finally:
            os.chdir(old_cwd)
    out["saver_matrix_jan1_wrong_cwd"] = "PASS"

    # 3. Post-delivery stale repair during entry window -> grading -> 08:30 day-outcome -> next-day automatic read
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        day1 = date(2027, 7, 15)
        day2 = date(2027, 7, 16)
        now1 = datetime(2027, 7, 15, 17, 0, tzinfo=ET)
        with walk.offline(now1):
            # Setup automatic state: streak=7, saver=active, recorded 20h ago (stale)
            assert walk.install_state(pdir, day1, walk.State("automatic", 7, 12, "active", 0.85)) == []
            ds_before = contest_state.load_decision_streak_state(pdir, require_contest_state=True, now=now1)
            assert ds_before.streak == 7 and ds_before.saver_available is True

            # Simulate confirmed pick delivery on day1
            dp = picks.DailyPick(
                date=day1.isoformat(), run_time=now1.isoformat(),
                pick=picks.BatterPick("Steven Kwan", 101, "CLE", 1, "P1", 901, 0.85, "", 70001, "2027-07-15T23:05:00Z", "BOS"),
                double_down=None, runner_up=None, bluesky_posted=False, bluesky_uri=None,
                dm_delivered=True, dm_message_id="msg-101", delivery_attempted=True,
                chosen_on=picks.chosen_on_record(ds_before, required=True),
                automatic_reading=contest_state.automatic_reading_record(ds_before, now1),
            )
            picks.save_pick(dp, pdir)
            write_decision(
                pdir, date=day1.isoformat(), action="single", source="mdp",
                scoreable=True, delivery_status="delivered",
                primary={"batter_id": 101, "batter_name": "Steven Kwan", "team": "CLE", "game_pk": 70001, "p_game_hit": 0.85},
                double_down=None, streak=7, saver_available=True, state_source="contest", state_status="fresh",
            )
            pick_bytes_before = (pdir / f"{day1.isoformat()}.json").read_bytes()
            dec_bytes_before = (pdir / day1.isoformat() / "decision.json").read_bytes()

            # Operator follows E2 delivery warning during entry window:
            # 1) set-contest-streak --date 2027-07-15 --streak 12 --best-streak 15 --saver used --picks-dir DIR
            c_set, _ = walk._invoke(
                "set-contest-streak", "--date", "2027-07-15",
                "--streak", "12", "--best-streak", "15", "--saver", "used",
                "--picks-dir", pdir,
            )
            assert c_set == 0
            # Verify sent pick and saver flag stayed as they were!
            assert (pdir / f"{day1.isoformat()}.json").read_bytes() == pick_bytes_before
            assert (pdir / day1.isoformat() / "decision.json").read_bytes() == dec_bytes_before
            assert saver_state.load_saver_state(pdir, season=2027).state == "active"

            # 2) saver-state --use --picks-dir DIR
            c_use, _ = walk._invoke("saver-state", "--use", "--picks-dir", pdir)
            assert c_use == 0
            assert saver_state.load_saver_state(pdir, season=2027).state == "used"

            # Same-day scheduler restart does not re-pick or refuse committed pick
            cfg = walk._config(pdir)
            cfg["orchestrator"]["heartbeat_path"] = str(Path(tmp) / "hb")
            with patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 0), \
                 redirect_stderr(io.StringIO()):
                r_re = scheduler.run_single_check(day1.isoformat(), [70001], set(), cfg, 0.03)
            assert r_re["pick_result"].locked is True and r_re["should_post"] is False
            assert (pdir / f"{day1.isoformat()}.json").read_bytes() == pick_bytes_before

            # Next day (2027-07-16 12:00 ET): dated observation expired -> returns to automatic with saver=used!
            walk._set_clock(datetime(2027, 7, 16, 12, 0, tzinfo=ET))
            ds_day2 = contest_state.load_decision_streak_state(pdir, require_contest_state=True, now=datetime(2027, 7, 16, 12, 0, tzinfo=ET))
            assert ds_day2.observation_kind == "automatic"
            assert ds_day2.saver_flag_state == "used" and ds_day2.saver_available is False
    out["post_delivery_repair_and_day2_saver"] = "PASS"

    # 4. Pre-delivery provisional pick + changed dated observation -> 5a refuses stale saved pick
    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        day1 = date(2027, 7, 15)
        now1 = datetime(2027, 7, 15, 12, 0, tzinfo=ET)
        with walk.offline(now1):
            walk._invoke("set-contest-streak", "--date", "2027-07-15", "--streak", "7", "--best-streak", "10", "--saver", "active", "--picks-dir", pdir)
            ds1 = contest_state.load_decision_streak_state(pdir, require_contest_state=True, now=now1)
            dp = picks.DailyPick(
                date=day1.isoformat(), run_time=now1.isoformat(),
                pick=picks.BatterPick("Steven Kwan", 101, "CLE", 1, "P1", 901, 0.85, "", 70001, "2027-07-15T23:05:00Z", "BOS"),
                double_down=None, runner_up=None, bluesky_posted=False, bluesky_uri=None,
                chosen_on=picks.chosen_on_record(ds1, required=True),
            )
            walk._invoke("set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "15", "--saver", "used", "--picks-dir", pdir)
            ds2 = contest_state.load_decision_streak_state(pdir, require_contest_state=True, now=now1)
            refusal = picks.saved_pick_refusal(dp, ds2)
            assert refusal is not None and "changed" in refusal or refusal is not None
    out["provisional_pick_5a_refusal"] = "PASS"

    return out


def test_part_1c_shims_and_doc_findings() -> dict:
    """Exercise deploy.yml, cron-setup-hetzner.sh, systemd scripts under shims, and reproduce doc findings."""
    extract_october_619abda7()
    findings = {}

    # --- Repro 1: Checklist Row A5 (`docs/ops/2027-season-start.md:14`) on restored 619abda7 box ---
    # A5 says: "On the restored box follow B8c steps 2 and 5–7 in order, with B8d governing the switch."
    # If followed literally (steps 2, 5, 6, 7 without steps 1, 3 deploy, 4 revoke) on October 619abda7:
    with tempfile.TemporaryDirectory() as tmp:
        box_home = Path(tmp) / "home_bts"
        bts_dir = box_home / "projects/bts"
        bts_dir.mkdir(parents=True)
        # Symlink October 619abda7 scripts and src into bts_dir
        os.symlink(OCT_DIR / "scripts", bts_dir / "scripts")
        os.symlink(OCT_DIR / "src", bts_dir / "src")
        (bts_dir / ".env").write_text("HEALTHCHECKS_PING_URL=https://hc-ping.com/fake\nBTS_BLUESKY_APP_PASSWORD=fake\n")

        # Step 2 + Step 5 config in ~/.bts-orchestrator.toml
        (box_home / ".bts-orchestrator.toml").write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\nentry_intent = "enter"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        # Create shim for ~/.local/bin/uv and crontab
        bin_dir = box_home / ".local/bin"
        bin_dir.mkdir(parents=True)
        crontab_store = box_home / "crontab.txt"
        uv_shim = bin_dir / "uv"
        uv_shim.write_text(
            f'#!/usr/bin/env bash\n'
            f'shift # drop "run"\n'
            f'cmd="$1"; shift\n'
            f'if [ "$cmd" = "bts" ]; then\n'
            f'  PYTHONPATH="{OCT_DIR / "src"}" exec "{REPO / ".venv/bin/python"}" -m bts.cli "$@" || PYTHONPATH="{OCT_DIR / "src"}" exec "{REPO / ".venv/bin/python"}" -c "from bts.cli import cli; cli()" "$@"\n'
            f'fi\n'
        )
        # Simpler exact python entrypoint for bts:
        uv_shim.write_text(
            f'#!/usr/bin/env bash\n'
            f'shift\n'
            f'shift\n'
            f'PYTHONPATH="{OCT_DIR / "src"}" exec "{REPO / ".venv/bin/python"}" -c "import sys; from bts.cli import cli; cli()" "$@"\n'
        )
        uv_shim.chmod(0o755)
        crontab_shim = bin_dir / "crontab"
        crontab_shim.write_text(
            f'#!/usr/bin/env bash\n'
            f'if [ "${{1:-}}" = "-l" ]; then\n'
            f'  if [ -f "{crontab_store}" ]; then cat "{crontab_store}"; exit 0; else echo "no crontab for bts" >&2; exit 1; fi\n'
            f'elif [ "${{1:-}}" = "-" ]; then\n'
            f'  cat > "{crontab_store}"\n'
            f'fi\n'
        )
        crontab_shim.chmod(0o755)

        env = os.environ.copy()
        env["HOME"] = str(box_home)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"

        # Step 5 on 619abda7: entry-intent
        p_ei = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", "from bts.cli import cli; cli()", "entry-intent", "--config", str(box_home / ".bts-orchestrator.toml")],
            env={**env, "PYTHONPATH": str(OCT_DIR / "src")},
            capture_output=True, text=True, timeout=15,
        )
        # Step 6 on 619abda7: cron-setup-hetzner.sh install
        p_cron = subprocess.run(
            ["bash", "-c", "cd ~/projects/bts && set -a && . ./.env && set +a && bash scripts/cron-setup-hetzner.sh install"],
            env=env, capture_output=True, text=True, timeout=15,
        )
        installed_cron = crontab_store.read_text() if crontab_store.exists() else ""
        # Check what 619abda7 strategy._days_remaining('2027-03-24') returns!
        p_days = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", "from bts import strategy; print(strategy._days_remaining('2027-03-24'))"],
            env={**env, "PYTHONPATH": str(OCT_DIR / "src")},
            capture_output=True, text=True, timeout=15, check=True,
        )
        findings["repro_a5_on_619abda7"] = {
            "entry_intent_exit": p_ei.returncode,
            "entry_intent_stdout": p_ei.stdout.strip(),
            "cron_install_exit": p_cron.returncode,
            "has_hardcoded_2026_build": "--seasons 2026" in installed_cron,
            "missing_day_outcome": "day-outcome" not in installed_cron,
            "has_entry_checker": "check-pick-entered" in installed_cron,
            "october_2027_days_left": int(p_days.stdout.strip()),
        }
        assert p_ei.returncode == 0 and p_ei.stdout.strip() == "enter"
        assert p_cron.returncode == 0 and "--seasons 2026" in installed_cron and "day-outcome" not in installed_cron
        assert int(p_days.stdout.strip()) == 0

    # --- Repro 2: Checklist Row B4 (`docs/ops/2027-season-start.md:26`) without `.env` exported on Linux ---
    with tempfile.TemporaryDirectory() as tmp:
        drill = Path(tmp) / "drill"
        picks_copy = drill / "picks"
        (picks_copy / "2026-09-10").mkdir(parents=True)
        # Remove decision.json as B4 instructs
        cfg_file = Path(tmp) / "orch.toml"
        cfg_file.write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\nentry_intent = "enter"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        # Run day-outcome with clean Linux-like env (no BTS_BLUESKY_* env vars, security binary absent)
        code_b4 = (
            "import os, subprocess; from unittest.mock import patch; from click.testing import CliRunner; "
            "from bts.cli import cli; from bts import scheduler; "
            "orig_run = subprocess.run; "
            "def no_keychain(args, *a, **kw): "
            "    if args and args[0] == 'security': raise FileNotFoundError('security not on linux')\n"
            "    return orig_run(args, *a, **kw)\n"
            "sent = []; "
            "with patch.object(subprocess, 'run', no_keychain), "
            "     patch.object(scheduler, 'fetch_schedule', lambda d: [{'gamePk': 1, 'status': {'detailedState': 'Final'}}]): "
            f"    res = CliRunner().invoke(cli, ['day-outcome', '--date', '2026-09-10', '--config', '{cfg_file}', '--picks-dir', '{picks_copy}', '--expected-username', 'stonehengee', '--dm-recipient', 'stonehengee.bsky.social']); "
            "    import json; print(json.dumps({'exit_code': res.exit_code, 'output': res.output}))"
        )
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("BTS_BLUESKY")}
        clean_env["PYTHONPATH"] = str(REPO / "src")
        p_b4 = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c", code_b4],
            env=clean_env, capture_output=True, text=True, timeout=15, check=True,
        )
        b4_res = json.loads(p_b4.stdout)
        assert b4_res["exit_code"] == 1
        assert '"state": "no_decision"' in b4_res["output"]
        assert "day-outcome: DM failed: RuntimeError" in b4_res["output"]
        findings["repro_b4_missing_dotenv"] = b4_res

    # --- Repro 3: INCIDENT.md §4 (`INCIDENT.md:129-131`) `install-systemd-hetzner.sh` while held ---
    with tempfile.TemporaryDirectory() as tmp:
        box_home = Path(tmp) / "home"
        systemd_dir = box_home / ".config/systemd/user"
        systemd_dir.mkdir(parents=True)
        # Apply B8c step 1 masks (-> /dev/null)
        for u in ("bts-scheduler.service", "bts-dashboard.service", "bts-leaderboard.timer", "bts-leaderboard.service", "bts-lineup-collect.service"):
            os.symlink("/dev/null", systemd_dir / u)
        shim_bin = Path(tmp) / "bin"
        shim_bin.mkdir()
        # Case A: Offline before normal boot (systemctl --user daemon-reload fails, AFTER install -m 0644 already unmasked!)
        systemctl_offline = shim_bin / "systemctl"
        systemctl_offline.write_text('#!/usr/bin/env bash\necho "Failed to connect to bus: No medium found" >&2\nexit 1\n')
        systemctl_offline.chmod(0o755)
        p_inst_offline = subprocess.run(
            ["bash", str(REPO / "scripts/install-systemd-hetzner.sh")],
            env={**os.environ, "HOME": str(box_home), "PATH": f"{shim_bin}:{os.environ['PATH']}"},
            capture_output=True, text=True, timeout=15,
        )
        sched_is_symlink_after = (systemd_dir / "bts-scheduler.service").is_symlink()
        dash_is_symlink_after = (systemd_dir / "bts-dashboard.service").is_symlink()
        lineup_is_symlink_after = (systemd_dir / "bts-lineup-collect.service").is_symlink()
        assert p_inst_offline.returncode == 1
        assert not sched_is_symlink_after and not dash_is_symlink_after and not lineup_is_symlink_after
        findings["repro_incident_s4_install_while_held"] = {
            "offline_exit_code": p_inst_offline.returncode,
            "scheduler_mask_destroyed": not sched_is_symlink_after,
            "dashboard_mask_destroyed": not dash_is_symlink_after,
            "writer_service_mask_destroyed": not lineup_is_symlink_after,
        }

        # Also check B8e: cp in install-leaderboard-systemd.sh when masked to a dummy null target
        fake_null = Path(tmp) / "fake_dev_null"
        fake_null.write_text("")
        (systemd_dir / "bts-leaderboard.service").unlink()
        os.symlink(fake_null, systemd_dir / "bts-leaderboard.service")
        # POSIX cp into directory follows destination symlink!
        subprocess.run(["cp", str(REPO / "deploy/systemd/bts-leaderboard.service"), f"{systemd_dir}/"], check=True, timeout=10)
        findings["leaderboard_cp_follows_mask_symlink"] = (systemd_dir / "bts-leaderboard.service").is_symlink()

    # --- Repro 4: INCIDENT.md §3 (`INCIDENT.md:93`) `python scripts/capture_bts_cookies.py` ---
    p_py = subprocess.run(["bash", "-c", "command -v python"], capture_output=True, text=True, timeout=5)
    # Test capture_bts_cookies.py store_cookies / main without --print under mocked playwright & security
    code_cookies = (
        "import io, sys, subprocess; from contextlib import redirect_stdout, redirect_stderr; from unittest.mock import MagicMock, patch; "
        "fake_pw = MagicMock(); sys.modules['playwright'] = MagicMock(); sys.modules['playwright.sync_api'] = MagicMock(sync_playwright=fake_pw); "
        "import scripts.capture_bts_cookies as cap; "
        "calls = []; "
        "ctx = MagicMock(); ctx.cookies.return_value = [{'name': 'oktaid', 'value': '123'}]; "
        "browser = MagicMock(); browser.new_context.return_value = ctx; "
        "fake_pw.return_value.__enter__.return_value.chromium.launch.return_value = browser; "
        "out_no_print, err_no_print = io.StringIO(), io.StringIO(); "
        "with patch.object(sys, 'platform', 'darwin'), "
        "     patch.object(sys, 'argv', ['capture_bts_cookies.py']), "
        "     patch('builtins.input', return_value=''), "
        "     patch.object(subprocess, 'run', lambda a, **kw: calls.append(a) or MagicMock(returncode=0)), "
        "     redirect_stdout(out_no_print), redirect_stderr(err_no_print): "
        "    cap.main()\n"
        "import json; print(json.dumps({'stdout_without_print': out_no_print.getvalue(), 'stderr': err_no_print.getvalue(), 'keychain_calls': len(calls)}))"
    )
    p_cap = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", code_cookies],
        cwd=REPO, env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}"},
        capture_output=True, text=True, timeout=15, check=True,
    )
    cap_res = json.loads(p_cap.stdout)
    assert cap_res["stdout_without_print"] == ""
    findings["repro_incident_s3_cookies"] = {
        "bare_python_exists_on_mac": p_py.returncode == 0,
        "capture_without_print_stdout": cap_res["stdout_without_print"],
        "capture_stderr": cap_res["stderr"].strip(),
    }

    # --- Full B8c Steps 1-7 + deploy.yml shell body under shims (canary pass, canary fail, early sync stop) ---
    workflow_text = (REPO / ".github/workflows/deploy.yml").read_text()
    ssh_script = workflow_text.split("script: |\n", 1)[1]
    ssh_script = "\n".join(line[12:] if line.startswith("            ") else line for line in ssh_script.splitlines())

    deploy_outcomes = {}
    for scenario in ("canary_pass", "canary_fail", "early_sync_fail"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin_repo = root / "origin.git"
            box_home = root / "home_bts"
            box_repo = box_home / "projects/bts"
            (box_home / "projects").mkdir(parents=True)
            (box_repo.parent).mkdir(parents=True, exist_ok=True)
            # Create lightweight git origin with two commits (OLD_SHA and TARGET_SHA)
            subprocess.run(["git", "init", "--bare", str(origin_repo)], check=True, capture_output=True, timeout=10)
            work = root / "work"
            subprocess.run(["git", "clone", str(origin_repo), str(work)], check=True, capture_output=True, timeout=10)
            subprocess.run(["git", "-C", str(work), "config", "user.email", "test@local"], check=True, timeout=5)
            subprocess.run(["git", "-C", str(work), "config", "user.name", "test"], check=True, timeout=5)
            (work / "rev.txt").write_text("october-619abda7\n")
            subprocess.run(["git", "-C", str(work), "add", "rev.txt"], check=True, timeout=5)
            subprocess.run(["git", "-C", str(work), "commit", "-m", "october"], check=True, capture_output=True, timeout=5)
            old_sha = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
            (work / "rev.txt").write_text("season-2027-target\n")
            subprocess.run(["git", "-C", str(work), "add", "rev.txt"], check=True, timeout=5)
            subprocess.run(["git", "-C", str(work), "commit", "-m", "target2027"], check=True, capture_output=True, timeout=5)
            target_sha = subprocess.run(["git", "-C", str(work), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
            subprocess.run(["git", "-C", str(work), "push", "origin", "HEAD:refs/heads/deploy"], check=True, capture_output=True, timeout=10)

            subprocess.run(["git", "clone", str(origin_repo), str(box_repo)], check=True, capture_output=True, timeout=10)
            subprocess.run(["git", "-C", str(box_repo), "reset", "--hard", old_sha], check=True, capture_output=True, timeout=5)
            (box_repo / "data").mkdir(parents=True)

            # Build shims in root/shims
            shims = root / "shims"
            shims.mkdir()
            state_log = root / "shim_events.log"
            (box_home / ".local/bin").mkdir(parents=True)
            uv_box = box_home / ".local/bin/uv"
            uv_box.write_text(
                f'#!/usr/bin/env bash\n'
                f'echo "uv $*" >> "{state_log}"\n'
                f'if [ "{scenario}" = "early_sync_fail" ] && [ "$1" = "sync" ]; then exit 1; fi\n'
                f'exit 0\n'
            )
            uv_box.chmod(0o755)

            for name, body in {
                "sleep": "#!/usr/bin/env bash\nexit 0\n",
                "id": "#!/usr/bin/env bash\necho 1000\n",
                "tailscale": "#!/usr/bin/env bash\necho 100.64.0.1\n",
                "curl": (
                    f'#!/usr/bin/env bash\n'
                    f'if [ "{scenario}" = "canary_fail" ]; then printf "500"; exit 0; else printf "200"; exit 0; fi\n'
                ),
                "systemctl": (
                    f'#!/usr/bin/env bash\n'
                    f'echo "systemctl $*" >> "{state_log}"\n'
                    f'if [[ "$*" == *"is-active --quiet"* ]]; then exit 0; fi\n'
                    f'if [[ "$*" == *"is-active"* ]]; then echo "active"; exit 0; fi\n'
                    f'exit 0\n'
                ),
                "sudo": (
                    f'#!/usr/bin/env bash\n'
                    f'# Strip -Hu bts or -u bts and map /home/bts -> {box_home}\n'
                    f'while [[ "$1" == -* ]]; do\n'
                    f'  if [[ "$1" == "-u" || "$1" == "-Hu" ]]; then shift 2; else shift; fi\n'
                    f'done\n'
                    f'while [[ "$1" == *=* ]]; do export "$1"; shift; done\n'
                    f'args=()\n'
                    f'for a in "$@"; do args+=("${{a///home/bts/{box_home}}}"); done\n'
                    f'HOME="{box_home}" GIT_CONFIG_GLOBAL="{box_home}/.gitconfig" "${{args[@]}}"\n'
                ),
            }.items():
                p = shims / name
                p.write_text(body)
                p.chmod(0o755)

            adapted_script = ssh_script.replace("cd /home/bts/projects/bts", f'cd "{box_repo}"')
            proc = subprocess.run(
                ["bash", "-c", adapted_script],
                env={**os.environ, "PATH": f"{shims}:{os.environ['PATH']}", "TARGET_SHA": target_sha},
                capture_output=True, text=True, timeout=20,
            )
            final_sha = subprocess.run(
                ["git", "-C", str(box_repo), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            deploy_outcomes[scenario] = {
                "exit_code": proc.returncode,
                "final_equals_target": final_sha == target_sha,
                "final_equals_old": final_sha == old_sha,
                "canary_passed_printed": "CANARY PASSED" in proc.stdout,
                "rolled_back_printed": "Rolled back cleanly" in proc.stdout,
            }

    assert deploy_outcomes["canary_pass"] == {
        "exit_code": 0, "final_equals_target": True, "final_equals_old": False,
        "canary_passed_printed": True, "rolled_back_printed": False,
    }
    assert deploy_outcomes["canary_fail"] == {
        "exit_code": 1, "final_equals_target": False, "final_equals_old": True,
        "canary_passed_printed": False, "rolled_back_printed": True,
    }
    assert deploy_outcomes["early_sync_fail"] == {
        "exit_code": 1, "final_equals_target": True, "final_equals_old": False,
        "canary_passed_printed": False, "rolled_back_printed": False,
    }
    findings["deploy_shim_outcomes"] = deploy_outcomes

    # Test B8c steps 5-6 & A7 revert with HEAD cron-setup-hetzner.sh
    with tempfile.TemporaryDirectory() as tmp:
        box_home = Path(tmp) / "home_bts"
        bts_dir = box_home / "projects/bts"
        bts_dir.mkdir(parents=True)
        os.symlink(REPO / "scripts", bts_dir / "scripts")
        os.symlink(REPO / "src", bts_dir / "src")
        (bts_dir / ".env").write_text("HEALTHCHECKS_PING_URL=https://hc-ping.com/ok\n")
        bin_dir = box_home / ".local/bin"
        bin_dir.mkdir(parents=True)
        crontab_store = box_home / "crontab.txt"
        uv_shim = bin_dir / "uv"
        uv_shim.write_text(
            f'#!/usr/bin/env bash\nshift\nshift\n'
            f'PYTHONPATH="{REPO / "src"}" exec "{REPO / ".venv/bin/python"}" -c "from bts.cli import cli; cli()" "$@"\n'
        )
        uv_shim.chmod(0o755)
        crontab_shim = bin_dir / "crontab"
        crontab_shim.write_text(
            f'#!/usr/bin/env bash\n'
            f'if [ "${{1:-}}" = "-l" ]; then\n'
            f'  if [ -f "{crontab_store}" ]; then cat "{crontab_store}"; exit 0; else echo "no crontab for bts" >&2; exit 1; fi\n'
            f'elif [ "${{1:-}}" = "-" ]; then cat > "{crontab_store}"; fi\n'
        )
        crontab_shim.chmod(0o755)
        env = {**os.environ, "HOME": str(box_home), "PATH": f"{bin_dir}:{os.environ['PATH']}", "PYTHONPATH": str(REPO / "src")}

        # B8c step 5 & 6: DM + enter
        (box_home / ".bts-orchestrator.toml").write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "dm"\nentry_intent = "enter"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        p_inst_enter = subprocess.run(
            ["bash", "-c", "cd ~/projects/bts && set -a && . ./.env && set +a && bash scripts/cron-setup-hetzner.sh show && bash scripts/cron-setup-hetzner.sh install"],
            env=env, capture_output=True, text=True, timeout=15, check=True,
        )
        cron_enter = [ln for ln in crontab_store.read_text().splitlines() if ln and not ln.startswith("#")]
        assert len(cron_enter) == 19
        assert any("check-pick-entered" in ln for ln in cron_enter)
        assert any("day-outcome" in ln for ln in cron_enter)
        assert sum("fetch-contest-streak" in ln and "--dm-recipient" in ln for ln in cron_enter) == 4

        # A7 revert: private + research
        (box_home / ".bts-orchestrator.toml").write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\ncontest_state_required = true\npick_delivery = "private"\nentry_intent = "research"\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        subprocess.run(
            ["bash", "-c", "cd ~/projects/bts && set -a && . ./.env && set +a && bash scripts/cron-setup-hetzner.sh install"],
            env=env, capture_output=True, text=True, timeout=15, check=True,
        )
        cron_res = [ln for ln in crontab_store.read_text().splitlines() if ln and not ln.startswith("#")]
        assert len(cron_res) == 18
        assert not any("check-pick-entered" in ln for ln in cron_res)
        assert any("day-outcome" in ln for ln in cron_res)
        assert sum("fetch-contest-streak" in ln and "--dm-recipient" not in ln for ln in cron_res) == 4

        # Check A7 08:30 day-outcome on delivered scoreable pick under research -> delivered_under_research
        res_do = day_outcome.classify_day(
            date="2027-07-15", games=10, void_games=0,
            decision={"action": "single", "source": "mdp", "scoreable": True, "delivery_status": "delivered",
                      "primary": {"batter_name": "Steven Kwan"}},
            decision_sha256="abc", receipts=[], intent="research", expected_username="stonehengee",
        )
        assert res_do.state == "delivered_under_research" and not res_do.ok
    findings["head_cron_and_a7_revert"] = "PASS"

    return findings


def main() -> None:
    r1a = test_part_1a_character_for_character()
    r1b = test_part_1b_cli_and_worst_states()
    r1c = test_part_1c_shims_and_doc_findings()
    print(json.dumps({"1a": r1a, "1b": r1b, "1c": r1c}, indent=2))


if __name__ == "__main__":
    main()
