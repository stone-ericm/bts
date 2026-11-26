"""Probes for Items 0, 1, and 4:
- Exact character-by-character verification of E1, E2, E3 runtime texts against Contract Version 2 + Amendment 2
- Reproduction/verification of all 14 Round-1 findings (Codex B1-B8, Carbon B1-B6) and their applied replacements on HEAD
- Per-sentence worst-state execution across CLI commands, shimmed deploy.yml, shimmed cron-setup-hetzner.sh,
  shimmed install-systemd-hetzner.sh, install-leaderboard-systemd.sh, and capture_bts_cookies.py
"""
from __future__ import annotations

import contextlib
import html
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT_DIR = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")

from bts import (
    contest_state,
    day_outcome,
    dm,
    orchestrator,
    picks,
    saver_state,
    scheduler,
    web,
)
from bts.health import alert as health_alert, contest_state as health_contest_state
from bts.leaderboard import auth as lb_auth
from scripts.audit.season_walk import walk


def _make_state(date_str: str = "2027-07-15", **kw):
    return scheduler.SchedulerState(
        date=date_str,
        schedule_fetched_at=f"{date_str}T10:00:00-04:00",
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
        **kw,
    )


def run_subproc(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, timeout: int = 20):
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_exact_runtime_texts() -> dict:
    results = {}
    day = date(2027, 7, 15)
    now = datetime(2027, 7, 15, 12, 0, tzinfo=ET)

    # K2 with BL-N3 status removal
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        cfg = walk._config(tp / "picks")
        cfg["orchestrator"]["heartbeat_path"] = str(tp / "hb")
        (tp / "picks").mkdir(parents=True)
        frame = walk._predictions(day, 0.76)
        frame.loc[0, "p_game_hit"] = 0.999
        statuses = {
            70000: {"abstract": "F", "detailed": "Final"},
            70001: {"abstract": "P", "detailed": "Scheduled"},
            70002: {"abstract": "P", "detailed": "Scheduled"},
        }
        err_buf = io.StringIO()
        with walk.offline(now), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")), \
                patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses), \
                patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 0), \
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: contextlib.nullcontext()), \
                contextlib.redirect_stderr(err_buf):
            assert walk.install_state(tp / "picks", day, walk.State("dated", 10, 10, "used", 0.75)) == []
            ret_k2 = scheduler.run_single_check(
                date=day.isoformat(), all_game_pks=list(statuses), confirmed_sides=set(),
                config=cfg, early_lock_gap=0.03,
            )
        sum_k2 = ret_k2["skip_summary"]
        k2_expected_body = (
            "Candidate used in the decision: Batter 1 (T1) 75.0%, below the streak-10 bar (79.6%) "
            "using unrounded values. Streak used at this check: 10."
        )
        assert scheduler.format_skip_dm(day.isoformat(), sum_k2) == f"BTS 2027-07-15: No pick at this check. {k2_expected_body}"
        assert f"  SKIP — {k2_expected_body}" in err_buf.getvalue().splitlines()
        assert web.render_skip_banner(sum_k2) == (
            f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(k2_expected_body)}</span></div>'
        )
        results["E1_K2_BL_N3"] = "exact"

    # K1: real tail stop at streak 0, best 18, 8 days left (2027-09-19)
    day_k1 = date(2027, 9, 19)
    now_k1 = datetime(2027, 9, 19, 12, 0, tzinfo=ET)
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        cfg = walk._config(tp / "picks")
        cfg["orchestrator"]["heartbeat_path"] = str(tp / "hb")
        (tp / "picks").mkdir(parents=True)
        frame = walk._predictions(day_k1, 0.76)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        err_buf = io.StringIO()
        with walk.offline(now_k1), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")), \
                patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses), \
                patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 0), \
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: contextlib.nullcontext()), \
                contextlib.redirect_stderr(err_buf):
            assert walk.install_state(tp / "picks", day_k1, walk.State("dated", 0, 18, "used", 0.76)) == []
            ret_k1 = scheduler.run_single_check(
                date=day_k1.isoformat(), all_game_pks=list(statuses), confirmed_sides=set(),
                config=cfg, early_lock_gap=0.03,
            )
        sum_k1 = ret_k1["skip_summary"]
        reason_k1 = sum_k1["reason"]
        k1_expected_body = (
            f"Candidate used in the decision: Batter 0 (T0) 76.0%. Reason: {reason_k1}. "
            "Streak used at this check: 0."
        )
        assert scheduler.format_skip_dm(day_k1.isoformat(), sum_k1) == f"BTS 2027-09-19: No pick at this check. {k1_expected_body}"
        assert f"  SKIP — {k1_expected_body}" in err_buf.getvalue().splitlines()
        assert web.render_skip_banner(sum_k1) == (
            f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(k1_expected_body)}</span></div>'
        )
        results["E1_K1_real_tail_stop"] = "exact"

    # K3: Aug 30 at streak 56, top_p=0.99 skipped over 82.5% bar
    day_k3 = date(2027, 8, 30)
    now_k3 = datetime(2027, 8, 30, 12, 0, tzinfo=ET)
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        cfg = walk._config(tp / "picks")
        cfg["orchestrator"]["heartbeat_path"] = str(tp / "hb")
        (tp / "picks").mkdir(parents=True)
        frame = walk._predictions(day_k3, 0.99)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        err_buf = io.StringIO()
        with walk.offline(now_k3), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")), \
                patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses), \
                patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 0), \
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: contextlib.nullcontext()), \
                contextlib.redirect_stderr(err_buf):
            assert walk.install_state(tp / "picks", day_k3, walk.State("dated", 56, 56, "used", 0.99)) == []
            ret_k3 = scheduler.run_single_check(
                date=day_k3.isoformat(), all_game_pks=list(statuses), confirmed_sides=set(),
                config=cfg, early_lock_gap=0.03,
            )
        sum_k3 = ret_k3["skip_summary"]
        k3_expected_body = (
            "Candidate used in the decision: Batter 0 (T0) 99.0%, at or above the streak-56 bar (82.5%). "
            "Streak used at this check: 56."
        )
        assert scheduler.format_skip_dm(day_k3.isoformat(), sum_k3) == f"BTS 2027-08-30: No pick at this check. {k3_expected_body}"
        assert f"  SKIP — {k3_expected_body}" in err_buf.getvalue().splitlines()
        assert web.render_skip_banner(sum_k3) == (
            f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(k3_expected_body)}</span></div>'
        )
        results["E1_K3_real_top_bin"] = "exact"

    # K4: July 15 at streak 56 (74 days left, all bins skip -> bar is None)
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        cfg = walk._config(tp / "picks")
        cfg["orchestrator"]["heartbeat_path"] = str(tp / "hb")
        (tp / "picks").mkdir(parents=True)
        frame = walk._predictions(day, 0.96)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        err_buf = io.StringIO()
        with walk.offline(now), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")), \
                patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses), \
                patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 0), \
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: contextlib.nullcontext()), \
                contextlib.redirect_stderr(err_buf):
            assert walk.install_state(tp / "picks", day, walk.State("dated", 56, 56, "used", 0.96)) == []
            ret_k4 = scheduler.run_single_check(
                date=day.isoformat(), all_game_pks=list(statuses), confirmed_sides=set(),
                config=cfg, early_lock_gap=0.03,
            )
        sum_k4 = ret_k4["skip_summary"]
        k4_expected_body = (
            "Candidate used in the decision: Batter 0 (T0) 96.0%. "
            "No probability comparison is available for this check. Streak used at this check: 56."
        )
        assert scheduler.format_skip_dm(day.isoformat(), sum_k4) == f"BTS 2027-07-15: No pick at this check. {k4_expected_body}"
        assert f"  SKIP — {k4_expected_body}" in err_buf.getvalue().splitlines()
        assert web.render_skip_banner(sum_k4) == (
            f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(k4_expected_body)}</span></div>'
        )
        results["E1_K4_real_all_skip"] = "exact"

    # K5: all games Final -> no_eligible selection
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        cfg = walk._config(tp / "picks")
        cfg["orchestrator"]["heartbeat_path"] = str(tp / "hb")
        (tp / "picks").mkdir(parents=True)
        frame = walk._predictions(day, 0.96)
        statuses = {int(pk): {"abstract": "F", "detailed": "Final"} for pk in frame["game_pk"]}
        err_buf = io.StringIO()
        with walk.offline(now), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")), \
                patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses), \
                patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 0), \
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: contextlib.nullcontext()), \
                contextlib.redirect_stderr(err_buf):
            assert walk.install_state(tp / "picks", day, walk.State("dated", 10, 10, "used", 0.96)) == []
            ret_k5 = scheduler.run_single_check(
                date=day.isoformat(), all_game_pks=list(statuses), confirmed_sides=set(),
                config=cfg, early_lock_gap=0.03,
            )
        sum_k5 = ret_k5["skip_summary"]
        k5_expected_body = (
            "No candidate was chosen. Top model prediction (eligibility unverified): Batter 0 (T0) 96.0%. "
            "Streak used at this check: 10."
        )
        assert scheduler.format_skip_dm(day.isoformat(), sum_k5) == f"BTS 2027-07-15: No pick at this check. {k5_expected_body}"
        assert f"  SKIP — {k5_expected_body}" in err_buf.getvalue().splitlines()
        assert web.render_skip_banner(sum_k5) == (
            f'<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>{html.escape(k5_expected_body)}</span></div>'
        )
        results["E1_K5_real_no_eligible"] = "exact"

    # 2. E2: delivery & morning stale warnings across old / future / unreadable and max 128-char stamp
    max_stamp = "0001-01-01T00:00:00." + "0" * 102 + "+00:00"
    future_stamp = "9999-12-31T23:59:59." + "9" * 102 + "+00:00"
    for kind, stamp in [("old_max", max_stamp), ("future_max", future_stamp), ("unreadable", "not-a-timestamp")]:
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            picks_dir = tp / "d1" / "picks"
            picks_dir.mkdir(parents=True)
            st = contest_state.DecisionStreakState(
                10, False, False, "contest", "fresh", None, None,
                observation_kind="automatic", contest_recorded_at=stamp,
            )
            rec = contest_state.automatic_reading_record(st, now)
            if rec["reason"] == "old":
                detail = f"last accepted fetch {rec['recorded_at']}, about {rec['age_hours']:.2f} hours ago"
            elif rec["reason"] == "future":
                detail = f"its recorded_at {rec['recorded_at']} is in the future"
            else:
                detail = "it has no usable recorded_at"

            sent_deliv = []
            err_deliv = io.StringIO()
            with patch.object(scheduler, "_now_et", lambda: now), \
                    patch.object(health_alert, "send_dm", lambda r, b: sent_deliv.append(b) or "msg1"), \
                    contextlib.redirect_stderr(err_deliv):
                scheduler._warn_stale_contest_reading(rec, {"bluesky": {"dm_recipient": "u.test"}}, picks_dir)
            exp_deliv_body = (
                f"The pick was delivered using an automatic contest reading flagged stale ({detail}). Check MLB. "
                "If MLB differs from status (B8a), use bts set-contest-streak --date 2027-07-15 --streak N "
                "--best-streak B --saver S --picks-dir DIR. DIR is the absolute scheduler picks path. The sent "
                "pick and saver flag stay as they were. Write only if the pick's slate and today's ET date both "
                "equal 2027-07-15. For a later date, follow B8b(1). On 2027-07-15 ET away from midnight, if MLB "
                "shows this season's saver used, check bts saver-state --picks-dir DIR. For active or not_earned, "
                "run bts saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."
            )
            exp_deliv_dm = "BTS health WARN attention:\n- [contest_state] " + exp_deliv_body
            assert sent_deliv == [exp_deliv_dm]
            assert f"  STALE CONTEST READING — {exp_deliv_body}" in err_deliv.getvalue().splitlines()

            picks_dir2 = tp / "d2" / "picks"
            picks_dir2.mkdir(parents=True)
            sent_morn = []
            err_morn = io.StringIO()
            with patch.object(contest_state, "load_decision_streak_state", lambda *a, **kw: st), \
                    patch.object(health_alert, "send_dm", lambda r, b: sent_morn.append(b) or "msg2"), \
                    contextlib.redirect_stderr(err_morn):
                scheduler._morning_state_check(
                    {"bluesky": {"dm_recipient": "u.test"}}, picks_dir2, "2027-07-15", now, now + timedelta(hours=2)
                )
            exp_morn_body = (
                f"Morning check for 2027-07-15: the automatic contest reading was stale at startup ({detail}). "
                "This reading is eligible if it remains in use. Check MLB. If prior entries are settled or absent "
                "and MLB differs from status (B8a), use bts set-contest-streak --date 2027-07-15 --streak N "
                "--best-streak B --saver S --picks-dir DIR on 2027-07-15 ET only. DIR is the absolute scheduler "
                "picks path. This leaves the saver flag unchanged. On 2027-07-15 ET away from midnight, if MLB "
                "shows this season's saver used, check bts saver-state --picks-dir DIR. For active or not_earned, "
                "run bts saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status."
            )
            exp_morn_dm = "BTS health WARN attention:\n- [contest_state] " + exp_morn_body
            assert sent_morn == [exp_morn_dm]
            assert f"  MORNING STATE CHECK — stale contest reading: {detail}" in err_morn.getvalue().splitlines()
            results[f"E2_{kind}_lengths"] = {"delivery": len(exp_deliv_dm), "morning": len(exp_morn_dm)}
            assert len(exp_deliv_dm) < 900 and len(exp_morn_dm) < 900

    assert results["E2_old_max_lengths"] == {"delivery": 881, "morning": 882}

    # 3. E3: Late & Void game delivery refusals
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "d1" / "picks"
        tp.mkdir(parents=True)
        p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 70001, "2027-07-15T23:05:00Z")
        p2 = picks.Pick("Luis Arraez", 650333, "SD", 1, "Logan Webb", 657277, 0.81, [], False, 70002, "2027-07-15T23:10:00Z")
        p_double = picks.DailyPick("2027-07-15", "2027-07-15T19:10:00-04:00", p1, p2, None)
        picks.save_pick(p_double, tp)
        st_l = _make_state("2027-07-15")
        cutoff = datetime(2027, 7, 15, 19, 0, tzinfo=ET)
        now_l = datetime(2027, 7, 15, 19, 10, tzinfo=ET)
        sent_l = []
        with patch.object(health_alert, "dispatch_dm_for_health_alerts", lambda alerts, *a, **kw: sent_l.extend(alerts)), \
                contextlib.redirect_stderr(io.StringIO()):
            scheduler._refuse_late_delivery(p_double, {}, tp, st_l, "2027-07-15", "primary", cutoff, now_l)
        exp_tail = (
            "This pick was withheld. A later check may produce another recommendation. "
            "Use a later official pick message only while the entry window is open. "
            "A manual entry can differ from a later recommendation. "
            "The entry check may report a mismatch if it verifies the entry in time."
        )
        assert sent_l[0].source == "late_delivery" and sent_l[0].level == "CRITICAL"
        assert sent_l[0].message == (
            "LATE DELIVERY REFUSED: Steven Kwan + Luis Arraez would have been sent 10 min after the 19:00 ET cutoff. "
            + exp_tail
        )

        tp2 = Path(tmp) / "d2" / "picks"
        tp2.mkdir(parents=True)
        picks.save_pick(p_double, tp2)
        st_v = _make_state("2027-07-15")
        sent_v = []
        with patch.object(scheduler, "_now_et", lambda: now_l), \
                patch.object(health_alert, "dispatch_dm_for_health_alerts", lambda alerts, *a, **kw: sent_v.extend(alerts)), \
                contextlib.redirect_stderr(io.StringIO()):
            scheduler._refuse_void_game_delivery(
                p_double, {}, tp2, st_v, "2027-07-15", "primary", {70002: "Cancelled", 70001: "Postponed"}
            )
        assert sent_v[0].source == "postponed_pick" and sent_v[0].level == "CRITICAL"
        assert sent_v[0].message == (
            "DELIVERY REFUSED for 2027-07-15: game 70001 is Postponed, game 70002 is Cancelled. "
            + exp_tail
        )
        results["E3_refusals"] = "exact"

    return results


def check_round1_and_operator_states() -> dict:
    out = {}

    # --- R1 Codex B1: Morning warning at 00:20 ET July 15 while July 14 game is still live ---
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        picks_dir = tp / "d" / "picks"
        picks_dir.mkdir(parents=True)
        day15 = date(2027, 7, 15)
        at_0020 = datetime(2027, 7, 15, 0, 20, tzinfo=ET)
        with walk.offline(at_0020):
            assert walk.install_state(picks_dir, day15, walk.State("automatic", 13, 14, "used", 0.82)) == []
            walk._set_clock(at_0020)
            auto_path = picks_dir / "account_state" / "contest_streak.json"
            auto_data = json.loads(auto_path.read_text())
            auto_data["recorded_at"] = "2027-07-14T04:00:00Z"
            auto_path.write_text(json.dumps(auto_data))

            sent_morn = []
            with patch.object(health_alert, "send_dm", lambda r, b: sent_morn.append(b) or "m1"), \
                    contextlib.redirect_stderr(io.StringIO()):
                scheduler._morning_state_check(
                    {"bluesky": {"dm_recipient": "u.test"}},
                    picks_dir, "2027-07-15", at_0020, datetime(2027, 7, 15, 16, 10, tzinfo=ET),
                )
            assert "If prior entries are settled or absent and MLB differs from status (B8a), use" in sent_morn[0]

            at_0900 = datetime(2027, 7, 15, 9, 0, tzinfo=ET)
            walk._set_clock(at_0900)
            code, cli_out = walk._invoke(
                "set-contest-streak", "--date", "2027-07-15", "--streak", "0",
                "--best-streak", "14", "--saver", "used", "--picks-dir", str(picks_dir),
            )
            assert code == 0, cli_out
            walk._set_clock(datetime(2027, 7, 15, 12, 0, tzinfo=ET))
            frame = walk._predictions(day15, 0.82)
            statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
            with patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")), \
                    patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses):
                _, sel, _ = orchestrator.run_and_pick(walk._config(picks_dir), "2027-07-15")
            assert sel.action == "double"
            out["codex_B1_morning_settlement"] = {"action_after_settled_miss": sel.action, "status": "fixed"}

    # --- Post-midnight delivery warning & C1 historical alert & dated-day 5a refusal ---
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        picks_dir = tp / "d" / "picks"
        picks_dir.mkdir(parents=True)
        at_0005 = datetime(2027, 7, 15, 0, 5, tzinfo=ET)
        with walk.offline(at_0005):
            code_yest, out_yest = walk._invoke(
                "set-contest-streak", "--date", "2027-07-14", "--streak", "14",
                "--best-streak", "14", "--saver", "used", "--picks-dir", str(picks_dir),
            )
            assert code_yest == 2 and "must be today (2027-07-15)" in out_yest
            hist_alerts = health_contest_state.check(
                picks_dir, expected=True,
                now=datetime(2027, 7, 14, 23, 59, 59, 999999, tzinfo=ET),
                historical_day=date(2027, 7, 14),
            )
            assert hist_alerts and all("set-contest-streak" not in a.message for a in hist_alerts)

            at_noon = datetime(2027, 7, 15, 12, 0, tzinfo=ET)
            walk._set_clock(at_noon)
            walk._invoke("set-contest-streak", "--date", "2027-07-15", "--streak", "10", "--best-streak", "10", "--saver", "used", "--picks-dir", str(picks_dir))
            st1 = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=at_noon)
            p1 = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 70001, "2027-07-15T23:05:00Z")
            saved_dp = picks.DailyPick("2027-07-15", at_noon.isoformat(), p1, None, None)
            saved_dp.chosen_on = picks.chosen_on_record(st1, required=True)
            walk._set_clock(datetime(2027, 7, 15, 14, 0, tzinfo=ET))
            walk._invoke("set-contest-streak", "--date", "2027-07-15", "--streak", "0", "--best-streak", "10", "--saver", "used", "--picks-dir", str(picks_dir))
            st2 = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=datetime(2027, 7, 15, 14, 0, tzinfo=ET))
            refusal_5a = picks.saved_pick_refusal(saved_dp, st2)
            assert refusal_5a is not None
            out["post_midnight_c1_and_5a"] = {
                "yesterday_date_exit": code_yest,
                "c1_command_free": True,
                "dated_change_refuses_saved_pick": True,
            }

    # --- R1 Codex B2 & Carbon B6 & Carbon N4 ---
    for bad_num in (1e308, 1.5e308, -0.1, 1.0000001, 2):
        s_bad = {"best_batter": "A", "best_team": "T", "best_p": bad_num, "bar": bad_num, "streak": 5}
        rendered = scheduler.format_skip_summary(s_bad)
        assert "inf%" not in rendered and "?" in rendered and "No probability comparison" in rendered
    s_ws = {"best_batter": "A", "best_team": "T", "best_p": 0.75, "bar": 0.80, "streak": 5, "reason": " \r\n\t "}
    assert "Reason:" not in scheduler.format_skip_summary(s_ws) and "below the streak-5 bar" in scheduler.format_skip_summary(s_ws)
    out["codex_B2_carbon_B6_N4"] = "fixed"

    # --- R1 Codex B3 & Carbon B2: A5 text ---
    a5_text = [ln for ln in (REPO / "docs/ops/2027-season-start.md").read_text().splitlines() if ln.startswith("| A5 |")][0]
    assert "follow B8c steps 1–7 in order, with B8d governing step 5" in a5_text
    assert "steps 2 and 5–7" not in a5_text
    out["codex_B3_carbon_B2_a5_text"] = "fixed"

    # --- Saver matrix: not_earned, active, used, uninitialized + Jan 1 + wrong cwd (Codex B6) ---
    saver_outcomes = {}
    for init_st in ("uninitialized", "not_earned", "active", "used"):
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            live_picks = tp / "live" / "picks"
            wrong_cwd = tp / "other_checkout"
            (wrong_cwd / "data" / "picks").mkdir(parents=True)
            live_picks.mkdir(parents=True)
            at_day = datetime(2027, 7, 15, 14, 0, tzinfo=ET)
            with walk.offline(at_day):
                if init_st != "uninitialized":
                    c0, o0 = walk._invoke("saver-state", "--init", init_st, "--season", "2027", "--picks-dir", str(live_picks))
                    assert c0 == 0, o0
                old_cwd = os.getcwd()
                try:
                    os.chdir(wrong_cwd)
                    c_use, o_use = walk._invoke("saver-state", "--use", "--picks-dir", str(live_picks))
                finally:
                    os.chdir(old_cwd)
                final_live = saver_state.load_saver_state(live_picks, season=2027).state
                final_wrong = saver_state.load_saver_state(wrong_cwd / "data" / "picks", season=2027).state
                assert final_wrong == "uninitialized"
                saver_outcomes[init_st] = {"exit": c_use, "final_live": final_live, "out": o_use.strip().splitlines()[0]}
    assert saver_outcomes["uninitialized"]["exit"] == 1 and saver_outcomes["uninitialized"]["final_live"] == "uninitialized"
    assert saver_outcomes["not_earned"]["exit"] == 0 and saver_outcomes["not_earned"]["final_live"] == "used"
    assert saver_outcomes["active"]["exit"] == 0 and saver_outcomes["active"]["final_live"] == "used"
    assert saver_outcomes["used"]["exit"] == 0 and saver_outcomes["used"]["final_live"] == "used"

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "picks"
        tp.mkdir()
        with walk.offline(datetime(2027, 12, 31, 20, 0, tzinfo=ET)):
            walk._invoke("saver-state", "--init", "active", "--season", "2027", "--picks-dir", str(tp))
        with walk.offline(datetime(2028, 1, 1, 0, 5, tzinfo=ET)):
            c_ny, o_ny = walk._invoke("saver-state", "--picks-dir", str(tp))
            assert c_ny == 0 and "uninitialized (season 2028)" in o_ny
    out["saver_matrix"] = saver_outcomes

    # --- R1 Carbon B3: B4 drill on copy with `. ./.env` exported while live picks untouched ---
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        live_picks = tp / "data" / "picks"
        (live_picks / "2027-07-14").mkdir(parents=True)
        dec_obj = {
            "schema_version": 1, "date": "2027-07-14", "action": "single", "source": "mdp",
            "scoreable": True, "delivery_status": "delivered",
            "primary": {"batter_id": 101, "batter_name": "Steven Kwan"},
        }
        (live_picks / "2027-07-14" / "decision.json").write_text(json.dumps(dec_obj))
        cfg_file = tp / "orch.toml"
        cfg_file.write_text(
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        drill_dir = tp / "drill"
        drill_dir.mkdir()
        shutil.copytree(live_picks, drill_dir / "picks")
        (drill_dir / "picks" / "2027-07-14" / "decision.json").unlink()

        sent_drill = []
        with walk.offline(datetime(2027, 7, 15, 9, 0, tzinfo=ET)), \
                patch.dict(os.environ, {"BTS_BLUESKY_APP_PASSWORD": "test-app-password"}, clear=False), \
                patch.object(scheduler, "fetch_schedule", lambda d: [{"game_pk": 1, "status": {"detailedState": "Final"}}]), \
                patch.object(dm, "retry_urlopen", lambda req, **kw: io.BytesIO(
                    json.dumps({"accessJwt": "j", "did": "did:plc:1", "convo": {"id": "c1"}, "id": "m1"}).encode()
                ) if (sent_drill.append(req.full_url) or True) else None):
            code_b4, out_b4 = walk._invoke(
                "day-outcome", "--date", "2027-07-14", "--config", str(cfg_file),
                "--picks-dir", str(drill_dir / "picks"),
                "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            )
        assert code_b4 == 1 and '"state": "no_decision"' in out_b4 and "DM failed" not in out_b4
        assert any("sendMessage" in u for u in sent_drill)
        assert (live_picks / "2027-07-14" / "decision.json").is_file()
        out["carbon_B3_b4_drill"] = {"exit": code_b4, "dm_sent": True, "live_decision_intact": True}

    # --- A7 revert to private/research + cron-setup-hetzner.sh + 08:30 delivered_under_research + B8d switch ---
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        fake_home = tp / "home"
        bts_repo = fake_home / "projects" / "bts"
        bin_dir = fake_home / ".local" / "bin"
        bts_repo.mkdir(parents=True)
        bin_dir.mkdir(parents=True)
        (bts_repo / ".env").write_text("HEALTHCHECKS_PING_URL=https://hc-ping.com/fake-uuid\nBTS_BLUESKY_APP_PASSWORD=pw\n")
        orch_toml = fake_home / ".bts-orchestrator.toml"
        orch_toml.write_text(
            '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\ncontest_state_required = true\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        crontab_store = tp / "crontab.txt"
        uv_shim = bin_dir / "uv"
        uv_shim.write_text(
            f"#!/usr/bin/env bash\nshift\nif [ \"$1\" = \"bts\" ]; then shift; exec {REPO / '.venv/bin/bts'} \"$@\"; fi\nexec \"$@\"\n"
        )
        uv_shim.chmod(0o755)
        crontab_shim = bin_dir / "crontab"
        crontab_shim.write_text(
            f"#!/usr/bin/env bash\nif [ \"$1\" = \"-l\" ]; then\n"
            f"  if [ -f {crontab_store} ]; then cat {crontab_store}; exit 0; else echo 'no crontab for bts' >&2; exit 1; fi\n"
            f"elif [ \"$1\" = \"-\" ]; then cat > {crontab_store}; exit 0;\n"
            f"elif [ \"$1\" = \"-r\" ]; then rm -f {crontab_store}; exit 0; fi\n"
        )
        crontab_shim.chmod(0o755)
        env = os.environ.copy()
        env["HOME"] = str(fake_home)
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        env["HEALTHCHECKS_PING_URL"] = "https://hc-ping.com/fake-uuid"

        r_res = run_subproc(["bash", str(REPO / "scripts/cron-setup-hetzner.sh"), "install"], env=env, cwd=bts_repo)
        assert r_res.returncode == 0, r_res.stderr
        research_cron = crontab_store.read_text()
        assert "check-pick-entered" not in research_cron and "day-outcome" in research_cron

        picks_a7 = tp / "picks_a7"
        (picks_a7 / "2027-07-14").mkdir(parents=True)
        (picks_a7 / "2027-07-14" / "decision.json").write_text(json.dumps({
            "schema_version": 1, "date": "2027-07-14", "action": "single", "source": "mdp",
            "scoreable": True, "delivery_status": "delivered",
            "primary": {"batter_id": 101, "batter_name": "Steven Kwan"},
        }))
        with walk.offline(datetime(2027, 7, 15, 8, 30, tzinfo=ET)), \
                patch.object(scheduler, "fetch_schedule", lambda d: [{"game_pk": 1, "status": {"detailedState": "Final"}}]):
            code_a7, out_a7 = walk._invoke(
                "day-outcome", "--date", "2027-07-14", "--config", str(orch_toml),
                "--picks-dir", str(picks_a7), "--expected-username", "stonehengee",
            )
        assert code_a7 == 1 and '"state": "delivered_under_research"' in out_a7

        orch_toml.write_text(
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        r_show = run_subproc(["bash", str(REPO / "scripts/cron-setup-hetzner.sh"), "show"], env=env, cwd=bts_repo)
        r_enter = run_subproc(["bash", str(REPO / "scripts/cron-setup-hetzner.sh"), "install"], env=env, cwd=bts_repo)
        enter_cron = crontab_store.read_text()
        assert r_show.returncode == 0 and r_enter.returncode == 0
        assert "check-pick-entered" in enter_cron
        out["A7_and_B8d_cron_switch"] = {
            "research_lines": len([l for l in research_cron.splitlines() if l.endswith("# BTS-HETZNER")]),
            "enter_lines": len([l for l in enter_cron.splitlines() if l.endswith("# BTS-HETZNER")]),
            "show_before_later_install_had_research_lines": "fetch-contest-streak" in r_show.stdout.split("Would install:")[0],
            "a7_0830_state": "delivered_under_research",
        }

    # --- R1 Codex B4 & B5 ---
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        local = tp / "local"
        run_subproc(["git", "init", "-b", "main", str(local)])
        run_subproc(["git", "config", "user.email", "t@example.com"], cwd=local)
        run_subproc(["git", "config", "user.name", "Test"], cwd=local)
        (local / "f.txt").write_text("stale\n")
        run_subproc(["git", "add", "f.txt"], cwd=local)
        run_subproc(["git", "commit", "-m", "stale main"], cwd=local)
        stale_sha = run_subproc(["git", "rev-parse", "main"], cwd=local).stdout.strip()
        run_subproc(["git", "checkout", "-b", "reviewed"], cwd=local)
        (local / "f.txt").write_text("approved\n")
        run_subproc(["git", "commit", "-am", "approved"], cwd=local)
        approved_sha = run_subproc(["git", "rev-parse", "HEAD"], cwd=local).stdout.strip()
        main_sha = run_subproc(["git", "rev-parse", "main"], cwd=local).stdout.strip()
        assert main_sha == stale_sha and main_sha != approved_sha
        out["codex_B4_rev_parse_main"] = {"caught_mismatch": True}

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "picks"
        tp.mkdir()
        with walk.offline(datetime(2026, 12, 20, 12, 0, tzinfo=ET)):
            c_dec, _ = walk._invoke("season-contract", "revoke", "--season", "2027", "--reason", "restore", "--picks-dir", str(tp))
        with walk.offline(datetime(2027, 1, 2, 12, 0, tzinfo=ET)):
            c_jan, _ = walk._invoke("season-contract", "revoke", "--season", "2027", "--reason", "restore", "--picks-dir", str(tp))
        assert c_dec == 2 and c_jan == 0
        out["codex_B5_december_revoke"] = {"dec_2026_exit": c_dec, "jan_2027_exit": c_jan}

    # --- Shimmed deploy.yml execution ---
    deploy_yml = (REPO / ".github/workflows/deploy.yml").read_text()
    script_body = deploy_yml.split("script: |\n", 1)[1]
    dedented = "\n".join(line[12:] if line.startswith("            ") else line for line in script_body.splitlines())
    deploy_outcomes = {}
    for scenario in ("canary_pass", "canary_fail", "early_sync_fail"):
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            origin = tp / "origin.git"
            box = tp / "box_repo"
            (box / "data").mkdir(parents=True)
            run_subproc(["git", "init", "--bare", str(origin)])
            run_subproc(["git", "init", "-b", "main", str(box)])
            run_subproc(["git", "config", "user.email", "t@example.com"], cwd=box)
            run_subproc(["git", "config", "user.name", "Test"], cwd=box)
            (box / "ver.txt").write_text("old\n")
            run_subproc(["git", "add", "ver.txt"], cwd=box)
            run_subproc(["git", "commit", "-m", "old"], cwd=box)
            old_sha = run_subproc(["git", "rev-parse", "HEAD"], cwd=box).stdout.strip()
            run_subproc(["git", "remote", "add", "origin", str(origin)], cwd=box)
            run_subproc(["git", "push", "origin", "main:main"], cwd=box)

            pusher = tp / "pusher"
            run_subproc(["git", "clone", str(origin), str(pusher)])
            run_subproc(["git", "config", "user.email", "t@example.com"], cwd=pusher)
            run_subproc(["git", "config", "user.name", "Test"], cwd=pusher)
            (pusher / "ver.txt").write_text("target\n")
            run_subproc(["git", "commit", "-am", "target"], cwd=pusher)
            target_sha = run_subproc(["git", "rev-parse", "HEAD"], cwd=pusher).stdout.strip()
            run_subproc(["git", "push", "origin", "HEAD:deploy"], cwd=pusher)
            run_subproc(["git", "reset", "--hard", old_sha], cwd=box)

            bin_d = tp / "bin"
            bin_d.mkdir()
            state_log = tp / "systemd_log.txt"
            (bin_d / "sudo").write_text(
                "#!/usr/bin/env bash\nwhile [[ \"$1\" == -* ]]; do\n"
                "  if [[ \"$1\" == \"-u\" || \"$1\" == \"-Hu\" ]]; then shift 2; else shift; fi\n"
                "done\nwhile [[ \"$1\" == *=* ]]; do export \"$1\"; shift; done\nexec \"$@\"\n"
            )
            (bin_d / "id").write_text("#!/usr/bin/env bash\necho 1000\n")
            (bin_d / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
            (bin_d / "tailscale").write_text("#!/usr/bin/env bash\necho 127.0.0.1\n")
            http_code = "500" if scenario == "canary_fail" else "204"
            (bin_d / "curl").write_text(f"#!/usr/bin/env bash\nprintf '{http_code}'\n")
            uv_exit = "1" if scenario == "early_sync_fail" else "0"
            uv_path = tp / "uv_bin"
            uv_path.write_text(f"#!/usr/bin/env bash\nexit {uv_exit}\n")
            uv_path.chmod(0o755)
            (bin_d / "systemctl").write_text(
                f"#!/usr/bin/env bash\necho \"$*\" >> {state_log}\n"
                "if [[ \"$*\" == *\"is-active --quiet\"* ]]; then exit 0; fi\n"
                "if [[ \"$*\" == *\"is-active\"* ]]; then echo active; exit 0; fi\n"
                "exit 0\n"
            )
            for p in bin_d.iterdir():
                p.chmod(0o755)

            adapted = (
                dedented
                .replace("/home/bts/projects/bts", str(box))
                .replace("/home/bts/.local/bin/uv", str(uv_path))
                .replace("git config --global --add safe.directory", "git config --local --add safe.directory")
                .replace("/etc/systemd/system/bts-scheduler.service", str(tp / "nonexistent_legacy_unit"))
            )
            script_file = tp / "run_deploy.sh"
            script_file.write_text(adapted)
            env = os.environ.copy()
            env["PATH"] = f"{bin_d}:{env.get('PATH', '')}"
            env["TARGET_SHA"] = target_sha
            res = run_subproc(["bash", str(script_file)], env=env, cwd=tp)
            box_head = run_subproc(["git", "rev-parse", "HEAD"], cwd=box).stdout.strip()
            origin_deploy = run_subproc(["git", "rev-parse", "refs/heads/deploy"], cwd=origin).stdout.strip()
            deploy_outcomes[scenario] = {
                "exit": res.returncode,
                "box_head_is": "target" if box_head == target_sha else ("old" if box_head == old_sha else "other"),
                "origin_deploy_is": "target" if origin_deploy == target_sha else "other",
                "canary_passed_printed": "CANARY PASSED" in res.stdout,
                "rolled_back_printed": "Rolled back cleanly" in res.stdout,
            }
    assert deploy_outcomes["canary_pass"]["box_head_is"] == "target"
    assert deploy_outcomes["canary_fail"]["box_head_is"] == "old" and deploy_outcomes["canary_fail"]["origin_deploy_is"] == "target"
    assert deploy_outcomes["early_sync_fail"]["box_head_is"] == "target"
    out["deploy_yml_shims"] = deploy_outcomes

    # --- CRITICAL PROBE A: INCIDENT.md:98 `uv run python scripts/capture_bts_cookies.py --print` & A4 & §3 table ---
    cookie_probe = {}
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "picks"
        tp.mkdir()
        for err_name, exc_obj in [
            ("rate_limit", lb_auth.RateLimitedLoginError("429")),
            ("transient", lb_auth.TransientAuthError("503")),
            ("auth", lb_auth.AuthError("expired")),
        ]:
            with patch.object(lb_auth, "load_session_cookies", side_effect=exc_obj):
                c_err, o_err = walk._invoke("fetch-contest-streak", "--picks-dir", str(tp))
            cookie_probe[f"cli_{err_name}"] = {"exit": c_err, "line": o_err.strip().splitlines()[0]}

    r_repo_capture = run_subproc(
        ["uv", "run", "--offline", "--no-sync", "python", "scripts/capture_bts_cookies.py", "--help"],
        env={**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache"},
    )
    cookie_probe["synced_repo_capture_exit"] = r_repo_capture.returncode
    cookie_probe["synced_repo_capture_err"] = r_repo_capture.stderr.strip().splitlines()[-1] if r_repo_capture.stderr else ""

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        pkg_src = tp / "dummy_pw"
        (pkg_src / "playwright").mkdir(parents=True)
        (pkg_src / "playwright" / "__init__.py").write_text("")
        (pkg_src / "playwright" / "sync_api.py").write_text("def sync_playwright(): raise RuntimeError('stub')\n")
        proj = tp / "scratch_bts"
        (proj / "scripts").mkdir(parents=True)
        shutil.copy2(REPO / "scripts" / "capture_bts_cookies.py", proj / "scripts" / "capture_bts_cookies.py")
        (proj / "pyproject.toml").write_text(
            '[project]\nname = "scratch-bts"\nversion = "0.1.0"\nrequires-python = ">=3.12"\ndependencies = []\n'
        )
        uv_env = {**os.environ, "UV_CACHE_DIR": str(tp / "uv-cache")}
        r_lock = run_subproc(["uv", "lock", "--offline"], cwd=proj, env=uv_env)
        r_venv = run_subproc(["uv", "venv", "--python", str(REPO / ".venv/bin/python")], cwd=proj, env=uv_env)
        sp = list((proj / ".venv" / "lib").glob("python*/site-packages"))[0]
        shutil.copytree(pkg_src / "playwright", sp / "playwright")
        dist_info = sp / "playwright-1.40.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: playwright\nVersion: 1.40.0\n")
        (dist_info / "INSTALLER").write_text("uv\n")
        (dist_info / "RECORD").write_text("playwright/__init__.py,,\nplaywright/sync_api.py,,\nplaywright-1.40.0.dist-info/METADATA,,\nplaywright-1.40.0.dist-info/INSTALLER,,\nplaywright-1.40.0.dist-info/RECORD,,\n")

        r_before = run_subproc([str(proj / ".venv/bin/python"), "scripts/capture_bts_cookies.py", "--help"], cwd=proj, env=uv_env)
        r_uv_run = run_subproc(
            ["uv", "run", "--offline", "python", "scripts/capture_bts_cookies.py", "--help"],
            cwd=proj, env=uv_env,
        )
        cookie_probe["scratch_uv_repro"] = {
            "lock_rc": r_lock.returncode,
            "venv_rc": r_venv.returncode,
            "direct_venv_before_uv_run_rc": r_before.returncode,
            "uv_run_rc": r_uv_run.returncode,
            "uv_run_stderr": r_uv_run.stderr.strip(),
            "playwright_still_in_venv_after_uv_run": (sp / "playwright").exists(),
        }

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        pw_mod = ModuleType("playwright")
        pw_sync = ModuleType("playwright.sync_api")

        class FakeCtx:
            def new_page(self):
                return SimpleNamespace(goto=lambda u: None)
            def cookies(self):
                return [{"name": "oktaid", "value": "u1"}]

        class FakeBrowser:
            def new_context(self):
                return FakeCtx()
            def close(self):
                pass

        @contextlib.contextmanager
        def fake_sync_pw():
            yield SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kw: FakeBrowser()))

        pw_sync.sync_playwright = fake_sync_pw
        sec_calls = []
        stdout_no_print = io.StringIO()
        stderr_no_print = io.StringIO()
        with patch.dict(sys.modules, {"playwright": pw_mod, "playwright.sync_api": pw_sync}), \
                patch.object(sys, "platform", "darwin"), \
                patch.object(sys, "argv", ["capture_bts_cookies.py"]), \
                patch("builtins.input", lambda *a: ""), \
                patch("subprocess.run", lambda cmd, **kw: sec_calls.append(cmd) or SimpleNamespace(returncode=0)), \
                contextlib.redirect_stdout(stdout_no_print), \
                contextlib.redirect_stderr(stderr_no_print):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_cap_test", REPO / "scripts/capture_bts_cookies.py")
            cap_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cap_mod)
            cap_mod.main()
        cookie_probe["without_print_stdout"] = stdout_no_print.getvalue()
        cookie_probe["without_print_files_created"] = [str(p) for p in tp.iterdir()]
        cookie_probe["keychain_called"] = any("add-generic-password" in c for c in sec_calls)
    out["cookie_probe"] = cookie_probe

    # --- CRITICAL PROBE B: INCIDENT.md:137-140 `scripts/install-systemd-hetzner.sh` vs `/dev/null` masks & daemon-reload ---
    systemd_probe = {}
    for install_mode in ("gnu_linux_install", "macos_bsd_install"):
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            fake_home = tp / "home"
            user_units = fake_home / ".config" / "systemd" / "user"
            user_units.mkdir(parents=True)
            shutil.copy2(REPO / "scripts/systemd/bts-scheduler.service", user_units / "bts-scheduler.service")
            shutil.copy2(REPO / "scripts/systemd/bts-dashboard.service", user_units / "bts-dashboard.service")
            for masked_name in (
                "bts-leaderboard.service", "bts-leaderboard.timer",
                "bts-live-forward-capture.service", "bts-lineup-collect.service",
            ):
                os.symlink("/dev/null", user_units / masked_name)

            bin_d = tp / "bin"
            bin_d.mkdir()
            reload_snapshot_file = tp / "at_daemon_reload.json"
            (bin_d / "systemctl").write_text(
                f"#!/usr/bin/env bash\n"
                f"if [[ \"$*\" == *\"daemon-reload\"* ]]; then\n"
                f"  {REPO / '.venv/bin/python'} -c 'import json, os; from pathlib import Path; d = Path(\"{user_units}\"); "
                f"print(json.dumps({{p.name: (\"symlink->\" + os.readlink(p) if p.is_symlink() else \"regular_file\") for p in sorted(d.iterdir())}}))' > {reload_snapshot_file}\n"
                f"fi\nexit 0\n"
            )
            (bin_d / "systemctl").chmod(0o755)

            if install_mode == "gnu_linux_install":
                (bin_d / "install").write_text(
                    "#!/usr/bin/env bash\n"
                    "mode=\"0755\"\nif [ \"$1\" = \"-m\" ]; then mode=\"$2\"; shift 2; fi\n"
                    "src=\"$1\"; dst=\"$2\"\nrm -f \"$dst\"\ncp \"$src\" \"$dst\"\nchmod \"$mode\" \"$dst\"\n"
                )
                (bin_d / "install").chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = str(fake_home)
            env["PATH"] = f"{bin_d}:{env.get('PATH', '')}"
            res = run_subproc(["bash", str(REPO / "scripts/install-systemd-hetzner.sh")], env=env)
            at_reload = json.loads(reload_snapshot_file.read_text()) if reload_snapshot_file.exists() else None
            systemd_probe[install_mode] = {
                "exit": res.returncode,
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip(),
                "units_at_internal_daemon_reload": at_reload,
            }

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        fake_home = tp / "home"
        user_units = fake_home / ".config" / "systemd" / "user"
        user_units.mkdir(parents=True)
        os.symlink("/dev/null", user_units / "bts-leaderboard.service")
        os.symlink("/dev/null", user_units / "bts-leaderboard.timer")
        bin_d = tp / "bin"
        bin_d.mkdir()
        (bin_d / "systemctl").write_text(
            f"#!/usr/bin/env bash\n"
            f"if [[ \"$*\" == *\"enable --now bts-leaderboard.timer\"* ]]; then\n"
            f"  if [ -L \"{user_units}/bts-leaderboard.timer\" ] && [ \"$(readlink \"{user_units}/bts-leaderboard.timer\")\" = \"/dev/null\" ]; then\n"
            f"    echo 'Failed to enable unit: Unit file bts-leaderboard.timer is masked.' >&2\n"
            f"    exit 1\n"
            f"  fi\n"
            f"fi\nexit 0\n"
        )
        (bin_d / "systemctl").chmod(0o755)
        env = {**os.environ, "HOME": str(fake_home), "PATH": f"{bin_d}:{os.environ.get('PATH', '')}"}
        r_lb = run_subproc(["bash", str(REPO / "scripts/install-leaderboard-systemd.sh"), "install"], env=env)
        systemd_probe["b8e_leaderboard_install_over_mask"] = {
            "exit": r_lb.returncode,
            "stderr": r_lb.stderr.strip(),
            "still_symlink_to_devnull": (
                (user_units / "bts-leaderboard.service").is_symlink()
                and os.readlink(user_units / "bts-leaderboard.service") == "/dev/null"
            ),
        }

    out["systemd_probe"] = systemd_probe
    return out


def main() -> None:
    report = {
        "exact_runtime_texts": check_exact_runtime_texts(),
        "round1_and_operator_states": check_round1_and_operator_states(),
    }
    out_path = OUT_DIR / "probe_0_1_4_results.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
