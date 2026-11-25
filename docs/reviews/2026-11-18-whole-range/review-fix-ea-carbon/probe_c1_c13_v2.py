"""Probe items 1 and 2 (v2): C1-C13 must-still-catch states, exact contract strings,
and every sentence of every message + the B8a bullet under adversarial states.
Offline only (network sockets refused), synthetic state, pinned ET clocks.
"""
from __future__ import annotations

import contextlib
import datetime as dt_mod
import html
import io
import json
import os
import socket
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from bts import (  # noqa: E402
    cli as cli_mod,
    contest_fetch,
    contest_state,
    daily_decision,
    dm,
    orchestrator,
    picks,
    saver_state,
    scheduler,
    season_contract,
    strategy,
    web,
)
from bts.health import alert, contest_state as health_contest_state  # noqa: E402
from bts.leaderboard import auth  # noqa: E402
from scripts.audit.season_walk import walk  # noqa: E402

ET = ZoneInfo("America/New_York")
REAL_DATETIME = dt_mod.datetime


def _block_network(*args, **kwargs):
    raise OSError("network disabled in review probe")


@contextlib.contextmanager
def pinned_world(at: datetime):
    walk._Clock.at = at
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(dt_mod, "datetime", walk._Clock))
        for mod in (
            cli_mod, contest_state, contest_fetch, daily_decision, orchestrator,
            picks, saver_state, scheduler, season_contract, strategy, web,
            alert, health_contest_state,
        ):
            if getattr(mod, "datetime", None) is REAL_DATETIME:
                stack.enter_context(patch.object(mod, "datetime", walk._Clock))
        stack.enter_context(patch.object(scheduler, "_now_et", lambda: walk._Clock.at.astimezone(ET)))
        stack.enter_context(patch.object(cli_mod, "_now_et", lambda: walk._Clock.at.astimezone(ET)))
        for m in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, m, _block_network))
        stack.enter_context(patch.object(socket, "getaddrinfo", _block_network))
        yield


def invoke_cli(*args, cwd: Path | None = None) -> tuple[int, str]:
    runner = CliRunner()
    prev = Path.cwd()
    try:
        if cwd is not None:
            os.chdir(cwd)
        res = runner.invoke(cli_mod.cli, [str(a) for a in args])
        out = res.output
        if res.exception is not None and not isinstance(res.exception, SystemExit):
            out += f"\n[{type(res.exception).__name__}: {res.exception}]"
        return res.exit_code, out
    finally:
        if cwd is not None:
            os.chdir(prev)


def make_daily_pick(day_str: str, batter: str = "Steven Kwan", batter_id: int = 680757,
                    team: str = "CLE", p: float = 0.82, game_pk: int = 70001,
                    game_hour_et: int = 19, game_min_et: int = 10,
                    dd: picks.Pick | None = None) -> picks.DailyPick:
    d = date.fromisoformat(day_str)
    gt_utc = REAL_DATETIME(d.year, d.month, d.day, game_hour_et, game_min_et, tzinfo=ET).astimezone(timezone.utc)
    gt_iso = gt_utc.isoformat().replace("+00:00", "Z")
    primary = picks.Pick(
        batter_name=batter, batter_id=batter_id, team=team, lineup_position=1,
        pitcher_name="Pitcher A", pitcher_id=9001, p_game_hit=p, flags=[],
        projected_lineup=False, game_pk=game_pk, game_time=gt_iso, pitcher_team="NYY",
    )
    return picks.DailyPick(
        date=day_str,
        run_time=REAL_DATETIME(d.year, d.month, d.day, 18, 25, tzinfo=ET).isoformat(),
        pick=primary,
        double_down=dd,
        runner_up=None,
    )


def check_exact_contract_strings() -> dict:
    """Verify exact character-for-character rendering on all surfaces against contract."""
    out = {}
    cases = {
        "K1": (
            {"best_batter": "B1", "best_team": "CLE", "best_p": 0.78, "streak": 0,
             "reason": "season-best 18 can't be beaten with 8 days left (max reachable 16); no pick",
             "basis": "decision", "candidate_verified": True},
            "Candidate used in the decision: B1 (CLE) 78.0%. Reason: season-best 18 can't be beaten with 8 days left (max reachable 16); no pick.",
        ),
        "K1_punct": (
            {"best_batter": "B1", "best_team": "CLE", "best_p": 0.78, "streak": 0,
             "reason": "Already punctuated!", "basis": "decision", "candidate_verified": True},
            "Candidate used in the decision: B1 (CLE) 78.0%. Reason: Already punctuated!",
        ),
        "K2": (
            {"best_batter": "B2", "best_team": "BOS", "best_p": 0.7956, "bar": 0.7960, "streak": 10,
             "basis": "decision", "candidate_verified": True},
            "Candidate used in the decision: B2 (BOS) 79.6%, below the streak-10 bar (79.6%) using unrounded values.",
        ),
        "K2_nostreak": (
            {"best_batter": "B2", "best_team": "BOS", "best_p": 0.75, "bar": 0.80, "streak": None,
             "basis": "decision", "candidate_verified": False},
            "Top model prediction (eligibility unverified): B2 (BOS) 75.0%, below the bar (80.0%) using unrounded values.",
        ),
        "K3": (
            {"best_batter": "Batter 0", "best_team": "T0", "best_p": 0.99, "bar": 0.825, "streak": 56,
             "basis": "decision", "candidate_verified": True},
            "Candidate used in the decision: Batter 0 (T0) 99.0%, at or above the streak-56 bar (82.5%).",
        ),
        "K4": (
            {"best_batter": "Batter 0", "best_team": "T0", "best_p": 0.96, "streak": 56,
             "basis": "decision", "candidate_verified": True},
            "Candidate used in the decision: Batter 0 (T0) 96.0%. No probability comparison is available for this check.",
        ),
        "K5": (
            {"best_batter": "Batter 0", "best_team": "T0", "best_p": 0.999, "bar": 0.796, "streak": 10,
             "basis": "no_decision", "candidate_verified": False},
            "No candidate was chosen. Top model prediction (eligibility unverified): Batter 0 (T0) 99.9%.",
        ),
    }
    for name, (summary, expected_sentence) in cases.items():
        s = summary.get("streak")
        suffix = f" Streak used at this check: {s}." if type(s) is int and s >= 0 else ""
        expected_body = expected_sentence + suffix
        dm_txt = scheduler.format_skip_dm("2027-07-15", summary)
        banner_txt = web.render_skip_banner(summary)
        assert dm_txt == f"BTS 2027-07-15: No pick at this check. {expected_body}", (name, dm_txt)
        assert banner_txt == (
            '<div class="skip-banner"><strong>LAST RECORDED NO-PICK SUMMARY</strong>'
            f"<span>{html.escape(expected_body)}</span></div>"
        ), (name, banner_txt)
    out["skip_cases_verified"] = list(cases.keys())

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        now = REAL_DATETIME(2027, 7, 15, 10, 0, tzinfo=ET)
        max_stamp = "0001-01-01T00:00:00." + "0" * 102 + "+00:00"
        details = {}
        for kind, stamp in [
            ("old_max", max_stamp),
            ("future_max", "9999-12-31T23:59:59." + "9" * 102 + "+00:00"),
            ("unreadable", "not-a-timestamp"),
        ]:
            st = contest_state.DecisionStreakState(
                10, True, True, "contest", "fresh", 0, True,
                observation_kind="automatic", contest_recorded_at=stamp,
            )
            rec = contest_state.automatic_reading_record(st, now)
            sent_deliv, sent_morn = [], []
            err_buf = io.StringIO()
            with pinned_world(now), \
                    patch.object(alert, "send_dm", lambda r, b: sent_deliv.append(b) or "msg1"), \
                    contextlib.redirect_stderr(err_buf):
                scheduler._warn_stale_contest_reading(
                    rec, {"bluesky": {"dm_recipient": "op.test"}}, picks_dir
                )
            for f in (Path(tmp) / "health_state").glob("*.json"):
                f.unlink()
            with pinned_world(now), \
                    patch.object(contest_state, "load_decision_streak_state", lambda *a, **k: st), \
                    patch.object(alert, "send_dm", lambda r, b: sent_morn.append(b) or "msg2"), \
                    contextlib.redirect_stderr(io.StringIO()):
                scheduler._morning_state_check(
                    {"bluesky": {"dm_recipient": "op.test"}}, picks_dir,
                    "2027-07-15", now, now + timedelta(hours=2),
                )
            for f in (Path(tmp) / "health_state").glob("*.json"):
                f.unlink()
            details[kind] = {
                "delivery_len": len(sent_deliv[0]),
                "morning_len": len(sent_morn[0]),
                "delivery_body": sent_deliv[0],
                "morning_body": sent_morn[0],
                "journal_matches_body": (
                    "  STALE CONTEST READING — " + sent_deliv[0].split("- [contest_state] ", 1)[1]
                ) in err_buf.getvalue().splitlines(),
            }
            assert details[kind]["delivery_len"] < 900
            assert details[kind]["morning_len"] < 900
            assert details[kind]["journal_matches_body"]

        d_str = "2027-07-15"
        det = details["old_max"]["delivery_body"].split("flagged stale (", 1)[1].split("). Check MLB.", 1)[0]
        assert len(det) == 177
        want_deliv = (
            "BTS health WARN attention:\n- [contest_state] "
            f"The pick was delivered using an automatic contest reading flagged stale ({det}). "
            f"Check MLB. If MLB differs from status (B8a), use bts set-contest-streak --date {d_str} "
            "--streak N --best-streak B --saver S --picks-dir DIR. DIR is the absolute scheduler picks path. "
            f"The sent pick and saver flag stay as they were. Write only if the pick's slate and today's ET date both equal {d_str}. "
            f"For a later date, see B8a. On {d_str} ET away from midnight, if MLB shows this season's saver used, "
            "check bts saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir DIR. "
            "For uninitialized, see B8a. Check the latest fetch status."
        )
        want_morn = (
            "BTS health WARN attention:\n- [contest_state] "
            f"Morning check for {d_str}: the automatic contest reading was stale at startup ({det}). "
            "This reading is eligible if it remains in use. Before the first check, if prior entries are settled or absent "
            f"and MLB differs from status (B8a), use bts set-contest-streak --date {d_str} --streak N --best-streak B "
            f"--saver S --picks-dir DIR on {d_str} ET only. DIR is the absolute scheduler picks path. "
            f"This leaves the saver flag unchanged. On {d_str} ET away from midnight, if MLB shows this season's saver used, "
            "check bts saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir DIR. "
            "For uninitialized, see B8a. Check the latest fetch status."
        )
        assert details["old_max"]["delivery_body"] == want_deliv
        assert details["old_max"]["morning_body"] == want_morn
        assert details["old_max"]["delivery_len"] == 875
        assert details["old_max"]["morning_len"] == 895
        out["stale_warnings"] = {k: {"delivery_len": v["delivery_len"], "morning_len": v["morning_len"]}
                                 for k, v in details.items()}

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        now = REAL_DATETIME(2027, 7, 15, 19, 8, tzinfo=ET)
        cutoff = REAL_DATETIME(2027, 7, 15, 19, 5, tzinfo=ET)
        daily = make_daily_pick("2027-07-15")
        picks.save_pick(daily, picks_dir)
        st = scheduler.SchedulerState(
            date="2027-07-15", schedule_fetched_at=now.isoformat(), games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
            pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        late_sent, void_sent = [], []
        with pinned_world(now), \
                patch.object(alert, "send_dm", lambda r, b: late_sent.append(b) or "m1"), \
                contextlib.redirect_stderr(io.StringIO()):
            scheduler._refuse_late_delivery(
                daily, {"bluesky": {"dm_recipient": "op.test"}}, picks_dir, st,
                "2027-07-15", "lineup", cutoff, now,
            )
        for f in (Path(tmp) / "health_state").glob("*.json"):
            f.unlink()
        picks.save_pick(daily, picks_dir)
        with pinned_world(now), \
                patch.object(alert, "send_dm", lambda r, b: void_sent.append(b) or "m2"), \
                contextlib.redirect_stderr(io.StringIO()):
            scheduler._refuse_void_game_delivery(
                daily, {"bluesky": {"dm_recipient": "op.test"}}, picks_dir, st,
                "2027-07-15", "lineup", {70001: "Postponed"},
            )
        tail5 = (
            "This pick was withheld. A later check may produce another recommendation. "
            "Use a later official pick message only while the entry window is open. "
            "A manual entry can differ from a later recommendation. "
            "The entry check may report a mismatch if it verifies the entry in time."
        )
        assert late_sent == [
            "BTS health CRITICAL alert(s):\n- [late_delivery] "
            f"LATE DELIVERY REFUSED: Steven Kwan would have been sent 3 min after the 19:05 ET cutoff. {tail5}"
        ]
        assert void_sent == [
            "BTS health CRITICAL alert(s):\n- [postponed_pick] "
            f"DELIVERY REFUSED for 2027-07-15: game 70001 is Postponed. {tail5}"
        ]
        out["refusals_verified"] = True
    return out


def check_c1_to_c13() -> dict:
    results = {}

    # C1: SW1 streak 56 on 2027-08-30, 99% candidate, shipped table skips top bin (bar ~82.5%)
    r_c1 = walk.walk_one(date(2027, 8, 30), walk.State("dated", 56, 56, "used", 0.99))
    assert r_c1["action"] == "skip" and f"{r_c1['pick_bar']:.1%}" == "82.5%"
    assert "below" not in r_c1["skip_dm"]
    assert "Candidate used in the decision: Batter 0 (T0) 99.0%, at or above the streak-56 bar (82.5%)." in r_c1["skip_dm"]
    results["C1"] = {"dm": r_c1["skip_dm"], "raw_bar": r_c1["pick_bar"]}

    # C2: SW1 streak 56 with >=74 days left (2027-07-15 -> 74 days left), every bin skips
    r_c2 = walk.walk_one(date(2027, 7, 15), walk.State("dated", 56, 56, "used", 0.96))
    assert r_c2["action"] == "skip" and r_c2["pick_bar"] is None and r_c2["days_left"] == 74
    assert "below" not in r_c2["skip_dm"] and "~80%" not in r_c2["skip_dm"]
    assert "No probability comparison is available for this check." in r_c2["skip_dm"]
    results["C2"] = r_c2["skip_dm"]

    # C3 & C4: Real run_single_check with removed 99.9% batter (C3) and no-decision cases (C4)
    c4_cases = {}
    for case in ("removed", "no_eligible", "status_failure", "no_valid_predictions", "sel_none"):
        with tempfile.TemporaryDirectory() as tmp:
            picks_dir = Path(tmp) / "picks"
            picks_dir.mkdir()
            now = REAL_DATETIME(2027, 7, 15, 16, 0, tzinfo=ET)
            cfg = walk._config(picks_dir)
            cfg["orchestrator"]["heartbeat_path"] = str(Path(tmp) / ".hb")
            frame = walk._predictions(date(2027, 7, 15), 0.76)
            frame.loc[0, "p_game_hit"] = 0.999
            if case == "no_valid_predictions":
                frame.loc[1:, "p_game_hit"] = float("nan")
            statuses = {
                70000: {"abstract": "F", "detailed": "Final"},
                70001: {"abstract": "P", "detailed": "Scheduled"},
                70002: {"abstract": "P", "detailed": "Scheduled"},
            }
            if case == "no_eligible":
                statuses = {pk: {"abstract": "P", "detailed": "Postponed"} for pk in statuses}

            def get_statuses(_d, case=case, statuses=statuses):
                if case == "status_failure":
                    raise OSError("MLB API down")
                return statuses

            err = io.StringIO()
            with pinned_world(now), \
                    patch.object(orchestrator, "run_cascade", lambda *a, f=frame: (f.copy(), "local")), \
                    patch.object(picks, "get_game_statuses_detailed", get_statuses), \
                    patch.object(scheduler, "count_new_confirmations", lambda *a, **k: 1), \
                    patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext()), \
                    contextlib.redirect_stderr(err):
                assert walk.install_state(picks_dir, date(2027, 7, 15), walk.State("dated", 10, 10, "used", 0.75)) == []
                if case == "sel_none":
                    with patch.object(orchestrator, "run_and_pick", lambda *a, **k: (frame.copy(), None, "local")):
                        ret = scheduler.run_single_check("2027-07-15", list(statuses), set(), cfg, 0.03)
                else:
                    ret = scheduler.run_single_check("2027-07-15", list(statuses), set(), cfg, 0.03)
            summary = ret["skip_summary"]
            dm_text = scheduler.format_skip_dm("2027-07-15", summary)
            j_lines = [ln for ln in err.getvalue().splitlines() if ln.startswith("  SKIP —")]
            if case == "removed":
                assert "Batter 0" not in dm_text and "99.9%" not in dm_text
                assert "Candidate used in the decision: Batter 1 (T1) 75.0%, below the streak-10 bar (" in dm_text
                assert "using unrounded values." in dm_text
                results["C3"] = {"dm": dm_text, "journal": j_lines[0]}
            else:
                assert "No candidate was chosen. Top model prediction (eligibility unverified): Batter 0 (T0) 99.9%." in dm_text
                assert "bar" not in dm_text
                c4_cases[case] = dm_text
    results["C4"] = c4_cases

    # C5 (same-day): MLB used saver, flag says active -> saver-state --use -> next automatic day counts saver spent
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        d1 = date(2027, 7, 15)
        d2 = date(2027, 7, 16)
        with pinned_world(REAL_DATETIME(2027, 7, 15, 12, 0, tzinfo=ET)):
            assert walk.install_state(picks_dir, d1, walk.State("automatic", 14, 14, "active", 0.79)) == []
            c_show, out_show = invoke_cli("saver-state", "--picks-dir", picks_dir)
            assert c_show == 0 and "saver_state: active (season 2027)" in out_show
            c_use, out_use = invoke_cli("saver-state", "--use", "--picks-dir", picks_dir)
            assert c_use == 0 and "saver_state: used (season 2027)" in out_use
        frame79 = walk._predictions(d2, 0.79)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame79["game_pk"]}
        with pinned_world(REAL_DATETIME(2027, 7, 16, 12, 0, tzinfo=ET)), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame79.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses):
            _, sel_after_use, _ = orchestrator.run_and_pick(walk._config(picks_dir), d2.isoformat())
        assert sel_after_use.action == "skip" and sel_after_use.saver_available is False
        results["C5_same_day"] = {
            "saver_after_use": "used",
            "next_day_action": sel_after_use.action,
            "next_day_bar": sel_after_use.decision.pick_bar,
        }

    # ADVERSARIAL REPRODUCTION 1 (C5 + C6 combined: night-game delivery stale warning read next morning)
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        t_deliv = REAL_DATETIME(2027, 7, 14, 21, 25, tzinfo=ET)
        sent_warn = []
        with pinned_world(t_deliv):
            assert walk.install_state(picks_dir, date(2027, 7, 14), walk.State("automatic", 12, 12, "active", 0.85)) == []
            auto_p = picks_dir / "account_state" / "contest_streak.json"
            auto_d = json.loads(auto_p.read_text())
            auto_d["recorded_at"] = "2027-07-14T01:10:00Z"
            auto_p.write_text(json.dumps(auto_d))
            st_14 = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=t_deliv)
            rec_14 = contest_state.automatic_reading_record(st_14, t_deliv)
            assert rec_14["stale"] is True
            with patch.object(alert, "send_dm", lambda r, b: sent_warn.append(b) or "w1"), \
                    contextlib.redirect_stderr(io.StringIO()):
                scheduler._warn_stale_contest_reading(rec_14, {"bluesky": {"dm_recipient": "op.test"}}, picks_dir)
        assert "On 2027-07-14 ET away from midnight, if MLB shows this season's saver used" in sent_warn[0]
        assert "For a later date, see B8a." in sent_warn[0]

        # 2027-07-15 01:10 ET: scheduled fetch-contest-streak recovers (streak=13, best=13); saver_state stays active!
        t_fetch15 = REAL_DATETIME(2027, 7, 15, 1, 10, tzinfo=ET)
        with pinned_world(t_fetch15), walk._fetch_stubs(13, 13, date(2027, 7, 14)):
            c_f, _ = invoke_cli("fetch-contest-streak", "--picks-dir", picks_dir, "--expected-username", walk.ACCOUNT)
            assert c_f == 0
        assert saver_state.load_saver_state(picks_dir, season=2027).state == "active"

        # 2027-07-15 08:30 ET: operator reads 2027-07-14 warning; MLB shows streak=13, best=13, saver=USED.
        t_morn15 = REAL_DATETIME(2027, 7, 15, 8, 30, tzinfo=ET)
        with pinned_world(t_morn15):
            c_st, out_st = invoke_cli("season-contract", "status", "--picks-dir", picks_dir)
            assert c_st == 0 and "saver: active; season: 2027" in out_st
            c_set, _ = invoke_cli(
                "set-contest-streak", "--date", "2027-07-15",
                "--streak", "13", "--best-streak", "13", "--saver", "used", "--picks-dir", picks_dir,
            )
            assert c_set == 0

        # 2027-07-16 01:10 ET: fetch updates streak=14, best=14; 2027-07-15 dated observation expired;
        # morning check and health check on 2027-07-16 are silent; run_and_pick picks SINGLE instead of SKIP!
        t_fetch16 = REAL_DATETIME(2027, 7, 16, 1, 10, tzinfo=ET)
        with pinned_world(t_fetch16), walk._fetch_stubs(14, 14, date(2027, 7, 15)):
            assert invoke_cli("fetch-contest-streak", "--picks-dir", picks_dir, "--expected-username", walk.ACCOUNT)[0] == 0
        t_play16 = REAL_DATETIME(2027, 7, 16, 10, 0, tzinfo=ET)
        morn_alerts_16 = []
        frame79_16 = walk._predictions(date(2027, 7, 16), 0.79)
        statuses16 = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame79_16["game_pk"]}
        with pinned_world(t_play16), \
                patch.object(alert, "send_dm", lambda r, b: morn_alerts_16.append(b) or "m"), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame79_16.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses16), \
                contextlib.redirect_stderr(io.StringIO()):
            scheduler._morning_state_check(
                {"bluesky": {"dm_recipient": "op.test"}}, picks_dir, "2027-07-16",
                t_play16, t_play16 + timedelta(hours=2),
            )
            hc_alerts_16 = health_contest_state.check(picks_dir, expected=True, now=t_play16)
            _, sel_bug_16, _ = orchestrator.run_and_pick(walk._config(picks_dir), "2027-07-16")
        assert morn_alerts_16 == []
        assert [a for a in hc_alerts_16 if a.level in ("WARN", "CRITICAL")] == []
        assert sel_bug_16.saver_available is True and sel_bug_16.action == "single"
        results["BLOCK_CANDIDATE_1_later_date_saver"] = {
            "warning_date_D": "2027-07-14",
            "read_at": "2027-07-15T08:30:00-04:00",
            "saver_flag_on_2027_07_16": saver_state.load_saver_state(picks_dir, season=2027).state,
            "morning_alerts_2027_07_16": morn_alerts_16,
            "health_warn_crit_2027_07_16": [a.message for a in hc_alerts_16 if a.level in ("WARN", "CRITICAL")],
            "action_on_2027_07_16_with_phantom_saver": sel_bug_16.action,
            "action_on_2027_07_16_with_spent_saver": results["C5_same_day"]["next_day_action"],
            "bar_with_spent_saver": results["C5_same_day"]["next_day_bar"],
        }

    # C6 + ADVERSARIAL REPRODUCTION 2 (B8a "For a later date" from wrong CWD)
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "real_box" / "data" / "picks"
        picks_dir.mkdir(parents=True)
        wrong_cwd = Path(tmp) / "operator_home"
        wrong_cwd.mkdir()
        with pinned_world(REAL_DATETIME(2027, 7, 14, 21, 0, tzinfo=ET)):
            assert walk.install_state(picks_dir, date(2027, 7, 14), walk.State("automatic", 0, 0, "not_earned", 0.80)) == []
        with pinned_world(REAL_DATETIME(2027, 7, 15, 0, 5, tzinfo=ET)):
            c_old_d, out_old_d = invoke_cli(
                "set-contest-streak", "--date", "2027-07-14",
                "--streak", "15", "--best-streak", "15", "--saver", "used", "--picks-dir", picks_dir,
                cwd=wrong_cwd,
            )
            assert c_old_d == 2 and "must be today (2027-07-15)" in out_old_d
        with pinned_world(REAL_DATETIME(2027, 7, 15, 8, 30, tzinfo=ET)):
            c_b4, out_b4 = invoke_cli(
                "set-contest-streak", "--date", "2027-07-15",
                "--streak", "15", "--best-streak", "15", "--saver", "used",
                cwd=wrong_cwd,
            )
            assert c_b4 == 0 and "in effect today" in out_b4
            assert (wrong_cwd / "data/picks/account_state/contest_streak.manual.json").is_file()
            assert not (picks_dir / "account_state/contest_streak.manual.json").exists()
            frame80 = walk._predictions(date(2027, 7, 15), 0.80)
            frame80.loc[1, "p_game_hit"] = 0.80
            statuses15 = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame80["game_pk"]}
            with patch.object(orchestrator, "run_cascade", lambda *a: (frame80.copy(), "local")), \
                    patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses15):
                _, sel_wrong_dir, _ = orchestrator.run_and_pick(walk._config(picks_dir), "2027-07-15")
            c_right, _ = invoke_cli(
                "set-contest-streak", "--date", "2027-07-15",
                "--streak", "15", "--best-streak", "15", "--saver", "used", "--picks-dir", picks_dir,
                cwd=wrong_cwd,
            )
            assert c_right == 0
            with patch.object(orchestrator, "run_cascade", lambda *a: (frame80.copy(), "local")), \
                    patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses15):
                _, sel_right_dir, _ = orchestrator.run_and_pick(walk._config(picks_dir), "2027-07-15")
        assert sel_wrong_dir.action != sel_right_dir.action
        results["C6_and_BLOCK_CANDIDATE_2_wrong_cwd"] = {
            "old_date_exit": c_old_d,
            "b8a_bullet4_without_picks_dir_exit": c_b4,
            "scheduler_action_after_wrong_cwd": sel_wrong_dir.action,
            "scheduler_action_with_explicit_picks_dir": sel_right_dir.action,
        }

    # Saver matrix + Jan 1 rollover + post-delivery grading check
    with tempfile.TemporaryDirectory() as tmp:
        saver_cases = {}
        for idx, prior in enumerate(("active", "not_earned", "used", "uninitialized")):
            pdir = Path(tmp) / f"p_{idx}"
            pdir.mkdir()
            with pinned_world(REAL_DATETIME(2027, 7, 15, 14, 0, tzinfo=ET)):
                if prior != "uninitialized":
                    invoke_cli("saver-state", "--init", prior, "--season", "2027", "--picks-dir", pdir)
                if prior in ("active", "not_earned"):
                    code, out = invoke_cli("saver-state", "--use", "--picks-dir", pdir)
                elif prior == "used":
                    code, out = invoke_cli("saver-state", "--picks-dir", pdir)
                else:
                    code, out = invoke_cli("saver-state", "--init", "used", "--season", "2027", "--picks-dir", pdir)
                saver_cases[prior] = {
                    "exit": code,
                    "final_state": saver_state.load_saver_state(pdir, season=2027).state,
                    "out_first": out.strip().splitlines()[0],
                }
                assert code == 0 and saver_cases[prior]["final_state"] == "used"
        p_jan = Path(tmp) / "p_jan"
        p_jan.mkdir()
        with pinned_world(REAL_DATETIME(2027, 12, 31, 20, 0, tzinfo=ET)):
            invoke_cli("saver-state", "--init", "active", "--season", "2027", "--picks-dir", p_jan)
        with pinned_world(REAL_DATETIME(2028, 1, 1, 0, 5, tzinfo=ET)):
            c_jan, out_jan = invoke_cli("saver-state", "--use", "--picks-dir", p_jan)
            assert c_jan == 1 and "season 2028" in out_jan
            assert saver_state.load_saver_state(p_jan, season=2027).state == "active"

        # Grading check: post-delivery set-contest-streak + saver-state --use leaves check-results identical
        p_grade = Path(tmp) / "p_grade"
        p_grade.mkdir()
        dp_g = make_daily_pick("2027-07-15")
        dp_g.notification_sent = True
        dp_g.notification_id = "g1"
        picks.save_pick(dp_g, p_grade)
        daily_decision.write_decision(
            "2027-07-15", p_grade, action="single", source="mdp",
            primary={"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "game_pk": 70001, "p_game_hit": 0.82},
            double_down=None, delivery_status="delivered", scoreable=True,
        )
        with pinned_world(REAL_DATETIME(2027, 7, 15, 20, 0, tzinfo=ET)):
            invoke_cli("saver-state", "--init", "active", "--season", "2027", "--picks-dir", p_grade)
            invoke_cli("set-contest-streak", "--date", "2027-07-15", "--streak", "12", "--best-streak", "12", "--saver", "used", "--picks-dir", p_grade)
            invoke_cli("saver-state", "--use", "--picks-dir", p_grade)
        with pinned_world(REAL_DATETIME(2027, 7, 16, 1, 0, tzinfo=ET)), \
                patch.object(picks, "resolve_daily_slot_results", lambda *a: {"pick": "hit"}):
            c_gr, out_gr = invoke_cli("check-results", "--date", "2027-07-15", "--picks-dir", p_grade)
        assert c_gr == 0 and picks.load_pick("2027-07-15", p_grade).result == "hit"
        results["saver_matrix_jan1_grading"] = {"saver_cases": saver_cases, "grading_exit": c_gr, "grading_out": out_gr.strip()}

    # C7: Refusal while a later game can still be picked + entry check window behavior
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        daily = make_daily_pick("2027-07-15", batter="Player Late", game_pk=70001, game_hour_et=19, game_min_et=10)
        later = make_daily_pick("2027-07-15", batter="Player Later", batter_id=660271, game_pk=70002, game_hour_et=22, game_min_et=10)
        now_late = REAL_DATETIME(2027, 7, 15, 19, 7, tzinfo=ET)
        cutoff = REAL_DATETIME(2027, 7, 15, 19, 5, tzinfo=ET)
        st = scheduler.SchedulerState(
            date="2027-07-15", schedule_fetched_at=now_late.isoformat(), games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
            pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        picks.save_pick(daily, picks_dir)
        ref_dms = []
        with pinned_world(now_late), \
                patch.object(alert, "send_dm", lambda r, b: ref_dms.append(b) or "r1"), \
                contextlib.redirect_stderr(io.StringIO()):
            scheduler._refuse_late_delivery(
                daily, {"bluesky": {"dm_recipient": "op.test"}}, picks_dir, st,
                "2027-07-15", "lineup", cutoff, now_late,
            )
        assert "manually if you want" not in ref_dms[0]
        assert picks.load_pick("2027-07-15", picks_dir) is None
        t_later_deliv = REAL_DATETIME(2027, 7, 15, 21, 25, tzinfo=ET)
        later.notification_sent = True
        later.notification_id = "dm_ok"
        later.delivered_at = t_later_deliv.isoformat()
        picks.save_pick(later, picks_dir)
        daily_decision.write_decision(
            "2027-07-15", picks_dir, action="single", source="mdp",
            primary={"batter_id": 660271, "batter_name": "Player Later", "team": "CLE", "game_pk": 70002, "p_game_hit": 0.82},
            double_down=None, delivery_status="delivered", scoreable=True,
        )
        entry_dms = []
        sess = SimpleNamespace(user_id=42, username="op", xsid="xs")
        with pinned_world(REAL_DATETIME(2027, 7, 15, 21, 30, tzinfo=ET)), \
                patch.object(auth, "load_session_cookies", lambda: {}), \
                patch.object(auth, "extract_uid", lambda c: 42), \
                patch.object(auth, "fetch_login_session", lambda **k: sess), \
                patch.object(contest_fetch, "fetch_profile", lambda *a: {"predictions": []}), \
                patch.object(contest_fetch, "fetch_pending_predictions", lambda *a: [{"roundId": 1, "playerId": 999}]), \
                patch.object(cli_mod, "_fetch_rounds", lambda: {1: date(2027, 7, 15)}), \
                patch.object(cli_mod, "_fetch_bts_to_mlb", lambda: {999: 111111}), \
                patch.object(dm, "send_dm", lambda r, b: entry_dms.append(b) or "e1"):
            c_ent, _ = invoke_cli("check-pick-entered", "--picks-dir", picks_dir, "--dm-recipient", "op.test")
        assert c_ent == 1 and "does NOT match the recommended pick" in entry_dms[0]
        with pinned_world(REAL_DATETIME(2027, 7, 15, 22, 6, tzinfo=ET)):
            c_out, out_txt = invoke_cli("check-pick-entered", "--picks-dir", picks_dir, "--dm-recipient", "op.test")
        assert c_out == 0 and "outside window" in out_txt
        results["C7"] = {"refusal_dm": ref_dms[0], "entry_mismatch_dm": entry_dms[0]}

    # C8 & C9: Morning warning at 00:20 ET (C8) and filing before guard vs during send after first check started (C9)
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        t_0020 = REAL_DATETIME(2027, 7, 15, 0, 20, tzinfo=ET)
        morn_dm = []
        with pinned_world(t_0020):
            assert walk.install_state(picks_dir, date(2027, 7, 15), walk.State("automatic", 0, 0, "not_earned", 0.80)) == []
            auto_p = picks_dir / "account_state" / "contest_streak.json"
            auto_d = json.loads(auto_p.read_text())
            auto_d["recorded_at"] = "2027-07-14T01:00:00Z"
            auto_p.write_text(json.dumps(auto_d))
            with patch.object(alert, "send_dm", lambda r, b: morn_dm.append(b) or "m8"), \
                    contextlib.redirect_stderr(io.StringIO()):
                scheduler._morning_state_check(
                    {"bluesky": {"dm_recipient": "op.test"}}, picks_dir, "2027-07-15",
                    t_0020, REAL_DATETIME(2027, 7, 15, 18, 25, tzinfo=ET),
                )
        assert "Before the first check, if prior entries are settled or absent and MLB differs from status (B8a)" in morn_dm[0]
        results["C8"] = morn_dm[0]

        t_chk = REAL_DATETIME(2027, 7, 15, 18, 25, tzinfo=ET)
        cfg = walk._config(picks_dir)
        cfg["bluesky"] = {"dm_recipient": "op.test"}
        frame80 = walk._predictions(date(2027, 7, 15), 0.80)
        frame80.loc[1, "p_game_hit"] = 0.80
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame80["game_pk"]}
        with pinned_world(t_chk), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame80.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses):
            _, sel_c9, _ = orchestrator.run_and_pick(cfg, "2027-07-15")
            invoke_cli("set-contest-streak", "--date", "2027-07-15", "--streak", "15", "--best-streak", "15",
                       "--saver", "used", "--picks-dir", picks_dir)
            st_c9 = scheduler.SchedulerState(
                date="2027-07-15", schedule_fetched_at=t_chk.isoformat(), games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            sent_picks_before = []
            with patch.object(dm, "send_dm", lambda r, b: sent_picks_before.append(b) or "p1"), \
                    patch.object(alert, "send_dm", lambda r, b: "a1"), \
                    patch.object(scheduler, "_delivery_game_statuses", lambda _d: statuses), \
                    contextlib.redirect_stderr(io.StringIO()):
                ok_before = scheduler._deliver_and_lock_pick(
                    sel_c9.pick_result.daily, cfg, picks_dir, st_c9, "2027-07-15", "lineup", selection=sel_c9
                )
        assert ok_before is False and sent_picks_before == []

        (picks_dir / "account_state" / "contest_streak.manual.json").unlink()
        sent_picks_during = []

        def send_and_race(recipient, body):
            invoke_cli("set-contest-streak", "--date", "2027-07-15", "--streak", "15", "--best-streak", "15",
                       "--saver", "used", "--picks-dir", picks_dir)
            sent_picks_during.append(body)
            return "p2"

        with pinned_world(t_chk), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame80.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses), \
                patch.object(scheduler, "_delivery_game_statuses", lambda _d: statuses), \
                patch.object(scheduler, "_trigger_live_forward_capture_on_lock", lambda *a: None), \
                patch.object(dm, "send_dm", send_and_race), \
                patch.object(alert, "send_dm", lambda r, b: "a2"), \
                contextlib.redirect_stderr(io.StringIO()):
            _, sel_c9b, _ = orchestrator.run_and_pick(cfg, "2027-07-15")
            st_c9b = scheduler.SchedulerState(
                date="2027-07-15", schedule_fetched_at=t_chk.isoformat(), games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            ok_during = scheduler._deliver_and_lock_pick(
                sel_c9b.pick_result.daily, cfg, picks_dir, st_c9b, "2027-07-15", "lineup", selection=sel_c9b
            )
        assert ok_during is True and len(sent_picks_during) == 1
        results["C9"] = {"stale_action": sel_c9.action, "withheld_before_guard": not ok_before, "delivered_during_send": ok_during}

    # C10
    s_c10 = {"best_batter": "A", "best_team": "T", "best_p": 1e308, "bar": 1.5e308, "streak": 10,
             "basis": "decision", "candidate_verified": True}
    dm_c10 = scheduler.format_skip_dm("2027-07-15", s_c10)
    ban_c10 = web.render_skip_banner(s_c10)
    assert "inf" not in dm_c10.lower() and "inf" not in ban_c10.lower()
    assert "Candidate used in the decision: A (T) ?. No probability comparison is available for this check. Streak used at this check: 10." in dm_c10
    results["C10"] = dm_c10

    # C11
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        t_1835 = REAL_DATETIME(2027, 7, 15, 18, 35, tzinfo=ET)
        t_1855 = REAL_DATETIME(2027, 7, 15, 18, 55, tzinfo=ET)
        cfg = walk._config(picks_dir)
        cfg["orchestrator"]["heartbeat_path"] = str(Path(tmp) / ".hb")
        frame72 = walk._predictions(date(2027, 7, 15), 0.72)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame72["game_pk"]}
        st = scheduler.SchedulerState(
            date="2027-07-15", schedule_fetched_at=t_1835.isoformat(), games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
            pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        with pinned_world(t_1835), \
                patch.object(orchestrator, "run_cascade", lambda *a: (frame72.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda _d: statuses), \
                patch.object(scheduler, "count_new_confirmations", lambda *a, **k: 1), \
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext()), \
                contextlib.redirect_stderr(io.StringIO()):
            walk.install_state(picks_dir, date(2027, 7, 15), walk.State("dated", 15, 15, "used", 0.72))
            r1 = scheduler.run_single_check("2027-07-15", list(statuses), set(), cfg, 0.03)
            assert r1["skip_summary"] is not None
            st.skip_summary = r1["skip_summary"]
            scheduler.save_state(st, picks_dir)
        with pinned_world(t_1855), \
                patch.object(orchestrator, "run_cascade", lambda *a: (None, None)), \
                patch.object(scheduler, "count_new_confirmations", lambda *a, **k: 0), \
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext()), \
                patch.object(web, "PICKS_DIR", picks_dir), \
                patch.object(web, "HEALTH_STATE_DIR", Path(tmp) / "hs"), \
                patch.object(web, "VALIDATION_DIR", Path(tmp) / "val"), \
                patch.object(web, "fetch_bluesky_posts", lambda *a, **k: []), \
                contextlib.redirect_stderr(io.StringIO()):
            r2 = scheduler.run_single_check("2027-07-15", list(statuses), set(), cfg, 0.03)
            assert r2["skip_summary"] is None
            if r2.get("skip_summary"):
                st.skip_summary = r2["skip_summary"]
            scheduler.save_state(st, picks_dir)
            page_html = web.render_page()
        assert "NO PICK AT LAST CHECK" not in page_html
        assert "LAST RECORDED NO-PICK SUMMARY" in page_html
        assert "Batter 0 (T0) 72.0%" in page_html
        results["C11"] = "LAST RECORDED NO-PICK SUMMARY present; NO PICK AT LAST CHECK absent"

    # C12
    class RaisingClassCandidate:
        @property
        def __class__(self):
            raise RuntimeError("boom on __class__")

    df_c12 = pd.DataFrame([{"batter_name": "Top", "team": "NYY", "p_game_hit": 0.75}])
    dec_c12 = strategy.PolicyDecision(
        "skip", "skip", "mdp", "tail_season_best", 0, 8, 18, "trusted", 18, "sha", None, "\n", None
    )
    sum_c12 = scheduler.build_skip_summary(df_c12, 0, decision=dec_c12, candidate=RaisingClassCandidate())
    assert sum_c12 is not None and sum_c12["candidate_verified"] is False and sum_c12["best_batter"] == "Top"
    rendered_c12 = scheduler.format_skip_summary(sum_c12)
    assert "Reason:" not in rendered_c12
    assert "No probability comparison is available for this check." in rendered_c12
    results["C12"] = rendered_c12

    # C13
    bad_s = {"best_batter": "<b>X</b>\nY", "best_team": "<script>", "best_p": "not_num",
             "bar": [1], "streak": True, "reason": " \n ", "basis": 123}
    ban_c13 = web.render_skip_banner(bad_s)
    assert "<b>" not in ban_c13 and "<script>" not in ban_c13 and "&lt;b&gt;X&lt;/b&gt; Y" in ban_c13
    assert "below the" not in ban_c13
    results["C13"] = ban_c13

    return results


if __name__ == "__main__":
    exact = check_exact_contract_strings()
    c_table = check_c1_to_c13()
    out_path = Path(__file__).with_name("probe_c1_c13.json")
    out_path.write_text(json.dumps({"exact_contract": exact, "c1_c13": c_table}, indent=2))
    print("probe_c1_c13_v2: ALL CHECKS COMPLETED")
    print("BLOCK_CANDIDATE_1:", json.dumps(c_table["BLOCK_CANDIDATE_1_later_date_saver"]))
    print("BLOCK_CANDIDATE_2:", json.dumps(c_table["C6_and_BLOCK_CANDIDATE_2_wrong_cwd"]))
