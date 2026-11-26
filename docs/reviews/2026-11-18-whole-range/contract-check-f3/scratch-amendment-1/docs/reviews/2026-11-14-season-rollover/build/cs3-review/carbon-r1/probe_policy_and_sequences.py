#!/usr/bin/env python3
"""Adversarial probe for CS3 review: items 2 (policy sweep + first week) and 3 (operator sequences)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base" / "src"
BRANCH_SRC = REPO / "src"
PYTHON = REPO / ".venv" / "bin" / "python"

POLICY_WORKER = r'''
import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ["TZ"] = "America/New_York"
import time
time.tzset()

mode = sys.argv[1]
out_file = Path(sys.argv[2])

import bts.contest_state as cs
import bts.saver_state as ss
import bts.strategy as strat
from bts.cli import cli
from click.testing import CliRunner

ET = ZoneInfo("America/New_York")
NOW_2027 = datetime(2027, 7, 15, 16, 0, tzinfo=timezone.utc)

# Ensure real checked-in mdp_policy.npz and mdp_tail_policy.npz are loaded
mdp = strat._load_mdp()
assert mdp is not None and mdp["policy_table"] is not None and mdp["tail"] is not None, f"MDP failed: {mdp}"
assert mdp["tail_error"] is None and mdp["base_error"] is None

boundaries = sorted(set(
    [round(0.70 + 0.01 * i, 4) for i in range(21)]
    + [round(float(b) - 1e-6, 6) for b in mdp["boundaries"] if 0.70 <= float(b) <= 0.90]
    + [round(float(b), 6) for b in mdp["boundaries"] if 0.70 <= float(b) <= 0.90]
    + [round(float(b) + 1e-6, 6) for b in mdp["boundaries"] if 0.70 <= float(b) <= 0.90]
))

# 1. Through REAL load_decision_streak_state: map (flag_kind, contest_kind, streak) -> saver_available
flag_kinds = ["missing", "2026:not_earned", "2026:active", "2026:used", "2027:not_earned", "2027:active", "2027:used"]
contest_kinds = ["2026_obs", "2027_obs", "model_only"]

state_bit_map = {}
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    (picks / "account_state").mkdir(parents=True, exist_ok=True)
    for fk in flag_kinds:
        fp = picks / "account_state" / "saver_state.json"
        fp.unlink(missing_ok=True)
        if fk != "missing":
            yr_s, st_s = fk.split(":")
            fp.write_text(json.dumps({"season": int(yr_s), "state": st_s}))
        for ck in contest_kinds:
            cp = picks / "account_state" / "contest_streak.json"
            sp = picks / "streak.json"
            cp.unlink(missing_ok=True)
            sp.unlink(missing_ok=True)
            for s in range(58):
                if ck == "model_only":
                    sp.write_text(json.dumps({"season": 2027, "streak": s, "saver_available": True}))
                else:
                    src_d = "2026-09-28" if ck == "2026_obs" else "2027-06-18"
                    cp.write_text(json.dumps({
                        "schema_version": "bts_contest_streak_auto_v1",
                        "active_streak": s,
                        "best_streak": s,
                        "source": "mlb_bts_profile",
                        "source_date": src_d,
                    }))
                ds = cs.load_decision_streak_state(picks, now=NOW_2027)
                assert ds.streak == s
                state_bit_map[f"{fk}|{ck}|{s}"] = {
                    "saver": ds.saver_available,
                    "flag_state": getattr(ds, "saver_flag_state", None),
                    "best_status": ds.best_status,
                }

# 2. Through REAL resolve_policy_decision: evaluate (date, streak, prob, has_diff, saver=False vs True)
# across ALL 365 dates of 2027 and streaks 0..57!
start_2027 = date(2027, 1, 1)
all_dates_2027 = [(start_2027 + timedelta(days=i)).isoformat() for i in range(365)]

# Because resolve_policy_decision depends on (days_remaining(date), streak, saver, primary_p, second_p, has_diff, best),
# group 2027 dates by _days_remaining(date) so we still verify all 365 dates exactly!
days_to_dates = {}
for d_iso in all_dates_2027:
    dr = strat._days_remaining(d_iso)
    days_to_dates.setdefault(dr, []).append(d_iso)

saver_flip_cells = []  # cells where action(saver=False) != action(saver=True)
for dr, d_list in sorted(days_to_dates.items()):
    rep_date = d_list[0]
    for s in range(58):
        for has_diff in (True, False):
            for p in boundaries:
                ctx = strat.DecisionContext(
                    primary_p=p,
                    second_p=p if has_diff else None,
                    has_diff_game=has_diff,
                    date=rep_date,
                    allow_double=True,
                    mdp=mdp,
                    best_streak=s,
                    best_status="trusted",
                )
                a_false = strat.resolve_policy_decision(ctx, s, False)
                a_true = strat.resolve_policy_decision(ctx, s, True)
                if a_false.action != a_true.action:
                    saver_flip_cells.append({
                        "days_remaining": dr,
                        "rep_date": rep_date,
                        "num_dates": len(d_list),
                        "first_date": d_list[0],
                        "last_date": d_list[-1],
                        "streak": s,
                        "has_diff": has_diff,
                        "p": p,
                        "objective": a_true.objective,
                        "action_saver_false": a_false.action,
                        "action_saver_true": a_true.action,
                    })

# Also check Week 1 day-by-day (2027-03-24 .. 2027-03-30) under all 5 initial flag setups:
# 1) 2026:not_earned, 2) 2026:active, 3) 2026:used, 4) no flag file, 5) after bts saver-state --init not_earned --season 2027
from bts.health import contest_state as hcs
week1_dates = [date(2027, 3, 24) + timedelta(days=i) for i in range(7)]
week1_scenarios = {}
for scen in ("2026:not_earned", "2026:active", "2026:used", "no_file", "init_not_earned_2027"):
    # Sub-scenario A: before any 2027 settled fetch (e.g. 2026 contest_streak.json at streak 6 still on disk, or manual streak climbing 0..6 without settled auto fetch)
    # Sub-scenario B: realistic daily settled fetch climbing streak 0, 1, 2, 3, 4, 5, 6
    for sub in ("unsettled_with_last_year_obs_s6", "daily_settled_fetch_climbing_0_to_6"):
        with tempfile.TemporaryDirectory() as td:
            picks = Path(td)
            (picks / "account_state").mkdir(parents=True, exist_ok=True)
            fp = picks / "account_state" / "saver_state.json"
            if scen.startswith("2026:"):
                fp.write_text(json.dumps({"season": 2026, "state": scen.split(":")[1]}))
            elif scen == "init_not_earned_2027":
                CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks), "--season", "2027", "--init", "not_earned"])
            days_out = []
            for idx, d_cur in enumerate(week1_dates):
                now_cur = datetime(d_cur.year, d_cur.month, d_cur.day, 15, 0, tzinfo=timezone.utc)
                cp = picks / "account_state" / "contest_streak.json"
                if sub == "unsettled_with_last_year_obs_s6":
                    cp.write_text(json.dumps({
                        "schema_version": "bts_contest_streak_auto_v1",
                        "active_streak": 6, "best_streak": 18,
                        "source": "mlb_bts_profile", "source_date": "2026-09-28",
                    }))
                else:
                    # Day 1 (idx=0): Opening Night pre-game, no settled 2027 round yet -> source_date None, streak 0
                    # Day 2..7 (idx=1..6): yesterday's pick settled as hit -> streak = idx, best = idx, source_date = yesterday
                    if idx == 0:
                        cp.write_text(json.dumps({
                            "schema_version": "bts_contest_streak_auto_v1",
                            "active_streak": 0, "best_streak": 0,
                            "source": "mlb_bts_profile", "source_date": None,
                            "recorded_at": now_cur.isoformat(),
                        }))
                        if mode == "base":
                            ss.maybe_auto_earn_saver(picks, best_streak=0, season=ss.season_for(None, now_year=2027))
                        # On branch, null source_date does NOT call maybe_auto_earn_saver
                    else:
                        prev_d = d_cur - timedelta(days=1)
                        cp.write_text(json.dumps({
                            "schema_version": "bts_contest_streak_auto_v1",
                            "active_streak": idx, "best_streak": idx,
                            "source": "mlb_bts_profile", "source_date": prev_d.isoformat(),
                            "recorded_at": now_cur.isoformat(),
                        }))
                        ss.maybe_auto_earn_saver(picks, best_streak=idx, season=2027)
                ds = cs.load_decision_streak_state(picks, now=now_cur)
                ctx80 = strat.DecisionContext(0.80, 0.80, True, d_cur.isoformat(), True, mdp, ds.best_streak, ds.best_status)
                act80 = strat.resolve_policy_decision(ctx80, ds.streak, ds.saver_available).action
                alerts = hcs.check(picks, expected=True, now=now_cur)
                days_out.append({
                    "date": d_cur.isoformat(),
                    "streak": ds.streak,
                    "flag_2027": ss.load_saver_state(picks, season=2027).state,
                    "policy_saver": ds.saver_available,
                    "action_at_0_80": act80,
                    "health_alerts": [(a.level, a.source) for a in alerts],
                })
            week1_scenarios[f"{scen}__{sub}"] = days_out

out_file.write_text(json.dumps({
    "mode": mode,
    "boundaries": mdp["boundaries"],
    "state_bit_map": state_bit_map,
    "saver_flip_cells": saver_flip_cells,
    "week1_scenarios": week1_scenarios,
}, indent=2))
'''


SEQUENCE_WORKER = r'''
import json
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ["TZ"] = "America/New_York"
import time
time.tzset()

out_file = Path(sys.argv[1])

import bts.contest_state as cs
import bts.saver_state as ss
import bts.strategy as strat
import bts.web as web
from bts.cli import cli
from bts.health import contest_state as hcs
from click.testing import CliRunner
from pytest import MonkeyPatch
from tests.test_contest_fetch import _mock_fetch

ET = ZoneInfo("America/New_York")
mdp = strat._load_mdp()


def snapshot_step(picks: Path, now_utc: datetime, date_iso: str, label: str) -> dict:
    ds = cs.load_decision_streak_state(picks, now=now_utc)
    ctx80 = strat.DecisionContext(0.80, 0.80, True, date_iso, True, mdp, ds.best_streak, ds.best_status)
    ctx82 = strat.DecisionContext(0.82, 0.82, True, date_iso, True, mdp, ds.best_streak, ds.best_status)
    dec80 = strat.resolve_policy_decision(ctx80, ds.streak, ds.saver_available)
    dec82 = strat.resolve_policy_decision(ctx82, ds.streak, ds.saver_available)
    alerts = hcs.check(picks, expected=True, now=now_utc)
    dctx = web.saver_dashboard_context(picks, now=now_utc)
    html = web.render_saver_section(dctx, decision_streak=ds.streak)
    audit_p = picks / "account_state" / "saver_transitions.jsonl"
    audits = [json.loads(x) for x in audit_p.read_text().splitlines()] if audit_p.exists() else []
    raw_flag_p = picks / "account_state" / "saver_state.json"
    raw_flag = json.loads(raw_flag_p.read_text()) if raw_flag_p.exists() else None
    return {
        "step": label,
        "date": date_iso,
        "streak": ds.streak,
        "best_streak": ds.best_streak,
        "saver_flag_state": ds.saver_flag_state,
        "raw_flag_season": raw_flag.get("season") if raw_flag else None,
        "policy_saver": ds.saver_available,
        "action_p80": dec80.action,
        "action_p82": dec82.action,
        "health": [(a.level, a.source, a.message) for a in alerts],
        "dash_state": dctx.state,
        "dash_button": dctx.button,
        "dash_nudge": dctx.nudge,
        "dash_warning": dctx.warning,
        "dash_html": html,
        "audit_len": len(audits),
    }


def run_real_fetch(picks: Path, now_et: datetime, active: int, best: int, predictions: list, rounds: dict):
    import bts.cli as climod
    mp = MonkeyPatch()
    _mock_fetch(mp, {"activeStreak": active, "seasonBestStreak": best, "predictions": predictions}, rounds)
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_et.astimezone(tz) if tz else now_et.replace(tzinfo=None)
    mp.setattr("datetime.datetime", FakeDT)
    try:
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks), "--expected-username", "stonehengee"])
        assert r.exit_code == 0, r.output
        return r.output
    finally:
        mp.undo()


steps = []
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    (picks / "account_state").mkdir(parents=True, exist_ok=True)
    # Step 1: Season start (2027-03-24) with last year's (2026) `used` flag and 2026 contest file on disk
    (picks / "account_state" / "saver_state.json").write_text(json.dumps({
        "season": 2026, "state": "used", "source": "cli", "updated_at": "2026-07-01T00:00:00+00:00",
    }))
    (picks / "account_state" / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 4, "best_streak": 18,
        "source": "mlb_bts_profile", "source_date": "2026-09-28",
        "recorded_at": "2026-09-29T06:00:00Z", "user_id": 50311, "username": "stonehengee",
    }))
    steps.append(snapshot_step(picks, datetime(2027, 3, 24, 15, tzinfo=timezone.utc), "2027-03-24", "1_season_start_2026_used_flag"))
    # Verify dashboard POSTs (mark_used and undo) both reject with 409 on uninitialized 2027 state without clobbering 2026 file!
    b_before = (picks / "account_state" / "saver_state.json").read_bytes()
    r_mu_1 = web.saver_transition_response(picks, expected_prior="active", new_state="used", same_origin=True, now=datetime(2027, 3, 24, 15, tzinfo=timezone.utc))
    r_un_1 = web.saver_transition_response(picks, expected_prior="used", new_state="active", same_origin=True, now=datetime(2027, 3, 24, 15, tzinfo=timezone.utc))
    assert r_mu_1[0] == 409 and r_un_1[0] == 409
    assert (picks / "account_state" / "saver_state.json").read_bytes() == b_before

    # Step 2a: First 2027 fetch with NO settled round (source_date=None, active=0, best=0)
    run_real_fetch(picks, datetime(2027, 3, 24, 12, tzinfo=ET), 0, 0, [], {})
    steps.append(snapshot_step(picks, datetime(2027, 3, 24, 16, tzinfo=timezone.utc), "2027-03-24", "2a_first_fetch_no_settled_round"))

    # Step 2b: Fetch with ONE settled 2027 round (2027-03-24 hit -> active=1, best=1) -> initializes 2027 not_earned!
    run_real_fetch(picks, datetime(2027, 3, 25, 10, tzinfo=ET), 1, 1,
                   [{"roundId": 1, "result": "hit", "streak": 1}], {1: date(2027, 3, 24)})
    steps.append(snapshot_step(picks, datetime(2027, 3, 25, 15, tzinfo=timezone.utc), "2027-03-25", "2b_fetch_one_settled_round"))

    # Step 3: Streak climbing 8, 9, 10, 11 with fetch promoting not_earned -> active at 10
    for st_k, dt_d in [(8, date(2027, 4, 2)), (9, date(2027, 4, 3)), (10, date(2027, 4, 4)), (11, date(2027, 4, 5))]:
        run_real_fetch(picks, datetime(dt_d.year, dt_d.month, dt_d.day, 10, tzinfo=ET), st_k, st_k,
                       [{"roundId": st_k, "result": "hit", "streak": st_k}], {st_k: dt_d})
        steps.append(snapshot_step(
            picks, datetime(dt_d.year, dt_d.month, dt_d.day, 15, tzinfo=timezone.utc),
            dt_d.isoformat(), f"3_climb_streak_{st_k}",
        ))

    # Step 4: Miss in 10..15 zone forgiven by contest (at streak 11: two fetches confirm held not_hit at 11 -> nudge=True),
    # then `bts saver-state --use --season 2027`, then next fetch at 11 and 12 (does anything resurrect active?)
    preds_forgiven = [
        {"roundId": 11, "result": "hit", "streak": 11, "roundPredictions": [{"playerId": 1}]},
        {"roundId": 12, "result": "not_hit", "streak": 11, "roundPredictions": [{"playerId": 2}]},
    ]
    rounds_forgiven = {11: date(2027, 4, 5), 12: date(2027, 4, 6)}
    run_real_fetch(picks, datetime(2027, 4, 7, 2, tzinfo=ET), 11, 11, preds_forgiven, rounds_forgiven)
    run_real_fetch(picks, datetime(2027, 4, 7, 10, tzinfo=ET), 11, 11, preds_forgiven, rounds_forgiven)
    steps.append(snapshot_step(picks, datetime(2027, 4, 7, 15, tzinfo=timezone.utc), "2027-04-07", "4a_miss_forgiven_at_11_before_use"))

    r_use = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks), "--season", "2027", "--use"])
    assert r_use.exit_code == 0
    steps.append(snapshot_step(picks, datetime(2027, 4, 7, 16, tzinfo=timezone.utc), "2027-04-07", "4b_after_cli_use"))

    # Next fetch at streak 12, best 12 -> must NOT resurrect active!
    run_real_fetch(picks, datetime(2027, 4, 8, 10, tzinfo=ET), 12, 12,
                   preds_forgiven + [{"roundId": 13, "result": "hit", "streak": 12}],
                   {**rounds_forgiven, 13: date(2027, 4, 7)})
    steps.append(snapshot_step(picks, datetime(2027, 4, 8, 15, tzinfo=timezone.utc), "2027-04-08", "4c_next_fetch_after_use_no_resurrect"))

    # Test dashboard Undo then Mark-Used buttons on this directory!
    r_dash_undo = web.saver_transition_response(picks, expected_prior="used", new_state="active", same_origin=True, now=datetime(2027, 4, 8, 16, tzinfo=timezone.utc), peer_ip="100.64.1.2")
    assert r_dash_undo[0] == 200
    steps.append(snapshot_step(picks, datetime(2027, 4, 8, 16, tzinfo=timezone.utc), "2027-04-08", "4d_after_dashboard_undo"))
    r_dash_use = web.saver_transition_response(picks, expected_prior="active", new_state="used", same_origin=True, now=datetime(2027, 4, 8, 16, 5, tzinfo=timezone.utc), peer_ip="100.64.1.2")
    assert r_dash_use[0] == 200
    steps.append(snapshot_step(picks, datetime(2027, 4, 8, 16, 5, tzinfo=timezone.utc), "2027-04-08", "4e_after_dashboard_mark_used"))

# Sequence B: Saver earned AND consumed between two fetches while flag still says `not_earned`, then `--use`, then fetch(best=10)
seq_between_fetch = []
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    run_real_fetch(picks, datetime(2027, 5, 1, 10, tzinfo=ET), 9, 9,
                   [{"roundId": 1, "result": "hit", "streak": 9}], {1: date(2027, 4, 30)})
    seq_between_fetch.append(snapshot_step(picks, datetime(2027, 5, 1, 15, tzinfo=timezone.utc), "2027-05-01", "B1_at_9_not_earned"))
    # Dashboard cannot mark not_earned used (no button, POST 409)
    d_post_ne = web.saver_transition_response(picks, expected_prior="not_earned", new_state="used", same_origin=True, now=datetime(2027, 5, 3, 12, tzinfo=timezone.utc))
    assert d_post_ne[0] == 409
    # Operator runs `bts saver-state --use --season 2027` while flag is still `not_earned`
    r_u = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks), "--season", "2027", "--use"])
    assert r_u.exit_code == 0
    seq_between_fetch.append(snapshot_step(picks, datetime(2027, 5, 3, 12, tzinfo=timezone.utc), "2027-05-03", "B2_after_cli_use_from_not_earned"))
    # Now delayed fetch arrives showing best=10, active=10 (or 0 after second miss) -> stays `used`!
    run_real_fetch(picks, datetime(2027, 5, 3, 13, tzinfo=ET), 10, 10,
                   [{"roundId": 3, "result": "not_hit", "streak": 10}], {3: date(2027, 5, 2)})
    seq_between_fetch.append(snapshot_step(picks, datetime(2027, 5, 3, 15, tzinfo=timezone.utc), "2027-05-03", "B3_fetch_best10_after_use_stays_used"))

# Sequence C: Mistaken `--use` from `not_earned` (at streak 4, unearned!), followed by `--undo` -> ends `active`!
# Evaluate policy, health, and dashboard at streaks 5, 10, and 16 (including dates where saver flips action, e.g. 2027-08-23 & 2027-08-29)!
seq_mistaken_undo = []
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    run_real_fetch(picks, datetime(2027, 8, 20, 10, tzinfo=ET), 4, 4,
                   [{"roundId": 1, "result": "hit", "streak": 4}], {1: date(2027, 8, 19)})
    assert CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks), "--season", "2027", "--use"]).exit_code == 0
    assert CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks), "--season", "2027", "--undo"]).exit_code == 0
    # Flag is now `active` even though best_streak is 4 (< 10)!
    for s_test, best_test, d_iso in [
        (5, 5, "2027-08-29"),
        (10, 10, "2027-08-23"),
        (16, 16, "2027-08-23"),
    ]:
        (picks / "account_state" / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": s_test,
            "best_streak": best_test,
            "source": "mlb_bts_profile",
            "source_date": d_iso,
        }))
        now_t = datetime.fromisoformat(f"{d_iso}T15:00:00+00:00")
        snap_active = snapshot_step(picks, now_t, d_iso, f"C_mistaken_undo_active_streak_{s_test}")
        # Compare against what a true `not_earned` (at s=5) or `used` (if spent) would produce on that exact date!
        ctx82 = strat.DecisionContext(0.82, 0.82, True, d_iso, True, mdp, best_test, "trusted")
        snap_active["compare_policy_if_saver_false_p82"] = strat.resolve_policy_decision(ctx82, s_test, False).action
        snap_active["compare_policy_if_saver_true_p82"] = strat.resolve_policy_decision(ctx82, s_test, True).action
        seq_mistaken_undo.append(snap_active)

# Sequence D: Midnight ET year change between read and write!
midnight_races = {}
# D1: Dashboard GET at 2026-12-31 23:59:55 ET -> POST at 2027-01-01 00:00:05 ET
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    (picks / "account_state").mkdir(parents=True)
    fp = picks / "account_state" / "saver_state.json"
    fp.write_text(json.dumps({"season": 2026, "state": "active"}))
    before = fp.read_bytes()
    t_pre = datetime(2026, 12, 31, 23, 59, 55, tzinfo=ET).astimezone(timezone.utc)
    t_post = datetime(2027, 1, 1, 0, 0, 5, tzinfo=ET).astimezone(timezone.utc)
    ctx_pre = web.saver_dashboard_context(picks, now=t_pre)
    post_after_midnight = web.saver_transition_response(
        picks, expected_prior=ctx_pre.expected_prior, new_state="used", same_origin=True, now=t_post
    )
    midnight_races["dash_get_2026_post_2027"] = {
        "ctx_pre_state": ctx_pre.state,
        "ctx_pre_button": ctx_pre.button,
        "post_after_midnight": post_after_midnight,
        "file_2026_preserved": fp.read_bytes() == before,
    }

# D2: web.render_page() where load_dashboard_streak_context runs at 2026-12-31 23:59:59 ET
# and saver_dashboard_context runs at 2027-01-01 00:00:01 ET!
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    (picks / "account_state").mkdir(parents=True)
    (picks / "streak.json").write_text(json.dumps({"season": 2026, "streak": 14, "saver_available": True}))
    (picks / "account_state" / "saver_state.json").write_text(json.dumps({"season": 2027, "state": "not_earned"}))
    # Tick 1 (in load_decision_streak_state): 2026-12-31 23:59:59 ET -> loads 2026 model streak = 14!
    # Tick 2 (in render_page saver_dashboard_context): 2027-01-01 00:00:01 ET -> loads 2027 saver = not_earned!
    ticks = iter([
        datetime(2026, 12, 31, 23, 59, 59, tzinfo=ET).astimezone(timezone.utc),
        datetime(2027, 1, 1, 0, 0, 1, tzinfo=ET),
        datetime(2027, 1, 1, 0, 0, 1, tzinfo=ET).astimezone(timezone.utc),
    ])
    orig_cs_dt = cs.datetime
    class SplitCSDT(datetime):
        @classmethod
        def now(cls, tz=None):
            t = datetime(2026, 12, 31, 23, 59, 59, tzinfo=ET)
            return t.astimezone(tz) if tz else t.replace(tzinfo=None)
    class SplitWebDT(datetime):
        @classmethod
        def now(cls, tz=None):
            t = datetime(2027, 1, 1, 0, 0, 1, tzinfo=ET)
            return t.astimezone(tz) if tz else t.replace(tzinfo=None)
    with patch.object(web, "PICKS_DIR", picks), \
         patch.object(web, "fetch_bluesky_posts", return_value=[]), \
         patch.object(cs, "datetime", SplitCSDT), \
         patch.object(web, "datetime", SplitWebDT):
        page_html = web.render_page()
    midnight_races["render_page_midnight_split"] = {
        "has_not_yet_earned": "Streak Saver: not yet earned" in page_html,
        "has_reserved": "Streak Saver: reserved (not yet earned)" in page_html,
    }

out_file.write_text(json.dumps({
    "main_sequence": steps,
    "between_fetch": seq_between_fetch,
    "mistaken_undo": seq_mistaken_undo,
    "midnight_races": midnight_races,
}, indent=2))
'''


def run_py(code: str, src_dir: Path, *extra_args: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_p = Path(tf.name)
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{src_dir}:{REPO}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TZ"] = "America/New_York"
        proc = subprocess.run([str(PYTHON), "-c", code, *extra_args, str(out_p)], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(f"Worker failed (rc={proc.returncode}):\n{proc.stderr}\n")
            raise SystemExit(proc.returncode)
        return json.loads(out_p.read_text())
    finally:
        out_p.unlink(missing_ok=True)


def main() -> None:
    pol_base = run_py(POLICY_WORKER, BASE_SRC, "base")
    pol_branch = run_py(POLICY_WORKER, BRANCH_SRC, "branch")
    seq_branch = run_py(SEQUENCE_WORKER, BRANCH_SRC)

    # Analyze saver_available bit differences across (flag_kind, contest_kind, streak)
    bit_diffs = []
    for k, b_val in pol_base["state_bit_map"].items():
        br_val = pol_branch["state_bit_map"][k]
        if b_val["saver"] != br_val["saver"]:
            fk, ck, s_str = k.split("|")
            bit_diffs.append({
                "flag": fk, "contest": ck, "streak": int(s_str),
                "base_saver": b_val["saver"], "branch_saver": br_val["saver"],
                "branch_flag_state": br_val["flag_state"],
            })

    # Verify policy flip streaks across all 365 dates of 2027
    flip_streaks = sorted({c["streak"] for c in pol_branch["saver_flip_cells"]})
    flip_Streaks_below_10 = sorted({c["streak"] for c in pol_branch["saver_flip_cells"] if c["streak"] < 10})
    flip_streaks_10_to_15 = sorted({c["streak"] for c in pol_branch["saver_flip_cells"] if 10 <= c["streak"] <= 15})
    flip_streaks_above_15 = sorted({c["streak"] for c in pol_branch["saver_flip_cells"] if c["streak"] > 15})

    summary = {
        "bit_diffs_count": len(bit_diffs),
        "bit_diffs": bit_diffs,
        "flip_streaks_all": flip_streaks,
        "flip_streaks_below_10": flip_Streaks_below_10,
        "flip_streaks_10_to_15": flip_streaks_10_to_15,
        "flip_streaks_above_15": flip_streaks_above_15,
        "saver_flip_cells_count": len(pol_branch["saver_flip_cells"]),
        "sample_flips_below_10": [c for c in pol_branch["saver_flip_cells"] if c["streak"] < 10][:15],
        "sample_flips_above_15": [c for c in pol_branch["saver_flip_cells"] if c["streak"] > 15][:10],
        "week1_base": pol_base["week1_scenarios"],
        "week1_branch": pol_branch["week1_scenarios"],
        "sequences": seq_branch,
    }
    out_json = REVIEW_DIR / "probe_policy_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")
    print(f"bit_diffs_count: {len(bit_diffs)}")
    print(f"flip_streaks_all (p in 0.70..0.90): {flip_streaks}")
    print(f"flip_streaks_below_10: {flip_Streaks_below_10}")
    print(f"flip_streaks_10_to_15: {flip_streaks_10_to_15}")
    print(f"flip_streaks_above_15: {flip_streaks_above_15}")


if __name__ == "__main__":
    main()
