#!/usr/bin/env python3
"""Adversarial probe v3 for CS3 review: items 1, 4, 5, 6, 7.

Fixes SchedulerState initialization from v2 without editing v1/v2.
Compares archived base (7a153ec9 in .codex-review/cs3-review/base/src) and
branch (HEAD in src) in isolated subprocesses over real files and real calls.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base" / "src"
BRANCH_SRC = REPO / "src"
PYTHON = REPO / ".venv" / "bin" / "python"

WORKER_CODE = r'''
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ["TZ"] = "America/New_York"
import time
time.tzset()

mode = sys.argv[1]  # "base" or "branch"
out_file = Path(sys.argv[2])

import bts.contest_state as cs
import bts.saver_state as ss
import bts.web as web
from bts.cli import cli
from bts.health import alert as halert
from bts.health import attention as hattn
from bts.health import contest_state as hcs
from click.testing import CliRunner

ET = ZoneInfo("America/New_York")
NOW_2027 = datetime(2027, 7, 15, 16, 0, tzinfo=timezone.utc)


def call_capture(fn):
    try:
        val = fn()
        return {"ok": True, "val": val}
    except BaseException as exc:
        return {"ok": False, "exc": type(exc).__name__, "msg": str(exc)}


def seed_saver(picks: Path, kind: str) -> Path:
    d = picks / "account_state"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "saver_state.json"
    if p.exists() or p.is_symlink():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.chmod(0o600)
            p.unlink()
    if kind == "missing":
        return p
    if kind == "chmod000":
        p.write_text('{"season": 2027, "state": "active"}')
        p.chmod(0o000)
        return p
    if kind == "non_utf8":
        p.write_bytes(b"\xff\xfe")
        return p
    if kind == "non_object":
        p.write_text("[]")
        return p
    if kind == "malformed":
        p.write_text("{bad")
        return p
    if kind == "invalid_state_2027":
        p.write_text('{"season": 2027, "state": "bogus"}')
        return p
    if kind == "bool_season_true":
        p.write_text('{"season": true, "state": "active"}')
        return p
    if kind == "directory":
        p.mkdir()
        return p
    yr_s, st = kind.split(":", 1)
    p.write_text(json.dumps({"season": int(yr_s), "state": st, "source": "seed", "updated_at": "2027-01-01T00:00:00+00:00"}))
    return p


def seed_contest(picks: Path, kind: str, streak: int = 7, best: int | None = None) -> None:
    d = picks / "account_state"
    d.mkdir(parents=True, exist_ok=True)
    auto_p = d / "contest_streak.json"
    man_p = d / "contest_streak.manual.json"
    if best is None:
        best = max(streak, 0)
    if kind == "none":
        return
    if kind == "undated":
        auto_p.write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": streak,
            "best_streak": best,
            "source": "mlb_bts_profile",
            "source_date": None,
            "recorded_at": "2027-07-15T12:00:00Z",
        }))
    elif kind == "2026":
        auto_p.write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": streak,
            "best_streak": best,
            "source": "mlb_bts_profile",
            "source_date": "2026-09-28",
            "recorded_at": "2026-09-29T06:00:00Z",
        }))
    elif kind == "2027":
        auto_p.write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": streak,
            "best_streak": best,
            "source": "mlb_bts_profile",
            "source_date": "2027-06-18",
            "recorded_at": "2027-06-18T15:00:00Z",
        }))
    elif kind == "manual_unexpired_2026_date":
        man_p.write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v2",
            "active_streak": streak,
            "best_streak": best,
            "source": "manual_cli",
            "source_date": "2026-09-28",
            "override_expires_at": "2027-07-16T16:00:00Z",
        }))


results = {"mode": mode}

# --- 1. Row 1: policy_saver_available & is_available ---
row1 = {}
has_psa = hasattr(ss, "policy_saver_available")
row1["has_policy_saver_available"] = has_psa
for st in ("uninitialized", "not_earned", "active", "used", "bogus"):
    s_obj = ss.SaverState(st, 2027)
    row1[f"is_available:{st}"] = s_obj.is_available
    if has_psa:
        for n in (-5, -1, 0, 1, 9, 10, 15, 16, 57, True, False):
            key = f"psa:{st}:streak={n!r}"
            row1[key] = call_capture(lambda s_obj=s_obj, n=n: ss.policy_saver_available(s_obj, n))
        for bad_n in (None, "7", [5]):
            key = f"psa:{st}:streak={bad_n!r}"
            row1[key] = call_capture(lambda s_obj=s_obj, bad_n=bad_n: ss.policy_saver_available(s_obj, bad_n))
results["row1"] = row1

# --- 2. Decision matrix (Rows 2, 3, 4) ---
decision_matrix = []
for saver_kind in ("missing", "chmod000", "2026:active", "2026:used", "2026:not_earned", "2027:not_earned", "2027:active", "2027:used"):
    for contest_kind in ("none", "undated", "2026", "2027", "manual_unexpired_2026_date"):
        for streak in (0, 7, 9, 10, 12, 15, 16, 57):
            with tempfile.TemporaryDirectory() as td:
                picks = Path(td)
                p = seed_saver(picks, saver_kind)
                try:
                    if contest_kind == "none":
                        (picks / "streak.json").write_text(json.dumps({
                            "season": 2027, "streak": streak, "saver_available": (streak < 12),
                        }))
                    else:
                        seed_contest(picks, contest_kind, streak=streak)
                    out = call_capture(lambda: cs.load_decision_streak_state(picks, now=NOW_2027))
                    if out["ok"]:
                        st_val = out["val"]
                        out["val"] = {
                            "streak": st_val.streak,
                            "saver_available": st_val.saver_available,
                            "saver_flag_state": getattr(st_val, "saver_flag_state", "<absent>"),
                            "source": st_val.source,
                            "status": st_val.status,
                        }
                finally:
                    if saver_kind == "chmod000" and p.exists():
                        p.chmod(0o600)
                decision_matrix.append({
                    "saver": saver_kind, "contest": contest_kind, "streak": streak, "res": out,
                })
results["decision_matrix"] = decision_matrix

# --- 3. Item 4 adversarial probe: model-only branch streak.json JSON values ---
item4_cases = []
model_json_payloads = [
    ("empty_dict", "{}"),
    ("season_only", '{"season": 2027}'),
    ("prior_season_null_streak", '{"season": 2026, "streak": null}'),
    ("streak_null", '{"season": 2027, "streak": null}'),
    ("streak_str_7", '{"season": 2027, "streak": "7"}'),
    ("streak_str_12", '{"season": 2027, "streak": "12"}'),
    ("streak_list", '{"season": 2027, "streak": [7]}'),
    ("streak_dict", '{"season": 2027, "streak": {}}'),
    ("streak_bool_true", '{"season": 2027, "streak": true}'),
    ("streak_bool_false", '{"season": 2027, "streak": false}'),
    ("streak_neg_1", '{"season": 2027, "streak": -1}'),
    ("streak_neg_45", '{"season": 2027, "streak": -45}'),
    ("streak_float_9_5", '{"season": 2027, "streak": 9.5}'),
    ("streak_float_10_5", '{"season": 2027, "streak": 10.5}'),
    ("streak_nan", '{"season": 2027, "streak": NaN}'),
    ("streak_neginf", '{"season": 2027, "streak": -Infinity}'),
    ("streak_int_7", '{"season": 2027, "streak": 7}'),
    ("streak_int_12", '{"season": 2027, "streak": 12}'),
]
for label, raw_json in model_json_payloads:
    for saver_kind in ("missing", "2027:not_earned", "2027:active", "2027:used"):
        with tempfile.TemporaryDirectory() as td:
            picks = Path(td)
            seed_saver(picks, saver_kind)
            (picks / "streak.json").write_text(raw_json)
            out = call_capture(lambda: cs.load_decision_streak_state(picks, now=NOW_2027))
            if out["ok"]:
                st_val = out["val"]
                out["val"] = {
                    "streak_repr": repr(st_val.streak),
                    "streak_type": type(st_val.streak).__name__,
                    "saver_available": st_val.saver_available,
                    "saver_flag_state": getattr(st_val, "saver_flag_state", "<absent>"),
                }
            item4_cases.append({
                "payload": label, "saver": saver_kind, "res": out,
            })

# Downstream caller 1: scheduler._deliver_and_lock_pick with streak=null + not_earned!
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    cur_yr = datetime.now(ET).year
    seed_saver(picks, f"{cur_yr}:not_earned")
    (picks / "streak.json").write_text(json.dumps({"season": cur_yr, "streak": None}))
    from bts.picks import DailyPick, Pick, save_pick
    import bts.scheduler as sched
    dp = DailyPick(
        date=f"{cur_yr}-07-15",
        run_time=f"{cur_yr}-07-15T14:00:00+00:00",
        pick=Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 1, f"{cur_yr}-07-15T23:10:00Z", "NYY"),
        double_down=None,
        runner_up=None,
    )
    save_pick(dp, picks)
    st_sched = sched.SchedulerState(
        date=f"{cur_yr}-07-15",
        schedule_fetched_at="",
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
    )
    cfg = {
        "orchestrator": {"picks_dir": str(picks)},
        "scheduler": {"pick_delivery": "dm", "contest_state_required": False},
        "bluesky": {"dm_recipient": "eric.test"},
    }
    sent_dms = []
    with patch("bts.dm.send_dm", side_effect=lambda r, b: sent_dms.append((r, b)) or "msg-1"), \
         patch.object(sched, "_now_et", return_value=datetime(cur_yr, 7, 15, 12, 0, tzinfo=ET)), \
         patch.object(sched, "_void_committed_games", return_value={}):
        deliver_res = call_capture(lambda: sched._deliver_and_lock_pick(
            dp, cfg, picks, st_sched, f"{cur_yr}-07-15", "check-1"
        ))
    results["item4_deliver_and_lock_streak_null"] = {
        "deliver_res": deliver_res,
        "sent_dms_count": len(sent_dms),
        "sent_body": sent_dms[0][1] if sent_dms else None,
    }

# Downstream caller 2: negative streak (-45) reaching resolve_policy_decision!
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    seed_saver(picks, "2027:not_earned")
    (picks / "streak.json").write_text('{"season": 2027, "streak": -45}')
    st_neg = cs.load_decision_streak_state(picks, now=NOW_2027)
    from bts.strategy import DecisionContext, _load_mdp, resolve_policy_decision
    ctx_neg = DecisionContext(0.80, 0.80, True, "2027-08-29", True, _load_mdp())
    dec_neg = resolve_policy_decision(ctx_neg, st_neg.streak, st_neg.saver_available)
    results["item4_neg45_policy"] = {
        "streak": st_neg.streak,
        "saver_available": st_neg.saver_available,
        "action": dec_neg.action,
        "objective": dec_neg.objective,
    }

# Downstream caller 3: orchestrator.orchestrate with streak=null + not_earned!
with tempfile.TemporaryDirectory() as td:
    picks = Path(td)
    cur_yr = datetime.now(ET).year
    seed_saver(picks, f"{cur_yr}:not_earned")
    (picks / "streak.json").write_text(json.dumps({"season": cur_yr, "streak": None}))
    import bts.orchestrator as orch
    import pandas as pd
    cfg_p = Path(td) / "cfg.toml"
    cfg_p.write_text(
        f'[orchestrator]\npicks_dir = "{picks}"\nmodels_dir = "data/models"\n'
        '[bluesky]\ndm_recipient = "eric.test"\n'
        '[tiers.local]\ntype = "local"\n'
    )
    preds_df = pd.DataFrame([{
        "batter_name": "Steven Kwan", "batter_id": 680757, "team": "CLE",
        "lineup": 1, "pitcher_name": "Gerrit Cole", "pitcher_id": 543037,
        "p_game_hit": 0.82, "flags": "", "game_pk": 1,
        "game_time": f"{cur_yr}-07-15T23:10:00Z", "pitcher_team": "NYY",
    }])
    orch_dms = []
    with patch.object(orch, "run_cascade", return_value=(preds_df, "local")), \
         patch("bts.picks.get_game_statuses_detailed", return_value={1: {"abstract": "P", "detailed": "Pre-Game"}}), \
         patch("bts.dm.send_dm", side_effect=lambda r, b: orch_dms.append((r, b)) or "dm1"):
        orch_res = call_capture(lambda: orch.orchestrate(cfg_p, f"{cur_yr}-07-15"))
    results["item4_orchestrate_streak_null"] = {
        "orch_res": orch_res,
        "orch_dms": len(orch_dms),
    }

results["item4_cases"] = item4_cases

# --- 4. Item 5: Health & attention matrix + command paste verification + Day 1/2/30 DMs ---
health_matrix = []
cmd_paste_results = []
for saver_kind in ("missing", "chmod000", "non_utf8", "invalid_state_2027", "2026:not_earned", "2026:active", "2026:used", "2027:not_earned", "2027:active", "2027:used"):
    for contest_kind in ("none", "undated", "2026", "2027"):
        for streak in (0, 9, 10, 15, 16, 57):
            for expected in (False, True):
                with tempfile.TemporaryDirectory() as td:
                    picks = Path(td)
                    p = seed_saver(picks, saver_kind)
                    try:
                        seed_contest(picks, contest_kind, streak=streak, best=max(streak, 10 if streak >= 10 else streak))
                        alerts = hcs.check(picks, expected=expected, now=NOW_2027)
                        alert_rows = [{"level": a.level, "source": a.source, "message": a.message} for a in alerts]
                        health_matrix.append({
                            "saver": saver_kind, "contest": contest_kind,
                            "streak": streak, "expected": expected,
                            "alerts": alert_rows,
                        })
                        for a in alerts:
                            if "Streak Saver" in a.message:
                                m = re.search(r"`([^`]+)`", a.message)
                                cmd_str = m.group(1) if m else None
                                if cmd_str and len(cmd_paste_results) < 60:
                                    argv = shlex.split(cmd_str)
                                    assert argv[0] == "bts"
                                    cli_args = argv[1:] + ["--picks-dir", str(picks)]
                                    r = CliRunner().invoke(cli, cli_args)
                                    after_alerts = hcs.check(picks, expected=expected, now=NOW_2027)
                                    after_saver = [x.message for x in after_alerts if "Streak Saver" in x.message]
                                    cmd_paste_results.append({
                                        "saver": saver_kind, "contest": contest_kind,
                                        "streak": streak, "expected": expected,
                                        "cmd": cmd_str,
                                        "exit_code": r.exit_code,
                                        "output_last": r.output.splitlines()[-1] if r.output.splitlines() else "",
                                        "post_state": ss.load_saver_state(picks, season=2027).state,
                                        "remaining_saver_alerts": after_saver,
                                    })
                    finally:
                        if p.exists() and not p.is_dir():
                            p.chmod(0o600)
results["health_matrix"] = health_matrix
results["cmd_paste_results"] = cmd_paste_results

# Simulate Day 1, Day 2, ..., Day 30 DMs for uninitialized saver in 2027
dm_timeline = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    picks = root / "picks"
    health_dir = root / "health_state"
    seed_saver(picks, "2026:used")
    seed_contest(picks, "2027", streak=0, best=0)
    health_dir.mkdir(parents=True, exist_ok=True)
    (health_dir / "warn_attention_state.json").write_text(json.dumps({
        "schema_version": "bts_warn_attention_state_v1",
        "sources": {"contest_state": {"last_seen": "2027-03-23", "streak": 5, "last_message": "old"}},
    }))
    start = date(2027, 3, 24)
    for day_idx in range(1, 31):
        d_cur = start + timedelta(days=day_idx - 1)
        now_cur = datetime(d_cur.year, d_cur.month, d_cur.day, 23, 0, tzinfo=timezone.utc)
        alerts = hcs.check(picks, expected=True, now=now_cur)
        pol_crit, warn_att = hattn.apply_warn_attention_policy(
            alerts, state_path=health_dir / "warn_attention_state.json", today=d_cur
        )
        all_alerts = alerts + pol_crit
        sent_bodies = []
        with patch.object(halert, "send_dm", side_effect=lambda rec, body: sent_bodies.append(body)):
            sent_first = halert.dispatch_dm_for_health_alerts(
                all_alerts, "eric.test", warn_attention=warn_att,
                status_path=health_dir / "health_dm_delivery_status.json", now_et_date=d_cur,
            )
            sent_repeat_same_day = halert.dispatch_dm_for_health_alerts(
                all_alerts, "eric.test", warn_attention=warn_att,
                status_path=health_dir / "health_dm_delivery_status.json", now_et_date=d_cur,
            )
        if day_idx in (1, 2, 3, 15, 30):
            dm_timeline.append({
                "day": day_idx,
                "date": d_cur.isoformat(),
                "alerts": [(a.level, a.source, a.message) for a in alerts],
                "warn_att": [(a.source, a.message) for a in warn_att],
                "sent_first": sent_first,
                "sent_repeat_same_day": sent_repeat_same_day,
                "dm_count": len(sent_bodies),
                "body": sent_bodies[0] if sent_bodies else None,
            })
results["dm_timeline_days_1_2_30"] = dm_timeline

# --- 5. Item 6: Dashboard matrix + Dec 31 23:59 ET vs Jan 1 00:01 ET ---
dash_cases = []
bad_contest_kinds = [
    "missing", "chmod000", "non_utf8", "non_object", "malformed",
    "overlong_int", "refused_58", "refused_best_lt_streak", "directory",
    "manual_chmod000", "manual_non_utf8", "manual_non_object", "manual_directory",
]
for bad in bad_contest_kinds:
    with tempfile.TemporaryDirectory() as td:
        picks = Path(td)
        flag_p = seed_saver(picks, "2027:active")
        d = picks / "account_state"
        auto_p = d / "contest_streak.json"
        man_p = d / "contest_streak.manual.json"
        cleanup_chmod = []
        if bad == "missing":
            pass
        elif bad == "chmod000":
            auto_p.write_text('{"active_streak": 5}')
            auto_p.chmod(0o000)
            cleanup_chmod.append(auto_p)
        elif bad == "non_utf8":
            auto_p.write_bytes(b"\xff")
        elif bad == "non_object":
            auto_p.write_text("[]")
        elif bad == "malformed":
            auto_p.write_text("{")
        elif bad == "overlong_int":
            auto_p.write_text('{"active_streak": ' + ("9" * 5000) + "}")
        elif bad == "refused_58":
            auto_p.write_text('{"active_streak": 58, "best_streak": 58, "source_date": "2027-06-18"}')
        elif bad == "refused_best_lt_streak":
            auto_p.write_text('{"active_streak": 10, "best_streak": 9, "source_date": "2027-06-18"}')
        elif bad == "directory":
            auto_p.mkdir()
        elif bad == "manual_chmod000":
            man_p.write_text('{"active_streak": 5}')
            man_p.chmod(0o000)
            cleanup_chmod.append(man_p)
        elif bad == "manual_non_utf8":
            man_p.write_bytes(b"\xff")
        elif bad == "manual_non_object":
            man_p.write_text("[]")
        elif bad == "manual_directory":
            man_p.mkdir()

        try:
            ctx_res = call_capture(lambda: web.saver_dashboard_context(picks, now=NOW_2027))
            if ctx_res["ok"]:
                c = ctx_res["val"]
                ctx_res["val"] = {"state": c.state, "button": c.button, "nudge": c.nudge, "warning": c.warning}
            post_res = call_capture(lambda: web.saver_transition_response(
                picks, expected_prior="active", new_state="used", same_origin=True, now=NOW_2027, peer_ip="100.64.0.1"
            ))
            flag_bytes = flag_p.read_text() if flag_p.exists() else None
            audit_p = d / "saver_transitions.jsonl"
            audit_lines = [json.loads(x) for x in audit_p.read_text().splitlines()] if audit_p.exists() else []
            class _FixedWebClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return NOW_2027.astimezone(tz) if tz else NOW_2027.replace(tzinfo=None)
            with patch.object(web, "PICKS_DIR", picks), \
                 patch.object(web, "datetime", _FixedWebClock), \
                 patch.object(web, "fetch_bluesky_posts", return_value=[]):
                html_res = call_capture(lambda: "saver-section" in web.render_page())
            dash_cases.append({
                "bad": bad, "ctx": ctx_res, "post": post_res,
                "flag_after": json.loads(flag_bytes) if flag_bytes else None,
                "audit_count": len(audit_lines),
                "last_audit": audit_lines[-1] if audit_lines else None,
                "page_has_saver_section": html_res,
            })
        finally:
            for cp in cleanup_chmod:
                cp.chmod(0o600)
results["dash_cases"] = dash_cases

# Dashboard boundary timestamps: Dec 31 23:59 ET, Jan 1 00:01 ET, and 2026 flag on disk in 2027
dash_boundary = {}
DEC31_2359_ET = datetime(2026, 12, 31, 23, 59, tzinfo=ET).astimezone(timezone.utc)  # 2027-01-01T04:59:00Z
JAN01_0001_ET = datetime(2027, 1, 1, 0, 1, tzinfo=ET).astimezone(timezone.utc)     # 2027-01-01T05:01:00Z
for name, ts, seed_yr in [
    ("dec31_2359_et_with_2026_active", DEC31_2359_ET, 2026),
    ("jan01_0001_et_with_2027_active_and_2026_contest", JAN01_0001_ET, 2027),
    ("2026_active_flag_in_2027", JAN01_0001_ET, 2026),
]:
    with tempfile.TemporaryDirectory() as td:
        picks = Path(td)
        fp = seed_saver(picks, f"{seed_yr}:active")
        seed_contest(picks, "2026", streak=10, best=10)
        before_bytes = fp.read_bytes()
        resp = call_capture(lambda: web.saver_transition_response(
            picks, expected_prior="active", new_state="used", same_origin=True, now=ts, peer_ip="100.64.1.1"
        ))
        after_bytes = fp.read_bytes()
        ap = picks / "account_state/saver_transitions.jsonl"
        audit_lines = [json.loads(x) for x in ap.read_text().splitlines()] if ap.exists() else []
        dash_boundary[name] = {
            "resp": resp,
            "bytes_unchanged": before_bytes == after_bytes,
            "flag_json": json.loads(after_bytes.decode()),
            "audit": audit_lines[-1] if audit_lines else None,
        }
results["dash_boundary"] = dash_boundary

# --- 6. Item 7: CLI saver-state flag combinations + auto-earn sites ---
cli_combos = []
flag_arg_sets = [
    [],
    ["--show"],
    ["--use"],
    ["--undo"],
    ["--use", "--force"],
    ["--undo", "--force"],
    ["--init", "not_earned"],
    ["--init", "active"],
    ["--init", "used"],
    ["--init", "used", "--force"],
    ["--init", "not_earned", "--use"],
    ["--init", "active", "--undo"],
    ["--use", "--undo"],
    ["--show", "--use"],
]
for prior_kind in ("missing", "2026:active", "2027:not_earned", "2027:active", "2027:used"):
    for fargs in flag_arg_sets:
        with tempfile.TemporaryDirectory() as td:
            picks = Path(td)
            fp = seed_saver(picks, prior_kind)
            before_b = fp.read_bytes() if fp.exists() else None
            r = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks), "--season", "2027", *fargs])
            after_b = fp.read_bytes() if fp.exists() else None
            ap = picks / "account_state" / "saver_transitions.jsonl"
            audits = [json.loads(x) for x in ap.read_text().splitlines()] if ap.exists() else []
            lines = r.output.splitlines()
            cli_combos.append({
                "prior": prior_kind,
                "args": fargs,
                "exit_code": r.exit_code,
                "last_line": lines[-1] if lines else "",
                "output": r.output,
                "state_2027": ss.load_saver_state(picks, season=2027).state,
                "file_untouched": before_b == after_b,
                "audit_count": len(audits),
                "audits": audits,
            })
results["cli_combos"] = cli_combos

# Race test on `bts saver-state --use`: not_earned promoted to active between initial read and lock
if mode == "branch":
    with tempfile.TemporaryDirectory() as td:
        picks = Path(td)
        seed_saver(picks, "2027:not_earned")
        real_tx = ss.transition_saver_state
        calls = []
        def racing_tx(pdir, *, expected_prior, new_state, season, source, force=False, peer=None):
            calls.append((expected_prior, new_state))
            if len(calls) == 1 and expected_prior == "not_earned":
                real_tx(pdir, expected_prior="not_earned", new_state="active", season=season, source="auto_earn")
            return real_tx(pdir, expected_prior=expected_prior, new_state=new_state, season=season, source=source, force=force, peer=peer)
        with patch.object(ss, "transition_saver_state", side_effect=racing_tx):
            r_race = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks), "--season", "2027", "--use"])
        results["cli_use_race_not_earned_to_active"] = {
            "exit_code": r_race.exit_code,
            "output": r_race.output,
            "calls": calls,
            "final_state": ss.load_saver_state(picks, season=2027).state,
            "audits": [json.loads(x) for x in (picks / "account_state/saver_transitions.jsonl").read_text().splitlines()],
        }

# Row 11: fetch-contest-streak and set-contest-streak auto-earn matrix
auto_earn_cases = []
cur_year = datetime.now(ET).year
for cmd_name in ("fetch-contest-streak", "set-contest-streak"):
    for src_kind in ("null", "prev_year", "cur_year", "next_year"):
        if src_kind == "null":
            src_arg = None
        elif src_kind == "prev_year":
            src_arg = date(cur_year - 1, 6, 18)
        elif src_kind == "cur_year":
            src_arg = date(cur_year, 6, 18)
        elif src_kind == "next_year":
            src_arg = date(cur_year + 1, 6, 18)

        for prior_flag in ("missing", f"{cur_year - 1}:active", f"{cur_year}:not_earned", f"{cur_year}:used"):
            for best_val in (None, 0, 7, 10, 12):
                if cmd_name == "fetch-contest-streak" and best_val is None:
                    continue
                with tempfile.TemporaryDirectory() as td:
                    picks = Path(td)
                    fp = seed_saver(picks, prior_flag)
                    before_b = fp.read_bytes() if fp.exists() else None
                    if cmd_name == "fetch-contest-streak":
                        from tests.test_contest_fetch import _mock_fetch
                        preds = [] if src_arg is None else [{"roundId": 1, "result": "hit"}]
                        rnds = {} if src_arg is None else {1: src_arg}
                        from pytest import MonkeyPatch
                        mp = MonkeyPatch()
                        _mock_fetch(mp, {"activeStreak": 0, "seasonBestStreak": best_val, "predictions": preds}, rnds)
                        try:
                            r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks), "--expected-username", "stonehengee"])
                        finally:
                            mp.undo()
                    else:
                        args = ["set-contest-streak", "--picks-dir", str(picks), "--streak", "0"]
                        if best_val is not None:
                            args += ["--best-streak", str(best_val)]
                        if src_arg is not None:
                            args += ["--source-date", src_arg.isoformat()]
                        r = CliRunner().invoke(cli, args)
                    after_b = fp.read_bytes() if fp.exists() else None
                    auto_earn_cases.append({
                        "cmd": cmd_name, "src_kind": src_kind, "prior": prior_flag, "best": best_val,
                        "exit_code": r.exit_code,
                        "file_untouched": before_b == after_b,
                        "cur_year_state": ss.load_saver_state(picks, season=cur_year).state,
                        "raw_flag": json.loads(after_b.decode()) if after_b else None,
                    })
results["auto_earn_cases"] = auto_earn_cases

out_file.write_text(json.dumps(results, indent=2))
'''


def run_worker(mode: str, src_dir: Path) -> dict:
    with tempfile.NamedTemporaryFile(suffix=f"_{mode}.json", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{src_dir}:{REPO}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TZ"] = "America/New_York"
        proc = subprocess.run(
            [str(PYTHON), "-c", WORKER_CODE, mode, str(out_path)],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(f"Worker {mode} failed (rc={proc.returncode}):\n{proc.stderr}\n")
            raise SystemExit(proc.returncode)
        return json.loads(out_path.read_text())
    finally:
        out_path.unlink(missing_ok=True)


def main() -> None:
    base = run_worker("base", BASE_SRC)
    branch = run_worker("branch", BRANCH_SRC)

    summary = {
        "decision_diffs": [],
        "item4_diffs": [],
        "item4_deliver_base": base["item4_deliver_and_lock_streak_null"],
        "item4_deliver_branch": branch["item4_deliver_and_lock_streak_null"],
        "item4_orchestrate_base": base["item4_orchestrate_streak_null"],
        "item4_orchestrate_branch": branch["item4_orchestrate_streak_null"],
        "item4_neg45_base": base["item4_neg45_policy"],
        "item4_neg45_branch": branch["item4_neg45_policy"],
        "health_diffs_count": 0,
        "cmd_paste_base": base["cmd_paste_results"],
        "cmd_paste_branch": branch["cmd_paste_results"],
        "dm_timeline_base": base["dm_timeline_days_1_2_30"],
        "dm_timeline_branch": branch["dm_timeline_days_1_2_30"],
        "dash_diffs": [],
        "dash_boundary_base": base["dash_boundary"],
        "dash_boundary_branch": branch["dash_boundary"],
        "cli_combo_diffs": [],
        "cli_use_race": branch.get("cli_use_race_not_earned_to_active"),
        "auto_earn_diffs": [],
    }

    for b0, b1 in zip(base["decision_matrix"], branch["decision_matrix"]):
        r0 = dict(b0["res"])
        r1 = dict(b1["res"])
        v0 = {k: v for k, v in r0.get("val", {}).items() if k != "saver_flag_state"} if r0["ok"] else r0
        v1 = {k: v for k, v in r1.get("val", {}).items() if k != "saver_flag_state"} if r1["ok"] else r1
        if v0 != v1:
            summary["decision_diffs"].append({
                "saver": b0["saver"], "contest": b0["contest"], "streak": b0["streak"],
                "base": r0, "branch": r1,
            })

    for i0, i1 in zip(base["item4_cases"], branch["item4_cases"]):
        r0 = i0["res"]
        r1 = i1["res"]
        v0 = {k: v for k, v in r0.get("val", {}).items() if k != "saver_flag_state"} if r0["ok"] else r0
        v1 = {k: v for k, v in r1.get("val", {}).items() if k != "saver_flag_state"} if r1["ok"] else r1
        if v0 != v1:
            summary["item4_diffs"].append({
                "payload": i0["payload"], "saver": i0["saver"],
                "base": r0, "branch": r1,
            })

    for h0, h1 in zip(base["health_matrix"], branch["health_matrix"]):
        if h0["alerts"] != h1["alerts"]:
            summary["health_diffs_count"] += 1

    for d0, d1 in zip(base["dash_cases"], branch["dash_cases"]):
        if (d0["ctx"] != d1["ctx"]) or (d0["post"] != d1["post"]) or (d0["page_has_saver_section"] != d1["page_has_saver_section"]):
            summary["dash_diffs"].append({"bad": d0["bad"], "base": d0, "branch": d1})

    for c0, c1 in zip(base["cli_combos"], branch["cli_combos"]):
        if (c0["exit_code"], c0["state_2027"], c0["file_untouched"], c0["audit_count"]) != (
            c1["exit_code"], c1["state_2027"], c1["file_untouched"], c1["audit_count"]
        ) or c0["last_line"] != c1["last_line"]:
            summary["cli_combo_diffs"].append({
                "prior": c0["prior"], "args": c0["args"],
                "base": {"rc": c0["exit_code"], "state": c0["state_2027"], "last": c0["last_line"], "audits": c0["audit_count"], "out": c0["output"]},
                "branch": {"rc": c1["exit_code"], "state": c1["state_2027"], "last": c1["last_line"], "audits": c1["audit_count"], "out": c1["output"]},
            })

    for a0, a1 in zip(base["auto_earn_cases"], branch["auto_earn_cases"]):
        f0 = {k: v for k, v in (a0["raw_flag"] or {}).items() if k != "updated_at"}
        f1 = {k: v for k, v in (a1["raw_flag"] or {}).items() if k != "updated_at"}
        if (a0["file_untouched"], a0["cur_year_state"], f0) != (a1["file_untouched"], a1["cur_year_state"], f1):
            summary["auto_earn_diffs"].append({
                "cmd": a0["cmd"], "src_kind": a0["src_kind"], "prior": a0["prior"], "best": a0["best"],
                "base": {"untouched": a0["file_untouched"], "state": a0["cur_year_state"], "flag": f0},
                "branch": {"untouched": a1["file_untouched"], "state": a1["cur_year_state"], "flag": f1},
            })

    out_json = REVIEW_DIR / "probe_matrix_results.json"
    out_json.write_text(json.dumps({
        "summary": summary,
        "base_row1": base["row1"],
        "branch_row1": branch["row1"],
        "branch_health_matrix": branch["health_matrix"],
        "branch_dash_cases": branch["dash_cases"],
    }, indent=2))
    print(f"Wrote {out_json}")
    print(f"decision_diffs: {len(summary['decision_diffs'])} / {len(base['decision_matrix'])}")
    print(f"item4_diffs: {len(summary['item4_diffs'])} / {len(base['item4_cases'])}")
    print(f"item4_deliver_base: {summary['item4_deliver_base']}")
    print(f"item4_deliver_branch: {summary['item4_deliver_branch']}")
    print(f"item4_orchestrate_base: {summary['item4_orchestrate_base']}")
    print(f"item4_orchestrate_branch: {summary['item4_orchestrate_branch']}")
    print(f"item4_neg45_base: {summary['item4_neg45_base']}")
    print(f"item4_neg45_branch: {summary['item4_neg45_branch']}")
    print(f"health_diffs_count: {summary['health_diffs_count']} / {len(base['health_matrix'])}")
    print(f"cmd_paste_branch: {len(summary['cmd_paste_branch'])}, non-zero exits: {sum(1 for x in summary['cmd_paste_branch'] if x['exit_code'] != 0)}, remaining alerts: {sum(1 for x in summary['cmd_paste_branch'] if x['remaining_saver_alerts'])}")
    print(f"dash_diffs: {len(summary['dash_diffs'])}")
    print(f"cli_combo_diffs: {len(summary['cli_combo_diffs'])}")
    print(f"auto_earn_diffs: {len(summary['auto_earn_diffs'])}")


if __name__ == "__main__":
    main()
