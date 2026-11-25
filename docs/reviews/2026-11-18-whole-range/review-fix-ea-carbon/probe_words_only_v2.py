"""Probe item 3 (v2): Words-only differential check against 63669c42.
Extracts 63669c42:src via git archive into .carbon-review/wr-ea/base_63669c42,
confirms distinct bts.__file__, and compares all non-text behavior in isolated
subprocesses with socket network access refused.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt_mod
import inspect
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_SHA = "63669c42"
BASE_DIR = HERE / "base_63669c42"


def _run_worker(out_file: Path) -> None:
    forced = os.environ.get("FORCE_BTS_SRC")
    if forced:
        sys.path.insert(0, str(REPO))
        sys.path.insert(0, forced)
    import pandas as pd
    import bts
    from bts import (
        contest_state,
        daily_decision,
        orchestrator,
        picks,
        scheduler,
        strategy,
        web,
    )
    from bts.health import alert, late_delivery as health_late_delivery
    from scripts.audit.season_walk import walk

    ET = ZoneInfo("America/New_York")
    REAL_DT = dt_mod.datetime

    def _no_net(*a, **k):
        raise OSError("network refused in words-only probe")

    @contextlib.contextmanager
    def pinned(at: datetime):
        walk._Clock.at = at
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(dt_mod, "datetime", walk._Clock))
            for mod in (
                contest_state, daily_decision, orchestrator, picks, scheduler,
                strategy, web, alert, health_late_delivery,
            ):
                if getattr(mod, "datetime", None) is REAL_DT:
                    stack.enter_context(patch.object(mod, "datetime", walk._Clock))
            stack.enter_context(patch.object(scheduler, "_now_et", lambda: walk._Clock.at.astimezone(ET)))
            for m in ("connect", "connect_ex", "sendto"):
                stack.enter_context(patch.object(socket.socket, m, _no_net))
            stack.enter_context(patch.object(socket, "getaddrinfo", _no_net))
            yield

    report: dict = {
        "bts_file": str(Path(bts.__file__).resolve()),
        "scheduler_file": str(Path(scheduler.__file__).resolve()),
        "web_file": str(Path(web.__file__).resolve()),
        "strategy_file": str(Path(strategy.__file__).resolve()),
    }

    has_candidate_param = "candidate" in inspect.signature(scheduler.build_skip_summary).parameters

    class RaisingColumns:
        @property
        def columns(self):
            raise RuntimeError("columns boom")

    class RaisingGetitem:
        columns = ["batter_name", "team", "p_game_hit"]

        def __getitem__(self, key):
            raise RuntimeError("getitem boom")

    class RaisingClass:
        @property
        def __class__(self):
            raise RuntimeError("__class__ boom")

    class RaisingMapping(dict):
        def get(self, key, default=None):
            raise RuntimeError("get boom")

    good = pd.DataFrame([
        {"batter_name": "Removed", "team": "NYY", "p_game_hit": 0.999},
        {"batter_name": "Second", "team": "BOS", "p_game_hit": 0.750},
    ])
    frames = [
        ("none", None),
        ("raising_cols", RaisingColumns()),
        ("raising_getitem", RaisingGetitem()),
        ("empty_df", pd.DataFrame()),
        ("zero_rows", good.iloc[:0]),
        ("no_batter_name", good.drop(columns=["batter_name"])),
        ("no_p", good.drop(columns=["p_game_hit"])),
        ("no_team", good.drop(columns=["team"])),
        ("dup_idx", pd.concat([good.iloc[:1], good.iloc[:1]])),
        ("all_nan", good.assign(p_game_hit=float("nan"))),
        ("all_pos_inf", good.assign(p_game_hit=float("inf"))),
        ("all_neg_inf", good.assign(p_game_hit=float("-inf"))),
        ("non_numeric", good.assign(p_game_hit=["abc", "def"])),
        ("valid_two", good),
        ("nan_plus_valid", pd.DataFrame([
            {"batter_name": "Bad", "team": "X", "p_game_hit": float("nan")},
            {"batter_name": "Good", "team": "Y", "p_game_hit": 0.81},
        ])),
    ]
    dec_reach = strategy.PolicyDecision(
        "skip", "skip", "mdp", "reach57", 10, 80, 10, "trusted", None, None, None, None, 0.796
    )
    dec_tail = strategy.PolicyDecision(
        "skip", "skip", "mdp", "tail_season_best", 0, 8, 18, "trusted", 18, "sha", None,
        "season-best 18 can't be beaten", None
    )
    valid_cand = {"batter_name": "Chosen", "team": "LAD", "p_game_hit": 0.75}
    na_cand = {"batter_name": None, "team": pd.NA, "p_game_hit": 1.5}
    candidates = [
        ("none", None),
        ("int", 42),
        ("raising_class", RaisingClass()),
        ("raising_get", RaisingMapping(valid_cand)),
        ("empty_dict", {}),
        ("bool_p", {"batter_name": "A", "team": "T", "p_game_hit": True}),
        ("str_p", {"batter_name": "A", "team": "T", "p_game_hit": "0.75"}),
        ("nan_p", {"batter_name": "A", "team": "T", "p_game_hit": float("nan")},),
        ("inf_p", {"batter_name": "A", "team": "T", "p_game_hit": float("inf")}),
        ("valid", valid_cand),
        ("valid_na_out_of_range", na_cand),
    ]

    summary_rows = []
    for f_name, frame in frames:
        for s_val in (None, 0, 10, "not_int"):
            for bar_val in (None, 0.0, 0.796, float("nan"), float("inf")):
                for d_name, dec in (("none", None), ("reach", dec_reach), ("tail", dec_tail)):
                    cand_iter = candidates if has_candidate_param else [("none", None)]
                    for c_name, cand in cand_iter:
                        if has_candidate_param:
                            res = scheduler.build_skip_summary(frame, s_val, pick_bar=bar_val, decision=dec, candidate=cand)
                        else:
                            res = scheduler.build_skip_summary(frame, s_val, pick_bar=bar_val, decision=dec)
                        if res is None:
                            core = None
                        else:
                            core = {k: v for k, v in res.items()
                                    if k not in ("best_batter", "best_team", "best_p", "basis", "candidate_verified")}
                            if has_candidate_param:
                                assert res["basis"] == ("decision" if dec is not None else "no_decision")
                                want_ver = dec is not None and c_name in ("valid", "valid_na_out_of_range")
                                assert res["candidate_verified"] is want_ver
                        summary_rows.append({
                            "key": f"{f_name}|s={s_val!r}|b={bar_val!r}|d={d_name}",
                            "cand": c_name,
                            "is_none": res is None,
                            "core": core,
                        })
    report["summary_corpus_count"] = len(summary_rows)
    collapsed = {}
    for row in summary_rows:
        k = row["key"]
        entry = {"is_none": row["is_none"], "core": row["core"]}
        if k in collapsed:
            assert collapsed[k] == entry, (k, collapsed[k], entry)
        collapsed[k] = entry
    report["summary_collapsed"] = collapsed

    def norm_check_ret(ret: dict) -> dict:
        out = dict(ret)
        pr = out.get("pick_result")
        if pr is not None:
            d_dict = dataclasses.asdict(pr.daily)
            d_dict.pop("model_git_sha", None)
            out["pick_result"] = {"locked": pr.locked, "daily": d_dict}
        sel = out.get("selection")
        if sel is not None:
            s_dict = dataclasses.asdict(sel)
            if s_dict.get("pick_result") and s_dict["pick_result"].get("daily"):
                s_dict["pick_result"]["daily"].pop("model_git_sha", None)
            out["selection"] = s_dict
        if "skip_summary" in out and out["skip_summary"] is not None:
            out["skip_summary"] = {
                k: v for k, v in out["skip_summary"].items()
                if k not in ("best_batter", "best_team", "best_p", "basis", "candidate_verified")
            }
        return out

    check_cases = {}
    for sc in (
        "mdp_skip_removed_top", "tail_stop", "heuristic_skip", "no_eligible",
        "status_failure", "no_valid_predictions", "sel_none", "empty_predictions",
        "fresh_single_lock", "pre_cascade_scoreable_lock", "contest_state_error",
    ):
        with tempfile.TemporaryDirectory() as tmp:
            picks_dir = Path(tmp) / "picks"
            picks_dir.mkdir()
            day = date(2027, 9, 19) if sc == "tail_stop" else date(2027, 7, 15)
            now = REAL_DT(day.year, day.month, day.day, 16, 0, tzinfo=ET)
            cfg = walk._config(picks_dir)
            cfg["orchestrator"]["heartbeat_path"] = str(Path(tmp) / ".hb")
            frame = walk._predictions(day, 0.76)
            frame.loc[0, "p_game_hit"] = 0.999
            if sc == "fresh_single_lock":
                frame = walk._predictions(day, 0.95)
            elif sc == "no_valid_predictions":
                frame.loc[1:, "p_game_hit"] = float("nan")
            elif sc == "empty_predictions":
                frame = pd.DataFrame()
            statuses = {
                70000: {"abstract": "F", "detailed": "Final"},
                70001: {"abstract": "P", "detailed": "Scheduled"},
                70002: {"abstract": "P", "detailed": "Scheduled"},
            }
            if sc == "fresh_single_lock":
                statuses[70000] = {"abstract": "P", "detailed": "Scheduled"}
            elif sc == "no_eligible":
                statuses = {pk: {"abstract": "F", "detailed": "Final"} for pk in statuses}

            def read_st(_d, sc=sc, statuses=statuses):
                if sc == "status_failure":
                    raise OSError("status offline")
                return statuses

            with pinned(now), \
                    patch.object(orchestrator, "run_cascade", lambda *a, f=frame: (f.copy(), "local")), \
                    patch.object(picks, "get_game_statuses_detailed", read_st), \
                    patch.object(scheduler, "count_new_confirmations", lambda *a, **k: 1), \
                    patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext()), \
                    patch.object(alert, "send_dm", lambda *a: "a"), \
                    contextlib.redirect_stderr(io.StringIO()):
                if sc != "contest_state_error":
                    st_install = (walk.State("dated", 0, 18, "not_earned", 0.75)
                                  if sc == "tail_stop"
                                  else walk.State("dated", 10, 10, "used", 0.75))
                    assert walk.install_state(picks_dir, day, st_install) == []
                if sc == "pre_cascade_scoreable_lock":
                    dp = picks.DailyPick(
                        date=day.isoformat(), run_time=now.isoformat(),
                        pick=picks.Pick("P1", 1, "CLE", 1, "PA", 2, 0.85, [], False, 70001, "2027-07-15T23:05:00Z"),
                        double_down=None, runner_up=None, notification_sent=True, notification_id="id1",
                    )
                    picks.save_pick(dp, picks_dir)
                    daily_decision.write_decision(
                        day.isoformat(), picks_dir, action="single", source="mdp",
                        primary={"batter_id": 1, "batter_name": "P1", "team": "CLE", "game_pk": 70001, "p_game_hit": 0.85},
                        double_down=None, delivery_status="delivered", scoreable=True,
                    )
                if sc == "heuristic_skip":
                    with patch.object(strategy, "_load_mdp", lambda: None):
                        ret = scheduler.run_single_check(day.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
                elif sc == "sel_none":
                    with patch.object(orchestrator, "run_and_pick", lambda *a, **k: (frame.copy(), None, "local")):
                        ret = scheduler.run_single_check(day.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
                else:
                    ret = scheduler.run_single_check(day.isoformat(), [70000, 70001, 70002], set(), cfg, 0.03)
            check_cases[sc] = norm_check_ret(ret)
    report["run_single_check"] = check_cases

    lifecycle = {}
    for mode in ("dm", "public", "private"):
        for outcome in ("ok", "no_id", "err", "kb", "sysexit"):
            with tempfile.TemporaryDirectory() as tmp:
                pdir = Path(tmp) / "picks"
                pdir.mkdir()
                st = scheduler.SchedulerState(
                    date="2027-07-15", schedule_fetched_at="2027-07-15T10:00:00-04:00",
                    games=[], confirmed_game_pks=[], runs_completed=[],
                    pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                    skip_summary={"best_batter": "A", "best_team": "T", "best_p": 0.75, "streak": 10},
                )
                calls = []

                def fake_send(rec, body, outcome=outcome, calls=calls):
                    calls.append(rec)
                    if len(calls) == 1:
                        if outcome == "err":
                            raise ValueError("fail")
                        if outcome == "kb":
                            raise KeyboardInterrupt("kb")
                        if outcome == "sysexit":
                            raise SystemExit(3)
                        if outcome == "no_id":
                            return None
                    return "msg_1"

                cfg = {"scheduler": {"pick_delivery": mode}, "bluesky": {"dm_recipient": "op.test"}}
                raised = None
                ret1 = None
                with contextlib.redirect_stderr(io.StringIO()):
                    try:
                        ret1 = scheduler.maybe_notify_skip(st, st.skip_summary, cfg, now_iso="2027-07-15T12:00:00-04:00", send=fake_send)
                    except BaseException as exc:
                        raised = type(exc).__name__
                scheduler.save_state(st, pdir)
                restarted = scheduler.SchedulerState(
                    date="2027-07-15", schedule_fetched_at="2027-07-15T13:00:00-04:00",
                    games=[], confirmed_game_pks=[], runs_completed=[],
                    pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                scheduler.carry_forward_skip_state(restarted, scheduler.load_state("2027-07-15", pdir))
                with contextlib.redirect_stderr(io.StringIO()):
                    ret2 = scheduler.maybe_notify_skip(restarted, restarted.skip_summary, cfg, now_iso="2027-07-15T13:05:00-04:00", send=fake_send)
                lifecycle[f"{mode}|{outcome}"] = {
                    "ret1": ret1, "raised": raised, "notified_after_1": st.skip_notified_at,
                    "ret2": ret2, "notified_after_2": restarted.skip_notified_at, "calls": calls,
                }
    for gate_case in ("no_skip", "has_skip_summary", "has_final_skip"):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            now = REAL_DT(2027, 7, 15, 19, 2, tzinfo=ET)
            dp = picks.DailyPick(
                date="2027-07-15", run_time=now.isoformat(),
                pick=picks.Pick("P1", 1, "CLE", 1, "PA", 2, 0.85, [], False, 70001, "2027-07-15T23:10:00Z"),
                double_down=None, runner_up=None,
            )
            picks.save_pick(dp, pdir)
            st = scheduler.SchedulerState(
                date="2027-07-15", schedule_fetched_at=now.isoformat(), games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                pick_locked_at=None, result_status=None, next_wakeup=None,
                skip_summary={"best_batter": "A", "best_team": "T", "best_p": 0.75, "streak": 10} if gate_case == "has_skip_summary" else None,
                final_skip_candidate={"primary": {}} if gate_case == "has_final_skip" else None,
            )
            mp_sent = []
            cfg = walk._config(pdir)
            cfg["bluesky"] = {"dm_recipient": "op.test"}
            with pinned(now), \
                    patch.object(alert, "send_dm", lambda r, b: mp_sent.append(b) or "mp1"), \
                    contextlib.redirect_stderr(io.StringIO()):
                scheduler._maybe_alert_missed_pick(cfg, "2027-07-15", pdir, 10, None, st)
            lifecycle[f"missed_pick|{gate_case}"] = mp_sent
    report["skip_lifecycle"] = lifecycle

    warn_effects = {}
    for route_seq in ("morn_then_deliv", "morn_fail_then_deliv", "deliv_after_midnight", "morn_jan1"):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            if route_seq == "deliv_after_midnight":
                now = REAL_DT(2027, 7, 16, 0, 5, tzinfo=ET)
                day_str = "2027-07-15"
            elif route_seq == "morn_jan1":
                now = REAL_DT(2028, 1, 1, 10, 0, tzinfo=ET)
                day_str = "2028-01-01"
            else:
                now = REAL_DT(2027, 7, 15, 10, 0, tzinfo=ET)
                day_str = "2027-07-15"
            st_dec = contest_state.DecisionStreakState(
                10, True, True, "contest", "fresh", 0, True,
                observation_kind="automatic", contest_recorded_at="2027-07-14T00:00:00Z",
            )
            rec = contest_state.automatic_reading_record(st_dec, now)
            dispatched = []

            def fake_alert_dm(recipient, body, route_seq=route_seq, dispatched=dispatched):
                dispatched.append(recipient)
                if route_seq == "morn_fail_then_deliv" and len(dispatched) == 1:
                    raise RuntimeError("morn send failed")
                return "ok_id"

            with pinned(now), \
                    patch.object(contest_state, "load_decision_streak_state", lambda *a, **k: st_dec), \
                    patch.object(alert, "send_dm", fake_alert_dm), \
                    contextlib.redirect_stderr(io.StringIO()):
                cfg = {"bluesky": {"dm_recipient": "op.test"}}
                if route_seq in ("morn_then_deliv", "morn_fail_then_deliv", "morn_jan1"):
                    scheduler._morning_state_check(cfg, pdir, day_str, now, now + timedelta(hours=2))
                if route_seq in ("morn_then_deliv", "morn_fail_then_deliv", "deliv_after_midnight"):
                    scheduler._warn_stale_contest_reading(rec, cfg, pdir)
            st_file = Path(tmp) / "health_state" / "contest_stale_reading_dm_delivery_status.json"
            st_json = json.loads(st_file.read_text())
            st_json.pop("body_sha256", None)
            st_json.pop("body_first_line", None)
            warn_effects[route_seq] = {"calls": len(dispatched), "status_file": st_json}

    containment = {}
    for exc_cls in (RuntimeError, KeyboardInterrupt, SystemExit):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            now = REAL_DT(2027, 7, 15, 10, 0, tzinfo=ET)
            st_dec = contest_state.DecisionStreakState(
                10, True, True, "contest", "fresh", 0, True,
                observation_kind="automatic", contest_recorded_at="2027-07-14T00:00:00Z",
            )
            rec = contest_state.automatic_reading_record(st_dec, now)

            def raise_exc(*a, exc_cls=exc_cls, **k):
                raise exc_cls("boom")

            deliv_escaped = None
            with pinned(now), \
                    patch.object(alert, "dispatch_dm_for_health_alerts", raise_exc), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    scheduler._warn_stale_contest_reading(rec, {"bluesky": {"dm_recipient": "op.test"}}, pdir)
                except BaseException as e:
                    deliv_escaped = type(e).__name__

            morn_escaped = None
            cfg = walk._config(pdir)
            cfg["bluesky"] = {"dm_recipient": "op.test"}
            cfg["orchestrator"]["heartbeat_path"] = str(Path(tmp) / ".hb")
            game = {"gamePk": 70001, "gameType": "R", "gameDate": "2027-07-15T23:10:00Z"}
            with pinned(now), \
                    patch.object(scheduler, "fetch_schedule", lambda _d: [game]), \
                    patch.object(scheduler, "_morning_state_check", raise_exc), \
                    patch.object(scheduler, "_watchdog_ping_sleep", lambda _s: None), \
                    patch.object(scheduler, "run_single_check", lambda **k: {
                        "skipped": False, "new_lineups": 0, "should_post": False,
                        "pick_result": None, "pick_name": None, "pick_p": None, "selection": None,
                    }), \
                    patch.object(scheduler, "_idle_until_next_wakeup", lambda *a: None), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    scheduler.run_day("2027-07-15", cfg)
                except BaseException as e:
                    morn_escaped = type(e).__name__
            containment[exc_cls.__name__] = {"delivery_escaped": deliv_escaped, "morning_run_day_escaped": morn_escaped}
    warn_effects["containment"] = containment
    report["stale_warnings"] = warn_effects

    refusal_effects = {}
    for kind in ("late", "void"):
        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            now = REAL_DT(2027, 7, 15, 19, 8, tzinfo=ET)
            cutoff = REAL_DT(2027, 7, 15, 19, 5, tzinfo=ET)
            dp = picks.DailyPick(
                date="2027-07-15", run_time="2027-07-15T18:25:00-04:00",
                pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "PA", 2, 0.82, [], False, 70001, "2027-07-15T23:10:00Z"),
                double_down=None, runner_up=None,
            )
            picks.save_pick(dp, pdir)
            st = scheduler.SchedulerState(
                date="2027-07-15", schedule_fetched_at=now.isoformat(), games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            captured_alerts = []
            with pinned(now), \
                    patch.object(alert, "dispatch_dm_for_health_alerts",
                                 lambda al, rec, status_path=None, **kw: captured_alerts.append(
                                     [(a.level, a.source) for a in al] + [str(Path(status_path).name)]
                                 )), \
                    contextlib.redirect_stderr(io.StringIO()):
                if kind == "late":
                    scheduler._refuse_late_delivery(dp, {"bluesky": {"dm_recipient": "op.test"}}, pdir, st, "2027-07-15", "lineup", cutoff, now)
                else:
                    scheduler._refuse_void_game_delivery(dp, {"bluesky": {"dm_recipient": "op.test"}}, pdir, st, "2027-07-15", "lineup", {70001: "Postponed"})
                eod_alerts = [(a.level, a.source, a.message) for a in health_late_delivery.check(pdir, today=date(2027, 7, 15))]
            saved_st = scheduler.load_state("2027-07-15", pdir)
            arch_path = pdir / "2027-07-15" / saved_st.delivery_refusals[0]["archive"]
            arch_data = json.loads(arch_path.read_text())
            arch_data.pop("model_git_sha", None)
            refusal_effects[kind] = {
                "pick_gone": picks.load_pick("2027-07-15", pdir) is None,
                "pick_locked": saved_st.pick_locked,
                "delivery_refusals": saved_st.delivery_refusals,
                "archive_data": arch_data,
                "dispatched_meta": captured_alerts,
                "eod_late_delivery": eod_alerts,
            }
    report["refusals"] = refusal_effects

    mdp_art = strategy._load_mdp()
    end = walk.LAST_GAME_DAY + timedelta(days=1)
    bars = []
    for days in [0, *range(1, 181), 181, 200]:
        day_s = (end - timedelta(days=days)).isoformat()
        for s in range(57):
            regime, _ = strategy._regime(mdp_art, s, day_s)
            bests = [(s, "trusted")]
            if regime == strategy.OBJECTIVE_TAIL:
                bests.extend([(56, "trusted"), (56, "untrusted"), (None, "missing")])
            for sav in (False, True):
                for b_val, trust in bests:
                    bars.append(strategy.effective_pick_bar(s, day_s, sav, mdp_art, b_val, trust))
    report["bars_count"] = len(bars)
    report["bars"] = bars

    out_file.write_text(json.dumps(report))


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _run_worker(Path(sys.argv[2]))
        return

    diff_res = subprocess.run(
        ["git", "diff", BASE_SHA, "HEAD", "--", "src/bts/strategy.py"],
        cwd=REPO, check=True, capture_output=True, text=True, timeout=10,
    )
    assert diff_res.stdout == "", f"strategy.py has diff: {diff_res.stdout}"

    if BASE_DIR.exists():
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir(parents=True)
    arch = subprocess.run(
        ["git", "archive", BASE_SHA, "src"],
        cwd=REPO, check=True, capture_output=True, timeout=15,
    )
    with tarfile.open(fileobj=io.BytesIO(arch.stdout), mode="r:") as tf:
        tf.extractall(BASE_DIR, filter="data")

    base_out = HERE / "words_only_base.json"
    head_out = HERE / "words_only_head.json"

    for src_root, out_path in ((BASE_DIR / "src", base_out), (REPO / "src", head_out)):
        env = dict(os.environ)
        env["FORCE_BTS_SRC"] = str(src_root.resolve())
        env["PYTHONPATH"] = f"{src_root.resolve()}{os.pathsep}{REPO}"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(out_path)],
            cwd=REPO, env=env, check=True, timeout=90,
        )

    base_data = json.loads(base_out.read_text())
    head_data = json.loads(head_out.read_text())

    assert base_data["bts_file"] == str((BASE_DIR / "src/bts/__init__.py").resolve())
    assert head_data["bts_file"] == str((REPO / "src/bts/__init__.py").resolve())
    assert base_data["bts_file"] != head_data["bts_file"]

    keys_to_compare = [
        "summary_collapsed",
        "run_single_check",
        "skip_lifecycle",
        "stale_warnings",
        "refusals",
        "bars_count",
        "bars",
    ]
    for k in keys_to_compare:
        assert base_data[k] == head_data[k], f"Mismatch in {k}!"

    summary = {
        "strategy_py_diff_bytes": len(diff_res.stdout),
        "base_bts_file": base_data["bts_file"],
        "head_bts_file": head_data["bts_file"],
        "base_summary_calls": base_data["summary_corpus_count"],
        "head_summary_calls": head_data["summary_corpus_count"],
        "collapsed_summary_states": len(head_data["summary_collapsed"]),
        "run_single_check_scenarios": list(head_data["run_single_check"].keys()),
        "skip_lifecycle_cases": len(head_data["skip_lifecycle"]),
        "stale_warning_containment": head_data["stale_warnings"]["containment"],
        "bars_compared": head_data["bars_count"],
        "all_equal": True,
    }
    (HERE / "probe_words_only_summary.json").write_text(json.dumps(summary, indent=2))
    print("probe_words_only_v2: ALL PARITY CHECKS PASSED:", json.dumps(summary))


if __name__ == "__main__":
    main()
