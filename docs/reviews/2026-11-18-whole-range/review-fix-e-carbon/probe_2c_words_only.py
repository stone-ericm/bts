"""Probe 2c: Words-only differential verification against base commit 63669c42.

Runs identical corpora against both `.carbon-review/wr-e/base_63669c42/src` and HEAD `src`,
confirming `bts.__file__` in each subprocess, and checks every non-text behavior named in brief item 2.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_SRC = REPO / ".carbon-review/wr-e/base_63669c42/src"
HEAD_SRC = REPO / "src"

WORKER_CODE = r"""
import io
import json
import math
import sys
import tempfile
from contextlib import ExitStack, nullcontext, redirect_stderr
from datetime import date, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import bts
from bts import contest_state, orchestrator, picks, scheduler, strategy
from bts.health import alert, late_delivery
from scripts.audit.season_walk import walk

ET = ZoneInfo("America/New_York")
mode = sys.argv[1]
expected_root = Path(sys.argv[2]).resolve()
actual_file = Path(bts.__file__).resolve()
assert actual_file.is_relative_to(expected_root), f"{actual_file} not under {expected_root}"

out = {"bts_file": str(actual_file)}

class RaisingCols:
    @property
    def columns(self):
        raise RuntimeError("columns boom")

class RaisingGetItem:
    columns = ["batter_name", "team", "p_game_hit"]
    def __getitem__(self, key):
        raise KeyError("getitem boom")

class RaisingDecision:
    objective = "tail"
    @property
    def reason(self):
        raise ValueError("reason boom")

class RaisingCandidateGet(dict):
    def get(self, key, default=None):
        if key == "team":
            raise RuntimeError("team boom")
        return super().get(key, default)

class RaisingCandidateProb(dict):
    def get(self, key, default=None):
        if key == "p_game_hit":
            raise RuntimeError("prob boom")
        return super().get(key, default)

good_df = pd.DataFrame([
    {"batter_name": "TopRemoved", "team": "NYY", "p_game_hit": 0.991},
    {"batter_name": "EligibleRunner", "team": "CLE", "p_game_hit": 0.752},
])
na_name_df = pd.DataFrame([{"batter_name": pd.NA, "team": None, "p_game_hit": 0.812}])
no_team_df = pd.DataFrame([{"batter_name": "Solo", "p_game_hit": 0.654}])
dup_idx_df = pd.concat([good_df.iloc[[0]], good_df.iloc[[0]]])

frames = [
    ("none", None),
    ("int", 42),
    ("dict", {"batter_name": ["A"], "p_game_hit": [0.8]}),
    ("raising_cols", RaisingCols()),
    ("raising_getitem", RaisingGetItem()),
    ("empty_df", pd.DataFrame()),
    ("zero_rows", good_df.iloc[:0]),
    ("no_p_col", good_df.drop(columns=["p_game_hit"])),
    ("no_name_col", good_df.drop(columns=["batter_name"])),
    ("all_nan", good_df.assign(p_game_hit=float("nan"))),
    ("all_pos_inf", good_df.assign(p_game_hit=float("inf"))),
    ("all_neg_inf", good_df.assign(p_game_hit=float("-inf"))),
    ("all_non_numeric", good_df.assign(p_game_hit="not_a_number")),
    ("dup_idx", dup_idx_df),
    ("good", good_df),
    ("na_name", na_name_df),
    ("no_team", no_team_df),
    ("mixed_nan_finite", pd.DataFrame([
        {"batter_name": "NaNBatter", "team": "BOS", "p_game_hit": float("nan")},
        {"batter_name": "InfBatter", "team": "BOS", "p_game_hit": float("inf")},
        {"batter_name": "FiniteBatter", "team": "BOS", "p_game_hit": 0.745},
    ])),
]

tail_skip_dec = strategy.PolicyDecision(
    action="skip", policy_action="skip", source="mdp", objective="tail",
    streak=0, days_effective=8, best_supplied=18, best_status="trusted",
    effective_best=18, tail_sha256="deadbeef", degraded_reason=None,
    reason="season-best 18 can't be beaten with 8 days left (max reachable 16); no pick",
    pick_bar=None,
)
reach57_skip_dec = strategy.PolicyDecision(
    action="skip", policy_action="skip", source="mdp", objective="reach57",
    streak=10, days_effective=100, best_supplied=10, best_status="trusted",
    effective_best=None, tail_sha256=None, degraded_reason=None,
    reason=None, pick_bar=0.796,
)

decisions = [("none", None), ("tail_skip", tail_skip_dec), ("reach57_skip", reach57_skip_dec), ("raising_dec", RaisingDecision())]
candidates = [
    ("missing", None),
    ("non_mapping_int", 123),
    ("series", pd.Series({"batter_name": "SeriesB", "team": "SEA", "p_game_hit": 0.71})),
    ("empty_dict", {}),
    ("bool_p", {"batter_name": "Keep", "team": "CLE", "p_game_hit": True}),
    ("str_p", {"batter_name": "Keep", "team": "CLE", "p_game_hit": "0.75"}),
    ("nan_p", {"batter_name": "Keep", "team": "CLE", "p_game_hit": float("nan")}),
    ("inf_p", {"batter_name": "Keep", "team": "CLE", "p_game_hit": float("inf")}),
    ("overflow_p", {"batter_name": "Keep", "team": "CLE", "p_game_hit": 10**400}),
    ("raising_prob", RaisingCandidateProb({"batter_name": "Keep", "team": "CLE", "p_game_hit": 0.75})),
    ("raising_team", RaisingCandidateGet({"batter_name": "Keep", "team": "CLE", "p_game_hit": 0.75})),
    ("valid_float", {"batter_name": "Keep", "team": "CLE", "p_game_hit": 0.752}),
    ("valid_np_float", {"batter_name": "Keep", "team": pd.NA, "p_game_hit": np.float64(0.752)}),
    ("valid_fraction", {"batter_name": None, "team": 99, "p_game_hit": Fraction(3, 4)}),
]

summary_results = []
for f_name, frame in frames:
    for s_val in (None, 0, 10, "not_an_int"):
        for b_val in (None, 0.796, float("nan"), float("inf"), 10**400):
            for d_name, dec in decisions:
                for c_name, cand in candidates:
                    if mode == "base":
                        res = scheduler.build_skip_summary(frame, s_val, pick_bar=b_val, decision=dec)
                    else:
                        res = scheduler.build_skip_summary(frame, s_val, pick_bar=b_val, decision=dec, candidate=cand)
                    if res is not None:
                        json.dumps(res, allow_nan=False)
                    summary_results.append({
                        "case": f"{f_name}|s={s_val}|b={'big' if b_val == 10**400 else b_val}|{d_name}|{c_name}",
                        "is_none": res is None,
                        "res": res,
                    })
out["summary_count"] = len(summary_results)
out["summaries"] = summary_results

def mk_pick(name, bid, team, gpk, p=0.82):
    return picks.Pick(name, bid, team, 1, "P1", 901, p, [], False, gpk, "2027-07-15T23:05:00Z", "BOS")

def raise_oserror(_d):
    raise OSError("network down")

check_cases = []
for case_name in ("removed_k2", "k3_top_bin", "k4_all_skip", "k1_tail_stop", "none_eligible",
                  "status_failure", "sel_none", "contest_error", "single_pick", "double_clamped",
                  "pre_lock_scoreable"):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        config = walk._config(picks_dir)
        config["orchestrator"]["heartbeat_path"] = str(tmp_path / "hb")
        if case_name == "k3_top_bin":
            day = date(2027, 8, 30)
            st = walk.State("dated", 56, 56, "used", 0.99)
            frame = walk._predictions(day, 0.99)
        elif case_name == "k4_all_skip":
            day = date(2027, 7, 15)
            st = walk.State("dated", 56, 56, "used", 0.96)
            frame = walk._predictions(day, 0.96)
        elif case_name == "k1_tail_stop":
            day = date(2027, 9, 19)
            st = walk.State("dated", 0, 18, "used", 0.75)
            frame = walk._predictions(day, 0.75)
            frame.loc[0, "p_game_hit"] = 0.999
        elif case_name == "single_pick":
            day = date(2027, 7, 15)
            st = walk.State("dated", 12, 12, "active", 0.85)
            frame = walk._predictions(day, 0.85)
        elif case_name == "double_clamped":
            day = date(2027, 7, 15)
            st = walk.State("dated", 0, 0, "not_earned", 0.85)
            frame = walk._predictions(day, 0.85)
            frame["game_pk"] = 70000
        else:
            day = date(2027, 7, 15)
            st = walk.State("dated", 10, 10, "used", 0.76)
            frame = walk._predictions(day, 0.76)
            frame.loc[0, "p_game_hit"] = 0.999

        now = datetime(day.year, day.month, day.day, 10, 0, tzinfo=ET)
        statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in (70000, 70001, 70002)}
        if case_name in ("removed_k2", "k1_tail_stop"):
            statuses[70000] = {"abstract": "F", "detailed": "Final"}
        elif case_name == "none_eligible":
            statuses = {k: {"abstract": "F", "detailed": "Final"} for k in statuses}

        with walk.offline(now), \
             patch.object(orchestrator, "run_cascade", lambda *a, _f=frame: (_f.copy(), "test")), \
             patch.object(picks, "get_game_statuses_detailed", raise_oserror if case_name == "status_failure" else (lambda _d, _s=statuses: _s)), \
             patch.object(picks, "get_game_statuses", raise_oserror if case_name == "status_failure" else (lambda _d, _s=statuses: {k: v["abstract"] for k, v in _s.items()})), \
             patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 2), \
             patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()), \
             patch.object(scheduler, "_alert_contest_state_failure", lambda *a, **kw: None):
            if case_name != "contest_error":
                assert walk.install_state(picks_dir, day, st) == []
            if case_name == "pre_lock_scoreable":
                dp = picks.DailyPick(
                    date=day.isoformat(), run_time=now.isoformat(),
                    pick=mk_pick("LockedB", 101, "NYY", 70001, 0.82),
                    double_down=None, runner_up=None, bluesky_posted=True, bluesky_uri="dm://1"
                )
                picks.save_pick(dp, picks_dir)
                (picks_dir / day.isoformat()).mkdir(exist_ok=True)
                (picks_dir / day.isoformat() / "decision.json").write_text(json.dumps({
                    "schema_version": "bts_daily_decision_v2", "date": day.isoformat(),
                    "action": "single", "source": "mdp", "scoreable": True,
                    "delivery_status": "delivered",
                }))
            stack = ExitStack()
            if case_name == "sel_none":
                stack.enter_context(patch.object(orchestrator, "run_and_pick", lambda *a, _f=frame, **kw: (_f.copy(), None, "test")))
            with stack, redirect_stderr(io.StringIO()):
                res = scheduler.run_single_check(
                    date=day.isoformat(), all_game_pks=[70000, 70001, 70002],
                    confirmed_sides=set(), config=config, early_lock_gap=0.03,
                )
        norm = {
            "keys": sorted(res.keys()),
            "skipped": res["skipped"],
            "new_lineups": res["new_lineups"],
            "should_post": res["should_post"],
            "pick_name": res["pick_name"],
            "pick_p": res["pick_p"],
            "has_pick_result": res["pick_result"] is not None,
            "pick_locked": res["pick_result"].locked if res["pick_result"] is not None else None,
            "daily_batter": res["pick_result"].daily.pick.batter_name if res["pick_result"] is not None else None,
            "daily_dd": (res["pick_result"].daily.double_down.batter_name
                         if res["pick_result"] is not None and res["pick_result"].daily.double_down else None),
            "sel_action": res["selection"].action if res["selection"] is not None else None,
            "sel_reason": res["selection"].no_pick_reason if res["selection"] is not None else None,
            "skip_summary": res.get("skip_summary"),
        }
        check_cases.append({"case": case_name, "norm": norm})
out["check_cases"] = check_cases

# 3. _maybe_alert_missed_pick gate equivalence
missed_cases = []
for mp_case in ("none", "with_summary", "with_final_skip"):
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        st_obj = scheduler.SchedulerState(
            date="2027-07-15", schedule_fetched_at="2027-07-15T10:00:00-04:00",
            games=[], confirmed_game_pks=[], runs_completed=[],
            pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            skip_summary={"best_batter": "A", "best_team": "T", "best_p": 0.75, "streak": 5} if mp_case == "with_summary" else None,
            final_skip_candidate={"primary": None} if mp_case == "with_final_skip" else None,
        )
        def fail_fetch(_d):
            raise AssertionError("gate passed")
        with patch.object(scheduler, "fetch_schedule", fail_fetch):
            try:
                scheduler._maybe_alert_missed_pick(
                    {"orchestrator": {"picks_dir": str(picks_dir)}},
                    "2027-07-15", picks_dir, 10, None, st_obj,
                )
                gated = True
            except AssertionError:
                gated = False
        missed_cases.append({"case": mp_case, "gated": gated})
out["missed_cases"] = missed_cases

# 4. Stale warnings metadata, dedup, retry, midnight/Jan 1, and interrupt containment
stale_meta = []
for route in ("delivery_old", "morning_old", "delivery_interrupt", "morning_interrupt", "cross_dedup", "retry_after_fail", "jan1_boundary"):
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        health_dir = Path(tmp) / "health_state"
        cfg = {"bluesky": {"dm_recipient": "op.bsky.social"}, "orchestrator": {"picks_dir": str(picks_dir)}}
        now = datetime(2027, 12, 31, 23, 55, tzinfo=ET) if route == "jan1_boundary" else datetime(2027, 7, 15, 10, 0, tzinfo=ET)
        ds = contest_state.DecisionStreakState(
            10, False, False, "contest", "fresh", None, None,
            observation_kind="automatic", contest_recorded_at="2027-07-14T10:00:00+00:00",
        )
        rec = contest_state.automatic_reading_record(ds, now)
        sent_dms = []
        calls = [0]
        def fake_send(rcpt, body, _route=route, _calls=calls, _sent=sent_dms):
            _calls[0] += 1
            if _route in ("delivery_interrupt", "morning_interrupt"):
                raise KeyboardInterrupt("sigint in dm")
            if _route == "retry_after_fail" and _calls[0] == 1:
                raise RuntimeError("transient network failure")
            _sent.append((rcpt, body))
            return "msg-1"
        with patch.object(contest_state, "load_decision_streak_state", lambda *a, **kw: ds), \
             patch.object(scheduler, "_now_et", lambda: now), \
             patch.object(alert, "send_dm", fake_send), \
             redirect_stderr(io.StringIO()):
            if route == "delivery_old":
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                status = json.loads((health_dir / "contest_stale_reading_dm_delivery_status.json").read_text())
                status.pop("updated_at", None)
                stale_meta.append({"route": route, "sent_count": len(sent_dms), "status": status})
            elif route == "morning_old":
                scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                status = json.loads((health_dir / "contest_stale_reading_dm_delivery_status.json").read_text())
                status.pop("updated_at", None)
                stale_meta.append({"route": route, "sent_count": len(sent_dms), "status": status})
            elif route == "delivery_interrupt":
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "contained": True})
            elif route == "morning_interrupt":
                try:
                    scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                    contained = True
                except KeyboardInterrupt:
                    contained = False
                stale_meta.append({"route": route, "contained": contained})
            elif route == "cross_dedup":
                scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "sent_count": len(sent_dms)})
            elif route == "retry_after_fail":
                scheduler._morning_state_check(cfg, picks_dir, now.date().isoformat(), now, now + timedelta(hours=1))
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "calls": calls[0], "sent_count": len(sent_dms)})
            elif route == "jan1_boundary":
                scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                now_jan1 = datetime(2028, 1, 1, 0, 5, tzinfo=ET)
                with patch.object(scheduler, "_now_et", lambda: now_jan1):
                    scheduler._warn_stale_contest_reading(rec, cfg, picks_dir)
                stale_meta.append({"route": route, "sent_count": len(sent_dms), "d1_has_dec31": "2027-12-31" in sent_dms[0][1], "d2_has_jan1": "2028-01-01" in sent_dms[1][1]})
out["stale_meta"] = stale_meta

# 5. Refusals side-effects & EOD late_delivery check
refusal_meta = []
for rkind in ("late", "void"):
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        now = datetime(2027, 7, 15, 19, 10, tzinfo=ET)
        dp = picks.DailyPick(
            date="2027-07-15", run_time=now.isoformat(),
            pick=mk_pick("Steven Kwan", 101, "CLE", 70001, 0.82),
            double_down=None, runner_up=None, bluesky_posted=False, bluesky_uri=None,
        )
        picks.save_pick(dp, picks_dir)
        st_obj = scheduler.SchedulerState(
            date="2027-07-15", schedule_fetched_at=now.isoformat(),
            games=[], confirmed_game_pks=[], runs_completed=[],
            pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        captured = []
        with patch.object(scheduler, "_now_et", lambda: now), \
             patch.object(alert, "send_dm", lambda r, b, _c=captured: (_c.append((r, b)) or "id-1")), \
             redirect_stderr(io.StringIO()):
            if rkind == "late":
                scheduler._refuse_late_delivery(dp, {"bluesky": {"dm_recipient": "op"}}, picks_dir, st_obj, "2027-07-15", "fallback", now - timedelta(minutes=10), now)
            else:
                scheduler._refuse_void_game_delivery(dp, {"bluesky": {"dm_recipient": "op"}}, picks_dir, st_obj, "2027-07-15", "fallback", {70001: "Postponed"})
        eod_alerts = [(a.level, a.source, a.message) for a in late_delivery.check(picks_dir, today=date(2027, 7, 15))]
        status_doc = json.loads((Path(tmp) / "health_state/health_dm_delivery_status.json").read_text())
        status_doc.pop("updated_at", None)
        saved_st = scheduler.load_state("2027-07-15", picks_dir)
        archive_doc = json.loads((picks_dir / "2027-07-15" / saved_st.delivery_refusals[0]["archive"]).read_text())
        refusal_meta.append({
            "kind": rkind,
            "pick_gone": picks.load_pick("2027-07-15", picks_dir) is None,
            "pick_locked": saved_st.pick_locked,
            "refusal_record": saved_st.delivery_refusals,
            "archive_refusal": archive_doc["refused_delivery"],
            "status_doc": status_doc,
            "eod_alerts": eod_alerts,
        })
out["refusal_meta"] = refusal_meta

# 6. effective_pick_bar across all streaks 0..56, days 0..185, both savers, both regimes
mdp = strategy._load_mdp()
end = date(2027, 9, 27)
bars = []
regimes_seen = set()
for days in range(0, 186):
    day_str = (end - timedelta(days=days)).isoformat()
    for streak in range(57):
        regime, _ = strategy._regime(mdp, streak, day_str)
        regimes_seen.add(regime)
        best_variants = [(streak, "trusted")]
        if regime == strategy.OBJECTIVE_TAIL:
            best_variants.extend([(56, "trusted"), (56, "untrusted"), (None, None)])
        for saver in (False, True):
            for b_strk, b_trust in best_variants:
                val = strategy.effective_pick_bar(streak, day_str, saver, mdp, b_strk, b_trust)
                bars.append(val)
out["bar_count"] = len(bars)
out["regimes_seen"] = sorted(regimes_seen)
out["bars"] = bars

print(json.dumps(out))
"""


def run_worker(mode: str, src_dir: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_dir}:{REPO}"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", WORKER_CODE, mode, str(src_dir)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if proc.returncode != 0:
        print(f"Worker {mode} failed:\n{proc.stderr}", file=sys.stderr)
        raise SystemExit(proc.returncode)
    return json.loads(proc.stdout)


def main() -> None:
    diff = subprocess.run(
        ["git", "diff", "63669c42", "HEAD", "--", "src/bts/strategy.py"],
        cwd=REPO, capture_output=True, text=True, timeout=10, check=True,
    ).stdout
    assert diff == "", f"strategy.py has unexpected diff: {diff}"

    base = run_worker("base", BASE_SRC)
    head = run_worker("head", HEAD_SRC)

    print("base bts.__file__:", base["bts_file"])
    print("head bts.__file__:", head["bts_file"])
    assert base["bts_file"] != head["bts_file"]

    assert base["summary_count"] == head["summary_count"]
    for b_item, h_item in zip(base["summaries"], head["summaries"], strict=True):
        assert b_item["case"] == h_item["case"]
        assert b_item["is_none"] == h_item["is_none"], f"None mismatch on {b_item['case']}"
        if not b_item["is_none"]:
            b_res = dict(b_item["res"])
            h_res = dict(h_item["res"])
            basis = h_res.pop("basis")
            verified = h_res.pop("candidate_verified")
            has_dec = "|none|" not in b_item["case"].split("|b=")[1]
            assert basis == ("decision" if has_dec else "no_decision"), b_item["case"]
            assert isinstance(verified, bool)
            for key in ("streak", "bar", "objective", "reason"):
                assert b_res.get(key) == h_res.get(key), f"Key {key} mismatch on {b_item['case']}: {b_res} vs {h_res}"
            if not verified:
                assert b_res == h_res, f"Unverified candidate changed base fields on {b_item['case']}: {b_res} vs {h_res}"
    print(f"1. build_skip_summary corpus: {head['summary_count']} combinations matched.")

    for b_case, h_case in zip(base["check_cases"], head["check_cases"], strict=True):
        assert b_case["case"] == h_case["case"]
        bn = dict(b_case["norm"])
        hn = dict(h_case["norm"])
        bs = bn.pop("skip_summary")
        hs = hn.pop("skip_summary")
        assert bn == hn, f"run_single_check non-summary mismatch on {b_case['case']}: {bn} vs {hn}"
        assert (bs is None) == (hs is None), f"skip_summary presence mismatch on {b_case['case']}"
        if bs is not None:
            for k in ("streak", "bar", "objective", "reason"):
                assert bs.get(k) == hs.get(k), f"skip_summary[{k}] mismatch on {b_case['case']}"
    print(f"2. run_single_check cases: {len(head['check_cases'])} cases matched all non-candidate fields.")

    assert base["missed_cases"] == head["missed_cases"]
    print("3. _maybe_alert_missed_pick gate matched:", head["missed_cases"])

    assert base["stale_meta"] == head["stale_meta"]
    print("4. Stale warning metadata/dedup/retry/interrupts matched:", len(head["stale_meta"]), "routes.")

    assert base["refusal_meta"] == head["refusal_meta"]
    print("5. Refusal side effects and EOD late_delivery matched:", len(head["refusal_meta"]), "kinds.")

    assert base["regimes_seen"] == ["reach57", "tail"] == head["regimes_seen"]
    assert base["bars"] == head["bars"]
    print(f"6. effective_pick_bar matched across {head['bar_count']} states in both regimes.")


if __name__ == "__main__":
    main()
