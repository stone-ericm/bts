"""Probe for Item 2 (Words only):
Extracts base 63669c42 into .carbon-review/wr-e-r2/base_63669c42/src, confirms distinct bts.__file__,
and compares HEAD vs BASE across:
- strategy.py diff and effective_pick_bar over the entire shipped table in both regimes (streaks 0..56, days 0..185, savers, best/trust)
- build_skip_summary over unusable & valid prediction frames x decisions x candidates
- run_single_check across 13 scenarios
- maybe_notify_skip, carry_forward_skip_state, and _maybe_alert_missed_pick skip gate
- _warn_stale_contest_reading and _morning_state_check (triggers, dedup, keys, status files, midnight/Jan 1, BaseException vs Exception)
- _refuse_late_delivery, _refuse_void_game_delivery, and EOD health.late_delivery
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE_EXTRACT = OUT_DIR / "base_63669c42"
ET = ZoneInfo("America/New_York")


def _make_state(scheduler_mod, date_str: str = "2027-07-15", **kw):
    return scheduler_mod.SchedulerState(
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


def _make_daily(picks_mod, date_str: str = "2027-07-15", game_pk: int = 70001, game_time: str = "2027-07-15T23:05:00Z"):
    p1 = picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, game_pk, game_time)
    return picks_mod.DailyPick(date_str, f"{date_str}T19:10:00-04:00", p1, None, None)


def run_worker_payload() -> dict:
    import bts
    from bts import contest_state, orchestrator, picks, scheduler, strategy
    from bts.health import alert as health_alert, late_delivery as health_late_delivery
    from scripts.audit.season_walk import walk

    is_head = "base_63669c42" not in Path(bts.__file__).as_posix()
    out: dict = {"bts_file": str(Path(bts.__file__).resolve()), "is_head": is_head}

    # 1. effective_pick_bar over streaks 0..56, days 0..185, saver False/True, both regimes + tail best/trust
    artifact = strategy._load_mdp()
    end = walk.LAST_GAME_DAY + timedelta(days=1)
    bars = []
    regimes = set()
    for days in range(0, 186):
        day_iso = (end - timedelta(days=days)).isoformat()
        for streak in range(57):
            regime, _ = strategy._regime(artifact, streak, day_iso)
            regimes.add(regime)
            bests = [(streak, "trusted")]
            if regime == strategy.OBJECTIVE_TAIL:
                bests.extend([(56, "trusted"), (56, "untrusted"), (None, "missing")])
            for saver in (False, True):
                for best, trust in bests:
                    val = strategy.effective_pick_bar(streak, day_iso, saver, artifact, best, trust)
                    bars.append(val)
    out["bars_count"] = len(bars)
    out["bars_regimes"] = sorted(regimes)
    out["bars"] = bars

    # 2. build_skip_summary corpus
    class RaisingColumns:
        @property
        def columns(self):
            raise RuntimeError("bad columns")

    class RaisingGetItem:
        columns = ["batter_name", "p_game_hit"]
        def __getitem__(self, key):
            raise RuntimeError("bad getitem")

    class RaisingDecisionReason:
        objective = "emax_season_best"
        @property
        def reason(self):
            raise RuntimeError("bad reason")

    class RaisingCandidate(dict):
        def get(self, k, d=None):
            if k == "batter_name":
                raise RuntimeError("bad candidate get")
            return super().get(k, d)

    good = pd.DataFrame([
        {"batter_name": "TopRemoved", "team": "NYY", "p_game_hit": 0.999},
        {"batter_name": "RunnerUp", "team": "BOS", "p_game_hit": 0.750},
    ])
    na_row = pd.DataFrame([{"batter_name": pd.NA, "team": None, "p_game_hit": 0.81}])
    frames = [
        ("none", None),
        ("int", 42),
        ("raising_cols", RaisingColumns()),
        ("raising_getitem", RaisingGetItem()),
        ("empty_df", pd.DataFrame()),
        ("zero_rows", good.iloc[:0]),
        ("no_batter_name", good.drop(columns=["batter_name"])),
        ("no_p_game_hit", good.drop(columns=["p_game_hit"])),
        ("no_team", good.drop(columns=["team"])),
        ("all_nan", good.assign(p_game_hit=float("nan"))),
        ("all_pos_inf", good.assign(p_game_hit=float("inf"))),
        ("all_neg_inf", good.assign(p_game_hit=float("-inf"))),
        ("dup_index", pd.concat([good.iloc[:1], good.iloc[:1]])),
        ("mixed_nan_valid", pd.DataFrame([
            {"batter_name": "Bad", "team": "X", "p_game_hit": float("nan")},
            {"batter_name": "Good", "team": "Y", "p_game_hit": 0.77},
        ])),
        ("na_names", na_row),
        ("good", good),
    ]
    tail_dec = strategy.PolicyDecision(
        "skip", "skip", "mdp", "emax_season_best", 0, 8, 18, "trusted",
        None, None, "season-best 18 can't be beaten with 8 days left", None, 18,
    )
    plain_dec = strategy.PolicyDecision(
        "skip", "skip", "mdp", "reach57", 10, 80, 10, "trusted",
        None, None, None, 0.796, None,
    )
    decisions = [
        ("none", None),
        ("tail_reason", tail_dec),
        ("plain_no_reason", plain_dec),
        ("raising_reason", RaisingDecisionReason()),
    ]
    candidates = [
        ("none", None),
        ("str", "not_a_mapping"),
        ("list", [1, 2]),
        ("empty_dict", {}),
        ("bool_p", {"batter_name": "C", "team": "T", "p_game_hit": True}),
        ("str_p", {"batter_name": "C", "team": "T", "p_game_hit": "0.75"}),
        ("nan_p", {"batter_name": "C", "team": "T", "p_game_hit": float("nan")}),
        ("inf_p", {"batter_name": "C", "team": "T", "p_game_hit": float("inf")}),
        ("huge_int_p", {"batter_name": "C", "team": "T", "p_game_hit": 10**400}),
        ("raising_cand", RaisingCandidate({"batter_name": "C", "team": "T", "p_game_hit": 0.75})),
        ("valid_cand", {"batter_name": "Kept", "team": "LAD", "p_game_hit": 0.75}),
        ("valid_cand_na_names", {"batter_name": pd.NA, "team": 123, "p_game_hit": 0.75}),
    ]

    summary_records = []
    for f_name, frame in frames:
        for streak in (None, 0, 10):
            for pbar in (None, float("nan"), 0.796):
                for d_name, dec in decisions:
                    if is_head:
                        for c_name, cand in candidates:
                            res = scheduler.build_skip_summary(frame, streak, pick_bar=pbar, decision=dec, candidate=cand)
                            summary_records.append((f_name, streak, str(pbar), d_name, c_name, res))
                    else:
                        res = scheduler.build_skip_summary(frame, streak, pick_bar=pbar, decision=dec)
                        for c_name, _ in candidates:
                            summary_records.append((f_name, streak, str(pbar), d_name, c_name, res))
    out["summary_records"] = summary_records

    # 3. run_single_check scenarios (13 distinct paths)
    check_results = {}
    day = date(2027, 7, 15)
    scenarios = [
        "scoreable_short_circuit",
        "past_cutoff_reselect_skip",
        "contest_state_error",
        "predictions_none",
        "predictions_empty",
        "k1_tail_stop",
        "k2_removed_top",
        "k3_above_bar",
        "k4_all_skip",
        "k5_no_eligible",
        "k5_status_error",
        "k5_sel_none",
        "double_clamped_to_single_pick",
    ]
    for sc in scenarios:
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp)
            picks_dir = tp / "picks"
            picks_dir.mkdir()
            cfg = walk._config(picks_dir)
            cfg["orchestrator"]["heartbeat_path"] = str(tp / "hb")
            cfg["orchestrator"]["models_dir"] = str(REPO / "data/models")
            frame = walk._predictions(day, 0.85 if sc == "double_clamped_to_single_pick" else 0.76)
            if sc == "k2_removed_top":
                frame.loc[0, "p_game_hit"] = 0.999
            if sc == "k3_above_bar":
                frame = walk._predictions(date(2027, 8, 30), 0.99)
            if sc == "k4_all_skip":
                frame = walk._predictions(day, 0.96)
            if sc == "double_clamped_to_single_pick":
                frame = frame.iloc[:1].copy()

            statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
            if sc == "k2_removed_top":
                statuses[70000] = {"abstract": "F", "detailed": "Final"}
            elif sc == "k5_no_eligible":
                statuses = {int(pk): {"abstract": "F", "detailed": "Final"} for pk in frame["game_pk"]}

            check_day = date(2027, 8, 30) if sc == "k3_above_bar" else (
                date(2027, 9, 19) if sc == "k1_tail_stop" else day
            )
            check_now = datetime.combine(check_day, datetime.min.time().replace(hour=12), tzinfo=ET)

            with walk.offline(check_now), \
                    patch.object(scheduler, "count_new_confirmations", lambda *a, **kw: 2), \
                    patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: contextlib.nullcontext()), \
                    contextlib.redirect_stderr(io.StringIO()):
                if sc != "contest_state_error":
                    st_install = (
                        walk.State("dated", 0, 18, "used", 0.76) if sc == "k1_tail_stop" else
                        walk.State("dated", 56, 56, "used", 0.99) if sc == "k3_above_bar" else
                        walk.State("dated", 56, 56, "used", 0.96) if sc == "k4_all_skip" else
                        walk.State("dated", 0, 0, "not_earned", 0.85) if sc == "double_clamped_to_single_pick" else
                        walk.State("dated", 10, 10, "used", 0.75)
                    )
                    assert walk.install_state(picks_dir, check_day, st_install) == []

                if sc == "scoreable_short_circuit":
                    dp = _make_daily(picks, check_day.isoformat(), 70000, "2027-07-15T23:05:00Z")
                    picks.save_pick(dp, picks_dir)
                    (picks_dir / check_day.isoformat()).mkdir(exist_ok=True)
                    (picks_dir / check_day.isoformat() / "decision.json").write_text(json.dumps({
                        "schema_version": 1, "date": check_day.isoformat(), "action": "single",
                        "source": "mdp", "scoreable": True, "delivery_status": "delivered",
                    }))
                elif sc == "past_cutoff_reselect_skip":
                    dp = _make_daily(picks, check_day.isoformat(), 69999, "2027-07-15T15:05:00Z")
                    picks.save_pick(dp, picks_dir)

                def fake_cascade(*a):
                    if sc == "predictions_none":
                        return None, "probe"
                    if sc == "predictions_empty":
                        return frame.iloc[:0].copy(), "probe"
                    return frame.copy(), "probe"

                def fake_statuses(*a):
                    if sc == "k5_status_error":
                        raise OSError("status fetch failed")
                    return statuses

                with patch.object(orchestrator, "run_cascade", fake_cascade), \
                        patch.object(picks, "get_game_statuses_detailed", fake_statuses), \
                        patch.object(scheduler, "_alert_contest_state_failure", lambda *a, **kw: None):
                    if sc == "k5_sel_none":
                        with patch.object(orchestrator, "run_and_pick", lambda *a, **kw: (frame.copy(), None, "probe")):
                            res = scheduler.run_single_check(
                                date=check_day.isoformat(), all_game_pks=list(statuses),
                                confirmed_sides=set(), config=cfg, early_lock_gap=0.03,
                            )
                    else:
                        unavail = {69999} if sc == "past_cutoff_reselect_skip" else None
                        res = scheduler.run_single_check(
                            date=check_day.isoformat(), all_game_pks=list(statuses),
                            confirmed_sides=set(), config=cfg, early_lock_gap=0.03,
                            unavailable_game_pks=unavail,
                        )
            norm = {
                "skipped": res.get("skipped"),
                "new_lineups": res.get("new_lineups"),
                "should_post": res.get("should_post"),
                "pick_name": res.get("pick_name"),
                "pick_p": res.get("pick_p"),
                "has_pick_result": res.get("pick_result") is not None,
                "locked": res["pick_result"].locked if res.get("pick_result") is not None else None,
                "has_selection": res.get("selection") is not None,
                "sel_action": res["selection"].action if res.get("selection") is not None else None,
                "has_skip_summary": "skip_summary" in res and res["skip_summary"] is not None,
                "skip_summary": res.get("skip_summary"),
            }
            check_results[sc] = norm
    out["check_results"] = check_results

    # 4. Missed-pick gate & maybe_notify_skip
    gate_outcomes = []
    for summ in (None, {}, {"best_batter": "A", "best_team": "T", "best_p": 0.75, "streak": 5},
                 {"best_batter": "A", "best_team": "T", "best_p": 0.75, "streak": 5, "basis": "no_decision"}):
        st = _make_state(scheduler, "2027-07-15", skip_summary=summ)
        called = []
        with patch.object(scheduler, "_alert_missed_pick", lambda *a, **kw: called.append(1)), \
                patch.object(scheduler, "fetch_schedule", lambda d: [{"game_pk": 1, "game_time": "2027-07-15T16:10:00Z", "status": "Scheduled"}]), \
                patch.object(scheduler, "_now_et", lambda: datetime(2027, 7, 15, 12, 0, tzinfo=ET)), \
                patch.object(scheduler, "_watchdog_ping_sleep", lambda *a: None), \
                patch("bts.picks.load_pick", lambda *a: None), \
                patch("bts.daily_decision.is_scoreable_commit", lambda *a: False):
            scheduler._maybe_alert_missed_pick(
                {"orchestrator": {"picks_dir": "/tmp/p"}}, "2027-07-15", Path("/tmp/p"), 15, None, st
            )
        gate_outcomes.append(len(called))
    out["missed_pick_gate"] = gate_outcomes

    # 5. Stale warnings: cross-route dedup, interrupt containment (BaseException vs Exception), Jan 1 date
    stale_effects = {}
    for route_order in (("morning", "delivery"), ("delivery", "morning")):
        with tempfile.TemporaryDirectory() as tmp:
            tp = Path(tmp) / "d" / "picks"
            tp.mkdir(parents=True)
            now_jan1 = datetime(2028, 1, 1, 0, 15, tzinfo=ET)
            st_auto = contest_state.DecisionStreakState(
                5, True, True, "contest", "fresh", None, None,
                observation_kind="automatic", contest_recorded_at="2027-12-31T01:00:00Z",
            )
            rec = contest_state.automatic_reading_record(st_auto, now_jan1)
            sent_routes = []
            with patch.object(scheduler, "_now_et", lambda: now_jan1), \
                    patch.object(contest_state, "load_decision_streak_state", lambda *a, **kw: st_auto), \
                    patch.object(health_alert, "send_dm", lambda r, b: sent_routes.append(
                        "morning" if "Morning check" in b else "delivery"
                    ) or "id1"), \
                    contextlib.redirect_stderr(io.StringIO()):
                for r_name in route_order:
                    if r_name == "morning":
                        scheduler._morning_state_check(
                            {"bluesky": {"dm_recipient": "u.test"}}, tp, "2028-01-01",
                            now_jan1, now_jan1 + timedelta(hours=2),
                        )
                    else:
                        scheduler._warn_stale_contest_reading(rec, {"bluesky": {"dm_recipient": "u.test"}}, tp)
            status_file = tp.parent / "health_state" / "contest_stale_reading_dm_delivery_status.json"
            status_json = json.loads(status_file.read_text())
            stale_effects["->".join(route_order)] = {
                "sent_routes": sent_routes,
                "status": status_json["status"],
                "sent_day": status_json["sent_day"],
                "sent_sources": status_json["sent_sources"],
                "critical_count": status_json["critical_count"],
                "warn_attention_count": status_json["warn_attention_count"],
            }

    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "d" / "picks"
        tp.mkdir(parents=True)
        now_j = datetime(2027, 7, 15, 10, 0, tzinfo=ET)
        st_auto = contest_state.DecisionStreakState(
            5, True, True, "contest", "fresh", None, None,
            observation_kind="automatic", contest_recorded_at="2027-07-14T01:00:00Z",
        )
        rec = contest_state.automatic_reading_record(st_auto, now_j)
        with patch.object(scheduler, "_now_et", lambda: now_j), \
                patch.object(health_alert, "send_dm", side_effect=KeyboardInterrupt("boom")), \
                contextlib.redirect_stderr(io.StringIO()):
            scheduler._warn_stale_contest_reading(rec, {"bluesky": {"dm_recipient": "u.test"}}, tp)
            deliv_contained = True
        morn_propagated = False
        tp2 = Path(tmp) / "d2" / "picks"
        tp2.mkdir(parents=True)
        with patch.object(contest_state, "load_decision_streak_state", lambda *a, **kw: st_auto), \
                patch.object(health_alert, "send_dm", side_effect=KeyboardInterrupt("boom")), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                scheduler._morning_state_check(
                    {"bluesky": {"dm_recipient": "u.test"}}, tp2, "2027-07-15", now_j, now_j + timedelta(hours=2)
                )
            except KeyboardInterrupt:
                morn_propagated = True
        stale_effects["interrupts"] = {
            "delivery_contained_KeyboardInterrupt": deliv_contained,
            "morning_propagated_KeyboardInterrupt": morn_propagated,
        }
    out["stale_effects"] = stale_effects

    # 6. Refusal side-effects + EOD health.late_delivery.check
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp) / "d1" / "picks"
        tp.mkdir(parents=True)
        dp = _make_daily(picks, "2027-07-15", 70001, "2027-07-15T23:05:00Z")
        picks.save_pick(dp, tp)
        st_r = _make_state(scheduler, "2027-07-15")
        cutoff = datetime(2027, 7, 15, 19, 0, tzinfo=ET)
        now_r = datetime(2027, 7, 15, 19, 10, tzinfo=ET)
        captured_alerts = []
        captured_status_paths = []
        with patch.object(
            health_alert, "dispatch_dm_for_health_alerts",
            lambda alerts, rec_h, status_path=None, **kw: (
                captured_alerts.extend([(a.level, a.source) for a in alerts]),
                captured_status_paths.append(status_path.name if status_path else None),
            ),
        ), contextlib.redirect_stderr(io.StringIO()):
            scheduler._refuse_late_delivery(dp, {"bluesky": {"dm_recipient": "u"}}, tp, st_r, "2027-07-15", "p", cutoff, now_r)
        eod = health_late_delivery.check("2027-07-15", picks_dir=tp)
        arch_name = st_r.delivery_refusals[0]["archive"]
        arch_data = json.loads((tp / "2027-07-15" / arch_name).read_text())

        tp_v = Path(tmp) / "d2" / "picks"
        tp_v.mkdir(parents=True)
        picks.save_pick(dp, tp_v)
        st_v = _make_state(scheduler, "2027-07-15")
        with patch.object(scheduler, "_now_et", lambda: now_r), patch.object(
            health_alert, "dispatch_dm_for_health_alerts",
            lambda alerts, rec_h, status_path=None, **kw: (
                captured_alerts.extend([(a.level, a.source) for a in alerts]),
                captured_status_paths.append(status_path.name if status_path else None),
            ),
        ), contextlib.redirect_stderr(io.StringIO()):
            scheduler._refuse_void_game_delivery(dp, {"bluesky": {"dm_recipient": "u"}}, tp_v, st_v, "2027-07-15", "p", {70001: "Postponed"})
        arch_v = json.loads((tp_v / "2027-07-15" / st_v.delivery_refusals[0]["archive"]).read_text())

        out["refusal_effects"] = {
            "alerts": captured_alerts,
            "status_files": captured_status_paths,
            "late_refusal_record": [{k: v for k, v in r.items() if k != "archive"} for r in st_r.delivery_refusals],
            "void_refusal_record": [{k: v for k, v in r.items() if k != "archive"} for r in st_v.delivery_refusals],
            "late_archive_meta": {k: v for k, v in arch_data.get("refused_delivery", {}).items() if k != "refused_at"},
            "void_archive_meta": {k: v for k, v in arch_v.get("refused_delivery", {}).items() if k != "refused_at"},
            "late_pick_removed": picks.load_pick("2027-07-15", tp) is None,
            "void_pick_removed": picks.load_pick("2027-07-15", tp_v) is None,
            "eod_late_delivery_alerts": [(a.level, a.source, a.message) for a in eod],
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["run"], default=None)
    args = ap.parse_args()
    if args.worker == "run":
        print(json.dumps(run_worker_payload(), default=str))
        return

    if BASE_EXTRACT.exists():
        shutil.rmtree(BASE_EXTRACT)
    BASE_EXTRACT.mkdir(parents=True)
    arch = subprocess.run(["git", "archive", "63669c42", "src"], cwd=REPO, check=True, capture_output=True, timeout=15)
    subprocess.run(["tar", "-x", "-C", str(BASE_EXTRACT)], input=arch.stdout, check=True, timeout=15)

    strat_diff = subprocess.run(
        ["git", "diff", "63669c42", "HEAD", "--", "src/bts/strategy.py"],
        cwd=REPO, check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    assert strat_diff == "", f"Unexpected strategy.py diff: {strat_diff}"

    py = str(REPO / ".venv/bin/python")
    head_env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}"}
    base_env = {**os.environ, "PYTHONPATH": f"{BASE_EXTRACT / 'src'}:{REPO}"}

    r_head = subprocess.run([py, __file__, "--worker", "run"], cwd=REPO, env=head_env, capture_output=True, text=True, timeout=60)
    if r_head.returncode != 0:
        raise RuntimeError(f"HEAD worker failed:\n{r_head.stderr}")
    r_base = subprocess.run([py, __file__, "--worker", "run"], cwd=REPO, env=base_env, capture_output=True, text=True, timeout=60)
    if r_base.returncode != 0:
        raise RuntimeError(f"BASE worker failed:\n{r_base.stderr}")

    head = json.loads(r_head.stdout)
    base = json.loads(r_base.stdout)

    assert head["is_head"] is True and head["bts_file"] == str((REPO / "src/bts/__init__.py").resolve())
    assert base["is_head"] is False and base["bts_file"] == str((BASE_EXTRACT / "src/bts/__init__.py").resolve())

    assert head["bars_count"] == base["bars_count"] and head["bars"] == base["bars"]

    assert len(head["summary_records"]) == len(base["summary_records"])
    none_matches = 0
    non_none_matches = 0
    for h_rec, b_rec in zip(head["summary_records"], base["summary_records"], strict=True):
        assert h_rec[:5] == b_rec[:5]
        h_sum, b_sum = h_rec[5], b_rec[5]
        if b_sum is None:
            assert h_sum is None, f"Mismatch None gate on {h_rec[:5]}: {h_sum}"
            none_matches += 1
        else:
            assert h_sum is not None, f"HEAD returned None where BASE succeeded on {h_rec[:5]}"
            for k in ("streak", "bar", "objective", "reason"):
                assert h_sum.get(k) == b_sum.get(k), f"Key {k} differed on {h_rec[:5]}: {h_sum} vs {b_sum}"
            assert h_sum["basis"] in ("decision", "no_decision")
            assert isinstance(h_sum["candidate_verified"], bool)
            if not h_sum["candidate_verified"]:
                for k in ("best_batter", "best_team", "best_p"):
                    assert h_sum[k] == b_sum[k], f"Unverified candidate changed {k} on {h_rec[:5]}"
            non_none_matches += 1

    for sc in head["check_results"]:
        h_c = dict(head["check_results"][sc])
        b_c = dict(base["check_results"][sc])
        h_s = h_c.pop("skip_summary")
        b_s = b_c.pop("skip_summary")
        assert h_c == b_c, f"run_single_check differed on {sc}: {h_c} vs {b_c}"
        if b_s is None:
            assert h_s is None
        else:
            for k in ("streak", "bar", "objective", "reason"):
                assert h_s.get(k) == b_s.get(k)

    assert head["missed_pick_gate"] == base["missed_pick_gate"]
    assert head["stale_effects"] == base["stale_effects"]
    assert head["refusal_effects"] == base["refusal_effects"]

    summary = {
        "head_bts_file": head["bts_file"],
        "base_bts_file": base["bts_file"],
        "strategy_py_diff_empty": True,
        "bar_comparisons": head["bars_count"],
        "regimes": head["bars_regimes"],
        "summary_combinations": len(head["summary_records"]),
        "summary_none_matches": none_matches,
        "summary_non_none_matches": non_none_matches,
        "run_single_check_scenarios": len(head["check_results"]),
        "missed_pick_gate_identical": True,
        "stale_effects_identical": True,
        "refusal_effects_identical": True,
    }
    out_path = OUT_DIR / "probe_2_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
