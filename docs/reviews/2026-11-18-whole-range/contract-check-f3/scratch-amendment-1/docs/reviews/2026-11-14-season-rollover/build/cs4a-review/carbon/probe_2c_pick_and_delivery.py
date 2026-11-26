"""Adversarial Probe 2c: Pick Policy Sweep & Real Orchestrator/Scheduler Delivery (Item 2)."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = REPO_ROOT / ".codex-review" / "cs4a-review"
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BRANCH_SRC = REPO_ROOT / "src"

ET = ZoneInfo("America/New_York")
SCHEMA = "bts_contest_streak_manual_v3"


def install_audit_hooks(flag_reads: list[str]) -> None:
    forbidden = {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.getnameinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.sendto",
    }

    def hook(event: str, args: tuple) -> None:
        if event in forbidden:
            raise RuntimeError(f"Network access forbidden in probe: {event} {args}")
        if event == "open" and args:
            p = str(args[0])
            if p.endswith("saver_state.json"):
                mode = str(args[1]) if len(args) > 1 else ""
                if "r" in mode or mode == "":
                    flag_reads.append(p)

    sys.addaudithook(hook)


def make_v3(day: str, streak: int, best: int, saver: str, **extra) -> dict:
    d = dt.date.fromisoformat(day)
    eod = dt.datetime.combine(d, dt.time.max, ET).isoformat()
    out = {
        "schema_version": SCHEMA,
        "effective_date": day,
        "season": d.year,
        "expires_at": eod,
        "active_streak": streak,
        "best_streak": best,
        "saver_state": saver,
        "recorded_at": f"{day}T15:00:00Z",
        "source": "manual_cli",
    }
    out.update(extra)
    return out


def run_policy_sweep_and_delivery() -> dict:
    flag_reads: list[str] = []
    install_audit_hooks(flag_reads)
    sys.path.insert(0, str(BRANCH_SRC))

    import pandas as pd
    import bts.contest_state as cs
    import bts.orchestrator as orch
    import bts.scheduler as sched
    import bts.strategy as strat
    from bts.saver_state import SaverState, policy_saver_available

    logging.getLogger("bts.contest_state").setLevel(logging.ERROR)

    mdp = strat._load_mdp()
    assert mdp is not None and mdp.get("tail") is not None, "Checked-in MDP artifacts failed to load!"

    probs = [round(0.70 + 0.02 * i, 2) for i in range(11)]  # 0.70, 0.72, ..., 0.90
    flag_states = {
        "absent": None,
        "not_earned": json.dumps({"state": "not_earned", "season": 2027}),
        "active": json.dumps({"state": "active", "season": 2027}),
        "used": json.dumps({"state": "used", "season": 2027}),
        "other_season_2026": json.dumps({"state": "active", "season": 2026}),
        "corrupt": "{corrupt",
        "overlong_int_5000": '{"state":"active","season":' + ("9" * 5000) + "}",
    }

    failures: list[str] = []
    cells_checked = 0
    differing_saver_examples: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        state_dir = picks_dir / "account_state"
        state_dir.mkdir(parents=True)

        (state_dir / "contest_streak.json").write_text(
            json.dumps({"active_streak": 58, "source_date": "2027-07-14"}), encoding="utf-8"
        )
        (picks_dir / "streak.json").write_text("{corrupt_model_json", encoding="utf-8")
        (picks_dir / "2027-07-16.json").write_text('{"result": "miss"}', encoding="utf-8")

        day = "2027-07-15"
        now = dt.datetime(2027, 7, 15, 13, 0, tzinfo=ET)
        flag_path = state_dir / "saver_state.json"
        manual_path = state_dir / "contest_streak.manual.json"

        ref_actions: dict[tuple[int, bool, float], str] = {}
        for s in range(58):
            for sv_bool in (False, True):
                for p in probs:
                    ctx = strat.DecisionContext(
                        primary_p=p, second_p=max(0.70, round(p - 0.01, 2)),
                        has_diff_game=True, date=day, allow_double=True,
                        mdp=mdp, best_streak=max(s, 15), best_status="trusted",
                    )
                    ref_actions[(s, sv_bool, p)] = strat.resolve_policy_decision(ctx, s, sv_bool).action

        for s in range(58):
            for p in probs:
                a_true = ref_actions[(s, True, p)]
                a_false = ref_actions[(s, False, p)]
                if a_true != a_false and len(differing_saver_examples) < 15:
                    differing_saver_examples.append({
                        "date": day, "streak": s, "p_game_hit": p,
                        "second_p": max(0.70, round(p - 0.01, 2)),
                        "action_saver_true": a_true, "action_saver_false": a_false,
                        "not_earned_action": a_true if s < 10 else a_false,
                        "active_action": a_true,
                        "used_action": a_false,
                    })

        flag_reads.clear()
        for f_label, f_content in flag_states.items():
            if flag_path.exists():
                flag_path.unlink()
            if f_content is not None:
                flag_path.write_text(f_content, encoding="utf-8")

            for obs_saver in ("not_earned", "active", "used"):
                for s in range(58):
                    best = max(s, 15)
                    manual_path.write_text(
                        json.dumps(make_v3(day, s, best, obs_saver, override_expires_at="2000-01-01T00:00:00Z")),
                        encoding="utf-8",
                    )
                    ds = cs.load_decision_streak_state(picks_dir, require_contest_state=True, now=now)
                    exp_bool = policy_saver_available(SaverState(obs_saver, 2027), s)
                    if (ds.streak != s or ds.saver_available != exp_bool or ds.saver_flag_state != obs_saver
                            or ds.status != "fresh" or ds.best_status != "trusted" or ds.best_streak != best):
                        failures.append(f"Mismatch at flag={f_label} saver={obs_saver} s={s}: {ds}")
                        break
                    for p in probs:
                        cells_checked += 1
                        ctx = strat.DecisionContext(
                            primary_p=p, second_p=max(0.70, round(p - 0.01, 2)),
                            has_diff_game=ds.allow_double, date=day, allow_double=ds.allow_double,
                            mdp=mdp, best_streak=ds.best_streak, best_status=ds.best_status,
                        )
                        dec = strat.resolve_policy_decision(ctx, ds.streak, ds.saver_available)
                        if dec.action != ref_actions[(s, exp_bool, p)]:
                            failures.append(f"Action mismatch at {f_label}/{obs_saver}/{s}/{p}: {dec.action}")

        if flag_reads:
            failures.append(f"saver_state.json was opened during dated sweep: {flag_reads[:5]}")

        # Tail regime at last microsecond of day (23:59:59.999999 ET)
        tail_day = "2027-09-22"
        tail_now = dt.datetime(2027, 9, 22, 23, 59, 59, 999999, tzinfo=ET)
        flag_path.write_text('{"state":"active","season":' + ("9" * 5000) + "}", encoding="utf-8")
        flag_reads.clear()
        manual_path.write_text(json.dumps(make_v3(tail_day, 2, 30, "active")), encoding="utf-8")
        ds_tail_stop = cs.load_decision_streak_state(picks_dir, require_contest_state=True, now=tail_now)
        ctx_tail_stop = strat.DecisionContext(
            primary_p=0.88, second_p=0.87, has_diff_game=True, date=tail_day,
            allow_double=ds_tail_stop.allow_double, mdp=mdp,
            best_streak=ds_tail_stop.best_streak, best_status=ds_tail_stop.best_status,
        )
        dec_tail_stop = strat.resolve_policy_decision(ctx_tail_stop, ds_tail_stop.streak, ds_tail_stop.saver_available)
        if ds_tail_stop.best_status != "trusted" or dec_tail_stop.objective != "tail" or dec_tail_stop.action != "skip":
            failures.append(f"Tail stop at last microsecond failed: ds={ds_tail_stop}, dec={dec_tail_stop}")
        if flag_reads:
            failures.append(f"Flag read during tail decision: {flag_reads}")

        # Part B: Real orchestrator + scheduler delivery end-to-end
        e2e_results = []
        for e2e_case, obs_s, obs_b, obs_sv, exp_sv_bool, deliv_mode in [
            ("streak6_not_earned_dm", 6, 6, "not_earned", True, "dm"),
            ("streak12_not_earned_dm", 12, 14, "not_earned", False, "dm"),
            ("streak12_active_public", 12, 14, "active", True, "public"),
            ("streak12_used_private", 12, 14, "used", False, "private"),
        ]:
            e2e_dir = Path(tmp) / f"e2e_{e2e_case}"
            (e2e_dir / "account_state").mkdir(parents=True)
            (e2e_dir / "account_state" / "saver_state.json").write_text(
                '{"state":"used","season":' + ("9" * 5000) + "}", encoding="utf-8"
            )
            (e2e_dir / "account_state" / "contest_streak.json").write_text(
                json.dumps({"active_streak": 0, "best_streak": 0, "source_date": "2027-07-14"}), encoding="utf-8"
            )
            (e2e_dir / "streak.json").write_text("{corrupt", encoding="utf-8")
            (e2e_dir / "account_state" / "contest_streak.manual.json").write_text(
                json.dumps(make_v3(
                    "2027-07-15", obs_s, obs_b, obs_sv,
                    override_expires_at="2020-01-01T00:00:00Z",
                )),
                encoding="utf-8",
            )

            df_preds = pd.DataFrame([
                {
                    "batter_name": "Bobby Witt Jr.", "batter_id": 677951, "team": "KC",
                    "lineup": 2, "pitcher_name": "Gerrit Cole", "pitcher_id": 543037,
                    "pitcher_team": "NYY", "p_game_hit": 0.85, "p_hit_pa": 0.34,
                    "flags": "", "game_pk": 7001, "game_time": "2027-07-15T23:10:00Z",
                },
                {
                    "batter_name": "Luis Arraez", "batter_id": 650333, "team": "SD",
                    "lineup": 1, "pitcher_name": "Logan Webb", "pitcher_id": 657277,
                    "pitcher_team": "SF", "p_game_hit": 0.84, "p_hit_pa": 0.33,
                    "flags": "", "game_pk": 7002, "game_time": "2027-07-16T01:40:00Z",
                },
            ])
            fake_statuses = {
                7001: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
                7002: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
            }
            cfg = {
                "orchestrator": {"picks_dir": str(e2e_dir), "models_dir": str(REPO_ROOT / "data" / "models")},
                "tiers": [{"name": "local", "type": "local"}],
                "scheduler": {
                    "pick_delivery": deliv_mode,
                    "contest_state_required": True,
                    "live_forward_capture_on_lock": False,
                },
                "bluesky": {"dm_recipient": "operator.bsky.social"},
            }

            sent_dms: list[tuple[str, str]] = []
            sent_posts: list[str] = []
            fixed_now_et = dt.datetime(2027, 7, 15, 14, 0, tzinfo=ET)
            fixed_now_utc = fixed_now_et.astimezone(dt.timezone.utc)

            class FixedDateTime(dt.datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed_now_utc.astimezone(tz) if tz else fixed_now_utc.replace(tzinfo=None)

            flag_reads.clear()
            with (
                patch.object(cs, "datetime", FixedDateTime),
                patch("bts.orchestrator.run_cascade", return_value=(df_preds, "local")),
                patch("bts.picks.get_game_statuses_detailed", return_value=fake_statuses),
                patch("bts.scheduler._delivery_game_statuses", return_value=fake_statuses),
                patch("bts.scheduler._now_et", return_value=fixed_now_et),
                patch("bts.dm.send_dm", side_effect=lambda rec, txt: (sent_dms.append((rec, txt)) or "dm-msg-123")),
                patch("bts.posting.post_to_bluesky", side_effect=lambda txt: (sent_posts.append(txt) or "at://post/1")),
            ):
                preds_out, sel, tier = orch.run_and_pick(cfg, "2027-07-15")
                assert sel is not None and sel.pick_result is not None, f"Expected pick for {e2e_case}, got {sel}"
                sched_state = sched.SchedulerState(
                    date="2027-07-15",
                    schedule_fetched_at="2027-07-15T10:00:00-04:00",
                    games=[],
                    confirmed_game_pks=[7001, 7002],
                    runs_completed=[],
                    pick_locked=False,
                    pick_locked_at=None,
                    result_status=None,
                    next_wakeup=None,
                )
                delivered_ok = sched._deliver_and_lock_pick(
                    sel.pick_result.daily, cfg, e2e_dir, sched_state, "2027-07-15", "lineup_lock", selection=sel
                )

            pick_record = json.loads((e2e_dir / "2027-07-15.json").read_text(encoding="utf-8"))
            decision_record = json.loads((e2e_dir / "2027-07-15" / "decision.json").read_text(encoding="utf-8"))

            if not delivered_ok:
                failures.append(f"{e2e_case}: _deliver_and_lock_pick returned False")
            if flag_reads:
                failures.append(f"{e2e_case}: saver_state.json opened: {flag_reads}")
            if decision_record.get("streak") != obs_s or decision_record.get("saver_available") is not exp_sv_bool:
                failures.append(f"{e2e_case}: decision.json wrong streak/saver: {decision_record}")
            if (decision_record.get("state_source") != "contest"
                    or decision_record.get("state_status") != "fresh"
                    or decision_record.get("contest_source_date") != "2027-07-15"
                    or decision_record.get("best_status") != "trusted"
                    or decision_record.get("best_streak") != obs_b):
                failures.append(f"{e2e_case}: decision.json metadata mismatch: {decision_record}")
            if deliv_mode == "dm" and (len(sent_dms) != 1 or str(obs_s) not in sent_dms[0][1]):
                failures.append(f"{e2e_case}: DM text did not carry observation streak {obs_s}: {sent_dms}")
            if deliv_mode == "public" and (len(sent_posts) != 1 or str(obs_s) not in sent_posts[0]):
                failures.append(f"{e2e_case}: Post text did not carry observation streak {obs_s}: {sent_posts}")

            e2e_results.append({
                "case": e2e_case,
                "mode": deliv_mode,
                "delivered_ok": delivered_ok,
                "flag_reads_count": len(flag_reads),
                "transport_text": sent_dms[0][1] if sent_dms else (sent_posts[0] if sent_posts else None),
                "pick_action": "double" if pick_record.get("double_down") else "single",
                "decision_json": {
                    k: decision_record.get(k)
                    for k in (
                        "action", "streak", "saver_available", "state_source", "state_status",
                        "contest_source_date", "best_streak", "best_status", "delivery_status", "scoreable",
                    )
                },
            })

    return {
        "policy_cells_checked": cells_checked,
        "differing_saver_examples": differing_saver_examples,
        "tail_stop_verified": {
            "date": tail_day, "streak": 2, "best": 30,
            "best_status": ds_tail_stop.best_status,
            "objective": dec_tail_stop.objective,
            "action": dec_tail_stop.action,
        },
        "e2e_deliveries": e2e_results,
        "failures": failures,
    }


def run_without_in_effect_vs_base() -> dict:
    script = """
import datetime as dt, json, sys, tempfile
from pathlib import Path
from zoneinfo import ZoneInfo
import bts.contest_state as cs
import bts.strategy as strat

ET = ZoneInfo("America/New_York")
now = dt.datetime(2027, 7, 15, 13, 0, tzinfo=ET)
mdp = strat._load_mdp()
mode = sys.argv[1]

out = {}
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    sd = p / "account_state"
    sd.mkdir()
    for auto_kind in ("valid", "absent"):
        if auto_kind == "valid":
            (sd / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v1",
                "active_streak": 7, "best_streak": 9, "source": "mlb_bts_profile",
                "source_date": "2027-07-14", "recorded_at": "2027-07-15T10:00:00Z"
            }))
        else:
            if (sd / "contest_streak.json").exists():
                (sd / "contest_streak.json").unlink()
        for flag_kind in ("not_earned", "used", "overlong"):
            if flag_kind == "overlong":
                (sd / "saver_state.json").write_text('{"state":"active","season":' + ("9"*5000) + '}')
            else:
                (sd / "saver_state.json").write_text(json.dumps({"state": flag_kind, "season": 2027}))

            manual_variants = ["none"] if mode == "base" else ["none", "yesterday", "tomorrow", "invalid_v3"]
            for mv in manual_variants:
                mp = sd / "contest_streak.manual.json"
                if mp.exists():
                    mp.unlink()
                if mv == "yesterday":
                    mp.write_text(json.dumps({
                        "schema_version": "bts_contest_streak_manual_v3",
                        "effective_date": "2027-07-14", "season": 2027,
                        "expires_at": "2027-07-14T23:59:59.999999-04:00",
                        "active_streak": 25, "best_streak": 25, "saver_state": "active",
                        "recorded_at": "2027-07-14T15:00:00Z", "source": "manual_cli"
                    }))
                elif mv == "tomorrow":
                    mp.write_text(json.dumps({
                        "schema_version": "bts_contest_streak_manual_v3",
                        "effective_date": "2027-07-16", "season": 2027,
                        "expires_at": "2027-07-16T23:59:59.999999-04:00",
                        "active_streak": 25, "best_streak": 25, "saver_state": "active",
                        "recorded_at": "2027-07-15T15:00:00Z", "source": "manual_cli"
                    }))
                elif mv == "invalid_v3":
                    mp.write_text(json.dumps({
                        "schema_version": "bts_contest_streak_manual_v3",
                        "effective_date": "2027-07-14", "season": 2027,
                        "expires_at": "2027-07-14T23:59:59.999999-04:00",
                        "active_streak": 25, "best_streak": 20, "saver_state": "active",
                        "recorded_at": "2027-07-14T15:00:00Z", "source": "manual_cli"
                    }))
                key = f"{auto_kind}|{flag_kind}|{mv}"
                try:
                    ds = cs.load_decision_streak_state(p, require_contest_state=False, now=now)
                    ctx = strat.DecisionContext(
                        primary_p=0.78, second_p=0.77, has_diff_game=True,
                        date="2027-07-15", allow_double=ds.allow_double, mdp=mdp,
                        best_streak=ds.best_streak, best_status=ds.best_status,
                    )
                    dec = strat.resolve_policy_decision(ctx, ds.streak, ds.saver_available)
                    out[key] = {"ok": [ds.streak, ds.saver_available, ds.source, ds.status, dec.action]}
                except Exception as exc:
                    out[key] = {"err": type(exc).__name__}
print(json.dumps(out))
"""
    res_base = json.loads(subprocess.check_output(
        [sys.executable, "-c", script, "base"],
        env=dict(os.environ, PYTHONPATH=str(BASE_SRC), PYTHONDONTWRITEBYTECODE="1"),
        text=True,
    ))
    res_branch = json.loads(subprocess.check_output(
        [sys.executable, "-c", script, "branch"],
        env=dict(os.environ, PYTHONPATH=str(BRANCH_SRC), PYTHONDONTWRITEBYTECODE="1"),
        text=True,
    ))

    mismatches = []
    for k, v in res_branch.items():
        auto_k, flag_k, mv = k.split("|")
        if mv == "invalid_v3":
            if v != {"err": "ContestStateError"}:
                mismatches.append(f"{k}: expected ContestStateError, got {v}")
        else:
            base_ref = res_base[f"{auto_k}|{flag_k}|none"]
            if v != base_ref:
                mismatches.append(f"{k}: branch {v} != base {base_ref}")

    return {"cases_checked": len(res_branch), "mismatches": mismatches, "sample_results": res_branch}


def main() -> None:
    part1 = run_policy_sweep_and_delivery()
    part2 = run_without_in_effect_vs_base()
    out = {"with_observation_and_e2e": part1, "without_observation_vs_base": part2}
    (REVIEW_DIR / "probe2_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if part1["failures"] or part2["mismatches"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
