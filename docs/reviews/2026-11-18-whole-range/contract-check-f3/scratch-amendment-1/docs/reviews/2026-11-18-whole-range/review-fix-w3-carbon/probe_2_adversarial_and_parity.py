"""Probe 2: Adversarial counterexamples, regression breakages, standing rows, and base-vs-head parity."""

from __future__ import annotations

import contextvars
from dataclasses import replace
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_ceab9854" / "src"
HEAD_SRC = REPO / "src"
DAY = "2027-03-24"


def load_offline(tmp_root: Path):
    source = REPO / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("probe2_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_sub(tree: str, case: str) -> dict:
    src_dir = str(BASE_SRC if tree == "base" else HEAD_SRC)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src_dir}:{REPO}"
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), __file__, "--worker", tree, case],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-1500:],
            "stderr": proc.stderr[-1500:],
        }
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE2_JSON ")]
    if not lines:
        return {"ok": False, "error": "no PROBE2_JSON", "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}
    return json.loads(lines[-1][len("PROBE2_JSON "):])


class AlarmStop(BaseException):
    pass


def worker_main(tree: str, case: str) -> dict:
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, dm, picks, posting, scheduler, strategy

    with TemporaryDirectory(prefix=f"p2-{tree}-{case}-", dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        offline = load_offline(tmp_path)

        # 1. Existing fast-suite tests in test_wr_a_pick.py & test_wr_a_review_carbon.py
        if case in {"fastsuite_wr_a_pick_dm", "fastsuite_wr_a_pick_public", "fastsuite_wr_a_pick_fixed_demo"}:
            mode = "public" if case.endswith("public") else "dm"
            with offline.offline(offline.instant(DAY, 18, 25)) as (world, patch):
                if case == "fastsuite_wr_a_pick_fixed_demo" and hasattr(picks, "save_pick_with_bytes"):
                    # Simulate the smallest fix in picks.py: save_pick owns the write + ContextVar box;
                    # save_pick_with_bytes delegates through picks.save_pick.
                    sink_var: contextvars.ContextVar[list[bytes] | None] = contextvars.ContextVar("sink", default=None)

                    def fixed_save_pick(daily_obj, picks_dir):
                        picks_dir = Path(picks_dir)
                        picks_dir.mkdir(parents=True, exist_ok=True)
                        p = picks_dir / f"{daily_obj.date}.json"
                        txt = json.dumps(picks.asdict(daily_obj), indent=2)
                        picks.atomic_write_text(p, txt)
                        box = sink_var.get()
                        if box is not None and not box:
                            box.append(txt.encode("ascii"))
                        try:
                            picks.append_lineup_evolution(daily_obj, picks_dir)
                        except Exception:
                            pass
                        return p

                    def fixed_save_pick_with_bytes(daily_obj, picks_dir):
                        box: list[bytes] = []
                        tok = sink_var.set(box)
                        try:
                            p = picks.save_pick(daily_obj, picks_dir)
                        finally:
                            sink_var.reset(tok)
                        return p, (box[0] if box else None)

                    patch.setattr(picks, "save_pick", fixed_save_pick)
                    patch.setattr(picks, "save_pick_with_bytes", fixed_save_pick_with_bytes)

                world.predictions = lambda: world.frame((0.99, 0.98))
                world.dated(0)
                save = picks.save_pick
                corrected = False

                def save_and_correct(daily_obj, directory):
                    nonlocal corrected
                    saved = save(daily_obj, directory)
                    if not corrected:
                        corrected = True
                        assert daily_obj.chosen_on["streak"] == 0 and daily_obj.double_down
                        world.dated(30)
                    return saved

                patch.setattr(picks, "save_pick", save_and_correct)
                world.config["scheduler"]["pick_delivery"] = mode

                deliver = scheduler._deliver_and_lock_pick
                refusals = []
                first_call_delivered = None

                def delivery(daily_obj, config, directory, state, day, label, *, selection=None, **kw):
                    nonlocal first_call_delivered
                    delivered = deliver(daily_obj, config, directory, state, day, label, selection=selection, **kw)
                    if first_call_delivered is None:
                        first_call_delivered = delivered
                        if not delivered and not state.pick_locked:
                            refusals.append(label)
                    return delivered

                patch.setattr(scheduler, "_deliver_and_lock_pick", delivery)
                scheduler.run_day(DAY, world.config)
                final_pick = picks.load_pick(DAY, world.picks)
                return {
                    "ok": True,
                    "corrected_during_check": corrected,
                    "first_call_delivered": first_call_delivered,
                    "refusals": refusals,
                    "final_chosen_streak": final_pick.chosen_on["streak"] if final_pick else None,
                    "final_is_single": (final_pick.double_down is None) if final_pick else None,
                    "sent_count": len(world.sends("dm") + world.sends("post")),
                    "contest_state_alerts": sum("contest_state" in b for b in world.sends("health-dm")),
                    "passes_test_wr_a_pick": (
                        first_call_delivered is False
                        and refusals == ["lineup"]
                        and final_pick is not None
                        and final_pick.chosen_on["streak"] == 30
                        and final_pick.double_down is None
                        and len(world.sends("dm") + world.sends("post")) == 1
                        and sum("contest_state" in b for b in world.sends("health-dm")) == 1
                    ),
                }

        if case in {"fastsuite_wr_a_carbon_midnight", "fastsuite_wr_a_carbon_midnight_fixed_demo"}:
            with offline.offline(offline.instant(DAY, 23, 58)) as (world, patch):
                if case.endswith("fixed_demo") and hasattr(picks, "save_pick_with_bytes"):
                    sink_var2: contextvars.ContextVar[list[bytes] | None] = contextvars.ContextVar("sink2", default=None)

                    def fixed_save2(daily_obj, picks_dir):
                        picks_dir = Path(picks_dir)
                        picks_dir.mkdir(parents=True, exist_ok=True)
                        p = picks_dir / f"{daily_obj.date}.json"
                        txt = json.dumps(picks.asdict(daily_obj), indent=2)
                        picks.atomic_write_text(p, txt)
                        box = sink_var2.get()
                        if box is not None and not box:
                            box.append(txt.encode("ascii"))
                        try:
                            picks.append_lineup_evolution(daily_obj, picks_dir)
                        except Exception:
                            pass
                        return p

                    def fixed_save_bytes2(daily_obj, picks_dir):
                        box: list[bytes] = []
                        tok = sink_var2.set(box)
                        try:
                            p = picks.save_pick(daily_obj, picks_dir)
                        finally:
                            sink_var2.reset(tok)
                        return p, (box[0] if box else None)

                    patch.setattr(picks, "save_pick", fixed_save2)
                    patch.setattr(picks, "save_pick_with_bytes", fixed_save_bytes2)

                world.schedule[DAY] = [
                    offline.game(101, offline.instant("2027-03-25", 0, 15)),
                    offline.game(102, offline.instant("2027-03-25", 0, 20)),
                ]
                world.predictions = lambda: world.frame((0.99, 0.98))
                world.automatic(0)
                acc = world.account / "contest_streak.json"
                acc.write_text(json.dumps({**json.loads(acc.read_text()), "source": "mlb_bts_profile"}))
                world.dated(0)

                save_orig = picks.save_pick
                rolled = False

                def save_and_roll_midnight(daily_obj, directory):
                    nonlocal rolled
                    saved = save_orig(daily_obj, directory)
                    if not rolled:
                        rolled = True
                        world.now = offline.instant("2027-03-25", 0, 1)
                    return saved

                patch.setattr(picks, "save_pick", save_and_roll_midnight)
                res = scheduler.run_single_check(DAY, [101, 102], set(), world.config, 0.03)
                st = scheduler.SchedulerState(DAY, world.now.isoformat(), [], [], [], False, None, None, None)
                delivered = scheduler._deliver_and_lock_pick(
                    res["pick_result"].daily, world.config, world.picks, st, DAY, "lineup",
                    selection=res["selection"],
                )
                return {
                    "ok": True,
                    "rolled_during_check": rolled,
                    "delivered": delivered,
                    "pick_locked": st.pick_locked,
                    "dm_count": len(world.sends("dm")),
                    "contest_alerts": sum("contest_state" in b for b in world.sends("health-dm")),
                    "passes_test_wr_a_review_carbon": (
                        delivered is False
                        and not st.pick_locked
                        and len(world.sends("dm")) == 0
                        and sum("contest_state" in b for b in world.sends("health-dm")) == 1
                    ),
                }

        # 2. Discrepancy 2A & 2B: Postponed game when check overruns cutoff (deliver) and when defer crosses cutoff
        if case == "bug_void_deliver_overrunning_cutoff":
            with offline.offline(offline.instant(DAY, 18, 25)) as (world, mp):
                world.schedule[DAY] = world.schedule[DAY][:1]
                world.dated(0)
                real_check = scheduler.run_single_check
                starts = []

                def forecast():
                    starts.append(world.now.strftime("%H:%M:%S"))
                    if len(starts) == 1:
                        # 18:25 check overruns to 19:05:30 (30s past 19:05 cutoff)
                        world.advance(40 * 60 + 30)
                        return world.frame((0.99,), (True,))
                    return None

                def postpone_during_check(*a, **kw):
                    res = real_check(*a, **kw)
                    if len(starts) == 1:
                        world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    return res

                world.predictions = forecast
                mp.setattr(scheduler, "run_single_check", postpone_during_check)
                scheduler.run_day(DAY, world.config)
                st = scheduler.load_state(DAY, world.picks)
                alerts = world.sends("health-dm")
                return {
                    "ok": True,
                    "starts": starts,
                    "refusal_reasons": [r["reason"] for r in (st.delivery_refusals or [])],
                    "alerts": alerts,
                    "has_postponed_alert": any("[postponed_pick]" in a for a in alerts),
                    "has_late_alert": any("[late_delivery]" in a for a in alerts),
                    "names_batter_in_dm": any("Synthetic Batter 101" in a for a in alerts),
                }

        if case == "bug_void_defer_timely_status_crossing_cutoff":
            at = lambda h, m, s=0: offline.instant(DAY, h, m, s)
            with offline.offline(at(18, 25)) as (world, mp):
                world.dated(0)
                world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(18, 50), "game_pks": [102]},
                ])
                mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
                    pk: {"away", "home"} for pk in pks if pk == 101
                })
                real_check = scheduler.run_single_check
                real_statuses = scheduler._delivery_game_statuses
                real_save_state = scheduler.save_state
                starts = []
                got_timely_void = False

                def forecast():
                    starts.append(world.now.strftime("%H:%M:%S"))
                    if len(starts) == 1:
                        # Finishes at 19:04:56 -> 4s before 19:05:00 cutoff (> 3s, so status poll runs!)
                        world.advance(39 * 60 + 56)
                        return world.frame((0.99, 0.40), (True, True))
                    return None

                def postpone_after_first(*a, **kw):
                    res = real_check(*a, **kw)
                    if len(starts) == 1:
                        world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    return res

                def timely_status(d):
                    nonlocal got_timely_void
                    out = real_statuses(d)
                    if len(starts) == 1:
                        got_timely_void = True
                    return out

                def slow_save_state(st_obj, pdir):
                    res = real_save_state(st_obj, pdir)
                    if got_timely_void and world.now < at(19, 5):
                        world.advance(5)  # save_state crosses 19:05:00 after defer_void_games was already obtained!
                    return res

                world.predictions = forecast
                mp.setattr(scheduler, "run_single_check", postpone_after_first)
                mp.setattr(scheduler, "_delivery_game_statuses", timely_status)
                mp.setattr(scheduler, "save_state", slow_save_state)
                scheduler.run_day(DAY, world.config)
                st = scheduler.load_state(DAY, world.picks)
                alerts = world.sends("health-dm")
                return {
                    "ok": True,
                    "got_timely_void": got_timely_void,
                    "refusal_reasons": [r["reason"] for r in (st.delivery_refusals or [])],
                    "has_postponed_alert": any("[postponed_pick]" in a for a in alerts),
                    "has_late_alert": any("[late_delivery]" in a for a in alerts),
                    "names_batter_in_dm": any("Synthetic Batter 101" in a for a in alerts),
                }

        # 3. Item 2: Days ending with no pick and no alert on HEAD where base alerted (or delivered)
        if case == "item2_two_game_projected_defer_silent_day":
            # Ordinary 2-game slate (19:10 and 20:10), 18:25 overruns to 19:00 with projected primary,
            # subsequent cascades take 6 min and fail (None).
            with offline.offline(offline.instant(DAY, 18, 25)) as (world, _):
                world.dated(0)
                starts = []

                def forecast():
                    starts.append(world.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        world.advance(35 * 60)
                        return world.frame((0.99, 0.40), (True, True))
                    world.advance(6 * 60)
                    return None

                world.predictions = forecast
                scheduler.run_day(DAY, world.config)
                dec = daily_decision.load_decision(DAY, world.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "pick_dms": len(world.sends("dm")),
                    "health_alerts": [a.splitlines()[-1] for a in world.sends("health-dm")],
                    "decision": dec,
                }

        if case == "item2_two_game_abandoned_reuse_loses_both_contest_and_missed_alerts":
            # Exact 2-game version of test_missed_pick_alert_survives_abandoned_reuse:
            # Game 101 at 19:10, Game 102 at 20:10 (check at 19:25).
            # Confirmed Batter 101 (0.99) vs Projected Batter 102 (0.98) -> gap_no_feasible_window -> plan = deliver!
            # Before the extra regular_only read, saved bytes get +b"\n" and dated state is corrected to best=1;
            # later cascades return None.
            with offline.offline(offline.instant(DAY, 18, 25)) as (world, mp):
                world.dated(0)
                path = world.picks / f"{DAY}.json"
                starts = []
                real_load_bytes = picks.load_pick_bytes
                real_check = scheduler.run_single_check
                mutated = False

                def forecast():
                    starts.append(world.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w_adv = 35 * 60
                        world.advance(w_adv)
                        return world.frame((0.99, 0.98), (False, True))
                    return None

                def mutate_on_base(*a, **kw):
                    nonlocal mutated
                    res = real_check(*a, **kw)
                    if tree == "base" and not mutated:
                        mutated = True
                        path.write_bytes(path.read_bytes() + b"\n")
                        world.dated(0, best=1)
                    return res

                def mutate_on_extra_read(*a, **kw):
                    nonlocal mutated
                    if kw.get("regular_only") and not mutated:
                        mutated = True
                        path.write_bytes(path.read_bytes() + b"\n")
                        world.dated(0, best=1)
                    return real_load_bytes(*a, **kw)

                world.predictions = forecast
                mp.setattr(scheduler, "run_single_check", mutate_on_base)
                mp.setattr(picks, "load_pick_bytes", mutate_on_extra_read)
                scheduler.run_day(DAY, world.config)
                dec = daily_decision.load_decision(DAY, world.picks)
                return {
                    "ok": True,
                    "mutated": mutated,
                    "starts": starts,
                    "pick_dms": len(world.sends("dm")),
                    "health_alerts": [a.splitlines()[-1] for a in world.sends("health-dm")],
                    "stale_archives": [p.name for p in (world.picks / DAY).glob("stale_pick*.json")],
                    "decision": dec,
                }

        # 4. Standing rows: interrupts & restarts across all points
        if case.startswith("standing_interrupt_"):
            # standing_interrupt_<exc>_<point>
            rest = case[len("standing_interrupt_"):]
            exc_name, point = rest.split("_", 1)
            exc_cls = KeyboardInterrupt if exc_name == "ki" else SystemExit
            at = lambda h, m: offline.instant(DAY, h, m)
            with offline.offline(at(18, 25)) as (world, mp):
                two_games = point in {"during_confirm_poll", "during_status_poll"}
                if not two_games:
                    world.schedule[DAY] = world.schedule[DAY][:1]
                else:
                    world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                    mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                        {"time_et": at(18, 25), "game_pks": [101]},
                        {"time_et": at(18, 50), "game_pks": [102]},
                    ])
                world.automatic(4, age_hours=16)
                mp.setattr(scheduler, "plan_run_times", mp.getattr if False else (
                    (lambda *a, **kw: [{"time_et": at(18, 25), "game_pks": [101]}, {"time_et": at(18, 50), "game_pks": [102]}])
                    if two_games else (lambda *a, **kw: [{"time_et": at(18, 25), "game_pks": [101]}])
                ))
                real_load_bytes = picks.load_pick_bytes
                real_plan = scheduler.plan_fallback_action
                real_format = scheduler._format_pick_delivery_text
                real_send = dm.send_dm
                real_commit = scheduler._commit_decision_for_pick
                real_auto_rec = contest_state.automatic_reading_record
                real_statuses = scheduler._delivery_game_statuses
                starts = []
                checked = False
                real_check = scheduler.run_single_check

                def forecast():
                    starts.append(world.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        world.advance((22 if two_games else 35) * 60)
                        return world.frame((0.99, 0.70)[:len(world.schedule[DAY])], (True, True)[:len(world.schedule[DAY])])
                    return None

                def mark_checked(*a, **kw):
                    nonlocal checked
                    r = real_check(*a, **kw)
                    checked = True
                    return r

                world.predictions = forecast
                mp.setattr(scheduler, "run_single_check", mark_checked)

                if point == "before_reload":
                    mp.setattr(picks, "load_pick_bytes", lambda *a, **kw: (_ for _ in ()).throw(exc_cls("stop")) if checked else real_load_bytes(*a, **kw))
                elif point == "before_planner":
                    mp.setattr(scheduler, "plan_fallback_action", lambda **kw: (_ for _ in ()).throw(exc_cls("stop")))
                elif point == "during_confirm_poll":
                    mp.setattr(scheduler, "check_confirmed_lineups",
                               lambda pks: (_ for _ in ()).throw(exc_cls("stop")) if (checked and pks == [102]) else {101: {"away", "home"}})
                elif point == "during_status_poll":
                    mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {pk: {"away", "home"} for pk in pks if pk == 101})
                    mp.setattr(scheduler, "_delivery_game_statuses",
                               lambda d: (_ for _ in ()).throw(exc_cls("stop")) if checked else real_statuses(d))
                elif point == "before_marker":
                    mp.setattr(scheduler, "_format_pick_delivery_text", lambda *a, **kw: (_ for _ in ()).throw(exc_cls("stop")))
                elif point == "inside_4c_telemetry":
                    mp.setattr(contest_state, "automatic_reading_record",
                               lambda *a, **kw: (_ for _ in ()).throw(exc_cls("stop")) if checked else real_auto_rec(*a, **kw))
                elif point == "after_marker":
                    mp.setattr(dm, "send_dm", lambda *a, **kw: (_ for _ in ()).throw(exc_cls("stop")))
                elif point == "after_confirmation":
                    mp.setattr(scheduler, "_commit_decision_for_pick", lambda *a, **kw: (_ for _ in ()).throw(exc_cls("stop")))

                escaped = False
                try:
                    scheduler.run_day(DAY, world.config)
                except (KeyboardInterrupt, SystemExit):
                    escaped = True

                p_mid = picks.load_pick(DAY, world.picks)
                dec_mid = daily_decision.load_decision(DAY, world.picks)
                mid_summary = {
                    "escaped": escaped,
                    "attempted": p_mid.delivery_attempted if p_mid else None,
                    "delivered": picks.pick_was_delivered(p_mid) if p_mid else None,
                    "dm_count": len(world.sends("dm")),
                    "decision_exists": dec_mid is not None,
                }

                # Restore hooks and simulate a fresh process restart on the durable state!
                mp.setattr(picks, "load_pick_bytes", real_load_bytes)
                mp.setattr(scheduler, "plan_fallback_action", real_plan)
                mp.setattr(scheduler, "_format_pick_delivery_text", real_format)
                mp.setattr(contest_state, "automatic_reading_record", real_auto_rec)
                mp.setattr(dm, "send_dm", real_send)
                mp.setattr(scheduler, "_commit_decision_for_pick", real_commit)
                mp.setattr(scheduler, "_delivery_game_statuses", real_statuses)
                mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {pk: {"away", "home"} for pk in pks})

                scheduler.run_day(DAY, world.config)
                dec_after = daily_decision.load_decision(DAY, world.picks)
                st_after = scheduler.load_state(DAY, world.picks)
                return {
                    "ok": True,
                    "mid": mid_summary,
                    "restart_delivery_status": dec_after["delivery_status"] if dec_after else None,
                    "restart_dm_count": len(world.sends("dm")),
                    "restart_unknown_alert": any("OUTCOME UNKNOWN" in a for a in world.sends("health-dm")),
                    "restart_reuse_count": sum(1 for r in (st_after.fallback_refreshes or []) if r.get("reused_check_cascade")),
                }

        raise ValueError(f"Unknown case {case}")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        out = worker_main(sys.argv[2], sys.argv[3])
        print("PROBE2_JSON " + json.dumps(out), flush=True)
        return

    paired_cases = [
        "fastsuite_wr_a_pick_dm",
        "fastsuite_wr_a_pick_public",
        "fastsuite_wr_a_carbon_midnight",
        "bug_void_deliver_overrunning_cutoff",
        "item2_two_game_projected_defer_silent_day",
        "item2_two_game_abandoned_reuse_loses_both_contest_and_missed_alerts",
    ]
    head_cases = [
        "fastsuite_wr_a_pick_fixed_demo",
        "fastsuite_wr_a_carbon_midnight_fixed_demo",
        "bug_void_defer_timely_status_crossing_cutoff",
    ]
    for exc in ("ki", "se"):
        for pt in (
            "before_reload",
            "before_planner",
            "during_confirm_poll",
            "during_status_poll",
            "before_marker",
            "inside_4c_telemetry",
            "after_marker",
            "after_confirmation",
        ):
            head_cases.append(f"standing_interrupt_{exc}_{pt}")

    results = {"base": {}, "head": {}}
    for c in paired_cases:
        results["base"][c] = run_sub("base", c)
        results["head"][c] = run_sub("head", c)
        print(f"[paired] {c}: base.ok={results['base'][c].get('ok')} head.ok={results['head'][c].get('ok')}", flush=True)
    for c in head_cases:
        results["head"][c] = run_sub("head", c)
        print(f"[head] {c}: head.ok={results['head'][c].get('ok')}", flush=True)

    out_path = REVIEW_DIR / "probe_2_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
