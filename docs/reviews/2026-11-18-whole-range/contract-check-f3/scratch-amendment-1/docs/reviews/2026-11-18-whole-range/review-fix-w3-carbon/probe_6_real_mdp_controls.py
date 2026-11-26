"""Probe 6: Real shipped-MDP-policy controls (streak=30 single/skip) and fix verification for void-vs-cutoff ordering."""

from __future__ import annotations

from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import shutil
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
    spec = spec_from_file_location("probe6_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_case(src_path: Path, case: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), __file__, "--worker", case],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )
    if proc.returncode != 0:
        return {"ok": False, "rc": proc.returncode, "stderr": proc.stderr[-1500:], "stdout": proc.stdout[-1500:]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE6_JSON ")]
    return json.loads(lines[-1][len("PROBE6_JSON "):])


def worker_main(case: str) -> dict:
    from bts import daily_decision, picks, scheduler

    with TemporaryDirectory(prefix=f"p6-{case}-", dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        offline = load_offline(tmp_path)
        at = lambda h, m, s=0: offline.instant(DAY, h, m, s)

        if case == "real_mdp_single_runner_up_reuse":
            # At streak=30, real shipped MDP chooses single on (0.99, 0.98).
            # Confirmed 0.99 vs Projected 0.98 (gap=1% < 3%) -> should_lock=False (gap block).
            # Game 102 check at 19:25 is infeasible before 19:05 cutoff -> fallback reuses & delivers!
            with offline.offline(at(18, 25)) as (w, _):
                w.dated(30, best=30)
                starts = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99, 0.98), (False, True))
                    return None

                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "reused": st.fallback_refreshes[0]["reused_check_cascade"],
                    "action": dec["action"] if dec else None,
                    "source": dec["source"] if dec else None,
                    "primary": dec["primary"] if dec else None,
                    "runner_up_in_double_down_meta": dec["double_down"] if dec else None,
                }

        if case in {"real_mdp_skip_survives_warmup", "real_mdp_skip_yields_to_genuine"}:
            # At streak=30, real shipped MDP genuinely skips p=0.60 (no monkeypatching resolve_policy_decision!).
            first = at(17, 40)
            second = at(18, 25)
            with offline.offline(first) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(30, best=30)
                w.predictions = lambda: w.frame((0.99,), (True,))
                scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)
                real_check = scheduler.run_single_check
                calls = 0

                def wrap_check(*a, **kw):
                    nonlocal calls
                    calls += 1
                    res = real_check(*a, **kw)
                    if calls == 1 and case == "real_mdp_skip_survives_warmup":
                        w.schedule[DAY][0]["status"].update(detailedState="Warmup", statusCode="PW")
                    return res

                def forecast():
                    if w.now == first:
                        return w.frame((0.60,), (True,))
                    if w.now == second:
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                mp.setattr(scheduler, "run_single_check", wrap_check)
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": first, "game_pks": [101]},
                    {"time_et": second, "game_pks": [101]},
                ])
                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                dec = daily_decision.load_decision(DAY, w.picks)
                pick_dms = [b for b in w.sends("dm") if b.startswith("Today's pick")]
                return {
                    "ok": True,
                    "pick_dms": len(pick_dms),
                    "decision_action": dec["action"] if dec else None,
                    "decision_source": dec["source"] if dec else None,
                    "scoreable": dec["scoreable"] if dec else None,
                }

        if case == "real_mdp_void_defer":
            # At streak=30, real shipped MDP chooses single on (0.99, 0.40).
            # Game 101 projected -> would defer for 18:50 -> status lookup sees Game 101 Postponed -> postponed_pick!
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(30, best=30)
                w.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(18, 50), "game_pks": [102]},
                ])
                mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
                    pk: {"away", "home"} for pk in pks if pk == 101
                })
                real_check = scheduler.run_single_check
                starts = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(22 * 60)
                        return w.frame((0.99, 0.40), (True, True))
                    if w.now >= at(18, 50):
                        return w.frame((0.40, 0.99), (True, False)).iloc[[1, 0]]
                    return None

                def postpone_first(*a, **kw):
                    res = real_check(*a, **kw)
                    if len(starts) == 1:
                        w.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    return res

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", postpone_first)
                scheduler.run_day(DAY, w.config)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "postponed_alerts": sum("[postponed_pick]" in a for a in w.sends("health-dm")),
                    "committed_pk": dec["primary"]["game_pk"] if dec else None,
                }

        if case == "void_deliver_past_cutoff_check":
            with offline.offline(at(18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                real_check = scheduler.run_single_check
                starts = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M:%S"))
                    if len(starts) == 1:
                        w.advance(40 * 60 + 30)
                        return w.frame((0.99,), (True,))
                    return None

                def postpone(*a, **kw):
                    res = real_check(*a, **kw)
                    if len(starts) == 1:
                        w.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    return res

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", postpone)
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                alerts = w.sends("health-dm")
                return {
                    "ok": True,
                    "refusal_reasons": [r["reason"] for r in (st.delivery_refusals or [])],
                    "postponed_alerts": sum("[postponed_pick]" in a for a in alerts),
                    "late_alerts": sum("[late_delivery]" in a for a in alerts),
                    "names_batter": any("Synthetic Batter 101" in a for a in alerts),
                }

        if case == "void_defer_crossing_cutoff_check":
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(30, best=30)
                w.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
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
                got_void = False

                def forecast():
                    starts.append(w.now.strftime("%H:%M:%S"))
                    if len(starts) == 1:
                        w.advance(39 * 60 + 56)
                        return w.frame((0.99, 0.40), (True, True))
                    return None

                def postpone(*a, **kw):
                    res = real_check(*a, **kw)
                    if len(starts) == 1:
                        w.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    return res

                def spy_status(d):
                    nonlocal got_void
                    out = real_statuses(d)
                    if len(starts) == 1:
                        got_void = True
                    return out

                def slow_save(st_obj, pdir):
                    r = real_save_state(st_obj, pdir)
                    if got_void and w.now < at(19, 5):
                        w.advance(5)
                    return r

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", postpone)
                mp.setattr(scheduler, "_delivery_game_statuses", spy_status)
                mp.setattr(scheduler, "save_state", slow_save)
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                alerts = w.sends("health-dm")
                return {
                    "ok": True,
                    "got_void": got_void,
                    "refusal_reasons": [r["reason"] for r in (st.delivery_refusals or [])],
                    "postponed_alerts": sum("[postponed_pick]" in a for a in alerts),
                    "late_alerts": sum("[late_delivery]" in a for a in alerts),
                }

        raise ValueError(case)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        print("PROBE6_JSON " + json.dumps(worker_main(sys.argv[2])), flush=True)
        return

    out = {"base": {}, "head": {}, "head_fixed_void_order": {}}
    for c in ("real_mdp_single_runner_up_reuse", "real_mdp_skip_survives_warmup", "real_mdp_skip_yields_to_genuine", "real_mdp_void_defer"):
        out["base"][c] = run_case(BASE_SRC, c)
        out["head"][c] = run_case(HEAD_SRC, c)
        print(f"[mdp] {c}: base.ok={out['base'][c].get('ok')} head.ok={out['head'][c].get('ok')}", flush=True)

    # Also test void_deliver_past_cutoff_check and void_defer_crossing_cutoff_check on base, head, and patched scratch head!
    fixed_src = REVIEW_DIR / "scratch_void_fix_src"
    if fixed_src.exists():
        shutil.rmtree(fixed_src)
    shutil.copytree(HEAD_SRC, fixed_src)
    sched_file = fixed_src / "bts/scheduler.py"
    old_block = """                    if reused_check_cascade:
                        now = _now_et()
                        if now >= cutoff:
                            _refuse_late_delivery(
                                daily, config, picks_dir, state, date, "fallback", cutoff, now)
                            continue
                        if (defer_void_games
                                and len(defer_void_games) == len(_committed_pick_game_pks(daily))):
                            _refuse_void_game_delivery(
                                daily, config, picks_dir, state, date, "fallback", defer_void_games)
                            continue"""
    new_block = """                    if reused_check_cascade and plan.action == "defer":
                        if (defer_void_games
                                and len(defer_void_games) == len(_committed_pick_game_pks(daily))):
                            _refuse_void_game_delivery(
                                daily, config, picks_dir, state, date, "fallback", defer_void_games)
                            continue
                        now = _now_et()
                        if now >= cutoff:
                            _refuse_late_delivery(
                                daily, config, picks_dir, state, date, "fallback", cutoff, now)
                            continue"""
    txt = sched_file.read_text()
    assert old_block in txt
    sched_file.write_text(txt.replace(old_block, new_block, 1))
    try:
        for c in ("void_deliver_past_cutoff_check", "void_defer_crossing_cutoff_check"):
            out["base"][c] = run_case(BASE_SRC, c)
            out["head"][c] = run_case(HEAD_SRC, c)
            out["head_fixed_void_order"][c] = run_case(fixed_src, c)
            print(f"[void_order] {c}: done", flush=True)
    finally:
        shutil.rmtree(fixed_src, ignore_errors=True)

    out_path = REVIEW_DIR / "probe_6_results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
