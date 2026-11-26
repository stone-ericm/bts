"""Probe 8: Verify 59b72c04-save-hook.patch on target_59b72c04/src across save_pick hooks, hostile wrappers, and void-vs-cutoff."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
TARGET_SRC = REVIEW_DIR / "target_59b72c04" / "src"
DAY = "2027-03-24"


def load_offline(tmp_root: Path):
    source = REPO / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("probe8_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_case(case: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TARGET_SRC)
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
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE8_JSON ")]
    return json.loads(lines[-1][len("PROBE8_JSON "):])


def worker_main(case: str) -> dict:
    from bts import daily_decision, picks, scheduler
    assert Path(picks.__file__).resolve().is_relative_to(TARGET_SRC.resolve())

    with TemporaryDirectory(prefix=f"p8-{case}-", dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        offline = load_offline(tmp_path)
        at = lambda h, m, s=0: offline.instant(DAY, h, m, s)

        if case in {"wr_a_pick_dm", "wr_a_pick_public"}:
            mode = "dm" if case.endswith("dm") else "public"
            with offline.offline(at(18, 25)) as (world, patch):
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

                def delivery(daily_obj, config, directory, state, day, label, *, selection=None):
                    delivered = deliver(daily_obj, config, directory, state, day, label, selection=selection)
                    if not refusals:
                        assert delivered is False and not state.pick_locked
                        refusals.append(label)
                    return delivered

                patch.setattr(scheduler, "_deliver_and_lock_pick", delivery)
                scheduler.run_day(DAY, world.config)
                final_pick = picks.load_pick(DAY, world.picks)
                return {
                    "ok": True,
                    "refusals": refusals,
                    "final_streak": final_pick.chosen_on["streak"],
                    "final_single": final_pick.double_down is None,
                    "sent": len(world.sends("dm") + world.sends("post")),
                    "contest_alerts": sum("contest_state" in b for b in world.sends("health-dm")),
                }

        if case == "wr_a_carbon_midnight":
            with offline.offline(at(23, 58)) as (world, patch):
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

                def save_and_roll(daily_obj, directory):
                    nonlocal rolled
                    saved = save_orig(daily_obj, directory)
                    if not rolled:
                        rolled = True
                        world.now = offline.instant("2027-03-25", 0, 1)
                    return saved

                patch.setattr(picks, "save_pick", save_and_roll)
                res = scheduler.run_single_check(DAY, [101, 102], set(), world.config, 0.03)
                st = scheduler.SchedulerState(DAY, world.now.isoformat(), [], [], [], False, None, None, None)
                delivered = scheduler._deliver_and_lock_pick(
                    res["pick_result"].daily, world.config, world.picks, st, DAY, "lineup",
                    selection=res["selection"],
                )
                return {
                    "ok": True,
                    "rolled": rolled,
                    "delivered": delivered,
                    "contest_alerts": sum("contest_state" in b for b in world.sends("health-dm")),
                }

        if case == "slice3_cached_fallbacks_in_loop":
            with offline.offline(at(18, 20)) as (w, mp):
                w.dated(0)
                w.schedule[DAY] = w.schedule[DAY][:1]
                orig_save = picks.save_pick
                corrected = False

                def save_and_correct(d, directory):
                    nonlocal corrected
                    p = orig_save(d, directory)
                    if not corrected:
                        corrected = True
                        w.dated(30)
                    return p

                mp.setattr(picks, "save_pick", save_and_correct)
                w.predictions = lambda: (None if corrected else w.frame((0.99,), (True,)))
                scheduler.run_day(DAY, w.config)
                return {
                    "ok": True,
                    "predict_times": [t.strftime("%H:%M") for k, t, _ in w.events if k == "predict"],
                    "dm_count": len(w.sends("dm")),
                    "contest_alerts": sum("Saved pick" in b for b in w.sends("health-dm")),
                }

        if case in {"wrapper_deepcopy", "wrapper_noop_mock", "wrapper_inplace_second_save", "wrapper_audit_replace"}:
            with offline.offline(at(18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                orig_save = picks.save_pick
                orig_append = picks.append_lineup_evolution
                starts = []
                hooked = False

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                def deepcopy_save(d, directory):
                    return orig_save(deepcopy(d), directory)

                def noop_save(d, directory):
                    nonlocal hooked
                    if not hooked:
                        hooked = True
                        return Path(directory) / f"{d.date}.json"
                    return orig_save(d, directory)

                def inplace_second_save(d, directory):
                    nonlocal hooked
                    p = orig_save(d, directory)
                    if not hooked:
                        hooked = True
                        d.pick = replace(d.pick, batter_id=999, batter_name="Mutated InPlace")
                        return orig_save(d, directory)
                    return p

                def audit_replace(d, directory):
                    orig_append(d, directory)
                    if not hooked and d.pick.batter_id != 999:
                        nonlocal hooked
                        hooked = True
                        other = deepcopy(d)
                        other.pick = replace(d.pick, batter_id=999, batter_name="Audit Replacement")
                        orig_save(other, directory)

                w.predictions = forecast
                if case == "wrapper_deepcopy":
                    mp.setattr(picks, "save_pick", deepcopy_save)
                elif case == "wrapper_noop_mock":
                    mp.setattr(picks, "save_pick", noop_save)
                elif case == "wrapper_inplace_second_save":
                    mp.setattr(picks, "save_pick", inplace_second_save)
                elif case == "wrapper_audit_replace":
                    mp.setattr(picks, "append_lineup_evolution", audit_replace)

                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                dms = w.sends("dm")
                return {
                    "ok": True,
                    "starts": starts,
                    "reused": st.fallback_refreshes[0]["reused_check_cascade"] if st.fallback_refreshes else None,
                    "dm_count": len(dms),
                    "dm_matches_commit": bool(dec and dms and dec["primary"]["batter_name"] in dms[0]),
                    "committed_id": dec["primary"]["batter_id"] if dec else None,
                }

        if case in {"void_deliver_past_cutoff", "void_defer_crossing_cutoff"}:
            with offline.offline(at(18, 25)) as (w, mp):
                if case == "void_deliver_past_cutoff":
                    w.schedule[DAY] = w.schedule[DAY][:1]
                    w.dated(0)
                else:
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
                        w.advance(40 * 60 + 30 if case == "void_deliver_past_cutoff" else 39 * 60 + 56)
                        return w.frame((0.99, 0.40)[:len(w.schedule[DAY])], (True, True)[:len(w.schedule[DAY])])
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
                    if case == "void_defer_crossing_cutoff" and got_void and w.now < at(19, 5):
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
                    "refusal_reasons": [r["reason"] for r in (st.delivery_refusals or [])],
                    "postponed_alerts": sum("[postponed_pick]" in a for a in alerts),
                    "late_alerts": sum("[late_delivery]" in a for a in alerts),
                    "names_batter": any("Synthetic Batter 101" in a for a in alerts),
                }

        raise ValueError(case)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        print("PROBE8_JSON " + json.dumps(worker_main(sys.argv[2])), flush=True)
        return

    cases = [
        "wr_a_pick_dm",
        "wr_a_pick_public",
        "wr_a_carbon_midnight",
        "slice3_cached_fallbacks_in_loop",
        "wrapper_deepcopy",
        "wrapper_noop_mock",
        "wrapper_inplace_second_save",
        "wrapper_audit_replace",
        "void_deliver_past_cutoff",
        "void_defer_crossing_cutoff",
    ]
    out = {}
    for c in cases:
        out[c] = run_case(c)
        print(f"[59b72c04] {c}: ok={out[c].get('ok')}", flush=True)

    out_path = REVIEW_DIR / "probe_8_59b72c04.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
