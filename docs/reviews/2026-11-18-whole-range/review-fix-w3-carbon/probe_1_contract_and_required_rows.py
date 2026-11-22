"""Probe 1: Required rows M1-M12, Check 1 & Check 2 items, and contract rulings on base vs head."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event
import time

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_ceab9854" / "src"
HEAD_SRC = REPO / "src"
DAY = "2027-03-24"


def load_offline(tmp_root: Path):
    source = REPO / "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py"
    spec = spec_from_file_location("probe1_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_worker(tree: str, case: str) -> dict:
    src_dir = str(BASE_SRC if tree == "base" else HEAD_SRC)
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir
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
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE1_JSON ")]
    if not lines:
        return {"ok": False, "error": "missing PROBE1_JSON", "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:]}
    return json.loads(lines[-1][len("PROBE1_JSON "):])


class BoundedAlarm(BaseException):
    pass


def worker_main(tree: str, case: str) -> dict:
    from tempfile import TemporaryDirectory
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy

    with TemporaryDirectory(prefix=f"p1-{tree}-{case}-", dir=REVIEW_DIR) as tmp:
        tmp_path = Path(tmp)
        offline = load_offline(tmp_path)

        if case == "ruling1_signature":
            sig = inspect.signature(scheduler._deliver_and_lock_pick)
            param = sig.parameters.get("reused_check_cascade")
            return {
                "ok": True,
                "has_kw": param is not None,
                "kind": str(param.kind) if param else None,
                "default": param.default if param else None,
            }

        if case == "m1_overrun":
            with offline.offline(offline.instant(DAY, 10)) as (w, _):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                starts = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if w.now < offline.instant(DAY, 18, 25):
                        return None
                    if w.now == offline.instant(DAY, 18, 25):
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    w.advance(6 * 60)
                    return None

                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                dms = [(t.strftime("%H:%M:%S"), b.splitlines()[0]) for k, t, b in w.events if k == "dm"]
                alerts = w.sends("health-dm")
                return {
                    "ok": True,
                    "starts": starts,
                    "refreshes": st.fallback_refreshes,
                    "dms": dms,
                    "alerts": [a.splitlines()[-1] for a in alerts],
                    "scoreable": dec["scoreable"] if dec else False,
                    "committed_batter": (dec.get("primary") or {}).get("batter_name") if dec else None,
                }

        if case in {"m2_replacement", "m2_audit_replacement", "m2_whitespace", "m2_wrong_pick_date"}:
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                path = w.picks / f"{DAY}.json"
                starts = []
                real_check = scheduler.run_single_check
                real_append = picks.append_lineup_evolution
                checked = False

                def replace_pick(daily_obj):
                    other = deepcopy(daily_obj)
                    other.pick = replace(daily_obj.pick, batter_id=999, batter_name="Replacement Batter")
                    picks.save_pick(other, w.picks)

                def append_hook(daily_obj, directory):
                    real_append(daily_obj, directory)
                    if case == "m2_audit_replacement" and daily_obj.pick.batter_id != 999 and not checked:
                        replace_pick(daily_obj)

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                def check_hook(*a, **kw):
                    nonlocal checked
                    res = real_check(*a, **kw)
                    checked = True
                    if case == "m2_replacement":
                        replace_pick(res["pick_result"].daily)
                    elif case == "m2_whitespace":
                        path.write_bytes(path.read_bytes() + b"\n")
                    elif case == "m2_wrong_pick_date":
                        payload = json.loads(path.read_text())
                        payload["date"] = "2027-03-23"
                        path.write_text(json.dumps(payload, indent=2))
                    return res

                w.predictions = forecast
                mp.setattr(picks, "append_lineup_evolution", append_hook)
                mp.setattr(scheduler, "run_single_check", check_hook)
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                dms = w.sends("dm")
                return {
                    "ok": True,
                    "starts": starts,
                    "reused": st.fallback_refreshes[0]["reused_check_cascade"],
                    "dm_count": len(dms),
                    "dm_matches_commit": bool(dec and dms and dec["primary"]["batter_name"] in dms[0]),
                    "committed_id": dec["primary"]["batter_id"] if dec else None,
                }

        if case == "m2_warmup_cache":
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, _):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(12, best=18, saver="active")
                w.predictions = lambda: w.frame((0.99,), (True,))
                first = scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)
                cached = first["pick_result"].daily
                w.schedule[DAY][0]["status"].update(detailedState="Warmup", statusCode="PW")

                def forecast():
                    w.advance(35 * 60)
                    return w.frame((0.99,), (True,))

                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "reused": st.fallback_refreshes[0]["reused_check_cascade"],
                    "primary": dec["primary"] if dec else None,
                    "expected_primary": scheduler._row_from_daily(cached.pick),
                    "objective": dec.get("objective") if dec else None,
                    "best_streak": dec.get("best_streak") if dec else None,
                }

        if case == "m2_single_runner_up":
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, _):
                w.dated(30, best=30)
                starts = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99, 0.85), (False, True))
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
                    "primary_pk": dec["primary"]["game_pk"] if dec and dec.get("primary") else None,
                    "runner_up_in_dd_meta": dec["double_down"]["game_pk"] if dec and dec.get("double_down") else None,
                }

        if case == "m2_nonmatching_selection":
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                real_check = scheduler.run_single_check

                def forecast():
                    w.advance(35 * 60)
                    return w.frame((0.99,), (True,))

                def tamper_selection(*a, **kw):
                    res = real_check(*a, **kw)
                    if res.get("selection") is not None:
                        bad_cand = dict(res["selection"].primary_candidate, p_game_hit=0.51)
                        res["selection"] = replace(res["selection"], primary_candidate=bad_cand)
                    return res

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", tamper_selection)
                scheduler.run_day(DAY, w.config)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "committed_p": dec["primary"]["p_game_hit"] if dec else None,
                }

        if case in {"m3_sleep", "m3_equality"}:
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, _):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                starts = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance((0 if case == "m3_sleep" else 10) * 60)
                        return w.frame((0.99,), (True,))
                    return None

                w.predictions = forecast
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "reused": st.fallback_refreshes[0]["reused_check_cascade"],
                    "duration_sec": st.fallback_refreshes[0]["duration_sec"],
                    "dm_count": len(w.sends("dm")),
                }

        if case == "m3_defer_no_coalesce_no_inherit":
            at = lambda h, m: offline.instant(DAY, h, m)
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(0)
                w.schedule[DAY].append(offline.game(103, at(21, 10)))
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(18, 45), "game_pks": [102]},
                    {"time_et": at(19, 25), "game_pks": [103]},
                ])
                mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
                    pk: {"away", "home"} for pk in pks if pk in {101, 102}
                })
                checks = []
                real_check = scheduler.run_single_check

                def wrap_check(*a, **kw):
                    checks.append(w.now.strftime("%H:%M"))
                    res = real_check(*a, **kw)
                    if len(checks) == 2 and "lock_decision" in res:
                        # Second check finishes at 19:00 before 19:35 deadline for game 102 (20:10):
                        # verify sleep happens and no reuse leaks.
                        pass
                    return res

                def forecast():
                    if len(checks) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99, 0.40, 0.40), (True, True, True))
                    if len(checks) == 2:
                        return w.frame((0.40, 0.99, 0.40), (True, False, True)).iloc[[1, 0, 2]]
                    return None

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", wrap_check)
                scheduler.run_day(DAY, w.config)
                st = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "checks": checks,
                    "coalesced": [r for r in st.runs_completed if r.get("reason") == "coalesced_after_fallback"],
                    "refreshes": st.fallback_refreshes,
                    "committed_pk": dec["primary"]["game_pk"] if dec else None,
                }

        if case in {"m4_skip_survives_warmup", "m4_skip_yields_to_genuine"}:
            first = offline.instant(DAY, 17, 40)
            second = offline.instant(DAY, 18, 25)
            with offline.offline(first) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                w.predictions = lambda: w.frame((0.99,), (True,))
                scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)
                orig_resolve = strategy.resolve_policy_decision

                def force_mdp_skip(*a, **kw):
                    d = orig_resolve(*a, **kw)
                    return replace(d, source="mdp") if d.action == "skip" else d

                real_check = scheduler.run_single_check
                calls = 0

                def wrap_check(*a, **kw):
                    nonlocal calls
                    calls += 1
                    res = real_check(*a, **kw)
                    if calls == 1 and case == "m4_skip_survives_warmup":
                        w.schedule[DAY][0]["status"].update(detailedState="Warmup", statusCode="PW")
                    return res

                def forecast():
                    if w.now == first:
                        return w.frame((0.10,), (True,))
                    if w.now == second:
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                mp.setattr(strategy, "resolve_policy_decision", force_mdp_skip)
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
                    "scoreable": dec["scoreable"] if dec else None,
                }

        if case in {"m5_marker_cutoff_dm", "m5_marker_cutoff_public"}:
            mode = "dm" if case.endswith("dm") else "public"
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.config["scheduler"]["pick_delivery"] = mode
                w.automatic(5, age_hours=16)
                real_write = picks.atomic_write_text
                starts = []
                attempt_flags = []

                def forecast():
                    starts.append(w.now.strftime("%H:%M:%S"))
                    if len(starts) == 1:
                        w.advance(2399)
                        return w.frame((0.99,), (True,))
                    w.advance(2)
                    return None

                def slow_marker_write(p, text):
                    real_write(p, text)
                    if p == w.picks / f"{DAY}.json":
                        flag = json.loads(text)["delivery_attempted"]
                        attempt_flags.append(flag)
                        if flag:
                            w.advance(1.25)

                w.predictions = forecast
                mp.setattr(picks, "atomic_write_text", slow_marker_write)
                scheduler.run_day(DAY, w.config)
                archived = list((w.picks / DAY).glob("refused_delivery*.json"))
                arch_data = json.loads(archived[0].read_text()) if archived else None
                return {
                    "ok": True,
                    "starts": starts,
                    "attempt_flags": attempt_flags,
                    "sent": len(w.sends("dm") + w.sends("post")),
                    "late_alerts": sum("[late_delivery]" in a for a in w.sends("health-dm")),
                    "stale_warns": sum("stale" in a.lower() for a in w.sends("health-dm")),
                    "archived_attempt_flag": arch_data["delivery_attempted"] if arch_data else None,
                    "archived_auto_reading": arch_data["automatic_reading"] if arch_data else "missing",
                    "decision": daily_decision.load_decision(DAY, w.picks),
                }

        if case == "m5_reload_cutoff_with_pending_window":
            at = lambda h, m: offline.instant(DAY, h, m)
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(0)
                w.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(18, 50), "game_pks": [102]},
                ])
                mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
                    pk: {"away", "home"} for pk in pks if pk == 101
                })
                starts = []
                real_load_bytes = picks.load_pick_bytes
                advanced = False

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(22 * 60)
                        return w.frame((0.99, 0.70), (True, True))
                    if w.now >= at(19, 5):
                        f = w.frame((0.70, 0.99), (True, False))
                        return f.iloc[[1, 0]]
                    return w.frame((0.99, 0.70), (True, True))

                def slow_reload(*a, **kw):
                    nonlocal advanced
                    out = real_load_bytes(*a, **kw)
                    if starts and not advanced and not kw.get("regular_only"):
                        advanced = True
                        w.advance(18 * 60)
                    return out

                w.predictions = forecast
                mp.setattr(picks, "load_pick_bytes", slow_reload)
                scheduler.run_day(DAY, w.config)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "starts": starts,
                    "late_alerts": sum("[late_delivery]" in a for a in w.sends("health-dm")),
                    "deferred_archives": len(list((w.picks / DAY).glob("deferred_fallback*.json"))),
                    "refused_archives": len(list((w.picks / DAY).glob("refused_delivery*.json"))),
                    "committed_pk": dec["primary"]["game_pk"] if dec else None,
                }

        if case in {"m6_best", "m6_saver", "m6_auto_to_auto"}:
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                if case == "m6_auto_to_auto":
                    w.automatic(2)
                else:
                    w.dated(0)
                real_check = scheduler.run_single_check

                def forecast():
                    if w.now == offline.instant(DAY, 18, 25):
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                def mutate(*a, **kw):
                    res = real_check(*a, **kw)
                    if case == "m6_best":
                        w.dated(0, best=5)
                    elif case == "m6_saver":
                        w.dated(0, saver="not_earned")
                    else:
                        w.automatic(9)
                    return res

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", mutate)
                scheduler.run_day(DAY, w.config)
                return {
                    "ok": True,
                    "dms": w.sends("dm"),
                    "contest_alerts": sum("[contest_state]" in a for a in w.sends("health-dm")),
                    "missed_alerts": sum("[missed_pick]" in a for a in w.sends("health-dm")),
                    "decision": daily_decision.load_decision(DAY, w.picks),
                }

        if case in {"m7_stale_dm", "m7_stale_public", "m7_unconfirmed_no_warn", "m7_failed_retries"}:
            mode = "public" if case == "m7_stale_public" else "dm"
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.config["scheduler"]["pick_delivery"] = mode
                w.automatic(3, age_hours=18)
                real_dm = dm.send_dm
                calls = 0

                def forecast():
                    if w.now == offline.instant(DAY, 18, 25):
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                def custom_dm(rec, body):
                    nonlocal calls
                    calls += 1
                    if case == "m7_unconfirmed_no_warn":
                        return ""
                    if case == "m7_failed_retries" and calls == 1:
                        raise RuntimeError("transient 502")
                    return real_dm(rec, body)

                w.predictions = forecast
                if mode == "dm" and case in {"m7_unconfirmed_no_warn", "m7_failed_retries"}:
                    mp.setattr(dm, "send_dm", custom_dm)
                scheduler.run_day(DAY, w.config)
                saved = picks.load_pick(DAY, w.picks)
                return {
                    "ok": True,
                    "calls": calls,
                    "sent": len(w.sends("dm" if mode == "dm" else "post")),
                    "stale_flag": (saved.automatic_reading or {}).get("stale") if saved else None,
                    "stale_warns": sum("stale" in a.lower() for a in w.sends("health-dm")),
                }

        if case in {"m8_void_deliver", "m8_partial_double", "m8_void_defer"}:
            at = lambda h, m: offline.instant(DAY, h, m)
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(0)
                if case == "m8_void_deliver":
                    w.schedule[DAY] = w.schedule[DAY][:1]
                elif case == "m8_void_defer":
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
                        w.advance((22 if case == "m8_void_defer" else 35) * 60)
                        if case == "m8_partial_double":
                            return w.frame((0.99, 0.98), (True, True))
                        return w.frame((0.99, 0.40)[:len(w.schedule[DAY])], (True, True)[:len(w.schedule[DAY])])
                    if case == "m8_void_defer" and w.now >= at(18, 50):
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
                    "dms": w.sends("dm"),
                    "postponed_alerts": sum("[postponed_pick]" in a for a in w.sends("health-dm")),
                    "decision_action": dec["action"] if dec else None,
                    "committed_pk": dec["primary"]["game_pk"] if dec and dec.get("primary") else None,
                }

        if case.startswith("m9_"):
            kind = case[len("m9_"):]
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                path = w.picks / f"{DAY}.json"
                real_check = scheduler.run_single_check
                real_path_open = Path.open
                checked = False
                expected_bytes = None

                def forecast():
                    if not checked:
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                def mutate_pick(*a, **kw):
                    nonlocal checked, expected_bytes
                    res = real_check(*a, **kw)
                    checked = True
                    raw = path.read_bytes()
                    if kind == "absence":
                        path.unlink()
                    elif kind == "dangling":
                        path.unlink()
                        path.symlink_to(w.picks / "missing_target.json")
                    elif kind == "symlink_regular":
                        target = w.picks / "real_target.json"
                        target.write_bytes(raw)
                        path.unlink()
                        path.symlink_to(target)
                    elif kind == "json":
                        path.write_text("{")
                        expected_bytes = b"{"
                    elif kind == "shape":
                        path.write_text("[]")
                        expected_bytes = b"[]"
                    elif kind == "invalid_time":
                        obj = json.loads(raw)
                        obj["pick"]["game_time"] = "not-a-timestamp"
                        path.write_text(json.dumps(obj))
                        expected_bytes = path.read_bytes()
                    elif kind == "directory":
                        path.unlink()
                        path.mkdir()
                    elif kind == "fifo":
                        path.unlink()
                        os.mkfifo(path)
                    elif kind == "unreadable":
                        expected_bytes = raw
                    return res

                def deny_open(self_path, *a, **kw):
                    if checked and self_path == path:
                        raise PermissionError("denied")
                    return real_path_open(self_path, *a, **kw)

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", mutate_pick)
                if kind == "unreadable":
                    mp.setattr(Path, "open", deny_open)

                def on_alarm(*_):
                    raise BoundedAlarm("fifo_blocked")

                old_handler = signal.signal(signal.SIGALRM, on_alarm)
                signal.alarm(2)
                raised = None
                try:
                    scheduler.run_day(DAY, w.config)
                except BoundedAlarm as exc:
                    raised = str(exc)
                except Exception as exc:
                    raised = type(exc).__name__
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                st = scheduler.load_state(DAY, w.picks)
                return {
                    "ok": True,
                    "raised": raised,
                    "dm_count": len(w.sends("dm")),
                    "lexists": os.path.lexists(path),
                    "bytes_preserved": (path.read_bytes() == expected_bytes) if expected_bytes is not None else True,
                    "reused": (st.fallback_refreshes[0]["reused_check_cascade"]
                               if st and st.fallback_refreshes else None),
                }

        if case.startswith("c2_dec_"):
            kind = case[len("c2_dec_"):]
            with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                dec_path = daily_decision.decision_path(DAY, w.picks)
                polled = []
                mp.setattr(scheduler, "run_result_polling", lambda pk, *a, **kw: polled.append(pk) or "final")
                real_check = scheduler.run_single_check
                real_os_open = os.open
                starts = []
                saved_evidence = None

                def forecast():
                    starts.append(w.now.strftime("%H:%M"))
                    if len(starts) == 1:
                        w.advance(35 * 60)
                        return w.frame((0.99,), (True,))
                    return None

                def plant_decision(*a, **kw):
                    nonlocal saved_evidence
                    res = real_check(*a, **kw)
                    dec_path.parent.mkdir(parents=True, exist_ok=True)
                    if kind in {"scoreable_mismatch", "scoreable_match", "wrong_date", "unscoreable"}:
                        prim = (scheduler._row_from_daily(res["pick_result"].daily.pick)
                                if kind == "scoreable_match"
                                else {"batter_id": 999, "batter_name": "Other", "game_pk": 102})
                        daily_decision.write_decision(
                            DAY, w.picks, action="single", source="mdp",
                            scoreable=(kind != "unscoreable"),
                            delivery_status="private_locked", primary=prim,
                        )
                        if kind == "wrong_date":
                            rec = json.loads(dec_path.read_text())
                            rec["date"] = "2027-03-23"
                            dec_path.write_text(json.dumps(rec))
                        saved_evidence = dec_path.read_bytes()
                    elif kind == "invalid_json":
                        dec_path.write_text("{")
                    elif kind == "unhashable_action":
                        dec_path.write_text(json.dumps({
                            "schema_version": "bts_daily_decision_v3",
                            "date": DAY, "action": ["bad"], "scoreable": True,
                        }))
                    elif kind == "empty":
                        dec_path.write_text("")
                    elif kind == "directory":
                        dec_path.mkdir()
                    elif kind == "dangling":
                        dec_path.symlink_to(dec_path.with_name("missing.json"))
                    elif kind == "fifo":
                        os.mkfifo(dec_path)
                    elif kind == "unreadable":
                        dec_path.write_text("{}")
                    return res

                def deny_dec_open(p, flags, *a, **kw):
                    if Path(p) == dec_path and (flags & os.O_NONBLOCK):
                        raise PermissionError("denied decision open")
                    return real_os_open(p, flags, *a, **kw)

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", plant_decision)
                if kind == "unreadable":
                    mp.setattr(os, "open", deny_dec_open)

                def on_alarm(*_):
                    raise BoundedAlarm("decision_fifo_blocked")

                old_handler = signal.signal(signal.SIGALRM, on_alarm)
                signal.alarm(2)
                raised = None
                try:
                    scheduler.run_day(DAY, w.config)
                except BoundedAlarm as exc:
                    raised = str(exc)
                except Exception as exc:
                    raised = type(exc).__name__
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                st = scheduler.load_state(DAY, w.picks)
                return {
                    "ok": True,
                    "raised": raised,
                    "starts": starts,
                    "dms": len(w.sends("dm")),
                    "polled": polled,
                    "reused": (st.fallback_refreshes[0]["reused_check_cascade"]
                               if st and st.fallback_refreshes else None),
                    "scoreable_preserved": (
                        dec_path.read_bytes() == saved_evidence
                        if kind in {"scoreable_mismatch", "scoreable_match"} else None
                    ),
                }

        if case.startswith("lw3_2_"):
            # lw3_2_<day>_<kind>_<crossing>
            parts = case[len("lw3_2_"):].split("_")
            day_key, state_kind, crossing = parts[0], parts[1], "_".join(parts[2:])
            day_str = "2027-06-18" if day_key == "jun" else "2026-12-31"
            start = offline.instant(day_str, 23, 20)
            with offline.offline(start) as (w, mp):
                w.schedule[day_str] = [offline.game(101, start + timedelta(minutes=70))]
                w.config["scheduler"].update(fallback_deadline_min=40, fallback_deadline_min_morning=40)
                if state_kind == "auto":
                    w.automatic(4)
                else:
                    w.dated(0)
                starts = []
                reads = []
                real_check = scheduler.run_single_check
                real_read = contest_state.load_decision_streak_state
                real_write = picks.atomic_write_text
                checked = False

                def forecast():
                    starts.append(w.now.isoformat())
                    if len(starts) == 1:
                        w.advance(2399)
                        return w.frame((0.99,), (True,))
                    w.advance(2)
                    return None

                def check_wrap(*a, **kw):
                    nonlocal checked
                    res = real_check(*a, **kw)
                    checked = True
                    if crossing == "before_read":
                        w.advance(2)
                    return res

                def read_wrap(*a, **kw):
                    if checked:
                        reads.append(w.now.date().isoformat())
                    out = real_read(*a, **kw)
                    if checked and crossing == "during_read" and len(reads) == 1:
                        w.advance(2)
                    elif checked and crossing == "double_rollover" and len(reads) <= 2:
                        w.advance(2 if len(reads) == 1 else 86400)
                    return out

                def write_wrap(p, text):
                    real_write(p, text)
                    if (crossing == "during_marker" and p == w.picks / f"{day_str}.json"
                            and json.loads(text)["delivery_attempted"]):
                        w.advance(2)

                w.predictions = forecast
                mp.setattr(strategy, "_load_mdp", lambda: None)
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": start, "game_pks": [101]}])
                mp.setattr(scheduler, "run_single_check", check_wrap)
                mp.setattr(contest_state, "load_decision_streak_state", read_wrap)
                mp.setattr(picks, "atomic_write_text", write_wrap)
                scheduler.run_day(day_str, w.config)
                p_after = picks.load_pick(day_str, w.picks)
                return {
                    "ok": True,
                    "dms": len(w.sends("dm")),
                    "contest_alerts": [a for a in w.sends("health-dm") if "[contest_state]" in a],
                    "reads": reads,
                    "attempt_cleared": (not p_after.delivery_attempted) if p_after else True,
                    "decision": daily_decision.load_decision(day_str, w.picks),
                }

        if case == "shared_budget_and_late_daemon":
            at = lambda h, m, s=0: offline.instant(DAY, h, m, s)
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(0)
                w.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                w.schedule[DAY].append(offline.game(103, at(19, 50)))
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(18, 50), "game_pks": [102]},
                    {"time_et": at(19, 5), "game_pks": [103]},
                ])
                release_slow = Event()
                slow_done = Event()
                starts = []
                observed_confirmed_snapshots = []
                real_plan = scheduler.plan_fallback_action

                def forecast():
                    starts.append(w.now.strftime("%H:%M:%S"))
                    if len(starts) == 1:
                        w.advance(22 * 60)
                        return w.frame((0.99, 0.70, 0.60), (True, True, True))
                    return w.frame((0.70, 0.99, 0.60), (True, False, True)).iloc[[1, 0, 2]]

                def confirm_lookup(pks):
                    if starts and pks == [102]:
                        time.sleep(0.4)
                        return {102: {"away"}}
                    if starts and pks == [103]:
                        release_slow.wait(timeout=6)
                        slow_done.set()
                        return {103: {"away", "home"}}
                    return {pk: {"away", "home"} for pk in pks if pk == 101}

                def status_lookup(d):
                    if len(starts) == 1:
                        release_slow.wait(timeout=6)
                        return {101: {"detailed": "Postponed"}}
                    return {101: {"detailed": "Scheduled"}, 102: {"detailed": "Scheduled"}, 103: {"detailed": "Scheduled"}}

                def spy_plan(**kw):
                    observed_confirmed_snapshots.append(sorted(kw["confirmed_sides"]))
                    return real_plan(**kw)

                w.predictions = forecast
                mp.setattr(scheduler, "check_confirmed_lineups", confirm_lookup)
                mp.setattr(scheduler, "_delivery_game_statuses", status_lookup)
                mp.setattr(scheduler, "plan_fallback_action", spy_plan)

                t0 = time.monotonic()
                try:
                    scheduler.run_day(DAY, w.config)
                    elapsed = time.monotonic() - t0
                    still_blocked = not slow_done.is_set()
                finally:
                    release_slow.set()
                    slow_done.wait(timeout=2)
                st_after_late = scheduler.load_state(DAY, w.picks)
                dec = daily_decision.load_decision(DAY, w.picks)
                return {
                    "ok": True,
                    "elapsed_sec": round(elapsed, 2),
                    "later_check_ran_before_slow_finished": still_blocked,
                    "starts": starts,
                    "confirmed_snapshots": observed_confirmed_snapshots,
                    "state_confirmed_pks": st_after_late.confirmed_game_pks,
                    "committed_pk": dec["primary"]["game_pk"] if dec else None,
                }

        if case == "cutoff_within_3s_skips_polls":
            at = lambda h, m, s=0: offline.instant(DAY, h, m, s)
            with offline.offline(at(18, 25)) as (w, mp):
                w.dated(0)
                w.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                    {"time_et": at(18, 25), "game_pks": [101]},
                    {"time_et": at(19, 10), "game_pks": [102]},
                ])
                poll_calls = []
                status_calls = []
                real_statuses = scheduler._delivery_game_statuses
                checked = False
                real_check = scheduler.run_single_check

                def forecast():
                    if not checked:
                        # Advance to 19:04:57.5 -> 2.5s before 19:05:00 cutoff (<= 3s)
                        w.advance(39 * 60 + 57.5)
                        return w.frame((0.99, 0.70), (True, True))
                    return w.frame((0.70, 0.99), (True, False)).iloc[[1, 0]]

                def wrap_check(*a, **kw):
                    nonlocal checked
                    res = real_check(*a, **kw)
                    checked = True
                    return res

                def spy_confirm(pks):
                    if checked and w.now < at(19, 5):
                        poll_calls.append(list(pks))
                    return {pk: {"away", "home"} for pk in pks if pk == 101}

                def spy_status(d):
                    if checked and w.now < at(19, 5):
                        status_calls.append(w.now.isoformat())
                    return real_statuses(d)

                w.predictions = forecast
                mp.setattr(scheduler, "run_single_check", wrap_check)
                mp.setattr(scheduler, "check_confirmed_lineups", spy_confirm)
                mp.setattr(scheduler, "_delivery_game_statuses", spy_status)
                scheduler.run_day(DAY, w.config)
                return {
                    "ok": True,
                    "poll_calls_near_cutoff": poll_calls,
                    "status_calls_near_cutoff": status_calls,
                    "deferred_archives": len(list((w.picks / DAY).glob("deferred_fallback*.json"))),
                }

        if case == "m11_early_returns":
            shapes = {}
            for kind in ("no_predictions", "mdp_skip", "contest_error", "status_locked", "scoreable_locked", "write_error"):
                with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
                    w.schedule[DAY] = w.schedule[DAY][:1]
                    w.dated(30, best=30)
                    w.predictions = lambda: w.frame((0.99,), (True,))
                    if kind == "status_locked":
                        d = scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)["pick_result"].daily
                        d.delivery_attempted = True
                        picks.save_pick(d, w.picks)
                    elif kind == "scoreable_locked":
                        d = scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)["pick_result"].daily
                        daily_decision.write_decision(
                            DAY, w.picks, action="single", source="mdp",
                            scoreable=True, delivery_status="private_locked",
                            primary=scheduler._row_from_daily(d.pick),
                        )
                    elif kind == "no_predictions":
                        w.predictions = lambda: None
                    elif kind == "mdp_skip":
                        w.predictions = lambda: w.frame((0.10,), (True,))
                    elif kind == "contest_error":
                        w.predictions = lambda: (_ for _ in ()).throw(contest_state.ContestStateError("bad state"))
                    elif kind == "write_error":
                        mp.setattr(picks, "atomic_write_text", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
                        try:
                            scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)
                        except OSError:
                            shapes[kind] = "raised_OSError"
                        continue
                    res = scheduler.run_single_check(DAY, [101], set(), w.config, 0.03)
                    shapes[kind] = {
                        "keys": sorted(res.keys()),
                        "has_lock_decision": "lock_decision" in res,
                        "has_saved_bytes": "saved_pick_bytes" in res,
                    }
            return {"ok": True, "shapes": shapes}

        raise ValueError(f"Unknown case {case}")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        out = worker_main(sys.argv[2], sys.argv[3])
        print("PROBE1_JSON " + json.dumps(out), flush=True)
        return

    cases_both = [
        "ruling1_signature",
        "m1_overrun",
        "m2_replacement",
        "m2_audit_replacement",
        "m2_whitespace",
        "m2_wrong_pick_date",
        "m2_warmup_cache",
        "m2_single_runner_up",
        "m2_nonmatching_selection",
        "m3_sleep",
        "m3_equality",
        "m3_defer_no_coalesce_no_inherit",
        "m4_skip_survives_warmup",
        "m4_skip_yields_to_genuine",
        "m5_marker_cutoff_dm",
        "m5_marker_cutoff_public",
        "m5_reload_cutoff_with_pending_window",
        "m6_best",
        "m6_saver",
        "m6_auto_to_auto",
        "m7_stale_dm",
        "m7_stale_public",
        "m7_unconfirmed_no_warn",
        "m7_failed_retries",
        "m8_void_deliver",
        "m8_partial_double",
        "m8_void_defer",
        "m9_absence",
        "m9_dangling",
        "m9_symlink_regular",
        "m9_json",
        "m9_shape",
        "m9_invalid_time",
        "m9_unreadable",
        "m9_directory",
        "m9_fifo",
        "m11_early_returns",
    ]
    cases_head_only = [
        "c2_dec_scoreable_mismatch",
        "c2_dec_scoreable_match",
        "c2_dec_unscoreable",
        "c2_dec_wrong_date",
        "c2_dec_invalid_json",
        "c2_dec_unhashable_action",
        "c2_dec_empty",
        "c2_dec_directory",
        "c2_dec_dangling",
        "c2_dec_fifo",
        "c2_dec_unreadable",
        "lw3_2_jun_dated_before_read",
        "lw3_2_jun_dated_during_read",
        "lw3_2_jun_dated_during_marker",
        "lw3_2_jun_dated_double_rollover",
        "lw3_2_jun_auto_during_read",
        "lw3_2_jun_auto_during_marker",
        "lw3_2_ny_dated_during_read",
        "lw3_2_ny_auto_during_read",
        "shared_budget_and_late_daemon",
        "cutoff_within_3s_skips_polls",
    ]

    results = {"base": {}, "head": {}}
    for c in cases_both:
        results["base"][c] = run_worker("base", c)
        results["head"][c] = run_worker("head", c)
        print(f"[both] {c}: base.ok={results['base'][c].get('ok')} head.ok={results['head'][c].get('ok')}", flush=True)
    for c in cases_head_only:
        results["head"][c] = run_worker("head", c)
        print(f"[head] {c}: head.ok={results['head'][c].get('ok')}", flush=True)

    out_path = REVIEW_DIR / "probe_1_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
