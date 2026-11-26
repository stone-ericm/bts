"""Round-2 Probe 2c: Adversarial checks of R1-R5 changes (second planner run, double-fail & interrupt in marker-clear retry, receipt dedup across C2 restart, untrusted decision + attempt recovery, void/late order, shadow on all 4 break paths, and operator text)."""

from __future__ import annotations

from contextlib import redirect_stderr
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import current_thread, main_thread

REPO = Path(__file__).resolve().parents[3]
R2_DIR = Path(__file__).resolve().parent
HEAD_SRC = REPO / "src"
DAY = "2027-03-24"


def load_offline(tmp_root: Path):
    source = REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "slice-3-codex" / "offline.py"
    spec = spec_from_file_location("r2_p2c_offline", source)
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = REPO
    mod.HERE = tmp_root
    return mod


def run_case(case: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HEAD_SRC)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), __file__, "--worker", case],
        capture_output=True,
        text=True,
        timeout=25,
        env=env,
    )
    if proc.returncode != 0:
        return {"ok": False, "rc": proc.returncode, "stderr": proc.stderr[-1500:], "stdout": proc.stdout[-1500:]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("R2_P2C_JSON ")]
    return json.loads(lines[-1][len("R2_P2C_JSON ") :])


def worker_main(case: str) -> dict:
    from bts import daily_decision, picks, scheduler, strategy

    assert Path(scheduler.__file__).resolve().is_relative_to(HEAD_SRC.resolve())

    with TemporaryDirectory(prefix=f"r2-p2c-{case}-", dir=R2_DIR) as tmp:
        offline = load_offline(Path(tmp))
        at = lambda h, m=0, s=0, *, day=DAY: offline.instant(day, h, m, s)

        if case in {"r2_flip_void_consistent", "r2_flip_void_transient_second_read"}:
            with offline.offline(at(18, 25)) as (world, mp):
                world.dated(30, best=30)
                mp.setattr(
                    scheduler,
                    "plan_run_times",
                    lambda *a, **kw: [
                        {"time_et": at(18, 25), "game_pks": [101]},
                        {"time_et": at(18, 42), "game_pks": [102]},
                    ],
                )
                real_check = scheduler.run_single_check
                real_status = scheduler._delivery_game_statuses
                starts = []
                status_calls = []
                stderr = io.StringIO()

                def forecast():
                    starts.append(world.now)
                    if len(starts) == 1:
                        world.advance(17 * 60 + 59)
                        return world.frame((0.99, 0.98), (False, True))
                    return None

                def check_postpone(*a, **kw):
                    res = real_check(*a, **kw)
                    if len(starts) == 1:
                        world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    return res

                def statuses(date):
                    is_worker = current_thread() is not main_thread()
                    status_calls.append("poll_worker" if is_worker else "delivery_main")
                    if is_worker:
                        world.advance(2)
                        return real_status(date)
                    if case == "r2_flip_void_transient_second_read":
                        raise OSError("transient status timeout on second unbudgeted read")
                    return real_status(date)

                world.predictions = forecast
                mp.setattr(scheduler, "run_single_check", check_postpone)
                mp.setattr(scheduler, "_delivery_game_statuses", statuses)
                mp.setattr(
                    scheduler,
                    "check_confirmed_lineups",
                    lambda pks: {pk: {"away", "home"} for pk in pks if pk == 101},
                )
                with redirect_stderr(stderr):
                    scheduler.run_day(DAY, world.config)
                st = scheduler.load_state(DAY, world.picks)
                alerts = world.sends("health-dm")
                dms = world.sends("dm")
                return {
                    "ok": True,
                    "status_calls": status_calls,
                    "refresh_action": st.fallback_refreshes[0]["action"],
                    "dm_count": len(dms),
                    "postponed_alerts": sum("[postponed_pick]" in a for a in alerts),
                    "printed_delivering": "FALLBACK — delivering" in stderr.getvalue(),
                }

        if case == "r2_poll_crosses_cutoff_keeps_defer_void_order":
            with offline.offline(at(18, 25)) as (world, mp):
                world.dated(30, best=30)
                mp.setattr(
                    scheduler,
                    "plan_run_times",
                    lambda *a, **kw: [
                        {"time_et": at(18, 25), "game_pks": [101]},
                        {"time_et": at(18, 50), "game_pks": [102]},
                    ],
                )
                real_check = scheduler.run_single_check
                real_status = scheduler._delivery_game_statuses
                starts = []
                status_calls = 0

                def forecast():
                    starts.append(world.now)
                    if len(starts) == 1:
                        world.advance(39 * 60 + 56)
                        return world.frame((0.99, 0.40), (True, True))
                    return None

                def check_postpone(*a, **kw):
                    res = real_check(*a, **kw)
                    if len(starts) == 1:
                        world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                    return res

                def statuses(date):
                    nonlocal status_calls
                    if current_thread() is not main_thread():
                        status_calls += 1
                        world.advance(5)
                    return real_status(date)

                world.predictions = forecast
                mp.setattr(scheduler, "run_single_check", check_postpone)
                mp.setattr(scheduler, "_delivery_game_statuses", statuses)
                mp.setattr(
                    scheduler,
                    "check_confirmed_lineups",
                    lambda pks: {pk: {"away", "home"} for pk in pks if pk == 101},
                )
                scheduler.run_day(DAY, world.config)
                st = scheduler.load_state(DAY, world.picks)
                alerts = world.sends("health-dm")
                ok = (
                    status_calls == 1
                    and st.fallback_refreshes[0]["action"] == "defer"
                    and sum("[postponed_pick]" in a for a in alerts) == 1
                    and sum("[late_delivery]" in a for a in alerts) == 0
                )
                return {"ok": ok, "status_calls": status_calls, "action": st.fallback_refreshes[0]["action"]}

        if case in {"r3_double_fail_cutoff_dm", "r3_double_fail_cutoff_public", "r3_double_fail_midnight_dm"}:
            is_midnight = "midnight" in case
            mode = "public" if "public" in case else "dm"
            day = "2027-06-18" if is_midnight else DAY
            start = at(23, 20, day=day) if is_midnight else at(18, 25, day=day)
            with offline.offline(start) as (world, mp):
                world.config["scheduler"]["pick_delivery"] = mode
                if is_midnight:
                    world.schedule[day] = [offline.game(101, start + timedelta(minutes=70))]
                    world.config["scheduler"].update(fallback_deadline_min=40, fallback_deadline_min_morning=40)
                    mp.setattr(strategy, "_load_mdp", lambda: None)
                    mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": start, "game_pks": [101]}])
                else:
                    world.schedule[day][1]["gameDate"] = at(20, 10, day=day).isoformat()
                    mp.setattr(
                        scheduler,
                        "plan_run_times",
                        lambda *a, **kw: [
                            {"time_et": at(18, 25, day=day), "game_pks": [101]},
                            {"time_et": at(19, 15, day=day), "game_pks": [102]},
                        ],
                    )
                world.dated(0)
                real_write = picks.atomic_write_text
                marker_set = False
                clear_attempts = 0

                def forecast():
                    if world.now == start:
                        world.advance(2399)
                        n = len(world.schedule[day])
                        return world.frame((0.99, 0.95)[:n], (True, False)[:n])
                    return world.frame((0.40, 0.95), (True, False)).iloc[[1, 0]]

                def failing_write(path, text):
                    nonlocal marker_set, clear_attempts
                    if Path(path) == world.picks / f"{day}.json":
                        attempted = json.loads(text)["delivery_attempted"]
                        if marker_set and not attempted:
                            clear_attempts += 1
                            raise OSError(f"persistent EIO #{clear_attempts}")
                        real_write(path, text)
                        if attempted and not marker_set:
                            marker_set = True
                            world.advance(2)
                    else:
                        real_write(path, text)

                world.predictions = forecast
                mp.setattr(picks, "atomic_write_text", failing_write)
                raised = None
                try:
                    scheduler.run_day(day, world.config)
                except OSError as exc:
                    raised = str(exc)
                after_crash = picks.load_pick(day, world.picks)
                alerts_before_restart = list(world.sends("health-dm"))

                mp.setattr(picks, "atomic_write_text", real_write)
                scheduler.run_day(day, world.config)
                rec = daily_decision.load_decision(day, world.picks)
                alerts_after_restart = list(world.sends("health-dm"))
                ok = (
                    clear_attempts == 2
                    and raised == "persistent EIO #2"
                    and after_crash.delivery_attempted is True
                    and len(alerts_before_restart) == 0
                    and not world.sends("dm")
                    and not world.sends("post")
                    and rec is not None
                    and rec["delivery_status"] == "locked_unconfirmed"
                    and sum("OUTCOME UNKNOWN" in a for a in alerts_after_restart) == 1
                )
                return {
                    "ok": ok,
                    "clear_attempts": clear_attempts,
                    "raised": raised,
                    "marker_on_disk_after_crash": after_crash.delivery_attempted,
                    "alerts_before_restart": len(alerts_before_restart),
                    "restart_delivery_status": rec and rec["delivery_status"],
                }

        if case == "r3_interrupts_in_marker_clear":
            outcomes = {}
            for exc_cls in (KeyboardInterrupt, SystemExit):
                for stage in ("first_clear_before_replace", "retry_clear_before_replace", "clear_after_replace_in_audit"):
                    with offline.offline(at(18, 25)) as (world, mp):
                        world.schedule[DAY] = world.schedule[DAY][:1]
                        world.automatic(0, age_hours=15)
                        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": at(18, 25), "game_pks": [101]}])
                        real_write = picks.atomic_write_text
                        real_append = picks.append_lineup_evolution
                        marker_set = False
                        first_failed = False
                        cleared_on_disk = False

                        def forecast():
                            if world.now == at(18, 25):
                                world.advance(2399)
                                return world.frame((0.99,), (True,))
                            return world.frame((0.99,), (True,))

                        def hooked_write(path, text):
                            nonlocal marker_set, first_failed, cleared_on_disk
                            if Path(path) == world.picks / f"{DAY}.json":
                                attempted = json.loads(text)["delivery_attempted"]
                                if marker_set and not attempted:
                                    if stage == "first_clear_before_replace":
                                        raise exc_cls("interrupt on 1st clear")
                                    if stage == "retry_clear_before_replace":
                                        if not first_failed:
                                            first_failed = True
                                            raise OSError("transient EIO before interrupt")
                                        raise exc_cls("interrupt on retry clear")
                                    real_write(path, text)
                                    cleared_on_disk = True
                                    return
                                real_write(path, text)
                                if attempted and not marker_set:
                                    marker_set = True
                                    world.advance(2)
                            else:
                                real_write(path, text)

                        def hooked_append(daily_obj, directory):
                            real_append(daily_obj, directory)
                            if stage == "clear_after_replace_in_audit" and cleared_on_disk:
                                raise exc_cls("interrupt in append_lineup_evolution after clear")

                        world.predictions = forecast
                        mp.setattr(picks, "atomic_write_text", hooked_write)
                        mp.setattr(picks, "append_lineup_evolution", hooked_append)
                        caught = False
                        try:
                            scheduler.run_day(DAY, world.config)
                        except exc_cls:
                            caught = True
                        saved = picks.load_pick(DAY, world.picks)
                        expect_attempted = stage != "clear_after_replace_in_audit"
                        ok = (
                            caught
                            and saved.delivery_attempted is expect_attempted
                            and (saved.automatic_reading is None if not expect_attempted else saved.automatic_reading is not None)
                        )
                        outcomes[f"{exc_cls.__name__}:{stage}"] = {
                            "ok": ok,
                            "delivery_attempted": saved.delivery_attempted,
                            "prior_reading_restored": saved.automatic_reading is None,
                        }
                        assert ok, outcomes[f"{exc_cls.__name__}:{stage}"]
            return {"ok": True, "cases": len(outcomes), "details": outcomes}

        if case == "r1_receipt_dedup_matrix":
            from bts.health import alert as alert_mod

            results = {}
            for sub in ("reuse_then_same_day_and_next_day_restart", "c2_restart_then_reuse", "failed_reuse_alert_retried_on_restart"):
                with offline.offline(at(18, 25)) as (world, mp):
                    world.schedule[DAY] = world.schedule[DAY][:1]
                    world.dated(0)
                    mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": at(18, 25), "game_pks": [101]}])
                    real_check = scheduler.run_single_check
                    real_alert_send = alert_mod.send_dm
                    fail_first_alert = sub == "failed_reuse_alert_retried_on_restart"
                    failed_once = False

                    def custom_alert_send(recipient, body):
                        nonlocal failed_once
                        if fail_first_alert and not failed_once and "OUTCOME UNKNOWN" in body:
                            failed_once = True
                            return ""
                        return real_alert_send(recipient, body)

                    mp.setattr(alert_mod, "send_dm", custom_alert_send)

                    if sub == "c2_restart_then_reuse":
                        scheduler._alert_delivery_outcome_unknown(world.config, world.picks, DAY)
                        assert sum("OUTCOME UNKNOWN" in a for a in world.sends("health-dm")) == 1

                    def forecast():
                        if world.now == at(18, 25):
                            world.advance(35 * 60)
                            return world.frame((0.99,), (True,))
                        return None

                    def publish_locked_unconfirmed(*a, **kw):
                        res = real_check(*a, **kw)
                        daily_obj = res["pick_result"].daily
                        daily_decision.write_decision(
                            DAY,
                            world.picks,
                            action="single",
                            source="mdp",
                            primary=scheduler._row_from_daily(daily_obj.pick),
                            delivery_status="locked_unconfirmed",
                            scoreable=True,
                        )
                        return res

                    world.predictions = forecast
                    mp.setattr(scheduler, "run_single_check", publish_locked_unconfirmed)
                    scheduler.run_day(DAY, world.config)
                    receipt_path = world.picks.parent / "health_state" / "delivery_outcome_unknown" / f"{DAY}.json"
                    receipt_1 = json.loads(receipt_path.read_text())

                    mp.setattr(scheduler, "run_single_check", real_check)
                    cur = picks.load_pick(DAY, world.picks)
                    cur.delivery_attempted = True
                    picks.save_pick(cur, world.picks)
                    world.now = at(19, 2)
                    scheduler.run_day(DAY, world.config)
                    if sub == "reuse_then_same_day_and_next_day_restart":
                        world.now = at(0, 15, day="2027-03-25")
                        scheduler._alert_delivery_outcome_unknown(world.config, world.picks, DAY)

                    unknown_alerts = [a for a in world.sends("health-dm") if "OUTCOME UNKNOWN" in a]
                    receipt_final = json.loads(receipt_path.read_text())
                    ok = len(unknown_alerts) == 1 and receipt_final["status"] == "sent" and receipt_final["sent_day"] == DAY
                    results[sub] = {
                        "ok": ok,
                        "unknown_alert_count": len(unknown_alerts),
                        "receipt_1_status": receipt_1["status"],
                        "receipt_final_status": receipt_final["status"],
                    }
                    assert ok, results[sub]
            return {"ok": True, "details": results}

        if case == "r1_attempt_with_all_untrusted_decision_shapes":
            shapes = ["malformed_json", "invalid_action", "wrong_date", "dangling", "fifo", "unreadable", "directory"]
            details = {}
            real_open = os.open
            for shp in shapes:
                with offline.offline(at(18, 25)) as (world, mp):
                    world.dated(0)
                    world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                    mp.setattr(
                        scheduler,
                        "plan_run_times",
                        lambda *a, **kw: [
                            {"time_et": at(18, 25), "game_pks": [101]},
                            {"time_et": at(18, 50), "game_pks": [102]},
                        ],
                    )
                    mp.setattr(scheduler, "run_result_polling", lambda *_a, **_kw: "polled")
                    dec_path = daily_decision.decision_path(DAY, world.picks)
                    first_done = False
                    mutated = False
                    real_check = scheduler.run_single_check

                    def forecast():
                        if world.now == at(18, 25):
                            world.advance(22 * 60)
                            return world.frame((0.99, 0.70), (True, True))
                        return None

                    def check_wrap(*a, **kw):
                        nonlocal first_done
                        res = real_check(*a, **kw)
                        first_done = True
                        return res

                    def conf_wrap(pks):
                        nonlocal mutated
                        if first_done and pks == [102] and not mutated:
                            mutated = True
                            cur = picks.load_pick(DAY, world.picks)
                            cur.delivery_attempted = True
                            picks.save_pick(cur, world.picks)
                            dec_path.parent.mkdir(parents=True, exist_ok=True)
                            if shp == "malformed_json":
                                dec_path.write_text("{")
                            elif shp == "invalid_action":
                                dec_path.write_text(
                                    json.dumps(
                                        {"schema_version": daily_decision.DECISION_SCHEMA, "date": DAY, "action": [], "scoreable": False}
                                    )
                                )
                            elif shp == "wrong_date":
                                dec_path.write_text(
                                    json.dumps(
                                        {"schema_version": daily_decision.DECISION_SCHEMA, "date": "2027-03-23", "action": "single", "scoreable": True}
                                    )
                                )
                            elif shp == "dangling":
                                dec_path.symlink_to(dec_path.with_name("missing_target.json"))
                            elif shp == "fifo":
                                os.mkfifo(dec_path)
                            elif shp == "unreadable":
                                dec_path.write_text("{}")
                            elif shp == "directory":
                                dec_path.mkdir()
                        return {pk: {"away", "home"} for pk in pks if pk == 101}

                    def open_hook(name, flags, *a, **kw):
                        if shp == "unreadable" and mutated and Path(name) == dec_path and (flags & os.O_NONBLOCK):
                            raise PermissionError("denied")
                        return real_open(name, flags, *a, **kw)

                    world.predictions = forecast
                    mp.setattr(scheduler, "run_single_check", check_wrap)
                    mp.setattr(scheduler, "check_confirmed_lineups", conf_wrap)
                    mp.setattr(os, "open", open_hook)

                    def timeout_handler(*_):
                        raise TimeoutError("blocked on special file")

                    prev = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(3)
                    err = None
                    try:
                        scheduler.run_day(DAY, world.config)
                    except Exception as exc:
                        err = type(exc).__name__
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, prev)

                    unknown_alerts = sum("OUTCOME UNKNOWN" in a for a in world.sends("health-dm"))
                    st = scheduler.load_state(DAY, world.picks)
                    if shp == "directory":
                        ok = err == "IsADirectoryError" and unknown_alerts == 1 and st.pick_locked
                    else:
                        rec = daily_decision.load_decision(DAY, world.picks)
                        ok = (
                            err is None
                            and unknown_alerts == 1
                            and st.pick_locked
                            and rec is not None
                            and rec["delivery_status"] == "locked_unconfirmed"
                        )
                    details[shp] = {"ok": ok, "error": err, "unknown_alerts": unknown_alerts, "locked": st.pick_locked}
                    assert ok, details[shp]
            return {"ok": True, "details": details}

        if case == "r5_shadow_all_four_breaks_and_optext":
            subcases = [
                "pre_poll_scoreable_match",
                "pre_poll_scoreable_mismatch",
                "pre_poll_attempted",
                "post_poll_scoreable",
                "post_poll_attempted",
                "post_poll_delivered",
            ]
            details = {}
            for sub in subcases:
                with offline.offline(at(18, 25)) as (world, mp):
                    world.dated(0)
                    world.config["scheduler"]["shadow_model"] = True
                    if sub.startswith("post_poll"):
                        world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
                        mp.setattr(
                            scheduler,
                            "plan_run_times",
                            lambda *a, **kw: [
                                {"time_et": at(18, 25), "game_pks": [101]},
                                {"time_et": at(18, 50), "game_pks": [102]},
                            ],
                        )
                    else:
                        world.schedule[DAY] = world.schedule[DAY][:1]
                        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": at(18, 25), "game_pks": [101]}])
                    mp.setattr(scheduler, "run_result_polling", lambda *_a, **_kw: "polled")
                    shadow_calls = []
                    mp.setattr(
                        scheduler,
                        "_trigger_shadow_prediction_on_lock",
                        lambda cfg, d, b: shadow_calls.append((d, b)),
                    )
                    real_check = scheduler.run_single_check
                    real_save = picks.save_pick
                    first_done = False
                    stderr = io.StringIO()

                    def forecast():
                        if world.now == at(18, 25):
                            world.advance(22 * 60 if sub.startswith("post_poll") else 35 * 60)
                            n = len(world.schedule[DAY])
                            return world.frame((0.99, 0.70)[:n], (True, True)[:n])
                        return None

                    def save_pre_attempted(daily_obj, directory):
                        if sub == "pre_poll_attempted" and not first_done:
                            daily_obj.delivery_attempted = True
                        return real_save(daily_obj, directory)

                    def check_wrap(*a, **kw):
                        nonlocal first_done
                        res = real_check(*a, **kw)
                        first_done = True
                        if sub.startswith("pre_poll_scoreable"):
                            prim = (
                                scheduler._row_from_daily(res["pick_result"].daily.pick)
                                if sub.endswith("match")
                                else {"batter_id": 999, "batter_name": "External Batter 999", "team": "NYY", "game_pk": 999, "p_game_hit": 0.88}
                            )
                            daily_decision.write_decision(
                                DAY,
                                world.picks,
                                action="single",
                                source="mdp",
                                primary=prim,
                                delivery_status="locked_unconfirmed",
                                scoreable=True,
                            )
                        return res

                    def conf_wrap(pks):
                        if first_done and pks == [102]:
                            cur = picks.load_pick(DAY, world.picks)
                            if sub == "post_poll_scoreable":
                                daily_decision.write_decision(
                                    DAY,
                                    world.picks,
                                    action="single",
                                    source="mdp",
                                    primary=scheduler._row_from_daily(cur.pick),
                                    delivery_status="locked_unconfirmed",
                                    scoreable=True,
                                )
                            elif sub == "post_poll_attempted":
                                cur.delivery_attempted = True
                                picks.save_pick(cur, world.picks)
                            elif sub == "post_poll_delivered":
                                cur.delivery_attempted = True
                                cur.notification_sent = True
                                cur.notification_id = "ext-msg-1"
                                cur.delivered_at = world.now.isoformat()
                                picks.save_pick(cur, world.picks)
                        return {pk: {"away", "home"} for pk in pks if pk == 101}

                    world.predictions = forecast
                    mp.setattr(picks, "save_pick", save_pre_attempted)
                    mp.setattr(scheduler, "run_single_check", check_wrap)
                    if sub.startswith("post_poll"):
                        mp.setattr(scheduler, "check_confirmed_lineups", conf_wrap)
                    with redirect_stderr(stderr):
                        scheduler.run_day(DAY, world.config)

                    err_txt = stderr.getvalue()
                    ok = (
                        shadow_calls == [(DAY, "Synthetic Batter 101")]
                        and err_txt.count("Pick locked. Stopping lineup checks.") == 1
                        and "FALLBACK: reusing this check's fresh cascade decision" not in err_txt
                    )
                    details[sub] = {
                        "ok": ok,
                        "shadow_calls": shadow_calls,
                        "has_unknown_stderr": "DELIVERY OUTCOME UNKNOWN" in err_txt,
                        "unknown_dm_count": sum("OUTCOME UNKNOWN" in a for a in world.sends("health-dm")),
                    }
                    assert ok, details[sub]
            return {"ok": True, "details": details}

    raise ValueError(case)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        res = worker_main(sys.argv[2])
        print("R2_P2C_JSON " + json.dumps(res), flush=True)
        return

    cases = [
        "r2_flip_void_consistent",
        "r2_flip_void_transient_second_read",
        "r2_poll_crosses_cutoff_keeps_defer_void_order",
        "r3_double_fail_cutoff_dm",
        "r3_double_fail_cutoff_public",
        "r3_double_fail_midnight_dm",
        "r3_interrupts_in_marker_clear",
        "r1_receipt_dedup_matrix",
        "r1_attempt_with_all_untrusted_decision_shapes",
        "r5_shadow_all_four_breaks_and_optext",
    ]
    out = {}
    for c in cases:
        res = run_case(c)
        out[c] = res
        print(f"[r2_probe_2c] {c}: ok={res.get('ok')}", flush=True)
        if not res.get("ok"):
            raise RuntimeError(f"Case {c} failed: {res}")

    out_path = R2_DIR / "r2_probe_2_results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
