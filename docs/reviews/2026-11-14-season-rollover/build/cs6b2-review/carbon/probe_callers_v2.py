"""Adversarial probe for CS6b-2 item 4 (using an identical isolated workspace path
across missing, array, and nested variants so embedded --picks-dir paths match
byte-for-byte without string normalization).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TZ"] = "America/New_York"
sys.dont_write_bytecode = True
if hasattr(time, "tzset"):
    time.tzset()

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
SCRATCH_BASE_SRC = REVIEW_DIR / "scratch_base" / "src"

_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}
_real_getaddrinfo = socket.getaddrinfo
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _install_network_guard() -> None:
    def guarded_getaddrinfo(host, *a, **kw):
        if host not in _ALLOWED_HOSTS:
            raise AssertionError(f"Forbidden DNS lookup: {host!r}")
        return _real_getaddrinfo(host, *a, **kw)

    def guarded_connect(self, addr):
        if isinstance(addr, tuple) and addr and addr[0] not in _ALLOWED_HOSTS:
            raise AssertionError(f"Forbidden connect: {addr!r}")
        return _real_connect(self, addr)

    def guarded_connect_ex(self, addr):
        if isinstance(addr, tuple) and addr and addr[0] not in _ALLOWED_HOSTS:
            raise AssertionError(f"Forbidden connect_ex: {addr!r}")
        return _real_connect_ex(self, addr)

    socket.getaddrinfo = guarded_getaddrinfo
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex


ALL_CASES = [
    "check",
    "state_read",
    "saved_pick",
    "in_loop",
    "final",
    "missed_direct",
    "missed_run_day",
    "late",
    "void",
    "unconfirmed",
    "classified_unconfirmed",
    "eod",
    "morning_critical",
    "morning_stale_warn",
    "delivery_stale_warn",
]


def run_worker(mode: str, shared_parent: Path) -> dict:
    src_dir = SCRATCH_BASE_SRC if mode == "base" else (REPO_ROOT / "src")
    sys.path = [str(src_dir), str(REPO_ROOT)] + [
        p for p in sys.path if p not in {str(REPO_ROOT / "src"), str(SCRATCH_BASE_SRC), str(REPO_ROOT)}
    ]
    _install_network_guard()

    import pandas as pd
    from zoneinfo import ZoneInfo

    from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.daily_decision import load_decision
    from bts.health import alert, runner
    from bts.model import predict

    assert Path(alert.__file__).resolve() == (src_dir / "bts" / "health" / "alert.py").resolve()
    assert sys.getrecursionlimit() == 1000

    DAY = "2027-07-15"
    NOW = datetime.fromisoformat(f"{DAY}T18:35:00-04:00")
    START = NOW.replace(hour=10, minute=0)

    def set_dated(directory: Path, now_dt: datetime = NOW, streak: int = 7, best: int = 9, saver: str = "used"):
        day = now_dt.astimezone(ZoneInfo("America/New_York")).date()
        target = directory / "account_state" / "contest_streak.manual.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v3",
            "season": day.year,
            "effective_date": day.isoformat(),
            "active_streak": streak,
            "best_streak": best,
            "saver_state": saver,
            "recorded_at": now_dt.isoformat(),
            "expires_at": f"{day}T23:59:59.999999-04:00",
            "source": "manual_cli",
        }))

    def seed_stale_automatic(directory: Path, now_dt: datetime, age_hours: int = 20, streak: int = 30):
        year = now_dt.year
        recorded = (now_dt - timedelta(hours=age_hours)).isoformat()
        acc = directory / "account_state"
        acc.mkdir(parents=True, exist_ok=True)
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2",
            "season": year,
            "activation_id": "a" * 32,
            "active_streak": streak,
            "best_streak": streak,
            "source": "manual_cli",
            "source_date": now_dt.date().isoformat(),
            "recorded_at": recorded,
        }))
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": year,
            "status": "automatic",
            "activation_id": "a" * 32,
            "switched_at": now_dt.isoformat(),
            "revoked_at": None,
            "note": None,
        }))

    def mk_daily_pick():
        return picks.DailyPick(
            DAY, NOW.isoformat(),
            picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                       0.82, [], False, 101, f"{DAY}T23:10:00Z"),
            None, None,
        )

    def fresh_selection(daily):
        policy = strategy.PolicyDecision(
            "single", "single", "heuristic", "reach57", 7, 80, None, "missing",
            None, None, None, None, None,
        )
        return strategy.SelectionResult(
            strategy.PickResult(daily), "single", "heuristic", None, None, None,
            streak=7, decision=policy,
        )

    lines, first_line = inspect.getsourcelines(scheduler.run_day)
    fallback_sites = [
        first_line + idx for idx, line in enumerate(lines)
        if "_alert_contest_state_failure(config, e)" in line
    ]
    assert len(fallback_sites) == 2
    in_loop_site, final_site = fallback_sites

    nested_bytes = b"[" * 20_000 + b"0" + b"]" * 20_000
    status_variants = {
        "missing": None,
        "array": b"[]",
        "nested": nested_bytes,
    }

    results: dict[str, dict[str, dict]] = {}
    orig_cwd = os.getcwd()
    tmp_path = shared_parent / "workspace"

    for case in ALL_CASES:
        results[case] = {}
        for variant, seed_bytes in status_variants.items():
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            tmp_path.mkdir(parents=True)
            os.chdir(tmp_path)
            strategy._mdp_cache = None

            now_box = [START if case in ("morning_critical", "morning_stale_warn") else NOW]

            class DecisionClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

            directory = tmp_path / "picks"
            statuses = {
                pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                for pk in (101, 102)
            }
            predictions = pd.DataFrame([
                {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                 "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                 "p_hit_pa": 0.24, "p_game_hit": 0.82, "flags": "", "projected_lineup": False,
                 "game_pk": 101, "game_time": f"{DAY}T23:10:00Z"},
                {"batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
                 "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD",
                 "p_hit_pa": 0.23, "p_game_hit": 0.81, "flags": "", "projected_lineup": False,
                 "game_pk": 102, "game_time": f"{DAY}T23:40:00Z"},
            ])
            config = {
                "orchestrator": {
                    "picks_dir": str(directory),
                    "heartbeat_path": str(tmp_path / "heartbeat"),
                    "models_dir": str(tmp_path / "models"),
                },
                "bluesky": {"dm_recipient": "offline.test"},
                "scheduler": {
                    "contest_state_required": True,
                    "pick_delivery": "dm",
                    "live_forward_capture_on_lock": False,
                },
                "health_checks": {"enabled": False},
                "tiers": [],
            }

            health_dir = directory.parent / "health_state"
            if case == "morning_critical":
                lane_file = health_dir / "contest_morning_check_dm_delivery_status.json"
            elif case in ("morning_stale_warn", "delivery_stale_warn"):
                lane_file = health_dir / "contest_stale_reading_dm_delivery_status.json"
            else:
                lane_file = health_dir / "health_dm_delivery_status.json"

            if seed_bytes is not None:
                lane_file.parent.mkdir(parents=True, exist_ok=True)
                lane_file.write_bytes(seed_bytes)

            patches: list[tuple[object, str, object]] = []

            def set_attr(obj, name, val):
                patches.append((obj, name, getattr(obj, name)))
                setattr(obj, name, val)

            try:
                set_attr(contest_state, "datetime", DecisionClock)
                set_attr(daily_decision, "datetime", DecisionClock)
                set_attr(picks, "datetime", DecisionClock)
                set_attr(strategy, "datetime", DecisionClock)
                set_attr(cli, "datetime", DecisionClock)
                set_attr(scheduler, "datetime", DecisionClock)
                set_attr(alert, "datetime", DecisionClock)
                set_attr(scheduler, "_now_et", lambda: now_box[0])
                set_attr(cli, "_now_et", lambda: now_box[0])
                set_attr(predict, "run_pipeline", Mock(return_value=predictions))
                set_attr(orchestrator, "run_cascade", Mock(return_value=(predictions, "offline")))
                set_attr(picks, "get_game_statuses_detailed", lambda *a: statuses)
                set_attr(strategy, "get_game_statuses_detailed", lambda *a: statuses)
                set_attr(picks, "get_game_statuses", lambda *a: {pk: "P" for pk in statuses})

                games = [{
                    "gamePk": pk, "gameDate": f"{DAY}T{gtime}:00Z",
                    "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                    "teams": {"away": {"team": {"name": f"Away {pk}"}},
                              "home": {"team": {"name": f"Home {pk}"}}},
                } for pk, gtime in ((101, "23:10"), (102, "23:40"))]
                set_attr(scheduler, "fetch_schedule", Mock(side_effect=lambda d: games if d == DAY else []))
                set_attr(scheduler, "check_confirmed_lineups", lambda *a: {})
                idle_mock = Mock()
                set_attr(scheduler, "_idle_until_next_wakeup", idle_mock)
                set_attr(scheduler, "_watchdog_ping_sleep", lambda *a: None)
                set_attr(scheduler, "run_result_polling", Mock(return_value="final"))
                set_attr(scheduler, "notify_ready", Mock())
                set_attr(scheduler, "notify_watchdog", Mock())
                set_attr(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext())
                set_attr(scheduler, "retry_urlopen", Mock(side_effect=AssertionError("MLB net")))

                pick_dm_mock = Mock(return_value="pick-dm")
                health_dm_mock = Mock(return_value="health-dm")
                post_mock = Mock(return_value="at://post")
                set_attr(dm, "send_dm", pick_dm_mock)
                set_attr(alert, "send_dm", health_dm_mock)
                set_attr(posting, "post_to_bluesky", post_mock)
                set_attr(posting, "should_post_now", Mock(return_value=True))

                real_dispatch = alert.dispatch_dm_for_health_alerts
                fallback_hits: list[int] = []
                dispatch_calls: list[tuple[str, int]] = []

                def spy_dispatch(*args, **kwargs):
                    frame = inspect.currentframe().f_back
                    dispatch_calls.append((frame.f_code.co_name, frame.f_lineno))
                    if frame.f_code.co_name == "_alert_contest_state_failure":
                        parent = frame.f_back
                        if parent.f_code.co_name == "run_day":
                            fallback_hits.append(parent.f_lineno)
                    return real_dispatch(*args, **kwargs)

                set_attr(alert, "dispatch_dm_for_health_alerts", spy_dispatch)
                set_attr(runner, "dispatch_dm_for_health_alerts", spy_dispatch)

                state = scheduler.SchedulerState(
                    date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                    confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                    pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                daily = mk_daily_pick()
                raised_exc = None
                ret_val = None

                def alarm_handler(*_):
                    raise TimeoutError(f"Caller {case}/{variant} timed out")

                old_sig = signal.signal(signal.SIGALRM, alarm_handler)
                signal.setitimer(signal.ITIMER_REAL, 5.0)
                try:
                    if case == "check":
                        r = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
                        ret_val = r["pick_result"] is None
                    elif case == "state_read":
                        picks.save_pick(daily, directory)
                        ret_val = scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, "state_read")
                    elif case == "saved_pick":
                        set_dated(directory)
                        picks.save_pick(daily, directory)
                        ret_val = scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, "saved_pick")
                    elif case == "in_loop":
                        set_dated(directory)
                        picks.save_pick(daily, directory)
                        set_attr(
                            scheduler, "_lock_decision_from_predictions",
                            lambda *a, **kw: scheduler.LockDecision(False, None, False),
                        )
                        now_box[0] = NOW - timedelta(minutes=10)

                        def predict_in_loop(*_):
                            if orchestrator.run_cascade.call_count > 1:
                                (directory / "account_state" / "contest_streak.manual.json").unlink(missing_ok=True)
                            return predictions, "offline"

                        orchestrator.run_cascade.side_effect = predict_in_loop
                        scheduler.run_day(DAY, config)
                        ret_val = idle_mock.call_count
                    elif case == "final":
                        picks.save_pick(daily, directory)
                        orchestrator.run_cascade.side_effect = [(None, None), (predictions, "offline")]
                        scheduler.run_day(DAY, config)
                        ret_val = idle_mock.call_count
                    elif case == "missed_direct":
                        picks.save_pick(daily, directory)
                        scheduler._maybe_alert_missed_pick(config, DAY, directory, 10, None, state)
                        ret_val = True
                    elif case == "missed_run_day":
                        config["scheduler"]["contest_state_required"] = False
                        picks.save_pick(daily, directory)
                        orchestrator.run_cascade.return_value = (None, None)
                        pick_dm_mock.side_effect = RuntimeError("pick transport down")
                        scheduler.run_day(DAY, config)
                        ret_val = idle_mock.call_count
                    elif case == "late":
                        set_dated(directory)
                        now_box[0] = NOW + timedelta(hours=1)
                        picks.save_pick(daily, directory)
                        ret_val = scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, "late")
                    elif case == "void":
                        set_dated(directory)
                        statuses[101] = {"abstract": "P", "detailed": "Cancelled", "code": "C"}
                        set_attr(scheduler, "_delivery_game_statuses", lambda _: statuses)
                        picks.save_pick(daily, directory)
                        ret_val = scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, "void")
                    elif case == "unconfirmed":
                        set_dated(directory)
                        daily.delivery_attempted = True
                        picks.save_pick(daily, directory)
                        ret_val = scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, "unconfirmed")
                    elif case == "classified_unconfirmed":
                        daily.delivery_attempted = True
                        picks.save_pick(daily, directory)
                        scheduler.run_day(DAY, config)
                        ret_val = idle_mock.call_count
                    elif case == "eod":
                        config["health_checks"]["enabled"] = True
                        orchestrator.run_cascade.return_value = (None, None)
                        set_attr(runner, "get_self_pid", lambda: None)
                        set_attr(runner, "read_systemd_nrestarts", lambda: None)
                        set_attr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
                        scheduler.run_day(DAY, config)
                        ret_val = idle_mock.call_count
                    elif case == "morning_critical":
                        set_attr(scheduler, "time", Mock(monotonic=lambda: 0))

                        def advance_and_fix_state(secs):
                            now_box[0] += timedelta(seconds=secs)
                            if not (directory / "account_state" / "contest_streak.manual.json").exists():
                                set_dated(directory, now_dt=now_box[0], streak=30, best=30)

                        set_attr(scheduler, "_watchdog_ping_sleep", advance_and_fix_state)
                        high_preds = (predictions.assign(p_game_hit=[0.99, 0.81]), "offline")
                        orchestrator.run_cascade.return_value = high_preds
                        scheduler.run_day(DAY, config)
                        ret_val = idle_mock.call_count
                    elif case == "morning_stale_warn":
                        set_attr(scheduler, "time", Mock(monotonic=lambda: 0))
                        seed_stale_automatic(directory, now_dt=START, age_hours=20, streak=30)

                        def advance(secs):
                            now_box[0] += timedelta(seconds=secs)

                        set_attr(scheduler, "_watchdog_ping_sleep", advance)
                        high_preds = (predictions.assign(p_game_hit=[0.99, 0.81]), "offline")
                        orchestrator.run_cascade.return_value = high_preds
                        scheduler.run_day(DAY, config)
                        ret_val = idle_mock.call_count
                    elif case == "delivery_stale_warn":
                        seed_stale_automatic(directory, now_dt=NOW, age_hours=20, streak=7)
                        picks.save_pick(daily, directory)
                        ret_val = scheduler._deliver_and_lock_pick(
                            daily, config, directory, state, DAY, "fixture", selection=fresh_selection(daily)
                        )
                except Exception as exc:
                    raised_exc = f"{type(exc).__name__}: {exc}"
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_sig)

                disk_state = scheduler.load_state(DAY, directory)
                state_obj = disk_state if disk_state is not None else state
                state_dict = asdict(state_obj)
                if state_dict.get("delivery_refusals"):
                    norm_refs = []
                    for ref in state_dict["delivery_refusals"]:
                        r_copy = dict(ref)
                        arch_name = r_copy.get("archive")
                        r_copy["archive_exists"] = bool(arch_name and (directory / DAY / arch_name).is_file())
                        norm_refs.append(r_copy)
                    state_dict["delivery_refusals"] = norm_refs

                dec = load_decision(DAY, directory)
                loaded_pick = picks.load_pick(DAY, directory)
                pick_summary = None
                if loaded_pick is not None:
                    pick_summary = {
                        "batter_id": loaded_pick.pick.batter_id,
                        "delivery_attempted": loaded_pick.delivery_attempted,
                        "notification_sent": loaded_pick.notification_sent,
                        "bluesky_posted": loaded_pick.bluesky_posted,
                    }

                lane_status = None
                if lane_file.exists():
                    raw_lane = lane_file.read_bytes()
                    try:
                        parsed_lane = json.loads(raw_lane)
                        lane_status = (
                            {"status": parsed_lane.get("status"), "sent_sources": parsed_lane.get("sent_sources")}
                            if isinstance(parsed_lane, dict) else f"non_dict:{type(parsed_lane).__name__}:len={len(raw_lane)}"
                        )
                    except RecursionError:
                        lane_status = f"recursion_error:len={len(raw_lane)}"

                results[case][variant] = {
                    "raised": raised_exc,
                    "ret_val": ret_val,
                    "health_dms": [c.args for c in health_dm_mock.call_args_list],
                    "pick_dms": [c.args for c in pick_dm_mock.call_args_list],
                    "posts": [c.args for c in post_mock.call_args_list],
                    "fallback_hits": fallback_hits,
                    "in_loop_site": in_loop_site,
                    "final_site": final_site,
                    "lane_status": lane_status,
                    "state": state_dict,
                    "decision": dec,
                    "pick": pick_summary,
                }
            finally:
                os.chdir(orig_cwd)
                for obj, name, orig in reversed(patches):
                    setattr(obj, name, orig)
                if tmp_path.exists():
                    shutil.rmtree(tmp_path)

    return results


def verify_and_summarize(base_res: dict, branch_res: dict) -> dict:
    unwrapped = {
        "check", "state_read", "in_loop", "final",
        "missed_direct", "missed_run_day", "late", "void", "unconfirmed",
    }
    wrapped = {
        "saved_pick", "eod", "morning_critical", "morning_stale_warn", "delivery_stale_warn",
    }

    summary = {}
    for case in ALL_CASES:
        b_arr = base_res[case]["array"]
        b_nest = base_res[case]["nested"]

        h_miss = branch_res[case]["missing"]
        h_arr = branch_res[case]["array"]
        h_nest = branch_res[case]["nested"]

        if case == "in_loop":
            assert h_miss["in_loop_site"] in h_miss["fallback_hits"]
            assert h_arr["in_loop_site"] in h_arr["fallback_hits"]
            assert h_nest["in_loop_site"] in h_nest["fallback_hits"]
        elif case == "final":
            assert h_miss["final_site"] in h_miss["fallback_hits"]
            assert h_miss["in_loop_site"] not in h_miss["fallback_hits"]
            assert h_arr["final_site"] in h_arr["fallback_hits"]
            assert h_arr["in_loop_site"] not in h_arr["fallback_hits"]
            assert h_nest["final_site"] in h_nest["fallback_hits"]
            assert h_nest["in_loop_site"] not in h_nest["fallback_hits"]

        if case in unwrapped:
            assert b_arr["raised"] and b_arr["raised"].startswith("AttributeError:"), f"Base {case}/array: {b_arr}"
            assert b_nest["raised"] and b_nest["raised"].startswith("RecursionError:"), f"Base {case}/nested: {b_nest}"
            assert len(b_arr["health_dms"]) == 0 and len(b_nest["health_dms"]) == 0
        elif case in wrapped:
            assert b_arr["raised"] is None and b_nest["raised"] is None, f"Base wrapped {case} raised: {b_arr} / {b_nest}"
            assert len(b_arr["health_dms"]) == 0 and len(b_nest["health_dms"]) == 0, (
                f"Base wrapped {case} unexpectedly sent health DM: {b_arr['health_dms']}"
            )
            assert str(b_arr["lane_status"]).startswith("non_dict:list")
            assert str(b_nest["lane_status"]).startswith("recursion_error:")
        elif case == "classified_unconfirmed":
            assert b_arr["raised"] is None and b_nest["raised"] is None
            assert len(h_arr["health_dms"]) == 0 and h_arr["lane_status"] == "non_dict:list:len=2"

        assert h_miss["raised"] is None and h_arr["raised"] is None and h_nest["raised"] is None, (
            f"Branch {case} raised: {h_miss['raised']} / {h_arr['raised']} / {h_nest['raised']}"
        )
        if case != "classified_unconfirmed":
            assert len(h_miss["health_dms"]) >= 1, f"Branch {case} sent 0 health DMs"
            assert h_arr["lane_status"]["status"] == "sent"
            assert h_nest["lane_status"]["status"] == "sent"
            assert h_arr["lane_status"] == h_miss["lane_status"]
            assert h_nest["lane_status"] == h_miss["lane_status"]

        for field in ("ret_val", "health_dms", "pick_dms", "posts", "state", "decision", "pick"):
            assert h_arr[field] == h_miss[field], (
                f"Branch {case} array != missing on {field}:\nARR={h_arr[field]}\nMISS={h_miss[field]}"
            )
            assert h_nest[field] == h_miss[field], (
                f"Branch {case} nested != missing on {field}:\nNEST={h_nest[field]}\nMISS={h_miss[field]}"
            )

        summary[case] = {
            "base_array": b_arr["raised"] or f"contained_0_health_dms(lane={b_arr['lane_status']})",
            "base_nested": b_nest["raised"] or f"contained_0_health_dms(lane={b_nest['lane_status']})",
            "branch_health_dms": len(h_arr["health_dms"]),
            "branch_pick_dms": len(h_arr["pick_dms"]),
            "branch_lane_status": h_arr["lane_status"],
            "branch_matches_missing_control": True,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["base", "branch"])
    parser.add_argument("--shared-parent")
    args = parser.parse_args()

    if args.worker:
        res = run_worker(args.worker, Path(args.shared_parent))
        sys.stdout.write(json.dumps(res))
        return

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as shared_td:
        base_proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "base", "--shared-parent", shared_td],
            capture_output=True, text=True, timeout=60,
        )
        assert base_proc.returncode == 0, f"Base worker failed:\n{base_proc.stderr}"
        branch_proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "branch", "--shared-parent", shared_td],
            capture_output=True, text=True, timeout=60,
        )
        assert branch_proc.returncode == 0, f"Branch worker failed:\n{branch_proc.stderr}"

    base_res = json.loads(base_proc.stdout)
    branch_res = json.loads(branch_proc.stdout)
    summary = verify_and_summarize(base_res, branch_res)
    out_file = REVIEW_DIR / "probe_callers_v2.json"
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
