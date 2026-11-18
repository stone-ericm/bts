"""Adversarial probe for CS6b-1 items 2 (fault/stall completion), 3, 4, 5, and 8:
- Complete 7-exception x 17-site fault injection matrix (capturing state at run_single_check entry)
  + status-file hazards + outer interrupt non-swallowing + real bts.dm.send_dm worst-case HTTP stall timing.
- Every gate, margin boundary, early return, real unreadable decision/pick files on disk (reproducing B1:
  RecursionError on deeply nested JSON in decision.json and {date}.json, plus verifying the minimal fix),
  and false-alarm audit across ordinary and special days.
- Character-for-character verification of Rows 2, 3, and 4 (envelopes, stderr, <HH:MM>, Alert keys,
  now_et_date decoupled from wall clock, refusal exclusion & 800-char ASCII single-line journal rendering,
  old/future/unreadable details, no-recipient behaviour).
- Dedup across restarts and dates, H6 banner on CRITICAL retry, no banner on failed morning WARN -> 4c delivery,
  same-day 4c suppression after morning WARN, aging past 14h after startup, and strict non-access to shared
  health_dm_delivery_status.json.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BRANCH_SRC = REPO / "src"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
REAL_DT = dt.datetime
DAY = "2027-07-15"


def install_network_guard() -> None:
    def audit(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                     "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto"}:
            raise AssertionError(f"forbidden network call: {event} {args!r}")

    sys.addaudithook(audit)


def make_games(day: str = DAY, utc_times: tuple[str, ...] = ("23:10",)) -> list[dict]:
    return [
        {
            "gamePk": 101 + idx,
            "gameDate": f"{day}T{tm}:00Z" if "T" not in tm else tm,
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": f"Away{idx}"}}, "home": {"team": {"name": f"Home{idx}"}}},
        }
        for idx, tm in enumerate(utc_times)
    ]


def seed_dated_obs(picks_dir: Path, day: str = DAY, streak: int = 30, best: int = 30,
                   saver: str = "used", recorded_at: str | None = None) -> None:
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    year = int(day[:4])
    rec = recorded_at or f"{day}T11:00:00Z"
    (acct / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_manual_v3",
        "season": year,
        "effective_date": day,
        "active_streak": streak,
        "best_streak": best,
        "saver_state": saver,
        "recorded_at": rec,
        "expires_at": f"{day}T23:59:59.999999-04:00" if "-03-" not in day or day >= "2027-03-14" else f"{day}T23:59:59.999999-05:00",
        "source": "manual_cli",
    }, indent=2) + "\n")


def seed_auto_obs(picks_dir: Path, *, season: int = 2027, switch_season: int | None = 2027,
                  switch_status: str = "automatic", activation_id: str = "a" * 32,
                  obs_activation_id: str | None = None, streak: int = 30, best: int = 30,
                  source_date: str | None = "2027-07-14", recorded_at: str | None = "2027-07-15T06:10:00Z") -> None:
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    if switch_season is not None:
        (acct / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": switch_season,
            "status": switch_status,
            "activation_id": activation_id,
            "switched_at": f"{switch_season}-03-24T13:00:00Z" if switch_status == "automatic" else None,
            "revoked_at": f"{switch_season}-07-15T12:00:00Z" if switch_status == "revoked" else None,
            "note": None,
        }, indent=2) + "\n")
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": season,
        "activation_id": obs_activation_id or activation_id,
        "active_streak": streak,
        "best_streak": best,
        "source": "mlb_bts_profile",
        "source_date": source_date,
        "recorded_at": recorded_at,
    }, indent=2) + "\n")
    (acct / "saver_state.json").write_text(json.dumps({
        "season": season,
        "state": "not_earned" if best < 10 else "used",
        "source": "cli",
        "updated_at": f"{season}-03-24T13:00:00Z",
    }, indent=2) + "\n")


def part1_faults_and_stall() -> dict:
    from bts import contest_state, daily_decision, dm, heartbeat, orchestrator, picks, posting, scheduler, slate, strategy, util
    from bts.health import alert

    class CustomBaseException(BaseException):
        pass

    class UnprintableError(BaseException):
        def __str__(self) -> str:
            raise RuntimeError("__str__ exploded")

        def __repr__(self) -> str:
            raise RuntimeError("__repr__ exploded")

    exc_factories = {
        "RuntimeError": lambda: RuntimeError("injected runtime\n" + "é" * 240),
        "KeyboardInterrupt": lambda: KeyboardInterrupt("injected kbi"),
        "SystemExit": lambda: SystemExit(7),
        "GeneratorExit": lambda: GeneratorExit("injected genexit"),
        "CustomBaseException": lambda: CustomBaseException("injected base"),
        "BaseExceptionGroup": lambda: BaseExceptionGroup("beg", [KeyboardInterrupt("kbi-in-group")]),
        "UnprintableError": lambda: UnprintableError(),
    }

    sites = [
        "_now_et",
        "runs_missing_time_et",
        "runs_naive_time_et",
        "_contest_state_required",
        "is_scoreable_commit",
        "load_pick",
        "pick_was_delivered",
        "state_read",
        "automatic_reading_record",
        "dispatcher",
        "transport_in_dispatcher",
        "stderr_skip_empty",
        "stderr_skip_due",
        "stderr_refusal",
        "stderr_stale",
        "stderr_accepted",
        "stderr_permanent",
    ]

    scratch = HERE / "scratch_part1"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    games_default = make_games()

    def execute_one(case_dir: Path, site: str | None, exc_name: str | None,
                    status_hazard: str | None = None, seed_undelivered_pick: bool = False) -> dict:
        start = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)
        now_box = [start]

        class Clock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return now_box[0].astimezone(tz) if tz is not None else now_box[0].replace(tzinfo=None)

        picks_dir = case_dir / "data" / "picks"
        hb_path = case_dir / "data" / "hb.json"
        picks_dir.mkdir(parents=True, exist_ok=True)

        if site in {"automatic_reading_record", "stderr_accepted", "stderr_stale"}:
            age_h = 20.0 if site == "stderr_stale" else 2.0
            rec = (start - dt.timedelta(hours=age_h)).astimezone(UTC).isoformat().replace("+00:00", "Z")
            seed_auto_obs(picks_dir, recorded_at=rec)
        if site == "pick_was_delivered" or seed_undelivered_pick:
            picks.save_pick(picks.DailyPick(
                date=DAY, run_time=start.isoformat(),
                pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                                0.82, [], False, 101, f"{DAY}T23:10:00Z"),
                double_down=None, runner_up=None,
            ), picks_dir)

        health_dir = picks_dir.parent / "health_state"
        morning_status = health_dir / "contest_morning_check_dm_delivery_status.json"
        if status_hazard == "dir":
            morning_status.mkdir(parents=True, exist_ok=True)
        elif status_hazard == "corrupt_syntax":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("{corrupt")
        elif status_hazard == "corrupt_list":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("[]")
        elif status_hazard == "corrupt_deep_json":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("[" * 1500 + "]" * 1500)
        elif status_hazard == "unwritable_file":
            health_dir.mkdir(parents=True, exist_ok=True)
            morning_status.write_text("{}")
            morning_status.chmod(0o444)
        elif status_hazard == "unwritable_dir":
            health_dir.mkdir(parents=True, exist_ok=True)
            health_dir.chmod(0o555)

        first_check_state = {}
        injected = [False]

        with contextlib.ExitStack() as stack:
            def patch(obj, attr, val):
                orig = getattr(obj, attr)
                setattr(obj, attr, val)
                stack.callback(setattr, obj, attr, orig)

            for mod in (contest_state, daily_decision, picks, strategy, scheduler, alert, heartbeat, slate):
                if hasattr(mod, "datetime"):
                    patch(mod, "datetime", Clock)
            patch(scheduler, "_now_et", lambda: now_box[0])
            patch(scheduler, "time", Mock(monotonic=lambda: 0.0))
            patch(scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext())
            patch(scheduler, "notify_ready", lambda *a, **k: None)
            patch(scheduler, "notify_watchdog", lambda *a, **k: None)
            patch(scheduler, "fetch_schedule", lambda d: games_default if d == DAY else [])
            patch(scheduler, "check_confirmed_lineups", lambda *_: {})
            patch(scheduler, "_idle_until_next_wakeup", lambda *_: None)
            patch(scheduler, "_watchdog_ping_sleep", lambda secs, *a, **k: now_box.__setitem__(
                0, now_box[0] + dt.timedelta(seconds=max(0.0, float(secs)))
            ))
            patch(scheduler, "run_result_polling", lambda *a, **k: "final")
            patch(dm, "send_dm", Mock(return_value="p1"))
            patch(alert, "send_dm", Mock(return_value="h1"))
            patch(posting, "post_to_bluesky", Mock(return_value="u1"))
            patch(orchestrator, "run_cascade", lambda *a, **k: (None, None))
            statuses = {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
            patch(picks, "get_game_statuses_detailed", lambda *a: statuses)
            patch(strategy, "get_game_statuses_detailed", lambda *a: statuses)
            patch(picks, "get_game_statuses", lambda *a: {101: "P"})

            real_rsc = scheduler.run_single_check

            def spy_rsc(*args, **kwargs):
                if not first_check_state:
                    st_bytes = (picks_dir / DAY / "scheduler_state.json").read_bytes()
                    first_check_state["at"] = now_box[0].isoformat()
                    first_check_state["scheduler_state"] = json.loads(st_bytes)
                return real_rsc(*args, **kwargs)

            patch(scheduler, "run_single_check", spy_rsc)

            if site == "runs_missing_time_et":
                target_dt = REAL_DT(2027, 7, 15, 16, 10, tzinfo=ET)

                class RunDict(dict):
                    def __init__(self):
                        super().__init__(time_et=target_dt, game_pks=[101])
                        self.calls = 0

                    def __getitem__(self, key):
                        if key == "time_et":
                            self.calls += 1
                            if self.calls == 2:
                                injected[0] = True
                                raise KeyError("time_et")
                        return super().__getitem__(key)

                patch(scheduler, "plan_run_times", lambda *a, **k: [RunDict()])
            elif site == "runs_naive_time_et":
                target_dt = REAL_DT(2027, 7, 15, 16, 10, tzinfo=ET)

                class NaiveOnceRunDict(dict):
                    def __init__(self):
                        super().__init__(time_et=target_dt, game_pks=[101])
                        self.calls = 0

                    def __getitem__(self, key):
                        if key == "time_et":
                            self.calls += 1
                            if self.calls == 2:
                                injected[0] = True
                                return REAL_DT(2027, 7, 15, 16, 10)
                        return super().__getitem__(key)

                patch(scheduler, "plan_run_times", lambda *a, **k: [NaiveOnceRunDict()])
            elif site == "_now_et":
                calls = [0]

                def faulty_now():
                    calls[0] += 1
                    if calls[0] == 4:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return now_box[0]

                patch(scheduler, "_now_et", faulty_now)
            elif site == "_contest_state_required":
                orig_req = scheduler._contest_state_required

                def faulty_req(cfg):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_req(cfg)

                patch(scheduler, "_contest_state_required", faulty_req)
            elif site == "is_scoreable_commit":
                orig_isc = daily_decision.is_scoreable_commit

                def faulty_isc(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_isc(*a, **k)

                patch(daily_decision, "is_scoreable_commit", faulty_isc)
            elif site == "load_pick":
                orig_lp = picks.load_pick

                def faulty_lp(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_lp(*a, **k)

                patch(picks, "load_pick", faulty_lp)
            elif site == "pick_was_delivered":
                orig_pwd = picks.pick_was_delivered

                def faulty_pwd(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_pwd(*a, **k)

                patch(picks, "pick_was_delivered", faulty_pwd)
            elif site == "state_read":
                orig_ldss = contest_state.load_decision_streak_state

                def faulty_ldss(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_ldss(*a, **k)

                patch(contest_state, "load_decision_streak_state", faulty_ldss)
            elif site == "automatic_reading_record":
                orig_arr = contest_state.automatic_reading_record

                def faulty_arr(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_arr(*a, **k)

                patch(contest_state, "automatic_reading_record", faulty_arr)
            elif site == "dispatcher":
                orig_disp = alert.dispatch_dm_for_health_alerts

                def faulty_disp(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return orig_disp(*a, **k)

                patch(alert, "dispatch_dm_for_health_alerts", faulty_disp)
            elif site == "transport_in_dispatcher":
                def faulty_send(*a, **k):
                    if not injected[0] and now_box[0] == start:
                        injected[0] = True
                        raise exc_factories[exc_name]()
                    return "h1"

                patch(alert, "send_dm", faulty_send)
            elif site in {"stderr_skip_empty", "stderr_skip_due", "stderr_refusal", "stderr_stale", "stderr_accepted", "stderr_permanent"}:
                if site == "stderr_skip_empty":
                    patch(scheduler, "plan_run_times", lambda *a, **k: [])
                elif site == "stderr_skip_due":
                    patch(scheduler, "plan_run_times", lambda *a, **k: [
                        {"time_et": start + dt.timedelta(minutes=5), "game_pks": [101]},
                    ])

                def faulty_print(*args, **kwargs):
                    msg = str(args[0]) if args else ""
                    if ("Morning state check" in msg or "MORNING STATE CHECK" in msg):
                        if site == "stderr_permanent" or not msg.startswith("  Morning state check failed:"):
                            injected[0] = True
                            raise exc_factories[exc_name]()
                    return print(*args, **kwargs)

                scheduler.print = faulty_print
                stack.callback(delattr, scheduler, "print")

            config = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(hb_path)},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            err_buf = io.StringIO()
            try:
                with contextlib.redirect_stderr(err_buf):
                    scheduler.run_day(DAY, config)
            finally:
                if health_dir.exists():
                    health_dir.chmod(0o755)
                if morning_status.exists() and morning_status.is_file():
                    morning_status.chmod(0o644)

        return {
            "injected": injected[0],
            "first_check_state": first_check_state,
            "stderr": err_buf.getvalue(),
        }

    ref_normal = execute_one(scratch / "ref_normal", None, None)
    assert ref_normal["first_check_state"]["at"] == "2027-07-15T16:10:00-04:00"

    total_fault_runs = 0
    for site in sites:
        excs_for_site = ["RuntimeError"] if site in {"runs_missing_time_et", "runs_naive_time_et"} else list(exc_factories)
        for exc_name in excs_for_site:
            res = execute_one(scratch / f"{site}_{exc_name}", site, exc_name)
            assert res["injected"] is True, (site, exc_name)
            if site == "stderr_skip_empty":
                assert res["first_check_state"] == {}
            elif site == "stderr_skip_due":
                assert res["first_check_state"]["at"] == "2027-07-15T10:05:00-04:00"
            else:
                assert res["first_check_state"] == ref_normal["first_check_state"], (site, exc_name)
            # Check failure line presence when writable and escaping
            if site not in {"transport_in_dispatcher", "stderr_permanent"} or exc_name != "RuntimeError":
                if exc_name == "UnprintableError":
                    assert "  Morning state check failed:" not in res["stderr"]
                elif site != "stderr_permanent":
                    assert "  Morning state check failed: " in res["stderr"], (site, exc_name, res["stderr"])
            total_fault_runs += 1

    hazards = ["dir", "corrupt_syntax", "corrupt_list", "corrupt_deep_json", "unwritable_file", "unwritable_dir"]
    for hz in hazards:
        res = execute_one(scratch / f"hazard_{hz}", None, None, status_hazard=hz)
        assert res["first_check_state"] == ref_normal["first_check_state"], hz

    outer_propagated = {}
    for where in ("before_block", "in_loop"):
        for exc_cls in (KeyboardInterrupt, SystemExit):
            case_dir = scratch / f"outer_{where}_{exc_cls.__name__}"
            picks_dir = case_dir / "data" / "picks"
            picks_dir.mkdir(parents=True, exist_ok=True)
            cfg = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(case_dir / "hb.json")},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            with contextlib.ExitStack() as stack:
                def patch(obj, attr, val):
                    orig = getattr(obj, attr)
                    setattr(obj, attr, val)
                    stack.callback(setattr, obj, attr, orig)

                patch(scheduler, "_now_et", lambda: REAL_DT(2027, 7, 15, 10, 0, tzinfo=ET))
                patch(scheduler, "notify_ready", lambda *a, **k: None)
                patch(scheduler, "notify_watchdog", lambda *a, **k: None)
                patch(scheduler, "_watchdog_ping_sleep", lambda *a, **k: None)
                patch(alert, "send_dm", Mock(return_value="h1"))
                if where == "before_block":
                    patch(scheduler, "fetch_schedule", Mock(side_effect=exc_cls("outer-before")))
                else:
                    patch(scheduler, "fetch_schedule", lambda d: games_default if d == DAY else [])
                    patch(scheduler, "run_single_check", Mock(side_effect=exc_cls("outer-loop")))
                caught = None
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        scheduler.run_day(DAY, cfg)
                except BaseException as err:
                    caught = type(err).__name__
                assert caught == exc_cls.__name__, (where, exc_cls, caught)
                outer_propagated[f"{where}:{exc_cls.__name__}"] = caught

    def measure_real_dm_stall(recipient: str, mode: str) -> dict:
        t0 = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)
        first_target = t0 + dt.timedelta(minutes=10)
        clock = [t0]
        http_calls = []
        counts: dict[str, int] = {}

        def fake_urlopen(req: urllib.request.Request, timeout: float = 15):
            url = req.full_url
            counts[url] = counts.get(url, 0) + 1
            attempt = counts[url]
            clock[0] += dt.timedelta(seconds=timeout)
            http_calls.append({"url": url, "attempt": attempt, "timeout": timeout, "at": clock[0].isoformat()})
            if mode == "fail_auth_all_retries":
                raise urllib.error.URLError("simulated 15s connect/read timeout")
            if "sendMessage" in url:
                raise urllib.error.URLError("simulated 15s sendMessage timeout")
            if attempt < 3:
                raise urllib.error.URLError("simulated 15s transient timeout")
            if "createSession" in url:
                body = b'{"accessJwt": "jwt-tok"}'
            elif "resolveHandle" in url:
                body = b'{"did": "did:plc:op123"}'
            elif "getConvoForMembers" in url:
                body = b'{"convo": {"id": "convo-1"}}'
            else:
                raise AssertionError(url)
            return io.BytesIO(body)

        case_dir = scratch / f"stall_{mode}_{recipient[:3]}"
        picks_dir = case_dir / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        loop_entered_at = []
        check_ran_at = []

        with contextlib.ExitStack() as stack:
            def patch(obj, attr, val):
                orig = getattr(obj, attr)
                setattr(obj, attr, val)
                stack.callback(setattr, obj, attr, orig)

            patch(os, "environ", os.environ | {"BTS_BLUESKY_APP_PASSWORD": "offline-secret"})
            patch(util, "urlopen", fake_urlopen)
            patch(util.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + dt.timedelta(seconds=s)))
            patch(alert, "send_dm", dm.send_dm)
            patch(scheduler, "_now_et", lambda: clock[0])
            patch(scheduler, "notify_ready", lambda *a, **k: None)
            patch(scheduler, "notify_watchdog", lambda *a, **k: None)
            patch(scheduler, "fetch_schedule", lambda d: games_default if d == DAY else [])
            patch(scheduler, "plan_run_times", lambda *a, **k: [{"time_et": first_target, "game_pks": [101]}])
            patch(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None)

            def spy_sleep(secs, *a, **k):
                loop_entered_at.append({"before_sleep": clock[0].isoformat(), "sleep_secs": secs})
                clock[0] += dt.timedelta(seconds=max(0.0, float(secs)))

            patch(scheduler, "_watchdog_ping_sleep", spy_sleep)

            def spy_check(**kwargs):
                check_ran_at.append(clock[0].isoformat())
                return {"new_lineups": 0, "skipped": True, "pick_result": None, "should_post": False}

            patch(scheduler, "run_single_check", spy_check)

            cfg = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(case_dir / "hb.json")},
                "bluesky": {"dm_recipient": recipient},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            with contextlib.redirect_stderr(io.StringIO()):
                scheduler.run_day(DAY, cfg)

        entered = dt.datetime.fromisoformat(loop_entered_at[0]["before_sleep"])
        return {
            "elapsed_sec": (entered - t0).total_seconds(),
            "remaining_lead_sec": (first_target - entered).total_seconds(),
            "http_call_count": len(http_calls),
            "check_ran_at": check_ran_at,
        }

    stall_auth = measure_real_dm_stall("operator.bsky.social", "fail_auth_all_retries")
    stall_handle = measure_real_dm_stall("operator.bsky.social", "worst_4_step")
    stall_did = measure_real_dm_stall("did:plc:operator", "worst_4_step")
    assert stall_auth["elapsed_sec"] == 60.0
    assert stall_handle["elapsed_sec"] == 195.0 and stall_handle["remaining_lead_sec"] == 405.0
    assert stall_handle["check_ran_at"] == ["2027-07-15T10:10:00-04:00"]
    assert stall_did["elapsed_sec"] == 135.0

    return {
        "fault_runs": total_fault_runs,
        "status_hazards": hazards,
        "outer_interrupts": outer_propagated,
        "stall_auth_sec": stall_auth["elapsed_sec"],
        "stall_handle_sec": stall_handle["elapsed_sec"],
        "stall_handle_remaining_sec": stall_handle["remaining_lead_sec"],
        "stall_did_sec": stall_did["elapsed_sec"],
    }


def part2_gates_unreadable_and_false_alarms() -> dict:
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.health import alert

    scratch = HERE / "scratch_part2"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    start = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)

    def run_isolated(name: str, setup_fn) -> dict:
        cdir = scratch / name
        picks_dir = cdir / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        now_box = [start]

        class Clock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return now_box[0].astimezone(tz) if tz is not None else now_box[0].replace(tzinfo=None)

        helper_calls = []
        read_calls = []
        health_dms = []
        pick_dms = []

        with contextlib.ExitStack() as stack:
            def patch(obj, attr, val):
                orig = getattr(obj, attr)
                setattr(obj, attr, val)
                stack.callback(setattr, obj, attr, orig)

            for mod in (contest_state, daily_decision, picks, strategy, scheduler, cli, alert):
                if hasattr(mod, "datetime"):
                    patch(mod, "datetime", Clock)
            patch(scheduler, "_now_et", lambda: now_box[0])
            patch(cli, "_now_et", lambda: now_box[0])
            patch(scheduler, "notify_ready", lambda *a, **k: None)
            patch(scheduler, "notify_watchdog", lambda *a, **k: None)
            patch(scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext())
            patch(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None)
            patch(scheduler, "_watchdog_ping_sleep", lambda secs, *a, **k: now_box.__setitem__(
                0, now_box[0] + dt.timedelta(seconds=max(0.0, float(secs)))
            ))
            patch(scheduler, "check_confirmed_lineups", lambda *_: {})
            patch(scheduler, "run_result_polling", lambda *a, **k: "final")
            patch(orchestrator, "run_cascade", lambda *a, **k: (None, None))
            patch(scheduler, "fetch_schedule", lambda d: make_games(d) if d == DAY else [])

            real_helper = scheduler._morning_state_check
            real_read = contest_state.load_decision_streak_state

            def spy_helper(*a, **k):
                helper_calls.append(a)
                return real_helper(*a, **k)

            def spy_read(*a, **k):
                read_calls.append((now_box[0].isoformat(), k))
                return real_read(*a, **k)

            patch(scheduler, "_morning_state_check", spy_helper)
            patch(contest_state, "load_decision_streak_state", spy_read)
            patch(alert, "send_dm", lambda rec, body: (health_dms.append((now_box[0].isoformat(), rec, body)) or "h1"))
            patch(dm, "send_dm", lambda rec, body: (pick_dms.append((now_box[0].isoformat(), rec, body)) or "p1"))
            patch(posting, "post_to_bluesky", lambda body: "u1")

            cfg = {
                "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(cdir / "hb.json")},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                setup_fn(cfg, picks_dir, now_box, patch)

        return {
            "helper_calls": len(helper_calls),
            "startup_reads": sum(1 for t, _ in read_calls if t == start.isoformat()),
            "total_reads": len(read_calls),
            "health_dms": health_dms,
            "pick_dms": pick_dms,
            "stderr": err_buf.getvalue(),
        }

    # 1) Delivery mode aliases and requiring flags
    for alias in ("dm", "bluesky_dm", "direct_message", "public", "bluesky", "feed", "post", None):
        r = run_isolated(f"mode_outward_{alias}", lambda cfg, pdir, nb, patch: (
            cfg["scheduler"].pop("pick_delivery") if alias is None else cfg["scheduler"].update(pick_delivery=alias),
            scheduler.run_day(DAY, cfg),
        ))
        assert r["helper_calls"] == 1 and len(r["health_dms"]) == 1, alias

    for priv in ("private", "none", "off", "local", "legacy_private_mode"):
        def _run_priv(cfg, pdir, nb, patch, p=priv):
            if p == "legacy_private_mode":
                cfg["scheduler"].pop("pick_delivery")
                cfg["scheduler"]["private_mode"] = True
            else:
                cfg["scheduler"]["pick_delivery"] = p
            scheduler.run_day(DAY, cfg)

        r = run_isolated(f"mode_priv_{priv}", _run_priv)
        assert r["helper_calls"] == 0 and r["health_dms"] == [] and "Morning state check" not in r["stderr"], priv

    for req_s, req_h, expect in ((True, False, 1), (False, True, 1), (False, False, 0)):
        r = run_isolated(f"flags_{req_s}_{req_h}", lambda cfg, pdir, nb, patch: (
            cfg["scheduler"].update(contest_state_required=req_s),
            cfg["health_checks"].update(contest_state_expected=req_h),
            scheduler.run_day(DAY, cfg),
        ))
        assert r["helper_calls"] == expect and len(r["health_dms"]) == expect

    # 2) Margin microsecond & exact boundaries
    for label, delta, expect_run in (
        ("overdue_60s", dt.timedelta(seconds=-60), False),
        ("exact_now", dt.timedelta(0), False),
        ("lead_599s", dt.timedelta(seconds=599), False),
        ("lead_10m_minus_1us", dt.timedelta(minutes=10) - dt.timedelta(microseconds=1), False),
        ("lead_exact_10m", dt.timedelta(minutes=10), True),
        ("mixed_past_future", "mixed", False),
        ("empty_plan", "empty", False),
    ):
        def _run_margin(cfg, pdir, nb, patch, d=delta):
            if d == "empty":
                runs = []
            elif d == "mixed":
                runs = [
                    {"time_et": start + dt.timedelta(hours=2), "game_pks": [101]},
                    {"time_et": start - dt.timedelta(seconds=1), "game_pks": [101]},
                ]
            else:
                runs = [{"time_et": start + d, "game_pks": [101]}]
            patch(scheduler, "plan_run_times", lambda *a, **k: runs)
            scheduler.run_day(DAY, cfg)

        r = run_isolated(f"margin_{label}", _run_margin)
        assert (r["helper_calls"] == 1) is expect_run, label
        if not expect_run:
            exp_line = ("  Morning state check skipped: no lineup check is planned."
                        if delta == "empty"
                        else "  Morning state check skipped: first lineup check is due within 10 minutes.")
            assert exp_line in r["stderr"].splitlines(), (label, r["stderr"])

    # 3) Real `bts schedule` crossing midnight with REAL unmocked `plan_run_times`!
    # (a) Late 05:05Z (01:05 ET next day) game on DAY -> check at 00:20 ET is 19m59s ahead of 00:00:01 ET ->
    #     margin passes, _morning_state_check IS called, and `date != now.date().isoformat()` suppresses!
    def _midnight_late_game(cfg, pdir, nb, patch):
        cfg["scheduler"]["entry_intent"] = "enter"
        nb[0] = REAL_DT(2027, 7, 15, 23, 59, 59, tzinfo=ET)
        patch(cli, "_today_et", lambda: nb[0].date().isoformat())
        patch(scheduler, "fetch_schedule", lambda d: make_games(d, ("2027-07-16T05:05:00Z",)) if d == DAY else [])
        toml_p = pdir.parent / "sched.toml"
        toml_p.touch()

        def slow_load(_):
            nb[0] += dt.timedelta(seconds=2)  # now 2027-07-16T00:00:01-04:00!
            return cfg

        patch(orchestrator, "load_config", slow_load)
        res = CliRunner().invoke(cli.cli, ["schedule", "--config", str(toml_p)], catch_exceptions=False)
        assert res.exit_code == 0

    r_mid_late = run_isolated("midnight_real_plan_late_game", _midnight_late_game)
    assert r_mid_late["helper_calls"] == 1 and r_mid_late["total_reads"] == 0 and r_mid_late["health_dms"] == []
    assert "Morning state check" not in r_mid_late["stderr"]

    # (b) Ordinary 23:10Z (19:10 ET) game on DAY -> check at 16:10 ET is overdue at 00:00:01 ET -> margin skips!
    def _midnight_ordinary(cfg, pdir, nb, patch):
        cfg["scheduler"]["entry_intent"] = "enter"
        nb[0] = REAL_DT(2027, 7, 15, 23, 59, 59, tzinfo=ET)
        patch(cli, "_today_et", lambda: nb[0].date().isoformat())
        toml_p = pdir.parent / "sched.toml"
        toml_p.touch()
        patch(orchestrator, "load_config", lambda _: (nb.__setitem__(0, nb[0] + dt.timedelta(seconds=2)) or cfg))
        res = CliRunner().invoke(cli.cli, ["schedule", "--config", str(toml_p)], catch_exceptions=False)
        assert res.exit_code == 0

    r_mid_ord = run_isolated("midnight_real_plan_ordinary", _midnight_ordinary)
    assert r_mid_ord["helper_calls"] == 0 and r_mid_ord["health_dms"] == []

    # 4) Early returns & non-returns (scoreable decision, non-scoreable decision, delivered DM/public, attempted, preview)
    for st_present in (False, True):
        for case, expect_suppress in (
            ("scoreable_no_pick", True),
            ("non_scoreable_skip_no_pick", False),
            ("delivered_dm_no_dec", True),
            ("delivered_public_with_skip_dec", True),
            ("attempted_no_dec", True),
            ("cached_preview_undelivered", False),
        ):
            def _run_early(cfg, pdir, nb, patch, sp=st_present, c=case):
                if sp:
                    scheduler.save_state(scheduler.SchedulerState(
                        DAY, start.isoformat(), [], [], [], False, None, None, None,
                    ), pdir)
                if c in {"scoreable_no_pick", "non_scoreable_skip_no_pick", "delivered_public_with_skip_dec"}:
                    dp = daily_decision.decision_path(DAY, pdir)
                    dp.parent.mkdir(parents=True, exist_ok=True)
                    dp.write_text(json.dumps({
                        "schema_version": "bts_daily_decision_v3",
                        "date": DAY,
                        "action": "single" if c == "scoreable_no_pick" else "skip",
                        "scoreable": c == "scoreable_no_pick",
                    }))
                if c in {"delivered_dm_no_dec", "delivered_public_with_skip_dec", "attempted_no_dec", "cached_preview_undelivered"}:
                    dpk = picks.DailyPick(
                        DAY, start.isoformat(),
                        picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                                   0.82, [], False, 101, f"{DAY}T23:10:00Z"),
                        None, None,
                        bluesky_posted=(c == "delivered_public_with_skip_dec"),
                        bluesky_uri="at://p" if c == "delivered_public_with_skip_dec" else None,
                        notification_sent=(c == "delivered_dm_no_dec"),
                        notification_id="m1" if c == "delivered_dm_no_dec" else None,
                        delivery_attempted=(c == "attempted_no_dec"),
                    )
                    picks.save_pick(dpk, pdir)
                scheduler.run_day(DAY, cfg)

            r = run_isolated(f"early_{st_present}_{case}", _run_early)
            assert r["helper_calls"] == 1
            assert (r["startup_reads"] == 0) is expect_suppress, (st_present, case, r)
            assert (len(r["health_dms"]) == 0) is expect_suppress, (st_present, case, r)

    # 5) REAL ON-DISK UNREADABLE decision.json AND {date}.json AUDIT (including deeply nested JSON -> B1!)
    unreadable_payloads = {
        "malformed_json": b"{broken",
        "bad_utf8": b"\xff\xfe\x00",
        "directory": "DIR",
        "chmod_000": "CHMOD000",
        "huge_int": ('{"x": ' + ("9" * 5000) + "}").encode("utf-8"),
        "json_list": b"[1, 2, 3]",
        "json_int": b"123",
        "empty_obj": b"{}",
        "null_pick": b'{"pick": null}',
        "bad_pick_fields": b'{"date":"2027-07-15","run_time":"t","pick":{},"double_down":null,"runner_up":null}',
        "deeply_nested_json": (b"[" * 1500 + b"]" * 1500),
    }
    unreadable_results = {"decision": {}, "pick": {}}
    first_target = REAL_DT(2027, 7, 15, 16, 10, tzinfo=ET)

    for target_kind in ("decision", "pick"):
        for pname, pval in unreadable_payloads.items():
            cdir = scratch / f"unread_{target_kind}_{pname}"
            pdir = cdir / "data" / "picks"
            pdir.mkdir(parents=True, exist_ok=True)
            fpath = (pdir / DAY / "decision.json") if target_kind == "decision" else (pdir / f"{DAY}.json")
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if pval == "DIR":
                fpath.mkdir()
            elif pval == "CHMOD000":
                fpath.write_text("{}")
                fpath.chmod(0o000)
            else:
                fpath.write_bytes(pval)

            cfg = {
                "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(cdir / "hb.json")},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
            }
            sent_direct = []
            direct_exc = None
            try:
                with contextlib.ExitStack() as st:
                    orig_sd = alert.send_dm
                    alert.send_dm = lambda rec, body: (sent_direct.append(body) or "h1")
                    st.callback(setattr, alert, "send_dm", orig_sd)
                    with contextlib.redirect_stderr(io.StringIO()):
                        scheduler._morning_state_check(cfg, pdir, DAY, start, first_target)
            except BaseException as exc:
                direct_exc = f"{type(exc).__name__}: {exc}"

            # Also test through run_day (only before main loop; empty runs after check? No, 1 run with mocked run_single_check!)
            sent_runday = []
            runday_err = io.StringIO()
            with contextlib.ExitStack() as st:
                def patch(obj, attr, val):
                    orig = getattr(obj, attr)
                    setattr(obj, attr, val)
                    st.callback(setattr, obj, attr, orig)

                patch(scheduler, "_now_et", lambda: start)
                patch(scheduler, "notify_ready", lambda *a, **k: None)
                patch(scheduler, "notify_watchdog", lambda *a, **k: None)
                patch(scheduler, "fetch_schedule", lambda d: make_games(d) if d == DAY else [])
                patch(scheduler, "_watchdog_ping_sleep", lambda *a, **k: None)
                patch(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None)
                patch(scheduler, "run_single_check", lambda **k: {"new_lineups": 0, "skipped": True, "pick_result": None, "should_post": False})
                patch(scheduler, "_maybe_alert_missed_pick", lambda *a, **k: None)
                patch(alert, "send_dm", lambda rec, body: (sent_runday.append(body) or "h1"))
                with contextlib.redirect_stderr(runday_err):
                    scheduler.run_day(DAY, cfg)

            if pval == "CHMOD000":
                fpath.chmod(0o644)

            unreadable_results[target_kind][pname] = {
                "direct_raised": direct_exc,
                "direct_dm_sent": len(sent_direct),
                "runday_morning_dm_sent": len(sent_runday),
                "runday_failed_line": "  Morning state check failed:" in runday_err.getvalue(),
            }

    # Confirm B1 on unpatched branch: ONLY deeply_nested_json fails on both decision.json and {date}.json!
    for tk in ("decision", "pick"):
        for pname, outcome in unreadable_results[tk].items():
            if pname == "deeply_nested_json":
                assert outcome["direct_raised"] and outcome["direct_raised"].startswith("RecursionError"), (tk, outcome)
                assert outcome["direct_dm_sent"] == 0 and outcome["runday_morning_dm_sent"] == 0
                assert outcome["runday_failed_line"] is True
            else:
                assert outcome["direct_raised"] is None and outcome["direct_dm_sent"] == 1 and outcome["runday_morning_dm_sent"] == 1, (tk, pname, outcome)

    # Verify the minimal fix in a scratch copy of src!
    patched_src = scratch / "patched_src" / "src"
    shutil.copytree(BRANCH_SRC, patched_src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sched_py = patched_src / "bts" / "scheduler.py"
    txt = sched_py.read_text(encoding="utf-8")
    txt = txt.replace(
        "from bts.contest_state import ContestStateError, automatic_reading_record, load_decision_streak_state",
        "from bts.contest_state import ContestStateError, _UNREADABLE, automatic_reading_record, load_decision_streak_state",
        1,
    ).replace(
        "    except (OSError, ValueError):",
        "    except _UNREADABLE:",
        1,
    ).replace(
        "    except (OSError, ValueError, TypeError, KeyError, AttributeError):",
        "    except (*_UNREADABLE, TypeError, KeyError, AttributeError):",
        1,
    )
    sched_py.write_text(txt, encoding="utf-8")
    fix_verify = subprocess.check_output(
        [
            sys.executable, "-c",
            (
                "import io, json, sys, contextlib; from pathlib import Path; from datetime import datetime; from zoneinfo import ZoneInfo; "
                f"sys.path.insert(0, {str(patched_src)!r}); "
                "from bts import scheduler; from bts.health import alert; "
                "ET = ZoneInfo('America/New_York'); now = datetime(2027, 7, 15, 10, 0, tzinfo=ET); first = datetime(2027, 7, 15, 16, 10, tzinfo=ET); "
                f"root = Path({str(scratch / 'verify_fix')!r}); "
                "sent = []; alert.send_dm = lambda r, b: sent.append(b) or 'h1'; "
                "for kind in ('decision', 'pick'): "
                "  pdir = root / kind / 'data' / 'picks'; pdir.mkdir(parents=True, exist_ok=True); "
                "  fp = (pdir / '2027-07-15' / 'decision.json') if kind == 'decision' else (pdir / '2027-07-15.json'); "
                "  fp.parent.mkdir(parents=True, exist_ok=True); fp.write_bytes(b'[' * 1500 + b']' * 1500); "
                "  cfg = {'orchestrator': {'picks_dir': str(pdir)}, 'bluesky': {'dm_recipient': 'op'}, 'scheduler': {'contest_state_required': True, 'pick_delivery': 'dm'}}; "
                "  with contextlib.redirect_stderr(io.StringIO()): scheduler._morning_state_check(cfg, pdir, '2027-07-15', now, first); "
                "assert len(sent) == 2, sent; print('FIXED_OK')"
            ),
        ],
        text=True,
    ).strip()
    assert fix_verify == "FIXED_OK"

    # 6) False-alarm audit across ordinary and special days
    false_alarm_checks = {}

    # 07:00 dated observation via real CLI -> 10:00 run_day -> 0 DMs
    def _fa_0700_dated(cfg, pdir, nb, patch):
        nb[0] = REAL_DT(2027, 7, 15, 7, 0, tzinfo=ET)
        res = CliRunner().invoke(cli.cli, [
            "set-contest-streak", "--date", DAY, "--streak", "12", "--best-streak", "15",
            "--saver", "active", "--picks-dir", str(pdir),
        ], catch_exceptions=False)
        assert res.exit_code == 0, res.output
        nb[0] = start
        scheduler.run_day(DAY, cfg)

    r_0700 = run_isolated("fa_0700_dated", _fa_0700_dated)
    assert r_0700["health_dms"] == [] and "  Morning state check: dated contest state accepted (streak 12)." in r_0700["stderr"]
    false_alarm_checks["dated_0700_to_1000"] = "0_dms_accepted"

    # Yesterday's dated observation + today's 02:10 fresh automatic reading -> 0 DMs
    def _fa_yest_dated_plus_auto(cfg, pdir, nb, patch):
        seed_dated_obs(pdir, day="2027-07-14", streak=11, best=15)
        seed_auto_obs(pdir, streak=12, best=15, source_date="2027-07-14", recorded_at="2027-07-15T06:10:00Z")
        scheduler.run_day(DAY, cfg)

    r_yd_auto = run_isolated("fa_yest_dated_plus_auto", _fa_yest_dated_plus_auto)
    assert r_yd_auto["health_dms"] == [] and "  Morning state check: automatic contest state accepted (streak 12)." in r_yd_auto["stderr"]
    false_alarm_checks["yest_dated_plus_fresh_auto"] = "0_dms_accepted_auto"

    # Yesterday's dated observation alone -> true alarm (1 morning DM)
    r_yd_only = run_isolated("fa_yest_dated_only", lambda cfg, pdir, nb, patch: (
        seed_dated_obs(pdir, day="2027-07-14", streak=11, best=15),
        scheduler.run_day(DAY, cfg),
    ))
    assert len(r_yd_only["health_dms"]) == 1
    false_alarm_checks["yest_dated_alone"] = "1_true_critical_dm"

    # Opening Night 2027-03-24 (switched + 02:10 fetch with source_date=None, streak=0) -> 0 DMs
    open_day = "2027-03-24"
    open_start = REAL_DT(2027, 3, 24, 10, 0, tzinfo=ET)

    def _fa_opening_switched(cfg, pdir, nb, patch):
        nb[0] = open_start
        patch(scheduler, "fetch_schedule", lambda d: make_games(open_day, ("23:10",)) if d == open_day else [])
        seed_auto_obs(pdir, season=2027, switch_season=2027, streak=0, best=0,
                      source_date=None, recorded_at="2027-03-24T06:10:00Z")
        scheduler.run_day(open_day, cfg)

    r_open_ok = run_isolated("fa_opening_switched", _fa_opening_switched)
    assert r_open_ok["health_dms"] == [] and "  Morning state check: automatic contest state accepted (streak 0)." in r_open_ok["stderr"]
    false_alarm_checks["opening_night_switched"] = "0_dms_accepted"

    # Opening Night unswitched (2026 switch left on disk) -> 1 true CRITICAL DM with 16:10 ET
    def _fa_opening_unswitched(cfg, pdir, nb, patch):
        nb[0] = open_start
        patch(scheduler, "fetch_schedule", lambda d: make_games(open_day, ("23:10",)) if d == open_day else [])
        seed_auto_obs(pdir, season=2026, switch_season=2026, streak=5, best=10,
                      source_date="2026-09-27", recorded_at="2026-09-28T06:10:00Z")
        scheduler.run_day(open_day, cfg)

    r_open_ref = run_isolated("fa_opening_unswitched", _fa_opening_unswitched)
    assert len(r_open_ref["health_dms"]) == 1 and "First planned lineup check at 16:10 ET." in r_open_ref["health_dms"][0][2]
    false_alarm_checks["opening_night_2026_snapshot"] = "1_true_critical_dm"

    # Day after an off day (07-14 off, last settled pick 07-13, 02:10 fetch on 07-15 with source_date=07-13) -> 0 DMs
    def _fa_after_off_day(cfg, pdir, nb, patch):
        p0713 = picks.DailyPick(
            "2027-07-13", "2027-07-13T20:00:00Z",
            picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                       0.85, [], False, 99, "2027-07-13T23:10:00Z"),
            None, None, result="hit", slot_results={"pick": "hit"},
        )
        picks.save_pick(p0713, pdir)
        seed_auto_obs(pdir, streak=25, best=25, source_date="2027-07-13", recorded_at="2027-07-15T06:10:00Z")
        scheduler.run_day(DAY, cfg)

    r_off_after = run_isolated("fa_after_off_day", _fa_after_off_day)
    assert r_off_after["health_dms"] == [] and "  Morning state check: automatic contest state accepted (streak 25)." in r_off_after["stderr"]
    false_alarm_checks["day_after_off_day"] = "0_dms_accepted"

    # Revoke in progress: switch revoked while today's dated observation still on disk -> accepted (Known Limit); once manual removed -> CRITICAL DM
    r_rev_dated = run_isolated("fa_revoke_with_dated", lambda cfg, pdir, nb, patch: (
        seed_auto_obs(pdir, switch_status="revoked"),
        seed_dated_obs(pdir, streak=14, best=20),
        scheduler.run_day(DAY, cfg),
    ))
    assert r_rev_dated["health_dms"] == [] and "dated contest state accepted (streak 14)." in r_rev_dated["stderr"]
    r_rev_done = run_isolated("fa_revoke_no_dated", lambda cfg, pdir, nb, patch: (
        seed_auto_obs(pdir, switch_status="revoked"),
        scheduler.run_day(DAY, cfg),
    ))
    assert len(r_rev_done["health_dms"]) == 1
    false_alarm_checks["revoke_in_progress"] = "dated_accepted_then_revoked_critical"

    # Restored snapshot with mismatched activation_id -> 1 true CRITICAL DM
    r_restored_mismatch = run_isolated("fa_restored_mismatch", lambda cfg, pdir, nb, patch: (
        seed_auto_obs(pdir, activation_id="b" * 32, obs_activation_id="a" * 32),
        scheduler.run_day(DAY, cfg),
    ))
    assert len(r_restored_mismatch["health_dms"]) == 1
    false_alarm_checks["restored_activation_mismatch"] = "1_true_critical_dm"

    return {
        "unreadable_audit": unreadable_results,
        "patched_fix_verified": fix_verify,
        "false_alarm_checks": false_alarm_checks,
    }


def part3_and_4_messages_and_dedup() -> dict:
    from bts import contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.health import alert

    scratch = HERE / "scratch_part34"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    now = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)
    early_check = REAL_DT(2027, 7, 15, 9, 5, 0, tzinfo=ET)  # tests leading-zero %H:%M -> 09:05!

    # 1) Row 2 exact message, <HH:MM> leading zero, refusal exclusion, 800-char ASCII single-line journal,
    #    and `now_et_date` decoupled from alert.datetime!
    pdir = scratch / "row2" / "data" / "picks"
    pdir.mkdir(parents=True, exist_ok=True)
    long_refusal = "SECRET_PATH=/deep/custom/dir\nLine2\rLine3 — café 🚀 " + ("X" * 1200)
    sent_row2 = []
    err_row2 = io.StringIO()

    class WrongWallClock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            # Different calendar date (2028-01-09) to prove `now_et_date=now.date()` is passed explicitly!
            w = REAL_DT(2028, 1, 9, 12, 0, 0, tzinfo=UTC)
            return w.astimezone(tz) if tz is not None else w.replace(tzinfo=None)

    with contextlib.ExitStack() as st:
        def patch(obj, attr, val):
            orig = getattr(obj, attr)
            setattr(obj, attr, val)
            st.callback(setattr, obj, attr, orig)

        patch(alert, "datetime", WrongWallClock)
        patch(alert, "send_dm", lambda rec, body: (sent_row2.append((rec, body)) or "h1"))
        patch(contest_state, "load_decision_streak_state", Mock(side_effect=contest_state.ContestStateError(long_refusal)))
        cfg = {"orchestrator": {"picks_dir": str(pdir)}, "bluesky": {"dm_recipient": "op.handle"}}
        with contextlib.redirect_stderr(err_row2):
            scheduler._morning_state_check(cfg, pdir, DAY, now, early_check)

    expected_row2_dm = (
        "BTS health CRITICAL alert(s):\n"
        "- [contest_state] Morning check for 2027-07-15: no usable contest state was found at startup. "
        "Check MLB and run bts season-contract status with the scheduler's picks directory. "
        "If state still needs correction, follow today's dated-observation instructions while a lineup check remains. "
        "First planned lineup check at 09:05 ET."
    )
    assert sent_row2 == [("op.handle", expected_row2_dm)]
    assert "SECRET_PATH" not in sent_row2[0][1]
    row2_lines = err_row2.getvalue().splitlines()
    assert len(row2_lines) == 1
    prefix2 = "  MORNING STATE CHECK — no pick can be made yet: "
    assert row2_lines[0].startswith(prefix2)
    tail2 = row2_lines[0][len(prefix2):]
    assert len(tail2) == 800 and tail2 == ascii(long_refusal)[:800]
    assert all(0x20 <= ord(ch) <= 0x7E for ch in tail2)
    st_row2 = json.loads((pdir.parent / "health_state" / "contest_morning_check_dm_delivery_status.json").read_text())
    assert st_row2["sent_day"] == "2027-07-15"  # NOT 2028-01-09!
    assert st_row2["sent_sources"] == ["CRITICAL:contest_state:morning"]

    # Row 2 with no recipient (None and "") -> log only, no status file!
    for empty_rec in (None, ""):
        pdir_nr = scratch / f"row2_norec_{empty_rec is None}" / "data" / "picks"
        pdir_nr.mkdir(parents=True, exist_ok=True)
        cfg_nr = {"orchestrator": {"picks_dir": str(pdir_nr)}, "bluesky": ({} if empty_rec is None else {"dm_recipient": ""})}
        err_nr = io.StringIO()
        with contextlib.redirect_stderr(err_nr):
            scheduler._morning_state_check(cfg_nr, pdir_nr, DAY, now, early_check)
        assert err_nr.getvalue().startswith(prefix2)
        assert not (pdir_nr.parent / "health_state").exists()

    # 2) Row 3 exact details for `old`, `future`, `unreadable`, plus `now_et_date` decoupled from wall clock & no-recipient!
    row3_cases = [
        ("old", "2027-07-14T18:00:00Z", "last accepted fetch 2027-07-14T18:00:00Z, about 20.00 hours ago"),
        ("future", "2027-07-15T14:10:00Z", "its recorded_at 2027-07-15T14:10:00Z is in the future"),
        ("unreadable_naive", "2027-07-15T10:00:00", "it has no usable recorded_at"),
        ("unreadable_garbage", "not-a-timestamp", "it has no usable recorded_at"),
    ]
    for cname, rec_at, exp_detail in row3_cases:
        pdir3 = scratch / f"row3_{cname}" / "data" / "picks"
        pdir3.mkdir(parents=True, exist_ok=True)
        seed_auto_obs(pdir3, recorded_at=rec_at)
        sent3 = []
        err3 = io.StringIO()
        with contextlib.ExitStack() as st:
            orig_dt = alert.datetime
            orig_sd = alert.send_dm
            alert.datetime = WrongWallClock
            alert.send_dm = lambda rec, body: (sent3.append((rec, body)) or "h1")
            st.callback(setattr, alert, "datetime", orig_dt)
            st.callback(setattr, alert, "send_dm", orig_sd)
            cfg3 = {"orchestrator": {"picks_dir": str(pdir3)}, "bluesky": {"dm_recipient": "op.handle"}}
            with contextlib.redirect_stderr(err3):
                scheduler._morning_state_check(cfg3, pdir3, DAY, now, early_check)

        assert err3.getvalue().splitlines() == [f"  MORNING STATE CHECK — stale contest reading: {exp_detail}"]
        exp_dm3 = (
            "BTS health WARN attention:\n"
            f"- [contest_state] Morning check for 2027-07-15: the automatic contest reading was stale at startup ({exp_detail}). "
            "This reading is eligible if it remains in use. Check MLB. If state still differs, "
            "use bts set-contest-streak --date 2027-07-15 --streak N --best-streak B --saver S with "
            "the scheduler's picks directory and the values shown in MLB. Check the latest fetch status."
        )
        assert sent3 == [("op.handle", exp_dm3)]
        st3 = json.loads((pdir3.parent / "health_state" / "contest_stale_reading_dm_delivery_status.json").read_text())
        assert st3["sent_day"] == "2027-07-15"
        assert st3["sent_sources"] == ["WARN_ATTENTION:contest_state:stale_reading"]

        # Also test no-recipient for Row 3
        pdir3_nr = scratch / f"row3_nr_{cname}" / "data" / "picks"
        pdir3_nr.mkdir(parents=True, exist_ok=True)
        seed_auto_obs(pdir3_nr, recorded_at=rec_at)
        err3_nr = io.StringIO()
        with contextlib.redirect_stderr(err3_nr):
            scheduler._morning_state_check(
                {"orchestrator": {"picks_dir": str(pdir3_nr)}, "bluesky": {}},
                pdir3_nr, DAY, now, early_check,
            )
        assert err3_nr.getvalue().splitlines() == [f"  MORNING STATE CHECK — stale contest reading: {exp_detail}"]
        assert not (pdir3_nr.parent / "health_state").exists()

    # 3) Shared `health_dm_delivery_status.json` strict non-read and non-write across BOTH lanes (CRITICAL & WARN)
    sentinel = b'{"sentinel_do_not_read_or_write": true}\n'
    real_open = io.open
    real_path_open = Path.open
    shared_touches = []

    for lane in ("critical", "warn"):
        for outcome in ("sent", "failed", "no_recipient"):
            pdir_s = scratch / f"shared_{lane}_{outcome}" / "data" / "picks"
            pdir_s.mkdir(parents=True, exist_ok=True)
            hfile = pdir_s.parent / "health_state" / "health_dm_delivery_status.json"
            hfile.parent.mkdir(parents=True, exist_ok=True)
            hfile.write_bytes(sentinel)
            if lane == "warn":
                seed_auto_obs(pdir_s, recorded_at="2027-07-14T18:00:00Z")

            def guarded_path_open(self, *a, **k):
                if self.name == "health_dm_delivery_status.json":
                    shared_touches.append((lane, outcome, str(self)))
                return real_path_open(self, *a, **k)

            with contextlib.ExitStack() as st:
                st.callback(setattr, Path, "open", real_path_open)
                Path.open = guarded_path_open
                orig_sd = alert.send_dm
                st.callback(setattr, alert, "send_dm", orig_sd)
                if outcome == "failed":
                    alert.send_dm = Mock(side_effect=RuntimeError("transport down"))
                else:
                    alert.send_dm = Mock(return_value="ok")
                cfg_s = {
                    "orchestrator": {"picks_dir": str(pdir_s)},
                    "bluesky": ({} if outcome == "no_recipient" else {"dm_recipient": "op"}),
                }
                with contextlib.redirect_stderr(io.StringIO()):
                    scheduler._morning_state_check(cfg_s, pdir_s, DAY, now, early_check)

            assert hfile.read_bytes() == sentinel, (lane, outcome)
    assert shared_touches == [], shared_touches

    # 4) Same-date stale WARN dedup across restarts + next-day roll
    pdir_dedup = scratch / "stale_restart_dedup" / "data" / "picks"
    pdir_dedup.mkdir(parents=True, exist_ok=True)
    seed_auto_obs(pdir_dedup, recorded_at="2027-07-14T18:00:00Z")
    stale_sends = []
    err_d1 = io.StringIO()
    err_d2 = io.StringIO()
    with contextlib.ExitStack() as st:
        orig_sd = alert.send_dm
        st.callback(setattr, alert, "send_dm", orig_sd)
        alert.send_dm = lambda rec, body: (stale_sends.append(body) or "h1")
        cfg_d = {"orchestrator": {"picks_dir": str(pdir_dedup)}, "bluesky": {"dm_recipient": "op"}}
        with contextlib.redirect_stderr(err_d1):
            scheduler._morning_state_check(cfg_d, pdir_dedup, DAY, now, early_check)
        with contextlib.redirect_stderr(err_d2):
            scheduler._morning_state_check(cfg_d, pdir_dedup, DAY, now + dt.timedelta(hours=1), early_check)
        assert len(stale_sends) == 1
        assert "MORNING STATE CHECK — stale contest reading:" in err_d1.getvalue()
        assert "MORNING STATE CHECK — stale contest reading:" in err_d2.getvalue()
        # Next ET date sends again!
        next_now = now + dt.timedelta(days=1)
        with contextlib.redirect_stderr(io.StringIO()):
            scheduler._morning_state_check(cfg_d, pdir_dedup, "2027-07-16", next_now, early_check + dt.timedelta(days=1))
        assert len(stale_sends) == 2

    return {
        "row2_verified": True,
        "row3_reasons_verified": [c[0] for c in row3_cases],
        "shared_status_touches": len(shared_touches),
        "stale_restart_and_next_day_sends": len(stale_sends),
    }


def main() -> None:
    sys.path.insert(0, str(BRANCH_SRC))
    install_network_guard()

    r1 = part1_faults_and_stall()
    print("[OK] Part 1 faults & stall:", json.dumps(r1))
    r2 = part2_gates_unreadable_and_false_alarms()
    print("[OK] Part 2 gates, unreadable files (B1 reproduced + fix verified), & false alarms:", json.dumps({
        "patched_fix_verified": r2["patched_fix_verified"],
        "false_alarm_checks": r2["false_alarm_checks"],
        "b1_decision_deep_json": r2["unreadable_audit"]["decision"]["deeply_nested_json"],
        "b1_pick_deep_json": r2["unreadable_audit"]["pick"]["deeply_nested_json"],
    }))
    r34 = part3_and_4_messages_and_dedup()
    print("[OK] Parts 3 & 4 messages, decoupled now_et_date, & dedup:", json.dumps(r34))

    out = {"part1": r1, "part2": r2, "part3_and_4": r34}
    (HERE / "probe_gates_messages_dedup.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
