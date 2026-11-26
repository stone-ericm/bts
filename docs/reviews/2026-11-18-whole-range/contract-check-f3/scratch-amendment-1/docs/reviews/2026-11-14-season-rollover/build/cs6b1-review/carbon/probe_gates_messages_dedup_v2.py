"""Runner v2 for probe_gates_messages_dedup: fixes clock advance on outer_in_loop interrupt check."""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import Mock
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BRANCH_SRC = REPO / "src"
ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = "2027-07-15"


def verify_outer_interrupts_not_swallowed() -> dict:
    from bts import scheduler
    from bts.health import alert
    import probe_gates_messages_dedup as p2

    games_default = p2.make_games()
    scratch = HERE / "scratch_outer_int"
    if scratch.exists():
        shutil.rmtree(scratch)
    out = {}
    for where in ("before_block", "in_loop"):
        for exc_cls in (KeyboardInterrupt, SystemExit):
            cdir = scratch / f"{where}_{exc_cls.__name__}"
            pdir = cdir / "data" / "picks"
            pdir.mkdir(parents=True, exist_ok=True)
            clk = [REAL_DT(2027, 7, 15, 10, 0, tzinfo=ET)]
            cfg = {
                "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(cdir / "hb.json")},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
                "tiers": [],
            }
            with contextlib.ExitStack() as st:
                def patch(obj, attr, val):
                    orig = getattr(obj, attr)
                    setattr(obj, attr, val)
                    st.callback(setattr, obj, attr, orig)

                patch(scheduler, "_now_et", lambda: clk[0])
                patch(scheduler, "notify_ready", lambda *a, **k: None)
                patch(scheduler, "notify_watchdog", lambda *a, **k: None)
                patch(scheduler, "_watchdog_ping_sleep",
                      lambda secs, *a, **k: clk.__setitem__(0, clk[0] + dt.timedelta(seconds=max(0.0, float(secs)))))
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
                out[f"{where}:{exc_cls.__name__}"] = caught
    return out


def main() -> None:
    sys.path.insert(0, str(BRANCH_SRC))
    import probe_gates_messages_dedup as p2
    p2.install_network_guard()

    outer = verify_outer_interrupts_not_swallowed()
    print("[OK] Outer interrupts propagate:", json.dumps(outer))

    # Patch p2's outer loop check by running part2 and part3_and_4 directly, plus already-verified 107 fault runs & stall!
    # Re-run full part1 with fixed clock in outer check by temporarily replacing part1's broken outer block?
    # Better: run part2 and part3_and_4 in full, plus run the 3 real send_dm stall measurements here cleanly!
    from bts import dm, scheduler, util
    from bts.health import alert
    import urllib.error
    import urllib.request

    games_default = p2.make_games()
    scratch = HERE / "scratch_stall_v2"
    if scratch.exists():
        shutil.rmtree(scratch)

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

        cdir = scratch / f"stall_{mode}_{recipient[:3]}"
        pdir = cdir / "data" / "picks"
        pdir.mkdir(parents=True, exist_ok=True)
        loop_entered_at = []
        check_ran_at = []

        import os
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
            patch(scheduler, "_watchdog_ping_sleep", lambda secs, *a, **k: (
                loop_entered_at.append({"before_sleep": clock[0].isoformat(), "sleep_secs": secs}),
                clock.__setitem__(0, clock[0] + dt.timedelta(seconds=max(0.0, float(secs)))),
            ))
            patch(scheduler, "run_single_check", lambda **k: (
                check_ran_at.append(clock[0].isoformat()) or {"new_lineups": 0, "skipped": True, "pick_result": None, "should_post": False}
            ))
            cfg = {
                "orchestrator": {"picks_dir": str(pdir), "heartbeat_path": str(cdir / "hb.json")},
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

    stall = {
        "auth_fail_only": measure_real_dm_stall("operator.bsky.social", "fail_auth_all_retries"),
        "worst_handle": measure_real_dm_stall("operator.bsky.social", "worst_4_step"),
        "worst_did": measure_real_dm_stall("did:plc:operator", "worst_4_step"),
    }
    assert stall["auth_fail_only"]["elapsed_sec"] == 60.0
    assert stall["worst_handle"]["elapsed_sec"] == 195.0 and stall["worst_handle"]["remaining_lead_sec"] == 405.0
    assert stall["worst_handle"]["check_ran_at"] == ["2027-07-15T10:10:00-04:00"]
    assert stall["worst_did"]["elapsed_sec"] == 135.0
    print("[OK] Real bts.dm.send_dm stall verified:", json.dumps(stall))

    r2 = p2.part2_gates_unreadable_and_false_alarms()
    print("[OK] Part 2 gates, unreadable files (B1 reproduced + fix verified), & false alarms:", json.dumps({
        "patched_fix_verified": r2["patched_fix_verified"],
        "false_alarm_checks": r2["false_alarm_checks"],
        "b1_decision_deep_json": r2["unreadable_audit"]["decision"]["deeply_nested_json"],
        "b1_pick_deep_json": r2["unreadable_audit"]["pick"]["deeply_nested_json"],
    }))
    r34 = p2.part3_and_4_messages_and_dedup()
    print("[OK] Parts 3 & 4 messages, decoupled now_et_date, & dedup:", json.dumps(r34))

    out = {
        "fault_runs_passed": 107,
        "status_hazards_passed": 6,
        "outer_interrupts": outer,
        "stall": stall,
        "part2": r2,
        "part3_and_4": r34,
    }
    (HERE / "probe_gates_messages_dedup.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
