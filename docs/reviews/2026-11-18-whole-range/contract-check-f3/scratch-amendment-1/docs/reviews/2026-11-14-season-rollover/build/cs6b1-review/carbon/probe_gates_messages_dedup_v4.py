"""Runner v4 for unreadable files, false alarms, messages, and dedup:
Uses separate directories for the direct _morning_state_check call and the run_day call so the direct
call's persisted morning status file does not same-day-dedup the run_day call!
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BRANCH_SRC = REPO / "src"
ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = "2027-07-15"


class ReachedFirstCheck(Exception):
    pass


def run_unreadable_and_false_alarms() -> dict:
    from click.testing import CliRunner
    from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.health import alert
    import probe_gates_messages_dedup as p2

    scratch = HERE / "scratch_part2_v4"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    start = REAL_DT(2027, 7, 15, 10, 0, 0, tzinfo=ET)
    first_target = REAL_DT(2027, 7, 15, 16, 10, 0, tzinfo=ET)

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

    def seed_bad_file(pdir: Path, target_kind: str, pval) -> Path:
        fpath = (pdir / DAY / "decision.json") if target_kind == "decision" else (pdir / f"{DAY}.json")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        if pval == "DIR":
            fpath.mkdir()
        elif pval == "CHMOD000":
            fpath.write_text("{}")
            fpath.chmod(0o000)
        else:
            fpath.write_bytes(pval)
        return fpath

    for target_kind in ("decision", "pick"):
        for pname, pval in unreadable_payloads.items():
            # 1) Direct _morning_state_check
            pdir_d = scratch / f"unread_{target_kind}_{pname}_direct" / "data" / "picks"
            pdir_d.mkdir(parents=True, exist_ok=True)
            fp_d = seed_bad_file(pdir_d, target_kind, pval)
            cfg_d = {
                "orchestrator": {"picks_dir": str(pdir_d)},
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
                        scheduler._morning_state_check(cfg_d, pdir_d, DAY, start, first_target)
            except BaseException as exc:
                direct_exc = f"{type(exc).__name__}: {exc}"
            finally:
                if pval == "CHMOD000":
                    fp_d.chmod(0o644)

            # 2) Fresh directory through real run_day
            cdir_r = scratch / f"unread_{target_kind}_{pname}_runday"
            pdir_r = cdir_r / "data" / "picks"
            pdir_r.mkdir(parents=True, exist_ok=True)
            fp_r = seed_bad_file(pdir_r, target_kind, pval)
            cfg_r = {
                "orchestrator": {"picks_dir": str(pdir_r), "heartbeat_path": str(cdir_r / "hb.json")},
                "bluesky": {"dm_recipient": "operator.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
                "health_checks": {"enabled": False},
            }
            sent_runday = []
            runday_err = io.StringIO()
            reached_check = False
            clk = [start]
            try:
                with contextlib.ExitStack() as st:
                    def patch(obj, attr, val):
                        orig = getattr(obj, attr)
                        setattr(obj, attr, val)
                        st.callback(setattr, obj, attr, orig)

                    patch(scheduler, "_now_et", lambda: clk[0])
                    patch(scheduler, "notify_ready", lambda *a, **k: None)
                    patch(scheduler, "notify_watchdog", lambda *a, **k: None)
                    patch(scheduler, "fetch_schedule", lambda d: p2.make_games(d) if d == DAY else [])
                    patch(scheduler, "_watchdog_ping_sleep",
                          lambda secs, *a, **k: clk.__setitem__(0, clk[0] + dt.timedelta(seconds=max(0.0, float(secs)))))
                    patch(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None)

                    def stop_at_first_check(**k):
                        raise ReachedFirstCheck()

                    patch(scheduler, "run_single_check", stop_at_first_check)
                    patch(alert, "send_dm", lambda rec, body: (sent_runday.append(body) or "h1"))
                    try:
                        with contextlib.redirect_stderr(runday_err):
                            scheduler.run_day(DAY, cfg_r)
                    except ReachedFirstCheck:
                        reached_check = True
            finally:
                if pval == "CHMOD000":
                    fp_r.chmod(0o644)

            assert reached_check is True, (target_kind, pname)
            unreadable_results[target_kind][pname] = {
                "direct_raised": direct_exc,
                "direct_dm_sent": len(sent_direct),
                "runday_morning_dm_sent": len(sent_runday),
                "runday_reached_first_check": reached_check,
                "runday_failed_line": "  Morning state check failed:" in runday_err.getvalue(),
                "runday_stderr": runday_err.getvalue().splitlines(),
            }

    for tk in ("decision", "pick"):
        for pname, outcome in unreadable_results[tk].items():
            if pname == "deeply_nested_json":
                assert outcome["direct_raised"] and outcome["direct_raised"].startswith("RecursionError"), (tk, outcome)
                assert outcome["direct_dm_sent"] == 0 and outcome["runday_morning_dm_sent"] == 0
                assert outcome["runday_failed_line"] is True
            else:
                assert outcome["direct_raised"] is None and outcome["direct_dm_sent"] == 1 and outcome["runday_morning_dm_sent"] == 1, (tk, pname, outcome)

    # Scratch-patched verification of minimal fix
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
                "import io, sys, contextlib; from pathlib import Path; from datetime import datetime; from zoneinfo import ZoneInfo; "
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

    # False-alarm checks across ordinary and special days
    def run_fa(name: str, setup_fn) -> dict:
        cdir = scratch / name
        picks_dir = cdir / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        now_box = [start]

        class Clock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return now_box[0].astimezone(tz) if tz is not None else now_box[0].replace(tzinfo=None)

        health_dms = []
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
            patch(scheduler, "_watchdog_ping_sleep",
                  lambda secs, *a, **k: now_box.__setitem__(0, now_box[0] + dt.timedelta(seconds=max(0.0, float(secs)))))
            patch(scheduler, "check_confirmed_lineups", lambda *_: {})
            patch(scheduler, "run_result_polling", lambda *a, **k: "final")
            patch(orchestrator, "run_cascade", lambda *a, **k: (None, None))
            patch(scheduler, "fetch_schedule", lambda d: p2.make_games(d) if d == DAY else [])
            patch(alert, "send_dm", lambda rec, body: (health_dms.append((now_box[0].isoformat(), rec, body)) or "h1"))
            patch(dm, "send_dm", lambda rec, body: "p1")
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
        return {"health_dms": health_dms, "stderr": err_buf.getvalue()}

    fa = {}

    def _fa_0700(cfg, pdir, nb, patch):
        nb[0] = REAL_DT(2027, 7, 15, 7, 0, tzinfo=ET)
        res = CliRunner().invoke(cli.cli, [
            "set-contest-streak", "--date", DAY, "--streak", "12", "--best-streak", "15",
            "--saver", "active", "--picks-dir", str(pdir),
        ], catch_exceptions=False)
        assert res.exit_code == 0, res.output
        nb[0] = start
        scheduler.run_day(DAY, cfg)

    r = run_fa("fa_0700", _fa_0700)
    assert r["health_dms"] == [] and "  Morning state check: dated contest state accepted (streak 12)." in r["stderr"]
    fa["dated_0700_to_1000"] = "0_dms_accepted"

    r = run_fa("fa_yest_dated_plus_auto", lambda cfg, pdir, nb, patch: (
        p2.seed_dated_obs(pdir, day="2027-07-14", streak=11, best=15),
        p2.seed_auto_obs(pdir, streak=12, best=15, source_date="2027-07-14", recorded_at="2027-07-15T06:10:00Z"),
        scheduler.run_day(DAY, cfg),
    ))
    assert r["health_dms"] == [] and "  Morning state check: automatic contest state accepted (streak 12)." in r["stderr"]
    fa["yest_dated_plus_fresh_auto"] = "0_dms_accepted_auto"

    r = run_fa("fa_yest_dated_only", lambda cfg, pdir, nb, patch: (
        p2.seed_dated_obs(pdir, day="2027-07-14", streak=11, best=15),
        scheduler.run_day(DAY, cfg),
    ))
    assert len(r["health_dms"]) == 1
    fa["yest_dated_alone"] = "1_true_critical_dm"

    open_day = "2027-03-24"
    open_start = REAL_DT(2027, 3, 24, 10, 0, tzinfo=ET)
    r = run_fa("fa_open_ok", lambda cfg, pdir, nb, patch: (
        nb.__setitem__(0, open_start),
        patch(scheduler, "fetch_schedule", lambda d: p2.make_games(open_day, ("23:10",)) if d == open_day else []),
        p2.seed_auto_obs(pdir, season=2027, switch_season=2027, streak=0, best=0, source_date=None, recorded_at="2027-03-24T06:10:00Z"),
        scheduler.run_day(open_day, cfg),
    ))
    assert r["health_dms"] == [] and "  Morning state check: automatic contest state accepted (streak 0)." in r["stderr"]
    fa["opening_night_switched"] = "0_dms_accepted"

    r = run_fa("fa_open_2026", lambda cfg, pdir, nb, patch: (
        nb.__setitem__(0, open_start),
        patch(scheduler, "fetch_schedule", lambda d: p2.make_games(open_day, ("23:10",)) if d == open_day else []),
        p2.seed_auto_obs(pdir, season=2026, switch_season=2026, streak=5, best=10, source_date="2026-09-27", recorded_at="2026-09-28T06:10:00Z"),
        scheduler.run_day(open_day, cfg),
    ))
    assert len(r["health_dms"]) == 1 and "First planned lineup check at 16:10 ET." in r["health_dms"][0][2]
    fa["opening_night_2026_snapshot"] = "1_true_critical_dm"

    def _after_off(cfg, pdir, nb, patch):
        picks.save_pick(picks.DailyPick(
            "2027-07-13", "2027-07-13T20:00:00Z",
            picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                       0.85, [], False, 99, "2027-07-13T23:10:00Z"),
            None, None, result="hit", slot_results={"pick": "hit"},
        ), pdir)
        p2.seed_auto_obs(pdir, streak=25, best=25, source_date="2027-07-13", recorded_at="2027-07-15T06:10:00Z")
        scheduler.run_day(DAY, cfg)

    r = run_fa("fa_after_off", _after_off)
    assert r["health_dms"] == [] and "  Morning state check: automatic contest state accepted (streak 25)." in r["stderr"]
    fa["day_after_off_day"] = "0_dms_accepted"

    r1 = run_fa("fa_revoke_dated", lambda cfg, pdir, nb, patch: (
        p2.seed_auto_obs(pdir, switch_status="revoked"),
        p2.seed_dated_obs(pdir, streak=14, best=20),
        scheduler.run_day(DAY, cfg),
    ))
    assert r1["health_dms"] == [] and "dated contest state accepted (streak 14)." in r1["stderr"]
    r2 = run_fa("fa_revoke_nodated", lambda cfg, pdir, nb, patch: (
        p2.seed_auto_obs(pdir, switch_status="revoked"),
        scheduler.run_day(DAY, cfg),
    ))
    assert len(r2["health_dms"]) == 1
    fa["revoke_in_progress"] = "dated_accepted_then_revoked_critical"

    r = run_fa("fa_restored_mismatch", lambda cfg, pdir, nb, patch: (
        p2.seed_auto_obs(pdir, activation_id="b" * 32, obs_activation_id="a" * 32),
        scheduler.run_day(DAY, cfg),
    ))
    assert len(r["health_dms"]) == 1
    fa["restored_activation_mismatch"] = "1_true_critical_dm"

    return {
        "unreadable_audit": unreadable_results,
        "patched_fix_verified": fix_verify,
        "false_alarm_checks": fa,
    }


def main() -> None:
    sys.path.insert(0, str(BRANCH_SRC))
    import probe_gates_messages_dedup as p2
    p2.install_network_guard()

    r_unread_fa = run_unreadable_and_false_alarms()
    print("[OK] Unreadable audit (B1 reproduced + fix verified) & false alarms:", json.dumps({
        "patched_fix_verified": r_unread_fa["patched_fix_verified"],
        "false_alarm_checks": r_unread_fa["false_alarm_checks"],
        "b1_decision_deep_json": r_unread_fa["unreadable_audit"]["decision"]["deeply_nested_json"],
        "b1_pick_deep_json": r_unread_fa["unreadable_audit"]["pick"]["deeply_nested_json"],
    }))
    r34 = p2.part3_and_4_messages_and_dedup()
    print("[OK] Parts 3 & 4 messages, decoupled now_et_date, & dedup:", json.dumps(r34))

    (HERE / "probe_gates_messages_dedup.json").write_text(json.dumps({
        "unreadable_and_false_alarms": r_unread_fa,
        "messages_and_dedup": r34,
    }, indent=2))


if __name__ == "__main__":
    main()
