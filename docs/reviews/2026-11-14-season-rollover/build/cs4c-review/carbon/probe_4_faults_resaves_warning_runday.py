"""Adversarial Probe 4: Fault injection, stamp precision, grading/archive/reader parity, warning edge cases, and real run_day."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REVIEW_DIR))

from netguard import forbid_network
import probe_1_2_7_delivery_stamp_row8 as p1
import probe_3_warning_run_day_and_remaining as p3

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
SCHEMA = "bts_pick_automatic_reading_v1"
WARN_FILE = "contest_stale_reading_dm_delivery_status.json"
WARN_KEY = "WARN_ATTENTION:contest_state:stale_reading"
WARN_PREFIX = "  STALE CONTEST READING — "


class DayClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def run_fault_injection_and_stamp_precision() -> None:
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
    from bts.health import alert

    class UnprintableError(RuntimeError):
        def __str__(self):
            raise RuntimeError("cannot stringify error")

    def make_env(td: str, mode: str = "dm"):
        root = Path(td)
        picks_dir = root / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        rec = p1._seed_reading(picks_dir, "old_auto", now=NOW)
        pk = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                        0.82, [], False, 101, f"{DAY}T23:10:00Z")
        daily = picks.DailyPick(DAY, NOW.isoformat(), pk, None, None)
        pol = strategy.PolicyDecision(
            "single", "single", "heuristic", "reach57", 12, 80, None, "missing",
            None, None, None, None, None,
        )
        sel = strategy.SelectionResult(
            strategy.PickResult(daily), "single", "heuristic",
            scheduler._row_from_daily(pk), None, None, streak=12, decision=pol,
        )
        cfg = {
            "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(root / "hb"),
                             "models_dir": str(root / "models")},
            "bluesky": {"dm_recipient": "operator.test"},
            "scheduler": {"contest_state_required": True, "pick_delivery": mode,
                          "live_forward_capture_on_lock": True},
            "health_checks": {"enabled": False},
        }
        st = scheduler.SchedulerState(
            date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
            pick_locked_at=None, result_status=None, next_wakeup=None,
        )
        return root, picks_dir, rec, daily, sel, cfg, st

    fault_cases = [
        "calc_runtime", "calc_unprintable",
        "disp_runtime", "disp_unprintable",
        "transport_runtime",
        "status_is_dir", "health_parent_is_file", "health_chmod_ro",
        "status_json_list", "status_json_corrupt",
        "record_str", "record_missing_keys", "record_bad_age_type", "record_unknown_reason",
        "stderr_closed", "stderr_broken_pipe",
        "warn_clock_raises", "bluesky_cfg_none_public",
    ]
    count_faults = 0
    for mode in ("dm", "public"):
        for fcase in fault_cases:
            if fcase == "bluesky_cfg_none_public" and mode != "public":
                continue
            count_faults += 1
            with tempfile.TemporaryDirectory() as td:
                root, picks_dir, rec, daily, sel, cfg, st = make_env(td, mode=mode)
                if fcase == "bluesky_cfg_none_public":
                    cfg["bluesky"] = None
                health_dir = picks_dir.parent / "health_state"
                warn_file = health_dir / WARN_FILE
                chmod_target = None
                if fcase == "status_is_dir":
                    warn_file.mkdir(parents=True)
                elif fcase == "health_parent_is_file":
                    health_dir.write_text("not a directory")
                elif fcase == "health_chmod_ro":
                    health_dir.mkdir(parents=True)
                    health_dir.chmod(0o500)
                    chmod_target = health_dir
                elif fcase == "status_json_list":
                    health_dir.mkdir(parents=True)
                    warn_file.write_text("[]")
                elif fcase == "status_json_corrupt":
                    health_dir.mkdir(parents=True)
                    warn_file.write_text("{bad")

                captures: list[str] = []
                pick_sends: list[str] = []
                warn_sends: list[str] = []
                after_capture = False

                def on_capture(_c, d):
                    nonlocal after_capture
                    captures.append(d)
                    after_capture = True
                    if fcase == "record_str":
                        daily.automatic_reading = "corrupted"  # type: ignore[assignment]
                    elif fcase == "record_missing_keys":
                        daily.automatic_reading = {"stale": True}
                    elif fcase == "record_bad_age_type":
                        daily.automatic_reading = {
                            "stale": True, "reason": "old", "recorded_at": rec, "age_hours": "bad"
                        }
                    elif fcase == "record_unknown_reason":
                        daily.automatic_reading = {
                            "stale": True, "reason": "mystery", "recorded_at": rec, "age_hours": None
                        }

                def now_fn():
                    if fcase == "warn_clock_raises" and after_capture:
                        raise RuntimeError("warning clock failed")
                    return NOW

                err_stream: object = io.StringIO()
                if fcase == "stderr_closed":
                    closed = io.StringIO()
                    closed.close()
                    err_stream = closed
                elif fcase == "stderr_broken_pipe":
                    class BrokenStderr:
                        def write(self, _s):
                            raise BrokenPipeError("broken pipe")
                        def flush(self):
                            pass
                    err_stream = BrokenStderr()

                try:
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(contest_state, "datetime", DayClock))
                        stack.enter_context(patch.object(daily_decision, "datetime", DayClock))
                        stack.enter_context(patch.object(picks, "datetime", DayClock))
                        stack.enter_context(patch.object(alert, "datetime", DayClock))
                        stack.enter_context(patch.object(scheduler, "_now_et", now_fn))
                        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                        stack.enter_context(patch.object(
                            scheduler, "_trigger_live_forward_capture_on_lock", on_capture
                        ))
                        stack.enter_context(patch.object(
                            dm, "send_dm", lambda r, t: (pick_sends.append(t), "dm-1")[1]
                        ))
                        stack.enter_context(patch.object(
                            posting, "post_to_bluesky", lambda t: (pick_sends.append(t), "at://1")[1]
                        ))
                        if fcase == "transport_runtime":
                            stack.enter_context(patch.object(
                                alert, "send_dm", Mock(side_effect=RuntimeError("warn dm down"))
                            ))
                        else:
                            stack.enter_context(patch.object(
                                alert, "send_dm", lambda r, t: (warn_sends.append(t), "wdm-1")[1]
                            ))
                        if fcase == "calc_runtime":
                            stack.enter_context(patch.object(
                                contest_state, "automatic_reading_record",
                                Mock(side_effect=RuntimeError("calc boom")),
                            ))
                        elif fcase == "calc_unprintable":
                            stack.enter_context(patch.object(
                                contest_state, "automatic_reading_record",
                                Mock(side_effect=UnprintableError()),
                            ))
                        elif fcase == "disp_runtime":
                            stack.enter_context(patch.object(
                                alert, "dispatch_dm_for_health_alerts",
                                Mock(side_effect=RuntimeError("disp boom")),
                            ))
                        elif fcase == "disp_unprintable":
                            stack.enter_context(patch.object(
                                alert, "dispatch_dm_for_health_alerts",
                                Mock(side_effect=UnprintableError()),
                            ))
                        stack.enter_context(patch.object(sys, "stderr", err_stream))

                        ok = scheduler._deliver_and_lock_pick(
                            daily, cfg, picks_dir, st, DAY, "fault", selection=sel,
                        )
                    assert ok is True, f"{mode}:{fcase} did not return True"
                    assert len(pick_sends) == 1
                    assert captures == [DAY]
                    saved = picks.load_pick(DAY, picks_dir)
                    assert saved is not None and saved.delivery_attempted is True
                    assert picks.pick_was_delivered(saved) is True
                    if fcase in {"calc_runtime", "calc_unprintable"}:
                        assert saved.automatic_reading is None
                    else:
                        assert saved.automatic_reading is not None and saved.automatic_reading["stale"] is True
                    assert st.pick_locked is True
                    assert scheduler.load_state(DAY, picks_dir).pick_locked is True
                    dec = daily_decision.load_decision(DAY, picks_dir)
                    assert dec is not None and dec["delivery_status"] == "delivered"
                    if fcase in {"stderr_closed", "stderr_broken_pipe"}:
                        assert len(warn_sends) == 1, f"{mode}:{fcase} dispatch blocked by stderr!"
                finally:
                    if chmod_target is not None:
                        chmod_target.chmod(0o755)

    # BaseException subclasses (KeyboardInterrupt, SystemExit) inside warning vs stamp
    for bexc_cls in (KeyboardInterrupt, SystemExit):
        for mode in ("dm", "public"):
            with tempfile.TemporaryDirectory() as td:
                root, picks_dir, rec, daily, sel, cfg, st = make_env(td, mode=mode)
                captures = []
                pick_sends = []
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(contest_state, "datetime", DayClock))
                    stack.enter_context(patch.object(daily_decision, "datetime", DayClock))
                    stack.enter_context(patch.object(picks, "datetime", DayClock))
                    stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                    stack.enter_context(patch.object(
                        scheduler, "_trigger_live_forward_capture_on_lock",
                        lambda _c, d: captures.append(d),
                    ))
                    stack.enter_context(patch.object(
                        dm, "send_dm", lambda r, t: (pick_sends.append(t), "dm-be")[1]
                    ))
                    stack.enter_context(patch.object(
                        posting, "post_to_bluesky", lambda t: (pick_sends.append(t), "at://be")[1]
                    ))
                    stack.enter_context(patch.object(
                        alert, "dispatch_dm_for_health_alerts", Mock(side_effect=bexc_cls("sig")),
                    ))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    raised = False
                    try:
                        scheduler._deliver_and_lock_pick(
                            daily, cfg, picks_dir, st, DAY, "bexc_warn", selection=sel,
                        )
                    except bexc_cls:
                        raised = True
                    assert raised
                    saved = picks.load_pick(DAY, picks_dir)
                    assert saved is not None and saved.delivery_attempted is True
                    assert picks.pick_was_delivered(saved) is True
                    assert saved.automatic_reading is not None and saved.automatic_reading["stale"] is True
                    assert st.pick_locked is True
                    assert scheduler.load_state(DAY, picks_dir).pick_locked is True
                    assert captures == [DAY]
                    dec = daily_decision.load_decision(DAY, picks_dir)
                    assert dec is not None and dec["delivery_status"] == "delivered"

                # Restart recovery after BaseException during warning
                alert_dm_after = Mock(return_value="a")
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(contest_state, "datetime", DayClock))
                    stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                    stack.enter_context(patch.object(
                        scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None
                    ))
                    stack.enter_context(patch.object(dm, "send_dm", lambda r, t: pick_sends.append(t)))
                    stack.enter_context(patch.object(posting, "post_to_bluesky", lambda t: pick_sends.append(t)))
                    stack.enter_context(patch.object(alert, "send_dm", alert_dm_after))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    st2 = scheduler.SchedulerState(
                        date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                        confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                        pick_locked_at=None, result_status=None, next_wakeup=None,
                    )
                    ok2 = scheduler._deliver_and_lock_pick(
                        picks.load_pick(DAY, picks_dir), cfg, picks_dir, st2, DAY, "restart",
                    )
                    assert ok2 is True
                    assert len(pick_sends) == 1
                    assert not alert_dm_after.called

            # BaseException inside _stamp_reading (before attempt save)
            with tempfile.TemporaryDirectory() as td:
                root, picks_dir, rec, daily, sel, cfg, st = make_env(td, mode=mode)
                path = picks.save_pick(daily, picks_dir)
                raw_before = path.read_bytes()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(contest_state, "datetime", DayClock))
                    stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                    stack.enter_context(patch.object(
                        contest_state, "automatic_reading_record", Mock(side_effect=bexc_cls("pre")),
                    ))
                    raised = False
                    try:
                        scheduler._deliver_and_lock_pick(
                            daily, cfg, picks_dir, st, DAY, "bexc_stamp", selection=sel,
                        )
                    except bexc_cls:
                        raised = True
                    assert raised
                    assert path.read_bytes() == raw_before

    # Item 2 Stamp Precision: distinct _now_et() samples, single state read, exact save_pick count
    for mode in ("dm", "public"):
        with tempfile.TemporaryDirectory() as td:
            root, picks_dir, _, daily, sel, cfg, st = make_env(td, mode=mode)
            daily.pick.game_time = "2027-07-16T05:00:00Z"
            rec_time = datetime(2027, 7, 15, 7, 0, 0, tzinfo=ET)
            (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v2",
                "season": 2027, "activation_id": "a" * 32,
                "active_streak": 12, "best_streak": 15,
                "source": "mlb_bts_profile", "source_date": DAY,
                "recorded_at": rec_time.isoformat(),
            }))
            t0_init = datetime(2027, 7, 15, 20, 0, 0, tzinfo=ET)   # 13.0h (fresh)
            t1_final = datetime(2027, 7, 15, 23, 0, 0, tzinfo=ET)  # 16.0h (stale!)
            t2_deliv = datetime(2027, 7, 15, 23, 1, 0, tzinfo=ET)
            t3_lock = datetime(2027, 7, 15, 23, 2, 0, tzinfo=ET)
            t4_warn = datetime(2027, 7, 16, 0, 10, 0, tzinfo=ET)   # next ET day!
            ticks = iter([t0_init, t1_final, t2_deliv, t3_lock, t4_warn])

            real_load_ds = contest_state.load_decision_streak_state
            ds_reads = []

            def spy_load_ds(*a, **kw):
                res = real_load_ds(*a, **kw)
                ds_reads.append(1)
                (picks_dir / "account_state" / "contest_streak.json").unlink()
                return res

            real_save = picks.save_pick
            saves = []

            def spy_save(d_obj, p_dir):
                res = real_save(d_obj, p_dir)
                saves.append(json.loads(res.read_text()))
                return res

            warn_bodies = []
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(contest_state, "datetime", DayClock))
                stack.enter_context(patch.object(daily_decision, "datetime", DayClock))
                stack.enter_context(patch.object(picks, "datetime", DayClock))
                stack.enter_context(patch.object(alert, "datetime", DayClock))
                stack.enter_context(patch.object(contest_state, "load_decision_streak_state", spy_load_ds))
                stack.enter_context(patch.object(scheduler, "_now_et", lambda: next(ticks)))
                stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                stack.enter_context(patch.object(
                    scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None
                ))
                stack.enter_context(patch.object(picks, "save_pick", spy_save))
                stack.enter_context(patch.object(dm, "send_dm", lambda *_: "dm-p"))
                stack.enter_context(patch.object(posting, "post_to_bluesky", lambda *_: "at://p"))
                stack.enter_context(patch.object(
                    alert, "send_dm", lambda r, b: (warn_bodies.append(b), "w-1")[1]
                ))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

                assert scheduler._deliver_and_lock_pick(
                    daily, cfg, picks_dir, st, DAY, "precision", selection=sel,
                ) is True

            assert len(ds_reads) == 1
            assert len(saves) == 2
            assert saves[0]["delivery_attempted"] is True
            assert saves[0]["automatic_reading"]["age_hours"] == 16.0
            assert saves[0]["automatic_reading"]["reason"] == "old"
            assert saves[1]["delivered_at"] == t2_deliv.isoformat()
            assert saves[1]["automatic_reading"] == saves[0]["automatic_reading"]
            assert st.pick_locked_at == t3_lock.isoformat()
            assert len(warn_bodies) == 1
            assert "--date 2027-07-16" in warn_bodies[0]
            assert "about 16.00 hours ago" in warn_bodies[0]

    print(
        f"[PASS] Fault injection ({count_faults} non-BaseException + 8 BaseException + restart cases) "
        "and stamp precision verified."
    )


def run_item_6_real_run_day_suite() -> None:
    import pandas as pd
    from bts import contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.health import alert

    for mode_case in (
        "old_20h", "fresh_3h", "dated_obs",
        "turns_stale_at_fallback", "turns_stale_reuse_at_fallback",
        "restart_after_delivery",
    ):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdir = root / "picks"
            pdir.mkdir(parents=True)

            games = [{
                "gamePk": 101, "gameDate": f"{DAY}T23:10:00Z",  # 19:10 ET
                "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": "Away 101"}}, "home": {"team": {"name": "Home 101"}}},
            }]

            now_box = [
                datetime(2027, 7, 15, 18, 20, 0, tzinfo=ET)
                if mode_case in {"turns_stale_at_fallback", "turns_stale_reuse_at_fallback"}
                else NOW
            ]

            class SimClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

            if mode_case in {"old_20h", "restart_after_delivery"}:
                rec = p1._seed_reading(pdir, "old_auto", now=NOW, streak=30)
            elif mode_case == "fresh_3h":
                rec = p1._seed_reading(pdir, "fresh_auto", now=NOW, streak=30)
            elif mode_case == "dated_obs":
                rec = p1._seed_reading(pdir, "dated", now=NOW, streak=30)
            else:
                # Recorded at 04:26 ET:
                # - At 18:25 ET lineup check: 13h59m old (FRESH when chosen!)
                # - At 18:35 ET fallback delivery: 14h09m old (STALE at send!)
                rec = datetime(2027, 7, 15, 4, 26, 0, tzinfo=ET).isoformat()
                p1._seed_reading(pdir, "fresh_auto", now=now_box[0], streak=30)
                (pdir / "account_state" / "contest_streak.json").write_text(json.dumps({
                    "schema_version": "bts_contest_streak_auto_v2",
                    "season": 2027, "activation_id": "a" * 32,
                    "active_streak": 30, "best_streak": 30,
                    "source": "mlb_bts_profile", "source_date": DAY,
                    "recorded_at": rec,
                }))

            if mode_case in {"turns_stale_at_fallback", "turns_stale_reuse_at_fallback"}:
                proj_preds = pd.DataFrame([{
                    "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                    "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                    "p_hit_pa": 0.28, "p_game_hit": 0.99, "flags": "PROJECTED", "projected_lineup": True,
                    "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
                }])
                cascade_calls = 0

                def cascade_fn(*_a, **_kw):
                    nonlocal cascade_calls
                    cascade_calls += 1
                    if cascade_calls == 1:
                        return proj_preds, "offline"
                    if mode_case == "turns_stale_at_fallback":
                        raise RuntimeError("fallback cascade failed -> cached pick with selection=None")
                    return proj_preds, "offline"

                def fake_sleep(secs):
                    if secs > 0:
                        now_box[0] = now_box[0] + timedelta(seconds=secs)
            else:
                preds = pd.DataFrame([{
                    "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                    "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                    "p_hit_pa": 0.28, "p_game_hit": 0.99, "flags": "", "projected_lineup": False,
                    "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
                }])
                cascade_fn = lambda *_a, **_kw: (preds, "offline")
                fake_sleep = lambda _s: None

            def status_detailed_fn(*_a):
                if mode_case == "turns_stale_reuse_at_fallback" and now_box[0] >= datetime(2027, 7, 15, 18, 30, tzinfo=ET):
                    return {101: {"abstract": "P", "detailed": "Warmup", "code": "PW"}}
                return {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}

            cfg = {
                "orchestrator": {
                    "picks_dir": str(pdir),
                    "heartbeat_path": str(root / "hb"),
                    "models_dir": str(REPO_ROOT / "data" / "models"),
                },
                "bluesky": {"dm_recipient": "op.test"},
                "scheduler": {
                    "contest_state_required": True,
                    "pick_delivery": "dm",
                    "entry_intent": "enter",
                    "live_forward_capture_on_lock": False,
                },
                "health_checks": {"enabled": False},
                "tiers": [],
            }

            pick_dms = []
            warn_dms = []
            err_buf = io.StringIO()
            with patch.object(contest_state, "datetime", SimClock), patch.object(
                daily_decision, "datetime", SimClock
            ), patch.object(
                picks, "datetime", SimClock
            ), patch.object(
                alert, "datetime", SimClock
            ), patch.object(
                scheduler, "_now_et", lambda: now_box[0]
            ), patch.object(
                scheduler, "fetch_schedule", lambda d: games if d == DAY else []
            ), patch.object(
                scheduler, "check_confirmed_lineups", lambda *_: {}
            ), patch.object(
                scheduler, "_idle_until_next_wakeup", lambda *_: None
            ), patch.object(
                scheduler, "_watchdog_ping_sleep", fake_sleep
            ), patch.object(
                scheduler, "heartbeat_watchdog", lambda *a, **k: contextlib.nullcontext()
            ), patch.object(
                scheduler, "run_result_polling", lambda *a, **k: "final"
            ), patch.object(
                orchestrator, "run_cascade", cascade_fn
            ), patch.object(
                picks, "get_game_statuses_detailed", status_detailed_fn
            ), patch.object(
                strategy, "get_game_statuses_detailed", status_detailed_fn
            ), patch.object(
                picks, "get_game_statuses", lambda *_: {101: "P"}
            ), patch.object(
                dm, "send_dm", lambda r, b: (pick_dms.append((r, b)), "pick-dm-1")[1]
            ), patch.object(
                posting, "post_to_bluesky", Mock(side_effect=AssertionError("unexpected public post"))
            ), patch.object(
                alert, "send_dm", lambda r, b: (warn_dms.append((r, b)), "warn-dm-1")[1]
            ), contextlib.redirect_stderr(err_buf):
                scheduler.run_day(date=DAY, config=cfg)

                if mode_case == "restart_after_delivery":
                    err_restart = io.StringIO()
                    with contextlib.redirect_stderr(err_restart):
                        scheduler.run_day(date=DAY, config=cfg)
                    assert WARN_PREFIX not in err_restart.getvalue()

            final_pick = picks.load_pick(DAY, pdir)
            assert final_pick is not None and final_pick.notification_sent is True
            assert scheduler.load_state(DAY, pdir).pick_locked is True
            assert daily_decision.load_decision(DAY, pdir)["delivery_status"] == "delivered"
            assert len(pick_dms) == 1, f"{mode_case}: got {len(pick_dms)}"

            warn_status_file = pdir.parent / "health_state" / WARN_FILE
            if mode_case in {"old_20h", "restart_after_delivery"}:
                assert final_pick.automatic_reading == {
                    "schema": SCHEMA, "recorded_at": rec, "age_hours": 20.0, "stale": True, "reason": "old"
                }
                assert len(warn_dms) == 1
                assert WARN_PREFIX in err_buf.getvalue()
                assert warn_status_file.exists()
            elif mode_case == "fresh_3h":
                assert final_pick.automatic_reading == {
                    "schema": SCHEMA, "recorded_at": rec, "age_hours": 3.0, "stale": False, "reason": None
                }
                assert warn_dms == []
                assert WARN_PREFIX not in err_buf.getvalue()
                assert not warn_status_file.exists()
            elif mode_case == "dated_obs":
                assert final_pick.automatic_reading is None
                assert warn_dms == []
                assert WARN_PREFIX not in err_buf.getvalue()
                assert not warn_status_file.exists()
            else:
                assert final_pick.chosen_on is not None and final_pick.chosen_on["kind"] == "automatic"
                assert final_pick.automatic_reading == {
                    "schema": SCHEMA, "recorded_at": rec, "age_hours": 14.15, "stale": True, "reason": "old"
                }
                assert len(warn_dms) == 1
                assert "about 14.15 hours ago" in warn_dms[0][1]
                assert WARN_PREFIX in err_buf.getvalue()

    print("[PASS] Item 6 all 6 real run_day scenarios (old 20h, fresh 3h, dated, aged cached fallback, aged reuse fallback, restart) verified.")


def main() -> None:
    with forbid_network(allow_loopback=False):
        run_fault_injection_and_stamp_precision()
        with patch.object(
            __import__("bts.contest_state", fromlist=["datetime"]), "datetime", DayClock
        ), patch.object(
            __import__("bts.picks", fromlist=["datetime"]), "datetime", DayClock
        ):
            p3._finish_probe_2_resaves_and_readers()
        # Run Item 5 checks (the first half of p3._test_item_5_warning_and_item_6_run_day, before run_day)
        from click.testing import CliRunner
        import pandas as pd
        from bts import cli, contest_state, dm, orchestrator, picks, posting, scheduler, strategy
        from bts.health import alert
        from bts.model import predict

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for label, delta, raw_override, exp_detail_fn in [
                ("old_14_00", timedelta(hours=14, microseconds=1), None,
                 lambda r: f"last accepted fetch {r}, about 14.00 hours ago"),
                ("old_20_50", timedelta(hours=20, minutes=30), None,
                 lambda r: f"last accepted fetch {r}, about 20.50 hours ago"),
                ("old_20_12", timedelta(seconds=int(20.12 * 3600)), None,
                 lambda r: f"last accepted fetch {r}, about 20.12 hours ago"),
                ("future", timedelta(minutes=-10), None,
                 lambda r: f"its recorded_at {r} is in the future"),
                ("unreadable", None, "unparseable-time",
                 lambda _r: "it has no usable recorded_at"),
            ]:
                sub = root / label / "picks"
                sub.mkdir(parents=True)
                rec = raw_override if raw_override is not None else (NOW - delta).isoformat()
                p1._seed_reading(sub, "old_auto", now=NOW)
                (sub / "account_state" / "contest_streak.json").write_text(json.dumps({
                    "schema_version": "bts_contest_streak_auto_v2",
                    "season": 2027, "activation_id": "a" * 32,
                    "active_streak": 10, "best_streak": 15,
                    "source": "mlb_bts_profile", "source_date": DAY,
                    "recorded_at": rec,
                }))
                pk = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                                0.82, [], False, 101, f"{DAY}T23:10:00Z")
                dp = picks.DailyPick(DAY, NOW.isoformat(), pk, None, None)
                pol = strategy.PolicyDecision(
                    "single", "single", "heuristic", "reach57", 10, 80, None, "missing",
                    None, None, None, None, None,
                )
                sel = strategy.SelectionResult(strategy.PickResult(dp), "single", "heuristic",
                                               None, None, None, streak=10, decision=pol)
                cfg = {
                    "orchestrator": {"picks_dir": str(sub)},
                    "bluesky": {"dm_recipient": "op.test"},
                    "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                                  "live_forward_capture_on_lock": False},
                }
                st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                sent_warns = []
                err = io.StringIO()

                class WrongAlertClock(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        wrong = datetime(2028, 1, 1, 12, 0, tzinfo=ET)
                        return wrong.astimezone(tz) if tz else wrong.replace(tzinfo=None)

                with patch.object(contest_state, "datetime", DayClock), patch.object(
                    alert, "datetime", WrongAlertClock
                ), patch.object(
                    scheduler, "_now_et", lambda: NOW
                ), patch.object(
                    scheduler, "_delivery_game_statuses", lambda _d: {}
                ), patch.object(
                    dm, "send_dm", lambda *_: "dm-ok"
                ), patch.object(
                    alert, "send_dm", lambda r, b: (sent_warns.append((r, b)), "w-ok")[1]
                ), contextlib.redirect_stderr(err):
                    assert scheduler._deliver_and_lock_pick(dp, cfg, sub, st, DAY, "fmt", selection=sel) is True

                exp_detail = exp_detail_fn(rec)
                assert sent_warns == [("op.test", p3._env(DAY, exp_detail))]
                assert err.getvalue().splitlines().count(WARN_PREFIX + p3._msg(DAY, exp_detail)) == 1
                st_doc = json.loads((sub.parent / "health_state" / WARN_FILE).read_text())
                assert st_doc["sent_day"] == DAY
                assert st_doc["sent_sources"] == [WARN_KEY]

            # Legacy commands making FRESH choices on 20h-old stale automatic state neither stamp nor warn
            leg_dir = root / "legacy_fresh" / "picks"
            leg_dir.mkdir(parents=True)
            p1._seed_reading(leg_dir, "old_auto", now=NOW, streak=10)
            preds = pd.DataFrame([{
                "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                "p_hit_pa": 0.24, "p_game_hit": 0.99, "flags": "", "projected_lineup": False,
                "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
            }])
            toml_leg = root / "leg.toml"
            toml_leg.write_text(
                'tiers = []\n[orchestrator]\n'
                f'picks_dir = {json.dumps(str(leg_dir))}\n'
                f'models_dir = {json.dumps(str(REPO_ROOT / "data" / "models"))}\n'
                '[bluesky]\ndm_recipient = "op.test"\n'
                '[scheduler]\ncontest_state_required = true\npick_delivery = "public"\n'
            )
            for cmd in ("preview", "run", "orchestrate"):
                if (leg_dir / f"{DAY}.json").exists():
                    (leg_dir / f"{DAY}.json").unlink()
                warn_mock = Mock(return_value="w")
                arr_spy = Mock(side_effect=AssertionError("legacy command called automatic_reading_record"))
                with patch.object(contest_state, "datetime", DayClock), patch.object(
                    cli, "_now_et", lambda: NOW
                ), patch.object(
                    predict, "run_pipeline", Mock(return_value=preds)
                ), patch.object(
                    orchestrator, "run_cascade", Mock(return_value=(preds, "offline"))
                ), patch.object(
                    picks, "get_game_statuses_detailed",
                    lambda *_: {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
                ), patch.object(
                    strategy, "get_game_statuses_detailed",
                    lambda *_: {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
                ), patch.object(
                    posting, "post_to_bluesky", Mock(return_value="at://leg")
                ), patch.object(
                    posting, "should_post_now", Mock(return_value=True)
                ), patch.object(
                    alert, "send_dm", warn_mock
                ), patch.object(
                    contest_state, "automatic_reading_record", arr_spy
                ):
                    args = (
                        ["orchestrate", "--date", DAY, "--config", str(toml_leg)]
                        if cmd == "orchestrate"
                        else [cmd, "--date", DAY, "--picks-dir", str(leg_dir),
                              "--models-dir", str(REPO_ROOT / "data" / "models")]
                    )
                    r_cli = CliRunner().invoke(cli.cli, args, catch_exceptions=False)
                    assert r_cli.exit_code == 0, r_cli.output
                    assert WARN_PREFIX not in r_cli.stderr
                assert not warn_mock.called
                assert not arr_spy.called
                assert picks.load_pick(DAY, leg_dir).automatic_reading is None

        print("[PASS] Item 5 warning formatting (.2f edges, future, unreadable), decoupled clock dedup, and legacy fresh-choice controls verified.")
        run_item_6_real_run_day_suite()
    print("PROBE_4_ALL_PASSED")


if __name__ == "__main__":
    main()
