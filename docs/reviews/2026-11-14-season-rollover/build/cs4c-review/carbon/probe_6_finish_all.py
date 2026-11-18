"""Adversarial Probe 6: Runs stderr-closed-during-warning (genuine closed StringIO after LOCKED line),
plus all remaining suites from Probe 5 (faults, stamp precision, resaves/readers, Item 5 warning, Item 6 run_day).
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from datetime import datetime
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
import probe_5_complete_suite as p5

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)


class CloseAfterLockedStringIO(io.StringIO):
    """Stays open for existing delivery stderr, then genuinely closes itself before warning runs."""

    def write(self, s: str) -> int:
        n = super().write(s)
        if "LOCKED (" in s and "\n" in s:
            self.close()
        if "LOCKED (" in self.getvalue() and s == "\n" and not self.closed:
            self.close()
        return n


class BrokenPipeAfterLocked:
    """Raises BrokenPipeError on every stderr write after the existing LOCKED line completes."""

    def __init__(self) -> None:
        self.buf = ""
        self.armed = False

    def write(self, s: str) -> int:
        if self.armed:
            raise BrokenPipeError("stderr pipe closed before warning")
        self.buf += s
        if "LOCKED (" in self.buf and self.buf.endswith("\n"):
            self.armed = True
        return len(s)

    def flush(self) -> None:
        if self.armed:
            raise BrokenPipeError("stderr flush broken")


def run_stderr_and_all_remaining() -> None:
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
    from bts.health import alert

    # Test both CloseAfterLockedStringIO (genuinely closed stream -> ValueError) and BrokenPipeAfterLocked
    # plus calc_runtime + closed stderr (where stderr is closed BEFORE _stamp_reading and stays closed until LOCKED?
    # Wait: if calc_runtime happens with BrokenPipe on warning lines, test that too!)
    for mode in ("dm", "public"):
        for kind in ("closed_after_locked", "broken_pipe_after_locked", "calc_err_with_closed_diag"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                pdir = root / "picks"
                pdir.mkdir(parents=True)
                p1._seed_reading(pdir, "old_auto", now=NOW)
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
                    "orchestrator": {"picks_dir": str(pdir)},
                    "bluesky": {"dm_recipient": "operator.test"},
                    "scheduler": {"contest_state_required": True, "pick_delivery": mode,
                                  "live_forward_capture_on_lock": True},
                }
                st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                captures = []
                warn_sends = []

                if kind == "closed_after_locked":
                    stream: object = CloseAfterLockedStringIO()
                elif kind == "broken_pipe_after_locked":
                    stream = BrokenPipeAfterLocked()
                else:
                    class FailOnlyWarningDiag:
                        def write(self, s: str) -> int:
                            if "Stale-reading warning failed" in s or "STALE CONTEST READING" in s:
                                raise ValueError("I/O operation on closed file")
                            return len(s)
                        def flush(self) -> None:
                            pass
                    stream = FailOnlyWarningDiag()

                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(contest_state, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(daily_decision, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(picks, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(alert, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                    stack.enter_context(patch.object(
                        scheduler, "_trigger_live_forward_capture_on_lock",
                        lambda _c, d: captures.append(d),
                    ))
                    stack.enter_context(patch.object(dm, "send_dm", lambda *_: "dm-1"))
                    stack.enter_context(patch.object(posting, "post_to_bluesky", lambda *_: "at://1"))
                    stack.enter_context(patch.object(
                        alert, "send_dm", lambda r, t: (warn_sends.append(t), "w-1")[1]
                    ))
                    if kind == "calc_err_with_closed_diag":
                        stack.enter_context(patch.object(
                            contest_state, "automatic_reading_record",
                            Mock(side_effect=RuntimeError("calc err")),
                        ))
                    stack.enter_context(patch.object(sys, "stderr", stream))

                    ok = scheduler._deliver_and_lock_pick(
                        daily, cfg, pdir, st, DAY, "stderr_probe", selection=sel,
                    )
                assert ok is True
                if kind == "closed_after_locked":
                    assert isinstance(stream, CloseAfterLockedStringIO) and stream.closed
                assert captures == [DAY]
                saved = picks.load_pick(DAY, pdir)
                assert saved is not None and picks.pick_was_delivered(saved)
                assert st.pick_locked is True
                assert daily_decision.load_decision(DAY, pdir)["delivery_status"] == "delivered"
                if kind != "calc_err_with_closed_diag":
                    assert len(warn_sends) == 1, f"{mode}:{kind} warning DM not sent despite closed stderr!"
                else:
                    assert saved.automatic_reading is None

    print("[PASS] Closed/broken stderr during warning and diagnostic writes verified (dispatch still fires, delivery unaffected).")


def main() -> None:
    with forbid_network(allow_loopback=False):
        run_stderr_and_all_remaining()

        # Run BaseException + Stamp Precision from p5 by temporarily filtering fault_cases in p5?
        # No, let's call the clean parts directly here without touching p5!
        from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
        from bts.health import alert

        # Run warn_clock_raises and bluesky_cfg_none_public (the 2 cases after stderr_closed in p5)
        for mode, fcase in [("dm", "warn_clock_raises"), ("public", "warn_clock_raises"), ("public", "bluesky_cfg_none_public")]:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                pdir = root / "picks"
                pdir.mkdir(parents=True)
                p1._seed_reading(pdir, "old_auto", now=NOW)
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
                    "orchestrator": {"picks_dir": str(pdir)},
                    "bluesky": None if fcase == "bluesky_cfg_none_public" else {"dm_recipient": "op.test"},
                    "scheduler": {"contest_state_required": True, "pick_delivery": mode,
                                  "live_forward_capture_on_lock": True},
                }
                st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                after_cap = False
                def on_cap(_c, _d):
                    nonlocal after_cap
                    after_cap = True
                def now_fn():
                    if fcase == "warn_clock_raises" and after_cap:
                        raise RuntimeError("warn clock boom")
                    return NOW
                err = io.StringIO()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(contest_state, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(daily_decision, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(picks, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(alert, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(scheduler, "_now_et", now_fn))
                    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                    stack.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", on_cap))
                    stack.enter_context(patch.object(dm, "send_dm", lambda *_: "dm-1"))
                    stack.enter_context(patch.object(posting, "post_to_bluesky", lambda *_: "at://1"))
                    stack.enter_context(patch.object(alert, "send_dm", lambda *_: "w-1"))
                    stack.enter_context(contextlib.redirect_stderr(err))
                    assert scheduler._deliver_and_lock_pick(daily, cfg, pdir, st, DAY, fcase, selection=sel) is True
                assert "Stale-reading warning failed:" in err.getvalue()
                assert picks.pick_was_delivered(picks.load_pick(DAY, pdir))
                assert st.pick_locked and daily_decision.load_decision(DAY, pdir)["delivery_status"] == "delivered"

        # BaseException + Stamp Precision
        for bexc_cls in (KeyboardInterrupt, SystemExit):
            for mode in ("dm", "public"):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    pdir = root / "picks"
                    pdir.mkdir(parents=True)
                    p1._seed_reading(pdir, "old_auto", now=NOW)
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
                        "orchestrator": {"picks_dir": str(pdir)},
                        "bluesky": {"dm_recipient": "op.test"},
                        "scheduler": {"contest_state_required": True, "pick_delivery": mode,
                                      "live_forward_capture_on_lock": True},
                    }
                    st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                    captures = []
                    sends = []
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(contest_state, "datetime", p5.DayClock))
                        stack.enter_context(patch.object(daily_decision, "datetime", p5.DayClock))
                        stack.enter_context(patch.object(picks, "datetime", p5.DayClock))
                        stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                        stack.enter_context(patch.object(
                            scheduler, "_trigger_live_forward_capture_on_lock", lambda _c, d: captures.append(d)
                        ))
                        stack.enter_context(patch.object(dm, "send_dm", lambda r, t: (sends.append(t), "dm-1")[1]))
                        stack.enter_context(patch.object(posting, "post_to_bluesky", lambda t: (sends.append(t), "at://1")[1]))
                        stack.enter_context(patch.object(
                            alert, "dispatch_dm_for_health_alerts", Mock(side_effect=bexc_cls("sig"))
                        ))
                        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                        caught = False
                        try:
                            scheduler._deliver_and_lock_pick(daily, cfg, pdir, st, DAY, "bexc", selection=sel)
                        except bexc_cls:
                            caught = True
                        assert caught
                    saved = picks.load_pick(DAY, pdir)
                    assert saved is not None and picks.pick_was_delivered(saved)
                    assert saved.automatic_reading is not None and saved.automatic_reading["stale"] is True
                    assert st.pick_locked and scheduler.load_state(DAY, pdir).pick_locked
                    assert captures == [DAY]
                    assert daily_decision.load_decision(DAY, pdir)["delivery_status"] == "delivered"

                    # Restart recovery
                    warn_after = Mock(return_value="w")
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(contest_state, "datetime", p5.DayClock))
                        stack.enter_context(patch.object(scheduler, "_now_et", lambda: NOW))
                        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                        stack.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None))
                        stack.enter_context(patch.object(dm, "send_dm", lambda r, t: sends.append(t)))
                        stack.enter_context(patch.object(posting, "post_to_bluesky", lambda t: sends.append(t)))
                        stack.enter_context(patch.object(alert, "send_dm", warn_after))
                        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                        st2 = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                        assert scheduler._deliver_and_lock_pick(saved, cfg, pdir, st2, DAY, "restart") is True
                    assert len(sends) == 1 and not warn_after.called

        # Stamp precision
        for mode in ("dm", "public"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                pdir = root / "picks"
                pdir.mkdir(parents=True)
                p1._seed_reading(pdir, "old_auto", now=NOW)
                rec_time = datetime(2027, 7, 15, 7, 0, 0, tzinfo=ET)
                (pdir / "account_state" / "contest_streak.json").write_text(json.dumps({
                    "schema_version": "bts_contest_streak_auto_v2",
                    "season": 2027, "activation_id": "a" * 32,
                    "active_streak": 12, "best_streak": 15,
                    "source": "mlb_bts_profile", "source_date": DAY,
                    "recorded_at": rec_time.isoformat(),
                }))
                pk = picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                                0.82, [], False, 101, "2027-07-16T05:00:00Z")
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
                    "orchestrator": {"picks_dir": str(pdir)},
                    "bluesky": {"dm_recipient": "op.test"},
                    "scheduler": {"contest_state_required": True, "pick_delivery": mode,
                                  "live_forward_capture_on_lock": False},
                }
                st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                t0 = datetime(2027, 7, 15, 20, 0, 0, tzinfo=ET)   # 13.0h (fresh)
                t1 = datetime(2027, 7, 15, 23, 0, 0, tzinfo=ET)   # 16.0h (stale!)
                t2 = datetime(2027, 7, 15, 23, 1, 0, tzinfo=ET)
                t3 = datetime(2027, 7, 15, 23, 2, 0, tzinfo=ET)
                t4 = datetime(2027, 7, 16, 0, 10, 0, tzinfo=ET)   # next ET day!
                ticks = iter([t0, t1, t2, t3, t4])
                real_load_ds = contest_state.load_decision_streak_state
                reads = []
                def spy_load_ds(*a, **kw):
                    r = real_load_ds(*a, **kw)
                    reads.append(1)
                    (pdir / "account_state" / "contest_streak.json").unlink()
                    return r
                real_save = picks.save_pick
                saves = []
                def spy_save(d_obj, pd_path):
                    res = real_save(d_obj, pd_path)
                    saves.append(json.loads(res.read_text()))
                    return res
                warns = []
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(contest_state, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(daily_decision, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(picks, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(alert, "datetime", p5.DayClock))
                    stack.enter_context(patch.object(contest_state, "load_decision_streak_state", spy_load_ds))
                    stack.enter_context(patch.object(scheduler, "_now_et", lambda: next(ticks)))
                    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: {}))
                    stack.enter_context(patch.object(picks, "save_pick", spy_save))
                    stack.enter_context(patch.object(dm, "send_dm", lambda *_: "dm-p"))
                    stack.enter_context(patch.object(posting, "post_to_bluesky", lambda *_: "at://p"))
                    stack.enter_context(patch.object(alert, "send_dm", lambda r, b: (warns.append(b), "w")[1]))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    assert scheduler._deliver_and_lock_pick(daily, cfg, pdir, st, DAY, "prec", selection=sel) is True
                assert len(reads) == 1 and len(saves) == 2
                assert saves[0]["automatic_reading"]["age_hours"] == 16.0
                assert saves[1]["delivered_at"] == t2.isoformat()
                assert st.pick_locked_at == t3.isoformat()
                assert len(warns) == 1 and "--date 2027-07-16" in warns[0] and "about 16.00 hours ago" in warns[0]

        print("[PASS] BaseException isolation, restart recovery, warn_clock/config=None faults, and 5-tick stamp precision verified.")

        with patch.object(contest_state, "datetime", p5.DayClock), patch.object(picks, "datetime", p5.DayClock):
            p3._finish_probe_2_resaves_and_readers()
        p5.run_item_5_warning_checks()
        p5.run_item_6_real_run_day_suite()
    print("PROBE_6_ALL_PASSED")


if __name__ == "__main__":
    main()
