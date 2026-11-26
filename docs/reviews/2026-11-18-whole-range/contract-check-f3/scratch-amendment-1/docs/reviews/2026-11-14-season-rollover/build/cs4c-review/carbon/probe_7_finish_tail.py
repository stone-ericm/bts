"""Adversarial Probe 7: Completes 5-tick stamp precision + Probe 2 resaves/readers + Item 5 warning + Item 6 run_day."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
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


def main() -> None:
    with forbid_network(allow_loopback=False):
        from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
        from bts.health import alert

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
                    stack.enter_context(patch.object(
                        scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None
                    ))
                    stack.enter_context(patch.object(scheduler, "_now_et", lambda: next(ticks, t4)))
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
        print("[PASS] 5-tick stamp precision, single state read, and save count verified in DM and public modes.")

        with patch.object(contest_state, "datetime", p5.DayClock), patch.object(picks, "datetime", p5.DayClock):
            p3._finish_probe_2_resaves_and_readers()
        p5.run_item_5_warning_checks()
        p5.run_item_6_real_run_day_suite()
    print("PROBE_7_ALL_PASSED")


if __name__ == "__main__":
    main()
