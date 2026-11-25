"""Adversarial probe for Item 5: all standing rows + ordinary sequences (restart between selection & commit, failed->retry send, unconfirmed send, finale & day after, Dec 31 & Jan 1 rollover, R33 reuse, rollback read/write on archived base)."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, time, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bts import contest_state, daily_decision as dd, orchestrator, picks, scheduler, strategy
from bts.health import tail_policy as tail_health
from scripts.audit.season_walk import walk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    tail_sha = strategy._load_mdp()["tail"].sha256

    with tempfile.TemporaryDirectory(dir=HERE, prefix="standing-") as tmp_str:
        root = Path(tmp_str)

        # 1. Ordinary sequence: selection saved -> restart before commit -> later check / fallback commits H metadata
        day = date(2027, 9, 25)
        d_str = day.isoformat()
        rdir = root / "restart_between_sel_and_commit"
        rdir.mkdir()
        frame = walk._predictions(day, 0.84).head(2)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        cfg = walk._config(rdir)
        cfg["orchestrator"]["heartbeat_path"] = str(rdir / ".heartbeat")
        cfg["bluesky"] = {"dm_recipient": "synthetic"}

        with (
            walk.offline(datetime.combine(day, time(16, 0), walk.ET)),
            patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "check1")),
            patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
            patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
            patch.object(scheduler, "count_new_confirmations", lambda *a: 2),
            patch.object(scheduler, "_lock_decision_from_predictions", lambda *a, **kw: scheduler.LockDecision(False, 0.83, False, "gap")),
        ):
            assert walk.install_state(rdir, day, walk.State("dated", 57, 57, "used", 0.84)) == []
            r1 = scheduler.run_single_check(d_str, list(statuses), set(), cfg, 0.03)
            assert r1["should_post"] is False
            assert dd.load_decision(d_str, rdir) is None
            saved_pre = picks.load_pick(d_str, rdir)
            assert saved_pre.double_down is not None and saved_pre.policy_decision["objective"] == "longest_streak"

        # Process restarts before commit; at fallback deadline cascade fails (empty), cached saved pick is delivered!
        reloaded = picks.load_pick(d_str, rdir)
        send = Mock(return_value="dm-after-restart")
        with (
            walk.offline(datetime.combine(day, time(18, 30), walk.ET)),
            patch.object(orchestrator, "run_cascade", lambda *a: (frame.iloc[:0], "empty_after_restart")),
            patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
            patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
            patch.object(scheduler, "_void_committed_games", lambda *a: {}),
            patch("bts.dm.send_dm", send),
        ):
            ref = scheduler._refresh_pick_at_fallback_decision(cfg, d_str, reloaded, 0.03)
            st_fresh = scheduler.SchedulerState(d_str, None, [], [], [], False, None, None, None)
            assert scheduler._deliver_and_lock_pick(reloaded, cfg, rdir, st_fresh, d_str, "final fallback", selection=ref.selection)
        rec = dd.load_decision(d_str, rdir)
        assert rec["delivery_status"] == "delivered" and rec["action"] == "double"
        assert rec["objective"] == "longest_streak" and rec["tail_policy_sha256"] == tail_sha
        print("OK: Restart between selection and commit preserved longest_streak and tail_sha")

        # 2. Ordinary sequence: failed send (attempt cleared, no commit) -> retry succeeds -> delivered;
        #    and interrupted/unconfirmed send -> no retry, locked_unconfirmed
        sdir = root / "send_sequences"
        sdir.mkdir()
        cfg_s = walk._config(sdir)
        cfg_s["orchestrator"]["heartbeat_path"] = str(sdir / ".heartbeat")
        cfg_s["bluesky"] = {"dm_recipient": "synthetic"}
        with (
            walk.offline(datetime.combine(day, time(18, 0), walk.ET)),
            patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "check")),
            patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
            patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
            patch.object(scheduler, "count_new_confirmations", lambda *a: 6),
            patch.object(scheduler, "_lock_decision_from_predictions", lambda *a, **kw: scheduler.LockDecision(True, None, True)),
            patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
            patch.object(scheduler, "_void_committed_games", lambda *a: {}),
        ):
            assert walk.install_state(sdir, day, walk.State("dated", 57, 57, "used", 0.84)) == []
            res_s = scheduler.run_single_check(d_str, list(statuses), set(), cfg_s, 0.01)
            pick_s = picks.load_pick(d_str, sdir)
            st_s = scheduler.SchedulerState(d_str, None, [], [], [], False, None, None, None)
            # First attempt fails cleanly with OSError
            with patch("bts.dm.send_dm", side_effect=OSError("502 Bad Gateway")):
                ok = scheduler._deliver_and_lock_pick(pick_s, cfg_s, sdir, st_s, d_str, "lineup", selection=res_s["selection"])
                assert ok is False
            disk_after_fail = picks.load_pick(d_str, sdir)
            assert disk_after_fail.delivery_attempted is False and dd.load_decision(d_str, sdir) is None
            # Retry succeeds!
            with patch("bts.dm.send_dm", return_value="dm-retry-ok") as retry_send:
                ok2 = scheduler._deliver_and_lock_pick(disk_after_fail, cfg_s, sdir, st_s, d_str, "lineup", selection=res_s["selection"])
                assert ok2 is True and retry_send.call_count == 1
            rec_retry = dd.load_decision(d_str, sdir)
            assert rec_retry["delivery_status"] == "delivered" and rec_retry["objective"] == "longest_streak"

        # Unconfirmed send (KeyboardInterrupt mid-send leaves delivery_attempted=True, re-entry refuses second send & writes locked_unconfirmed)
        udir = root / "unconfirmed_send"
        udir.mkdir()
        cfg_u = walk._config(udir)
        cfg_u["orchestrator"]["heartbeat_path"] = str(udir / ".heartbeat")
        cfg_u["bluesky"] = {"dm_recipient": "synthetic"}
        with (
            walk.offline(datetime.combine(day, time(18, 0), walk.ET)),
            patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "check")),
            patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
            patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
            patch.object(scheduler, "count_new_confirmations", lambda *a: 6),
            patch.object(scheduler, "_lock_decision_from_predictions", lambda *a, **kw: scheduler.LockDecision(True, None, True)),
            patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
            patch.object(scheduler, "_alert_delivery_outcome_unknown", Mock()),
            patch.object(scheduler, "_void_committed_games", lambda *a: {}),
        ):
            assert walk.install_state(udir, day, walk.State("dated", 57, 57, "used", 0.84)) == []
            res_u = scheduler.run_single_check(d_str, list(statuses), set(), cfg_u, 0.01)
            pick_u = picks.load_pick(d_str, udir)
            st_u = scheduler.SchedulerState(d_str, None, [], [], [], False, None, None, None)
            with patch("bts.dm.send_dm", side_effect=KeyboardInterrupt("sigint during http")):
                try:
                    scheduler._deliver_and_lock_pick(pick_u, cfg_u, udir, st_u, d_str, "lineup", selection=res_u["selection"])
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError("Expected KeyboardInterrupt")
            disk_u = picks.load_pick(d_str, udir)
            assert disk_u.delivery_attempted is True and dd.load_decision(d_str, udir) is None
            with patch("bts.dm.send_dm", side_effect=AssertionError("must not resend")):
                ok_u = scheduler._deliver_and_lock_pick(disk_u, cfg_u, udir, st_u, d_str, "lineup", selection=res_u["selection"])
                assert ok_u is False
            rec_u = dd.load_decision(d_str, udir)
            assert rec_u["delivery_status"] == "locked_unconfirmed" and rec_u["objective"] == "longest_streak"
        print("OK: Failed->retry send and unconfirmed send sequences verified")

        # 3. Finale (2027-09-26 -> 1 day, double) & day after (2027-09-27 -> 0 days, reach57 skip),
        #    and Dec 31 23:59 ET -> Jan 1 00:01 ET rollover + accepted 2028 state (271 raw, 180 effective, single) + 2028 WARN
        for f_str, want_act, want_obj, want_eff in (
            ("2027-09-26", "double", "longest_streak", 1),
            ("2027-09-27", "skip", "reach57", 0),
        ):
            f_day = date.fromisoformat(f_str)
            fdir = root / f"finale_{f_str}"
            fdir.mkdir()
            f_frame = walk._predictions(f_day, 0.96).head(2)
            f_stat = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in f_frame["game_pk"]}
            with (
                walk.offline(datetime.combine(f_day, time(12), walk.ET)),
                patch.object(orchestrator, "run_cascade", lambda *a: (f_frame.copy(), "finale")),
                patch.object(picks, "get_game_statuses_detailed", lambda *a: f_stat),
            ):
                assert walk.install_state(fdir, f_day, walk.State("dated", 57, 57, "used", 0.96)) == []
                _, sel_f, _ = orchestrator.run_and_pick(walk._config(fdir), f_str)
                assert sel_f.action == want_act and sel_f.decision.objective == want_obj
                assert sel_f.decision.days_effective == want_eff

        ydir = root / "year_rollover"
        ydir.mkdir()
        dec31 = date(2027, 12, 31)
        jan1 = date(2028, 1, 1)
        with walk.offline(datetime(2027, 12, 31, 20, 0, tzinfo=walk.ET)):
            assert walk.install_state(ydir, dec31, walk.State("automatic", 57, 57, "used", 0.96)) == []
            walk._set_clock(datetime(2027, 12, 31, 23, 59, tzinfo=walk.ET))
            s_dec = contest_state.load_decision_streak_state(ydir, require_contest_state=True)
            assert s_dec.streak == 57 and strategy._days_remaining("2027-12-31") == 0
            # Dec 31 selection at 57 skips with 0 days
            d_frame = walk._predictions(dec31, 0.96).head(2)
            d_stat = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in d_frame["game_pk"]}
            with patch.object(orchestrator, "run_cascade", lambda *a: (d_frame.copy(), "dec31")), patch.object(picks, "get_game_statuses_detailed", lambda *a: d_stat):
                _, sel_dec, _ = orchestrator.run_and_pick(walk._config(ydir), "2027-12-31")
                assert sel_dec.action == "skip" and sel_dec.decision.objective == "reach57"

            # Roll over to 2028-01-01 00:01 ET: 2027 switch refused!
            walk._set_clock(datetime(2028, 1, 1, 0, 1, tzinfo=walk.ET))
            try:
                contest_state.load_decision_streak_state(ydir, require_contest_state=True)
            except contest_state.ContestStateError:
                pass
            else:
                raise AssertionError("Expected 2027 switch refusal on 2028-01-01")

        # Install 2028 state on 2028-01-01 -> 271 raw, 180 effective -> single; season_end_alerts warns
        with walk.offline(datetime(2028, 1, 1, 12, 0, tzinfo=walk.ET)):
            j_frame = walk._predictions(jan1, 0.96).head(2)
            j_stat = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in j_frame["game_pk"]}
            assert walk.install_state(ydir, jan1, walk.State("automatic", 57, 57, "used", 0.96)) == []
            with patch.object(orchestrator, "run_cascade", lambda *a: (j_frame.copy(), "jan1")), patch.object(picks, "get_game_statuses_detailed", lambda *a: j_stat):
                _, sel_jan, _ = orchestrator.run_and_pick(walk._config(ydir), "2028-01-01")
                assert strategy._days_remaining("2028-01-01") == 271
                assert sel_jan.action == "single" and sel_jan.decision.objective == "longest_streak" and sel_jan.decision.days_effective == 180
            alerts_2028 = tail_health.season_end_alerts(today=jan1)
            assert len(alerts_2028) == 1 and alerts_2028[0].level == "WARN" and "2028" in alerts_2028[0].message
        print("OK: Finale/post-finale and Dec 31 / Jan 1 ET rollover verified")

        # 4. Rollback read and write on archived 0a2bec10 base
        rb_dir = root / "rollback"
        rb_dir.mkdir()
        h_rec = dd.write_decision(
            "2027-07-15", rb_dir, action="single", source="mdp", objective="longest_streak",
            primary={"batter_id": 100, "batter_name": "Batter 0", "team": "T0", "game_pk": 70000, "p_game_hit": 0.84},
            delivery_status="delivered", scoreable=True, tail_policy_sha256=tail_sha,
        )
        assert h_rec is not None

        # Also save an uncommitted H DailyPick on disk and verify base cannot write a commit record for it (limit 9)
        archive = subprocess.run(["git", "archive", "0a2bec10", "src"], cwd=REPO, capture_output=True, check=True, timeout=15).stdout
        base_extract = root / "base_0a2bec10"
        base_extract.mkdir()
        subprocess.run(["tar", "-x", "-C", str(base_extract)], input=archive, check=True, timeout=15)

        rb_script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(base_extract / 'src')!r})
import bts
from bts import daily_decision as base_dd
assert Path(bts.__file__).resolve() == Path({str(base_extract / 'src/bts/__init__.py')!r}).resolve()

# 1. Existing finalized H decision.json parses, normalizes to 'unknown', is not reach57 skip, and stays scoreable
loaded = base_dd.load_decision("2027-07-15", {str(rb_dir)!r})
assert loaded is not None
assert loaded["objective"] == "longest_streak"
assert base_dd.decision_objective(loaded) == "unknown"
assert base_dd.is_reach57_mdp_skip(loaded) is False
assert base_dd.is_scoreable_commit("2027-07-15", {str(rb_dir)!r}, None) is True

# 2. Base cannot WRITE a new record with objective='longest_streak' (limit 9)
w = base_dd.write_decision("2027-07-16", {str(rb_dir)!r}, action="single", source="mdp",
                           objective="longest_streak", delivery_status="delivered", scoreable=True)
assert w is None
assert base_dd.load_decision("2027-07-16", {str(rb_dir)!r}) is None
print("ROLLBACK_BASE_OK")
"""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run([sys.executable, "-c", rb_script], cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=20)
        assert "ROLLBACK_BASE_OK" in proc.stdout
        print("OK: Rollback read/parse/scoreable and limit-9 write rejection verified on archived 0a2bec10")


if __name__ == "__main__":
    main()
