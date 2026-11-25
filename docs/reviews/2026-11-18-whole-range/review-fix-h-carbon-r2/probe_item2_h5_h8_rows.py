"""Adversarial probe for Item 2 (part B): H1-H3 full grid, H5 (orchestrator + scheduler save/deliver/commit), H7, H8 (cached + real run_day restart)."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, time, timedelta
from itertools import product
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bts import daily_decision, orchestrator, picks, scheduler, strategy
from bts.simulate import tail_policy as tp
from scripts.audit.season_walk import walk


def main() -> None:
    shipped = strategy._load_mdp()
    tail_sha = shipped["tail"].sha256

    # 1. H1, H2, H3 across every raw day 1..271, streaks 57/58/60/100/10000, savers, probs, bests, clamps
    checked = 0
    for raw, streak, saver, prob in product(range(1, 272), (57, 58, 60, 100, 10000), (False, True), (0.01, 0.50, 0.80, 0.99)):
        day = (date(2028, 9, 28) - timedelta(days=raw)).isoformat()
        for allow, partner, (best, b_status, norm_status) in product(
            (True, False),
            (True, False),
            ((None, None, "missing"), (57, "trusted", "trusted"), (57, "untrusted", "untrusted")),
        ):
            if raw not in (1, 2, 3, 4, 115, 116, 180, 181, 271) and (best is not None or not allow or not partner):
                continue
            ctx = strategy.DecisionContext(prob, 0.91 if partner else None, partner, day, allow, shipped, best, b_status)
            dec = strategy.resolve_policy_decision(ctx, streak, saver)
            want_pol = "double" if raw <= 3 else "single"
            want_act = want_pol if (allow and partner) else "single"
            assert dec.policy_action == want_pol and dec.action == want_act
            assert dec.objective == "longest_streak" and dec.source == "mdp"
            assert dec.streak == streak and dec.days_effective == min(raw, 180)
            assert dec.best_supplied == best and dec.best_status == norm_status
            assert dec.effective_best is None and dec.guard is None
            assert dec.tail_sha256 == tail_sha and dec.degraded_reason is None
            assert dec.reason is None and dec.pick_bar is None
            checked += 1
    print(f"OK: H1-H3 checked {checked} PolicyDecision cases across raw days 1..271")

    with tempfile.TemporaryDirectory(prefix="wr-h2-h5-h8-") as tmp_str:
        root = Path(tmp_str)

        # 2. H5: real orchestrator.run_and_pick AND real scheduler check+save+deliver+commit
        # for both dated 57/57/used and automatic 57/57 on 2027-07-15, 09-24, 09-25, 09-26
        for kind in ("dated", "automatic"):
            for day_str, want_action in (
                ("2027-07-15", "single"),
                ("2027-09-24", "double"),
                ("2027-09-25", "double"),
                ("2027-09-26", "double"),
            ):
                day = date.fromisoformat(day_str)
                frame = walk._predictions(day, 0.84).head(2)
                statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}

                # 2a. orchestrator.run_and_pick
                orch_dir = root / f"orch_{kind}_{day_str}"
                orch_dir.mkdir()
                with (
                    walk.offline(datetime.combine(day, time(12), walk.ET)),
                    patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")),
                    patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
                ):
                    assert walk.install_state(orch_dir, day, walk.State(kind, 57, 57, "used", 0.84)) == []
                    _, sel, _ = orchestrator.run_and_pick(walk._config(orch_dir), day_str)
                    assert sel.action == want_action
                    daily = sel.pick_result.daily
                    valid_co = picks._valid_chosen_on(daily.chosen_on)
                    assert valid_co and valid_co["kind"] == kind and valid_co["required"] is True
                    assert daily.policy_decision["objective"] == "longest_streak"
                    assert daily.policy_decision["tail_sha256"] == tail_sha
                    assert (daily.double_down is not None) == (want_action == "double")

                # 2b. scheduler.run_single_check + load_pick + _deliver_and_lock_pick + load_decision
                sched_dir = root / f"sched_{kind}_{day_str}"
                sched_dir.mkdir()
                cfg = walk._config(sched_dir)
                cfg["orchestrator"]["heartbeat_path"] = str(sched_dir / ".heartbeat")
                cfg["bluesky"] = {"dm_recipient": "synthetic"}
                send = Mock(return_value="dm-id")
                with (
                    walk.offline(datetime.combine(day, time(12), walk.ET)),
                    patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")),
                    patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
                    patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
                    patch.object(scheduler, "count_new_confirmations", lambda *a: 6),
                    patch.object(scheduler, "_lock_decision_from_predictions", lambda *a, **kw: scheduler.LockDecision(True, None, True)),
                    patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
                    patch.object(scheduler, "_void_committed_games", lambda *a: {}),
                    patch("bts.dm.send_dm", send),
                ):
                    assert walk.install_state(sched_dir, day, walk.State(kind, 57, 57, "used", 0.84)) == []
                    r = scheduler.run_single_check(day_str, list(statuses), set(), cfg, 0.01)
                    saved = picks.load_pick(day_str, sched_dir)
                    assert picks._valid_chosen_on(saved.chosen_on)["kind"] == kind
                    st = scheduler.SchedulerState(day_str, None, [], [], [], False, None, None, None)
                    assert scheduler._deliver_and_lock_pick(saved, cfg, sched_dir, st, day_str, "lineup", selection=r["selection"])
                    rec = daily_decision.load_decision(day_str, sched_dir)
                    assert rec["action"] == want_action and rec["objective"] == "longest_streak"
                    assert rec["tail_policy_sha256"] == tail_sha and rec["scoreable"] is True
        print("OK: H5 verified for both orchestrator and scheduler across both state kinds and all 4 dates")

        # 3. H7: write_decision, decision_objective, is_reach57_mdp_skip
        h7_dir = root / "h7"
        h7_dir.mkdir()
        for act, sc in (("single", True), ("double", True), ("skip", False)):
            d_str = f"2027-07-1{5 + ('single', 'double', 'skip').index(act)}"
            rec = daily_decision.write_decision(
                d_str, h7_dir, action=act, source="mdp", objective="longest_streak",
                delivery_status="delivered" if sc else "not_applicable", scoreable=sc,
                tail_policy_sha256=tail_sha,
            )
            assert rec is not None
            loaded = daily_decision.load_decision(d_str, h7_dir)
            assert loaded == rec
            assert daily_decision.decision_objective(loaded) == "longest_streak"
            assert daily_decision.is_reach57_mdp_skip(loaded) is False
        print("OK: H7 verified")

        # 4. H8: cached fallback (empty refresh) AND real scheduler.run_day classification restart
        day = date(2027, 9, 26)
        day_str = day.isoformat()
        frame = walk._predictions(day, 0.84).head(2)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        schedule_games = [
            {
                "gamePk": int(row["game_pk"]),
                "gameDate": row["game_time"],
                "gameType": "R",
                "status": {"abstractGameState": "Preview", "detailedState": "Scheduled"},
                "teams": {"away": {"team": {"name": row["team"]}}, "home": {"team": {"name": row["pitcher_team"]}}},
            }
            for _, row in frame.iterrows()
        ]

        for mode in ("cached_fallback", "restart_unconfirmed_run_day", "restart_delivered_run_day"):
            mdir = root / f"h8_{mode}"
            mdir.mkdir()
            cfg = walk._config(mdir)
            cfg["orchestrator"]["heartbeat_path"] = str(mdir / ".heartbeat")
            cfg["bluesky"] = {"dm_recipient": "synthetic"}

            with (
                walk.offline(datetime.combine(day, time(18, 30), walk.ET)),
                patch.object(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe")),
                patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
                patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
                patch.object(scheduler, "count_new_confirmations", lambda *a: 6),
                patch.object(scheduler, "_lock_decision_from_predictions", lambda *a, **kw: scheduler.LockDecision(False, 0.83, False, "gap")),
            ):
                assert walk.install_state(mdir, day, walk.State("automatic", 57, 57, "used", 0.84)) == []
                scheduler.run_single_check(day_str, list(statuses), set(), cfg, 0.03)
                saved = picks.load_pick(day_str, mdir)
                assert saved.double_down is not None
                assert saved.policy_decision["objective"] == "longest_streak"

            if mode == "cached_fallback":
                send = Mock(return_value="dm-cached")
                empty_df = frame.iloc[:0]
                with (
                    walk.offline(datetime.combine(day, time(18, 40), walk.ET)),
                    patch.object(orchestrator, "run_cascade", lambda *a: (empty_df, "empty")),
                    patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
                    patch.object(scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
                    patch.object(scheduler, "_void_committed_games", lambda *a: {}),
                    patch("bts.dm.send_dm", send),
                ):
                    refresh = scheduler._refresh_pick_at_fallback_decision(cfg, day_str, saved, 0.03)
                    assert refresh.daily is saved and getattr(refresh.selection, "decision", None) is None
                    st = scheduler.SchedulerState(day_str, None, [], [], [], False, None, None, None)
                    assert scheduler._deliver_and_lock_pick(saved, cfg, mdir, st, day_str, "final fallback", selection=refresh.selection)
                rec = daily_decision.load_decision(day_str, mdir)
                assert rec["delivery_status"] == "delivered" and rec["action"] == "double"
                assert rec["objective"] == "longest_streak" and rec["tail_policy_sha256"] == tail_sha
            else:
                if mode == "restart_unconfirmed_run_day":
                    saved.delivery_attempted = True
                else:
                    saved.dm_delivered_at = "2027-09-26T22:31:00+00:00"
                picks.save_pick(saved, mdir)
                assert daily_decision.load_decision(day_str, mdir) is None
                send = Mock(return_value="should-not-send")
                with (
                    walk.offline(datetime.combine(day, time(18, 40), walk.ET)),
                    patch.object(scheduler, "fetch_schedule", lambda d: schedule_games),
                    patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses),
                    patch.object(scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
                    patch.object(scheduler, "_watchdog_ping_sleep", lambda s: None),
                    patch.object(scheduler, "_idle_until_next_wakeup", lambda *a: None),
                    patch.object(scheduler, "_next_day_wakeup", lambda d, sc: datetime(2027, 9, 27, 10, tzinfo=walk.ET)),
                    patch.object(scheduler, "run_result_polling", lambda *a, **kw: "final"),
                    patch.object(scheduler, "_alert_delivery_outcome_unknown", Mock()),
                    patch("bts.dm.send_dm", send),
                ):
                    scheduler.run_day(day_str, cfg)
                assert send.call_count == 0
                rec = daily_decision.load_decision(day_str, mdir)
                want_status = "locked_unconfirmed" if mode == "restart_unconfirmed_run_day" else "delivered"
                assert rec is not None and rec["delivery_status"] == want_status
                assert rec["action"] == "double" and rec["objective"] == "longest_streak"
                assert rec["tail_policy_sha256"] == tail_sha
        print("OK: H8 verified for cached fallback and both real scheduler.run_day classification restarts")


if __name__ == "__main__":
    main()
