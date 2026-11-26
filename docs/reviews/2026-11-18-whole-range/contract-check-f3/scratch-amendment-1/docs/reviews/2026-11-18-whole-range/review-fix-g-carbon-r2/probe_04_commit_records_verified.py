"""Verified adversarial probe for Item 4: commit records, Round-1 transitions, _commit_decision_for_pick, readers, and census."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
from datetime import date, datetime, timedelta
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import bts
from bts import daily_decision, dm, orchestrator, picks, posting, scheduler, skip_policy_shadow, strategy
from bts.cli import cli
from click.testing import CliRunner
from scripts.audit.boundary_shadow_census import CensusHalt, run_census
from scripts.audit.season_walk import walk

BASE_SRC = HERE / "base_f83e0d0b" / "src"
ROLLBACK_SRC = HERE / "rollback_619abda7" / "src"

DAY_SINGLE = "2027-03-24"
NOW_SINGLE = walk.REAL_DATETIME(2027, 3, 24, 18, 32, tzinfo=walk.ET)
GUARD_BIN4 = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}

DAY_DOUBLE = "2027-08-29"
NOW_DOUBLE = walk.REAL_DATETIME(2027, 8, 29, 18, 32, tzinfo=walk.ET)
GUARD_BIN1 = {"rule": "monotone_skip_v1", "bin": 1, "from_bin": 0, "table_action": "skip"}


def _make_state(day: str = DAY_SINGLE, now: datetime = NOW_SINGLE, **kwargs):
    return scheduler.SchedulerState(
        date=day,
        schedule_fetched_at=now.isoformat(),
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=False,
        pick_locked_at=None,
        result_status=None,
        next_wakeup=None,
        **kwargs,
    )


def _frame(day: str, top_p: float, second_p: float | None = None, start_id: int = 100, same_game: bool = False):
    df = walk._predictions(date.fromisoformat(day), top_p)
    df["batter_id"] = [start_id + i for i in range(len(df))]
    df["batter_name"] = [f"Batter {bid}" for bid in df["batter_id"]]
    if second_p is not None:
        df.loc[1, "p_game_hit"] = second_p
        df.loc[2, "p_game_hit"] = max(0.01, round(second_p - 0.02, 4))
    if same_game:
        df["game_pk"] = int(df.loc[0, "game_pk"])
    return df


def _setup_env(stack: ExitStack, root: Path, frame, *, day: str = DAY_SINGLE, now: datetime = NOW_SINGLE,
               streak: int = 50, best: int = 50, saver: str = "used", mode: str = "dm"):
    stack.enter_context(walk.offline(now))
    problems = walk.install_state(root, date.fromisoformat(day), walk.State("dated", streak, best, saver, 0.83))
    assert not problems, problems
    walk._set_clock(now)
    cfg = walk._config(root)
    cfg["orchestrator"]["heartbeat_path"] = str(root / "heartbeat")
    cfg["scheduler"].update(pick_delivery=mode, live_forward_capture_on_lock=False)
    cfg["bluesky"] = {"dm_recipient": "captured-dm"}
    cfg["health_checks"]["enabled"] = False
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
    stack.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *_: statuses))
    stack.enter_context(patch.object(orchestrator, "run_cascade", lambda *_: (frame, "probe")))
    stack.enter_context(patch.object(scheduler, "_now_et", lambda: now))
    stack.enter_context(patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()))
    stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda *_: statuses))
    for name in (
        "_trigger_live_forward_capture_on_lock",
        "_alert_delivery_outcome_unknown",
        "_idle_until_next_wakeup",
        "_watchdog_ping_sleep",
        "_morning_state_check",
    ):
        stack.enter_context(patch.object(scheduler, name, Mock()))
    stack.enter_context(patch.object(scheduler, "run_result_polling", Mock(return_value="hit")))
    stack.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *_: 0))
    games = [
        {
            "gamePk": int(pk),
            "gameDate": gt,
            "doubleHeader": "N",
            "status": {"detailedState": "Scheduled"},
            "teams": {s: {"team": {"name": f"{s}-{pk}"}} for s in ("home", "away")},
        }
        for pk, gt in zip(frame.game_pk, frame.game_time)
    ]
    stack.enter_context(patch.object(scheduler, "fetch_schedule", lambda *_: games))
    return cfg, statuses


def _run_side_transitions(side: str) -> dict:
    expected_prefix = BASE_SRC if side == "base" else (ROOT / "src")
    bts_file = str(Path(bts.__file__).resolve())
    assert Path(bts_file).is_relative_to(expected_prefix.resolve()), (side, bts_file, expected_prefix)

    artifact = strategy._load_mdp()
    b0, b1 = artifact["boundaries"][0], artifact["boundaries"][1]
    p_bin1 = (b0 + b1) / 2.0

    results = {"bts_file": bts_file, "unguarded": {}, "guarded": {}, "warmup": {}}
    scenarios = (
        "confirmed_after_recovery",
        "private_to_dm",
        "refused_then_repicked",
        "prior_process_missing_pick",
        "earlier_same_date_run",
        "undecodable",
    )

    for sc in scenarios:
        with TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            frame = _frame(DAY_SINGLE, 0.83)
            cfg, _ = _setup_env(stack, root, frame)
            _, sel_a, _ = orchestrator.run_and_pick(cfg, DAY_SINGLE)
            daily_a = sel_a.pick_result.daily
            st = _make_state()
            dec_path = daily_decision.decision_path(DAY_SINGLE, root)
            send = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-ok")))

            def deliver(pick_obj, sel_obj=None):
                return scheduler._deliver_and_lock_pick(
                    pick_obj, cfg, root, st, DAY_SINGLE, "probe", selection=sel_obj
                )

            expected_bid = 100
            if sc == "confirmed_after_recovery":
                def on_send(*_):
                    pending = picks.load_pick(DAY_SINGLE, root)
                    assert pending.delivery_attempted and not picks.pick_was_delivered(pending)
                    assert deliver(pending) is False
                    mid = daily_decision.load_decision(DAY_SINGLE, root)
                    assert mid["delivery_status"] == "locked_unconfirmed"
                    return "dm-confirmed"

                send.side_effect = on_send
                assert deliver(daily_a, sel_a) is True
            elif sc == "undecodable":
                dec_path.parent.mkdir(parents=True, exist_ok=True)
                dec_path.write_bytes(b"\xff")
                assert deliver(daily_a, sel_a) is True
            else:
                cfg["scheduler"]["pick_delivery"] = "private"
                assert deliver(daily_a, sel_a) is True
                assert daily_decision.load_decision(DAY_SINGLE, root)["delivery_status"] == "private_locked"
                cfg["scheduler"]["pick_delivery"] = "dm"
                if sc == "private_to_dm":
                    assert deliver(picks.load_pick(DAY_SINGLE, root)) is True
                else:
                    if sc == "refused_then_repicked":
                        from bts.health import alert
                        with patch.object(alert, "dispatch_dm_for_health_alerts", Mock()):
                            stale = picks.load_pick(DAY_SINGLE, root)
                            stale.pick.game_time = (NOW_SINGLE - timedelta(minutes=10)).isoformat()
                            picks.save_pick(stale, root)
                            assert deliver(stale) is False
                    else:
                        (root / f"{DAY_SINGLE}.json").rename(root / "moved_aside.json")
                    assert picks.load_pick(DAY_SINGLE, root) is None
                    if sc == "earlier_same_date_run":
                        (dec_path.parent / "scheduler_state.json").unlink()
                    frame["batter_id"] = [200, 201, 202]
                    frame["batter_name"] = ["Batter 200", "Batter 201", "Batter 202"]
                    scheduler.run_day(DAY_SINGLE, cfg)
                    expected_bid = 200

            send.assert_called_once()
            saved = picks.load_pick(DAY_SINGLE, root)
            assert saved.pick.batter_id == expected_bid and picks.pick_was_delivered(saved)
            rec = daily_decision.load_decision(DAY_SINGLE, root)
            assert rec is not None and rec["scoreable"] is True
            results["unguarded"][sc] = {
                "action": rec["action"],
                "primary_bid": (rec["primary"] or {}).get("batter_id"),
                "double_bid": (rec["double_down"] or {}).get("batter_id"),
                "delivery_status": rec["delivery_status"],
                "scoreable": rec["scoreable"],
                "committed_flag": bool(st.committed_pick_written) if sc in ("confirmed_after_recovery", "private_to_dm", "undecodable") else True,
                "has_guard_key": "policy_guard" in rec,
                "policy_guard": rec.get("policy_guard"),
            }

    # Compare unguarded Warmup single & double between base and head
    for kind, day, now, streak, best, top_p, sec_p in (
        ("single", DAY_SINGLE, NOW_SINGLE, 50, 50, 0.83, None),
        ("double", DAY_DOUBLE, NOW_DOUBLE, 1, 1, b0 - 0.01, b0 - 0.01),
    ):
        with TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            frame = _frame(day, top_p, second_p=sec_p)
            cfg, statuses = _setup_env(stack, root, frame, day=day, now=now, streak=streak, best=best)
            _, sel, _ = orchestrator.run_and_pick(cfg, day)
            daily = sel.pick_result.daily
            assert (daily.double_down is not None) == (kind == "double")
            picks.save_pick(daily, root)
            for pk in statuses:
                statuses[pk] = {"abstract": "L", "detailed": "Warmup", "code": "PW"}
            refresh = scheduler._refresh_pick_at_fallback_decision(cfg, day, daily, 0.03)
            assert refresh.selection is not None and refresh.selection.decision is None
            stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-warmup")))
            st = _make_state(day=day, now=now)
            assert scheduler._deliver_and_lock_pick(
                refresh.daily, cfg, root, st, day, "fallback", selection=refresh.selection
            )
            rec = daily_decision.load_decision(day, root)
            results["warmup"][kind] = {
                "action": rec["action"],
                "primary_bid": (rec["primary"] or {}).get("batter_id"),
                "double_bid": (rec["double_down"] or {}).get("batter_id"),
                "streak": rec["streak"],
                "delivery_status": rec["delivery_status"],
            }

    if side == "head":
        # Run all 6 transitions on guarded single AND guarded double
        for label, day, now, streak, best, top_p, sec_p, exp_act, exp_guard in (
            ("guarded_single", DAY_SINGLE, NOW_SINGLE, 50, 50, 0.96, None, "single", GUARD_BIN4),
            ("guarded_double", DAY_DOUBLE, NOW_DOUBLE, 1, 1, p_bin1, 0.90, "double", GUARD_BIN1),
        ):
            results["guarded"][label] = {}
            for sc in scenarios:
                with TemporaryDirectory() as d, ExitStack() as stack:
                    root = Path(d)
                    frame = _frame(day, top_p, second_p=sec_p)
                    cfg, _ = _setup_env(stack, root, frame, day=day, now=now, streak=streak, best=best)
                    _, sel_a, _ = orchestrator.run_and_pick(cfg, day)
                    daily_a = sel_a.pick_result.daily
                    assert sel_a.decision.guard == exp_guard
                    st = _make_state(day=day, now=now)
                    dec_path = daily_decision.decision_path(day, root)
                    send = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-ok")))

                    def deliver_g(pick_obj, sel_obj=None):
                        return scheduler._deliver_and_lock_pick(
                            pick_obj, cfg, root, st, day, "probe", selection=sel_obj
                        )

                    expected_bid = 100
                    expected_runner_or_dd = None if sc == "private_to_dm" and exp_act == "single" else 101
                    if sc == "confirmed_after_recovery":
                        def on_send_g(*_):
                            pending = picks.load_pick(day, root)
                            assert deliver_g(pending) is False
                            mid = daily_decision.load_decision(day, root)
                            assert mid["delivery_status"] == "locked_unconfirmed"
                            assert mid["policy_guard"] == exp_guard
                            return "dm-confirmed"

                        send.side_effect = on_send_g
                        assert deliver_g(daily_a, sel_a) is True
                    elif sc == "undecodable":
                        dec_path.parent.mkdir(parents=True, exist_ok=True)
                        dec_path.write_bytes(b"\xff")
                        assert deliver_g(daily_a, sel_a) is True
                    else:
                        cfg["scheduler"]["pick_delivery"] = "private"
                        assert deliver_g(daily_a, sel_a) is True
                        cfg["scheduler"]["pick_delivery"] = "dm"
                        if sc == "private_to_dm":
                            assert deliver_g(picks.load_pick(day, root)) is True
                        else:
                            if sc == "refused_then_repicked":
                                from bts.health import alert
                                with patch.object(alert, "dispatch_dm_for_health_alerts", Mock()):
                                    stale = picks.load_pick(day, root)
                                    stale.pick.game_time = (now - timedelta(minutes=10)).isoformat()
                                    picks.save_pick(stale, root)
                                    assert deliver_g(stale) is False
                            else:
                                (root / f"{day}.json").rename(root / "moved_aside.json")
                            if sc == "earlier_same_date_run":
                                (dec_path.parent / "scheduler_state.json").unlink()
                            frame["batter_id"] = [200, 201, 202]
                            frame["batter_name"] = ["Batter 200", "Batter 201", "Batter 202"]
                            scheduler.run_day(day, cfg)
                            expected_bid = 200
                            expected_runner_or_dd = 201

                    saved_g = picks.load_pick(day, root)
                    assert (saved_g.double_down is not None) == (exp_act == "double")
                    rec = daily_decision.load_decision(day, root)
                    assert rec["action"] == exp_act
                    assert rec["primary"]["batter_id"] == expected_bid
                    if expected_runner_or_dd is not None:
                        assert rec["double_down"]["batter_id"] == expected_runner_or_dd
                    else:
                        assert rec["double_down"] is None
                    assert rec["delivery_status"] == "delivered"
                    assert rec["policy_guard"] == exp_guard
                    results["guarded"][label][sc] = "ok"

    return results


def _run_reader_side(side: str, shared_dir: Path) -> dict:
    expected_prefix = BASE_SRC if side == "base" else ROLLBACK_SRC
    bts_file = str(Path(bts.__file__).resolve())
    assert Path(bts_file).is_relative_to(expected_prefix.resolve()), (side, bts_file, expected_prefix)

    fixtures = [
        ("2026-03-25", "single", True, GUARD_BIN4),
        ("2026-03-26", "single", True, None),
        ("2026-03-27", "skip", False, None),
    ]
    for day, exp_action, exp_scoreable, exp_guard in fixtures:
        dec = daily_decision.load_decision(day, shared_dir)
        daily = picks.load_pick(day, shared_dir)
        assert dec is not None
        assert dec["action"] == exp_action
        assert dec["scoreable"] is exp_scoreable
        assert dec["policy_guard"] == exp_guard
        assert daily is not None and daily.pick.batter_id == 100
        if exp_action != "skip":
            assert (daily.policy_decision or {}).get("guard") == exp_guard

    if side == "base":
        out = shared_dir / "written_by_base"
        out.mkdir(parents=True, exist_ok=True)
        day = "2026-03-26"
        frame = walk._predictions(date.fromisoformat(day), 0.83)
        now = walk.REAL_DATETIME(2026, 3, 26, 12, 0, tzinfo=walk.ET)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
        with walk.offline(now):
            sel = strategy.select_pick(
                frame, day, out, streak=50, saver_available=False,
                game_statuses_detailed=statuses, require_detailed_statuses=True,
            )
            daily = sel.pick_result.daily
            picks.save_pick(daily, out)
            st = _make_state(day=day, now=now)
            scheduler._commit_decision_for_pick(
                out, day, daily, selection=sel, delivery_status="private_locked", state=st
            )
    return {"bts_file": bts_file, "ok": True}


def _verify_write_commit_decision_diff() -> None:
    base_sched = subprocess.run(
        ["git", "show", "f83e0d0b:src/bts/scheduler.py"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    head_fn = inspect.getsource(scheduler._write_commit_decision)
    normalized = (
        head_fn.replace(", policy_guard=None", "")
        .replace(", policy_guard=policy_guard", "")
    )
    assert normalized in base_sched, "scheduler._write_commit_decision differs beyond policy_guard pass-through!"


def _adversarial_commit_decision_hunt() -> dict:
    artifact = strategy._load_mdp()
    b0, b1 = artifact["boundaries"][0], artifact["boundaries"][1]
    p_bin1 = (b0 + b1) / 2.0
    summary = {}

    # 1. Fresh, cached-failed, Warmup, and NaN-predictions commits across DM, public, private for single AND double
    for kind, day, now, streak, best, top_p, sec_p, exp_guard in (
        ("guarded_single", DAY_SINGLE, NOW_SINGLE, 50, 50, 0.96, None, GUARD_BIN4),
        ("guarded_double", DAY_DOUBLE, NOW_DOUBLE, 1, 1, p_bin1, 0.90, GUARD_BIN1),
    ):
        for path_kind in ("fresh", "cached_error", "warmup", "nan_predictions"):
            for mode in ("dm", "public", "private"):
                with TemporaryDirectory() as d, ExitStack() as stack:
                    root = Path(d)
                    frame = _frame(day, top_p, second_p=sec_p)
                    cfg, statuses = _setup_env(
                        stack, root, frame, day=day, now=now, streak=streak, best=best, mode=mode
                    )
                    _, sel, _ = orchestrator.run_and_pick(cfg, day)
                    daily = sel.pick_result.daily
                    picks.save_pick(daily, root)
                    assert daily.policy_decision["guard"] == exp_guard
                    json.dumps(json.loads((root / f"{day}.json").read_text()), allow_nan=False)

                    send_dm_mock = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-1")))
                    post_mock = stack.enter_context(patch.object(posting, "post_to_bluesky", Mock(return_value="at://1")))

                    if path_kind == "cached_error":
                        stack.enter_context(
                            patch.object(orchestrator, "run_cascade", Mock(side_effect=RuntimeError("tier down")))
                        )
                        ref = scheduler._refresh_pick_at_fallback_decision(cfg, day, picks.load_pick(day, root), 0.03)
                        assert ref.selection is None
                        commit_daily, commit_sel = ref.daily, ref.selection
                    elif path_kind == "warmup":
                        for pk in statuses:
                            statuses[pk] = {"abstract": "L", "detailed": "Warmup", "code": "PW"}
                        ref = scheduler._refresh_pick_at_fallback_decision(cfg, day, picks.load_pick(day, root), 0.03)
                        assert ref.selection is not None and ref.selection.decision is None
                        assert ref.selection.primary_candidate is None and ref.selection.double_candidate is None
                        commit_daily, commit_sel = ref.daily, ref.selection
                    elif path_kind == "nan_predictions":
                        nan_frame = frame.copy()
                        nan_frame["p_game_hit"] = float("nan")
                        stack.enter_context(patch.object(orchestrator, "run_cascade", lambda *_: (nan_frame, "probe")))
                        ref = scheduler._refresh_pick_at_fallback_decision(cfg, day, picks.load_pick(day, root), 0.03)
                        assert ref.selection is not None and ref.selection.decision is None
                        assert ref.selection.no_pick_reason == "no_valid_predictions"
                        commit_daily, commit_sel = ref.daily, ref.selection
                    else:
                        commit_daily, commit_sel = daily, sel

                    st = _make_state(day=day, now=now)
                    assert scheduler._deliver_and_lock_pick(
                        commit_daily, cfg, root, st, day, "probe", selection=commit_sel
                    ) is True

                    rec = daily_decision.load_decision(day, root)
                    json.dumps(rec, allow_nan=False)
                    exp_status = "private_locked" if mode == "private" else "delivered"
                    exp_action = "double" if kind == "guarded_double" else "single"
                    assert rec["action"] == exp_action
                    assert rec["delivery_status"] == exp_status
                    assert rec["scoreable"] is True
                    assert rec["policy_guard"] == exp_guard
                    assert rec["primary"]["batter_id"] == 100
                    if exp_action == "double":
                        assert rec["double_down"]["batter_id"] == 101
                    else:
                        if path_kind == "fresh":
                            assert rec["double_down"]["batter_id"] == 101
                        else:
                            assert rec["double_down"] is None
                    if mode == "dm":
                        send_dm_mock.assert_called_once()
                        assert "Batter 100" in send_dm_mock.call_args.args[1]
                        if exp_action == "double":
                            assert "Batter 101" in send_dm_mock.call_args.args[1]
                    elif mode == "public":
                        post_mock.assert_called_once()
                        assert "Batter 100" in post_mock.call_args.args[0]
                    else:
                        send_dm_mock.assert_not_called()
                        post_mock.assert_not_called()

    summary["matrix_24_commits"] = "ok"

    # 2. Adversarial mismatch hunt
    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        frame = _frame(DAY_DOUBLE, p_bin1, second_p=0.90)
        cfg, _ = _setup_env(stack, root, frame, day=DAY_DOUBLE, now=NOW_DOUBLE, streak=1, best=1, mode="dm")
        _, sel, _ = orchestrator.run_and_pick(cfg, DAY_DOUBLE)
        daily_committed = sel.pick_result.daily

        other_frame = _frame(DAY_DOUBLE, p_bin1, second_p=0.90, start_id=900)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in other_frame.game_pk}
        other_sel = strategy.select_pick(
            other_frame, DAY_DOUBLE, root, streak=1, saver_available=False,
            game_statuses_detailed=statuses, require_detailed_statuses=True,
        )
        other_daily = other_sel.pick_result.daily

        # Case 2a: decision=None, pick_result points to other_daily (900), but daily_committed (100+101) is delivered
        sel_wrong_pick_result = strategy.SelectionResult(
            pick_result=strategy.PickResult(daily=other_daily, locked=False),
            action=None, source=None, primary_candidate=None, double_candidate=None, no_pick_reason=None,
        )
        stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-2a")))
        st = _make_state(day=DAY_DOUBLE, now=NOW_DOUBLE)
        assert scheduler._deliver_and_lock_pick(
            daily_committed, cfg, root, st, DAY_DOUBLE, "probe", selection=sel_wrong_pick_result
        )
        rec_2a = daily_decision.load_decision(DAY_DOUBLE, root)
        assert rec_2a["primary"]["batter_id"] == 100
        assert rec_2a["double_down"]["batter_id"] == 101
        assert rec_2a["policy_guard"] == GUARD_BIN1

        # Case 2b: decision=None, but primary_candidate=888, double_candidate=889 populated on SelectionResult
        daily_committed_2 = picks.parse_daily_pick(json.dumps(picks.asdict(sel.pick_result.daily)))
        daily_committed_2.notification_sent = False
        daily_committed_2.notification_id = None
        daily_committed_2.delivery_attempted = False
        sel_stray_cands = strategy.SelectionResult(
            pick_result=None,
            action="double",
            source="mdp",
            primary_candidate={"batter_id": 888, "batter_name": "Wrong 888", "team": "bad", "game_pk": 1, "p_game_hit": 0.5},
            double_candidate={"batter_id": 889, "batter_name": "Wrong 889", "team": "bad", "game_pk": 2, "p_game_hit": 0.5},
            no_pick_reason=None,
            decision=None,
        )
        assert scheduler._deliver_and_lock_pick(
            daily_committed_2, cfg, root, st, DAY_DOUBLE, "probe", selection=sel_stray_cands
        )
        rec_2b = daily_decision.load_decision(DAY_DOUBLE, root)
        assert rec_2b["primary"]["batter_id"] == 100
        assert rec_2b["double_down"]["batter_id"] == 101
        assert rec_2b["policy_guard"] == GUARD_BIN1

    # Case 2c: Concurrent delivery_attempted=True before fallback refresh
    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        frame = _frame(DAY_SINGLE, 0.96)
        cfg, _ = _setup_env(stack, root, frame, mode="dm")
        _, sel, _ = orchestrator.run_and_pick(cfg, DAY_SINGLE)
        daily = sel.pick_result.daily
        daily.delivery_attempted = True
        picks.save_pick(daily, root)
        ref = scheduler._refresh_pick_at_fallback_decision(cfg, DAY_SINGLE, daily, 0.03)
        assert ref.selection is not None and ref.selection.decision is None
        send_spy = stack.enter_context(patch.object(dm, "send_dm", Mock()))
        st = _make_state()
        assert scheduler._deliver_and_lock_pick(
            ref.daily, cfg, root, st, DAY_SINGLE, "fallback", selection=ref.selection
        ) is False
        send_spy.assert_not_called()
        rec_2c = daily_decision.load_decision(DAY_SINGLE, root)
        assert rec_2c["delivery_status"] == "locked_unconfirmed"
        assert rec_2c["primary"]["batter_id"] == 100
        assert rec_2c["policy_guard"] == GUARD_BIN4

    summary["mismatch_hunt"] = "ok"

    # 3. Saved double whose second leg was dropped, replaced, or partially voided
    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        frame = _frame(DAY_DOUBLE, p_bin1, second_p=0.90)
        cfg, statuses = _setup_env(stack, root, frame, day=DAY_DOUBLE, now=NOW_DOUBLE, streak=1, best=1, mode="dm")
        _, sel, _ = orchestrator.run_and_pick(cfg, DAY_DOUBLE)
        daily = sel.pick_result.daily
        assert daily.policy_decision["action"] == "double"
        daily.double_down = None
        picks.save_pick(daily, root)
        for pk in statuses:
            statuses[pk] = {"abstract": "L", "detailed": "Warmup", "code": "PW"}
        ref = scheduler._refresh_pick_at_fallback_decision(cfg, DAY_DOUBLE, daily, 0.03)
        stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-drop")))
        assert scheduler._deliver_and_lock_pick(
            ref.daily, cfg, root, _make_state(day=DAY_DOUBLE, now=NOW_DOUBLE), DAY_DOUBLE, "fallback", selection=ref.selection
        )
        rec_3a = daily_decision.load_decision(DAY_DOUBLE, root)
        assert rec_3a["action"] == "single"
        assert rec_3a["primary"]["batter_id"] == 100
        assert rec_3a["double_down"] is None
        assert rec_3a["policy_guard"] == GUARD_BIN1

    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        frame = _frame(DAY_DOUBLE, p_bin1, second_p=0.90)
        cfg, statuses = _setup_env(stack, root, frame, day=DAY_DOUBLE, now=NOW_DOUBLE, streak=1, best=1, mode="dm")
        _, sel1, _ = orchestrator.run_and_pick(cfg, DAY_DOUBLE)
        picks.save_pick(sel1.pick_result.daily, root)
        frame.loc[1, "batter_id"] = 301
        frame.loc[1, "batter_name"] = "Batter 301"
        _, sel2, _ = orchestrator.run_and_pick(cfg, DAY_DOUBLE)
        picks.save_pick(sel2.pick_result.daily, root)
        for pk in statuses:
            statuses[pk] = {"abstract": "L", "detailed": "Warmup", "code": "PW"}
        ref = scheduler._refresh_pick_at_fallback_decision(cfg, DAY_DOUBLE, picks.load_pick(DAY_DOUBLE, root), 0.03)
        send_m = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-rep")))
        assert scheduler._deliver_and_lock_pick(
            ref.daily, cfg, root, _make_state(day=DAY_DOUBLE, now=NOW_DOUBLE), DAY_DOUBLE, "fallback", selection=ref.selection
        )
        assert "Batter 301" in send_m.call_args.args[1] and "Batter 101" not in send_m.call_args.args[1]
        rec_3b = daily_decision.load_decision(DAY_DOUBLE, root)
        assert rec_3b["action"] == "double"
        assert rec_3b["primary"]["batter_id"] == 100
        assert rec_3b["double_down"]["batter_id"] == 301
        assert rec_3b["policy_guard"] == GUARD_BIN1

    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        frame = _frame(DAY_DOUBLE, p_bin1, second_p=0.90)
        cfg, statuses = _setup_env(stack, root, frame, day=DAY_DOUBLE, now=NOW_DOUBLE, streak=1, best=1, mode="dm")
        _, sel, _ = orchestrator.run_and_pick(cfg, DAY_DOUBLE)
        daily = sel.pick_result.daily
        picks.save_pick(daily, root)
        dd_pk = daily.double_down.game_pk
        statuses[dd_pk] = {"abstract": "F", "detailed": "Postponed"}
        stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="dm-void1")))
        assert scheduler._deliver_and_lock_pick(
            daily, cfg, root, _make_state(day=DAY_DOUBLE, now=NOW_DOUBLE), DAY_DOUBLE, "fallback", selection=None
        )
        rec_3c = daily_decision.load_decision(DAY_DOUBLE, root)
        assert rec_3c["action"] == "double"
        assert rec_3c["primary"]["batter_id"] == 100
        assert rec_3c["double_down"]["batter_id"] == 101
        assert rec_3c["policy_guard"] == GUARD_BIN1

    summary["double_leg_mutations"] = "ok"

    # 4. Malformed / missing slot fields & EOD skip null enforcement
    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        frame = _frame(DAY_SINGLE, 0.96)
        cfg, _ = _setup_env(stack, root, frame, mode="private")
        _, sel, _ = orchestrator.run_and_pick(cfg, DAY_SINGLE)
        daily = sel.pick_result.daily
        raw_obj = json.loads(json.dumps(picks.asdict(daily)))
        del raw_obj["pick"]["batter_id"]
        (root / f"{DAY_SINGLE}.json").write_text(json.dumps(raw_obj))
        try:
            picks.load_pick(DAY_SINGLE, root)
        except TypeError:
            pass
        else:
            raise AssertionError("Expected TypeError on missing Pick.batter_id")

        st_skip = _make_state()
        st_skip.final_skip_candidate = {
            "primary": sel.primary_candidate,
            "source": "mdp",
            "streak": 50,
            "saver_available": False,
            "objective": "reach57",
            "policy_guard": GUARD_BIN4,
        }
        scheduler._write_endofday_skip(root, DAY_SINGLE, st_skip)
        skip_rec = daily_decision.load_decision(DAY_SINGLE, root)
        assert skip_rec["action"] == "skip" and "policy_guard" in skip_rec and skip_rec["policy_guard"] is None

    summary["malformed_and_skip_null"] = "ok"
    return summary


def _cross_version_and_consumers() -> dict:
    with TemporaryDirectory() as d:
        root = Path(d)
        fixtures = [
            ("2026-03-25", 50, 0.96, "single", GUARD_BIN4),
            ("2026-03-26", 50, 0.83, "single", None),
            ("2026-03-27", 56, 0.99, "skip", None),
        ]
        day_nows = {}
        for day, streak, p, exp_action, exp_guard in fixtures:
            frame = walk._predictions(date.fromisoformat(day), p)
            now = walk.REAL_DATETIME.fromisoformat(f"{day}T12:00:00").replace(tzinfo=walk.ET)
            day_nows[day] = now
            statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
            with walk.offline(now):
                sel = strategy.select_pick(
                    frame, day, root, streak=streak, saver_available=False,
                    game_statuses_detailed=statuses, require_detailed_statuses=True,
                )
                assert sel.action == exp_action
                st = _make_state(day=day, now=now)
                if sel.pick_result is not None:
                    daily = sel.pick_result.daily
                    picks.save_pick(daily, root)
                    scheduler._commit_decision_for_pick(
                        root, day, daily, selection=sel, delivery_status="private_locked", state=st
                    )
                else:
                    picks.save_pick(
                        picks.DailyPick(date=day, run_time=now.isoformat(), pick=picks.pick_from_row(frame.iloc[0]),
                                        double_down=None, runner_up=None),
                        root,
                    )
                    st.final_skip_candidate = {
                        "primary": sel.primary_candidate,
                        "double": sel.double_candidate,
                        "source": "mdp",
                        "streak": streak,
                        "saver_available": False,
                        **scheduler._selection_decision_meta(sel),
                    }
                    scheduler._write_endofday_skip(root, day, st)

        dec_files = [daily_decision.decision_path(day, root) for day, *_ in fixtures]
        before_bytes = [p.read_bytes() for p in dec_files]

        for side, src_dir in (("base", BASE_SRC), ("rollback", ROLLBACK_SRC)):
            out = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "reader", side, str(root)],
                cwd=ROOT, check=True, capture_output=True, text=True, timeout=20,
                env=os.environ | {"PYTHONPATH": f"{src_dir}:{ROOT}"},
            )
            info = json.loads(out.stdout.strip().splitlines()[-1])
            assert info["ok"] is True

        base_out = root / "written_by_base"
        base_pick_bytes = (base_out / "2026-03-26.json").read_bytes()
        base_dec_bytes = daily_decision.decision_path("2026-03-26", base_out).read_bytes()
        loaded_base_pick = picks.load_pick("2026-03-26", base_out)
        loaded_base_dec = daily_decision.load_decision("2026-03-26", base_out)
        assert loaded_base_pick.policy_decision.get("guard") is None
        assert loaded_base_dec.get("policy_guard") is None
        assert (base_out / "2026-03-26.json").read_bytes() == base_pick_bytes
        assert daily_decision.decision_path("2026-03-26", base_out).read_bytes() == base_dec_bytes

        ledger = root / "ledger.jsonl"
        ledger.write_text(
            "\n".join(json.dumps({"recorded_at": f"{day}T00:00:00Z", "active_streak": s}) for day, s, *_ in fixtures)
        )
        census = run_census(
            picks_dir=root, ledger_path=ledger, policy_path=ROOT / "data/models/mdp_policy.npz",
            saver_active=False, start="2026-03-25", end="2026-03-27", as_of="2026-03-27",
            output=root / "census.json",
        )
        rows = {r["date"]: r for r in census["rows"]}
        assert (rows["2026-03-25"]["raw_deployed"], rows["2026-03-25"]["parity"]) == ("skip", "guarded_play_excluded")
        assert (rows["2026-03-26"]["raw_deployed"], rows["2026-03-26"]["parity"]) == ("single", "parity")
        assert (rows["2026-03-27"]["raw_deployed"], rows["2026-03-27"]["parity"]) == ("skip", "parity")
        assert census["follow_up"]["primary_n_diff"] == 0

        cfg_toml = root / "config.toml"
        cfg_toml.write_text('[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n')
        for day, _, _, exp_action, _ in fixtures:
            now = day_nows[day]
            with walk.offline(now), ExitStack() as stack:
                stack.enter_context(patch.object(scheduler, "fetch_schedule", lambda *_: [{"gamePk": 70000}]))
                slots = stack.enter_context(
                    patch.object(picks, "resolve_daily_slot_results", Mock(return_value={"pick": "hit"}))
                )
                r1 = CliRunner().invoke(
                    cli, ["day-outcome", "--date", day, "--config", str(cfg_toml), "--picks-dir", str(root)]
                )
                assert r1.exit_code == 0, r1.output
                exp_state = "deliberate_skip" if exp_action == "skip" else "research_commit"
                assert json.loads(r1.output)["state"] == exp_state

                r2 = CliRunner().invoke(
                    cli, ["check-results", "--date", day, "--picks-dir", str(root), "--allow-stale-scoring"]
                )
                assert r2.exit_code == 0, r2.output
                assert slots.call_count == int(exp_action != "skip")
                if exp_action != "skip":
                    assert picks.load_pick(day, root).result == "hit"

            dec = daily_decision.load_decision(day, root)
            assert daily_decision.is_reach57_mdp_skip(dec) is (exp_action == "skip")
            sh = skip_policy_shadow.record_skip_from_decision(day, root, now=now)
            assert (sh is not None) is (exp_action == "skip")

        assert [p.read_bytes() for p in dec_files] == before_bytes

        neg_cases = [
            ("oob_bin", {"rule": "monotone_skip_v1", "bin": 5, "from_bin": 3, "table_action": "skip"}, 0.96, "single"),
            ("raw_not_skip", {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}, 0.83, "double"),
            ("bool_bin", {"rule": "monotone_skip_v1", "bin": True, "from_bin": 0, "table_action": "skip"}, 0.96, "single"),
            ("inverted_bins", {"rule": "monotone_skip_v1", "bin": 2, "from_bin": 2, "table_action": "skip"}, 0.96, "single"),
            ("unguarded_raw_skip", None, 0.96, "single"),
        ]
        for label, bad_guard, p_val, rec_act in neg_cases:
            with TemporaryDirectory() as nd:
                np_dir = Path(nd)
                led = np_dir / "ledger.jsonl"
                led.write_text(json.dumps({"recorded_at": "2026-03-25T00:00:00Z", "active_streak": 50}))
                daily_decision.write_decision(
                    "2026-03-25", np_dir, action=rec_act, source="mdp", objective="reach57",
                    primary={"batter_id": 100, "p_game_hit": p_val},
                    double_down={"batter_id": 101, "p_game_hit": 0.90} if rec_act == "double" else None,
                    streak=50, saver_available=False, delivery_status="delivered", scoreable=True,
                    policy_guard=bad_guard, now=datetime.fromisoformat("2026-03-25T22:00:00+00:00"),
                )
                try:
                    run_census(
                        picks_dir=np_dir, ledger_path=led, policy_path=ROOT / "data/models/mdp_policy.npz",
                        saver_active=False, start="2026-03-25", end="2026-03-25", as_of="2026-03-25",
                        output=np_dir / "census.json",
                    )
                except CensusHalt:
                    pass
                else:
                    raise AssertionError(f"Census failed to halt on negative case {label}")

    return {"cross_version_and_consumers": "ok"}


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "side":
        print(json.dumps(_run_side_transitions(sys.argv[2])))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "reader":
        print(json.dumps(_run_reader_side(sys.argv[2], Path(sys.argv[3]))))
        return

    _verify_write_commit_decision_diff()

    base_proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "side", "base"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=45,
        env=os.environ | {"PYTHONPATH": f"{BASE_SRC}:{ROOT}"},
    )
    head_proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "side", "head"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=45,
        env=os.environ | {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
    )
    base_res = json.loads(base_proc.stdout.strip().splitlines()[-1])
    head_res = json.loads(head_proc.stdout.strip().splitlines()[-1])

    for sc, b_row in base_res["unguarded"].items():
        h_row = head_res["unguarded"][sc]
        for k in ("action", "primary_bid", "double_bid", "delivery_status", "scoreable", "committed_flag"):
            assert b_row[k] == h_row[k], (sc, k, b_row[k], h_row[k])
        assert b_row["has_guard_key"] is False
        assert h_row["has_guard_key"] is True and h_row["policy_guard"] is None

    assert base_res["warmup"]["single"]["primary_bid"] is None
    assert head_res["warmup"]["single"]["primary_bid"] == 100
    assert base_res["warmup"]["double"]["primary_bid"] is None and base_res["warmup"]["double"]["double_bid"] is None
    assert head_res["warmup"]["double"]["primary_bid"] == 100 and head_res["warmup"]["double"]["double_bid"] == 101
    assert base_res["warmup"]["single"]["streak"] == head_res["warmup"]["single"]["streak"] == 0

    adv = _adversarial_commit_decision_hunt()
    cons = _cross_version_and_consumers()

    report = {
        "write_commit_decision_diff": "exact_passthrough_verified",
        "base_bts_file": base_res["bts_file"],
        "head_bts_file": head_res["bts_file"],
        "unguarded_transitions_6": head_res["unguarded"],
        "guarded_transitions_12": head_res["guarded"],
        "warmup_base_vs_head": {"base": base_res["warmup"], "head": head_res["warmup"]},
        "adversarial_hunt": adv,
        "consumers": cons,
    }
    (HERE / "probe_04_commit_records.json").write_text(json.dumps(report, indent=2))
    print("PROBE_04_PASS", json.dumps({"transitions": 18, "warmup_double_checked": True}))


if __name__ == "__main__":
    main()
