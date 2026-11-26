"""Match executed pick slots against scoreable records and render the real journal."""
from contextlib import ExitStack, redirect_stderr
from dataclasses import asdict
from datetime import date, timedelta
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import daily_decision, dm, picks, posting, scheduler, strategy
from bts.simulate.mdp import ACTIONS, lookup_action
from scripts.audit.season_walk import walk

END = date(2027, 9, 27)
ARTIFACT = strategy._load_mdp()
BOUNDS = ARTIFACT["boundaries"]
TABLE = ARTIFACT["policy_table"]
P = (BOUNDS[0] + BOUNDS[1]) / 2


def selection(root, d, s, saver, p, allow=True, different=True):
    day = END - timedelta(days=d)
    frame = walk._predictions(day, p)
    if not different:
        frame["game_pk"] = frame["game_pk"].iloc[0]
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
    return strategy.select_pick(frame, day.isoformat(), root, streak=s,
                                saver_available=saver, allow_double=allow,
                                game_statuses_detailed=statuses, require_detailed_statuses=True)


def check_journal(sel, s, d, saver, p, expected_action, expected_policy_action):
    buffer = StringIO()
    with redirect_stderr(buffer):
        scheduler._log_policy_decision(sel)
    decision = sel.decision
    assert decision.action == expected_action and decision.policy_action == expected_policy_action
    if decision.guard is None:
        assert buffer.getvalue() == ""
        return
    q = sum(p >= edge for edge in BOUNDS)
    v = int(saver)
    effective_days = min(d, 180)
    assert lookup_action(TABLE, BOUNDS, s, d, saver, p) == "skip"
    j = max(j for j in range(q) if TABLE[s, effective_days, v, j] in (1, 2))
    assert ACTIONS[TABLE[s, effective_days, v, j]] == expected_policy_action
    assert (decision.guard["bin"], decision.guard["from_bin"]) == (q, j)
    availability = "available" if v == 1 else "unavailable"
    line = (f"  POLICY GUARD — At streak {s} with {effective_days} policy days left and saver "
            f"input {availability}, the table skips bin {q}. Bin {j} supplies "
            f"{expected_policy_action}. Selected action is {expected_action}.\n")
    assert buffer.getvalue() == line
    print(line.strip())


def commit(root, sel, mode, recovered=False):
    daily = sel.pick_result.daily
    day = daily.date
    now = walk.REAL_DATETIME.fromisoformat(day + "T17:00:00").replace(tzinfo=walk.ET)
    state = scheduler.SchedulerState(date=day, schedule_fetched_at=now.isoformat(), games=[],
                                     confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                                     pick_locked_at=None, result_status=None, next_wakeup=None)
    config = walk._config(root)
    config["scheduler"].update(pick_delivery=mode, live_forward_capture_on_lock=False)
    config["bluesky"] = {"dm_recipient": "capture"}
    config["orchestrator"]["require_contest_state"] = False
    config["contest_state"] = {"required": False}
    sent = Mock(return_value="captured")
    path = picks.save_pick(daily, root)
    if recovered:
        daily = picks.load_pick(day, root)
    with ExitStack() as stack:
        stack.enter_context(walk.offline(now))
        stack.enter_context(patch.object(scheduler, "_now_et", return_value=now))
        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", return_value={}))
        stack.enter_context(patch.object(scheduler, "_trigger_live_forward_capture_on_lock"))
        stack.enter_context(patch.object(scheduler, "_contest_state_required", return_value=False))
        stack.enter_context(patch.object(dm, "send_dm", sent))
        stack.enter_context(patch.object(posting, "post_to_bluesky", sent))
        assert scheduler._deliver_and_lock_pick(daily, config, root, state, day, "probe",
                                                selection=None if recovered else sel)
    assert state.committed_pick_written
    payload = json.loads(path.read_bytes())
    record = daily_decision.load_decision(day, root)
    expected_status = "private_locked" if mode == "private" else "delivered"
    assert record["delivery_status"] == expected_status
    assert record["action"] == sel.decision.action and record["scoreable"] is True
    guard = sel.decision.guard
    assert json.loads(json.dumps(record))["policy_guard"] == guard
    assert json.loads(json.dumps(asdict(daily)))["policy_decision"]["guard"] == guard
    assert payload["policy_decision"]["guard"] == guard
    assert record["primary"]["batter_id"] == payload["pick"]["batter_id"]
    sent.assert_not_called() if mode == "private" else sent.assert_called_once()
    if mode != "private":
        text = sent.call_args.args[-1]
        assert daily.pick.batter_name in text
        if daily.double_down:
            assert daily.double_down.batter_name in text
        elif daily.runner_up:
            assert daily.runner_up["batter_name"] not in text
    actual_second = (record["double_down"] or {}).get("batter_id")
    expected_second = (payload["double_down"] or {}).get("batter_id")
    print(mode, "restart" if recovered else "fresh", day, sel.decision.policy_action,
          record["action"], "saved_second", expected_second, "record_second", actual_second)
    return actual_second, expected_second


if __name__ == "__main__":
    wrong = []
    with TemporaryDirectory() as temp:
        root = Path(temp)
        for allow, different, action in [(True, True, "double"), (False, True, "single"),
                                          (True, False, "single")]:
            sel = selection(root, 29, 1, False, P, allow, different)
            assert sel.pick_result.daily.pick.batter_id == 100
            assert bool(sel.pick_result.daily.double_down) == (action == "double")
            check_journal(sel, 1, 29, False, P, action, "double")
            if action == "double":
                ordinary = TABLE.copy()
                ordinary[1, 29, 0, 1] = 2
                with patch.object(strategy, "_load_mdp", return_value=ARTIFACT | {"policy_table": ordinary}):
                    raw = selection(root, 29, 1, False, P)
                assert raw.decision.guard is None
                assert raw.pick_result.daily.pick == sel.pick_result.daily.pick
                assert raw.pick_result.daily.double_down == sel.pick_result.daily.double_down
            for mode in ("dm", "public", "private"):
                for recovered in (False, True):
                    test_root = root / f"{allow}-{different}-{mode}-{recovered}"
                    fresh_sel = selection(test_root, 29, 1, False, P, allow, different)
                    observed, expected = commit(test_root, fresh_sel, mode, recovered)
                    if observed != expected:
                        wrong.append((allow, different, mode, recovered, observed, expected))
        for s, d, v, p in [(50, 187, False, .96), (50, 187, 0, .96),
                            (50, 187, "0", .96), (42, 38, True, .96),
                            (42, 38, 1, .96), (50, 187, False, .83)]:
            sel = selection(root, d, s, v, p)
            check_journal(sel, s, d, v, p, "single", "single")
            if (s, d, v, p) == (50, 187, False, .96):
                observed, expected = commit(root / "guarded-single", sel, "dm")
                if observed != expected:
                    wrong.append(("guarded single", observed, expected))
    print("Unexpected committed legs:", wrong)
    assert not wrong, wrong
