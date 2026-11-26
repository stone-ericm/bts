"""Exercise real scheduler classification and byte preservation for guarded picks."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import daily_decision, dm, picks, scheduler
from commit_lifecycle import DAY, GUARD, choose, data, deliver, identity, setup, state


class Interrupt(BaseException):
    pass


def check_record(root, status):
    record = daily_decision.load_decision(DAY, root)
    assert identity(record) == ("single", 100, status)
    assert record["policy_guard"] == GUARD and record["scoreable"] is True


if __name__ == "__main__":
    for case in ("direct_attempt", "classify_attempt", "classify_delivered", "decision_before_flag"):
        with TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            config, _ = setup(stack, root, data(.96))
            daily, sel = choose(config)
            pick_path = picks.save_pick(daily, root)
            record_path = daily_decision.decision_path(DAY, root)
            send = stack.enter_context(patch.object(dm, "send_dm", Mock()))
            if case in ("direct_attempt", "classify_attempt"):
                send.side_effect = RuntimeError("failed")
                assert not deliver(root, config, daily, sel)
                daily = picks.load_pick(DAY, root)
                assert not daily.delivery_attempted and not record_path.exists()
                send.side_effect = Interrupt("after durable marker")
                try:
                    deliver(root, config, daily, sel)
                except Interrupt:
                    assert picks.load_pick(DAY, root).delivery_attempted and not record_path.exists()
                else:
                    raise AssertionError("No interruption after durable marker")
                send.reset_mock(side_effect=True)
                if case == "direct_attempt":
                    assert not deliver(root, config, picks.load_pick(DAY, root))
                else:
                    scheduler.run_day(DAY, config)
                send.assert_not_called()
                scheduler._alert_delivery_outcome_unknown.assert_called_once()
                check_record(root, "locked_unconfirmed")
            elif case == "classify_delivered":
                daily.notification_sent = True
                daily.notification_id = "prior-dm"
                picks.save_pick(daily, root)
                assert not record_path.exists()
                scheduler.run_day(DAY, config)
                send.assert_not_called()
                check_record(root, "delivered")
            else:
                send.return_value = "confirmed"
                save = scheduler.save_state

                def interrupt_flag_save(current, directory):
                    if record_path.exists():
                        # Interrupt the save of the flag after the atomic record write.
                        raise Interrupt("decision on disk")
                    return save(current, directory)

                with patch.object(scheduler, "save_state", side_effect=interrupt_flag_save):
                    try:
                        deliver(root, config, daily, sel)
                    except Interrupt:
                        assert record_path.exists() and not scheduler.load_state(DAY, root).committed_pick_written
                    else:
                        raise AssertionError("No interruption after decision write")
                record = record_path.read_bytes()
                before_pick = pick_path.read_bytes()
                check_record(root, "delivered")
                scheduler.run_day(DAY, config)
                assert scheduler.load_state(DAY, root).committed_pick_written
                scheduler._write_endofday_skip(root, DAY, state(final_skip_candidate={
                    "source": "mdp", "policy_guard": GUARD,
                }))
                assert record_path.read_bytes() == record and pick_path.read_bytes() == before_pick
                send.assert_called_once()
            print(case, "verified")
