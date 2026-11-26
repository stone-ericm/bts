"""A real final fallback must record the cached guarded pick it sends after a restart."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import daily_decision, dm, orchestrator, picks, scheduler, strategy
from bts.simulate import mdp, tail_policy
from commit_lifecycle import DAY, GUARD, choose, data, setup


if __name__ == "__main__":
    with TemporaryDirectory() as temp, ExitStack() as stack:
        root = Path(temp)
        config, _ = setup(stack, root, data(.96))
        cached, sel = choose(config)
        assert cached.pick.batter_id == 100 and sel.decision.guard == GUARD
        picks.save_pick(cached, root)

        # A new daemon sees no policy artifacts. The first scheduled cascade yields
        # no predictions; the final fallback does get a weaker, different slate.
        stack.enter_context(patch.object(strategy, "_mdp_cache", None))
        stack.enter_context(patch.object(mdp, "DEFAULT_POLICY_PATH", root / "missing-base.npz"))
        stack.enter_context(patch.object(tail_policy, "DEFAULT_TAIL_POLICY_PATH", root / "missing-tail.npz"))
        late = data(.60, start_id=200)
        cascade = Mock(side_effect=[(None, None), (late, "late")])
        stack.enter_context(patch.object(orchestrator, "run_cascade", cascade))
        send = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="captured")))
        delivery = stack.enter_context(patch.object(scheduler, "_deliver_and_lock_pick",
                                                    wraps=scheduler._deliver_and_lock_pick))
        scheduler.run_day(DAY, config)

        delivery.assert_called_once()
        assert delivery.call_args.args[5] == "final fallback"
        refresh_selection = delivery.call_args.kwargs["selection"]
        assert refresh_selection.decision.source == "heuristic"
        assert refresh_selection.decision.action == "skip" and refresh_selection.pick_result is None
        assert refresh_selection.primary_candidate["batter_id"] == 200
        send.assert_called_once()
        assert "Fixture batter 100" in send.call_args.args[1]
        assert "Fixture batter 200" not in send.call_args.args[1]
        saved = picks.load_pick(DAY, root)
        assert picks.pick_was_delivered(saved)
        assert saved.pick.batter_id == 100 and saved.policy_decision["guard"] == GUARD
        record = daily_decision.load_decision(DAY, root)
        assert (record["action"], record["delivery_status"]) == ("single", "delivered")
        actual = (record["primary"]["batter_id"], record["source"], record["policy_guard"])
        print("Sent and saved single 100. Record primary, source, guard:", actual)
        assert actual == (100, "mdp", GUARD), actual
