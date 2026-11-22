"""Compare midnight before and during the saved-pick contest-state read."""

from dataclasses import asdict
from datetime import timedelta
import json
import sys
from unittest.mock import patch

import replay

from bts import contest_state, daily_decision, scheduler, strategy


def run(day, cross_in_reader):
    start = replay.offline.instant(day, 23, 20)
    with replay.offline.offline(start) as (w, mp):
        w.schedule[day][:] = [replay.offline.game(101, start + timedelta(minutes=70))]
        w.dated(0)
        path = w.picks / f"{day}.json"
        real_check = scheduler.run_single_check
        real_lock = scheduler._lock_decision_from_predictions
        real_read = contest_state.load_decision_streak_state
        starts = []
        first_return = False
        rolled = False

        def forecast():
            starts.append(w.now.isoformat())
            if len(starts) == 1:
                w.advance(39 * 60)
                return w.frame((0.99,), (True,))
            return None

        def rollover():
            nonlocal rolled
            rolled = True
            w.advance(5 * 60)
            w.dated(0)

        def check(*args, **kwargs):
            nonlocal first_return
            decisions = []

            def lock(*a, **kw):
                decision = real_lock(*a, **kw)
                decisions.append(decision)
                return decision

            with patch.object(scheduler, "_lock_decision_from_predictions", lock):
                result = real_check(*args, **kwargs)
            if not first_return:
                first_return = True
                assert not result["should_post"]
                raw = json.dumps(asdict(result["pick_result"].daily), indent=2).encode()
                assert path.read_bytes() == raw
                result["lock_decision"], = decisions
                result["saved_pick_bytes"] = raw
                if not cross_in_reader:
                    rollover()
            return result

        def read(*args, **kwargs):
            result = real_read(*args, **kwargs)
            if first_return and not rolled:
                rollover()
            return result

        w.predictions = forecast
        mp.setattr(strategy, "_load_mdp", lambda: None)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": start, "game_pks": [101]}])
        w.config["scheduler"].update({
            "fallback_deadline_min_morning": 40, "fallback_deadline_min": 40,
        })
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(contest_state, "load_decision_streak_state", read)
        scheduler.run_day(day, w.config)
        state = scheduler.load_state(day, w.picks)
        assert state.fallback_refreshes[0]["reused_check_cascade"]
        sent = bool(w.sends("dm"))
        refusal = any("[contest_state]" in body for body in w.sends("health-dm"))
        assert sent is cross_in_reader and refusal is not cross_in_reader
        assert bool(daily_decision.load_decision(day, w.picks)) is sent
        print("BOUNDARY", day, "during final read" if cross_in_reader else "before fallback",
              "sent" if sent else "refused", starts, flush=True)


if __name__ == "__main__":
    for day in sys.argv[1:]:
        for crossing in (False, True):
            run(day, crossing)
