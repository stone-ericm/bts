"""Check a save read-back race and an interrupt during pre-send telemetry."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import replay

from bts import contest_state, daily_decision, picks, scheduler


def read_back():
    with replay.offline.offline(replay.offline.instant(replay.DAY, 18, 25)) as (w, mp):
        w.schedule[w.day] = w.schedule[w.day][:1]
        w.dated(0)
        w.predictions = lambda: w.frame((0.99,), (True,))
        write = picks.atomic_write_text
        append = picks.append_lineup_evolution
        writes = []
        original = scheduler.run_single_check
        calculate = scheduler._lock_decision_from_predictions

        def writer(path, body):
            writes.append(body.encode())
            write(path, body)

        def append_and_replace(daily, directory):
            append(daily, directory)
            if len(writes) == 1:
                other = deepcopy(daily)
                other.pick = replace(daily.pick, batter_id=999, batter_name="Replacement Batter")
                picks.save_pick(other, directory)

        def check(*args, **kwargs):
            decisions = []

            def capture(*a, **kw):
                decision = calculate(*a, **kw)
                decisions.append(decision)
                return decision

            with patch.object(scheduler, "_lock_decision_from_predictions", capture):
                result = original(*args, **kwargs)
            _, read_back_bytes = picks.load_pick_bytes(w.day, w.picks)
            assert read_back_bytes == writes[1] and read_back_bytes != writes[0]
            result["lock_decision"], = decisions
            result["saved_pick_bytes"] = read_back_bytes
            w.advance(35 * 60)
            return result

        mp.setattr(picks, "atomic_write_text", writer)
        mp.setattr(picks, "append_lineup_evolution", append_and_replace)
        mp.setattr(scheduler, "run_single_check", check)
        scheduler.run_day(w.day, w.config)
        sent, = w.sends("dm")
        commit = daily_decision.load_decision(w.day, w.picks)
        assert "Replacement Batter" in sent and commit["primary"]["batter_name"] == "Synthetic Batter 101"
        print("ORDER read-back after save captured the other writer's bytes", flush=True)


def interrupt():
    with replay.offline.offline(replay.offline.instant(replay.DAY, 18, 25)) as (w, mp):
        w.schedule[w.day] = w.schedule[w.day][:1]
        w.dated(0)
        w.predictions = lambda: w.frame((0.99,), (True,))
        check = scheduler.run_single_check(w.day, [101], set(), w.config, 0.03)
        daily = check["pick_result"].daily
        state = scheduler.SchedulerState(w.day, "", [], [], [], False, None, None, None)

        def exit_before_marker(*args, **kwargs):
            raise SystemExit(33)

        mp.setattr(contest_state, "automatic_reading_record", exit_before_marker)
        assert scheduler._deliver_and_lock_pick(daily, w.config, w.picks, state, w.day, "fallback")
        assert len(w.sends("dm")) == 1 and daily_decision.load_decision(w.day, w.picks)
        print("ORDER SystemExit in pre-attempt telemetry was caught and the pick sent", flush=True)


if __name__ == "__main__":
    read_back()
    interrupt()
