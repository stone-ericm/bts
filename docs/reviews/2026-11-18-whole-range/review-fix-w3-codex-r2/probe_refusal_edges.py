"""Inject a second failed marker clear or an interrupt inside the refusal retry."""

from datetime import timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import review_world
from bts import daily_decision, picks, scheduler, strategy


review_world.fixture.HERE = Path(__file__).resolve().parent
DAY, at, offline = review_world.DAY, review_world.at, review_world.offline


def run(edge, mode, faults):
    start = at(23, 20) if edge == "midnight" else at(18, 25)
    with offline(start) as (world, mp):
        world.dated(0)
        if edge == "midnight":
            world.schedule[DAY] = [review_world.game(101, start + timedelta(minutes=70))]
            world.config["scheduler"].update(fallback_deadline_min=40, fallback_deadline_min_morning=40)
            mp.setattr(strategy, "_load_mdp", lambda: None)
        else:
            world.schedule[DAY] = world.schedule[DAY][:1]
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [{"time_et": start, "game_pks": [101]}])
        world.config["scheduler"]["pick_delivery"] = mode
        mark = False
        fired = []
        real_write = picks.atomic_write_text
        pick_file = world.picks / f"{DAY}.json"
        prediction_starts = []

        def predict():
            prediction_starts.append(world.now.isoformat())
            if len(prediction_starts) == 1:
                world.advance(2399)
                return world.frame((.99,), (True,))
            return None

        def write(file, text):
            nonlocal mark
            if file == pick_file:
                marker = json.loads(text)["delivery_attempted"]
                if mark and not marker and len(fired) < len(faults):
                    error = faults[len(fired)]
                    fired.append(error.__name__)
                    raise error("fixture failure clearing a known-unsent marker")
                if marker and not mark:
                    mark = True
                    real_write(file, text)
                    world.advance(2)
                    return
            real_write(file, text)

        world.predictions = predict
        mp.setattr(picks, "atomic_write_text", write)
        try:
            scheduler.run_day(DAY, world.config)
        except (OSError, SystemExit, KeyboardInterrupt) as error:
            assert type(error) is faults[-1]
        else:
            raise AssertionError("a failed retry unexpectedly returned")
        assert fired == [fault.__name__ for fault in faults]
        assert picks.load_pick(DAY, world.picks).delivery_attempted
        assert not review_world.health(world) and not world.sends("dm") and not world.sends("post")
        assert daily_decision.load_decision(DAY, world.picks) is None
        assert not list((world.picks / DAY).glob("refused_delivery*.json"))
        mp.setattr(picks, "atomic_write_text", real_write)
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        health = review_world.health(world)
        assert record["delivery_status"] == "locked_unconfirmed"
        assert len(health) == 1 and "OUTCOME UNKNOWN" in health[0]
        assert len(prediction_starts) == 1 and not world.sends("dm") and not world.sends("post")
        print("REFUSAL", json.dumps({"edge": edge, "mode": mode, "faults": fired,
                                     "first_run_marker_retained": True, "first_run_alerts": [],
                                     "restart_status": record["delivery_status"], "restart_alerts": health,
                                     "pick_transport_calls": 0}), flush=True)


if __name__ == "__main__":
    for edge, mode in [("cutoff", "dm"), ("midnight", "public")]:
        for errors in [(OSError, OSError), (OSError, KeyboardInterrupt), (OSError, SystemExit),
                       (KeyboardInterrupt,), (SystemExit,)]:
            run(edge, mode, errors)
