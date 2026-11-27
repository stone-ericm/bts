"""Check the receipt wording with a late delivery guard and the real dispatcher."""

import json
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from bts import picks, scheduler
from bts.health import runner
from rollover import clock, scenario
from scripts.audit.season_walk import walk


day = '2027-06-14'
start = clock(day, 23, 30)
first_pitch = clock(day, 23, 59)
after_midnight = first_pitch + timedelta(minutes=2)

with TemporaryDirectory() as tmp, scenario(Path(tmp), start, [first_pitch]) as (sim, mp):
    daily = sim.seed()
    state = scheduler.load_state(day, sim.picks)

    def delayed_status(_):
        walk._set_clock(after_midnight)
        return sim.statuses

    sim.delivery_status.side_effect = delayed_status
    assert not scheduler._deliver_and_lock_pick(daily, sim.config, sim.picks, state, day, 'lineup')
    first, = sim.dispatches
    assert first['wire'] and first['sent_day'] == after_midnight.date().isoformat()
    assert first['sources'] == ['late_delivery'] and not sim.pick_dms
    cutoff = picks.submission_cutoff_et(daily)
    scheduler._refuse_late_delivery(daily, sim.config, sim.picks, state, day, 'lineup', cutoff, after_midnight)
    second = sim.dispatches[-1]
    assert not second['attempt'] and not second['wire']

    original = runner._safe_run
    mp.setattr(runner, '_safe_run', lambda key, fn: original(key, fn) if key == 'late_delivery' else [])
    runner.run_all_checks(sim.picks, Path(tmp) / 'models', 'capture.invalid', today=start.date())
    digest = sim.dispatches[-1]
    assert digest['wire'] and digest['sent_day'] == day
    assert len(sim.dispatches) == 3
    print(f'First pitch {first_pitch.isoformat()}; delayed status returned {after_midnight.isoformat()}')
    print(json.dumps(sim.dispatches, indent=2))
