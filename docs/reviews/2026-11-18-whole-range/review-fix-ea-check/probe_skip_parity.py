"""Compare skip summary persistence and notice timing across the exact revisions."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from bts import scheduler
from scripts.audit.season_walk import walk

from probe_contest import WORK, config
from probe_parity import NOW, compare_source


def trace(module, directory):
    day = NOW.date().isoformat()
    stamp = NOW.isoformat()
    frame = walk._predictions(NOW.date(), .75)
    decision = SimpleNamespace(objective="reach57", reason="Observations below the expected count")
    candidate = frame.iloc[0].to_dict()
    generated = [module.build_skip_summary(frame, 10, .8, details, candidate)
                 for details in (None, decision)]
    legacy = {"best_batter": "Batter 0", "best_team": "T0", "best_p": .75, "streak": 10, "bar": .8}
    summaries = [legacy, *generated]
    events, bodies = [], []

    for summary in summaries:
        state = module.SchedulerState(day, stamp, [], [], [], False, None, None, None, skip_summary=summary)
        sends = []

        def send(recipient, body):
            assert recipient == "capture.invalid"
            sends.append(body)
            if len(sends) == 1:
                raise RuntimeError("offline skip transport failure")

        opts = {"now_iso": stamp, "send": send}
        first = module.maybe_notify_skip(state, summary, config(directory), **opts)
        assert first is False and state.skip_notified_at is None
        second = module.maybe_notify_skip(state, summary, config(directory), **opts)
        assert second is True and state.skip_notified_at == stamp
        serialized = module.save_state(state, directory)
        saved = json.loads(serialized.read_text())
        restarted = module.SchedulerState(day, stamp, [], [], [], False, None, None, None)
        module.carry_forward_skip_state(restarted, module.load_state(day, directory))
        third = module.maybe_notify_skip(restarted, summary, config(directory), **opts)
        assert third is False and len(sends) == 2 and restarted.skip_summary == summary
        events.append((first, second, third, len(sends), saved, restarted.skip_notified_at))
        bodies.append(sends[1])
    return events, bodies


prior = compare_source()
with TemporaryDirectory(dir=WORK) as temp, walk.offline(NOW):
    root = Path(temp)
    old_events, old_bodies = trace(prior, root / "old" / "picks")
    current_events, current_bodies = trace(scheduler, root / "new" / "picks")
    assert old_events == current_events
    assert old_bodies[0].replace("Streak used at", "Streak recorded with") == current_bodies[0]
    assert old_bodies[1] == current_bodies[1]
    assert old_bodies[2].replace("Observations below the", "Observations under the") == current_bodies[2]
    print("Wire summaries, notice retry and same-day restart suppression match on legacy, no-decision and decision inputs.")
