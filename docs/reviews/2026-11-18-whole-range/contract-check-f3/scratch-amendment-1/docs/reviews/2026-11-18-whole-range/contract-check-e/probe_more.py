"""Real lineup checks for the remaining skip cases and real 08:30/research behavior."""
import json
import socket
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from pytest import MonkeyPatch

from bts import scheduler, strategy, orchestrator
from probe_state import invoke, observation, sent_pick
from probe_text import offline, show
from tests.test_saved_pick_paths_cs5a import world


def picks_cases(root, mp):
    config, directory, clock, _ = world.__wrapped__(root, mp)
    mp.setattr(scheduler, "count_new_confirmations", lambda *a, **k: 0)
    import datetime as calendar
    from bts import contest_state
    mp.setattr(calendar, "datetime", contest_state.datetime)
    for day, streak, best, p in [("2027-09-19", 0, 18, .72), ("2027-08-30", 56, 56, .99),
                                 ("2027-07-15", 56, 56, .99), ("2027-07-16", 10, 10, .82)]:
        clock[0] = datetime.fromisoformat(f"{day}T18:35:00-04:00")
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[p, p]), tier)
        res = observation(day, directory, streak=streak, best=best)
        assert res.exit_code == 0, res.output
        result = scheduler.run_single_check(date=day, all_game_pks=[101, 102], confirmed_sides=set(),
                                            config=config, early_lock_gap=.03)
        show(day, result)


def research(root, mp):
    sent_pick(root, mp, True)
    cfg = root / "test.toml"
    cfg.write_text('[scheduler]\npick_delivery="private"\nentry_intent="research"\ncontest_state_required=true\n')
    result = invoke("day-outcome", "--date", "2027-07-15", "--config", str(cfg), "--dm-recipient", "offline.test",
                    directory=root / "picks")
    print("08:30 after A7", result.exit_code, json.loads(result.output)["state"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as name, MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", offline)
        mp.setattr(socket, "create_connection", offline)
        (picks_cases if sys.argv[1] == "lineup" else research)(Path(name), mp)
