"""Exercise the next day's real automatic selection for each prior saver flag."""
import datetime as calendar
import socket
import tempfile
from datetime import timedelta
from pathlib import Path

from pytest import MonkeyPatch

from bts import cli, contest_state, orchestrator
from probe_state import invoke, observation
from probe_text import offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW, set_automatic
from tests.test_saved_pick_paths_cs5a import world


def probe(root, mp):
    config, _, clock, _ = world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", contest_state.datetime)
    mp.setattr(cli, "datetime", contest_state.datetime)
    frame, tier = orchestrator.run_cascade.return_value
    tomorrow = NOW + timedelta(days=1)
    day = tomorrow.date().isoformat()
    orchestrator.run_cascade.return_value = (
        frame.assign(p_game_hit=[.82, .82], game_time=[f"{day}T23:10:00Z", f"{day}T23:40:00Z"]), tier,
    )
    for dated in (False, True):
        for prior in ("active", "not_earned", "used", "uninitialized"):
            states = []
            for mark in (False, True):
                directory = root / f"{dated}-{prior}-{mark}"
                config["orchestrator"]["picks_dir"] = str(directory)
                clock[0] = NOW
                set_automatic(directory, streak=10)
                if prior != "uninitialized":
                    res = invoke("saver-state", "--init", prior, directory=directory)
                    assert res.exit_code == 0, res.output
                if dated:
                    res = observation(DAY, directory, streak=10, best=10, saver="used")
                    assert res.exit_code == 0, res.output
                if mark:
                    res = invoke("saver-state", "--use", directory=directory)
                    assert res.exit_code == (1 if prior == "uninitialized" else 0), res.output
                clock[0] = tomorrow
                _, sel, _ = orchestrator.run_and_pick(config, day)
                assert sel.pick_result.daily.chosen_on["kind"] == "automatic"
                states.append((sel.saver_available, sel.action))
            print(dated, prior, states)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as name, MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", offline)
        mp.setattr(socket, "create_connection", offline)
        probe(Path(name), mp)
