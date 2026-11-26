"""Compare a morning repair at selection, guard and transport with real policy."""
import datetime as calendar
import socket
import sys
import tempfile
from pathlib import Path

from fixtures import HERE, fixture
from pytest import MonkeyPatch

ops = fixture("probe_operators")
archived = fixture("probe_text", "contract-check-e")


def run(root, mp, timing):
    config, directory, clock, _ = ops.world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", ops.contest_state.datetime)
    mp.setattr(ops.scheduler, "count_new_confirmations", lambda *a, **k: 0)
    ops.seed_automatic(directory, streak=0)
    frame, tier = ops.orchestrator.run_cascade.return_value
    ops.orchestrator.run_cascade.return_value = frame.assign(p_game_hit=[.80, .80]), tier
    sent = ops.run_case(config)["selection"]
    assert sent.decision.action in ("single", "double")
    daily = sent.pick_result.daily
    ops.picks.save_pick(daily, directory)
    target = root / "corrected"
    config["orchestrator"]["picks_dir"] = str(target)
    response = ops.observation(ops.DAY, target, streak=15, best=15, saver="used")
    assert response.exit_code == 0, response.output
    corrected = ops.run_case(config)["selection"]
    assert corrected.decision.action == "skip", corrected.decision
    config["orchestrator"]["picks_dir"] = str(directory)

    def repair():
        result = ops.observation(ops.DAY, directory, streak=15, best=15, saver="used")
        assert result.exit_code == 0, result.output

    transport = ops.dm.send_dm

    def after_guard(*args, **kwargs):
        repair()
        return transport(*args, **kwargs)

    if timing == "before":
        repair()
    else:
        mp.setattr(ops.dm, "send_dm", after_guard)
    ops.scheduler._deliver_and_lock_pick(
        daily, config, directory, ops.scheduler_state(), ops.DAY, "probe", selection=sent,
    )
    committed = ops.picks.load_pick(ops.DAY, directory)
    delivered = committed is not None and ops.picks.pick_was_delivered(committed)
    assert delivered == (timing == "during")
    print("Timing", timing, "original", sent.decision.action, "corrected", corrected.decision.action,
          "delivered", delivered, "transport calls", transport.call_count)


with tempfile.TemporaryDirectory(dir=HERE) as folder, MonkeyPatch.context() as mp:
    mp.setattr(socket.socket, "connect", archived.offline)
    mp.setattr(socket, "create_connection", archived.offline)
    mp.setattr(socket, "getaddrinfo", archived.offline)
    run(Path(folder), mp, sys.argv[1])
